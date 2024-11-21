"""
Train the velocity matching objective on an infinite 3D GeoData set.
"""

import os
import platform
import time
import warnings
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from matplotlib import patches
from torch.utils.data import DataLoader
from tqdm import tqdm

# Third-party libraries
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.core import LightningModule
from lightning.pytorch.loggers import WandbLogger

# Project-specific imports
from callbacks import EMACallback, InferenceCallback
from geogen.dataset import GeoData3DStreamingDataset
from flowtrain.interpolation import LinearInterpolant, StochasticInterpolator
from flowtrain.models import Unet3D
from flowtrain.solvers import ODEFlowSolver

from utils import (
    find_latest_checkpoint,
    plot_cat_view,
    plot_static_views,
)


def get_config() -> dict:
    """
    Generates the entire configuration as a dictionary.

    Returns:
        dict: Configuration dictionary.
    """
    config = {
        "resume": True,
        "devices": [1, 2],  # This will be adjusted automatically below
        # Project configurations
        "project": {
            "name": "cat-embeddings-18d-normed-64cubed",
            "root_dir": os.path.dirname(os.path.abspath(__file__)),
        },
        # Data loader configurations
        "data": {
            "shape": (64, 64, 64),  # [C, X, Y, Z]
            "bounds": (
                (-1920, 1920),
                (-1920, 1920),
                (-1920, 1920),
            ),
            "batch_size": 6,
            "epoch_size": 10_000,
        },
        # Categorical embedding parameters
        "embedding": {
            "num_categories": 15,
            "dim": 18,
        },
        # Model parameters
        "model": {
            "dim": 48,  # Base number of hidden channels in model
            "dim_mults": (
                1,
                1,
                2,
                3,
                4,
            ),  # Multipliers for hidden dims in each superblock, total 2x downsamples = len(dim_mults)-1
            "data_channels": 1,  # Data clamped down to fit categorical count
            "dropout": 0.1,  # Optional network dropout
            "self_condition": False,  # Optional conditioning on input data
            "time_sin_pos": False,  # Use fixed sin/cos positional embeddings for time
            "time_resolution": 1024,  # Resolution of time (number of random Fourier features)
            "time_bandwidth": 1000.0,  # Starting bandwidth of fourier frequencies, f ~ N(0, time_bandwidth)
            "time_learned_emb": True,  # Learnable fourier freqs and phases
            "attn_enabled": True,  # Enable or disable self attention before each (down/up sample) also feeds skip connections
            "attn_dim_head": 32,  # Size of attention hidden dimension heads
            "attn_heads": 4,  # Number of chunks to split hidden dimension into for attention
            "full_attn": None,  # defaults to full attention only for inner most layer final down, middle, first up
            "flash_attn": False,  # For high performance GPUs https://github.com/Dao-AILab/flash-attention
        },
        # Training parameters
        "training": {
            "lambda_angle": 10,
            "max_epochs": 2000,
            "learning_rate": 2.0e-4,
            "lr_decay": 0.997,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 24,
            "log_every_n_steps": 5,
        },
        # Inference parameters
        "inference": {
            "seed": None,
            "n_samples": 1,
            "batch_size": 4,
            "save_imgs": True,
        },
    }

    # Dynamically set device configurations
    if not config["devices"]:
        system = platform.system()
        if system == "Windows":
            config["devices"] = ["cuda"] if torch.cuda.is_available() else ["cpu"]
        elif system == "Linux":
            config["devices"] = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
        else:
            config["devices"] = ["cpu"]

    # Ensure model_params are updated with the embedding dimension
    config["model"]["data_channels"] = config["embedding"]["dim"]

    return config


def setup_directories(config):
    root_dir = config["project"]["root_dir"]
    project_name = config["project"]["name"]

    dirs = {
        "checkpoint_dir": os.path.join(root_dir, "saved_models", project_name),
        "photo_dir": os.path.join(root_dir, "images", project_name),
        "emb_dir": os.path.join(root_dir, "embeddings", project_name),
        "samples_dir": os.path.join(root_dir, "samples", project_name),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs


def create_callbacks(config, dirs) -> Dict[str, Callback]:
    """
    Create a dictionary of callbacks for the PyTorch Lightning Trainer.

    Includes options for top-k checkpointing, last checkpointing, inference during training, and more.

    Args:
        config: Configuration dataclass or dictionary.
        dirs: Dictionary of directories for saving outputs.

    Returns:
        A dictionary of callbacks with descriptive keys.
    """
    callbacks = {
        "top_k_checkpoint": ModelCheckpoint(
            dirpath=dirs["checkpoint_dir"],
            filename="topk-{epoch:02d}-{train_loss:.4f}",
            save_top_k=1,
            verbose=True,
            monitor="train_loss",
            mode="min",
        ),
        "last_checkpoint": ModelCheckpoint(
            dirpath=dirs["checkpoint_dir"],
            save_last=True,
            verbose=True,
            monitor="epoch",
            mode="max",
        ),
        "inference_callback": InferenceCallback(
            save_dir=dirs["photo_dir"],
            every_n_epochs=5,
            n_samples=4,
            n_steps=32,
            tf=0.999,
            seed=42,
        ),
    }

    return callbacks


def get_data_loader(config: dict, device: str) -> DataLoader:
    """
    Initialize the data loader for training.

    Args:
        config (dict): Configuration dictionary.
        device (str): Device to load data onto.
    """
    dataset = GeoData3DStreamingDataset(
        model_resolution=config["data"]["shape"],  # [C, X, Y, Z]
        model_bounds=config["data"]["bounds"],
        dataset_size=config["data"]["epoch_size"],
        device="cpu",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=16,
    )
    return dataloader


class Geo3DStochInterp(LightningModule):
    """
    A PyTorch Lightning module for stochastic interpolation of 3D GeoData.

    Parameters
    ----------
    data_shape : tuple
        Shape of the data, [X, Y, Z]
    time_range : list
        Time range for interpolation training. Reduced time range can help with variance near boundaries.
    num_categories : int
        Number of categorical classes.
    embedding_dim : int
        Dimension of the embedding vectors for categories, GeoGen has 15 categories to embed
    lambda_angle : float
        Weight for the angle (cosine similarity) loss if using learnable normalized embeddings.
    model_params : dict
        Parameters for the flow match ML model that predicts the stochastic interpolation objective.

    Attributes
    ----------
    net : nn.Module
        The machine learning model that predicts the stochastic interpolation objective.
    interpolant : LinearInterpolant
        The linear interpolant for stochastic interpolation.
    interpolator : StochasticInterpolator
        Manages calculations for stochastic interpolation.
    embedding : nn.Embedding
        Embedding layer for categorical data.
    """

    def __init__(
        self,
        data_shape: Tuple[int, int, int] = (32, 32, 32),
        time_range: List[float] = [0.0005, 0.9995],
        num_categories: int = 15,
        embedding_dim: int = 20,
        lambda_angle: float = 0.1,
        learning_rate: float = 2.0e-4,  # Left here due to saved model compatibility
        lr_decay: float = 0.997,
        **model_params: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_shape = data_shape
        self.time_range = time_range
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.lambda_angle = lambda_angle

        # Embedding layer setup
        self.embedding = nn.Embedding(self.num_categories, self.embedding_dim)
        self._initialize_embedding(self.num_categories, self.embedding_dim)
        # Freeze embedding weights after initialization
        self.embedding.weight.requires_grad = False

        # Update model_params to reflect the new input channels
        model_params["data_channels"] = self.embedding_dim
        self.net = Unet3D(**model_params)
        self.ema_shadow = {}  # To store EMA weights for U-Net (optional)

        # Stochastic Interpolant and interpolator setup
        self.interpolant = LinearInterpolant(one_sided=True)
        self.interpolator = StochasticInterpolator(self.interpolant)

    def _initialize_embedding(self, n_cats: int, n_dims: int) -> None:
        """
        Build an embedding matrix with n_cats categories and n_dims dimensions as
        a simplex shifted to be centered at the origin. This embedding has the property
        that the angle between any two embedding vectors is maximized, maximizing the
        difference between categories using cosine similarity. (Think of the points of a tetrahedron in 3D)

        Args:
            n_cats (int): Number of categories.
            n_dims (int): Dimension of each embedding vector.
        """
        with torch.no_grad():
            # Initial basis is n_cats unit vectors in the first n_cats dimensions with other dims set to zero
            init_matrix = torch.zeros(n_cats, n_dims)
            init_matrix[:, :n_cats] = torch.eye(n_cats)

            # Compute the centroid vector
            centroid_vector = torch.ones(n_cats) / n_cats
            centroid_vector = torch.cat([centroid_vector, torch.zeros(n_dims - n_cats)])

            # Shift the basis vectors to center the simplex at the origin
            init_matrix[:, :n_cats] -= centroid_vector[:n_cats].unsqueeze(0)

            # Normalize the rows to unit norm
            init_matrix = init_matrix / init_matrix.norm(dim=1, keepdim=True)

            self.embedding.weight.data.copy_(init_matrix)

    def forward(self, x, t):
        return self.net(x, t)

    def embed(self, x):
        """
        Convert [B, X, Y, Z] categorical indices to [B, E, X, Y, Z] embedding vectors.
        The E dimension holds the vector representation of the category.
        """

        indices = x.squeeze(1).long() + 1  # Adjust indices if needed
        embedded = self.embedding(indices)  # [B, X, Y, Z, E]
        embedded = embedded.permute(0, 4, 1, 2, 3).contiguous()  # [B, E, X, Y, Z]
        return embedded

    # TODO: Check efficiency of this function, the expanded broadcast might be computationally expensive
    def decode(self, x, return_logits=False):
        """
        Decode a tensor of embedding vectors back to categorical indices.

        The input tensor x is expected to have shape [B, E, X, Y, Z].
        For unit norm embedding vectors, the dot product is equivalent to nearest neighbor cosine similarity.
        """
        embedding_vecs = self.embedding.weight
        B, E, X, Y, Z = x.shape

        # Normalize over the embedding dimension E for cosine similarity
        x = F.normalize(x, dim=1)
        embedding_vecs = F.normalize(embedding_vecs, dim=1)

        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(1)  # [B, 1, E, X, Y, Z]
        embedding_vecs_expanded = embedding_vecs.view(
            1, self.num_categories, E, 1, 1, 1
        )  # [1, num_categories, E, 1, 1, 1]

        # Compute similarity (dot product)
        logits = (x_expanded * embedding_vecs_expanded).sum(
            dim=2
        )  # [B, num_categories, X, Y, Z]

        preds = torch.argmax(logits, dim=1)  # [B, X, Y, Z]

        if return_logits:
            return logits
        else:
            return preds

    def save_embedding(self, emb_dir: str) -> None:
        """
        Save an embedding layer to a file. Useful for learnable embeddings.

        Args:
            emb_dir (str): Directory to save the embedding.
        """
        embedding_save_path = os.path.join(emb_dir, "learned_embedding_full.pt")
        torch.save(self.embedding.state_dict(), embedding_save_path)
        print(f"Learned embedding saved to {embedding_save_path}")

    def training_step(self, batch):
        """
        Training step for the Lightning module.

        Args:
            batch (torch.Tensor): Batch of input data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        # Renormalize embeddings to unit ball, important for learnable embeddings and decoding metric used
        with torch.no_grad():
            self.embedding.weight.div_(
                self.embedding.weight.norm(p=2, dim=1, keepdim=True)
            )

        # Draw encoded geogen model. Small noise is added to prevent singularities.
        X1 = self.embed(batch)  # [B, E, X, Y, Z]
        X1 = X1 + 1e-3 * torch.randn_like(X1)

        X0 = torch.randn_like(X1)  # [B, E, X, Y, Z]

        # Restrict time range for training (See Albergo et al. 2023)
        T = torch.empty(X1.size(0), device=X1.device).uniform_(
            self.time_range[0], self.time_range[1]
        )  # [B,] (unifo)

        # Compute objectives
        XT, BT = self.interpolator.flow_objective(T, X0, X1)
        BT_hat = self.net(XT, T)  # [B, E, X, Y, Z]

        # Compute losses
        mse_loss = F.mse_loss(BT, BT_hat) / F.mse_loss(BT, torch.zeros_like(BT))

        # Orthogonality loss on embeddings (metric if using learnable embeddings)
        embed_norm = F.normalize(self.embedding.weight, dim=1)
        gram_matrix = torch.matmul(embed_norm, embed_norm.t())
        angle_loss = torch.sum(gram_matrix)  # optimal spread gives 0

        # Total loss includes MSE loss and angle loss (penalty for embedding similarity)
        loss = mse_loss + self.lambda_angle * angle_loss

        # Track the mean of the embedding vectors (useful for learnable embeddings)
        mean_embedding = self.embedding.weight.mean(dim=0)  # Mean of embedding vectors
        mean_embedding_norm = torch.norm(
            mean_embedding
        )  # Norm of the mean embedding vector

        # Log the training loss and orthogonality loss
        self.log_dict(
            {
                "train_loss": loss,
                "flow_loss": mse_loss,
                "angle_loss": angle_loss,
                "mean_embedding_norm": mean_embedding_norm,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )

        return loss

    def on_train_epoch_end(self, unused=None):
        # Log the learning rate at the end of each epoch
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_epoch=True, logger=True)
        # self._log_embedding_gram_matrix()

    def _log_embedding_gram_matrix(self):
        """
        Helper to log a heatmap of the Gram matrix of the embedding vectors.
        This computes the angle pariwise between all embedding vectors, useful
        for tracking the spread of learnable embeddings.
        """
        # Generate the Gram matrix (self dot products) for embedding vectors
        embed_vecs = self.embedding.weight
        gram_matrix = (
            torch.matmul(embed_vecs, embed_vecs.t()).detach().cpu().numpy()
        )  # Convert to numpy
        # Remove the diagonal elements for visualization
        np.fill_diagonal(gram_matrix, 0)

        # Create a heatmap plot with matplotlib/seaborn
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(gram_matrix, annot=False, cmap="coolwarm", cbar=True)
        plt.title(f"Embedding Gram Matrix at Epoch {self.current_epoch}")

        # Define groupings for the boxes and corresponding labels
        groupings = [[0], [1], [2, 3, 4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14]]
        labels = ["Air", "Basement", "Sediment", "Dike", "Intrusion", "Blob"]

        for group, label in zip(groupings, labels):
            # Draw a rectangle around each group
            start = group[0]
            end = group[-1]
            rect = patches.Rectangle(
                (start, start),
                end - start + 1,
                end - start + 1,
                fill=False,
                edgecolor="teal",
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(rect)

            # Add a label in the center of each box
            center_x = (start + end + 1) / 2
            center_y = (start + end + 1) / 2
            ax.text(
                center_x,
                center_y,
                label,
                color="black",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                alpha=0.7,
            )

        # Log the plot directly to Wandb via the logger
        self.logger.experiment.log({"embedding_gram_matrix": wandb.Image(plt)})

        # Close the plot to free memory
        plt.close()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_decay
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint):

        checkpoint["ema_shadow"] = (
            self.ema_shadow
        )  # Save the EMA weights to the checkpoint

    def on_load_checkpoint(self, checkpoint):
        self.ema_shadow = checkpoint[
            "ema_shadow"
        ]  # Load the EMA weights from the checkpoint


def launch_training(config, dirs, device: str) -> None:
    """
    Initialize and launch the training process.

    Args:
        config (Config): Configuration dataclass.
        dirs (Dict[str, str]): Paths to necessary directories.
        device (str): Device to train on.
    """
    data_loader = get_data_loader(config, device=device)

    # Initialize the model
    last_checkpoint = (
        find_latest_checkpoint(dirs["checkpoint_dir"]) if config["resume"] else None
    )
    if config["resume"] and last_checkpoint:
        model = Geo3DStochInterp.load_from_checkpoint(last_checkpoint)
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        model = Geo3DStochInterp(
            data_shape=config["data"]["shape"],
            num_categories=config["embedding"]["num_categories"],
            embedding_dim=config["embedding"]["dim"],
            lambda_angle=config["training"]["lambda_angle"],
            learning_rate=config["training"]["learning_rate"],
            lr_decay=config["training"]["lr_decay"],
            **config["model"],
        )
        last_checkpoint = None

    # Configure Weights & Biases logger
    logger = WandbLogger(
        project=config["project"]["name"], resume="allow", log_model=True
    )
    logger.log_hyperparams(config)

    # Create callbacks
    callbacks = create_callbacks(config, dirs)
    callbacks_as_list = list(callbacks.values())

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=config["training"]["max_epochs"],
        devices=config["devices"],
        logger=logger,
        callbacks=callbacks_as_list,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        log_every_n_steps=config["training"]["log_every_n_steps"],
    )

    # Test basic functionality before training begins
    test_inspect_data(data_loader)
    inference_callback = callbacks["inference_callback"]
    inference_callback.run_manual_inference(trainer, model)

    # Start training
    trainer.fit(model, data_loader, ckpt_path=last_checkpoint)


def load_model(
    checkpoint_dir: str, device: str, model_path: Optional[str] = None
) -> Geo3DStochInterp:
    """
    Load the model from a checkpoint.

    Args:
        checkpoint_dir (str): Directory containing checkpoints.
        device (str): Device to load model onto.
        model_path (Optional[str]): Specific path to a checkpoint. If None, finds the latest checkpoint.

    Returns:
        Geo3DStochInterp: The loaded model.
    """
    if model_path is None:
        last_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if not last_checkpoint:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    else:
        last_checkpoint = model_path

    model = Geo3DStochInterp.load_from_checkpoint(last_checkpoint)
    model.to(device)
    model.eval()
    return model


def run_inference(
    dirs,
    device,
    model=None,
    n_samples=1,
    batch_size=4,
    inference_seed=None,
    data_shape=None,
    save_imgs=True,
) -> None:
    """
    Run inference to generate samples using the trained model.

    Args:
        config (dict): Configuration dictionary.
        dirs (Dict[str, str]): Paths to necessary directories.
        device (str): Device to run inference on.
    """

    checkpoint_dir = dirs["checkpoint_dir"]
    samples_dir = dirs["samples_dir"]
    os.makedirs(samples_dir, exist_ok=True)

    if model == None:
        # Load model from checkpoint
        last_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if not last_checkpoint:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        model = Geo3DStochInterp.load_from_checkpoint(last_checkpoint)

    model.to(device)
    model.eval()

    if data_shape is None:
        data_shape = model.data_shape

    # Apply EMA weights if available
    if hasattr(model, "ema_callback"):
        model.ema_callback.apply_ema_weights(model)

    solver = ODEFlowSolver(model=model.net, rtol=1e-6)

    t0, tf = 0.001, 0.999
    n_steps = 32

    # Enable off-screen rendering if using PyVista or similar
    # pv.OFF_SCREEN = True  # Uncomment if using PyVista

    num_batches = (n_samples - 1) // batch_size + 1
    generator = (
        torch.Generator(device="cpu").manual_seed(inference_seed)
        if inference_seed
        else None
    )

    total_start = time.time()

    with tqdm(total=num_batches, desc="Generating Batches") as pbar:
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
            if generator:
                X0 = torch.randn(
                    current_batch_size,
                    model.embedding_dim,
                    *data_shape,
                    generator=generator,
                ).to(device)
            else:
                X0 = torch.randn(
                    current_batch_size,
                    model.embedding_dim,
                    *data_shape,
                ).to(device)

            # Solve ODEFlow
            start = time.time()
            solution = solver.solve(
                X0, t0=t0, tf=tf, n_steps=n_steps
            )  # [T, B, C, X, Y, Z]
            inference_time = time.time() - start
            print(
                f"Batch {batch_idx + 1}/{num_batches}: ODEFlow solved in {inference_time:.2f} seconds"
            )

            # Decode solution at final time step
            final_solution = solution[-1]  # [B, C, X, Y, Z]
            decoded = model.decode(final_solution)  # [B, X, Y, Z]

            for i in range(current_batch_size):
                sample_idx = batch_idx * batch_size + i
                sample_tensor = final_solution[i].detach().cpu()  # [C, X, Y, Z]
                tensor_path = os.path.join(samples_dir, f"sample_{sample_idx}.pt")
                torch.save(sample_tensor, tensor_path)
                print(f"Saved tensor to {tensor_path}")

                if save_imgs:
                    # Save static view
                    try:
                        static_plot_path = os.path.join(
                            samples_dir, f"static_view_{sample_idx}.png"
                        )
                        plot_static_views(sample_tensor, save_path=static_plot_path)
                        print(f"Saved static view to {static_plot_path}")
                    except Exception as e:
                        warnings.warn(
                            f"Failed to save static view for sample {sample_idx}: {e}"
                        )

                    # Save categorical view
                    try:
                        cat_plot_path = os.path.join(
                            samples_dir, f"cat_view_{sample_idx}.png"
                        )
                        plot_cat_view(
                            decoded[i].detach().cpu(), save_path=cat_plot_path
                        )
                        print(f"Saved categorical view to {cat_plot_path}")
                    except Exception as e:
                        warnings.warn(
                            f"Failed to save categorical view for sample {sample_idx}: {e}"
                        )

            pbar.update(1)

    total_time = time.time() - total_start
    print(
        f"Inference completed: Generated {n_samples} samples in {total_time:.2f} seconds."
    )


def test_inspect_data(data_loader: DataLoader) -> None:
    """
    Inspect samples from the data loader.

    Args:
        data_loader (DataLoader): DataLoader object.
    """
    batch = next(iter(data_loader))
    for b in batch:
        plot_static_views(b.detach().cpu()).show()


def main() -> None:
    """
    Main function to execute training or inference based on user needs.
    """
    config = get_config()
    dirs = setup_directories(config)

    device = config["devices"]

    run_training = False
    run_inference_flag = True

    if run_training:
        launch_training(config, dirs, device)

    if run_inference_flag:

        inference_device = "cuda:2"

        relative_checkpoint_path = os.path.join(
            "demo_model", "trained-model.ckpt"
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
        checkpoint_path = os.path.join(script_dir, relative_checkpoint_path)

        model = Geo3DStochInterp.load_from_checkpoint(checkpoint_path).to(
            inference_device
        )

        run_inference(
            dirs,
            inference_device,
            model=model,
            n_samples=4,
            batch_size=4,
            data_shape=(64, 64, 64),
            save_imgs=False,
        )


if __name__ == "__main__":
    main()
