import argparse
import os
import platform
import time
import warnings
from typing import Any, Dict, List, Tuple, Optional
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import json

# from cpu_binding import affinity, num_threads
# if affinity: # https://github.com/pytorch/pytorch/issues/99625
#    os.sched_setaffinity(os.getpid(), affinity)
# if num_threads > 0:
#    torch.set_num_threads(num_threads)
#    torch.set_num_interop_threads(num_threads)

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
from lightning.pytorch.loggers import WandbLogger, CSVLogger

# Project-specific imports
from boreholes import make_boreholes_mask, make_surface_mask, make_combined_mask
from callbacks import EMACallback, InferenceCallback
from geogen.dataset import GeoData3DStreamingDataset
from flowtrain.interpolation import LinearInterpolant, StochasticInterpolator
from flowtrain.models import Unet3DCondV3 as Unet3D
from flowtrain.solvers import ODEFlowSolver

from utils import (
    find_latest_checkpoint,
    plot_cat_view,
    plot_static_views,
)

ROOT_DIR = "/scratch/okhmakv/SI"
os.environ["WANDB_MODE"] = "disabled"
NUM_WORKERS = 16
CPUS_PER_TASK = "auto"  # 192
try:
    NUM_NODES = int(os.environ["SLURM_NNODES"])
except KeyError:
    NUM_NODES = 1


def get_config(args=None) -> dict:
    """
    Generates the entire configuration as a dictionary.

    Args:
        args: Command line arguments from argparse

    Returns:
        dict: Configuration dictionary.
    """
    config = {
        "resume": True,
        "devices": [0,1,2],  # Adjusted below with argparse
        # Project configurations
        "project": {
            "name": "15c_64_b8_ac4_lr1_3_64n_combined",
            "root_dir": ROOT_DIR,
        },
        # Data loader configurations
        "data": {
            "bounds": (
                (-1920, 1920),
                (-1920, 1920),
                (-1920, 1920),
            ),
            "batch_size": 8,
            "epoch_size": 20_000,
            "shape": (64, 64, 64),  # [C, X, Y, Z]
        },
        # Categorical embedding parameters
        "embedding": {
            "num_categories": 15,
            "dim": 15,
        },
        # Model parameters
        "model": {
            "dim": 48,  # Base number of hidden channels in model
            "dim_mults": (
                1,
                2,
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
            "max_epochs": 3000,
            "learning_rate": 1e-3,
            "lr_decay": 0.999,
            "gradient_clip_val": 3e-1,
            "accumulate_grad_batches": 4,
            "log_every_n_steps": 1,
            # --- EMA configuration ---
            "use_ema": True,
            "ema_decay": 0.9995,
            "ema_start_step": 0,
            "ema_update_every": 1,
            "ema_update_on_cpu": True,
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
    if config["devices"] is None:
        if args and hasattr(args, "device"):
            if args.device == "auto":
                system = platform.system()
                if system == "Windows":
                    config["devices"] = (
                        ["cuda"] if torch.cuda.is_available() else ["cpu"]
                    )
                elif system == "Linux":
                    config["devices"] = (
                        ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
                    )
                else:
                    config["devices"] = ["cpu"]
            else:
                config["devices"] = [args.device]
        else:
            # Default auto-detection
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
        "training_logs": os.path.join(root_dir, "training_logs", project_name),
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
            save_top_k=3,
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
    }

    if config["training"].get("use_ema", False):
        ema_cb = EMACallback(
            decay=config["training"]["ema_decay"],
            start_step=config["training"]["ema_start_step"],
            update_every=config["training"]["ema_update_every"],
            update_on_cpu=config["training"]["ema_update_on_cpu"],
        )
        callbacks["ema_callback"] = ema_cb

    return callbacks


def get_data_loader(config: dict, device: str = "cpu") -> DataLoader:
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
        device=device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
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
        time_range: List[float] = [0.0001, 0.9999],
        num_categories: int = 15,
        embedding_dim: int = 20,
        lambda_reconstruct: float = 1.0,
        learning_rate: float = 2e-3,
        lr_decay: float = 0.997,
        **model_params: Any,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

        self.data_shape = data_shape
        self.time_range = time_range
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.lambda_reconstruct = lambda_reconstruct

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

        indices = (
            x.squeeze(1).long() + 1
        )  # Adjust indices if needed (air starts as -1 so bump up by 1)
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

        # Draw encoded geogen model. Small noise is added to prevent singularities.
        X1 = self.embed(batch)  # [B, E, X, Y, Z]
        mask_boreholes = make_combined_mask(batch).expand(
            -1, X1.shape[1], -1, -1, -1
        )  # [B, E, X, Y, Z]
        b = X1[mask_boreholes]  # [N_masked, E]
        ATb = X1 * mask_boreholes
        X1 = X1 + 1e-4 * torch.randn_like(X1)

        X0 = torch.randn_like(X1)  # [B, E, X, Y, Z]

        # Restrict time range for training (See Albergo et al. 2023)
        T = torch.empty(X1.size(0), device=X1.device).uniform_(
            self.time_range[0], self.time_range[1]
        )  # [B,] (unifo)

        # Compute objectives
        XT, VT = self.interpolator.flow_objective(T, X0, X1)
        VT_hat = self.net(XT, ATb, T)  # [B, E, X, Y, Z]

        # Compute straight line estimated reconstruction
        T_broadcasted = T.view(-1, 1, 1, 1, 1)  # Shape: [6, 1, 1, 1, 1]
        b_hat = (
            XT[mask_boreholes] + ((1 - T_broadcasted) * VT_hat)[mask_boreholes]
        )  # [B, E, X, Y, Z]

        # Compute losses
        mse_loss = F.mse_loss(VT, VT_hat) / (
            F.mse_loss(VT, torch.zeros_like(VT)) + 1e-6
        )
        # Weight reconstruction loss (b_hat should match b and converge)
        weighted_reconstruct_loss = (
            T_broadcasted.squeeze()
            * F.mse_loss(b, b_hat)  # Multiply by T to make later values more important
        ) / (F.mse_loss(X1, torch.zeros_like(X1)) + 1e-6)
        weighted_reconstruct_loss = weighted_reconstruct_loss.mean()

        # Total loss includes MSE loss and angle loss (penalty for embedding similarity)
        loss = mse_loss + self.lambda_reconstruct * weighted_reconstruct_loss

        # Log the training loss and orthogonality loss
        self.log_dict(
            {
                "train_loss": loss,
                "flow_loss": mse_loss,
                "reconstruct_loss": weighted_reconstruct_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def on_train_epoch_end(self, unused=None):
        # Log the learning rate at the end of each epoch
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_epoch=True, logger=True)

        print(f"Epoch {self.current_epoch}: Learning Rate = {lr}")

    def on_after_backward(self):
        """Logs gradient norms after the backward pass."""
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)  # L2 norm of gradients
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5  # Compute final norm
        self.log("grad_norm", total_norm, on_step=True, sync_dist=True)
        print(f"Gradient Norm: {total_norm:.6f}")

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_decay
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def launch_training(config, dirs) -> None:
    """
    Initialize and launch the training process.

    Args:
        config (Config): Configuration dataclass.
        dirs (Dict[str, str]): Paths to necessary directories.
        device (str): Device to train on.
    """
    data_loader = get_data_loader(config)
    last_checkpoint = (
        find_latest_checkpoint(dirs["checkpoint_dir"]) if config["resume"] else None
    )

    if last_checkpoint is None:
        model = Geo3DStochInterp(
            data_shape=config["data"]["shape"],
            num_categories=config["embedding"]["num_categories"],
            embedding_dim=config["embedding"]["dim"],
            learning_rate=config["training"]["learning_rate"],
            lr_decay=config["training"]["lr_decay"],
            **config["model"],
        )
    else:
        model = Geo3DStochInterp.load_from_checkpoint(last_checkpoint)
        print(f"Resuming training from checkpoint: {last_checkpoint}")

    # Configure Weights & Biases logger
    logger = WandbLogger(
        project=config["project"]["name"],
        resume="allow",
        log_model=True,
    )

    csv_logger = CSVLogger(
        save_dir=dirs["training_logs"],
    )
    logger.log_hyperparams(config)

    csv_logger.log_hyperparams(config)

    # Create callbacks
    callbacks_dict = create_callbacks(config, dirs)
    callbacks_list = list(callbacks_dict.values())

    # Initialize Trainer
    # print(os.environ)
    trainer = Trainer(
        max_epochs=config["training"]["max_epochs"],
        # devices=config["devices"],
        devices=CPUS_PER_TASK,
        accelerator="cpu",
        # strategy="ddp",
        num_nodes=NUM_NODES,
        callbacks=callbacks_list,
        logger=[logger, csv_logger],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        log_every_n_steps=config["training"]["log_every_n_steps"],
    )

    print(
        "trainer", NUM_NODES, trainer.global_rank, trainer.local_rank, trainer.node_rank
    )
    # exit()
    trainer.ema_callback = callbacks_dict.get("ema_callback", None)

    # Test basic functionality before training begins
    # inference_callback: InferenceCallback = callbacks["inference_callback"]
    # inference_callback.run_manual_inference(trainer, model)

    # Restore case with None for model will load all callbacks (EMA etc)
    trainer.fit(model=model, train_dataloaders=data_loader, ckpt_path=last_checkpoint)


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


def test_inspect_data(data_loader: DataLoader) -> None:
    """
    Inspect samples from the data loader.

    Args:
        data_loader (DataLoader): DataLoader object.
    """
    batch = next(iter(data_loader))
    for b in batch:
        plot_static_views(b.detach().cpu()).show()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train conditional 3D geological models using stochastic interpolation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1", "cuda:2"],
        default="auto",
        help="Device to use for computation",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to execute training or inference based on user needs.
    """
    args = parse_arguments()
    config = get_config(args)
    dirs = setup_directories(config)

    print(f"Running conditional training on device: {config['devices']}")
    launch_training(config, dirs)


if __name__ == "__main__":
    main()
