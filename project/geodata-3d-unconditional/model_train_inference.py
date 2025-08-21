"""
Train the velocity matching objective on an infinite 3D GeoData set.
"""

import argparse
import os
import re
import platform
import time
import warnings
from typing import Any, Dict, List, Tuple, Optional, Union

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
    download_if_missing,
)


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
        "devices": [0],
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
            "epoch_size": 10_000, # Artificial epcoch size, all samples generated are unseen across epochs
        },
        # Categorical embedding parameters
        "embedding": {
            "num_categories": 15,  # Number of categories for the embedding
            "dim": 18,  # The 15D num_category simplex is centered at origin in 18D space
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

    # For TRAINING: parse --train-devices into Lightning-friendly values
    accelerator, trainer_devices, _ = _parse_devices_arg(
        getattr(args, 'train_devices', None) if args else None
    )
    config["accelerator"] = accelerator         # 'cpu' or 'gpu'
    config["devices"] = trainer_devices         # 1 or [0,1,...]

    # Ensure model_params are updated with the embedding dimension
    config["model"]["data_channels"] = config["embedding"]["dim"]

    return config

def _parse_devices_arg(devices_str: Optional[str]) -> Tuple[str, Union[int, List[int]], str]:
    """Return (accelerator, trainer_devices, inference_device_str).

    Accepted values for devices_str:
    - "cpu"
    - "auto" (all GPUs if available else CPU)
    - comma-separated GPU indices, e.g. "0" or "0,1,2"
    """
    # Defaults
    if devices_str is None:
        devices_str = "auto"

    s = devices_str.strip().lower()

    # CPU
    if s == "cpu":
        return "cpu", 1, "cpu"

    # AUTO
    if s == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpus = list(range(torch.cuda.device_count()))
            return "gpu", gpus, f"cuda:{gpus[0]}"
        else:
            return "cpu", 1, "cpu"


    # Explicit GPU index or indices
    if re.fullmatch(r"\d+(,\d+)*", s):
        idxs = [int(x) for x in s.split(",")]
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but GPU indices were provided.")
        max_idx = torch.cuda.device_count() - 1
        bad = [i for i in idxs if i < 0 or i > max_idx]
        if bad:
            raise ValueError(
            f"Requested GPU indices {bad} out of range [0, {max_idx}] for this machine."
            )
        return "gpu", idxs, f"cuda:{idxs[0]}"

    raise ValueError(
    f"Unrecognized --devices value '{devices_str}'. Use 'cpu', 'auto', or comma-separated GPU indices like '0,1,2'."
    )

def setup_directories(config):
    """
    Project name and root dir taken from config to form directories for training
    
    checkpoint_dir: where to save training models
    photo_dir: where to save intermediate inference run outputs during training
    emb_dir: where to save learned embeddings for categorical data, if applicable
    samples_dir: where to save generated samples during inference trials in training
    """
    
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
        learning_rate = None,  # Left here due to saved model compatibility
        lr_decay = None, # Left here for backward compatibility
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
        # Freeze embedding weights after initialization (non-learnable)
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

        # Nearest neighbor decoder
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

        Returns:
            torch.Tensor: Loss value.
        """
        # Draw encoded geogen model. Small noise is added to prevent singularities.
        X1 = self.embed(batch)  # [B, E, X, Y, Z]
        X1 = X1 + 1e-3 * torch.randn_like(X1)

        X0 = torch.randn_like(X1)  # [B, E, X, Y, Z]

        # Restrict time range for training (See Albergo et al. 2023)
        T = torch.empty(X1.size(0), device=X1.device).uniform_(
            self.time_range[0], self.time_range[1]
        )  # [B,] (unifo)

        # Compute objectives with flowtrain package
        XT, VT = self.interpolator.flow_objective(T, X0, X1)
        VT_hat = self.net(XT, T)  # [B, E, X, Y, Z]

        # Compute losses
        mse_loss = F.mse_loss(VT, VT_hat) / F.mse_loss(VT, torch.zeros_like(VT))

        # Log the training loss and orthogonality loss
        self.log_dict(
            {
                "train_loss": mse_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )

        return mse_loss

    def on_train_epoch_end(self, unused=None):
        # Log the learning rate at the end of each epoch
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_epoch=True, logger=True)
        # self._log_embedding_gram_matrix()

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
        accelerator=config["accelerator"],
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

    t0, tf = 0.001, 1.0
    n_steps = 16

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
                sample_tensor = decoded[i].detach().cpu()  # [C, X, Y, Z]
                tensor_path = os.path.join(
                    samples_dir, f"decoded_s{inference_seed}_{sample_idx}.pt"
                )
                torch.save(sample_tensor, tensor_path)

                # Save the nondecoded tensors with time steps as well
                sample_tensor = solution[:, i].detach().cpu()  # [T, C, X, Y, Z]
                tensor_path = os.path.join(
                    samples_dir, f"fullsol_s{inference_seed}_{sample_idx}.pt"
                )
                torch.save(sample_tensor, tensor_path)

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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train or run inference on 3D geological models using stochastic interpolation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'inference', 'both'], 
        default='inference',
        help='Mode to run: train, inference, or both'
    )
    
    parser.add_argument(
        '--train-devices',
        type=str,
        default='0',
        help="Training devices: 'cpu', 'auto', '0', or '0,1,2' (comma-separated GPU indices)"
    )

    parser.add_argument(
        '--infer-device',
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Inference device: 'cpu' or 'cuda'"
    )
    
    parser.add_argument(
        '--n-samples', 
        type=int, 
        default=8,
        help='Number of samples to generate during inference'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=100,
        help='Random seed for inference'
    )
    
    parser.add_argument(
        '--save-images', 
        action='store_true',
        help='Save visualization images during inference'
    )
    
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='Path to specific checkpoint file (if not provided, uses demo model)'
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Main function to execute training or inference based on user needs.
    """
    args = parse_arguments()
    config = get_config(args)
    dirs = setup_directories(config)

    device = config["devices"]
    
    print(f"Running in {args.mode} mode on device: {device}")

    if args.mode in ['train', 'both']:
        print("Starting training...")
        launch_training(config, dirs, device)

    if args.mode in ['inference', 'both']:
        print("Starting inference...")
        
        inference_device = args.infer_device
        if inference_device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("Requested --infer-device cuda but CUDA is not available.")

        # Use provided checkpoint or default demo model
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
        else:
            relative_checkpoint_path = os.path.join(
                "demo_model", "unconditional-weights.ckpt"
            )
            script_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(script_dir, relative_checkpoint_path)
            
            weights_url = "https://github.com/chipnbits/flowtrain_stochastic_interpolation/releases/download/v1.0.0/unconditional-weights.ckpt"
            download_if_missing(checkpoint_path, weights_url)

        print(f"Loading model from: {checkpoint_path}")
        model = Geo3DStochInterp.load_from_checkpoint(
            checkpoint_path, map_location=inference_device
        ).to(inference_device)

        run_inference(
            dirs,
            inference_device,
            model=model,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            data_shape=(64, 64, 64),
            inference_seed=args.seed,
            save_imgs=args.save_images,
        )
        
        print(f"Inference completed! Results saved to: {dirs['samples_dir']}")


if __name__ == "__main__":
    main()
