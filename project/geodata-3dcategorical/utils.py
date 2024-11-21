import os

import einops
import geogen.plot as geovis
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
from geogen.model import GeoModel


# Function to find the latest checkpoint file in a directory
def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [
        os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
    ]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint


def plot_2d_slices(volume, save_path=None, max_slices=64):
    """
    Plot and optionally save 2D slices from a 3D or 4D (with single batch dimension) volume data tensor.

    Args:
    - volume_data (np.ndarray or torch.Tensor): The 3D or 4D volume data with shape (x, y, z) or (1, x, y, z).
    - save_path (str, optional): Path to save the image file. If None, the plot is not saved.
    - max_slices (int): Maximum number of slices to display.

    Returns:
    - None
    """
    # Ensure numpy array for manipulation, handle both numpy and PyTorch tensors
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    # Check if tensor is 4D with the first dimension being 1, and squeeze if necessary
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = np.squeeze(volume, axis=0)

    volume = einops.rearrange(volume, "x y z -> x z y")

    # Determine the number of slices to show
    num_slices = min(volume.shape[0], max_slices)
    num_cols = int(np.sqrt(num_slices))
    num_rows = (num_slices + num_cols - 1) // num_cols  # Ceiling division

    # Calculate global min and max values for consistent color mapping
    global_min = volume.min()
    global_max = volume.max()

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    if num_rows == 1 or num_cols == 1:
        axes = np.atleast_1d(axes)  # Ensure axes is array for consistent indexing

    slice_indices = np.linspace(0, volume.shape[0] - 1, num_slices, dtype=int)
    for i, slice_idx in enumerate(slice_indices):
        ax = axes.flat[i]
        im = ax.imshow(
            volume[slice_idx],
            cmap="viridis",
            vmin=global_min,
            vmax=global_max,
            origin="lower",
        )
        ax.set_title(f"Slice {slice_idx}")
        ax.axis("off")  # Hide the axes

    # Hide any remaining empty subplots
    for j in range(i + 1, num_cols * num_rows):
        axes.flat[j].axis("off")

    plt.tight_layout()
    # Save to png if path provided
    if save_path is not None:
        plt.savefig(save_path)
    plt.colorbar(
        im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.05
    )
    plt.close(fig)


def _validate_tensor_bounds(tensor, bounds):
    """Helper function to validate tensor shape and bounds for GeoModel conversion.
    Tensor is validate as single channel 3D. If no bounds are provided, assume cubic voxels.
    """
    shape = tensor.shape
    dims = len(shape)
    if dims > 3:
        if any(dim != 1 for dim in shape[:-3]):
            raise ValueError(
                "Only tensors with singleton dimensions before the last 3 dimensions are supported."
            )
        tensor = (
            tensor.squeeze()
        )  # Remove singleton dimensions before the last 3 dimensions

    if len(tensor.shape) != 3:
        raise ValueError("The tensor must be 3D with shape (X, Y, Z).")
    if bounds is None:  # Assume cubic voxels if bounds not provided
        bounds = ((0, tensor.shape[0]), (0, tensor.shape[1]), (0, tensor.shape[2]))

    return tensor, bounds


def plot_static_views(tensor, bounds=None, save_path=None):
    """Plot 4 different static views and return pv plotter

    Parameters:
    - tensor (torch.Tensor): The 3d tensor to plot static views of
    - bounds (tuple): Optional metric bounds for the model ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    - save_path (str): Optional path to save the plot screenshot
    """
    tensor, bounds = _validate_tensor_bounds(
        tensor, bounds
    )  # Validate tensor shape and bounds
    model = GeoModel.from_tensor(
        data_tensor=tensor, bounds=bounds
    )  # Convert tensor to GeoModel

    p = pv.Plotter(
        shape=(2, 2), off_screen=True
    )  # Setup 2x2 subplots, offscreen plotting for screenshots

    # Set subplots with different camera positions
    for i in range(4):
        p.subplot(i // 2, i % 2)
        geovis.volview(model, plotter=p)
        p.camera.azimuth = i * 90
        p.camera.elevation = 3

    if save_path is not None:  # Optional save screenshot
        p.screenshot(save_path, scale=1, transparent_background=False)

    # p.off_screen = False  # Resume onscreen plotting for returned plotter, p.show() to display
    return p


def plot_cat_view(tensor, bounds=None, save_path=None):
    """Save a categorical view of the rounded data

        Parameters:
    - tensor (torch.Tensor): The 3d tensor to plot static views of
    - bounds (tuple): Optional metric bounds for the model ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    - save_path (str): Optional path to save the plot screenshot
    """
    tensor, bounds = _validate_tensor_bounds(
        tensor, bounds
    )  # Validate tensor shape and bounds
    model = GeoModel.from_tensor(
        data_tensor=tensor, bounds=bounds
    )  # Convert tensor to GeoModel
    # Make sure rock data is properly rounded to integers for limited discrete categories
    model.data = np.round(model.data)

    p = geovis.categorical_grid_view(model, text_annot=True, off_screen=True)

    if save_path is not None:  # Optional save screenshot
        p.screenshot(save_path, scale=1, transparent_background=False)

    p.off_screen = (
        False  # Resume onscreen plotting for returned plotter, p.show() to display
    )
    return p


def main():
    from geogen.dataset import DataLoader, GeoData3DStreamingDataset

    dataset = GeoData3DStreamingDataset(
        model_resolution=(64, 64, 32),  # Resolution of the model (remove color channel)
        dataset_size=10_000,
        device="cpu",
    )  # Number of samples in one epoch
    dataloader = DataLoader(dataset, batch_size=4, num_workers=4)
    batch = next(iter(dataloader))
    vol = batch[0]
    plot_2d_slices(vol, save_path="test.png")
    p = plot_static_views(vol, save_path="static_view.png")
    p = plot_cat_view(vol, save_path="cat_view.png")


if __name__ == "__main__":
    main()
