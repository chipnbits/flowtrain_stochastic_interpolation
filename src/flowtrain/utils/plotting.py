import einops
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from matplotlib.animation import FuncAnimation

from flowtrain.interpolation import StochasticInterpolator


def show_images(
    tensors, denormalize=True, title="", save_path=None, show_windowed=True
):
    """Shows the provided images as sub-pictures in a square"""

    if denormalize:
        images = denormalize_images(tensors)
    else:
        images = tensors

    print(f"images tensor is {images.shape}")
    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                # plt.imshow(images[idx][0], cmap="gray")
                # Handle color vs grayscale images
                if images[idx].shape[0] == 1:

                    plt.imshow(images[idx][0], cmap="gray")
                else:
                    # Einops to convert from CxHxW to HxWxC
                    plt.imshow(einops.rearrange(images[idx], "c h w -> h w c"))

                plt.axis("off")
                idx += 1

    fig.suptitle(title, fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300)
    # Showing the figure

    if show_windowed:
        plt.show()


def show_first_batch(loader, denormalize=False):
    for batch in loader:
        if type(batch) is list:
            show_images(batch[0], denormalize)
        else:
            show_images(batch, denormalize)
        break


def denormalize_images(imgs):
    """Normalize all channels of images to [0, 255], returns a clone of the images"""
    imgs = imgs.clone()
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    imgs = (imgs * 255).type(torch.uint8)
    return imgs


def make_interpolation_sequence(
    interpolator: StochasticInterpolator, X0, X1, Z=None, n_steps=32
):
    """Interpolates between two images X0 and X1 using the provided interpolator

    Params:
        interpolator: The interpolator object
        X0: Tensor of shape N x C x H x W containing the first image
        X1: Tensor of shape N x C x H x W containing the second image
        Z: Tensor of shape N x C x H x W containing the noise to use for interpolation
        n_steps: The number of interpolation steps to take

    Returns:
        image_frames: Tensor of shape T x N x C x H x W containing the interpolated frames
        t: The interpolation times
    """
    # Generate interpolation times
    time_frames = torch.linspace(0, 1, n_steps).to(X0)

    # Get all frames and store in T x N x C x H x W tensor
    image_frames = torch.zeros(
        n_steps, X0.size(0), X0.size(1), X0.size(2), X0.size(3)
    ).to(X0)
    for i, t in enumerate(time_frames):
        T = torch.ones(X0.size(0)).to(X0) * t
        XT = interpolator.get_XT(T, X0, X1, Z)
        image_frames[i] = XT

    show_time_series(image_frames)

    return image_frames, time_frames


def show_time_series(image_frames, save_path=None, denormalize=False):
    """Shows the time series of images as a single image
    Params:
        image_frames: Tensor of shape T x N x C x H x W containing the time series of batch of images
    """
    mosaic = einops.rearrange(image_frames, "t n c h w -> (n h) (t w) c")
    if denormalize:
        mosaic = denormalize_images(mosaic)
    fig = plt.figure(figsize=(16, 8))
    plt.imshow(mosaic.detach().cpu().numpy(), cmap="gray")

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def make_interpolation_gif(
    interpolator: StochasticInterpolator, X0, X1, Z=None, n_steps=32
):
    """Interpolates between two images X0 and X1 using the provided interpolator

    Params:
        interpolator: The interpolator object
        X0: Tensor of shape N x C x H x W containing the first image
        X1: Tensor of shape N x C x H x W containing the second image
        Z: Tensor of shape N x C x H x W containing the noise to use for interpolation
        n_steps: The number of interpolation steps to take

    Returns:
        image_frames: Tensor of shape T x N x C x H x W containing the interpolated frames
        t: The interpolation times
    """
    assert X0.size(0) % 4 == 0, "Batch size must be a multiple of 4"

    # Generate interpolation times
    time_frames = torch.linspace(0, 1, n_steps).to(X0)

    # Get all frames and store in T x N x C x H x W tensor
    image_frames = torch.zeros(
        n_steps, X0.size(0), X0.size(1), X0.size(2), X0.size(3)
    ).to(X0)
    for i, t in enumerate(time_frames):
        T = torch.ones(X0.size(0)).to(X0) * t
        XT = interpolator.get_XT(T, X0, X1, Z)
        image_frames[i] = XT

    animate_batch(image_frames, save_path="interpolated_image.gif")


def animate_batch(image_frames, save_path=None, denormalize=False, fps=10):
    """Takes a TxNxCxHxW tensor and animates it as a gif with the last frame held for extra frames"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    t, n, c, h, w = image_frames.shape

    if n % 4 != 0:
        raise ValueError("Batch size must be a multiple of 4")

    if save_path and save_path.split(".")[-1] != "gif":
        raise ValueError("Save path must be a .gif file, use the .gif extension")

    def make_mosaic(frame_t):
        b1 = 4
        b2 = frame_t.size(0) // b1
        mosaic = einops.rearrange(
            frame_t, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=b1, b2=b2
        )
        if denormalize:
            # Assuming denormalize_images is a function to convert data for display
            mosaic = denormalize_images(mosaic)
        return mosaic.cpu().numpy()

    # Setup initial image
    frame_t = image_frames[0]
    mosaic = make_mosaic(frame_t)
    img_ax = ax.imshow(mosaic, cmap="gray")

    def update(frame_index):
        # Show the last frame for extra frames without recalculating
        if frame_index >= t:
            frame_index = t - 1  # Hold on the last frame

        frame_t = image_frames[frame_index]
        mosaic = make_mosaic(frame_t)
        img_ax.set_data(mosaic)
        ax.set_title(f"Interpolation: Step {frame_index + 1} of {t}")
        return (img_ax,)

    # Extra frames for final image
    end_frames = 40

    # Create animation, adjusting total frames by adding linger frames
    anim = FuncAnimation(
        fig, update, frames=t + end_frames, interval=1000 / fps, blit=True
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)  # Close the figure to prevent display issues

    return anim


def plot_volume(volume):
    """Plot a 3D volume using PyVista"""
    # Create a PyVista mesh
    x, y, z = volume.shape
    x = np.arange(x)
    y = np.arange(y)
    z = np.arange(z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid = pv.StructuredGrid(X, Y, Z)
    values = volume.flatten(order="F")

    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars=values, show_edges=False)
    plotter.show()


def plot_trajectories(solution):
    solution_np = solution.cpu().detach().numpy()
    plt.figure(figsize=(10, 6))
    # Plot each trajectory
    for i in range(solution_np.shape[1]):  # Iterate over each batch
        traj = solution_np[
            :, i, :
        ]  # traj is now [T, Y], assuming Y=2 for 2D trajectories
        plt.plot(
            traj[:, 0], traj[:, 1], color="black", linewidth=0.1, alpha=0.5
        )  # Trajectory line
        # Mark the start of the trajectory
        plt.plot(
            traj[0, 0],
            traj[0, 1],
            "o",
            color="green",
            markersize=5,
            label="Start" if i == 0 else "",
        )
        # Mark the end of the trajectory
        plt.plot(
            traj[-1, 0],
            traj[-1, 1],
            "x",
            color="orange",
            markersize=5,
            label="End" if i == 0 else "",
        )

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Trajectories and Distributions")
    plt.legend()
    plt.axis("equal")
    plt.show()


def plot_2d_slices(volume_data, save_path=None):
    volume_data = einops.rearrange(volume_data, "x y z -> x z y")

    num_cols = 8
    num_rows = 8

    # Calculate global min and max values for consistent color mapping
    global_min = volume_data.min()
    global_max = volume_data.max()

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = np.array(axes)  # Ensure axes is an array for consistent indexing

    # Flatten axes array for easier iteration in case of a single row or column
    if num_rows == 1 or num_cols == 1:
        axes = axes.flatten()

    slice_indices = np.linspace(
        0, volume_data.shape[0] - 1, num_rows * num_cols
    ).astype(int)
    for i, slice_idx in enumerate(slice_indices):
        slice_data = volume_data[slice_idx]
        ax = axes.flat[i]
        # Use global min and max values for color scaling
        im = ax.imshow(
            slice_data, cmap="viridis", vmin=global_min, vmax=global_max, origin="lower"
        )
        ax.set_title(f"Slice {i}")
        ax.axis("off")  # Hide the axis

    # Hide any remaining empty subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes.flat[j])

    plt.tight_layout()
    # Save to png
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
