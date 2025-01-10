"""
Server does not allow rendering of PyVista, this script is to generate images from saved tensors
on a local machine.
"""

import os
import torch
import functools
import imageio

import torch.nn as nn
import torch.nn.functional as F

import utils
import pyvista as pv

import geogen.plot as geovis
from geogen.model import GeoModel

# Default bounds
DEFAULT_BOUNDS = ((-3840, 3840), (-3840, 3840), (-1920, 1920))

def main():
    proj_name = "cat-embeddings-final-time"
    saved_tensors_dir = os.path.join(os.path.dirname(__file__), "samples", proj_name)
    save_imgs_dir = os.path.join(os.path.dirname(__file__), "images", proj_name)
    save_emb_dir = os.path.join(os.path.dirname(__file__), "embeddings", proj_name)
    os.makedirs(save_imgs_dir, exist_ok=True)

    bounds = ((-1920,1920), (-1920, 1920), (-1920, 1920))
    
    tensor = get_single_tensor(saved_tensors_dir)    
    print(f"tensor shape: {tensor.shape}")
    
    make_views(saved_tensors_dir, save_imgs_dir, view_type='vol', bounds=bounds, )
    make_views(saved_tensors_dir, save_imgs_dir, view_type='cat', bounds=bounds,  )
    # make_views(saved_tensors_dir, save_imgs_dir, animate=True, view_type='vol', bounds=bounds, delay_on_last_frame=3,)
    # make_views(saved_tensors_dir, save_imgs_dir, animate=True, view_type='cat', bounds=bounds, delay_on_last_frame=3,)

def load_embedding(file_path: str):
    """Load the nn.Embedding layer from a saved state dict file."""
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    num_categories, embedding_dim = state_dict['weight'].shape
    embedding = nn.Embedding(num_categories, embedding_dim)
    embedding.load_state_dict(state_dict)
    embedding.weight.requires_grad = False
    
    return embedding
    
def decode_with_loaded_embedding(tensor, embedding: nn.Embedding):
    """Decode the tensor using the reloaded nn.Embedding object."""
    embedding_vectors = F.normalize(embedding.weight, dim=1)  # Normalize embeddings

    # Check if batch dimension is present
    if tensor.ndim == 5:
        B, E, X, Y, Z = tensor.shape
    elif tensor.ndim == 4:
        E, X, Y, Z = tensor.shape
        B = 1  # Set batch size to 1 if there's no batch dimension
        tensor = tensor.unsqueeze(0)  # Add batch dimension for processing
    else:
        raise ValueError("Input tensor must have 4 or 5 dimensions.")


    # Expand dimensions for broadcasting
    tensor_expanded = tensor.unsqueeze(1)  # [B, 1, E, X, Y, Z]
    embedding_expanded = embedding_vectors.view(1, embedding_vectors.shape[0], E, 1, 1, 1)

    # Compute similarity (dot product)
    logits = (tensor_expanded * embedding_expanded).sum(dim=2)  # [B, num_categories, X, Y, Z]
    preds = torch.argmax(logits, dim=1)  # [B, X, Y, Z]

    return preds

def process_folder_of_tensors(dir: str, action: callable, embedding=None):
    """Process a folder of tensors with a given action."""
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".pt"):
                tensor = torch.load(os.path.join(root, file), map_location=torch.device('cpu'))
                if embedding is not None:
                    tensor = decode_with_loaded_embedding(tensor, embedding)
                   
                tensor = tensor - 1
                action(tensor=tensor, filename=file)


def plot_tensor(tensor: torch.Tensor, view_type: str, bounds=None, plotter=None):
    """
    Create a plotter and plot the tensor according to the view_type.
    
    Parameters:
    - tensor (torch.Tensor): The tensor to plot.
    - view_type (str): 'vol' for volumetric view, 'cat' for categorical view.
    - bounds (tuple): Optional bounds for the model.
    - plotter (pv.Plotter): Optional existing plotter to use for plotting.
    
    Returns:
    - plotter (pv.Plotter): The plotter with the tensor plotted.
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    # Validate tensor and bounds
    tensor, bounds = utils._validate_tensor_bounds(tensor, bounds)
    model = GeoModel.from_tensor(data_tensor=tensor, bounds=bounds)

    if plotter is None:
        plotter = pv.Plotter(off_screen=True)

    # Plot according to view type
    if view_type == 'vol':
        geovis.volview(model, plotter=plotter, show_bounds=True, clim=(-1,15))
    elif view_type == 'cat':
        plotter = geovis.categorical_grid_view(model, text_annot=True, off_screen=True)
    else:
        raise ValueError(f"Invalid view_type: {view_type}. Must be 'vol' or 'cat'.")

    return plotter


def save_final_frame(tensor: torch.Tensor, save_dir: str, filename: str, view_type: str, bounds=None):
    """ Save the last frame of a tensor as an image for the specified view_type."""
    
    filename = filename.split(".")[0]
    # tensor = tensor[-1].round()

    # Get the plotter with the tensor plotted
    p = plot_tensor(tensor, view_type, bounds)

    # Save the screenshot
    file_path = os.path.join(save_dir, f"{filename}_{view_type}_final.png")
    p.screenshot(file_path, scale=1, transparent_background=True)
    p.close()


def animate_tensor_to_gif(tensor: torch.Tensor, save_dir: str, filename: str, view_type='vol', bounds=None, delay_on_last_frame: int = 3):
    """
    Animate the ODE process for a tensor over time and save it as a GIF, with camera rotation.

    Args:
        tensor (torch.Tensor): Input tensor.
        save_dir (str): Directory to save the GIF.
        filename (str): Filename for the GIF.
        view_type (str): 'vol' for volumetric view or 'cat' for categorical view.
        bounds (tuple, optional): Bounding box for the plot.
        delay_on_last_frame (int, optional): Number of times to repeat the last frame for delay.
        rotation_angle (int, optional): The amount to rotate the camera for each frame (in degrees).
    """
    filename = filename.split(".")[0]
    gif_path = os.path.join(save_dir, f"{filename}_{view_type}_animation.gif")

    tensor = tensor.round()
    rotation_angle = 180 / (tensor.shape[0]-1) # for looping animation
    fps = tensor.shape[0] // 6

    images = []
    accumulated_angle = 0

    # Iterate over each frame in the tensor
    for i in range(tensor.shape[0]):
        tensor_frame = tensor[i]
        if view_type == 'cat':
            # Hack to stabilize subplots for categorical views
            tensor[...,0,0, -12:-1] = torch.linspace(1, 11, 11).unsqueeze(0)
            tensor.clamp_(-1,15)
        p = plot_tensor(tensor_frame, view_type, bounds)

        if view_type == 'vol':
            # Rotate the camera for each frame
            accumulated_angle += rotation_angle
            p.camera.azimuth = accumulated_angle

        image = p.screenshot(scale=1, transparent_background=False)
        images.append(image)
        p.close()

    # Add delay to the last frame by appending it multiple times
    last_frame = images[-1]
    images.extend([last_frame] * delay_on_last_frame*fps)

    # Save the GIF with imageio
    imageio.mimsave(gif_path, images, fps=fps, loop=0)
    

def make_views(tensor_dir, save_dir, view_type='vol', animate=False, delay_on_last_frame=3, bounds=None, embedding = None):
    """
    Make views (volumetric or categorical) or animate tensors in a directory.

    Parameters:
    - tensor_dir (str): Directory with tensors.
    - save_dir (str): Directory to save the images.
    - view_type (str): 'vol' for volumetric view, 'cat' for categorical view.
    - animate (bool): If True, create an animation.
    - delay_on_last_frame (int): Delay on the last frame in the animation (for GIFs).
    - bounds (tuple): Optional bounds for the model ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
    """
    if animate:
        action = functools.partial(animate_tensor_to_gif, save_dir=save_dir, view_type=view_type, delay_on_last_frame=delay_on_last_frame, bounds=bounds)
    else:
        action = functools.partial(save_final_frame, save_dir=save_dir, view_type=view_type, bounds=bounds)

    process_folder_of_tensors(tensor_dir, action, embedding=embedding)


def get_single_tensor(tensor_dir):
    """Get a single tensor from a directory."""
    for root, dirs, files in os.walk(tensor_dir):
        for file in files:
            if file.endswith(".pt"):
                return torch.load(os.path.join(root, file), map_location=torch.device('cpu'))

    raise FileNotFoundError("No tensors found in the directory.")


if __name__ == "__main__":
    main()
