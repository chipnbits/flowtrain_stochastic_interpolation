""" Boreholing module for 3D conditional simulation. """

import torch

import torch
import math

def _jittered_grid_points(X, Y, n_bores, device="cpu"):
    """
    Generate 'jittered grid' 2D points (x, y) within a X-by-Y area.
    Returns a LongTensor of shape (n_points, 2).
    """
    # Number of cells in x/y to approximate n_bores
    n_x = int(math.floor(math.sqrt(n_bores)))
    n_y = int(math.ceil(n_bores / n_x))

    # Cell size
    cell_width_x = X / n_x
    cell_width_y = Y / n_y

    points = []
    for i in range(n_x):
        for j in range(n_y):
            center_x = (i + 0.5) * cell_width_x
            center_y = (j + 0.5) * cell_width_y
            # Jitter around the cell center
            rand_x = torch.rand(1, device=device) * cell_width_x - cell_width_x/2
            rand_y = torch.rand(1, device=device) * cell_width_y - cell_width_y/2

            px = center_x + rand_x
            py = center_y + rand_y

            px = torch.clamp(px, min=0, max=X-1)
            py = torch.clamp(py, min=0, max=Y-1)

            points.append((px.item(), py.item()))

    points = points[:n_bores]
    # Convert to tensor of shape (n_points, 2)
    points_tensor = torch.tensor(points, dtype=torch.long, device=device)
    return points_tensor

def make_boreholes_mask(X: torch.Tensor) -> torch.Tensor:
    """
    Create a boolean mask of shape (B, 1, X, Y, Z) with 'vertical boreholes'.
    
    For each batch item:
      1) Randomly choose n_bores between 8 and 64
      2) Generate jittered 2D points in the (X, Y) plane
      3) Mark the entire Z-depth at those (x, y) positions as True
    
    Arguments:
      X: a tensor of shape (B, C, size_x, size_y, size_z)
    
    Returns:
      mask: a bool tensor of shape (B, 1, size_x, size_y, size_z)
            with True in the borehole positions, False elsewhere.
    """
    B, C, size_x, size_y, size_z = X.shape
    device = X.device
    mask = torch.zeros((B, 1, size_x, size_y, size_z), dtype=torch.bool, device=device)

    for b in range(B):
        n_bores = torch.randint(2, 16, (1,), device=device).item()
        coords_2d = _jittered_grid_points(size_x, size_y, n_bores, device=device)
        
        x_coords = coords_2d[:, 0]
        y_coords = coords_2d[:, 1]
        mask[b, 0, x_coords, y_coords, :] = True

    return mask

