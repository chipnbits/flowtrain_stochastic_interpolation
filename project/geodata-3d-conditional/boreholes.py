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
            rand_x = torch.rand(1, device=device) * cell_width_x - cell_width_x / 2
            rand_y = torch.rand(1, device=device) * cell_width_y - cell_width_y / 2

            px = center_x + rand_x
            py = center_y + rand_y

            px = torch.clamp(px, min=0, max=X - 1)
            py = torch.clamp(py, min=0, max=Y - 1)

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
        n_bores = torch.randint(8, 32, (1,), device=device).item()
        coords_2d = _jittered_grid_points(size_x, size_y, n_bores, device=device)

        x_coords = coords_2d[:, 0]
        y_coords = coords_2d[:, 1]
        mask[b, 0, x_coords, y_coords, :] = True

    return mask

def make_surface_mask(X: torch.Tensor) -> torch.Tensor:
    """
    Create a boolean mask to identify surface positions in a 3D volume, based on the Air indexing.

    Steps for each batch item:
      1) Set the topmost slice along the z-axis to True, marking it as the surface, to prevent skipping in case of no Air above. 
      2) For any voxel with a value of -1 (Air), mark it and its immediately lower neighbor along the z-axis as part of the surface unless it is the bottommost slice.

    Arguments:
      X: A tensor of shape (B, C, size_x, size_y, size_z), where B is the batch size, C is the number of channels,
         and size_x, size_y, size_z are the dimensions of the volume.

    Returns:
      mask: A boolean tensor of shape (B, 1, size_x, size_y, size_z), where True values indicate surface positions. Elsewhere False.
    """

    B, C, size_x, size_y, size_z = X.shape  

    device = X.device

    # Initialize the mask as False everywhere
    mask = torch.zeros((B, 1, size_x, size_y, size_z), dtype=torch.bool, device=device)
    
    mask[:, 0, :, :, size_z-1] = True
    for b in range(B):
        positions = (X[b, 0] == -1).nonzero(as_tuple=True) 
        if positions[0].numel() > 0: 
            x_coords, y_coords, z_coords = positions
            mask[b, 0, x_coords, y_coords, z_coords] = True
            z_shifted = torch.clamp(z_coords - 1, min=0)
            mask[b, 0, x_coords, y_coords, z_shifted] = True

        #print(f"mask for {b}", mask[b, 0])
    return mask


def make_combined_mask(X: torch.Tensor) -> torch.Tensor:
    """
    Combines the borehole and surface masks into a single boolean mask.

    Arguments:
        X: a tensor of shape (B, C, size_x, size_y, size_z)

    Returns:
        combined_mask: a bool tensor of shape (B, 1, size_x, size_y, size_z)
                       with True for both boreholes and surface features.
    """
    borehole_mask = make_boreholes_mask(X)
    surface_mask = make_surface_mask(X)
    combined_mask = borehole_mask | surface_mask

    return combined_mask

def make_combined_reduced_mask(X: torch.Tensor) -> torch.Tensor:
    """
    Combines the borehole and surface masks into a single boolean mask.

    Arguments:
        X: a tensor of shape (B, C, size_x, size_y, size_z)

    Returns:
        combined_mask: a bool tensor of shape (B, 1, size_x, size_y, size_z)
                       with True for both boreholes and surface features.
    """
    B, C, size_x, size_y, size_z = X.shape  # Extract dimensions
    device = X.device

    # Initialize the mask as False everywhere
    combined_mask = torch.zeros((B, 1, size_x, size_y, size_z), dtype=torch.bool, device=device)

    for b in range(B):
        
        surface_positions = (X[b, 0] == -1).nonzero(as_tuple=True)  # Find where surface starts as indicated by -1 in the data
        if surface_positions[0].numel() > 0:
            x_coords, y_coords, z_coords = surface_positions
            combined_mask [b, 0, x_coords, y_coords, z_coords] = True
            z_shifted = torch.clamp(z_coords - 1, min=0)
            combined_mask [b, 0, x_coords, y_coords, z_shifted] = True

        n_bores = torch.randint(8, 64, (1,), device=device).item() 
        coords_2d = _jittered_grid_points(size_x, size_y, n_bores, device=device)

        # Apply boreholes to the mask
        for x, y in coords_2d:
            if x in x_coords and y in y_coords:
                min_z_index = z_coords[(x_coords == x) & (y_coords == y)].min()
                z_start = max(min_z_index - 16, 0)  
                combined_mask[b, 0, x, y, z_start:] = True

    return combined_mask

def make_boreholes_reduced_mask(X: torch.Tensor) -> torch.Tensor:

    B, C, size_x, size_y, size_z = X.shape  # Extract dimensions
    device = X.device

    mask = torch.zeros((B, 1, size_x, size_y, size_z), dtype=torch.bool, device=device)

    for b in range(B):

        surface_positions = (X[b, 0] == -1).nonzero(as_tuple=True) 
        if surface_positions[0].numel() > 0:
            x_coords, y_coords, z_coords = surface_positions
            mask [b, 0, x_coords, y_coords, z_coords] = True
           

        n_bores = torch.randint(8, 64, (1,), device=device).item()  
        coords_2d = _jittered_grid_points(size_x, size_y, n_bores, device=device)

        for x, y in coords_2d:
            if x in x_coords and y in y_coords:
                min_z_index = z_coords[(x_coords == x) & (y_coords == y)].min()
                z_start = max(min_z_index - 16, 0)  
                mask[b, 0, x, y, z_start:] = True

    return mask
        