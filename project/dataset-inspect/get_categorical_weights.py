"""
Script to sample multiple batches from GeoData3DStreamingDataset and count
the frequency of categories from -1 to 13.

Assumptions:
- The GeoData3DStreamingDataset returns tensors of shape (B, C, X, Y, Z),
  where C=15 corresponds to categories -1 to 13 mapped as follows:
    - Index 0 -> Category -1
    - Indices 1 to 14 -> Categories 0 to 13
- Categories are one-hot encoded.
"""

import time as clock
import torch

# Synthetic GeoData generation
from geogen.dataset import GeoData3DStreamingDataset, OneHotTransform
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define constants
DATA_SHAPE = (15, 16, 16, 16)  # C=15, spatial=16x16x16
BOUNDS = (
    (-1920, 1920),
    (-1920, 1920),
    (-1920, 1920),
)  # Viewing window bounds (in meters)
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loader(batch_size, device, shuffle=True):
    """
    Initializes the DataLoader for GeoData3DStreamingDataset.

    Args:
        batch_size (int): Number of samples per batch.
        device (str or torch.device): Device to load data on.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: Initialized DataLoader.
    """
    # Initialize the OneHotTransform if needed
    onehot_transform = OneHotTransform()

    # Initialize the dataset
    dataset = GeoData3DStreamingDataset(
        model_resolution=DATA_SHAPE[1:],  # (16, 16, 16)
        model_bounds=BOUNDS,  # ((-1920,1920), ...)
        dataset_size=10_000,  # Number of samples per epoch
        device=device,
        transform=onehot_transform,  # Apply the one-hot transform
    )

    # Initialize the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=16,  # Adjust based on your CPU cores
        pin_memory=True,  # Speeds up data transfer to GPU
    )

    return dataloader


def pool_categorical_weights(dataloader, num_categories=15, category_mapping=None):
    """
    Samples batches from the DataLoader and counts the frequency of each category.

    Args:
        dataloader (DataLoader): DataLoader to sample data from.
        num_categories (int): Number of categories (C).
        category_mapping (dict, optional): Mapping from index to category label.
                                            If None, assumes mapping {0: -1, 1:0, ..., 14:13}.

    Returns:
        dict: Dictionary mapping category labels to their frequencies.
    """
    if category_mapping is None:
        # Default mapping: index 0 -> -1, indices 1-14 -> categories 0-13
        category_mapping = {0: -1}
        for idx in range(1, num_categories):
            category_mapping[idx] = idx - 1

    # Initialize counts
    category_counts = torch.zeros(num_categories, dtype=torch.long)

    total_voxels = 0  # To keep track of total voxels sampled

    # Iterate over DataLoader
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Sampling Batches")):
        # batch shape: (B, C, X, Y, Z)
        # Sum over batch and spatial dimensions to get counts per category
        counts = batch.sum(dim=(0, 2, 3, 4))  # Shape: (C,)
        category_counts += counts.to(torch.long)
        total_voxels += (
            batch.shape[0] * batch.shape[2] * batch.shape[3] * batch.shape[4]
        )

    # Convert counts to frequencies
    frequencies = category_counts.float() / total_voxels

    # Create a dictionary mapping category labels to frequencies
    freq_dict = {}
    for idx in range(num_categories):
        label = category_mapping.get(idx, idx - 1)  # Default mapping if not provided
        freq_dict[label] = frequencies[idx].item()

    return freq_dict


def main():
    """
    Main function to execute the frequency counting.
    """
    # Initialize the DataLoader
    dataloader = get_data_loader(BATCH_SIZE, DEVICE, shuffle=True)

    # Define category mapping: index 0 -> -1, indices 1-14 -> 0-13
    category_mapping = {0: -1}
    for idx in range(1, DATA_SHAPE[0]):
        category_mapping[idx] = idx - 1

    # Perform frequency counting
    freq_dict = pool_categorical_weights(
        dataloader, num_categories=DATA_SHAPE[0], category_mapping=category_mapping
    )

    # Sort the dictionary by category label
    sorted_freq = dict(sorted(freq_dict.items(), key=lambda item: item[0]))

    # Display the frequencies
    print("\nCategory Frequencies:")
    for category, freq in sorted_freq.items():
        print(f"Category {category}: {freq*100:.2f}%")


if __name__ == "__main__":
    start_time = clock.time()
    main()
    end_time = clock.time()
    elapsed = end_time - start_time
    print(f"\nCompleted frequency counting in {elapsed:.2f} seconds.")
