import os

import torch


def save_model(model, path="saved_models/default_save.pth"):
    """Save the model state dictionary to a file.

    Params:
    model: torch.nn.Module
        The model to save.
    path: str
        The full path for model.
    """
    # Ensure the directory exists for the model
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save only the model state dictionary
    torch.save(model.state_dict(), path)


def load_model(model, path, device=None):
    """Load a model state dictionary from a file, handling both new simple and old detailed formats.

    Params:
    model: torch.nn.Module
        The model to load state into.
    path: str
        The full path to the model file.
    device: torch.device, optional
        The device to load the model onto. Defaults to GPU if available, otherwise CPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint
    checkpoint = torch.load(path, map_location=device)

    # Check if the loaded checkpoint is a dictionary with expected keys or just a state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Old style dictionary with more details
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # New simple style or unexpected format; assuming it is just a state_dict
        model.load_state_dict(checkpoint)

    model.to(device)
    print(f"Loaded model from {path}")
    return model
