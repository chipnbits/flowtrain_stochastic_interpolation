import os
import platform
import torch
import time
import math
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from boreholes import make_boreholes_mask, make_surface_mask, make_combined_mask
from geogen.dataset import GeoData3DStreamingDataset

from model_train_sh_inference_cond import (
    Geo3DStochInterp,
    ODEFlowSolver,
)

def get_config() -> dict:
    """
    Generates the entire configuration as a dictionary.

    Returns:
        dict: Configuration dictionary.
    """
    devices = [0, 1, 2]  # List of GPU devices to use

    config = {
        "resume": True,
        "devices": devices,
        # Project configurations
        "project": {
            "name": "lr1_3_combined",
            "root_dir": "/ibex/user/okhmakv/SI_checkpoints_new",
        },
        # Data loader configurations
        "data": {
            "shape": (64, 64, 64),  # [C, X, Y, Z]
            "bounds": (
                (-1920, 1920),
                (-1920, 1920),
                (-1920, 1920),
            ),
            "batch_size": 8,
            "epoch_size": 10_000,
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
            "learning_rate": 5.0e-4,
            "lr_decay": 0.999,
            "gradient_clip_val": 1e-2,
            "accumulate_grad_batches": int(4 * 3 / len(devices)),
            "log_every_n_steps": 16,
            # --- EMA configuration ---
            "use_ema": True,
            "ema_decay": 0.9995,
            "ema_start_step": 0,
            "ema_update_every": 1,
            "ema_update_on_cpu": False,
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
        "example_model_dir": os.path.join(root_dir, "example_model"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs

def model_test(
    dirs,
    device,
    model: Geo3DStochInterp = None,
    n_samples=9,
    preview_boreholes=True,
    run_num=0,
):
    """
    Draw a random model, create boreholes, run inference to reconstruct the model, save results.
    """
    save_dir = dirs["samples_dir"]
    example_model = dirs["example_model_dir"]
    # Put into a run number directory
    save_dir = os.path.join(save_dir, f"run_{run_num}")

    dataset = GeoData3DStreamingDataset(
        model_resolution=model.data_shape,
        model_bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)),
        dataset_size=10,
        device=device,
    )
    

    #synthetic_model = torch.load(os.path.join(example_model, "true_model.pt"),  map_location=device)
    
    synthetic_model = dataset[0].unsqueeze(0)  # [1, 1, X, Y, Z]
    boreholes_mask = make_combined_mask(synthetic_model)  # [1, 1, X, Y, Z]

    boreholes = synthetic_model.clone()
    boreholes[~boreholes_mask] = -1  # delete rock around boreholes

    # Save the original model and boreholes
    save_model_and_boreholes(synthetic_model, boreholes, save_dir)

    X1 = model.embed(synthetic_model)  # [1, C, X, Y, Z]
    ATb = X1.clone()
    ATb[~boreholes_mask.expand(-1, X1.shape[1], -1, -1, -1)] = 0
    print(f"Encoded model shape: {X1.shape}, ATb shape: {ATb.shape}")

    inv_solutions = run_inference(
        device,
        model=model,
        ATb=ATb,
        data_shape=model.data_shape,
        n_samples=n_samples,
        inference_seed=42,
    )  # [T, B, C, X, Y, Z]
    sol_tf = inv_solutions[-1]  # [B, C, X, Y, Z]
    sol_decoded = (
        model.decode(sol_tf).detach().cpu() - 1
    )  # [B, 1, X, Y, Z] (bump back down to -1)

    save_solutions(sol_decoded, save_dir)



def save_model_and_boreholes(model, boreholes, save_dir):
    # Save the tensor data for the model and boreholes
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model, os.path.join(save_dir, "true_model.pt"))
    torch.save(boreholes, os.path.join(save_dir, "boreholes.pt"))


def save_solutions(solutions, save_dir):
    # Save the tensor data for the solutions
    os.makedirs(save_dir, exist_ok=True)
    for i, sol in enumerate(solutions):
        torch.save(sol, os.path.join(save_dir, f"sol_{i}.pt"))


def run_inference(
    device,
    model: Geo3DStochInterp = None,
    ATb=None,
    data_shape=None,
    n_samples=10,
    inference_seed=None,
    save_imgs=True,
) -> None:
    """
    Run conditional inference to generate multiple samples using the trained model.

    Parameters
    ----------
    device : Which device to run the inference on.
    model : The trained loaded model.
    ATb : The conditioning data (boreholes) to use for inference (in embedded space).
    data_shape : The x,y,z shape of data to generate (defaults to model.data_shape).
    n_samples : Number of samples to generate, as a single batch
    inference_seed : Seed for reproducibility
    save_imgs : Whether to save the generated images
    """

    # checkpoint_dir = dirs["checkpoint_dir"]
    # samples_dir = dirs["samples_dir"]
    # os.makedirs(samples_dir, exist_ok=True)

    model.to(device)
    model.eval()

    if data_shape is None:
        data_shape = model.data_shape

    # Wrapper function to add conditioning data to the x,t based ODE
    def dxdt_cond(x, time, *args, **kwargs):
        return model.net.forward(x, ATb=ATb, time=time, *args, **kwargs)

    solver = ODEFlowSolver(model=dxdt_cond, rtol=1e-6)

    # Option for
    generator = (
        torch.Generator(device="cpu").manual_seed(inference_seed)
        if inference_seed
        else None
    )

    X0 = torch.randn(
        n_samples,
        model.embedding_dim,
        *data_shape,
        generator=generator,
    ).to(device)

    assert (
        ATb.shape[-4:] == X0.shape[-4:]
    ), "ATb must match generative data in the last 4 dimensions (c, x, y, z)"

    if ATb is None:
        ATb = torch.zeros_like(X0)
    else:
        ATb = ATb.to(device)
        # expand the batch dim to n_samples
        ATb = ATb.expand(n_samples, -1, -1, -1, -1)

    # Run the inference
    t0, tf = 0.001, 0.999
    n_steps = 150
    start = time.time()
    solution = solver.solve(
        X0, t0=0.001, tf=0.999, n_steps=n_steps
    )  # [T, B, C, X, Y, Z]

    print(f"Time taken for inference: {time.time() - start:.2f} seconds")

    return solution


def save_model_and_boreholes(model, boreholes, save_dir):
    # Save the tensor data for the model and boreholes
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model, os.path.join(save_dir, "true_model.pt"))
    torch.save(boreholes, os.path.join(save_dir, "boreholes.pt"))


def save_solutions(solutions, save_dir):
    # Save the tensor data for the solutions
    os.makedirs(save_dir, exist_ok=True)
    for i, sol in enumerate(solutions):
        torch.save(sol, os.path.join(save_dir, f"sol_{i}.pt"))


def load_model_and_boreholes(save_dir):
    # Load the tensor data for the model and boreholes
    model = torch.load(os.path.join(save_dir, "true_model.pt"))
    boreholes = torch.load(os.path.join(save_dir, "boreholes.pt"))
    return model, boreholes


def load_solutions(save_dir):
    # index all files starting with "sol_" in the save_dir
    sol_files = [f for f in os.listdir(save_dir) if f.startswith("sol_")]
    solutions = [None] * len(sol_files)
    for i, sol_file in enumerate(sol_files):
        solutions[i] = torch.load(os.path.join(save_dir, sol_file))

    # Turn into one tensor wit batch dimension
    return torch.stack(solutions)


def load_model_with_ema_option(
    ckpt_path: str,
    map_location: str = "cpu",
    use_ema: bool = False,
) -> Geo3DStochInterp:
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    model = Geo3DStochInterp.load_from_checkpoint(ckpt_path, map_location=map_location)

    if use_ema and "ema_shadow" in checkpoint:
        print("Applying EMA shadow to model...")
        for name, param in model.named_parameters():
            if name in checkpoint["ema_shadow"]:
                param.data.copy_(checkpoint["ema_shadow"][name].to(map_location))
    elif use_ema:
        print("WARNING: 'ema_shadow' not found in checkpoint. Using regular weights.")

    return model




def main() -> None:
    cfg = get_config()
    dirs = setup_directories(cfg)

    use_ema = True  # or False
    inference_device = "cuda"
    relative_checkpoint_path = os.path.join(
        "/home/okhmakv/SI_checkpoints_new1/15c_64_b8_ac4_lr1_3_64n_combined",
        "topk-epoch=1409-train_loss=0.0176.ckpt",
    )

    model = load_model_with_ema_option(
        ckpt_path=relative_checkpoint_path,
        map_location=inference_device,
        use_ema=use_ema,
    ).to(inference_device)

    
    for i in range(20):
        model_test( 
            dirs,
            inference_device,
            model=model,
            n_samples=6,
            run_num=i,
        )


if __name__ == "__main__":
    main()
