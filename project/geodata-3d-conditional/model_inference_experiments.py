import os
import torch
import time
import math
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from boreholes import make_boreholes_mask, make_combined_mask, make_surface_mask
from geogen.dataset import GeoData3DStreamingDataset
from geogen.model import GeoModel
import geogen.plot as geovis
from model_train_ema_mixedpres_lowvram import (
    Geo3DStochInterp,
    setup_directories,
    get_data_loader,
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
            "name": "kaust-training",
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

def create_cond_data(
    save_dir,
    cond_data_folder_title,
    device,
    num_folders,
):
    for i in range(num_folders):    
        run_dir = os.path.join(save_dir, f"{cond_data_folder_title}_{i}")

        dataset = GeoData3DStreamingDataset(
            model_resolution=(64,64,64),
            model_bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)),
            dataset_size=100_000,
            device=device,
        )

        synthetic_model = dataset[0].unsqueeze(0)  # [1, 1, X, Y, Z]
        boreholes_mask = make_combined_mask(synthetic_model)  # [1, 1, X, Y, Z]
        boreholes = synthetic_model.clone()
        boreholes[~boreholes_mask] = -1  # delete rock around boreholes

        # Save the original model and boreholes
        save_model_and_boreholes(synthetic_model, boreholes, run_dir)


def run_inference(
    device,
    model: Geo3DStochInterp = None,
    ATb=None,
    data_shape=None,
    n_samples=10,
    inference_seed=None,
    save_imgs=True,
) -> None:

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
    n_steps = 16
    start = time.time()
    solution = solver.solve(
        X0, t0=0.0001, tf=0.9999, n_steps=n_steps
    )  # [T, B, C, X, Y, Z]    
    print(f"Time taken for inference: {time.time() - start:.2f} seconds")

    return solution

def populate_solutions(
    save_dir,
    cond_data_folder_title,
    device,
    model: Geo3DStochInterp = None,
    inference_method= run_inference,
    n_samples_each=9,
    batch_size=16,
    sample_title="run", 
    make_extended_solutions=False,
):
    # Find all the folders with the cond_data_folder_title within save_dir
    cond_data_folders = [
        f for f in os.listdir(save_dir) if f.startswith(cond_data_folder_title)
    ]
    
    for folder in cond_data_folders:
        folder_path = os.path.join(save_dir, folder)
        geomodel, boreholes = load_model_and_boreholes(folder_path)
        
        # Send to same device as model
        geomodel = geomodel.to(device)
        boreholes = boreholes.to(device)
        # Reconstruct mask (air above, rocks below)
        boreholes_mask = (boreholes != -1) | (geomodel == -1)
        
        # Embed the ground truth model        
        X1 = model.embed(geomodel)
        ATb = X1.clone()

        mask_boreholes = boreholes_mask.expand(-1, X1.shape[1], -1, -1, -1)
        ATb = X1 * mask_boreholes 
        
        # Split into batches
        n_batches = math.ceil(n_samples_each / batch_size)
        
        samples_remaining = n_samples_each
        for i in range(n_batches):
            n_samples = min(samples_remaining, batch_size)
            samples_remaining -= n_samples
            
            inv_solutions = inference_method(
                device,
                model=model,
                ATb=ATb,
                data_shape=model.data_shape,
                n_samples=n_samples,
                inference_seed=42+i,
            )
            
            sol_tf = inv_solutions[-1]  # [B, C, X, Y, Z]

            sol_save = (
                model.decode(sol_tf).detach().cpu() - 1
            )  # [B, 1, X, Y, Z] (bump back down to -1)
            save_solutions(sol_save, folder_path, sample_title, start_index=i*batch_size)
            
            if make_extended_solutions:
                # Try exteneded integration for the solutions and save a version of them
                EXTRA_CONVERGENCE_STEPS = 5
                print(f"Running extended integration for {EXTRA_CONVERGENCE_STEPS} steps")
                
                for j in range(1, EXTRA_CONVERGENCE_STEPS):
                    # Push solution through last 1% of flow again five times
                    sol_extended = extend_solutions(model, sol_tf, ATb) # [B, C, X, Y, Z] -> [T, B, C, X, Y, Z]
                    sol_tf = sol_extended[-1]  # [T, B, C, X, Y, Z] -> [B, C, X, Y, Z]
                    sol_decoded = sol_tf.detach().cpu() - 1
                    save_solutions(sol_decoded, folder_path, f"{sample_title}_extended_{j}steps", start_index=i*batch_size)
        
def extend_solutions(model, initial_solutions, ATb):
    """
    Extend the solutions by n_steps.
    """
    ATb_expanded = ATb.expand(initial_solutions.shape[0], -1, -1, -1, -1) # Same batch size

    # Wrapper function to add conditioning data to the x,t based ODE
    def dxdt_cond(x, time, *args, **kwargs):
        return model.net.forward(x, ATb=ATb_expanded, time=time, *args, **kwargs)

    solver = ODEFlowSolver(model=dxdt_cond, rtol=1e-8)
    
    extended_solutions = solver.solve(
        initial_solutions, t0=0.99, tf=0.9999, n_steps=8
    )

    return extended_solutions


def show_solutions(solutions):
    n_samples = solutions.shape[0]
    n_cols = math.ceil(n_samples**0.5)
    n_rows = math.ceil(n_samples / n_cols)
    p2 = pv.Plotter(shape=(n_rows, n_cols))
    for i in range(n_samples):
        p2.subplot(i // n_cols, i % n_cols)
        geovis.volview(GeoModel.from_tensor(solutions[i]), plotter=p2)

    p2.show()

def show_model_and_boreholes(model, boreholes):
    """
    Plot the model and boreholes side by side.
    """
    # Make two pane pyvista plot
    p = BackgroundPlotter(shape=(1, 2))

    # Plot the synthetic model
    p.subplot(0, 0)
    m = GeoModel.from_tensor(model.squeeze().detach().cpu())
    geovis.volview(m, plotter=p, show_bounds=True)

    # Select 2nd pane
    p.subplot(0, 1)
    bh = GeoModel.from_tensor(boreholes.squeeze().detach().cpu())
    geovis.volview(bh, plotter=p, show_bounds=True)

    p.show()


def save_model_and_boreholes(model, boreholes, save_dir):
    # Save the tensor data for the model and boreholes
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model, os.path.join(save_dir, "true_model.pt"))
    torch.save(boreholes, os.path.join(save_dir, "boreholes.pt"))


def save_solutions(solutions, save_dir, sample_title = "sol", start_index = 0):
    # Save the tensor data for the solutions    
    os.makedirs(save_dir, exist_ok=True)
    for i, sol in enumerate(solutions):
        torch.save(sol, os.path.join(save_dir, f"{sample_title}_{i+start_index}.pt"))


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


def load_run_display(run_num):
    relative_sample_path = os.path.join(
        "samples",
        "15d-conditional-64x64x64-ema-mixedpres-lowvram",
        f"run_{run_num}",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
    sample_path = os.path.join(script_dir, relative_sample_path)

    model, boreholes = load_model_and_boreholes(sample_path)
    solutions = load_solutions(sample_path)

    show_model_and_boreholes(model, boreholes)
    show_solutions(solutions)

def ensemble_analysis():
    cfg = get_config()
    dirs = setup_directories(cfg)

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
    samples_dir = dirs["samples_dir"]
    samples_dir = os.path.join(samples_dir, "ensemble_method_18")

    sols_decoded = load_solutions(
        samples_dir, start_flag="sample_"
    )  # [B, 1, 64, 64, 64]

    # model_params = {
    #     "dim": 8,  # Base number of hidden channels in model
    #     "dim_mults": (
    #         1,
    #         2,
    #     ),  # Multipliers for hidden dims in each superblock, total 2x downsamples = len(dim_mults)-1
    #     "data_channels": 1,  # Data clamped down to fit categorical count
    #     "dropout": 0.1,  # Optional network dropout
    #     "self_condition": False,  # Optional conditioning on input data
    #     "time_sin_pos": False,  # Use fixed sin/cos positional embeddings for time
    #     "time_resolution": 1024,  # Resolution of time (number of random Fourier features)
    #     "time_bandwidth": 1000.0,  # Starting bandwidth of fourier frequencies, f ~ N(0, time_bandwidth)
    #     "time_learned_emb": True,  # Learnable fourier freqs and phases
    #     "attn_enabled": True,  # Enable or disable self attention before each (down/up sample) also feeds skip connections
    #     "attn_dim_head": 32,  # Size of attention hidden dimension heads
    #     "attn_heads": 4,  # Number of chunks to split hidden dimension into for attention
    #     "full_attn": None,  # defaults to full attention only for inner most layer final down, middle, first up
    #     "flash_attn": False,  # For high performance GPUs https://github.com/Dao-AILab/flash-attention
    # }

    # decoder = Geo3DStochInterp(
    #     data_shape=(16, 16, 16), num_categories=15, embedding_dim=15, **model_params
    # )
    # sols_decoded = decoder.decode(sols).detach().cpu() - 1  # [B, 1, 64, 64, 64]

    model, boreholes = load_model_and_boreholes(samples_dir)
    show_model_and_boreholes(model, boreholes)
    show_solutions(sols_decoded)

    num_categories = 15
    sols_one_hot = (
        torch.nn.functional.one_hot(sols_decoded + 1, num_categories)
        .permute(0, 4, 1, 2, 3)
        .float()
    )  # [B, 15, 64, 64, 64]
    probability_vector = sols_one_hot.mean(dim=0, keepdim=True)  # [1, 15, 64, 64, 64]

    eps = 1e-8
    entropy = -torch.sum(
        probability_vector * torch.log(probability_vector + eps), dim=1
    )  # [1, 64, 64, 64]
    entropy = entropy.squeeze(0)  # Remove batch dim -> [64, 64, 64]

    most_probable = torch.argmax(probability_vector, dim=1)  # [1, 64, 64, 64]
    most_probable = most_probable.squeeze(0) - 1  # Remove batch dim -> [64, 64, 64]

    # Clip out the air pixels
    entropy_masked = entropy.clone()
    entropy_masked[most_probable == -1] = -1

    p = pv.Plotter(shape=(1, 3))

    # Wrap most probable in GeoModel and plot
    p.subplot(0, 0)
    mp = GeoModel.from_tensor(most_probable)
    p = geovis.nsliceview(model=mp, plotter=p)

    p.subplot(0, 1)
    p = geovis.volview(model=mp, plotter=p)

    # Set to second pane
    p.subplot(0, 2)
    p = geovis.plot_array(data=entropy_masked, plotter=p)
    p.link_views()
    p.show()

    # Expand solutions out using 1 hot along c
    print("")

def inspect_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
    save_dir = os.path.join(script_dir, "samples", "kaust-training")
    cond_data_folder_title = "ensemble_method"
    cond_data_folders = [
        f for f in os.listdir(save_dir) if f.startswith(cond_data_folder_title)
    ]
    print(cond_data_folders)

    for folder in cond_data_folders:
        folder_path = os.path.join(save_dir, folder)
        geomodel, boreholes = load_model_and_boreholes(folder_path)
        show_model_and_boreholes(geomodel, boreholes)

        # Wait for user input to continue, press any key to continue
        print(f"Showing model and boreholes for {folder}")
        input("Press Enter to continue...")

def main() -> None:
    cfg = get_config()
    dirs = setup_directories(cfg)

    use_ema = True  # or False
    inference_device = "cuda:1"
    relative_checkpoint_path = os.path.join(
        "saved_models",
        "kaust-training",
        "combined_646464_topk-epoch=1493-train_loss=0.0160.ckpt"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
    checkpoint_path = os.path.join(script_dir, relative_checkpoint_path)

    model = load_model_with_ema_option(
        ckpt_path=checkpoint_path,
        map_location=inference_device,
        use_ema=use_ema,
    ).to(inference_device)

    cond_data_folder_title="normal_vs_extended"

    # # Create the conditioning data
    # num_folders = 10
    # create_cond_data(
    #     save_dir=dirs["samples_dir"],
    #     cond_data_folder_title=cond_data_folder_title,
    #     device=inference_device,
    #     num_folders=num_folders,
    # )
    
    # Populate the solutions
    populate_solutions(
        save_dir=dirs["samples_dir"],
        cond_data_folder_title=cond_data_folder_title,
        device=inference_device,
        model=model,
        inference_method=run_inference,
        n_samples_each=9,
        sample_title="sample",
        make_extended_solutions=True,
    )
    


if __name__ == "__main__":
    main()
    # load_run_display(0)
