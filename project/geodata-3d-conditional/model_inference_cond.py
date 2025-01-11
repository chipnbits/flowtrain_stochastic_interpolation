import os
import torch
import time
import math
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from boreholes import make_boreholes_mask
from geogen.dataset import GeoData3DStreamingDataset
from geogen.model import GeoModel
import geogen.plot as geovis
from model_train import (
    Geo3DStochInterp,
    get_config,
    setup_directories,
    get_data_loader,
    ODEFlowSolver,
)


def model_test(
    dirs, device, model: Geo3DStochInterp = None, n_samples=9, preview_boreholes=True
):
    """
    Draw a random model, create boreholes, run inference to reconstruct the model, save results.
    """
    dataset = GeoData3DStreamingDataset(
        model_resolution=model.data_shape,
        model_bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)),
        dataset_size=100_000,
        device=device,
    )

    synthetic_model = dataset[0].unsqueeze(0)  # [1, 1, X, Y, Z]
    boreholes_mask = make_boreholes_mask(synthetic_model)  # [1, 1, X, Y, Z]
    boreholes = synthetic_model.clone()
    boreholes[~boreholes_mask] = -1  # delete rock around boreholes

    preview_boreholes and show_model_and_boreholes(synthetic_model, boreholes)

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
    sol_decoded = model.decode(sol_tf).detach().cpu() - 1  # [B, 1, X, Y, Z] (bump back down to -1)
    show_solutions(sol_decoded)


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
    geovis.volview(m, plotter=p)

    # Select 2nd pane
    p.subplot(0, 1)
    bh = GeoModel.from_tensor(boreholes.squeeze().detach().cpu())
    geovis.volview(bh, plotter=p)

    p.show()


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

    # Wrapper function to align the call signature
    def dxdt_cond(x, time, *args, **kwargs):
        return model.net.forward(x, ATb=ATb, time=time, *args, **kwargs)

    solver = ODEFlowSolver(model=dxdt_cond, rtol=1e-6)

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
    n_steps = 16
    start = time.time()
    solution = solver.solve(
        X0, t0=0.001, tf=0.999, n_steps=n_steps
    )  # [T, B, C, X, Y, Z]

    print(f"Time taken for inference: {time.time() - start:.2f} seconds")

    return solution


def main() -> None:
    cfg = get_config()
    dirs = setup_directories(cfg)

    inference_device = "cuda"
    relative_checkpoint_path = os.path.join(
        "saved_models",
        "18d-embeddings-conditional",
        "topk-epoch=1371-train_loss=0.0245.ckpt",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
    checkpoint_path = os.path.join(script_dir, relative_checkpoint_path)

    model = Geo3DStochInterp.load_from_checkpoint(
        checkpoint_path, map_location=inference_device
    ).to(inference_device)

    model_test(
        dirs,
        inference_device,
        model=model,
        n_samples=9,
    )


if __name__ == "__main__":
    main()
