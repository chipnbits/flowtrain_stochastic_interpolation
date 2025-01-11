import os
import torch
import time

from boreholes import make_boreholes_mask
from geogen.dataset import GeoData3DStreamingDataset
from model_train import (
    Geo3DStochInterp,
    get_config,
    setup_directories,
    get_data_loader,
    ODEFlowSolver,
)


def model_test(dirs, device, model: Geo3DStochInterp = None, n_samples=10):
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
    boreholes = make_boreholes_mask(synthetic_model)  # [1, 1, X, Y, Z]
    
def run_inference(
    device,
    model=None,
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
        return model.forward(x, ATb=ATb, time=time, *args, **kwargs)

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
        # Add batch dimension and expand without duplicating memory
        ATb = ATb.unsqueeze(0).expand(n_samples, *ATb.shape)

    # Run the inference
    t0, tf = 0.001, 0.999
    n_steps = 16
    start = time.time()
    solution = solver.solve(X0, ATb=ATb, t0=0.001, tf=0.999)  # [T, B, C, X, Y, Z]

    print(f"Time taken for inference: {time.time() - start:.2f} seconds")

    return solution


def main() -> None:
    cfg = get_config()
    dirs = setup_directories(cfg)

    inference_device = "cuda:2"
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
    )


if __name__ == "__main__":
    main()
