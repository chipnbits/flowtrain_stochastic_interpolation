import os
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

import pyvista as pv
import matplotlib.pyplot as plt

import geogen.plot as geovis
from geogen.model import GeoModel
from geogen.generation import (
    BED_ROCK_VAL,
    SEDIMENT_VALS,
    DIKE_VALS,
    INTRUSION_VALS,
    BLOB_VALS,
)

COLOR_SCHEME = "gist_ncar"


def main():
    make_unconditioned_figures()
    make_dike_realization_figures()
    make_probability_map()

def make_probability_map():
    """ Figure generation from a computed probability map """
    # Get the current directory of this file
    dir = os.path.dirname(__file__)

    # Get all files in the tensor directory
    tensor_dir = os.path.join(dir, "ensemble")
    img_dir = os.path.join(dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    filename = "probability_tensor.pt"
    data = torch.load(os.path.join(tensor_dir, filename), map_location="cpu")
    probability_vector = data.squeeze()

    model, boreholes = load_model_and_boreholes(tensor_dir)
    true_model = model.detach().cpu().squeeze()
    boreholes = boreholes.detach().cpu().squeeze()

    true_grid = get_voxel_grid_from_tensor(
        true_model, bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)), threshold=-0.5
    )
    borehole_grid = get_voxel_grid_from_tensor(
        boreholes, bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)), threshold=-0.5
    )

    dike_probs = probability_vector[[d + 1 for d in DIKE_VALS], :]

    # eps = 1e-8
    # entropy = -torch.sum(
    #     probability_vector * torch.log(probability_vector + eps), dim=1
    # )  # [64, 64, 64]

    print(f"shape of dike_probs: {dike_probs.shape}")

    # Set to spatial coords
    x = np.linspace(-1920, 1920, 64)
    y = np.linspace(-1920, 1920, 64)
    z = np.linspace(-1920, 1920, 64)
    x, y, z = np.meshgrid(x, y, z, indexing="ij")  # Ensure correct shape order
    mesh = pv.StructuredGrid(x, y, z)

    # Extract the first channel
    dike1_data = dike_probs[1].detach().cpu().numpy().ravel(order="F")
    mesh["dike1"] = dike1_data  # Assign to mesh

    dike2_data = dike_probs[2].detach().cpu().numpy().ravel(order="F")
    mesh["dike2"] = dike2_data  # Assign to mesh

    dike1_true = true_grid.copy()
    dike1_true["values"] = np.where(
        ~np.isin(dike1_true["values"], DIKE_VALS[1]), -1, dike1_true["values"]
    )
    dike1_true = dike1_true.threshold(-0.5, all_scalars=True)

    dike2_true = true_grid.copy()
    dike2_true["values"] = np.where(
        ~np.isin(dike2_true["values"], DIKE_VALS[2]), -1, dike2_true["values"]
    )
    dike2_true = dike2_true.threshold(-0.5, all_scalars=True)

    dike1_samples = borehole_grid.copy()
    dike1_samples["values"] = np.where(
        ~np.isin(dike1_samples["values"], DIKE_VALS[1]), -1, dike1_samples["values"]
    )
    dike1_samples = dike1_samples.threshold(-0.5, all_scalars=True)

    dike2_samples = borehole_grid.copy()
    dike2_samples["values"] = np.where(
        ~np.isin(dike2_samples["values"], DIKE_VALS[2]), -1, dike2_samples["values"]
    )
    dike2_samples = dike2_samples.threshold(-0.5, all_scalars=True)

    ########## Plotting ##########

    views = ["isometric", "xz", "xy", "yz"]

    dike_true_list = [dike1_true, dike2_true]
    dike_samples_list = [dike1_samples, dike2_samples]
    n_list = [1, 2]

    mesh = mesh
    n = 1
    c = 1

    for dike_true, dike_samples, n in zip(dike_true_list, dike_samples_list, n_list):
        for view in views:
            p = pv.Plotter(
                shape=(c, 2),
                window_size=(2 * 700, 2 * 400 * c),
                off_screen=True,
                border=False,
            )
            for i in range(c):

                p.subplot(i, 0)
                plot_true_dike_with_samples(dike_true, dike_samples, p, n)
                add_bounds(p)

                if view == "xz":
                    p.view_xz()
                elif view == "xy":
                    p.view_xy()
                elif view == "yz":
                    p.view_yz()
                elif view == "isometric":
                    p.view_isometric()

                p.camera.Zoom(0.9)
                # add title

                p.subplot(i, 1)
                plot_estimated_dike_with_samples(
                    mesh,
                    dike_samples,
                    p,
                    n,
                    contour_values=[0.05, 0.33, 0.62, 0.90],
                    show_bounds=True,
                )

                if view == "xz":
                    p.view_xz()
                elif view == "xy":
                    p.view_xy()
                elif view == "yz":
                    p.view_yz()
                elif view == "isometric":
                    p.view_isometric()

                # zoom out a bit
                p.camera.Zoom(0.85)

            # Save the graphic
            output_path = os.path.join(img_dir, f"probability_map_dike_{view}_{n}.pdf")
            p.save_graphic(output_path)


def plot_true_dike_with_samples(true_dike, sampled_dike, plotter, n=1):
    plotter.add_mesh(
        true_dike,
        scalars=None,
        color="orange",
        show_scalar_bar=False,
        interpolate_before_map=False,
        opacity=0.5,
        label=f"Dike {n} True",
    )
    plotter.add_mesh(
        sampled_dike,
        scalars=None,
        color="red",
        show_scalar_bar=False,
        interpolate_before_map=False,
        opacity=1.0,
        label=f"Dike {n} Bore Samples",
    )


def plot_estimated_dike_with_samples(
    probability,
    sampled_dike,
    plotter,
    n=1,
    contour_values=[0.05, 0.33, 0.62, 0.9],
    show_bounds=True,
):
    plotter.add_mesh(
        sampled_dike,
        scalars=None,
        color="red",
        show_scalar_bar=False,
        interpolate_before_map=False,
        opacity=1.0,
        label=f"Dike {n} Bore Samples",
    )

    contour = probability.contour([0.05, 0.3, 0.6, 0.9], scalars=f"dike{n}")
    plotter.add_mesh(
        contour,
        opacity=0.3,
        cmap="Wistia",
        show_scalar_bar=False,
    )

    bar = plotter.add_scalar_bar(
        f"Probability Contour",
        vertical=False,
        title_font_size=24,
        label_font_size=24,
        fmt="%.2f",
        n_labels=len(contour_values),
    )

    if show_bounds:
        flat_bounds = [-1950, 1950, -1950, 1950, -1950, 1950]
        bounding_box = pv.Box(
            flat_bounds,
        )
        plotter.add_mesh(
            bounding_box, color="black", style="wireframe", line_width=2, opacity=0.2
        )


def make_unconditioned_figures():
    # Get the current directory of this file
    dir = os.path.dirname(__file__)

    # Get all files in the tensor directory
    tensor_dir = os.path.join(dir, "unconditional")
    img_dir = os.path.join(dir, "images")

    make_standalone_scalarbar(img_dir)

    os.makedirs(img_dir, exist_ok=True)  # Ensure the output directory exists
    # Get all files in the tensor directory
    tensor_files = os.listdir(tensor_dir)
    sample_labels = [
        "(a)",
        "(b)",
        "(c)",
    ]

    # Select which unconditional models to use
    file_nums = [114, 130, 171]
    tensor_files = [f"decoded_s100_{num}.pt" for num in file_nums]

    p = make_1x3_subplot_with_single_colorbar(
        sample_labels, tensor_files, tensor_dir, img_dir
    )
    p.save_graphic(
        os.path.join(img_dir, "unconditioned_1x3_subplot_with_single_colorbar.pdf")
    )


def make_dike_realization_figures():
    # Get the current directory of this file
    dir = os.path.dirname(__file__)

    # Get all files in the tensor directory
    tensor_dir = os.path.join(dir, "conditional")
    img_dir = os.path.join(dir, "images")

    os.makedirs(img_dir, exist_ok=True)  # Ensure the output directory exists
    model, boreholes = load_model_and_boreholes(tensor_dir)
    model = model.detach().cpu()
    boreholes = boreholes.detach().cpu()

    model = model.squeeze()
    boreholes = boreholes.squeeze()
    p = make_2x1_model_borehole_plot(model, boreholes)

    p.save_graphic(os.path.join(img_dir, "conditioned_2x1_dike_model_boreholes.pdf"))

    r = 3
    c = 4
    p = make_nxn_dike_realization_plot(tensor_dir, r, c)
    p.save_graphic(os.path.join(img_dir, f"conditioned_{r}x{c}_dike_realizations.pdf"))


def make_2x1_model_borehole_plot(model, boreholes):
    model_grid = get_voxel_grid_from_tensor(
        model, bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)), threshold=-0.5
    )
    borehole_grid = get_voxel_grid_from_tensor(
        boreholes, bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)), threshold=-0.5
    )

    p = pv.Plotter(shape=(2, 1), window_size=(900, 1800), border=False, off_screen=True)
    p.subplot(0, 0)
    p = plot_only_dikes(p, model_grid, show_bounds=True)

    p.subplot(1, 0)
    p = plot_only_dikes(p, borehole_grid, show_bounds=True)
    # add_lighting(p)
    p.link_views()
    return p


def make_nxn_dike_realization_plot(tensor_dir, r=3, c=4):
    # Second plotter look over all samples

    p = pv.Plotter(
        shape=(r, c), window_size=(400 * c, 400 * r), border=False, off_screen=True
    )
    # load all sample files from tensor dir
    tensor_files = os.listdir(tensor_dir)
    sample_files = [f for f in tensor_files if "sample" in f]
    sample_files = sorted(sample_files)

    a = 10
    sample_files = sample_files[a : a + int(r * c)]

    print("Plotting samples: {}".format(len(sample_files)))

    for i, file in enumerate(sample_files):
        p.subplot(i // int(c), i % int(c))
        tensor = torch.load(
            os.path.join(tensor_dir, file), map_location=torch.device("cpu")
        )
        tensor = tensor.squeeze()
        grid = get_voxel_grid_from_tensor(
            tensor, bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)), threshold=-0.5
        )
        p = plot_only_dikes(p, grid, show_skin=True, show_bounds=True)

    return p


def plot_only_dikes(plotter, imagedata, show_skin=True, show_bounds=False):
    grid = imagedata

    filtered_grid = grid.copy()
    filtered_grid["values"] = np.where(
        ~np.isin(filtered_grid["values"], DIKE_VALS), -1, filtered_grid["values"]
    )
    filtered_grid = filtered_grid.threshold(-0.5, all_scalars=True)

    plot_config = get_plot_config()
    plot_config["scalar_bar_args"] = None

    if show_skin:
        skin = grid.extract_surface()
        plotter.add_mesh(
            skin,
            scalars="values",
            **plot_config,
            opacity=0.2,
            show_scalar_bar=False,
        )
    plotter.add_mesh(
        filtered_grid,
        scalars="values",
        **plot_config,
        show_scalar_bar=False,
    )

    if show_bounds:
        flat_bounds = [-1950, 1950, -1950, 1950, -1950, 1950]
        bounding_box = pv.Box(
            flat_bounds,
        )
        plotter.add_mesh(
            bounding_box, color="black", style="wireframe", line_width=2, opacity=0.2
        )

    return plotter


def add_bounds(plotter, flat_bounds=[-1920, 1920, -1920, 1920, -1920, 1920]):
    bounding_box = pv.Box(
        flat_bounds,
    )
    plotter.add_mesh(
        bounding_box, color="black", style="wireframe", line_width=2, opacity=0.2
    )
    plotter.show_bounds(
        grid="back",
        location="outer",
        ticks="outside",
        n_xlabels=4,
        n_ylabels=4,
        n_zlabels=4,
        xtitle="Easting",
        ytitle="Northing",
        ztitle="Elevation",
        bounds=flat_bounds,
        all_edges=True,
        corner_factor=0.5,
        font_size=12,
    )


def load_model_and_boreholes(save_dir):
    # Load the tensor data for the model and boreholes
    model = torch.load(os.path.join(save_dir, "true_model.pt"))
    boreholes = torch.load(os.path.join(save_dir, "boreholes.pt"))
    return model, boreholes


def make_1x3_subplot_with_single_colorbar(
    subplot_labels, tensor_files, tensor_dir, img_dir
):
    shape = (1, 4)
    row_weights = [1]
    col_weights = [1, 1, 1, 0.5]  # Last col is for the scalar bar

    plotter = pv.Plotter(
        shape=shape,
        row_weights=row_weights,
        col_weights=col_weights,
        border=False,
        window_size=(1600, 450),
        off_screen=True,
    )
    plot_config = get_plot_config()

    bounds = ((-1920, 1920), (-1920, 1920), (-1920, 1920))

    for i, file in enumerate(tensor_files[:3]):
        file_path = os.path.join(tensor_dir, file)
        tensor = torch.load(file_path, map_location=torch.device("cpu")) - 1
        model = GeoModel.from_tensor(data_tensor=tensor, bounds=bounds)

        plotter.subplot(0, i)
        mesh = get_voxel_grid_from_model(model, threshold=-0.5)
        plot_config_no_bar = plot_config.copy()
        plot_config_no_bar["scalar_bar_args"] = None

        plotter.add_mesh(
            mesh,
            scalars="values",
            **plot_config_no_bar,
            show_scalar_bar=False,
            interpolate_before_map=False,
        )
        plotter.add_axes(line_width=2)
        plotter.add_text(subplot_labels[i], font_size=12, position="upper_left")

    plotter.subplot(0, 3)
    # Add a single scalar bar to the right
    cb_args = plot_config["scalar_bar_args"].copy()
    cb_args.update(
        {
            "position_x": 0.65,  # far right
            "position_y": 0.01,  # slightly up from bottom
            "width": 0.3,  # narrow bar
            "height": 0.9,  # fill nearly the entire vertical space
        }
    )
    plotter.add_scalar_bar(**cb_args)
    # # manually add a title for scalar bar above the bar
    plotter.add_text("Rock Type", position="upper_edge", font_size=12)
    plotter.link_views()
    add_lighting(plotter)
    return plotter


def make_2x3_subplot_with_single_colorbar(
    subplot_labels, tensor_files, tensor_dir, img_dir
):
    shape = (2, 4)
    row_weights = [1, 1]
    col_weights = [1, 1, 1, 0.5]  # Last col is for the scalar bar
    groups = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (np.s_[:], 3)]

    plotter = pv.Plotter(
        shape=shape,
        row_weights=row_weights,
        col_weights=col_weights,
        groups=groups,
        border=False,
        window_size=(1600, 900),
        off_screen=True,
    )
    plot_config = get_plot_config()

    bounds = ((-1920, 1920), (-1920, 1920), (-1920, 1920))

    for i, file in enumerate(tensor_files[:6]):  # Ensure we only take up to 6 files
        file_path = os.path.join(tensor_dir, file)
        tensor = torch.load(file_path, map_location=torch.device("cpu")) - 1
        model = GeoModel.from_tensor(data_tensor=tensor, bounds=bounds)

        plotter.subplot(i // 3, i % 3)
        mesh = get_voxel_grid_from_model(model, threshold=-0.5)
        plot_config_no_bar = plot_config.copy()
        plot_config_no_bar["scalar_bar_args"] = None

        plotter.add_mesh(
            mesh,
            scalars="values",
            **plot_config_no_bar,
            show_scalar_bar=False,
            interpolate_before_map=False,
        )
        plotter.add_axes(line_width=5)
        plotter.add_text(subplot_labels[i], font_size=12, position="upper_left")

    plotter.subplot(1, 3)
    # Add a single scalar bar to the right
    cb_args = plot_config["scalar_bar_args"].copy()
    cb_args.update(
        {
            "position_x": 0.65,  # far right
            "position_y": 0.05,  # slightly up from bottom
            "width": 0.3,  # narrow bar
            "height": 0.90,  # fill nearly the entire vertical space
        }
    )
    plotter.add_scalar_bar(**cb_args)
    # # manually add a title for scalar bar above the bar
    plotter.add_text("Rock Type", position="upper_edge", font_size=12)
    plotter.link_views()
    add_lighting(plotter)

    return plotter


def make_standalone_scalarbar(output_path):
    """Generate a standalone scalar bar and save it as a PDF."""
    plotter = pv.Plotter(window_size=(300, 900), off_screen=True)

    # add single invsivble point
    dummy_mesh = pv.PolyData(np.array([[0, 0, 0]]))
    dummy_mesh["values"] = [0]
    plot_config = get_plot_config()
    plot_config["scalar_bar_args"] = None
    plotter.add_mesh(
        dummy_mesh,
        scalars="values",
        **plot_config,
        show_scalar_bar=False,
        opacity=0,  # Make the point invisible
        show_edges=False,
    )

    plot_config = get_plot_config()
    cb_args = plot_config["scalar_bar_args"].copy()
    cb_args.update(
        {
            "position_x": 0.65,  # Far right
            "position_y": 0.05,  # Slightly up from bottom
            "width": 0.3,  # Narrow bar
            "height": 0.90,  # Fill nearly the entire vertical space
        }
    )

    # Add scalar bar
    plotter.add_scalar_bar(**cb_args)

    # Manually add title
    plotter.add_text("Rock Type", position="upper_edge", font_size=12)

    # Save to PDF
    output_file = os.path.join(output_path, "scalarbar_tall.pdf")
    plotter.save_graphic(output_file)
    print(f"Scalar bar saved to {output_file}")

    plotter = pv.Plotter(window_size=(300, 450), off_screen=True)
    # add single invsivble point
    dummy_mesh = pv.PolyData(np.array([[0, 0, 0]]))
    dummy_mesh["values"] = [0]
    plot_config = get_plot_config()
    plot_config["scalar_bar_args"] = None
    plotter.add_mesh(
        dummy_mesh,
        scalars="values",
        **plot_config,
        show_scalar_bar=False,
        opacity=0,  # Make the point invisible
        show_edges=False,
    )

    plot_config = get_plot_config()
    cb_args = plot_config["scalar_bar_args"].copy()
    cb_args.update(
        {
            "position_x": 0.65,  # far right
            "position_y": 0.01,  # slightly up from bottom
            "width": 0.3,  # narrow bar
            "height": 0.9,  # fill nearly the entire vertical space
        }
    )

    plotter.add_scalar_bar(**cb_args)
    # # manually add a title for scalar bar above the bar
    plotter.add_text("Rock Type", position="upper_edge", font_size=12)

    # Save to PDF
    output_file = os.path.join(output_path, "scalarbar_short.pdf")
    plotter.save_graphic(output_file)
    print(f"Scalar bar saved to {output_file}")


def add_lighting(plotter):
    # dim overall lighting
    light = pv.Light(position=(5000, 5000, -5000), color="white", intensity=0.30)
    light.positional = True
    plotter.add_light(light)


def get_plot_config():
    # color_range = (-1, 13)
    color_range = (0, 13)
    clim = (color_range[0] - 0.5, color_range[1] + 0.5)
    n_colors = color_range[1] - color_range[0] + 1
    my_cmap = plt.get_cmap(COLOR_SCHEME, n_colors)

    annotations = {}
    # annotations[float(-1)] = "Air"
    annotations[float(BED_ROCK_VAL)] = "Basement"

    for val in SEDIMENT_VALS:
        annotations[float(val)] = "Sedimentary"

    for val in DIKE_VALS:
        annotations[float(val)] = "Planar Dikes"

    for val in INTRUSION_VALS:
        annotations[float(val)] = "Magma Intrusion"

    for val in BLOB_VALS:
        annotations[float(val)] = "Minerals"

    plot_config = {
        "cmap": my_cmap,
        # Shift so each integer is the center of a bin:
        "clim": clim,
        "n_colors": n_colors,
        # Provide annotation dict for -1..15
        "annotations": annotations,
        # Optional: scalar bar styling
        "scalar_bar_args": {
            "label_font_size": 18,
            "vertical": True,
            "n_labels": 0,
            "position_x": 0.90,
            "position_y": 0.1,
            "width": 0.10,
            "height": 0.8,
            # You can omit n_labels here,
            # as the annotations override the default tick labeling
        },
    }
    return plot_config


def setup_plot(model: GeoModel, plotter: Optional[pv.Plotter] = None, threshold=-0.5):
    if plotter is None:
        plotter = pv.Plotter()

    if np.all(np.isnan(model.data)):
        plotter.add_text("No data to show, all values are NaN.", font_size=20)
        mesh = None
    else:
        mesh = get_voxel_grid_from_model(model, threshold)

    plot_config = get_plot_config()
    return plotter, mesh, plot_config


def volview(
    model: GeoModel,
    plotter: Optional[pv.Plotter] = None,
    threshold=-0.5,
    show_bounds=False,
    clim=None,
) -> pv.Plotter:
    """
    Visualize a volumetric view of the geological model with an optional bounding box.

    Parameters
    ----------
    model : GeoModel
        The geological model to be visualized. It contains the data and resolution information.
    plotter : pv.Plotter, optional
        The PyVista plotter to use for rendering the visualization. If not provided, a new one is created.
    threshold : float, optional
        Threshold value used to filter the voxel grid. Default is -0.5, values below this threshold are not shown.
    show_bounds : bool, optional
        If True, display the axis-aligned bounds and tick marks of the model. Default is False.

    Returns
    -------
    pv.Plotter
        The PyVista plotter object with the volumetric view rendered.
    """
    plotter, mesh, plot_config = setup_plot(model, plotter, threshold)
    if mesh is None:
        return plotter

    if clim:
        plot_config["clim"] = clim

    plotter.add_mesh(
        mesh, scalars="values", **plot_config, interpolate_before_map=False
    )
    plotter.add_axes(line_width=5)

    flat_bounds = [item for sublist in model.bounds for item in sublist]

    if show_bounds:
        plotter.show_bounds(
            mesh=mesh,
            grid="back",
            location="outer",
            ticks="outside",
            n_xlabels=4,
            n_ylabels=4,
            n_zlabels=4,
            xtitle="Easting",
            ytitle="Northing",
            ztitle="Elevation",
            bounds=flat_bounds,
            all_edges=True,
            corner_factor=0.5,
            font_size=12,
        )

    # flat_bounds = [item for sublist in model.bounds for item in sublist]
    # bounding_box = pv.Box(flat_bounds)
    # plotter.add_mesh(bounding_box, color="black", style="wireframe", line_width=1)

    return plotter


def get_voxel_grid_from_model(model, threshold=None):
    """
    Convert the geological model's data into a voxel grid for visualization. The voxel grid contains discrete rock types.

    Parameters
    ----------
    model : GeoModel
        The geological model containing rock type data and resolution.
    threshold : float, optional
        Threshold value used to filter the voxel grid. Default is None, meaning no filtering will occur.

    Returns
    -------
    pyvista.ImageData
        The voxel grid representation of the geological model, with discrete values for rock types.
    """
    if model.data is None or model.data.size == 0:
        raise ValueError(
            "Model data is empty or not computed, no data to show. Use compute model first."
        )
    if not all(res > 1 for res in model.resolution):
        raise ValueError(
            "Voxel grid requires a model resolution greater than 1 in each dimension."
        )

    # Create a padded grid with n+1 nodes and node spacing equal to model sample spacing
    dimensions = tuple(x + 1 for x in model.resolution)
    spacing = tuple(
        (x[1] - x[0]) / (r - 1) for x, r in zip(model.bounds, model.resolution)
    )
    # pad origin with a half cell size to center the grid
    origin = tuple(x[0] - cs / 2 for x, cs in zip(model.bounds, spacing))

    # Create a structured grid with n+1 nodes in each dimension forming n^3 cells
    grid = pv.ImageData(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
    )
    # Necessary to reshape data vector in Fortran order to match the grid
    grid["values"] = model.data.reshape(model.resolution).ravel(order="F")
    grid = grid.threshold(threshold, all_scalars=True)
    return grid


def get_voxel_grid_from_tensor(
    data, bounds=((-1920, 1920), (-1920, 1920), (-1920, 1920)), threshold=None
):
    """ """
    assert data.ndim == 3, "Data must be 3D"
    dims = data.shape

    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Create a padded grid with n+1 nodes and node spacing equal to model sample spacing
    dimensions = tuple(x + 1 for x in dims)
    spacing = tuple((x[1] - x[0]) / (r - 1) for x, r in zip(bounds, dims))
    # pad origin with a half cell size to center the grid
    origin = tuple(x[0] - cs / 2 for x, cs in zip(bounds, spacing))

    # Create a structured grid with n+1 nodes in each dimension forming n^3 cells
    grid = pv.ImageData(
        dimensions=dimensions,
        spacing=spacing,
        origin=origin,
    )
    # Necessary to reshape data vector in Fortran order to match the grid
    grid["values"] = data.flatten(order="F")
    grid = grid.threshold(threshold, all_scalars=True)

    return grid


if __name__ == "__main__":
    main()
