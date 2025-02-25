
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import torch
import os
import math
import gdown
from geogen.model import GeoModel
import geogen.plot as geovis



def load_model_and_boreholes(save_dir):
    # Load the tensor data for the model and boreholes
    model = torch.load(os.path.join(save_dir, "true_model.pt"),  map_location=torch.device("cpu"))
    boreholes = torch.load(os.path.join(save_dir, "boreholes.pt"),  map_location=torch.device("cpu"))
    return model, boreholes

def load_solutions(save_dir):
    # index all files starting with "sol_" in the save_dir
    sol_files = [f for f in os.listdir(save_dir) if f.startswith("sol_")]
    solutions = [None] * len(sol_files)
    for i, sol_file in enumerate(sol_files):
        solutions[i] = torch.load(os.path.join(save_dir, sol_file),  map_location=torch.device("cpu"))

    # Turn into one tensor wit batch dimension
    return torch.stack(solutions)

def load_run_display(file_path, run_num):
    relative_sample_path = os.path.join(
        file_path,
        run_name,
    )

    model, boreholes = load_model_and_boreholes(relative_sample_path)
    solutions = load_solutions(relative_sample_path)

    show_model_and_boreholes(model, boreholes)
    #show_categorical_grid_view(model, boreholes)
    m = GeoModel.from_tensor(model.squeeze().detach().cpu())
    #show_slised_view(m)

    show_solutions(solutions)

    #for i in range(solutions.shape[0]):
    #    show_slised_view(GeoModel.from_tensor(solutions[i]))

def show_solutions(solutions):
    n_samples = solutions.shape[0]
    n_cols = math.ceil(n_samples**0.5)
    n_rows = math.ceil(n_samples / n_cols)
    p2 = pv.Plotter(shape=(n_rows, n_cols))
    for i in range(n_samples):
        p2.subplot(i // n_cols, i % n_cols)
        geovis.volview(GeoModel.from_tensor(solutions[i]), plotter=p2)
    p2.show()

    for i in  range(n_samples):
        geovis.categorical_grid_view(GeoModel.from_tensor(solutions[i])).show()
       
def show_categorical_grid_view(model, boreholes):


    m = GeoModel.from_tensor(model.squeeze().detach().cpu())
    geovis.categorical_grid_view(m).show()
 
    bh = GeoModel.from_tensor(boreholes.squeeze().detach().cpu())
    geovis.categorical_grid_view(bh).show()

def show_slised_view(model):
    #geovis.onesliceview(m).show()
    geovis.nsliceview(model).show()



def show_model_and_boreholes(model, boreholes):
    """
    Plot the model and boreholes side by side.
    """
    # Make two pane pyvista plot
    p = BackgroundPlotter(shape=(1, 2))

    #Plot the synthetic model
    p.subplot(0, 0)
    m = GeoModel.from_tensor(model.squeeze().detach().cpu())
    #geovis.volview(m).show()
    geovis.volview(m, plotter=p, show_bounds=True)
    # Select 2nd pane
    p.subplot(0, 1)
    bh = GeoModel.from_tensor(boreholes.squeeze().detach().cpu())
    geovis.volview(bh, plotter=p, show_bounds=True)

    print("plotting the model")
    p.show()


file_path = "/Users/user/Documents/conditional_inference/updated_results/64_cubed/1200epoch/lr1_3_combined1200epoch"
run_name = "run_new_4"


load_run_display(file_path, run_name)
