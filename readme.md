## Install

This repo makes use of a flowtrain package for stochastic interpolation and storing some ML models. The code for the package is in the `src/flowtrain` directory. To install the package navigate to the directory with the `setup.py` file, then run the following command:

```bash
pip install -e .
```

## Usage

The usage of the package for stochastic interpolation is best understood by looking at the example notebooks in the `examples` directory. The numerical example is the best place to start. A 2D MNIST fashion demo shows how to scale up to larger datasets.

## GeoGen usage

The `project` directory contains code and supporting files for stochastic interpolation of 3D GeoGen data. The training and inference can be done through `model_train_inference.py` script.

Note that the code makes use of PyTorch, PyTorch Lightning, and Wandb but can be adapted to a different framework. The important part for stochastic interpolation of categorical data is the embedding, decoding, training steps, and inference using an ODE solver.

The overall structure is designed to allow for learned categorical embeddings, and preparation work for adding EMA and other techniques for better training stability.

The best trained model for 64^3 unconditional data and 64^3 conditional data is available in the `demo_model` folder for each application under the project folder.

### Unconditional Training and Generation
Model was trained using `model_train_inference.py` script. 
The same script is setup to demonstrate inference using the `main()` function. 

### Conditional Training and Generation
Training was done with `model_train_sh_inference_cond.py` script.
A demonstration of inference using the trained weights can be found in `model_inference_experiments.py` script.

```bash

