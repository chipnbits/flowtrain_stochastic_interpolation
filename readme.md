## 📦 Installation

This repository includes the `flowtrain` package for stochastic interpolation and managing machine learning models. The source code is located in `src/flowtrain`.

To install the package in editable mode, navigate to the project root (where `setup.py` is located) and run:

```bash
    pip install -e .
```

This package also installs a dependency for synthetic geological data generation:  
[`StructuralGeo`](https://github.com/eldadHaber/StructuralGeo/releases/tag/v1.0), which will be installed automatically via `pip`.

---

## StructuralGeo Integration

The `project/` directory contains code and supporting files for training and evaluating flow-based models on 3D StructuralGeo data. These models are designed for stochastic interpolation using flow-matching techniques.

The codebase is built on:

- **PyTorch** for deep learning
- **PyTorch Lightning** for cleaner training loops
- **Weights & Biases (wandb)** for experiment tracking

### Pretrained Models
Pretrained models for both **unconditional** and **conditional** generation at 64³ resolution are available. These weights are downloaded automatically on first use and stored in the `project/*/demo_model/` directory.

If desired, the weights can also be downloaded manually from the [v1.0.0 GitHub release](https://github.com/chipnbits/flowtrain_stochastic_interpolation/releases/tag/v1.0.0):

- [`conditional-weights.ckpt`](https://github.com/chipnbits/flowtrain_stochastic_interpolation/releases/download/v1.0.0/conditional-weights.ckpt) with training run [WandB](https://wandb.ai/sghyseli/cat-embeddings-18d-normed-64cubed?nw=nwusersghyseli)
- [`unconditional-weights.ckpt`](https://github.com/chipnbits/flowtrain_stochastic_interpolation/releases/download/v1.0.0/unconditional-weights.ckpt)

---

## Usage

### Unconditional Model

- **Training:** `project/geodata-3d-unconditional/train_unconditional.py`
Training parameters can be edited via the `get_config()` function in the script, currently set to values used in training the saved demo model. To train on multiple GPUs, use the `--train-devices` flag.

```bash
cd project/geodata-3d-unconditional

python train_inference_unconditional.py --mode train --train-devices 0,1
```

- **Inference:**
Pretrained weights are setup to load automatically, custom training checkpoint available with `--checkpoint_path` flag.


- **Inference demo:** Use the `main()` function in the same script to run inference with pretrained weights.

```bash
cd project/geodata-3d-unconditional

# Saves tensors + PNGs to project/samples/<project_name>/
python train_inference_unconditional.py --mode inference --n-samples 8 --batch-size 2 --seed 100 --save-images --infer-device cuda
```

### Conditional Training & Inference

Conditional training and inference requires an additional step to set up the surface and borehole data from a random generated StructuralGeo sample.

- **Training:** `model_train_sh_inference_cond.py`

Training parameters can be adjusted via the `get_config()` function in the script. Script is set to the training regime used for the pretrained conditional model provided.

```bash
cd project/geodata-3d-conditional

python model_train_sh_inference_cond.py
```


- **Inference:**
A Jupyter notebook `project/geodata-3d-conditional/inference_demo.ipynb` is provided to demonstrate generating conditional data, loading the saved weights, and running inference with the pretrained model. An additional probabilistic analysis using an ensemble of models is also included, making use of compressed data in the `dikes_ptpack.tar.gz` archive.

An automated python script has also been provided to automatically generate synthetic geology, extract boreholde data, and produce reconstructions:

```bash
cd project/geodata-3d-conditional

python model_inference_experiments.py --n-samples 4 --n-scenarios 1

```

[![DOI](https://zenodo.org/badge/891713525.svg)](https://doi.org/10.5281/zenodo.16924445)

