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

### Unconditional Training & Inference

- **Training script:** `model_train_inference.py`
- **Inference demo:** Use the `main()` function in the same script to run inference with pretrained weights.

### Conditional Training & Inference

- **Training script:** `model_train_sh_inference_cond.py`
- **Inference demo:** `model_inference_experiments.py` demonstrates conditional generation using pretrained weights.
