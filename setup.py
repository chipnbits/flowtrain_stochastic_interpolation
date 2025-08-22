from setuptools import setup, find_packages

setup(
  name='flowtrain',
  version='1.0',
  package_dir={"": "src"},
  packages=find_packages(where="src"),
  description='Generative AI for geogen data with stochastic interpolation',
  author='Simon Ghyselincks',
  author_email='sghyselincks@gmail.com',
  install_requires=[
      "scipy",
      "denoising-diffusion-pytorch",
      "torchdiffeq",
      "numpy",
      "torch",
      "lightning",
      "torchvision",
      "einops",
      "matplotlib",
      "tqdm",
      "seaborn",
      "wandb",
      "GeoGen @ git+https://github.com/eldadHaber/StructuralGeo.git@v1.0"
  ],
)
