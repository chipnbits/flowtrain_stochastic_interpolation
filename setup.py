from setuptools import setup, find_packages

setup(
  name='flowtrain',
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
      "torchvision",
      "einops",
      "matplotlib",
      "tqdm",
      "GeoGen @ git+https://github.com/eldadHaber/StructuralGeo.git@v1.0"
  ],
)