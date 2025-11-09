from setuptools import setup, find_packages

setup(
  name='flowtrain',
  version='1.0',
  package_dir={"": "src"},
  packages=find_packages(where="src"),
  description='Generative AI for geogen data with stochastic interpolation',
  author='Simon Ghyselincks',
  author_email='sghyselincks@gmail.com',
)
