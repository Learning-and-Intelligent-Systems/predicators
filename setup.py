
"""Setup script.
Usage examples:
    pip install -e .
    pip install -e .[develop]
"""
from setuptools import find_packages, setup

setup(name="predicators",
      version="0.1.0",
      # Tried:
      # packages=find_packages(include=["predicators.src.*"]),
      # packages=find_packages(include=["src.*"]),
      # package_dir = {"": "src"}
      packages=find_packages(include=["predicators", "predicators.src.*"]),
      install_requires=[
          "numpy>=1.22.3", "pytest", "gym==0.21.0", "matplotlib", "imageio",
          "imageio-ffmpeg", "pandas", "torch", "scipy", "tabulate", "dill",
          "pyperplan", "pathos", "requests", "slack_bolt", "pybullet>=3.2.0",
          "sklearn", "pyqt5", "graphlib-backport", "openai", "pyyaml",
          "types-PyYAML"
      ],
      include_package_data=True,
      extras_require={
          "develop": [
              "pytest-cov>=2.12.1", "pytest-pylint>=0.18.0", "yapf==0.32.0",
              "docformatter", "isort",
              "mypy@git+https://github.com/python/mypy.git@9bd651758e8ea2494" +
              "837814092af70f8d9e6f7a1"
          ]
      })