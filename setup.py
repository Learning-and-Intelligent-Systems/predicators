"""Setup script."""
from setuptools import find_packages, setup

setup(
    name="predicators",
    version="0.1.0",
    packages=find_packages(include=["predicators", "predicators.*"]),
    install_requires=[
        "numpy==1.23.5",
        "pytest==7.1.3",
        "mypy==1.8.0",
        "gym==0.26.2",
        "matplotlib==3.6.2",
        "imageio==2.22.2",
        "imageio-ffmpeg",
        "pandas==1.5.1",
        "torch==2.0.1",
        "scipy==1.9.3",
        "tabulate==0.9.0",
        "dill==0.3.5.1",
        "pyperplan",
        "pathos",
        "pillow==10.3.0",
        "requests",
        "slack_bolt",
        "pybullet>=3.2.0",
        "scikit-learn==1.1.2",
        "graphlib-backport",
        "openai==1.19.0",
        "pyyaml==6.0",
        "pylint==2.14.5",
        "types-PyYAML",
        "lisdf",
        "seaborn==0.12.1",
        "smepy@git+https://github.com/sebdumancic/structure_mapping.git",
        "pg3@git+https://github.com/tomsilver/pg3.git",
        "gym_sokoban@git+https://github.com/Learning-and-Intelligent-Systems/gym-sokoban.git",  # pylint: disable=line-too-long
        "ImageHash",
        "google-generativeai",
        "tenacity",
        "httpx==0.27.0"
    ],
    include_package_data=True,
    extras_require={
        "develop": [
            "pytest-cov==2.12.1", "pytest-pylint==0.18.0", "yapf==0.32.0",
            "docformatter==1.4", "isort==5.10.1", "mypy-extensions==1.0.0"
        ]
    })
