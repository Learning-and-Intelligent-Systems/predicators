"""Setup script."""
from setuptools import find_packages, setup

setup(
    name="predicators",
    version="0.1.0",
    packages=find_packages(include=["predicators", "predicators.*"]),
    install_requires=[
        "numpy>=1.22.3",
        "pytest",
        "mypy",
        "gym==0.26.2",
        "matplotlib",
        "imageio",
        "imageio-ffmpeg",
        "pandas",
        "torch==2.0.1",
        "scipy",
        "tabulate",
        "dill",
        "pyperplan",
        "pathos",
        "requests",
        "slack_bolt",
        "pybullet>=3.2.0",
        "scikit-learn",
        "graphlib-backport",
        "openai==0.28.1",
        "pyyaml",
        "pylint==2.14.5",
        "types-PyYAML",
        "lisdf",
        "seaborn",
        "smepy@git+https://github.com/sebdumancic/structure_mapping.git",
        "pg3@git+https://github.com/tomsilver/pg3.git",
        "gym_sokoban@git+https://github.com/Learning-and-Intelligent-Systems/gym-sokoban.git"  # pylint: disable=line-too-long
    ],
    include_package_data=True,
    extras_require={
        "develop": [
            "pytest-cov==2.12.1",
            "pytest-pylint==0.18.0",
            "yapf==0.32.0",
            "docformatter==1.4",
            "isort==5.10.1",
        ]
    })
