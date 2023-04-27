"""Setup script."""
from setuptools import find_packages, setup

path_to_myproject = "."

setup(
    name="predicators",
    version="0.1.0",
    packages=find_packages(include=["predicators", "predicators.*"]),
    install_requires=[
        "numpy>=1.22.3",
        "pytest",
        "gym==0.26.2",
        "matplotlib",
        "imageio",
        "imageio-ffmpeg",
        "pandas",
        "torch",
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
        "openai",
        "pyyaml",
        "pylint==2.14.5",
        "types-PyYAML",
        "lisdf",
        "seaborn",
        "smepy@git+https://github.com/sebdumancic/structure_mapping.git",
        "bosdyn-client >= 3.1",
        "opencv-python >= 3.4.2.17",
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
            "mypy@git+https://github.com/python/mypy.git@9bd651758e8ea2494" +
            "837814092af70f8d9e6f7a1",
        ]
    })
