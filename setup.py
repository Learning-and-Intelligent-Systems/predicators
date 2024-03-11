"""Setup script."""
from setuptools import find_packages, setup

path_to_myproject = "."

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
        "requests",
        "slack_bolt",
        "pybullet>=3.2.0",
        "scikit-learn==1.1.2",
        "graphlib-backport",
        "openai==0.28.1",
        "pyyaml==6.0",
        "pylint==2.14.5",
        "types-PyYAML",
        "lisdf",
        "seaborn==0.12.1",
        "moviepy==1.0.3",
        "apriltag==0.0.16",
        "scikit-image==0.19.3",
        "protobuf==4.22.0",
        "smepy@git+https://github.com/sebdumancic/structure_mapping.git",
        "bosdyn-client >= 3.1",
        "opencv-python == 4.7.0.72",
        "pg3@git+https://github.com/tomsilver/pg3.git",
        "gym_sokoban@git+https://github.com/Learning-and-Intelligent-Systems/gym-sokoban.git",  # pylint: disable=line-too-long
        "pbrspot@git+https://github.com/NishanthJKumar/pbrspot.git"
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
