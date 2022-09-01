"""This module installs and loads the IKFast module for a given robot specified
by its IKFastInfo."""

import glob
import importlib
import logging
import os
import sys
from importlib.abc import Loader
from types import ModuleType

from predicators.pybullet_helpers.ikfast import IKFastInfo
from predicators.utils import get_third_party_path


def install_ikfast_module(ikfast_dir: str) -> None:
    """One-time install an IKFast module for a specific robot.

    Assumes there is a subdirectory in envs/assets/ikfast with a
    setup.py file for the robot. See the panda_arm subdirectory for an
    example.
    """
    cmds = [
        # Go to the subdirectory with the setup.py file.
        f"cd {ikfast_dir}",
        # Run the setup.py file.
        "python setup.py",
    ]
    # Execute the command.
    cmd = "; ".join(cmds)
    logging.debug(f"Executing command: {cmd}")
    exit_value = os.system(cmd)
    if exit_value != 0:
        raise RuntimeError(
            f"IKFast install failed with exit code {exit_value}. "
            "Check messages above.")


def install_ikfast_if_required(ikfast_info: IKFastInfo) -> str:
    """If IKFast has been previously installed, there should be a file with
    extension .so, starting with name module_name, in the ikfast_dir.

    We check if this file exists, if not we install IKFast by compiling
    it.
    """
    ikfast_dir = os.path.join(get_third_party_path(), "ikfast",
                              ikfast_info.module_dir)
    glob_pattern = os.path.join(ikfast_dir, f"{ikfast_info.module_name}*.so")
    so_filepaths = glob.glob(glob_pattern)

    # We need to install.
    if not so_filepaths:
        logging.warning(
            f"IKFast module {ikfast_info.module_name} not found; installing.")
        install_ikfast_module(ikfast_dir)
        so_filepaths = glob.glob(glob_pattern)

    if len(so_filepaths) != 1:  # pragma: no cover
        # Shouldn't ever happen.
        raise ValueError(
            f"Found {len(so_filepaths)} .so files in {ikfast_dir}.")

    module_filepath = so_filepaths[0]
    return module_filepath


def import_ikfast(ikfast_info: IKFastInfo) -> ModuleType:
    """Imports the MoveIt IKFast solver for the given robot.

    If the solver is not already installed, it will be installed
    automatically when this function is called for the first time.
    """
    if ikfast_info.module_name in sys.modules:
        # IKFast already imported for robot so just return it.
        ikfast = sys.modules[ikfast_info.module_name]
        return ikfast

    # Otherwise, load IKFast module and build if it isn't already built.
    module_filepath = install_ikfast_if_required(ikfast_info)

    # Import the module.
    # See https://docs.python.org/3/library/importlib.html.
    spec = importlib.util.spec_from_file_location(ikfast_info.module_name,
                                                  module_filepath)
    if spec is None or spec.loader is None:  # pragma: no cover
        # Shouldn't ever happen unless there's corruption or tampering.
        raise ImportError("IKFast module could not be found.")

    ikfast = importlib.util.module_from_spec(spec)
    sys.modules[ikfast_info.module_name] = ikfast

    # Keeps mypy happy
    assert isinstance(spec.loader, Loader)
    spec.loader.exec_module(ikfast)
    logging.debug(
        f"Loaded IKFast module {ikfast_info.module_name} from {module_filepath}"
    )

    return ikfast
