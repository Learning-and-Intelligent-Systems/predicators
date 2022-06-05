import glob
import importlib
import logging
import os
import sys

from predicators.src import utils
from predicators.src.pybullet_helpers.ikfast import IKFastInfo


def _install_ikfast_module(ikfast_dir: str) -> None:
    """One-time install an IKFast module for a specific robot.

    Assumes there is a subdirectory in envs/assets/ikfast with a
    setup.py file for the robot. See the panda_arm subdirectory for an
    example.
    """
    # TODO: consider directly calling setup.py instead of using an OS command.
    cmds = [
        # Go to the subdirectory with the setup.py file.
        f"cd {ikfast_dir}",
        # Run the setup.py file.
        "python setup.py",
    ]
    # Execute the command.
    cmd = "; ".join(cmds)
    logging.debug(f"Executing command: {cmd}")
    os.system(cmd)


def _install_ikfast_if_required(ikfast_info: IKFastInfo) -> str:
    """
    If IKFast has been previously installed, there should be a file with
    extension .so, starting with name module_name, in the ikfast_dir.

    We check if this file exists, if not we install IKFast by compiling it.
    """
    ikfast_dir = os.path.join(
        utils.get_env_asset_path("ikfast"), ikfast_info.module_dir
    )
    glob_pattern = os.path.join(ikfast_dir, f"{ikfast_info.module_name}*.so")
    so_filepaths = glob.glob(glob_pattern)
    if len(so_filepaths) > 1:
        raise ValueError("More than one .so file found for IKFast.")

    # We need to install.
    if not so_filepaths:
        logging.warning(
            f"IKFast module {ikfast_info.module_name} not found; installing."
        )
        _install_ikfast_module(ikfast_dir)
        so_filepaths = glob.glob(glob_pattern)
        if len(so_filepaths) != 1:
            raise RuntimeError(
                f"Encountered {len(so_filepaths)} .so files after installing IKFast."
            )

    module_filepath = so_filepaths[0]
    return module_filepath


def import_ikfast(ikfast_info: IKFastInfo):
    """
    Imports the MoveIt IKFast solver for the given robot. If the solver is not
    already installed, it will be installed automatically when this function is
    called for the first time.
    """
    if ikfast_info.module_name in sys.modules:
        # IKFast already imported for robot so just return it.
        ikfast = sys.modules[ikfast_info.module_name]
        return ikfast

    # Otherwise, we load the IKFast module and build it if it isn't already built.
    module_filepath = _install_ikfast_if_required(ikfast_info)

    # Import the module.
    # See https://docs.python.org/3/library/importlib.html.
    spec = importlib.util.spec_from_file_location(
        ikfast_info.module_name, module_filepath
    )
    assert spec is not None, "IKFast module could not be found."
    ikfast = importlib.util.module_from_spec(spec)
    sys.modules[ikfast_info.module_name] = ikfast
    spec.loader.exec_module(ikfast)
    logging.debug(
        f"Loaded IKFast module {ikfast_info.module_name} from {module_filepath}"
    )

    return ikfast
