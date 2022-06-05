import glob
import importlib.util
import logging
import os
import random
import sys
import time
from itertools import chain
from typing import Sequence, List, NamedTuple, TYPE_CHECKING, Dict

import numpy as np

from predicators.src import utils
from predicators.src.pybullet_helpers.utils import matrix_from_quat
from predicators.src.structs import JointsState, Pose3D
from pybullet_tools.ikfast.ikfast import get_ik_joints
from pybullet_tools.utils import (
    get_joint_positions,
    interval_generator,
    INF,
    joints_from_names,
    get_min_limits,
    get_max_limits, get_length, get_difference_fn, wait_for_user,
)

"""
This implementation is heavily based on the pybullet-planning repository
by Caelan Garrett (https://github.com/caelan/pybullet-planning/).
"""

if TYPE_CHECKING:
    from predicators.src.pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot


class IKFastInfo(NamedTuple):
    module_dir: str
    module_name: str
    base_link: str
    ee_link: str
    free_joints: List[str]


_ROBOT_TO_IKFAST_INFO: Dict[str, IKFastInfo] = {
    "panda": IKFastInfo(
        module_dir="panda_arm",
        module_name="ikfast_panda_arm",
        # TODO: use the below things
        base_link="panda_link0",
        ee_link="panda_link8",
        free_joints=["panda_joint7"],
    )
}


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

    We check if this file exists, if not we install IKFast to compile it.
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


def _import_ikfast(ikfast_info: IKFastInfo):
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


def ikfast_inverse_kinematics(
    robot: "SingleArmPyBulletRobot",
    target_position: Pose3D,
    target_orientation: Sequence[float],
    physics_client_id: int,
) -> JointsState:
    """Runs IK and returns a joints state.

    TODO: describe the assumptions about the target position and orientation
    in terms of what joints they're referring to.

    Note that this will automatically compile IKFast for the given robot if it
    hasn't been compiled already when this function is called for the first time.

    """
    robot_name = robot.get_name()
    if robot_name not in _ROBOT_TO_IKFAST_INFO:
        raise ValueError(f"Robot {robot_name} not supported by IKFast.")

    ikfast_info = _ROBOT_TO_IKFAST_INFO[robot_name]
    ikfast = _import_ikfast(ikfast_info)

    # IKFast expects matrix representation of orientation.
    matrix_target_orn = matrix_from_quat(target_orientation, physics_client_id).tolist()

    # TODO: understand third argument. Waiting for Caelan reply on this.
    # This is a temporary thing until I understand what the hell is happening.
    # ipdb> self._joint_lower_limits[6]
    # -2.8973
    # ipdb> self._joint_upper_limits[6]
    # 2.8973
    robot_obj = robot
    current_conf = np.array(robot_obj.get_joints())[:7]
    robot = robot.robot_id

    max_distance = INF
    free_joints = joints_from_names(robot, ikfast_info.free_joints)
    current_positions = get_joint_positions(robot, free_joints)
    lower_limits = get_min_limits(robot, free_joints)
    upper_limits = get_max_limits(robot, free_joints)

    generator = chain(
        [current_positions],  # TODO: sample from a truncated Gaussian nearby
        interval_generator(lower_limits, upper_limits),
    )

    max_time = 5
    start_time = time.time()

    def violates_limit(q):
        violates_lower_limit = [qs < limit for qs, limit in zip(q, robot_obj.joint_lower_limits)]
        violates_upper_limit = [qs > limit for qs, limit in zip(q, robot_obj.joint_upper_limits)]
        return any(violates_lower_limit) or any(violates_upper_limit)

    for free_positions in generator:
        free_positions = list(free_positions)

        if max_time < time.time() - start_time:
            break

        solutions = ikfast.get_ik(
            matrix_target_orn, list(target_position), free_positions
        )
        if solutions is None:
            continue

        # Return closest solution
        # TODO: circular checking instead of plain norm
        valid_confs = [conf for conf in solutions if not violates_limit(conf)]
        if valid_confs:
            dists = [np.linalg.norm(np.array(conf) - current_conf) for conf in valid_confs]
            min_idx = np.argmin(dists)
            return valid_confs[min_idx]

    raise RuntimeError("IK Failed!")
