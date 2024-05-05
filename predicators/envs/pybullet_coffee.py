"""A PyBullet version of CoffeeEnv."""

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.coffee import CoffeeEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Array, EnvironmentTask, Object, State


class PyBulletCoffeeEnv(PyBulletEnv, CoffeeEnv):
    """PyBullet Coffee domain."""

    # Need to override a number of settings to conform to the actual dimensions
    # of the robots, table, etc.
    init_padding: ClassVar[float] = 0.05
    x_lb: ClassVar[float] = 1.1
    x_ub: ClassVar[float] = 1.6
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    z_lb: ClassVar[float] = 0.2
    z_ub: ClassVar[float] = 0.75
    robot_init_x: ClassVar[float] = (x_ub + x_lb) / 2.0
    robot_init_y: ClassVar[float] = (y_ub + y_lb) / 2.0
    robot_init_z: ClassVar[float] = z_ub - 0.1
    # Machine settings.
    machine_x_len: ClassVar[float] = 0.2 * (x_ub - x_lb)
    machine_y_len: ClassVar[float] = 0.1 * (y_ub - y_lb)
    machine_z_len: ClassVar[float] = 0.5 * (z_ub - z_lb)
    machine_x: ClassVar[float] = x_ub - machine_x_len - 0.01
    machine_y: ClassVar[float] = y_lb + machine_y_len + init_padding
    button_x: ClassVar[float] = machine_x
    button_y: ClassVar[float] = machine_y + machine_y_len / 2
    button_z: ClassVar[float] = z_lb + 3 * machine_z_len / 4
    button_radius: ClassVar[float] = 0.2 * machine_y_len
    # Jug settings.
    pick_jug_x_padding: ClassVar[float] = 0.05
    jug_radius: ClassVar[float] = (0.8 * machine_y_len) / 2.0
    jug_height: ClassVar[float] = 0.15 * (z_ub - z_lb)
    jug_init_y_lb: ClassVar[float] = machine_y - machine_y_len + init_padding
    jug_init_y_ub: ClassVar[float] = machine_y + machine_y_len - init_padding
    jug_init_x_lb: ClassVar[float] = x_lb + jug_radius + pick_jug_x_padding + \
                                     init_padding
    jug_init_x_ub: ClassVar[
        float] = machine_x - machine_x_len - jug_radius - init_padding
    jug_handle_offset: ClassVar[float] = 1.75 * jug_radius
    jug_handle_height: ClassVar[float] = jug_height / 2
    jug_handle_radius: ClassVar[float] = jug_handle_height / 3  # for rendering
    # Dispense area settings.
    dispense_area_x: ClassVar[float] = machine_x - 2.4 * jug_radius
    dispense_area_y: ClassVar[float] = machine_y + machine_y_len / 2
    # Cup settings.
    cup_radius: ClassVar[float] = 0.6 * jug_radius
    cup_init_x_lb: ClassVar[float] = jug_init_x_lb
    cup_init_x_ub: ClassVar[float] = jug_init_x_ub
    cup_init_y_lb: ClassVar[float] = machine_y + cup_radius + init_padding + jug_radius
    cup_init_y_ub: ClassVar[float] = y_ub - cup_radius - init_padding
    cup_capacity_lb: ClassVar[float] = 0.075 * (z_ub - z_lb)
    cup_capacity_ub: ClassVar[float] = 0.15 * (z_ub - z_lb)
    cup_target_frac: ClassVar[float] = 0.75  # fraction of the capacity
    # Table settings.
    table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)


    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Settings from CFG.
        self.jug_init_rot_lb = -CFG.coffee_jug_init_rot_amt
        self.jug_init_rot_ub = CFG.coffee_jug_init_rot_amt

    def initialize_pybullet(
            self, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle coffee-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Load table.
        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          self.table_pose,
                                          self.table_orientation,
                                          physicsClientId=physics_client_id)
        bodies["table_id"] = table_id

        # Load coffee jug.

        # Create the body.
        # This pose doesn't matter because it gets overwritten in reset.
        jug_pose = ((self.jug_init_x_lb + self.jug_init_x_ub) / 2,
                    (self.jug_init_y_lb + self.jug_init_y_ub) / 2,
                    self.z_lb + self.jug_height / 2)
        # The jug orientation updates based on the rotation of the state.
        rot = (self.jug_init_rot_lb + self.jug_init_rot_ub) / 2
        jug_orientation = p.getQuaternionFromEuler([0.0, 0.0, rot + np.pi])

        jug_id = p.loadURDF(
            utils.get_env_asset_path("urdf/kettle.urdf"),
            useFixedBase=True,
            globalScaling=0.075,
            physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(jug_id,
            jug_pose,
            jug_orientation,
            physicsClientId=physics_client_id)
        bodies["jug_id"] = jug_id

        # TODO remove
        for _ in range(10000):
            p.stepSimulation(physicsClientId=physics_client_id)
            import time
            time.sleep(0.01)

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._jug_id = pybullet_bodies["jug_id"]

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((cls.robot_init_x, cls.robot_init_y, cls.robot_init_z),
                       robot_ee_orn)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home)

    def _extract_robot_state(self, state: State) -> Array:
        import ipdb; ipdb.set_trace()
        # # The orientation is fixed in this environment.
        # qx, qy, qz, qw = self.get_robot_ee_home_orn()
        # f = self.fingers_state_to_joint(self._pybullet_robot,
        #                                 state.get(self._robot, "fingers"))
        # return np.array([
        #     state.get(self._robot, "pose_x"),
        #     state.get(self._robot, "pose_y"),
        #     state.get(self._robot, "pose_z"), qx, qy, qz, qw, f
        # ],
        #                 dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_coffee"

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle coffee-specific resetting."""
        super()._reset_state(state)

        import ipdb; ipdb.set_trace()

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.debug("Desired state:")
            logging.debug(state.pretty_str())
            logging.debug("Reconstructed state:")
            logging.debug(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state.

        Note that in addition to the state inside PyBullet itself, this
        uses self._block_id_to_block and self._held_obj_id. As long as
        the PyBullet internal state is only modified through reset() and
        step(), these all should remain in sync.
        """
        state_dict = {}

        # Get robot state.
        rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(rf)
        state_dict[self._robot] = np.array([rx, ry, rz, fingers],
                                           dtype=np.float32)
        joint_positions = self._pybullet_robot.get_joints()

        import ipdb; ipdb.set_trace()

        state = utils.PyBulletState(state_dict,
                                    simulator_state=joint_positions)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def _get_tasks(self, num: int, num_cups_lst: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num, num_cups_lst, rng)
        return self._add_pybullet_state_to_tasks(tasks)

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        import ipdb; ipdb.set_trace()

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        if CFG.pybullet_robot == "panda":
            # gripper rotated 90deg so parallel to x-axis
            normal = np.array([1., 0., 0.], dtype=np.float32)
        elif CFG.pybullet_robot == "fetch":
            # gripper parallel to y-axis
            normal = np.array([0., 1., 0.], dtype=np.float32)
        else:  # pragma: no cover
            # Shouldn't happen unless we introduce a new robot.
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        return {
            self._pybullet_robot.left_finger_id: normal,
            self._pybullet_robot.right_finger_id: -1 * normal,
        }
