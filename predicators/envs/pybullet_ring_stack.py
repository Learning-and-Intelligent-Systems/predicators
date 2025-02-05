"""A PyBullet version of Cover."""
import logging
import os
import subprocess
from typing import Any, ClassVar, Dict, List, Tuple
from pathlib import Path
import numpy as np
import pybullet as p
import shutil
import random

from predicators import utils
from predicators.envs.ring_stack import RingStackEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, Object, State


class PyBulletRingEnv(PyBulletEnv, RingStackEnv):
    """PyBullet Ring Stack domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._ring_id_to_ring: Dict[int, Object] = {}
        self._ring_id_to_target: Dict[int, Object] = {}

        # Create a copy of the pybullet robot for checking forward kinematics
        # in step() without changing the "real" robot state.
        fk_physics_id = p.connect(p.DIRECT)
        self._pybullet_robot_fk = self._create_pybullet_robot(fk_physics_id)

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool,
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle blocks-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)
        logging.info("Initialized pybullet")

        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)
        bodies["table_id"] = table_id

        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            # assert using_gui, \
            #     "using_gui must be True to use pybullet_draw_debug."
            # Draw the workspace on the table for clarity.
            p.addUserDebugLine([cls.x_lb, cls.y_lb, cls.table_height],
                               [cls.x_ub, cls.y_lb, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_ub, cls.table_height],
                               [cls.x_ub, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_lb, cls.table_height],
                               [cls.x_lb, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_ub, cls.y_lb, cls.table_height],
                               [cls.x_ub, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            # Draw coordinate frame labels for reference.
            p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            p.addUserDebugText("z", [0, 0, 0.25], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            # Draw the pick z location at the x/y midpoint.
            mid_x = (cls.x_ub + cls.x_lb) / 2
            mid_y = (cls.y_ub + cls.y_lb) / 2
            p.addUserDebugText("*", [mid_x, mid_y, cls.pick_z],
                               [1.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)

        # # Create ring & pole

        bodies["ring_ids"] = []
        for _ in range(CFG.ring_stack_max_num_rings):
            task_ring_idx = 2  # Initial rings do not matter.

            # Update the URDF file to point to the new .obj file
            urdf_file = utils.get_env_asset_path("urdf/ring.urdf")
            with open(urdf_file, 'r') as file:
                urdf_data = file.read()

            # Replace the old .obj file reference with the new one
            updated_urdf_data = urdf_data.replace('ring.obj', f'../rings/ring_{task_ring_idx}.obj')

            # Write the updated URDF data to a temporary file
            temp_urdf_file = f"predicators/envs/assets/urdf/temp/ring_temp_{task_ring_idx}.urdf"
            with open(temp_urdf_file, 'w') as file:
                file.write(updated_urdf_data)

            # Load the updated URDF
            ring_id = p.loadURDF(temp_urdf_file,
                                 useFixedBase=False,
                                 physicsClientId=physics_client_id)
            os.remove(temp_urdf_file)

            bodies["ring_ids"].append(ring_id)

        pole_id = p.loadURDF(utils.get_env_asset_path("urdf/pole.urdf"),
                             useFixedBase=True,
                             physicsClientId=physics_client_id)
        bodies["pole_id"] = pole_id

        return physics_client_id, pybullet_robot, bodies

    def generate_new_ring_models(self, state):
        logging.info(f"new rings for p_client_id {self._physics_client_id}")
        success = False

        while not success:
            try:
                new_ring_ids = []
                logging.info(f'pb bodies: {p.getNumBodies()}')
                if self._ring_ids is None:
                    self._ring_ids = []
                logging.info(f"ring_ids: {self._ring_ids}")
                for body_id in self._ring_ids:
                    p.removeBody(body_id)

                rings = state.get_objects(self._ring_type)
                logging.info(f"task init state: {state}")


                for ring in rings:
                    task_ring_idx = int(state.get(ring, "id"))

                    # Update the URDF file to point to the new .obj file
                    urdf_file = utils.get_env_asset_path("urdf/ring.urdf")
                    with open(urdf_file, 'r') as file:
                        urdf_data = file.read()

                    # Replace the old .obj file reference with the new one
                    updated_urdf_data = urdf_data.replace('ring.obj', f'../rings/ring_{task_ring_idx}.obj')

                    # Write the updated URDF data to a temporary file
                    temp_urdf_file = f"predicators/envs/assets/urdf/temp/ring_temp_{task_ring_idx}.urdf"
                    with open(temp_urdf_file, 'w') as file:
                        file.write(updated_urdf_data)

                    logging.info(f"Using ring: {task_ring_idx} for test task")

                    # Load the updated URDF
                    ring_id = p.loadURDF(temp_urdf_file,
                                         useFixedBase=False,
                                         physicsClientId=self._physics_client_id)

                    os.remove(temp_urdf_file)

                    new_ring_ids.append(ring_id)
                success = True
            except Exception as e:
                logging.info("Race condition error in generating rings")
                success = False
        logging.info(f'pb bodies: {p.getNumBodies()}')

        self._ring_ids = new_ring_ids

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._ring_ids = pybullet_bodies["ring_ids"]
        self._pole_ids = pybullet_bodies["pole_id"]

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((cls.robot_init_x, cls.robot_init_y, cls.robot_init_z),
                       robot_ee_orn)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home)

    def _extract_robot_state(self, state: State) -> Array:
        fingers = self.fingers_state_to_joint(self._pybullet_robot,
                                              state.get(self._robot, "fingers"))

        ry = state.get(self._robot, "pose_y")
        rx = state.get(self._robot, "pose_x")
        rz = state.get(self._robot, "pose_z")
        # The orientation is fixed in this environment.
        qx = state.get(self._robot, "orn_x")
        qy = state.get(self._robot, "orn_y")
        qz = state.get(self._robot, "orn_z")
        qw = state.get(self._robot, "orn_w")

        return np.array([rx, ry, rz, qx, qy, qz, qw, fingers],
                        dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_ring_stack"

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle blocks-specific resetting."""
        super()._reset_state(state)
        logging.info("resetting pybullet state")
        # Reset rings based on the state.

        ring_objs = state.get_objects(self._ring_type)
        self._ring_id_to_ring = {}
        for i, ring_obj in enumerate(ring_objs):
            ring_id = self._ring_ids[i]

            self._ring_id_to_ring[ring_id] = ring_obj
            bx = state.get(ring_obj, "pose_x")
            by = state.get(ring_obj, "pose_y")
            bz = state.get(ring_obj, "pose_z")
            # logging.info(f'ring pos from reset: {bx}, {by}, {bz}')

            p.resetBasePositionAndOrientation(
                ring_id, [bx, by, bz],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # Check if we're holding some ring.
        held_ring = self._get_held_ring(state)
        if held_ring is not None:
            self._force_grasp_object(held_ring)


        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(ring_objs), len(self._ring_ids)):
            ring_id = self._ring_ids[i]
            h = state.get(self._ring_id_to_ring[ring_id], "minor_radius")
            assert ring_id not in self._ring_id_to_ring
            p.resetBasePositionAndOrientation(
                ring_id, [i, 0, h],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # reset pole based on state
        pole_obj = state.get_objects(self._pole_type)[0]
        self._pole_id_to_pole = {}
        pole_id = self._pole_ids
        self._pole_id_to_pole[pole_id] = pole_obj
        bx = state.get(pole_obj, "pose_x")
        by = state.get(pole_obj, "pose_y")
        bz = state.get(pole_obj, "pose_z")
        # logging.info(f'pole pos from reset: {bx}, {by}, {bz}')
        p.resetBasePositionAndOrientation(
            pole_id, [bx, by, bz],
            self._default_orn,
            physicsClientId=self._physics_client_id)

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.info("Desired state:")
            logging.info(state.pretty_str())
            logging.info("Reconstructed state:")
            logging.info(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state.

        Note that in addition to the state inside PyBullet itself, this
        uses self._ring_id_to_ring and self._held_obj_id. As long as
        the PyBullet internal state is only modified through reset() and
        step(), these all should remain in sync.
        """
        state_dict = {}

        # Get robot state.
        rx, ry, rz, qx, qy, qz, qw, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(rf)
        state_dict[self._robot] = np.array([rx, ry, rz, qx, qy, qz, qw, fingers],
                                           dtype=np.float32)
        joint_positions = self._pybullet_robot.get_joints()

        # Get ring states.
        for ring_id, ring in self._ring_id_to_ring.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                ring_id, physicsClientId=self._physics_client_id)
            held = (ring_id == self._held_obj_id)
            # pose_x, pose_y, pose_z, held

            mesh = str(p.getVisualShapeData(ring_id, physicsClientId=self._physics_client_id)[0][4])
            mesh_data = mesh.split("'")[1]
            parts = mesh_data.split('/')
            last_part = parts[-1]
            # Split the last part by '_' and get the part containing the number
            number_part = last_part.split('_')[1]
            # Split the number part by '.' to remove the file extension
            mesh_id = number_part.split('.')[0]
            major_radius, minor_radius = self.retrieve_geometry_data_from_obj(mesh_data)

            state_dict[ring] = np.array([bx, by, bz, mesh_id, major_radius, minor_radius, held],
                                        dtype=np.float32)

        # Get pole states.
        for pole_id, pole in self._pole_id_to_pole.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                pole_id, physicsClientId=self._physics_client_id)
            # pose_x, pose_y, pose_z,
            state_dict[pole] = np.array([bx, by, bz],
                                        dtype=np.float32)

        state = utils.PyBulletState(state_dict,
                                    simulator_state=joint_positions)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def _get_tasks(self, num_tasks: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num_tasks, rng)
        return self._add_pybullet_state_to_tasks(tasks)

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._ring_id_to_ring)

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

    def _force_grasp_object(self, ring: Object) -> None:
        ring_to_ring_id = {r: i for i, r in self._ring_id_to_ring.items()}
        ring_id = ring_to_ring_id[ring]
        # The block should already be held. Otherwise, the position of the
        # block was wrong in the state.
        held_obj_id = self._detect_held_object()
        assert ring_id == held_obj_id
        # Create the grasp constraint.
        self._held_obj_id = ring_id
        self._create_grasp_constraint()

    @classmethod
    def fingers_state_to_joint(cls, pybullet_robot: SingleArmPyBulletRobot,
                               fingers_state: float) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either 0 or 1. Transform them to be
        either pybullet_robot.closed_fingers or
        pybullet_robot.open_fingers.
        """
        assert fingers_state in (0.0, 1.0)
        open_f = pybullet_robot.open_fingers
        closed_f = pybullet_robot.closed_fingers
        return closed_f if fingers_state == 0.0 else open_f

    def _fingers_joint_to_state(self, fingers_joint: float) -> float:
        """Convert the finger joint values in PyBullet to values for the State.

        The joint values given as input are the ones coming out of
        self._pybullet_robot.get_state().
        """
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
        # Fingers in the State should be either 0 or 1.
        return int(fingers_joint > (open_f + closed_f) / 2)

    @classmethod
    def retrieve_geometry_data_from_obj(cls, file_path):
        # Open the file and read the first line
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()

        # Check if the first line is a comment
        if first_line.startswith('#'):
            # Remove the '#' and split the comment by comma
            comment = first_line[1:].strip()
            values = comment.split(',')

            if len(values) == 2:
                major_radius = values[0].strip()
                minor_radius = values[1].strip()
                return float(major_radius), float(minor_radius)
            else:
                raise ValueError("Comment does not contain exactly two comma-separated values")
        else:
            raise ValueError("The first line is not a comment")
