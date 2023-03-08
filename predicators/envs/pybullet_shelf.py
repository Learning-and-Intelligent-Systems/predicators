"""A simple environment where a robot must pick a block from the table with a
top grasp and put it into a high-up shelf with a side grasp.

The main purpose of this environment is to develop PyBullet options that
involve changing the end-effector orientation.
"""

import logging
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import RGBA, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class PyBulletShelfEnv(PyBulletEnv):
    """PyBullet shelf domain."""

    # Parameters that aren't important enough to need to clog up settings.py
    # The table x bounds are (1.1, 1.6), but the workspace is smaller.
    _x_lb: ClassVar[float] = 1.2
    _x_ub: ClassVar[float] = 1.5
    # The table y bounds are (0.3, 1.2), but the workspace is smaller.
    _y_lb: ClassVar[float] = 0.4
    _y_ub: ClassVar[float] = 1.1

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)
    table_height: ClassVar[float] = 0.2

    # Wall parameters.
    _wall_thickness: ClassVar[float] = 0.01
    _wall_height: ClassVar[float] = 0.1
    _wall_x: ClassVar[float] = 1.12

    # Block parameters.
    _block_color: ClassVar[RGBA] = (1.0, 0.0, 0.0, 1.0)
    block_size: ClassVar[float] = 0.05
    _block_x_lb: ClassVar[float] = _x_lb + block_size
    _block_x_ub: ClassVar[float] = _x_ub - block_size
    _block_y_lb: ClassVar[float] = _y_ub - block_size
    _block_y_ub: ClassVar[float] = _y_ub - block_size / 2

    # Robot parameters.
    _robot_init_x: ClassVar[float] = (_block_x_lb + _block_x_ub) / 2
    _robot_init_y: ClassVar[float] = (_block_y_lb + _block_y_ub) / 2
    _robot_init_z: ClassVar[float] = 0.75

    # Shelf parameters.
    _shelf_width: ClassVar[float] = (_x_ub - _x_lb) * 0.4
    _shelf_length: ClassVar[float] = (_y_ub - _y_lb) * 0.6
    shelf_base_height: ClassVar[float] = _robot_init_z * 0.8
    _shelf_ceiling_height: ClassVar[float] = _robot_init_z * 0.2
    _shelf_ceiling_thickness: ClassVar[float] = 0.01
    _shelf_pole_diam: ClassVar[float] = 0.01
    _shelf_color: ClassVar[RGBA] = (0.5, 0.3, 0.05, 1.0)
    shelf_x: ClassVar[float] = _x_ub - _shelf_width / 2
    shelf_y: ClassVar[float] = _y_lb + _shelf_length / 2

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        self._robot_type = Type("robot", [
            "pose_x", "pose_y", "pose_z", "pose_q0", "pose_q1", "pose_q2",
            "pose_q3", "fingers"
        ])
        self._shelf_type = Type("shelf", ["pose_x", "pose_y"])
        self._block_type = Type("block",
                                ["pose_x", "pose_y", "pose_z", "held"])

        self._InShelf = Predicate("InShelf",
                                  [self._block_type, self._shelf_type],
                                  self._InShelf_holds)
        self._OnTable = Predicate("OnTable", [self._block_type],
                                  self._OnTable_holds)

        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._shelf = Object("shelfy", self._shelf_type)
        self._block = Object("blocky", self._block_type)

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InShelf, self._OnTable}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InShelf}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._block_type, self._shelf_type}

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle cover-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        bodies["table_id"] = table_id
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)

        # Create shelf.
        color = cls._shelf_color
        orientation = cls._default_orn
        base_pose = (cls.shelf_x, cls.shelf_y,
                     cls.table_height + cls.shelf_base_height / 2)
        # Shelf base.
        # Create the collision shape.
        base_half_extents = [
            cls._shelf_width / 2, cls._shelf_length / 2,
            cls.shelf_base_height / 2
        ]
        base_collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=base_half_extents,
            physicsClientId=physics_client_id)
        # Create the visual shape.
        base_visual_id = p.createVisualShape(p.GEOM_BOX,
                                             halfExtents=base_half_extents,
                                             rgbaColor=color,
                                             physicsClientId=physics_client_id)
        # Create the ceiling.
        link_positions = []
        link_collision_shape_indices = []
        link_visual_shape_indices = []
        pose = (
            0, 0,
            cls.shelf_base_height / 2 + cls._shelf_ceiling_height - \
                cls._shelf_ceiling_thickness / 2
        )
        link_positions.append(pose)
        half_extents = [
            cls._shelf_width / 2, cls._shelf_length / 2,
            cls._shelf_ceiling_thickness / 2
        ]
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=physics_client_id)
        link_collision_shape_indices.append(collision_id)
        visual_id = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=half_extents,
                                        rgbaColor=color,
                                        physicsClientId=physics_client_id)
        link_visual_shape_indices.append(visual_id)
        # Create poles connecting the base to the ceiling.
        for x_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                pose = (x_sign * (cls._shelf_width - cls._shelf_pole_diam) / 2,
                        y_sign * (cls._shelf_length - cls._shelf_pole_diam) /
                        2, cls.shelf_base_height / 2 +
                        cls._shelf_ceiling_height / 2)
                link_positions.append(pose)
                half_extents = [
                    cls._shelf_pole_diam / 2, cls._shelf_pole_diam / 2,
                    cls._shelf_ceiling_height / 2
                ]
                collision_id = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    physicsClientId=physics_client_id)
                link_collision_shape_indices.append(collision_id)
                visual_id = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=color,
                    physicsClientId=physics_client_id)
                link_visual_shape_indices.append(visual_id)

        # Create the whole body.
        num_links = len(link_positions)
        assert len(link_collision_shape_indices) == num_links
        assert len(link_visual_shape_indices) == num_links
        link_masses = [0.1 for _ in range(num_links)]
        link_orientations = [orientation for _ in range(num_links)]
        link_intertial_frame_positions = [[0, 0, 0] for _ in range(num_links)]
        link_intertial_frame_orns = [[0, 0, 0, 1] for _ in range(num_links)]
        link_parent_indices = [0 for _ in range(num_links)]
        link_joint_types = [p.JOINT_FIXED for _ in range(num_links)]
        link_joint_axis = [[0, 0, 0] for _ in range(num_links)]
        shelf_id = p.createMultiBody(
            baseCollisionShapeIndex=base_collision_id,
            baseVisualShapeIndex=base_visual_id,
            basePosition=base_pose,
            baseOrientation=orientation,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shape_indices,
            linkVisualShapeIndices=link_visual_shape_indices,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_intertial_frame_positions,
            linkInertialFrameOrientations=link_intertial_frame_orns,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axis,
            physicsClientId=physics_client_id)
        bodies["shelf_id"] = shelf_id

        # Create a wall in front of the block to force a top grasp.
        color = cls._shelf_color
        orientation = cls._default_orn
        pose = (cls._wall_x, (cls._y_lb + cls._y_ub) / 2,
                cls.table_height + cls._wall_height / 2)
        # Create the collision shape.
        half_extents = [
            cls._wall_thickness / 2, (cls._y_ub - cls._y_lb) / 2,
            cls._wall_height / 2
        ]
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=physics_client_id)
        # Create the visual shape.
        visual_id = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=half_extents,
                                        rgbaColor=color,
                                        physicsClientId=physics_client_id)
        wall_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id,
                                    basePosition=pose,
                                    baseOrientation=orientation,
                                    physicsClientId=physics_client_id)
        bodies["wall_id"] = wall_id

        # Create block.
        color = cls._block_color
        half_extents = (cls.block_size / 2.0, cls.block_size / 2.0,
                        cls.block_size / 2.0)
        block_id = create_pybullet_block(color, half_extents, cls._obj_mass,
                                         cls._obj_friction, cls._default_orn,
                                         physics_client_id)
        bodies["block_id"] = block_id

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._shelf_id = pybullet_bodies["shelf_id"]
        self._block_id = pybullet_bodies["block_id"]

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        ee_home = Pose(
            (cls._robot_init_x, cls._robot_init_y, cls._robot_init_z),
            cls.get_robot_ee_home_orn()
            # FETCH
            # p.getQuaternionFromEuler([0.0, np.pi, -np.pi])
            # PANDA
            # p.getQuaternionFromEuler([3.0, -1.0, 0.0])
        )
        robot = create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                 physics_client_id, ee_home)
        # import time
        # while True:
        #     p.stepSimulation(physics_client_id)
        #     rx, ry, rz, q0, q1, q2, q3, rf = robot.get_state()
        #     print(p.getEulerFromQuaternion([q0, q1, q2, q3]))
        #     time.sleep(0.001)

        return robot

    def _extract_robot_state(self, state: State) -> Array:
        fingers = self.fingers_state_to_joint(
            self._pybullet_robot, state.get(self._robot, "fingers"))
        return np.array([
            state.get(self._robot, "pose_x"),
            state.get(self._robot, "pose_y"),
            state.get(self._robot, "pose_z"),
            state.get(self._robot, "pose_q0"),
            state.get(self._robot, "pose_q1"),
            state.get(self._robot, "pose_q2"),
            state.get(self._robot, "pose_q3"),
            fingers,
        ],
                        dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_shelf"

    def _reset_state(self, state: State) -> None:
        super()._reset_state(state)

        # Reset the block based on the state.
        x = state.get(self._block, "pose_x")
        y = state.get(self._block, "pose_y")
        z = state.get(self._block, "pose_z")
        p.resetBasePositionAndOrientation(
            self._block_id, [x, y, z],
            self._default_orn,
            physicsClientId=self._physics_client_id)

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()

        if not reconstructed_state.allclose(state):
            logging.debug("Desired state:")
            logging.debug(state.pretty_str())
            logging.debug("Reconstructed state:")
            logging.debug(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state."""
        state_dict = {}

        # Get robot state.
        rx, ry, rz, q0, q1, q2, q3, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(rf)
        state_dict[self._robot] = {
            "pose_x": rx,
            "pose_y": ry,
            "pose_z": rz,
            "pose_q0": q0,
            "pose_q1": q1,
            "pose_q2": q2,
            "pose_q3": q3,
            "fingers": fingers,
        }
        joint_positions = self._pybullet_robot.get_joints()

        # Get the shelf state.
        state_dict[self._shelf] = {
            "pose_x": self.shelf_x,
            "pose_y": self.shelf_y,
        }

        # Get block state.
        (bx, by, bz), _ = p.getBasePositionAndOrientation(
            self._block_id, physicsClientId=self._physics_client_id)
        held = (self._block_id == self._held_obj_id)
        state_dict[self._block] = {
            "pose_x": bx,
            "pose_y": by,
            "pose_z": bz,
            "held": held,
        }

        state_without_sim = utils.create_state_from_dict(state_dict)
        state = utils.PyBulletState(state_without_sim.data,
                                    simulator_state=joint_positions)

        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        # import time
        # while True:
        #     p.stepSimulation(self._physics_client_id)
        #     time.sleep(0.001)

        return state

    def _get_tasks(self, num_tasks: int,
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        for _ in range(num_tasks):
            state_dict = {}
            # The only variation is in the position of the block.
            x = rng.uniform(self._block_x_lb, self._block_x_ub)
            y = rng.uniform(self._block_y_lb, self._block_y_ub)
            z = self.table_height + self.block_size / 2
            held = 0.0
            state_dict[self._block] = {
                "pose_x": x,
                "pose_y": y,
                "pose_z": z,
                "held": held,
            }
            state_dict[self._shelf] = {
                "pose_x": self.shelf_x,
                "pose_y": self.shelf_y
            }
            home_orn = self.get_robot_ee_home_orn()
            state_dict[self._robot] = {
                "pose_x": self._robot_init_x,
                "pose_y": self._robot_init_y,
                "pose_z": self._robot_init_z,
                "pose_q0": home_orn[0],
                "pose_q1": home_orn[1],
                "pose_q2": home_orn[2],
                "pose_q3": home_orn[3],
                "fingers": 1.0  # fingers start out open
            }
            state = utils.create_state_from_dict(state_dict)
            goal = {GroundAtom(self._InShelf, [self._block, self._shelf])}
            task = Task(state, goal)
            tasks.append(task)

        return self._add_pybullet_state_to_tasks(tasks)

    def _get_object_ids_for_held_check(self) -> List[int]:
        return {self._block_id}

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        if CFG.pybullet_robot == "panda":
            # gripper rotated 90deg so parallel to x-axis
            normal = np.array([1., 0., 0.], dtype=np.float32)
        elif CFG.pybullet_robot == "fetch":
            # TODO
            import ipdb
            ipdb.set_trace()
        else:  # pragma: no cover
            # Shouldn't happen unless we introduce a new robot.
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        return {
            self._pybullet_robot.left_finger_id: normal,
            self._pybullet_robot.right_finger_id: -1 * normal,
        }

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

    def _InShelf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, shelf = objects
        ds = ["x", "y"]
        sizes = [self._shelf_width, self._shelf_length]
        # TODO factor out
        return self._object_contained_in_object(block, shelf, state, ds, sizes)

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        x = state.get(block, "pose_x")
        y = state.get(block, "pose_y")
        return self._block_x_lb <= x <= self._block_x_ub and \
               self._block_y_lb <= y <= self._block_y_ub

    def _object_contained_in_object(self, obj: Object, container: Object,
                                    state: State, dims: List[str],
                                    sizes: List[float]) -> bool:
        assert len(dims) == len(sizes)
        for dim, size in zip(dims, sizes):
            obj_pose = state.get(obj, f"pose_{dim}")
            container_pose = state.get(container, f"pose_{dim}")
            container_lb = container_pose - size / 2.
            container_ub = container_pose + size / 2.
            if not container_lb - 1e-5 <= obj_pose <= container_ub + 1e-5:
                return False
        return True
