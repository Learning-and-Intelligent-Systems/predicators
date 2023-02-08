"""A PyBullet version of the sandwich environment."""

import logging
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Sequence, Tuple

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block, \
    create_pybullet_cylinder
from predicators.envs.sandwich import SandwichEnv
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, State, Task


class PyBulletSandwichEnv(PyBulletEnv, SandwichEnv):
    """PyBullet Sandwich domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Option parameters.

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    # Robot parameters.
    _move_to_pose_tol: ClassVar[float] = 1e-4

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Override options, keeping the types and parameter spaces the same.
        open_fingers_func = lambda s, _1, _2: (self._fingers_state_to_joint(
            s.get(self._robot, "fingers")), self._pybullet_robot.open_fingers)
        close_fingers_func = lambda s, _1, _2: (self._fingers_state_to_joint(
            s.get(self._robot, "fingers")), self._pybullet_robot.closed_fingers
                                                )

        ## Pick option
        types = self._Pick.types
        params_space = self._Pick.params_space
        self._Pick: ParameterizedOption = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Close fingers.
                create_change_fingers_option(
                    self._pybullet_robot_sim, "CloseFingers", types,
                    params_space, close_fingers_func, self._max_vel_norm,
                    self._grasp_tol),
            ])

        ## Stack option
        types = self._Stack.types
        params_space = self._Stack.params_space
        self._Stack: ParameterizedOption = \
            utils.LinearChainParameterizedOption("Stack",
            [
                # Open fingers.
                create_change_fingers_option(self._pybullet_robot_sim,
                    "OpenFingers", types, params_space, open_fingers_func,
                    self._max_vel_norm, self._grasp_tol),
            ])

        ## PutOnBoard option
        types = self._PutOnBoard.types
        params_space = self._PutOnBoard.params_space
        self._PutOnBoard: ParameterizedOption = \
            utils.LinearChainParameterizedOption("PutOnBoard",
            [
                # Open fingers.
                create_change_fingers_option(self._pybullet_robot_sim,
                    "OpenFingers", types, params_space, open_fingers_func,
                    self._max_vel_norm, self._grasp_tol),
            ])

        # We track the correspondence between PyBullet object IDs and Object
        # instances. This correspondence changes with the task.
        self._id_to_object: Dict[int, Object] = {}

    def _initialize_pybullet(self) -> None:
        """Run super(), then handle sandwich-specific initialization."""
        super()._initialize_pybullet()

        # Load table in both the main client and the copy.
        self._table_id = p.loadURDF(
            utils.get_env_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._table_id,
            self._table_pose,
            self._table_orientation,
            physicsClientId=self._physics_client_id)
        p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                   useFixedBase=True,
                   physicsClientId=self._physics_client_id2)
        p.resetBasePositionAndOrientation(
            self._table_id,
            self._table_pose,
            self._table_orientation,
            physicsClientId=self._physics_client_id2)

        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert self.using_gui, \
                "using_gui must be True to use pybullet_draw_debug."
            # Draw the workspace on the table for clarity.
            p.addUserDebugLine([self.x_lb, self.y_lb, self.table_height],
                               [self.x_ub, self.y_lb, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=self._physics_client_id)
            p.addUserDebugLine([self.x_lb, self.y_ub, self.table_height],
                               [self.x_ub, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=self._physics_client_id)
            p.addUserDebugLine([self.x_lb, self.y_lb, self.table_height],
                               [self.x_lb, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=self._physics_client_id)
            p.addUserDebugLine([self.x_ub, self.y_lb, self.table_height],
                               [self.x_ub, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=self._physics_client_id)
            # Draw coordinate frame labels for reference.
            p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0],
                               physicsClientId=self._physics_client_id)
            p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0],
                               physicsClientId=self._physics_client_id)
            p.addUserDebugText("z", [0, 0, 0.25], [0.0, 0.0, 0.0],
                               physicsClientId=self._physics_client_id)
            # Draw the pick z location at the x/y midpoint.
            mid_x = (self.x_ub + self.x_lb) / 2
            mid_y = (self.y_ub + self.y_lb) / 2
            p.addUserDebugText("*", [mid_x, mid_y, self.pick_z],
                               [1.0, 0.0, 0.0],
                               physicsClientId=self._physics_client_id)

        # Create board.
        # TODO: fix robot orientation in sandwich env.
        # The poses here are not important because they are overwritten by
        # the state values when a task is reset.
        pose = ((self.board_x_lb + self.board_x_ub) / 2,
                (self.board_y_lb + self.board_y_ub) / 2,
                self.table_height + self.board_thickness / 2)
        # Create the collision shape.
        half_extents = [
            self.board_width / 2, self.board_length / 2,
            self.board_thickness / 2
        ]
        color = self.board_color
        orientation = self._default_orn
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._physics_client_id)
        # Create the visual_shape.
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self._physics_client_id)
        # Create the body.
        self._board_id = p.createMultiBody(
            baseMass=-1,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=pose,
            baseOrientation=orientation,
            physicsClientId=self._physics_client_id)

        # Create holder.
        color = self.holder_color
        orientation = self._default_orn
        base_pose = ((self.holder_x_lb + self.holder_x_ub) / 2,
                     (self.holder_y_lb + self.holder_y_ub) / 2,
                     self.table_height + self.holder_thickness / 2)
        # Holder base.
        # Create the collision shape.
        base_half_extents = [
            self.holder_width / 2, self.holder_length / 2,
            self.holder_thickness / 2
        ]
        base_collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=base_half_extents,
            physicsClientId=self._physics_client_id)
        # Create the visual shape.
        base_visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=base_half_extents,
            rgbaColor=color,
            physicsClientId=self._physics_client_id)
        link_positions = []
        link_collision_shape_indices = []
        link_visual_shape_indices = []
        # Create links to prevent movement in the x direction.
        link_thickness = self.holder_thickness
        max_ingredient_radius = max(self.ingredient_radii.values())
        for x_offset in [
                max_ingredient_radius + link_thickness,
                -(max_ingredient_radius + link_thickness)
        ]:
            pose = (x_offset, 0,
                    self.holder_thickness / 2 + self.holder_height / 2)
            link_positions.append(pose)
            half_extents = [
                link_thickness / 2, self.holder_length / 2,
                self.holder_height / 2
            ]
            collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self._physics_client_id)
            link_collision_shape_indices.append(collision_id)
            visual_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=color,
                physicsClientId=self._physics_client_id)
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
        self._holder_id = p.createMultiBody(
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
            physicsClientId=self._physics_client_id)

        # Create ingredients.  Note that we create the maximum number once, and
        # then later on, in reset_state(), we will remove ingredients from the
        # workspace (teleporting them far away) based on the state.
        self._ingredient_ids = {}  # ingredient type to ids
        for ingredient in CFG.sandwich_ingredients_test:
            num_ing = max(max(CFG.sandwich_ingredients_train[ingredient]),
                          max(CFG.sandwich_ingredients_test[ingredient]))
            color = tuple(self.ingredient_colors[ingredient]) + (1.0, )
            shape = self.ingredient_shapes[ingredient]
            radius = self.ingredient_radii[ingredient]
            half_extents = (radius, radius, self.ingredient_thickness)
            self._ingredient_ids[ingredient] = []
            for _ in range(num_ing):
                if shape == 0:
                    pid = create_pybullet_block(color, half_extents,
                                                self._obj_mass,
                                                self._obj_friction,
                                                self._default_orn,
                                                self._physics_client_id)
                else:
                    assert shape == 1
                    pid = create_pybullet_cylinder(color, radius,
                                                   self.ingredient_thickness,
                                                   self._obj_mass,
                                                   self._obj_friction,
                                                   self._default_orn,
                                                   self._physics_client_id)

                self._ingredient_ids[ingredient].append(pid)

    def _create_pybullet_robot(
            self, physics_client_id: int) -> SingleArmPyBulletRobot:
        ee_home = (self.robot_init_x, self.robot_init_y, self.robot_init_z)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home)

    def _extract_robot_state(self, state: State) -> Array:
        # TODO: we're probably going to need orientation here as well
        return np.array([
            state.get(self._robot, "pose_x"),
            state.get(self._robot, "pose_y"),
            state.get(self._robot, "pose_z"),
            self._fingers_state_to_joint(state.get(self._robot, "fingers")),
        ],
                        dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_sandwich"

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle sandwich-specific resetting."""
        super()._reset_state(state)

        # Reset ingredients based on the state.
        ing_objs = state.get_objects(self._ingredient_type)
        self._id_to_object = {}
        unused_ids = {k: list(v) for k, v in self._ingredient_ids.items()}
        for ing_obj in ing_objs:
            ing_type = self._obj_to_ingredient(ing_obj, state)
            ing_id = unused_ids[ing_type].pop()
            self._id_to_object[ing_id] = ing_obj
            x = state.get(ing_obj, "pose_x")
            y = state.get(ing_obj, "pose_y")
            z = state.get(ing_obj, "pose_z")
            rot = state.get(ing_obj, "rot")
            if abs(rot - np.pi / 2) < 1e-3:
                orn = p.getQuaternionFromEuler([np.pi / 2, 0.0, 0.0])
            else:
                assert abs(rot - 0) < 1e-3
                orn = self._default_orn
            p.resetBasePositionAndOrientation(
                ing_id, [x, y, z],
                orn,
                physicsClientId=self._physics_client_id)

        import time
        while True:
            p.stepSimulation(self._physics_client_id)
            time.sleep(0.001)

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state."""
        import ipdb
        ipdb.set_trace()

    def _get_tasks(self, num_tasks: int, num_ingredients: Dict[str, List[int]],
                   rng: np.random.Generator) -> List[Task]:
        tasks = super()._get_tasks(num_tasks, num_ingredients, rng)
        return self._add_pybullet_state_to_tasks(tasks)

    def _load_task_from_json(self, json_file: Path) -> Task:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        import ipdb
        ipdb.set_trace()

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        import ipdb
        ipdb.set_trace()

    def _fingers_state_to_joint(self, fingers_state: float) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either 0 or 1. Transform them to be
        either self._pybullet_robot.closed_fingers or
        self._pybullet_robot.open_fingers.
        """
        assert fingers_state in (0.0, 1.0)
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
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
