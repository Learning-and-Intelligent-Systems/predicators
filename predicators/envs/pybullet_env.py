"""Base class for a PyBullet environment.

Contains useful common code.
"""

import abc
import logging
from pprint import pformat
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib
import numpy as np
import pybullet as p
from gym.spaces import Box
from PIL import Image

from predicators import utils
from predicators.envs import BaseEnv
from predicators.pybullet_helpers.camera import create_gui_connection
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.objects import update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, Mask, Object, \
    Observation, State, Video
from predicators.utils import PyBulletState


class PyBulletEnv(BaseEnv):
    """Base class for a PyBullet environment."""
    # Parameters that aren't important enough to need to clog up settings.py

    # General robot parameters.
    # grasp_tol: value for which the objects with distance below to are
    # considered to be grasped, and also the value change finger option can be
    # terminated.
    grasp_tol: ClassVar[float] = 5e-2  # for large objects
    grasp_tol_small: ClassVar[float] = 5e-4  # for small objects
    _finger_action_tol: ClassVar[float] = 1e-4
    open_fingers: ClassVar[float] = 0.04
    closed_fingers: ClassVar[float] = 0.01
    robot_base_pos: Optional[Tuple[float, float, float]] = None
    robot_base_orn: Optional[Tuple[float, float, float, float]] = None

    # Object parameters.
    _obj_mass: ClassVar[float] = 0.5
    _obj_friction: ClassVar[float] = 1.2
    _obj_colors_main: ClassVar[Sequence[Tuple[float, float, float, float]]] = [
        (0.95, 0.05, 0.1, 1.),
        (0.05, 0.95, 0.1, 1.),
        (0.1, 0.05, 0.95, 1.),
        (0.4, 0.05, 0.6, 1.),
        (0.6, 0.4, 0.05, 1.),
        (0.05, 0.04, 0.6, 1.),
        (0.95, 0.95, 0.1, 1.),
        (0.95, 0.05, 0.95, 1.),
        (0.05, 0.95, 0.95, 1.)]
    _obj_colors: ClassVar[Sequence[Tuple[float, float, float, float]]] =\
        _obj_colors_main + [
        (0.941, 0.196, 0.196, 1.),  # Red
        (0.196, 0.941, 0.196, 1.),  # Green
        (0.196, 0.196, 0.941, 1.),  # Blue
        (0.941, 0.941, 0.196, 1.),  # Yellow
        (0.941, 0.196, 0.941, 1.),  # Magenta
        (0.196, 0.941, 0.941, 1.),  # Cyan
        (0.941, 0.588, 0.196, 1.),  # Orange
        (0.588, 0.196, 0.941, 1.),  # Purple
        (0.196, 0.941, 0.588, 1.),  # Teal
        (0.941, 0.196, 0.588, 1.),  # Pink
        (0.588, 0.941, 0.196, 1.),  # Lime
        (0.196, 0.588, 0.941, 1.),  # Sky Blue
    ]
    _out_of_view_xy: ClassVar[Sequence[float]] = [10.0, 10.0]
    _default_orn: ClassVar[Sequence[float]] = [0.0, 0.0, 0.0, 1.0]

    # Camera parameters.
    _camera_distance: ClassVar[float] = 0.8
    _camera_yaw: ClassVar[float] = 90.0
    _camera_pitch: ClassVar[float] = -24
    _camera_target: ClassVar[Pose3D] = (1.65, 0.75, 0.42)
    _camera_fov: ClassVar[float] = 60
    _debug_text_position: ClassVar[Pose3D] = (1.65, 0.25, 0.75)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # When an object is held, a constraint is created to prevent slippage.
        self._held_constraint_id: Optional[int] = None
        self._held_obj_to_base_link: Optional[Any] = None
        self._held_obj_id: Optional[int] = None

        # Set up all the static PyBullet content.
        self._physics_client_id, self._pybullet_robot, pybullet_bodies = \
            self.initialize_pybullet(self.using_gui)
        self._store_pybullet_bodies(pybullet_bodies)

        # What are they used for??
        # It's used in get_state, reset_state and labeling state.
        # Should be populated at reset or reset state.
        self._objects: List[Object] = []

    def get_object_by_id(self, obj_id: int) -> Object:
        for obj in self._objects:
            if obj.id == obj_id:
                return obj
        raise ValueError(f"Object with ID {obj_id} not found")

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Initialize the PyBullet environment.

        This method initializes the PyBullet physics simulation, loads the robot
        and shared object models, and returns the physics client ID, the robot
        instance, and a dictionary containing other object IDs and any additional
        information that needs to be tracked.

        Args:
            using_gui (bool): If True, the PyBullet GUI will be used. Otherwise,
                            the simulation will run in headless mode.

        Returns:
            Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
                - int: The physics client ID.
                - SingleArmPyBulletRobot: The robot instance.
                - Dict[str, Any]: A dictionary containing object IDs and other
                                information from PyBullet that needs to be
                                tracked.

        Notes:
            - This is a public class method because it is also used by the
            oracle options.
            - This method loads object models that are shared across tasks.
            These objects can have different poses or colors, and the number of
            objects can vary across tasks (e.g., the number of blocks in the
            blocks domain). However, an object's size cannot be changed after
            loading.
            - Task-specific objects that need to be loaded with different sizes
            or other properties should be handled in the
            `_create_task_specific_objects` method, which is called during each
            task's reset.
            - Subclasses may override this method to load additional assets. In
            the subclass, register all object IDs here and move them out of view
            in the `reset_custom_env_state` method.
        """
        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if using_gui:  # pragma: no cover
            physics_client_id = create_gui_connection(
                camera_distance=cls._camera_distance,
                camera_yaw=cls._camera_yaw,
                camera_pitch=cls._camera_pitch,
                camera_target=cls._camera_target,
            )
        else:
            physics_client_id = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=physics_client_id)

        # Load plane.
        p.loadURDF(utils.get_env_asset_path("urdf/plane.urdf"), [0, 0, 0],
                   useFixedBase=True,
                   physicsClientId=physics_client_id)

        # Load robot.
        pybullet_robot = cls._create_pybullet_robot(physics_client_id)

        # Set gravity.
        p.setGravity(0., 0., -10., physicsClientId=physics_client_id)

        return physics_client_id, pybullet_robot, {}

    @abc.abstractmethod
    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store any bodies created in cls.initialize_pybullet().

        This is separate from the initialization because the
        initialization is a class method (which is needed for options).
        Subclasses should decide what bodies to keep.
        """
        raise NotImplementedError("Override me!")

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((cls.robot_init_x, cls.robot_init_y, cls.robot_init_z),
                       robot_ee_orn)

        if cls.robot_base_pos is None or cls.robot_base_orn is None:
            base_pose = None
        else:
            base_pose = Pose(cls.robot_base_pos, cls.robot_base_orn)

        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home,
                                                base_pose)

    def _extract_robot_state(self, state: State) -> Array:
        """Given a State, extract the robot state, to be passed into
        self._pybullet_robot.reset_state().

        This should be the same type as the return value of
        self._pybullet_robot.get_state().
        """

        # EE Position
        def get_pos_feature(state, feature_name):
            if feature_name in self._robot.type.feature_names:
                return state.get(self._robot, feature_name)
            elif f"pose_{feature_name}" in self._robot.type.feature_names:
                return state.get(self._robot, f"pose_{feature_name}")
            else:
                raise ValueError(f"Cannot find robot pos '{feature_name}'")

        rx = get_pos_feature(state, "x")
        ry = get_pos_feature(state, "y")
        rz = get_pos_feature(state, "z")

        # EE Orientation
        _, default_tilt, default_wrist = p.getEulerFromQuaternion(
            self.get_robot_ee_home_orn())
        if "tilt" in self._robot.type.feature_names:
            tilt = state.get(self._robot, "tilt")
        else:
            tilt = default_tilt
        if "wrist" in self._robot.type.feature_names:
            wrist = state.get(self._robot, "wrist")
        else:
            wrist = default_wrist
        qx, qy, qz, qw = p.getQuaternionFromEuler([0.0, tilt, wrist])

        # Fingers
        f = state.get(self._robot, "fingers")
        f = self._fingers_state_to_joint(self._pybullet_robot, f)

        return np.array([rx, ry, rz, qx, qy, qz, qw, f], dtype=np.float32)

    @abc.abstractmethod
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return a list of pybullet IDs corresponding to objects in the
        simulator that should be checked when determining whether one is
        held."""
        raise NotImplementedError("Override me!")

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        # Get the current state of the robot, including the orientation quaternion
        rx, ry, rz, qx, qy, qz, qw, rf = self._pybullet_robot.get_state()

        # Convert the quaternion to a rotation matrix
        rotation_matrix = p.getMatrixFromQuaternion([qx, qy, qz, qw])
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

        # Define the initial normal vectors for the fingers
        if CFG.pybullet_robot == "panda":
            # gripper rotated 90deg so parallel to x-axis
            normal = np.array([1., 0., 0.], dtype=np.float32)
        elif CFG.pybullet_robot == "fetch":
            # gripper parallel to y-axis
            normal = np.array([0., 1., 0.], dtype=np.float32)
        else:  # pragma: no cover
            # Shouldn't happen unless we introduce a new robot.
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        # Transform the normal vectors using the rotation matrix
        transformed_normal = rotation_matrix.dot(normal)
        transformed_normal_neg = rotation_matrix.dot(-1 * normal)

        return {
            self._pybullet_robot.left_finger_id: transformed_normal,
            self._pybullet_robot.right_finger_id: transformed_normal_neg,
        }

    @classmethod
    def _fingers_state_to_joint(cls, pybullet_robot: SingleArmPyBulletRobot,
                                finger_state: float) -> float:
        """Map the fingers in the given *State* to joint values for
        PyBullet."""
        # If open_fingers is undefined, use 1.0 as the default.
        subs = {
            cls.open_fingers: pybullet_robot.open_fingers,
            cls.closed_fingers: pybullet_robot.closed_fingers,
        }
        match = min(subs, key=lambda k: abs(k - finger_state))
        return subs[match]

    @classmethod
    def _fingers_joint_to_state(cls, pybullet_robot: SingleArmPyBulletRobot,
                                finger_joint: float) -> float:
        """Inverse of _fingers_state_to_joint()."""
        subs = {
            pybullet_robot.open_fingers: cls.open_fingers,
            pybullet_robot.closed_fingers: cls.closed_fingers,
        }
        match = min(subs, key=lambda k: abs(k - finger_joint))
        return subs[match]

    @property
    def action_space(self) -> Box:
        return self._pybullet_robot.action_space

    def simulate(self, state: State, action: Action) -> State:
        # Optimization: check if we're already in the right state.
        # self._current_observation is None at the beginning
        # state is not allclose to self._current_state when the state has been
        # updated, so it first calls _reset_state to update the pybullet state
        if self._current_observation is None or \
            not state.allclose(self._current_state):
            self._current_observation = state
            self._reset_state(state)
        return self.step(action)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise NotImplementedError("A PyBullet environment cannot render "
                                  "arbitrary states.")

    def reset(self,
              train_or_test: str,
              task_idx: int,
              render: bool = False) -> Observation:
        state = super().reset(train_or_test, task_idx)
        self._reset_state(state)
        # Converts the State into a PyBulletState.
        self._current_observation = self._get_state()
        # logging.debug(f"Reset state:\n{pformat(state.pretty_str())}")
        observation = self.get_observation(render=render)
        return observation

    def _reset_state(self, state: State) -> None:
        """Reset the PyBullet state to match the given state.

        Used in initialization and bilevel planning.
        """
        self._objects = list(state.data)
        # 1) Clear old constraint if we had a held object
        if self._held_constraint_id is not None:
            p.removeConstraint(self._held_constraint_id,
                               physicsClientId=self._physics_client_id)
            self._held_constraint_id = None
        self._held_obj_to_base_link = None
        self._held_obj_id = None

        # 2) Reset robot pose
        self._pybullet_robot.reset_state(self._extract_robot_state(state))

        # I want to have a step that creates task specific objects before reset
        # their positions, what should I call this?
        self._create_task_specific_objects(state)

        # 3) Reset all known objects (position, orientation, etc.)
        for obj in self._objects:
            if obj.type.name == "robot":
                continue
            if obj.type.name == "position":
                # abstract entity
                continue
            self._reset_single_object(obj, state)

        # 4) Let the subclass do any additional specialized resetting
        self._reset_custom_env_state(state)

        # 5) (Optional) Check for reconstruction mismatch in debug mode
        #    (Helps catch if the environment hook overwrites something.)
        reconstructed = self._get_state()
        if not reconstructed.allclose(state):
            logging.warning("Could not reconstruct state exactly in reset.")

    @abc.abstractmethod
    def _create_task_specific_objects(self, state: State) -> None:
        raise NotImplementedError("Override me!")

    def _reset_single_object(self, obj: Object, state: State) -> None:
        """Shared logic for setting position/orientation and constraints."""
        # If the environment doesn’t want the base class to handle it,
        # it can skip or override this method. By default, look for
        # standard features: x, y, z, rot, is_held.

        # 1) Position/orientation if those features exist
        # try:
        features = obj.type.feature_names
        cur_x, cur_y, cur_z = p.getBasePositionAndOrientation(
            obj.id, physicsClientId=self._physics_client_id)[0]
        # except:
        #     breakpoint()
        px = state.get(obj, "x") if "x" in obj.type.feature_names else cur_x
        py = state.get(obj, "y") if "y" in obj.type.feature_names else cur_y
        pz = state.get(obj, "z") if "z" in obj.type.feature_names else cur_z

        if "rot" in features:
            angle = state.get(obj, "rot")
            # Convert from 2D angle to a 3D quaternion (assuming rotation around z)
            orn = p.getQuaternionFromEuler([0.0, 0.0, angle])
        else:
            orn = self._default_orn  # e.g. (0,0,0,1)

        # 2) Update the object’s position/orientation in PyBullet
        update_object(obj.id, [px, py, pz],
                      orn,
                      physics_client_id=self._physics_client_id)

        # 3) If there's an is_held feature, reattach constraints if needed
        if "is_held" in features:
            if state.get(obj, "is_held") > 0.5:
                # attach constraint
                self._held_obj_id = obj.id
                self._create_grasp_constraint()
                # self._create_grasp_constraint_for_object(obj.id)
                # Optionally store the parent link transform
                world_to_base_link = get_link_state(
                    self._pybullet_robot.robot_id,
                    self._pybullet_robot.end_effector_id,
                    physics_client_id=self._physics_client_id).com_pose
                obj_to_base_link = p.invertTransform(*p.multiplyTransforms(
                    world_to_base_link[0], world_to_base_link[1], (px, py,
                                                                   pz), orn))
                self._held_obj_to_base_link = obj_to_base_link

    @abc.abstractmethod
    def _reset_custom_env_state(self, state: State) -> None:
        """Hook for environment-specific resetting (colors, water, etc.).

        Subclasses can override or extend this if needed.
        """
        raise NotImplementedError("Override me!")

    def _get_state(self, render_obs: bool = False) -> State:
        """Reads the PyBullet scene into a `State` (PyBulletState). It takes
        care of:

        * robot features [x, y, z, tilt, wrist, fingers]
        * object features [x, y, z, rot, is_held]
        the other feature extractors should be implemented in the subclasses via
        `_extract_feature`.
        """
        state_dict: Dict[Object, Dict[str, float]] = {}

        # --- 1) Robot ---
        robot_state = self._get_robot_state_dict()
        state_dict[self._robot] = robot_state

        # --- 2) Other Objects ---
        for obj in self._objects:
            if obj.type.name in ["robot"]:
                continue

            obj_features = obj.type.feature_names
            obj_dict = {}

            if obj.type.name == "position":
                for feature in obj_features:
                    obj_dict[feature] = self._extract_feature(obj, feature)
                state_dict[obj] = obj_dict
                continue

            # Basic features
            try:
                (px, py, pz), orn = p.getBasePositionAndOrientation(
                obj.id, physicsClientId=self._physics_client_id)
            except:
                breakpoint()
            if "x" in obj_features:
                obj_dict["x"] = px
            if "y" in obj_features:
                obj_dict["y"] = py
            if "z" in obj_features:
                obj_dict["z"] = pz
            if "rot" in obj_features:
                yaw = p.getEulerFromQuaternion(orn)[2]
                obj_dict["rot"] = yaw
            if "is_held" in obj_features:
                obj_dict["is_held"] = 1.0 if obj.id == self._held_obj_id \
                                            else 0.0

            if "r" in obj_features or "b" in obj_features or \
                "g" in obj_features:
                # TODO: also handle color_r, color_b, ...
                visual_data = p.getVisualShapeData(
                    obj.id, physicsClientId=self._physics_client_id)[0]
                (r, g, b, a) = visual_data[7]
                obj_dict["r"] = r
                obj_dict["g"] = g
                obj_dict["b"] = b

            # Additional features
            for feature in obj_features:
                if feature not in [
                        "x", "y", "z", "rot", "is_held", "r", "g", "b"
                ]:
                    obj_dict[feature] = self._extract_feature(obj, feature)

            state_dict[obj] = obj_dict

        # Convert to a PyBulletState
        # try:
        state = utils.create_state_from_dict(state_dict)
        # except:
        #     breakpoint()
        joint_positions = self._pybullet_robot.get_joints()
        pyb_state = PyBulletState(
            state.data, simulator_state={"joint_positions": joint_positions})
        return pyb_state

    @abc.abstractmethod
    def _extract_feature(self, obj: Object, feature: str) -> float:
        raise NotImplementedError("Override me!")

    def _get_robot_state_dict(self) -> None:
        """Get dict state of the robot."""
        r_dict = {}
        r_features = self._robot.type.feature_names
        if CFG.env == "pybullet_cover":
            rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
            hand = (ry - self.y_lb) / (self.y_ub - self.y_lb)
            r_dict.update({"hand": hand, "pose_x": rx, "pose_z": rz})
        elif CFG.env == "pybullet_blocks":
            rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
            fingers = self._fingers_joint_to_state(self._pybullet_robot, rf)
            r_dict.update({
                "pose_x": rx,
                "pose_y": ry,
                "pose_z": rz,
                "fingers": fingers
            })
        else:
            rx, ry, rz, qx, qy, qz, qw, rf = self._pybullet_robot.get_state()
            r_dict.update({"x": rx, "y": ry, "z": rz, "fingers": rf})
            _, tilt, wrist = p.getEulerFromQuaternion([qx, qy, qz, qw])
            if "tilt" in r_features:
                r_dict["tilt"] = tilt
            if "wrist" in r_features:
                r_dict["wrist"] = wrist
        return r_dict

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:  # pragma: no cover
        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        del action, caption  # unused

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._camera_target,
            distance=self._camera_distance,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._physics_client_id)

        width = CFG.pybullet_camera_width
        height = CFG.pybullet_camera_height

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self._camera_fov,
            aspect=float(width / height),
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self._physics_client_id)

        (_, _, px, _,
         _) = p.getCameraImage(width=width,
                               height=height,
                               viewMatrix=view_matrix,
                               projectionMatrix=proj_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL,
                               physicsClientId=self._physics_client_id)

        rgb_array = np.array(px).reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]
        return [rgb_array]

    def render_segmented_obj(
        self,
        action: Optional[Action] = None,
        caption: Optional[str] = None,
    ) -> Tuple[Image.Image, Dict[Object, Mask]]:
        """Render the scene and the segmented objects in the scene."""
        del action, caption  # unused
        # if not self.using_gui:
        #     raise Exception(
        #         "Rendering only works with GUI on. See "
        #         "https://github.com/bulletphysics/bullet3/issues/1157")

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._camera_target,
            distance=self._camera_distance,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._physics_client_id)

        width = CFG.pybullet_camera_width
        height = CFG.pybullet_camera_height

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width / height),
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self._physics_client_id)

        # Initialize an empty dictionary
        mask_dict: Dict[Object, Mask] = {}

        # Get the original image and segmentation mask
        (_, _, rgbImg, _,
         segImg) = p.getCameraImage(width=width,
                                    height=height,
                                    viewMatrix=view_matrix,
                                    projectionMatrix=proj_matrix,
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                    physicsClientId=self._physics_client_id)

        # Convert to numpy arrays
        original_image: np.ndarray = np.array(rgbImg, dtype=np.uint8).reshape(
            (height, width, 4))
        seg_image = np.array(segImg).reshape((height, width))

        state_img = Image.fromarray(  # type: ignore[no-untyped-call]
            original_image[:, :, :3])

        # Iterate over all bodies to be labeled
        for obj in self._objects:
            body_id = obj.id
            mask = seg_image == body_id
            mask_dict[obj] = mask

        return state_img, mask_dict

    def get_observation(self, render: bool = False) -> Observation:
        """Get the current observation of this environment.

        Currently, this just return a copy of the state and optionally a
        rendered image.
        """
        self._current_observation = self._get_state()
        assert isinstance(self._current_observation, PyBulletState)
        state_copy = self._current_observation.copy()

        if render:
            state_copy.add_images_and_masks(*self.render_segmented_obj())

        return state_copy

    def step(self, action: Action, render_obs: bool = False) -> Observation:
        """Execute one environment step with the given action.

        This method handles:
        1. Robot joint control by converting action to target positions
        2. Management of held objects and grasping constraints
        3. Physics simulation stepping
        4. Object grasp detection and constraint creation/removal
        5. `self._current_observation` update

        Args:
            action (Action): The action to execute, containing target joint
            positions
            render_obs (bool, optional): Whether to include RGB observation.
                Defaults to False.

        Returns:
            Observation: Updated environment observation after executing the
            action. May include an image if render_obs=True or
            CFG.rgb_observation=True.
        """
        # Send the action to the robot.
        target_joint_positions = action.arr.tolist()
        self._pybullet_robot.set_motors(target_joint_positions)

        # If we are setting the robot joints directly, and if there is a held
        # object, we need to reset the pose of the held object directly. This
        # is because the PyBullet constraints don't seem to play nicely with
        # resetJointState (the robot will sometimes drop the object).
        if CFG.pybullet_control_mode == "reset" and \
            self._held_obj_id is not None:
            world_to_base_link = get_link_state(
                self._pybullet_robot.robot_id,
                self._pybullet_robot.end_effector_id,
                physics_client_id=self._physics_client_id).com_pose
            base_link_to_held_obj = p.invertTransform(
                *self._held_obj_to_base_link)
            world_to_held_obj = p.multiplyTransforms(world_to_base_link[0],
                                                     world_to_base_link[1],
                                                     base_link_to_held_obj[0],
                                                     base_link_to_held_obj[1])
            p.resetBasePositionAndOrientation(
                self._held_obj_id,
                world_to_held_obj[0],
                world_to_held_obj[1],
                physicsClientId=self._physics_client_id)

        # Step the simulation here before adding or removing constraints
        # because detect_held_object() should use the updated state.
        if CFG.pybullet_control_mode != "reset":
            for _ in range(CFG.pybullet_sim_steps_per_action):
                p.stepSimulation(physicsClientId=self._physics_client_id)

        # If not currently holding something, and fingers are closing, check
        # for a new grasp.
        if self._held_constraint_id is None and self._fingers_closing(action):
            logging.debug("Finger closing")
            # Detect if an object is held. If so, create a grasp constraint.
            self._held_obj_id = self._detect_held_object()
            # logging.debug(f"Detected held object: {self._held_obj_id}")
            # breakpoint()
            if self._held_obj_id is not None:
                self._create_grasp_constraint()

        # If placing, remove the grasp constraint.
        if self._held_constraint_id is not None and \
            self._fingers_opening(action):
            p.removeConstraint(self._held_constraint_id,
                               physicsClientId=self._physics_client_id)
            self._held_constraint_id = None
            logging.debug("Finger opening")
            self._held_obj_id = None

        # self._current_observation = self._get_state()

        # Depending on the observation mode, either return object-centric state
        # or object_centric + rgb observation
        observation = self.get_observation(render=CFG.rgb_observation or\
                                                render_obs)

        return observation

    def _detect_held_object(self) -> Optional[int]:
        """Return the PyBullet object ID of the held object if one exists.

        If multiple objects are within the grasp tolerance, return the
        one that is closest.
        """
        expected_finger_normals = self._get_expected_finger_normals()
        closest_held_obj = None
        closest_held_obj_dist = float("inf")
        for obj_id in self._get_object_ids_for_held_check():
            for finger_id, expected_normal in expected_finger_normals.items():
                assert abs(np.linalg.norm(expected_normal) - 1.0) < 1e-5
                # Find points on the object that are within grasp_tol distance
                # of the finger. Note that we use getClosestPoints instead of
                # getContactPoints because we still want to consider the object
                # held even if there is a tiny distance between the fingers and
                # the object.
                closest_points = p.getClosestPoints(
                    bodyA=self._pybullet_robot.robot_id,
                    bodyB=obj_id,
                    distance=self.grasp_tol_small,
                    linkIndexA=finger_id,
                    physicsClientId=self._physics_client_id)
                for point in closest_points:
                    # If the contact normal is substantially different from
                    # the expected contact normal, this is probably an object
                    # on the outside of the fingers, rather than the inside.
                    # A perfect score here is 1.0 (normals are unit vectors).
                    contact_normal = point[7]
                    score = expected_normal.dot(contact_normal)
                    # logging.debug(f"With obj {obj_id}, score: {score}")
                    assert -1.0 <= score <= 1.0

                    # Take absolute as object/gripper could be rotated 180
                    # degrees in the given axis.
                    if np.abs(score) < 0.9:
                        continue
                    # Handle the case where multiple objects pass this check
                    # by taking the closest one. This should be rare, but it
                    # can happen when two objects are stacked and the robot is
                    # unstacking the top one.
                    contact_distance = point[8]
                    if contact_distance < closest_held_obj_dist:
                        closest_held_obj = obj_id
                        closest_held_obj_dist = contact_distance
        return closest_held_obj

    def _create_grasp_constraint(self) -> None:
        assert self._held_obj_id is not None
        base_link_to_world = np.r_[p.invertTransform(
            *p.getLinkState(self._pybullet_robot.robot_id,
                            self._pybullet_robot.end_effector_id,
                            physicsClientId=self._physics_client_id)[:2])]
        world_to_obj = np.r_[p.getBasePositionAndOrientation(
            self._held_obj_id, physicsClientId=self._physics_client_id)]
        self._held_obj_to_base_link = p.invertTransform(*p.multiplyTransforms(
            base_link_to_world[:3], base_link_to_world[3:], world_to_obj[:3],
            world_to_obj[3:]))
        self._held_constraint_id = p.createConstraint(
            parentBodyUniqueId=self._pybullet_robot.robot_id,
            parentLinkIndex=self._pybullet_robot.end_effector_id,
            childBodyUniqueId=self._held_obj_id,
            childLinkIndex=-1,  # -1 for the base
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=self._held_obj_to_base_link[0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=self._held_obj_to_base_link[1],
            physicsClientId=self._physics_client_id)

    def _fingers_closing(self, action: Action) -> bool:
        """Check whether this action is working toward closing the fingers."""
        f_delta = self._action_to_finger_delta(action)
        return f_delta < -self._finger_action_tol

    def _fingers_opening(self, action: Action) -> bool:
        """Check whether this action is working toward opening the fingers."""
        f_delta = self._action_to_finger_delta(action)
        # logging.debug(f"Finger delta: {f_delta}")
        return f_delta > self._finger_action_tol

    def _get_finger_position(self, state: State) -> float:
        # Arbitrarily use the left finger as reference.
        state = cast(utils.PyBulletState, state)
        finger_joint_idx = self._pybullet_robot.left_finger_joint_idx
        return state.joint_positions[finger_joint_idx]

    def _action_to_finger_delta(self, action: Action) -> float:
        assert isinstance(self._current_observation, State)
        finger_position = self._get_finger_position(self._current_observation)
        target = action.arr[-1]
        # logging.debug(f"Finger position: {finger_position}, target: {target}")
        return target - finger_position

    def _add_pybullet_state_to_tasks(
            self, tasks: List[EnvironmentTask]) -> List[EnvironmentTask]:
        """Converts the task initial states into PyBulletStates.

        This is used in generating tasks.
        """
        pybullet_tasks = []
        for task in tasks:
            # Reset the robot.
            init = task.init
            # Extract the joints.
            # YC: Probably need to reset_state here so I can then get an
            # observation, would it work without the reset_state?
            # Attempt 2: First reset it.
            self._current_observation = init
            self._reset_state(init)
            # Cast _current_observation from type State to PybulletState
            joint_positions = self._pybullet_robot.get_joints()
            self._current_observation = utils.PyBulletState(
                init.data.copy(), simulator_state=joint_positions)
            # Attempt 1: Let's try to get a rendering directly first
            pybullet_init = self.get_observation(render=CFG.render_init_state)
            pybullet_init.option_history = []
            # # <Original code
            # self._pybullet_robot.reset_state(self._extract_robot_state(init))
            # joint_positions = self._pybullet_robot.get_joints()
            # pybullet_init = utils.PyBulletState(
            #     init.data.copy(), simulator_state=joint_positions)
            # # >
            pybullet_task = EnvironmentTask(pybullet_init, task.goal)
            pybullet_tasks.append(pybullet_task)
        return pybullet_tasks

    @classmethod
    def get_robot_ee_home_orn(cls) -> Quaternion:
        """Public for use by oracle options."""
        robot_ee_orns = CFG.pybullet_robot_ee_orns[cls.get_name()]
        return robot_ee_orns[CFG.pybullet_robot]


def create_pybullet_block(
    color: Tuple[float, float, float, float],
    half_extents: Tuple[float, float, float],
    mass: float,
    friction: float,
    position: Sequence[Pose3D] = (0, 0, 0),
    orientation: Sequence[Quaternion] = (0, 0, 0, 1),
    physics_client_id: int = 0,
) -> int:
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by

    # Create the collision shape.
    collision_id = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=half_extents,
                                          physicsClientId=physics_client_id)

    # Create the visual_shape.
    visual_id = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=half_extents,
                                    rgbaColor=color,
                                    physicsClientId=physics_client_id)

    # Create the body.
    block_id = p.createMultiBody(baseMass=mass,
                                 baseCollisionShapeIndex=collision_id,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=position,
                                 baseOrientation=orientation,
                                 physicsClientId=physics_client_id)
    p.changeDynamics(
        block_id,
        linkIndex=-1,  # -1 for the base
        lateralFriction=friction,
        spinningFriction=friction,
        physicsClientId=physics_client_id)

    return block_id


def create_pybullet_sphere(
    color: Tuple[float, float, float, float],
    radius: float,
    mass: float,
    friction: float,
    position: Sequence[Pose3D] = (0, 0, 0),
    orientation: Sequence[Quaternion] = (0, 0, 0, 1),
    physics_client_id: int = 0,
) -> int:
    """A generic utility for creating a new sphere.

    Returns the PyBullet ID of the newly created sphere.
    """
    # Create the collision shape.
    collision_id = p.createCollisionShape(p.GEOM_SPHERE,
                                          radius=radius,
                                          physicsClientId=physics_client_id)

    # Create the visual shape.
    visual_id = p.createVisualShape(p.GEOM_SPHERE,
                                    radius=radius,
                                    rgbaColor=color,
                                    physicsClientId=physics_client_id)

    # Create the body.
    sphere_id = p.createMultiBody(baseMass=mass,
                                  baseCollisionShapeIndex=collision_id,
                                  baseVisualShapeIndex=visual_id,
                                  basePosition=position,
                                  baseOrientation=orientation,
                                  physicsClientId=physics_client_id)
    p.changeDynamics(
        sphere_id,
        linkIndex=-1,  # -1 for the base
        lateralFriction=friction,
        spinningFriction=friction,
        physicsClientId=physics_client_id)

    return sphere_id
