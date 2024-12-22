# pybullet_grow.py

import logging
import numpy as np
import pybullet as p
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple
from gym.spaces import Box

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.pybullet_helpers.geometry import Pose, Quaternion
from predicators import utils
from predicators.structs import Action, EnvironmentTask, State, Object, \
    Predicate, Type, GroundAtom, Array

from predicators.settings import CFG


class PyBulletGrowEnv(PyBulletEnv):
    """
    A simple PyBullet environment involving two cups (red and blue) and
    two jugs (red and blue). Each jug contains infinite liquid. Pouring
    from a jug of matching color into a cup increases the 'growth' in 
    that cup.

    We want the 'growth' of both cups to exceed some threshold as a goal.
    """

    # Workspace bounds. You can adjust these as you wish.
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = 0.2
    z_ub: ClassVar[float] = 0.75

    # Robot default pose and orientation
    robot_base_pos: ClassVar[Tuple[float, float, float]] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])  # pointing in +y

    # We define two cups and two jugs at fixed color-coded positions
    # or randomly sampled. They have some "growth" or "liquid" features.

    # For simplicity, we define just one predicate: CupGrown(cup).
    # We might also define "Holding", "JugPickedUp", etc.

    # Growth threshold for the cups to meet the goal
    growth_threshold: ClassVar[float] = 0.5

    # Hard-coded finger states for open/close
    open_fingers: ClassVar[float] = 0.4
    closed_fingers: ClassVar[float] = 0.1
    # Tolerance for "close enough" to pick up or hold
    grasp_tol: ClassVar[float] = 1e-2
    # How much we pour if we tilt near the max angle
    pour_rate: ClassVar[float] = 0.1

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Define Types:
        self._robot_type = Type(
            "robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
        self._cup_type = Type(
            "cup", ["x", "y", "z", "growth", "color"])
        self._jug_type = Type(
            "jug", ["x", "y", "z", "is_held", "tilt", "color"])

        # Define Objects (we will create them in tasks, but the "concept" is here).
        self._robot = Object("robot", self._robot_type)
        self._red_cup = Object("red_cup", self._cup_type)
        self._blue_cup = Object("blue_cup", self._cup_type)
        self._red_jug = Object("red_jug", self._jug_type)
        self._blue_jug = Object("blue_jug", self._jug_type)

        # Define Predicates
        self._CupGrown = Predicate(
            "CupGrown", [self._cup_type], self._CupGrown_holds)

        # For a simpler environment, we only define one 'goal' predicate in tasks:
        # "CupGrown" for each cup. Alternatively, you can define more.

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_grow"

    @property
    def predicates(self) -> Set[Predicate]:
        # We only define CupGrown in this simple example, but you could
        # define Holding, JugPickedUp, etc.
        return {self._CupGrown}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._CupGrown}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._cup_type, self._jug_type}

    @property
    def action_space(self) -> Box:
        """
        For example, we let the action be a 6D array:
            [delta_x, delta_y, delta_z, delta_tilt, delta_wrist, delta_fingers]
        all in [-1, 1], scaled inside the step function.
        """
        return Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    # -------------------------------------------------------------------------
    # Environment Setup

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """
        Create the PyBullet environment and the robot.
        Subclasses typically add additional environment assets here.
        """
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(using_gui)

        # You can add a table or floor plane, etc.
        # For example, let's add a table below the workspace:
        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        # Position the table so the top is at z=cls.z_lb
        p.resetBasePositionAndOrientation(
            table_id, (0.75, 1.35, 0.0),
            p.getQuaternionFromEuler([0., 0., np.pi / 2]),
            physicsClientId=physics_client_id
        )
        bodies["table_id"] = table_id

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """
        Store references to PyBullet IDs for environment assets.
        """
        self._table_id = pybullet_bodies["table_id"]

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        """
        Create a single-arm PyBullet robot.
        """
        # The EE orientation is usually set so that the gripper is down.
        ee_home_orn = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        ee_home = Pose(position=(0.75, 1.25, cls.z_ub - 0.1),
                       orientation=ee_home_orn)
        base_pose = Pose(cls.robot_base_pos, cls.robot_base_orn)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id,
                                                ee_home,
                                                base_pose)

    # -------------------------------------------------------------------------
    # Key Abstract Methods from PyBulletEnv

    def _extract_robot_state(self, state: State) -> np.ndarray:
        """
        Convert the State's stored robot features into the 8D (x, y, z, qx, qy, qz, qw, fingers)
        that the PyBullet robot expects in fetch.py. Or a 7D if ignoring orientation.
        
        For simplicity, let's store tilt/wrist as Euler angles in the State, 
        then convert to quaternion here.
        """
        # We'll just store a fixed orientation for the demonstration
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        z = state.get(self._robot, "z")
        fingers = state.get(self._robot, "fingers")
        tilt = state.get(self._robot, "tilt")
        wrist = state.get(self._robot, "wrist")
        # Convert Euler => quaternion
        orn = p.getQuaternionFromEuler([0.0, tilt, wrist])
        qx, qy, qz, qw = orn

        # Return as [x, y, z, qx, qy, qz, qw, fingers]
        return np.array([x, y, z, qx, qy, qz, qw, fingers], dtype=np.float32)

    def _get_state(self) -> State:
        """
        Create a State object from the current PyBullet simulation. 
        We read off robot state, plus the cups/jugs positions, plus any
        relevant features (like 'growth').
        """
        state_dict: Dict[Object, Dict[str, float]] = {}

        # 1) Robot
        rx, ry, rz, qx, qy, qz, qw, rf = self._pybullet_robot.get_state()
        # Convert orientation back into tilt/wrist
        roll, tilt, wrist = p.getEulerFromQuaternion([qx, qy, qz, qw])
        # We store the robot's features
        state_dict[self._robot] = {
            "x": rx,
            "y": ry,
            "z": rz,
            "fingers": self._fingers_joint_to_state(rf),
            "tilt": tilt,
            "wrist": wrist
        }

        # 2) For each cup and jug, we would have loaded them in _reset_state.
        #    The environment keeps track of their PyBullet IDs => we read their positions now.
        for body_id, obj in self._obj_id_to_obj.items():
            if obj.type == self._cup_type:
                (cx, cy, cz), _ = p.getBasePositionAndOrientation(
                    body_id, physicsClientId=self._physics_client_id)
                # "growth" from custom dictionary, or we can keep track in a side-dict
                growth = self._cup_growth[obj]
                color_val = 1.0 if "red" in obj.name else 2.0  # arbitrary code
                state_dict[obj] = {
                    "x": cx, "y": cy, "z": cz,
                    "growth": growth,
                    "color": color_val
                }
            elif obj.type == self._jug_type:
                (jx, jy, jz), orn = p.getBasePositionAndOrientation(
                    body_id, physicsClientId=self._physics_client_id)
                # is_held we track from constraints
                is_held = 1.0 if (body_id == self._held_obj_id) else 0.0
                # tilt = ...
                # but for simplicity let's store 0.0 for tilt
                tilt = 0.0
                color_val = 1.0 if "red" in obj.name else 2.0
                state_dict[obj] = {
                    "x": jx, "y": jy, "z": jz,
                    "is_held": is_held,
                    "tilt": tilt,
                    "color": color_val
                }

        state = utils.create_state_from_dict(state_dict)
        # Convert the dictionary to a State. Store simulator_state for rendering or debugging.
        joint_positions = self._pybullet_robot.get_joints()
        pyb_state = utils.PyBulletState(state, simulator_state=joint_positions)
        return pyb_state

    def _get_object_ids_for_held_check(self) -> List[int]:
        """
        Return IDs of jugs (since we can only hold jugs).
        """
        return [self._red_jug_id, self._blue_jug_id]

    def _get_expected_finger_normals(self) -> Dict[int, np.ndarray]:
        """
        For the default fetch robot in predicators. We assume a certain orientation
        where the left_finger_id is in +y direction, the right_finger_id is in -y.
        """
        normal_left = np.array([0., 1., 0.], dtype=np.float32)
        normal_right = np.array([0., -1., 0.], dtype=np.float32)
        return {
            self._pybullet_robot.left_finger_id: normal_left,
            self._pybullet_robot.right_finger_id: normal_right,
        }

    # -------------------------------------------------------------------------
    # Setting or updating the environment’s state.

    def _reset_state(self, state: State) -> None:
        """
        Called whenever we do reset() or simulate() on a new state that differs from
        the environment's current state. We must reflect that state in PyBullet.
        """
        super()._reset_state(state)  # Clears constraints, resets robot
        # Remove old bodies if we had them
        for body_id in getattr(self, "_cup_ids", []):
            p.removeBody(body_id, physicsClientId=self._physics_client_id)
        for body_id in getattr(self, "_jug_ids", []):
            p.removeBody(body_id, physicsClientId=self._physics_client_id)

        self._obj_id_to_obj = {}  # track body -> object
        self._cup_growth = {}     # track cup's growth

        # Re-create cups in PyBullet
        self._cup_ids = []
        for cup_obj in [self._red_cup, self._blue_cup]:
            cx = state.get(cup_obj, "x")
            cy = state.get(cup_obj, "y")
            cz = state.get(cup_obj, "z")
            body_id = self._create_cup_urdf(cx, cy, cz, "red" in cup_obj.name)
            self._cup_ids.append(body_id)
            self._obj_id_to_obj[body_id] = cup_obj
            # Store initial growth
            self._cup_growth[cup_obj] = state.get(cup_obj, "growth")

        # Re-create jugs in PyBullet
        self._jug_ids = []
        for jug_obj in [self._red_jug, self._blue_jug]:
            jx = state.get(jug_obj, "x")
            jy = state.get(jug_obj, "y")
            jz = state.get(jug_obj, "z")
            jug_body_id = self._create_jug_urdf(jx, jy, jz, "red" in jug_obj.name)
            self._jug_ids.append(jug_body_id)
            self._obj_id_to_obj[jug_body_id] = jug_obj

        # For convenience, store IDs for usage in _get_object_ids_for_held_check()
        self._red_jug_id = self._jug_ids[0]
        self._blue_jug_id = self._jug_ids[1]

        # Check if either jug is held => forcibly attach constraints. 
        # (Though in our tasks we often start is_held=0.)
        for jug_body_id, jug_obj in zip(self._jug_ids, [self._red_jug, self._blue_jug]):
            if state.get(jug_obj, "is_held") > 0.5:
                # Create a constraint with the robot's gripper
                self._create_grasp_constraint_for_object(jug_body_id)

        # After re-adding objects, compare with the new state
        reconstructed_state = self._get_state()
        # if not reconstructed_state.allclose(state):
        #     logging.warning("Could not reconstruct state exactly!")

    # -------------------------------------------------------------------------
    # Custom pouring logic: if we see the robot has a jug at max tilt over a cup
    # of matching color, we increase cup growth.

    def step(self, action: Action, render_obs: bool = False) -> State:
        """
        We let the parent class handle the robot stepping & constraints. Then,
        we post-process: if the robot is tilting a jug over a matching-color cup,
        we increase the cup's growth in self._cup_growth, and update the environment
        accordingly.
        """
        next_state = super().step(action, render_obs=render_obs)

        # If a jug is in the robot's hand, and tilt is large, check if over a cup
        if self._held_obj_id is not None:
            # Which jug is being held?
            jug_obj = self._obj_id_to_obj[self._held_obj_id]
            # Check tilt
            tilt = next_state.get(self._robot, "tilt")
            if abs(tilt - np.pi/4) < 0.1:
                # We assume that means "pouring"
                # Find if there's a cup below that is color-matched
                jug_color = next_state.get(jug_obj, "color")  # 1.0 for red, 2.0 for blue
                # We can do a simple proximity check in XY
                jug_x = next_state.get(jug_obj, "x")
                jug_y = next_state.get(jug_obj, "y")
                for cup_id, cup_obj in [(cid, obj) for cid, obj in self._obj_id_to_obj.items() if obj.type == self._cup_type]:
                    cx = next_state.get(cup_obj, "x")
                    cy = next_state.get(cup_obj, "y")
                    dist = np.hypot(jug_x - cx, jug_y - cy)
                    if dist < 0.1:  # "over" the cup
                        cup_color = next_state.get(cup_obj, "color")
                        if abs(cup_color - jug_color) < 0.1:
                            # Color match => increase growth
                            new_growth = min(1.0, self._cup_growth[cup_obj] + self.pour_rate)
                            self._cup_growth[cup_obj] = new_growth

        # Because self._cup_growth changed, we replicate that into self._get_state() next time.
        # So let's do one more read => next_state
        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    def _fingers_joint_to_state(self, finger_joint: float) -> float:
        """
        If the parent class uses "reset positions" for the joints, we map them back 
        to [closed_fingers, open_fingers].
        """
        # For a simple approach, pick whichever is closer.
        if abs(finger_joint - self._pybullet_robot.open_fingers) < abs(finger_joint - self._pybullet_robot.closed_fingers):
            return self.open_fingers
        return self.closed_fingers

    # -------------------------------------------------------------------------
    # Predicates

    @staticmethod
    def _CupGrown_holds(state: State, objects: Tuple[Object, ...]) -> bool:
        """A cup is "grown" if 'growth' > growth_threshold."""
        cup, = objects
        try:
            growth = state.get(cup, "growth")
        except:
            breakpoint()
        return growth > PyBulletGrowEnv.growth_threshold

    # -------------------------------------------------------------------------
    # Task Generation

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks, 
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks, 
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator
                    ) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            # Initialize random positions for cups & jugs
            red_cup_x = rng.uniform(self.x_lb, self.x_ub)
            red_cup_y = rng.uniform(self.y_lb, self.y_ub)
            blue_cup_x = rng.uniform(self.x_lb, self.x_ub)
            blue_cup_y = rng.uniform(self.y_lb, self.y_ub)
            red_jug_x = rng.uniform(self.x_lb, self.x_ub)
            red_jug_y = rng.uniform(self.y_lb, self.y_ub)
            blue_jug_x = rng.uniform(self.x_lb, self.x_ub)
            blue_jug_y = rng.uniform(self.y_lb, self.y_ub)

            # Robot at center
            robot_dict = {
                "x": (self.x_lb + self.x_ub)*0.5,
                "y": (self.y_lb + self.y_ub)*0.5,
                "z": self.z_ub - 0.05,
                "fingers": self.open_fingers,
                "tilt": 0.0,
                "wrist": 0.0
            }

            # Cup initial
            red_cup_dict = {
                "x": red_cup_x, "y": red_cup_y, "z": self.z_lb + 0.02,
                "growth": 0.0,  # empty plant
                "color": 1.0    # red
            }
            blue_cup_dict = {
                "x": blue_cup_x, "y": blue_cup_y, "z": self.z_lb + 0.02,
                "growth": 0.0,
                "color": 2.0    # blue
            }

            # Jug initial
            red_jug_dict = {
                "x": red_jug_x, "y": red_jug_y, "z": self.z_lb + 0.02,
                "is_held": 0.0,  # not in hand
                "tilt": 0.0,
                "color": 1.0
            }
            blue_jug_dict = {
                "x": blue_jug_x, "y": blue_jug_y, "z": self.z_lb + 0.02,
                "is_held": 0.0,
                "tilt": 0.0,
                "color": 2.0
            }

            init_dict = {
                self._robot: robot_dict,
                self._red_cup: red_cup_dict,
                self._blue_cup: blue_cup_dict,
                self._red_jug: red_jug_dict,
                self._blue_jug: blue_jug_dict,
            }

            init_state = utils.create_state_from_dict(init_dict)

            # The goal is for both cups to be "CupGrown"
            # i.e. growth > growth_threshold
            goal_atoms = {
                GroundAtom(self._CupGrown, [self._red_cup]),
                GroundAtom(self._CupGrown, [self._blue_cup]),
            }
            tasks.append(EnvironmentTask(init_state, goal_atoms))
        return tasks

    def _create_cup_urdf(self, x: float, y: float, z: float, is_red: bool) -> int:
        global_scale = 0.2
        cup_id = p.loadURDF(
            utils.get_env_asset_path("urdf/cup.urdf"),
            useFixedBase=True,
            globalScaling=global_scale,
            physicsClientId=self._physics_client_id
        )
        cup_orn = p.getQuaternionFromEuler([np.pi, np.pi, 0.0])
        p.resetBasePositionAndOrientation(
            cup_id, (x, y, z),
            cup_orn,
            physicsClientId=self._physics_client_id)

        if is_red:
            p.changeVisualShape(cup_id, -1, rgbaColor=(1, 0, 0, 1))
        else:
            p.changeVisualShape(cup_id, -1, rgbaColor=(0, 0, 1, 1))
        return cup_id

    def _create_jug_urdf(self, x: float, y: float, z: float, is_red: bool) -> int:
        jug_id = p.loadURDF(
            utils.get_env_asset_path("urdf/jug-pixel.urdf"),
            basePosition=(x, y, z),
            globalScaling=0.2,
            useFixedBase=False,
            physicsClientId=self._physics_client_id
        )
        jug_orn = p.getQuaternionFromEuler([np.pi, np.pi, 0.0])
        p.resetBasePositionAndOrientation(
            jug_id, (x, y, z),
            jug_orn,
            physicsClientId=self._physics_client_id)
        if is_red:
            p.changeVisualShape(jug_id, -1, rgbaColor=(1, 0, 0, 1))
        else:
            p.changeVisualShape(jug_id, -1, rgbaColor=(0, 0, 1, 1))
        return jug_id