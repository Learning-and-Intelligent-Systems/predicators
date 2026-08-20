"""A PyBullet environment with two switches and one light bulb.

The power switch controls whether the light is on/off.
The color switch cycles through colors (red, green, blue)
each time it's toggled.
The goal is to have the light display a specific target color.

python predicators/main.py --approach oracle --env pybullet_switch \
--seed 0 --num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1 --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900
"""

from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import cap_switch_joint_travel, \
    create_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletSwitchEnv(PyBulletEnv):
    """A PyBullet environment with two switches controlling a light bulb.

    - Power switch: toggles the light on/off
    - Color switch: cycles through red, green, blue when toggled (OFF->ON)
    - Goal: achieve a specific target color on the light
    """

    # Table / workspace config
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = tuple(
        p.getQuaternionFromEuler([0., 0., np.pi / 2]))

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    init_padding: ClassVar[float] = 0.05

    # Robot config
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Quaternion] = tuple(
        p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2]))
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2
    max_angular_vel: ClassVar[float] = np.pi / 4

    # Switch/light dimensions
    snap_width: ClassVar[float] = 0.05
    snap_height: ClassVar[float] = 0.05
    switch_width: ClassVar[float] = 0.06
    switch_height: ClassVar[float] = 0.08

    # Camera parameters
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # Light colors
    LIGHT_COLORS: ClassVar[List[Tuple[float, float, float, float]]] = [
        (1.0, 0.0, 0.0, 1.0),  # Red (index 0)
        (0.0, 1.0, 0.0, 1.0),  # Green (index 1)
        (0.0, 0.0, 1.0, 1.0),  # Blue (index 2)
    ]
    LIGHT_OFF_COLOR: ClassVar[Tuple[float, float, float,
                                    float]] = (0.8, 0.8, 0.8, 1.0)

    # Types
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])
    _power_switch_type = Type("power_switch", ["x", "y", "z", "rot", "is_on"],
                              sim_features=["id", "joint_id", "joint_scale"])
    _color_switch_type = Type(
        "color_switch", ["x", "y", "z", "rot", "is_on"],
        sim_features=["id", "joint_id", "joint_scale", "color_count"])
    _light_type = Type("light", ["x", "y", "z", "rot", "is_on", "color_index"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # Objects
        self._robot = Object("robot", self._robot_type)
        self._power_switch = Object("power_switch", self._power_switch_type)
        self._color_switch = Object("color_switch", self._color_switch_type)
        self._light = Object("light", self._light_type)

        super().__init__(use_gui, **kwargs)

        # Track previous switch states for edge detection
        self._prev_color_switch_on: bool = False
        self._pre_step_color_count: int = 0

        # Predicates
        self._PowerOn = Predicate("PowerOn", [self._power_switch_type],
                                  self._PowerOn_holds)
        self._LightOn = Predicate("LightOn", [self._light_type],
                                  self._LightOn_holds)
        self._LightIsRed = Predicate("LightIsRed", [self._light_type],
                                     self._LightIsRed_holds)
        self._LightIsGreen = Predicate("LightIsGreen", [self._light_type],
                                       self._LightIsGreen_holds)
        self._LightIsBlue = Predicate("LightIsBlue", [self._light_type],
                                      self._LightIsBlue_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_switch"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._PowerOn,
            self._LightOn,
            self._LightIsRed,
            self._LightIsGreen,
            self._LightIsBlue,
            self._HandEmpty,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._LightIsRed, self._LightIsGreen, self._LightIsBlue}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type,
            self._power_switch_type,
            self._color_switch_type,
            self._light_type,
        }

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Add table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=1,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id

        # Create the power switch
        power_switch_id = create_object(
            asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
            physics_client_id=physics_client_id,
            scale=1,
            use_fixed_base=True,
        )
        bodies["power_switch_id"] = power_switch_id

        # Create the color switch (same URDF, different instance)
        color_switch_id = create_object(
            asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
            physics_client_id=physics_client_id,
            scale=1,
            use_fixed_base=True,
        )
        bodies["color_switch_id"] = color_switch_id

        # Create the light
        light_id = create_object(
            asset_path="urdf/bulb_box_snap.urdf",
            physics_client_id=physics_client_id,
            scale=1,
            use_fixed_base=True,
        )
        bodies["light_id"] = light_id

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _get_joint_id(obj_id: int, joint_name: str) -> int:
        """Get the joint ID for a joint with a given name."""
        num_joints = p.getNumJoints(obj_id)
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(obj_id, joint_index)
            if joint_info[1].decode('utf-8') == joint_name:
                return joint_index
        return -1

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to PyBullet IDs for environment assets."""
        self._power_switch.id = pybullet_bodies["power_switch_id"]
        self._power_switch.joint_id = self._get_joint_id(
            self._power_switch.id, "joint_0")
        self._power_switch.joint_scale = 0.1
        cap_switch_joint_travel(self._power_switch.id,
                                self._power_switch.joint_id,
                                self._power_switch.joint_scale,
                                self._physics_client_id)

        self._color_switch.id = pybullet_bodies["color_switch_id"]
        self._color_switch.joint_id = self._get_joint_id(
            self._color_switch.id, "joint_0")
        self._color_switch.joint_scale = 0.1
        cap_switch_joint_travel(self._color_switch.id,
                                self._color_switch.joint_id,
                                self._color_switch.joint_scale,
                                self._physics_client_id)
        self._color_switch.color_count = 0  # Will be set in reset

        self._light.id = pybullet_bodies["light_id"]

    # -------------------------------------------------------------------------
    # State Management
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return IDs of objects that can be held (none in this env)."""
        return []

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._light_type and feature == "is_on":
            return float(self._is_power_switch_on())
        if obj.type == self._light_type and feature == "color_index":
            color_count = self._color_switch.color_count
            return float(int(color_count) % len(self.LIGHT_COLORS))
        if obj.type == self._power_switch_type and feature == "is_on":
            return float(self._is_switch_on(self._power_switch))
        if obj.type == self._color_switch_type and feature == "is_on":
            return float(self._is_switch_on(self._color_switch))
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        """Set switch positions, tracking vars, color count, and light
        visual."""
        power_on = state.get(self._power_switch, "is_on") > 0.5
        self._set_switch_state(self._power_switch, power_on)

        color_switch_on = state.get(self._color_switch, "is_on") > 0.5
        self._set_switch_state(self._color_switch, color_switch_on)

        self._prev_color_switch_on = color_switch_on

        color_index = int(state.get(self._light, "color_index"))
        self._color_switch.color_count = color_index

        self._update_light_visual(power_on, color_index)

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Save pre-step color count before kinematics."""
        self._pre_step_color_count = self._color_switch.color_count
        return super().step(action, render_obs=render_obs)

    def _domain_specific_step(self) -> None:
        # Detect color switch toggle (OFF -> ON transition)
        curr_color_switch_on = self._is_switch_on(self._color_switch)
        if not self._prev_color_switch_on and curr_color_switch_on:
            # Rising edge detected - increment color count
            self._color_switch.color_count = self._pre_step_color_count + 1

        self._prev_color_switch_on = curr_color_switch_on

        # Compute color index
        color_index = int(self._color_switch.color_count) % len(
            self.LIGHT_COLORS)

        # Check if power is on
        power_on = self._is_power_switch_on()

        # Update light visual
        self._update_light_visual(power_on, color_index)

    # -------------------------------------------------------------------------
    # Switch helpers
    def _is_switch_on(self, switch_obj: Object) -> bool:
        """Check if a switch is in the ON position."""
        joint_state = p.getJointState(switch_obj.id,
                                      switch_obj.joint_id,
                                      physicsClientId=self._physics_client_id
                                      )[0] / switch_obj.joint_scale
        joint_min = p.getJointInfo(switch_obj.id,
                                   switch_obj.joint_id,
                                   physicsClientId=self._physics_client_id)[8]
        joint_max = p.getJointInfo(switch_obj.id,
                                   switch_obj.joint_id,
                                   physicsClientId=self._physics_client_id)[9]
        joint_state = np.clip(
            (joint_state - joint_min) / (joint_max - joint_min), 0, 1)
        return bool(joint_state > 0.5)

    def _is_power_switch_on(self) -> bool:
        """Check if the power switch is on."""
        return self._is_switch_on(self._power_switch)

    def _set_switch_state(self, switch_obj: Object, power_on: bool) -> None:
        """Programmatically set a switch on/off."""
        joint_id = switch_obj.joint_id
        if joint_id < 0:
            return
        info = p.getJointInfo(switch_obj.id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = j_max if power_on else j_min
        p.resetJointState(
            switch_obj.id,
            joint_id,
            target_val * switch_obj.joint_scale,
            physicsClientId=self._physics_client_id,
        )

    # -------------------------------------------------------------------------
    # Light helpers
    def _update_light_visual(self, power_on: bool, color_index: int) -> None:
        """Update the light's visual appearance."""
        if self._light.id is None:
            return
        if power_on:
            color = self.LIGHT_COLORS[color_index]
        else:
            color = self.LIGHT_OFF_COLOR
        p.changeVisualShape(self._light.id,
                            3,
                            rgbaColor=color,
                            physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Predicates
    @staticmethod
    def _PowerOn_holds(state: State, objects: Sequence[Object]) -> bool:
        power_switch, = objects
        return state.get(power_switch, "is_on") > 0.5

    @staticmethod
    def _LightOn_holds(state: State, objects: Sequence[Object]) -> bool:
        light, = objects
        return state.get(light, "is_on") > 0.5

    @staticmethod
    def _LightIsRed_holds(state: State, objects: Sequence[Object]) -> bool:
        light, = objects
        return (state.get(light, "is_on") > 0.5
                and int(state.get(light, "color_index")) == 0)

    @staticmethod
    def _LightIsGreen_holds(state: State, objects: Sequence[Object]) -> bool:
        light, = objects
        return (state.get(light, "is_on") > 0.5
                and int(state.get(light, "color_index")) == 1)

    @staticmethod
    def _LightIsBlue_holds(state: State, objects: Sequence[Object]) -> bool:
        light, = objects
        return (state.get(light, "is_on") > 0.5
                and int(state.get(light, "color_index")) == 2)

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    # -------------------------------------------------------------------------
    # Task Generation
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int,
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            # Robot at center
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "roll": self.robot_init_roll,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }

            # Power switch position (left side)
            power_switch_x = self.x_lb + 3 * self.init_padding
            power_switch_dict = {
                "x": power_switch_x,
                "y": 1.3,
                "z": self.table_height,
                "rot": np.pi / 2,
                "is_on": 0.0,  # Start with power off
            }

            # Color switch position (right of power switch)
            color_switch_x = power_switch_x + 0.2
            init_color_index = int(rng.integers(0, 3))  # Random initial color
            color_switch_dict = {
                "x": color_switch_x,
                "y": 1.3,
                "z": self.table_height,
                "rot": np.pi / 2,
                "is_on": 0.0,  # Start with switch off
            }

            # Light position (right of color switch)
            light_x = color_switch_x + 0.2
            light_dict = {
                "x": light_x,
                "y": 1.3,
                "z": self.z_lb + self.snap_height / 2,
                "rot": -np.pi / 2,
                "is_on": 0.0,  # Light off (power switch is off)
                "color_index": float(init_color_index),
            }

            init_dict = {
                self._robot: robot_dict,
                self._power_switch: power_switch_dict,
                self._color_switch: color_switch_dict,
                self._light: light_dict,
            }
            init_state = utils.create_state_from_dict(init_dict)

            # Random target color
            color_predicates = [
                self._LightIsRed, self._LightIsGreen, self._LightIsBlue
            ]
            target_idx = int(rng.integers(0, len(color_predicates)))
            target_pred = color_predicates[target_idx]
            goal_atoms = {GroundAtom(target_pred, [self._light])}

            tasks.append(EnvironmentTask(init_state, goal_atoms))

        return self._add_pybullet_state_to_tasks(tasks)


if __name__ == "__main__":
    # Run a simple simulation to test the environment.
    import time

    CFG.seed = 0
    CFG.env = "pybullet_switch"
    CFG.num_train_tasks = 1
    env = PyBulletSwitchEnv(use_gui=True)
    task = env._generate_train_tasks()[0]  # pylint: disable=protected-access
    env._set_state(task.init)  # pylint: disable=protected-access

    while True:
        _joints = env._pybullet_robot.initial_joint_positions  # pylint: disable=protected-access
        _act = Action(np.array(_joints))
        env.step(_act)
        time.sleep(0.01)
