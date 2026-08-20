"""A PyBullet environment with switches controlling barriers.

Each switch controls one barrier independently. When a switch is ON, its
corresponding barrier rises. When OFF, the barrier lowers.

python predicators/main.py --approach oracle --env pybullet_barrier \
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
    create_object, create_pybullet_block
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletBarrierEnv(PyBulletEnv):
    """A PyBullet environment with switches controlling barriers.

    - Each switch controls one barrier
    - Switch ON -> barrier rises, Switch OFF -> barrier lowers
    - Animation happens gradually over simulation steps
    """

    # Number of switch/barrier pairs
    num_barriers: ClassVar[int] = 2

    # Barrier animation parameters
    barrier_speed: ClassVar[float] = 0.005  # units per step
    barrier_raised_height: ClassVar[float] = 0.15  # fully raised height
    barrier_tolerance: ClassVar[float] = 0.01  # tolerance for checking up/down

    # Barrier dimensions (thin vertical wall)
    barrier_half_extents: ClassVar[Tuple[float, float,
                                         float]] = (0.02, 0.1, 0.05)

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

    # Switch dimensions
    switch_width: ClassVar[float] = 0.06
    switch_height: ClassVar[float] = 0.08

    # Camera parameters
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # Barrier color
    BARRIER_COLOR: ClassVar[Tuple[float, float, float,
                                  float]] = (0.6, 0.3, 0.1, 1.0)  # brown

    # Types
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])
    _switch_type = Type("switch", ["x", "y", "z", "rot", "is_on"],
                        sim_features=["id", "joint_id", "joint_scale"])
    _barrier_type = Type("barrier", ["x", "y", "rot", "height"],
                         sim_features=["id", "base_z"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # Objects
        self._robot = Object("robot", self._robot_type)
        self._switches: List[Object] = [
            Object(f"switch{i}", self._switch_type)
            for i in range(self.num_barriers)
        ]
        self._barriers: List[Object] = [
            Object(f"barrier{i}", self._barrier_type)
            for i in range(self.num_barriers)
        ]

        super().__init__(use_gui, **kwargs)

        # Predicates
        self._SwitchOn = Predicate("SwitchOn", [self._switch_type],
                                   self._SwitchOn_holds)
        self._BarrierUp = Predicate("BarrierUp", [self._barrier_type],
                                    self._BarrierUp_holds)
        self._BarrierDown = Predicate("BarrierDown", [self._barrier_type],
                                      self._BarrierDown_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_barrier"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._SwitchOn,
            self._BarrierUp,
            self._BarrierDown,
            self._HandEmpty,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._BarrierUp, self._BarrierDown}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type,
            self._switch_type,
            self._barrier_type,
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

        # Create switches
        for i in range(cls.num_barriers):
            switch_id = create_object(
                asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
                physics_client_id=physics_client_id,
                scale=1,
                use_fixed_base=True,
            )
            bodies[f"switch{i}_id"] = switch_id

        # Create barriers (static blocks)
        for i in range(cls.num_barriers):
            barrier_id = create_pybullet_block(
                color=cls.BARRIER_COLOR,
                half_extents=cls.barrier_half_extents,
                mass=0.0,  # Fixed barrier
                friction=0.5,
                physics_client_id=physics_client_id,
            )
            bodies[f"barrier{i}_id"] = barrier_id

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _get_joint_id(obj_id: int, joint_name: str,
                      physics_client_id: int) -> int:
        """Get the joint ID for a joint with a given name."""
        num_joints = p.getNumJoints(obj_id, physicsClientId=physics_client_id)
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(obj_id,
                                        joint_index,
                                        physicsClientId=physics_client_id)
            if joint_info[1].decode('utf-8') == joint_name:
                return joint_index
        return -1

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to PyBullet IDs for environment assets."""
        for i, switch in enumerate(self._switches):
            switch.id = pybullet_bodies[f"switch{i}_id"]
            switch.joint_id = self._get_joint_id(switch.id, "joint_0",
                                                 self._physics_client_id)
            switch.joint_scale = 0.1
            cap_switch_joint_travel(switch.id, switch.joint_id,
                                    switch.joint_scale,
                                    self._physics_client_id)

        for i, barrier in enumerate(self._barriers):
            barrier.id = pybullet_bodies[f"barrier{i}_id"]
            # Store the base_z position (when lowered, barrier hides in table)
            # When height=0: top is below table surface
            # When height=raised_height: bottom is at table surface
            barrier.base_z = (self.z_lb - self.barrier_raised_height / 2 +
                              self.barrier_half_extents[2])

    # -------------------------------------------------------------------------
    # State Management
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return IDs of objects that can be held (none in this env)."""
        return []

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._switch_type and feature == "is_on":
            return float(self._is_switch_on(obj))
        if obj.type == self._barrier_type and feature == "height":
            # Get current z position and subtract base_z
            pos, _ = p.getBasePositionAndOrientation(
                obj.id, physicsClientId=self._physics_client_id)
            current_z = pos[2]
            return current_z - obj.base_z
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        """Reset environment state from a State object."""
        # Set switch states and positions
        for switch in self._switches:
            switch_on = state.get(switch, "is_on") > 0.5
            self._set_switch_state(switch, switch_on)

            # Set switch position
            switch_x = state.get(switch, "x")
            switch_y = state.get(switch, "y")
            switch_z = state.get(switch, "z")
            switch_rot = state.get(switch, "rot")
            switch_orn = p.getQuaternionFromEuler([0, 0, switch_rot])
            p.resetBasePositionAndOrientation(
                switch.id, (switch_x, switch_y, switch_z),
                switch_orn,
                physicsClientId=self._physics_client_id)

        # Set barrier positions at correct heights
        # Compute base_z so barriers hide in table when lowered
        base_z = (self.z_lb - self.barrier_raised_height +
                  self.barrier_half_extents[2])

        for barrier in self._barriers:
            barrier_x = state.get(barrier, "x")
            barrier_y = state.get(barrier, "y")
            barrier_rot = state.get(barrier, "rot")
            barrier_height = state.get(barrier, "height")
            barrier_orn = p.getQuaternionFromEuler([0, 0, barrier_rot])

            # Store base_z in sim_feature
            barrier.base_z = base_z

            # Position barrier at correct height
            p.resetBasePositionAndOrientation(
                barrier.id, (barrier_x, barrier_y, base_z + barrier_height),
                barrier_orn,
                physicsClientId=self._physics_client_id)

    def step(  # pylint: disable=redefined-outer-name
            self,
            action: Action,
            render_obs: bool = False) -> State:
        """Process a single action step and animate barriers."""
        # Execute the action
        super().step(action, render_obs=render_obs)

        # Animate barriers based on switch states
        for switch, barrier in zip(self._switches, self._barriers):
            switch_on = self._is_switch_on(switch)

            # Get current barrier position
            pos, orn = p.getBasePositionAndOrientation(
                barrier.id, physicsClientId=self._physics_client_id)
            current_z = pos[2]
            current_height = current_z - barrier.base_z

            # Determine target height
            if switch_on:
                target_height = self.barrier_raised_height
            else:
                target_height = 0.0

            # Gradual movement
            height_diff = target_height - current_height
            if abs(height_diff) > self.barrier_speed:
                if height_diff > 0:
                    new_height = current_height + self.barrier_speed
                else:
                    new_height = current_height - self.barrier_speed
            else:
                new_height = target_height

            # Update barrier position
            new_z = barrier.base_z + new_height
            p.resetBasePositionAndOrientation(
                barrier.id, (pos[0], pos[1], new_z),
                orn,
                physicsClientId=self._physics_client_id)

        # Get updated state
        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

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
    # Predicates
    @staticmethod
    def _SwitchOn_holds(state: State, objects: Sequence[Object]) -> bool:
        switch, = objects
        return state.get(switch, "is_on") > 0.5

    def _BarrierUp_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        barrier, = objects
        height = state.get(barrier, "height")
        return height >= self.barrier_raised_height - self.barrier_tolerance

    def _BarrierDown_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        barrier, = objects
        height = state.get(barrier, "height")
        return height <= self.barrier_tolerance

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

            init_dict: Dict[Object, Dict[str, float]] = {
                self._robot: robot_dict
            }

            # Position switches and barriers
            switch_spacing = 0.25
            start_x = self.x_lb + 3 * self.init_padding

            # Random initial switch states and barrier heights
            init_switch_states = [
                bool(rng.integers(0, 2)) for _ in range(self.num_barriers)
            ]

            for i, switch in enumerate(self._switches):
                switch_x = start_x + i * switch_spacing
                switch_dict = {
                    "x": switch_x,
                    "y": 1.3,
                    "z": self.table_height,
                    "rot": np.pi / 2,
                    "is_on": float(init_switch_states[i]),
                }
                init_dict[switch] = switch_dict

            for i, barrier in enumerate(self._barriers):
                barrier_x = start_x + i * switch_spacing
                # Initial height matches switch state
                init_height = (self.barrier_raised_height
                               if init_switch_states[i] else 0.0)
                barrier_dict = {
                    "x": barrier_x,
                    "y": 1.5,  # Behind the switches
                    "rot": 0.0,
                    "height": init_height,
                }
                init_dict[barrier] = barrier_dict

            init_state = utils.create_state_from_dict(init_dict)

            # Create goal: random target barrier states
            # Ensure at least one barrier needs to change
            goal_atoms: Set[GroundAtom] = set()

            # Randomly select target states for barriers
            target_states = [
                bool(rng.integers(0, 2)) for _ in range(self.num_barriers)
            ]

            # Ensure at least one change is required
            while target_states == init_switch_states:
                target_states = [
                    bool(rng.integers(0, 2)) for _ in range(self.num_barriers)
                ]

            for i, barrier in enumerate(self._barriers):
                if target_states[i]:
                    goal_atoms.add(GroundAtom(self._BarrierUp, [barrier]))
                else:
                    goal_atoms.add(GroundAtom(self._BarrierDown, [barrier]))

            tasks.append(EnvironmentTask(init_state, goal_atoms))

        return self._add_pybullet_state_to_tasks(tasks)


if __name__ == "__main__":
    # Run a simple simulation to test the environment.
    import time

    CFG.seed = 0
    CFG.env = "pybullet_barrier"
    CFG.num_train_tasks = 1
    env = PyBulletBarrierEnv(use_gui=True)
    task = env._generate_train_tasks()[0]  # pylint: disable=protected-access
    env._set_state(task.init)  # pylint: disable=protected-access

    print("PyBullet Barrier Environment Test")
    print("Barriers should animate when switches are toggled.")
    print("Press Ctrl+C to exit.")

    while True:
        _joints = env._pybullet_robot.initial_joint_positions  # pylint: disable=protected-access
        action = Action(np.array(_joints))
        env.step(action)
        time.sleep(0.01)
