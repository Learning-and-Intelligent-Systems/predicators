"""A PyBullet environment with blocks, a magic trash bin, and a switch.

When the switch is ON and a block is inside the bin, the block is teleported
to an out-of-view position (vanished). The goal is to make certain blocks
vanish (not be on the table).

python predicators/main.py --approach oracle --env pybullet_magic_bin \
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


class PyBulletMagicBinEnv(PyBulletEnv):
    """A PyBullet environment with blocks, a magic bin, and a switch.

    - Robot can pick and place blocks
    - Switch controls whether the magic bin is active
    - When switch is ON and a block is in the bin, the block vanishes
    - Goal: make specific blocks vanish
    """

    # Number of blocks
    num_blocks: ClassVar[int] = 3

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

    # Block dimensions
    block_size: ClassVar[float] = 0.05

    # Bin parameters
    bin_scale: ClassVar[float] = 0.15  # Scale down the bucket
    bin_radius: ClassVar[
        float] = 0.08  # Approximate radius for collision check
    bin_height: ClassVar[
        float] = 0.16  # Approximate height of bucket after scaling

    # Camera parameters
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # Types
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])
    _block_type = Type("block", ["x", "y", "z", "is_held", "vanished"])
    _switch_type = Type("switch", ["x", "y", "z", "rot", "is_on"],
                        sim_features=["id", "joint_id", "joint_scale"])
    _bin_type = Type("bin", ["x", "y", "z", "rot"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # Objects
        self._robot = Object("robot", self._robot_type)
        self._blocks: List[Object] = [
            Object(f"block{i}", self._block_type)
            for i in range(self.num_blocks)
        ]
        self._switch = Object("switch", self._switch_type)
        self._bin = Object("bin", self._bin_type)

        super().__init__(use_gui, **kwargs)

        # Predicates
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._block_type],
                                  self._Holding_holds)
        self._OnTable = Predicate("OnTable", [self._block_type],
                                  self._OnTable_holds)
        self._InBin = Predicate("InBin", [self._block_type, self._bin_type],
                                self._InBin_holds)
        self._SwitchOn = Predicate("SwitchOn", [self._switch_type],
                                   self._SwitchOn_holds)
        self._Vanished = Predicate("Vanished", [self._block_type],
                                   self._Vanished_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_magic_bin"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HandEmpty,
            self._Holding,
            self._OnTable,
            self._InBin,
            self._SwitchOn,
            self._Vanished,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Vanished}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type,
            self._block_type,
            self._switch_type,
            self._bin_type,
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

        # Create blocks
        block_ids = []
        for i in range(cls.num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            half_extents = (cls.block_size / 2, cls.block_size / 2,
                            cls.block_size / 2)
            block_id = create_pybullet_block(
                color=color,
                half_extents=half_extents,
                mass=cls._obj_mass,
                friction=cls._obj_friction,
                physics_client_id=physics_client_id,
            )
            block_ids.append(block_id)
        bodies["block_ids"] = block_ids

        # Create the switch
        switch_id = create_object(
            asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
            physics_client_id=physics_client_id,
            scale=1,
            use_fixed_base=True,
        )
        bodies["switch_id"] = switch_id

        # Create the magic bin (bucket) with concave mesh collision
        # Using URDF_USE_SELF_COLLISION_INCLUDE_PARENT flag to help with
        # concave collision detection for static objects
        bin_id = p.loadURDF(
            utils.get_env_asset_path(
                "urdf/partnet_mobility/bucket/100470/bucket.urdf"),
            useFixedBase=True,
            globalScaling=cls.bin_scale,
            flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT,
            physicsClientId=physics_client_id,
        )
        bodies["bin_id"] = bin_id

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
        # Store block IDs
        block_ids = pybullet_bodies["block_ids"]
        for block, block_id in zip(self._blocks, block_ids):
            block.id = block_id

        # Store switch ID and joint info
        self._switch.id = pybullet_bodies["switch_id"]
        self._switch.joint_id = self._get_joint_id(self._switch.id, "joint_0",
                                                   self._physics_client_id)
        self._switch.joint_scale = 0.1
        cap_switch_joint_travel(self._switch.id, self._switch.joint_id,
                                self._switch.joint_scale,
                                self._physics_client_id)

        # Store bin ID
        self._bin.id = pybullet_bodies["bin_id"]

    # -------------------------------------------------------------------------
    # State Management
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return IDs of objects that can be held (blocks)."""
        return [block.id for block in self._blocks]

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._switch_type and feature == "is_on":
            return float(self._is_switch_on())
        if obj.type == self._block_type and feature == "vanished":
            # Check if block is at out-of-view position
            pos, _ = p.getBasePositionAndOrientation(
                obj.id, physicsClientId=self._physics_client_id)
            return float(pos[0] > 5.0)  # Out of view if x > 5
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        """Reset environment state from a State object."""
        # Set switch state
        switch_on = state.get(self._switch, "is_on") > 0.5
        self._set_switch_state(switch_on)

        # Set block positions (including vanished blocks)
        for block in self._blocks:
            vanished = state.get(block, "vanished") > 0.5
            if vanished:
                # Move to out-of-view position
                idx = self._blocks.index(block)
                oov_x, oov_y = self._out_of_view_xy
                p.resetBasePositionAndOrientation(
                    block.id, [oov_x, oov_y, idx * self.block_size],
                    self._default_orn,
                    physicsClientId=self._physics_client_id)

    def _domain_specific_step(self) -> None:
        """If switch is on and block is in bin, vanish it."""
        if self._is_switch_on():
            bin_pos, _ = p.getBasePositionAndOrientation(
                self._bin.id, physicsClientId=self._physics_client_id)

            for block in self._blocks:
                # Skip already vanished blocks
                block_pos, _ = p.getBasePositionAndOrientation(
                    block.id, physicsClientId=self._physics_client_id)
                if block_pos[0] > 5.0:  # Already vanished
                    continue

                # Skip held blocks
                if block.id == self._held_obj_id:
                    continue

                # Check if block is in bin (horizontal distance check)
                dx = block_pos[0] - bin_pos[0]
                dy = block_pos[1] - bin_pos[1]
                dist = np.sqrt(dx * dx + dy * dy)

                # Check if block is above bin bottom and within radius
                if dist < self.bin_radius and block_pos[2] < bin_pos[2] + 0.15:
                    # Teleport block to out-of-view position
                    idx = self._blocks.index(block)
                    oov_x, oov_y = self._out_of_view_xy
                    p.resetBasePositionAndOrientation(
                        block.id, [oov_x, oov_y, idx * self.block_size],
                        self._default_orn,
                        physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Switch helpers
    def _is_switch_on(self) -> bool:
        """Check if the switch is in the ON position."""
        joint_state = p.getJointState(self._switch.id,
                                      self._switch.joint_id,
                                      physicsClientId=self._physics_client_id
                                      )[0] / self._switch.joint_scale
        joint_min = p.getJointInfo(self._switch.id,
                                   self._switch.joint_id,
                                   physicsClientId=self._physics_client_id)[8]
        joint_max = p.getJointInfo(self._switch.id,
                                   self._switch.joint_id,
                                   physicsClientId=self._physics_client_id)[9]
        joint_state = np.clip(
            (joint_state - joint_min) / (joint_max - joint_min), 0, 1)
        return bool(joint_state > 0.5)

    def _set_switch_state(self, power_on: bool) -> None:
        """Programmatically set the switch on/off."""
        joint_id = self._switch.joint_id
        if joint_id < 0:
            return
        info = p.getJointInfo(self._switch.id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = j_max if power_on else j_min
        p.resetJointState(
            self._switch.id,
            joint_id,
            target_val * self._switch.joint_scale,
            physicsClientId=self._physics_client_id,
        )

    # -------------------------------------------------------------------------
    # Predicates
    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, block = objects
        return (state.get(robot, "fingers") <= 0.02
                and state.get(block, "is_held") > 0.5)

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        # Not vanished and on the table surface
        if state.get(block, "vanished") > 0.5:
            return False
        if state.get(block, "is_held") > 0.5:
            return False
        z = state.get(block, "z")
        return abs(z - (self.table_height + self.block_size / 2)) < 0.05

    def _InBin_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, bin_obj = objects
        # Not vanished, not held, and within bin radius
        if state.get(block, "vanished") > 0.5:
            return False
        if state.get(block, "is_held") > 0.5:
            return False
        block_x = state.get(block, "x")
        block_y = state.get(block, "y")
        bin_x = state.get(bin_obj, "x")
        bin_y = state.get(bin_obj, "y")
        dx = block_x - bin_x
        dy = block_y - bin_y
        dist = np.sqrt(dx * dx + dy * dy)
        return dist < self.bin_radius

    @staticmethod
    def _SwitchOn_holds(state: State, objects: Sequence[Object]) -> bool:
        switch, = objects
        return state.get(switch, "is_on") > 0.5

    @staticmethod
    def _Vanished_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "vanished") > 0.5

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

            # Switch position (left side)
            switch_x = self.x_lb + 3 * self.init_padding
            switch_dict = {
                "x": switch_x,
                "y": 1.3,
                "z": self.table_height,
                "rot": np.pi / 2,
                "is_on": 0.0,  # Start with switch off
            }

            # Bin position (right of switch)
            # Bin origin is near center, so raise it so bottom sits on table
            bin_x = switch_x + 0.25
            bin_dict = {
                "x": bin_x,
                "y": 1.35,
                "z":
                self.table_height + 0.08,  # Offset to place bottom on table
                "rot": 0.0,
            }

            init_dict: Dict[Object, Dict[str, float]] = {
                self._robot: robot_dict,
                self._switch: switch_dict,
                self._bin: bin_dict,
            }

            # Place blocks on table
            block_start_x = bin_x + 0.2
            for i, block in enumerate(self._blocks):
                block_dict = {
                    "x": block_start_x + i * 0.1,
                    "y": 1.35,
                    "z": self.table_height + self.block_size / 2,
                    "is_held": 0.0,
                    "vanished": 0.0,
                }
                init_dict[block] = block_dict

            init_state = utils.create_state_from_dict(init_dict)

            # Goal: at least one random block should vanish
            num_to_vanish = rng.integers(1, min(3, self.num_blocks) + 1)
            blocks_to_vanish = rng.choice(self._blocks,
                                          size=num_to_vanish,
                                          replace=False)
            goal_atoms = {
                GroundAtom(self._Vanished, [block])
                for block in blocks_to_vanish
            }

            tasks.append(EnvironmentTask(init_state, goal_atoms))

        return self._add_pybullet_state_to_tasks(tasks)


if __name__ == "__main__":
    # Run a simple simulation to test the environment.
    import time

    CFG.seed = 0
    CFG.env = "pybullet_magic_bin"
    CFG.num_train_tasks = 1
    env = PyBulletMagicBinEnv(use_gui=True)
    task = env._generate_train_tasks()[0]  # pylint: disable=protected-access
    env._set_state(task.init)  # pylint: disable=protected-access

    print("PyBullet Magic Bin Environment Test")
    print("Blocks should vanish when in bin with switch ON.")
    print("Press Ctrl+C to exit.")

    while True:
        _joints = env._pybullet_robot.initial_joint_positions  # pylint: disable=protected-access
        _act = Action(np.array(_joints))
        env.step(_act)
        time.sleep(0.01)
