import logging
from itertools import product
from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block, \
    create_pybullet_sphere
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletFanEnv(PyBulletEnv):
    """A PyBullet environment where a ball is blown around by fans in a
    maze."""

    # -------------------------------------------------------------------------
    # Table / workspace / Maze
    # -------------------------------------------------------------------------
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2.0)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    init_padding: float = 0.05

    # -------------------------------------------------------------------------
    # Robot config
    # -------------------------------------------------------------------------
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.62, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])
    robot_init_tilt: ClassVar[float] = np.pi / 2.0
    robot_init_wrist: ClassVar[float] = -np.pi / 2.0

    # -------------------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------------------
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.25, 0.42)

    # -------------------------------------------------------------------------
    # Fan configuration
    # -------------------------------------------------------------------------
    num_front_fans: ClassVar[int] = 5
    num_back_fans: ClassVar[int] = 5
    num_left_fans: ClassVar[int] = 2
    num_right_fans: ClassVar[int] = 2
    num_pos_x, num_pos_y = 9, 4
    pos_gap = 0.08
    
    num_walls: ClassVar[int] = 4
    wall_x_len, wall_y_len, wall_z_len = 0.05, 0.04, 0.04

    fan_scale: ClassVar[float] = 0.08
    fan_x_len: ClassVar[float] = 0.2 * fan_scale  # length of the fan blades
    fan_z_len: ClassVar[float] = 1.5 * fan_scale  # height of the fan base
    fan_spin_velocity: ClassVar[float] = 100.0  # velocity for joint_0
    wind_force_magnitude: ClassVar[float] = 0.8  # force on the ball

    switch_joint_scale: ClassVar[float] = 0.1
    switch_on_threshold: ClassVar[float] = 0.5  # fraction of the joint range
    switch_x_len: ClassVar[float] = 0.10  # length of the switch
    switch_height: ClassVar[float] = 0.08

    # -------------------------------------------------------------------------
    # Types
    # -------------------------------------------------------------------------
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _fan_type = Type(
        "fan",
        [
            "x",  # fan base x
            "y",  # fan base y
            "z",  # fan base z
            "rot",  # base orientation (Z euler)
            "side",  # 0=left,1=right,2=back,3=front
            "is_on",  # whether the controlling switch is on
        ],
        sim_features=["id", "joint_id", "side_idx"])
    # New separate switch type:
    _switch_type = Type(
        "switch",
        [
            "x",
            "y",
            "z",
            "rot",  # switch orientation
            "side",  # matches fan side
            "is_on",  # is this switch on
        ],
        sim_features=["id", "joint_id", "side_idx"])
    _wall_type = Type("wall", ["x", "y", "z", "rot"])
    _ball_type = Type("ball", ["x", "y", "z"])
    _target_type = Type("target", ["x", "y", "z", "rot", "is_hit"])
    _position_type = Type("position", ["xx", "yy"])

    # -------------------------------------------------------------------------
    # Environment initialization
    # -------------------------------------------------------------------------
    def __init__(self, use_gui: bool = True) -> None:
        self._robot = Object("robot", self._robot_type)

        # Fans
        self._fans: List[Object] = []

        # Switches: now each is a distinct object of _switch_type
        self._switches: List[Object] = []
        self._switch_sides = ["left", "right", "back", "front"]
        for side_str in self._switch_sides:
            # Create a switch object using the new _switch_type
            switch_obj = Object(f"switch_{side_str}", self._switch_type)
            self._switches.append(switch_obj)

        # Maze walls
        self._walls = [Object(f"wall{i}", self._wall_type) for i in range(
                                                            self.num_walls)]
        self._positions = [Object(f"position{i}_{j}", self._position_type)
                                        for i in range(self.num_pos_y) 
                                        for j in range(self.num_pos_x)]
        self.pos_dict = dict()
        pos_y_lb, pos_y_ub = 1.305, 1.545
        pos_x_lb, pos_x_ub = 0.43, 1.07
        x_coords = np.linspace(pos_x_lb, pos_x_ub, self.num_pos_x, endpoint=True)
        y_coords = np.linspace(pos_y_lb, pos_y_ub, self.num_pos_y, endpoint=True,)
        self.grid_pos = list(product(x_coords, y_coords))
        for i, (x, y) in enumerate(self.grid_pos):
            self.pos_dict[self._positions[i]] = {"xx": x, "yy": y}

        # Ball
        self._ball = Object("ball", self._ball_type)

        # Target
        self._target = Object("target", self._target_type)


        super().__init__(use_gui=use_gui)

        # Define new predicates if desired
        self._FanOn = Predicate("FanOn", [self._fan_type], self._FanOn_holds)
        self._BallAtTarget = Predicate("BallAtTarget",
                                       [self._ball_type, self._target_type],
                                       self._BallAtTarget_holds)
        self._BallAtPos = Predicate("BallAtPos",
                                    [self._ball_type, self._position_type],
                                    self._BallAtPos_holds)
        self._LeftOf = Predicate("LeftOf", [self._position_type, 
                                            self._position_type],
                                            self._LeftOf_holds)
        self._RightOf = Predicate("RightOf", [self._position_type, 
                                            self._position_type],
                                            self._RightOf_holds)
        self._UpOf = Predicate("UpOf", [self._position_type, 
                                            self._position_type],
                                            self._UpOf_holds)
        self._DownOf = Predicate("DownOf", [self._position_type, 
                                            self._position_type],
                                            self._DownOf_holds)
        self._ClearPos = Predicate("ClearPos", [self._position_type],
                                   self._ClearPos_holds)
        self._LeftFanSwitch = Predicate("LeftFanSwitch", [self._switch_type],
                                        self._LeftFanSwitch_holds)
        self._RightFanSwitch = Predicate("RightFanSwitch", [self._switch_type],
                                        self._RightFanSwitch_holds)
        self._FrontFanSwitch = Predicate("FrontFanSwitch", [self._switch_type],
                                         self._FrontFanSwitch_holds)
        self._BackFanSwitch = Predicate("BackFanSwitch", [self._switch_type],
                                        self._BackFanSwitch_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_fan"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._FanOn, self._BallAtTarget, self._BallAtPos,
        self._ClearPos,
        self._LeftOf, self._RightOf, self._UpOf, self._DownOf,
        self._LeftFanSwitch, self._RightFanSwitch, self._FrontFanSwitch,
        self._BackFanSwitch
                }

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._fan_type, self._switch_type,
            self._wall_type, self._ball_type, self._target_type,
            self._position_type
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._BallAtTarget}

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    # -------------------------------------------------------------------------
    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Create a table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id

        # ---------------------------------------------------------------------
        # Create fans in four groups: left, right, back, front
        # We'll store them in the dictionary as fan_ids_left, fan_ids_right, ...
        # ---------------------------------------------------------------------
        fan_urdf = "urdf/partnet_mobility/fan/101450/mobility.urdf"

        left_fan_ids = []
        for _ in range(cls.num_left_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            left_fan_ids.append(fid)

        right_fan_ids = []
        for _ in range(cls.num_right_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            right_fan_ids.append(fid)

        back_fan_ids = []
        for _ in range(cls.num_back_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            back_fan_ids.append(fid)

        front_fan_ids = []
        for _ in range(cls.num_front_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            front_fan_ids.append(fid)

        bodies["fan_ids_left"] = left_fan_ids
        bodies["fan_ids_right"] = right_fan_ids
        bodies["fan_ids_back"] = back_fan_ids
        bodies["fan_ids_front"] = front_fan_ids

        # ---------------------------------------------------------------------
        # Create 4 switches at the requested positions
        #   order: left=0, right=1, back=2, front=3
        # ---------------------------------------------------------------------
        switch_urdf = "urdf/partnet_mobility/switch/102812/switch.urdf"

        mid_x = cls.x_lb + (cls.x_ub - cls.x_lb) / 2
        bottom_fan_y = cls.y_lb + 0.14
        top_fan_y = cls.y_ub + cls.fan_x_len / 2
        mid_y = (bottom_fan_y + top_fan_y) / 2
        switch_positions = [
            (cls.x_lb - cls.fan_x_len * 5, mid_y, 0.0),  # left  -> side=0
            (cls.x_ub + cls.fan_x_len * 5, mid_y, np.pi),  # right -> side=1
            (mid_x, top_fan_y, -np.pi / 2),  # back  -> side=2
            (mid_x, bottom_fan_y, np.pi / 2),  # front -> side=3
        ]

        switch_ids = []
        for (sx, sy, srot) in switch_positions:
            sid = create_object(asset_path=switch_urdf,
                                position=(sx, sy, cls.table_height),
                                orientation=p.getQuaternionFromEuler(
                                    [0, 0, srot]),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            switch_ids.append(sid)
        bodies["switch_ids"] = switch_ids

        # ---------------------------------------------------------------------
        # Maze walls
        # ---------------------------------------------------------------------
        wall_ids = []
        for _ in range(cls.num_walls):
            wall_id = create_pybullet_block(
            color=(0.5, 0.5, 0.5, 1.0),
            half_extents=(cls.wall_x_len/2, cls.wall_y_len/2, cls.wall_z_len/2),
            mass=0.0,
            friction=0.5,
            position=(0.75, 1.28, cls.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
            wall_ids.append(wall_id)
        bodies["wall_ids"] = wall_ids

        # ---------------------------------------------------------------------
        # Create the ball
        # ---------------------------------------------------------------------
        ball_id = create_pybullet_sphere(
            color=(0.0, 0.0, 1.0, 1),
            radius=0.05,
            mass=0.01,
            friction=10,
            position=(0.75, 1.35, cls.table_height + 0.05),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        p.changeDynamics(ball_id,
                         -1,
                         linearDamping=10.0,
                         angularDamping=10.0,
                         physicsClientId=physics_client_id)
        bodies["ball_id"] = ball_id

        # ---------------------------------------------------------------------
        # Create the target
        # ---------------------------------------------------------------------
        target_id = create_pybullet_block(color=(0, 1, 0, 1.0),
                                          half_extents=(0.03, 0.03, 0.0001),
                                          mass=0.0,
                                          friction=0.5,
                                          position=(0, 0, cls.table_height),
                                          orientation=p.getQuaternionFromEuler(
                                              [0, 0, 0]),
                                          physics_client_id=physics_client_id)
        bodies["target_id"] = target_id

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _get_joint_id(obj_id: int, joint_name: str) -> int:
        num_joints = p.getNumJoints(obj_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to all PyBullet object IDs and their joints."""
        self._fans.clear()
        # 0 = left, 1 = right, 2 = back, 3 = front

        for fid in pybullet_bodies["fan_ids_left"]:
            fan_obj = Object(f"fan_left_{len(self._fans)}", self._fan_type)
            fan_obj.id = fid
            fan_obj.side_idx = 0
            fan_obj.joint_id = self._get_joint_id(fid, "joint_0")
            self._fans.append(fan_obj)

        for fid in pybullet_bodies["fan_ids_right"]:
            fan_obj = Object(f"fan_right_{len(self._fans)}", self._fan_type)
            fan_obj.id = fid
            fan_obj.side_idx = 1
            fan_obj.joint_id = self._get_joint_id(fid, "joint_0")
            self._fans.append(fan_obj)

        for fid in pybullet_bodies["fan_ids_back"]:
            fan_obj = Object(f"fan_back_{len(self._fans)}", self._fan_type)
            fan_obj.id = fid
            fan_obj.side_idx = 2
            fan_obj.joint_id = self._get_joint_id(fid, "joint_0")
            self._fans.append(fan_obj)

        for fid in pybullet_bodies["fan_ids_front"]:
            fan_obj = Object(f"fan_front_{len(self._fans)}", self._fan_type)
            fan_obj.id = fid
            fan_obj.side_idx = 3
            fan_obj.joint_id = self._get_joint_id(fid, "joint_0")
            self._fans.append(fan_obj)

        # Switches
        for i, switch_obj in enumerate(self._switches):
            switch_obj.id = pybullet_bodies["switch_ids"][i]
            switch_obj.joint_id = self._get_joint_id(switch_obj.id, "joint_0")
            switch_obj.side_idx = i  # 0=left,1=right,2=back,3=front

        for wall, id in zip(self._walls, pybullet_bodies["wall_ids"]):
            wall.id = id
        self._ball.id = pybullet_bodies["ball_id"]
        self._target.id = pybullet_bodies["target_id"]

    # -------------------------------------------------------------------------
    # Read state from PyBullet
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        return []

    def _create_task_specific_objects(self, state: State) -> None:
        pass

    def _reset_custom_env_state(self, state: State) -> None:
        for switch_obj in self._switches:
            is_on_val = state.get(switch_obj, "is_on")
            self._set_switch_on(switch_obj.id, bool(is_on_val > 0.5))
        
        oov_x, oov_y = self._out_of_view_xy
        # Move irrelavent walls oov
        wall_obj = state.get_objects(self._wall_type)
        for i in range(len(wall_obj), len(self._walls)):
            update_object(self._walls[i].id, position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._fan_type:
            if feature == "side":
                return float(obj.side_idx)
            elif feature == "is_on":
                controlling_switch = self._switches[obj.side_idx]
                return float(self._is_switch_on(controlling_switch.id))
        elif obj.type == self._switch_type:
            if feature == "side":
                return float(obj.side_idx)
            elif feature == "is_on":
                return float(self._is_switch_on(obj.id))
        elif obj.type == self._target_type:
            if feature == "is_hit":
                bx = self._current_observation.get(self._ball, "x")
                by = self._current_observation.get(self._ball, "y")
                tx = self._current_observation.get(self._target, "x")
                ty = self._current_observation.get(self._target, "y")
                return 1.0 if self._is_ball_close_to_position(bx, by, tx, ty) \
                    else 0.0
        elif obj.type == self._position_type:
            if feature == "xx":
                try:
                    return self.pos_dict[obj]["xx"]
                except:
                    breakpoint()
            elif feature == "yy":
                return self.pos_dict[obj]["yy"]


        raise ValueError(f"Unknown feature {feature} for object {obj}")

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------
    def step(self, action: Action, render_obs: bool = False) -> State:
        """Execute a low-level action, then spin fans & blow the ball."""
        next_state = super().step(action, render_obs=render_obs)
        self._simulate_fans()
        final_state = self._get_state()
        self._current_observation = final_state
        # Draw a debug line at the ball's position
        bx, by = final_state.get(self._ball, "x"), final_state.get(self._ball, "y")
        p.addUserDebugLine([bx, by, self.table_height], 
                           [bx, by, self.table_height+0.2], 
                           [0, 1, 0],
                    lifeTime=0.2,  # short lifetime so each step refreshes
                    physicsClientId=self._physics_client_id)
        return final_state

    # -------------------------------------------------------------------------
    # Fan Simulation
    # -------------------------------------------------------------------------
    def _simulate_fans(self) -> None:
        """Spin any switched-on fans and blow the ball."""
        # For each switch, if on => spin all fans with same side_idx
        for side_idx, switch_obj in enumerate(self._switches):
            on = self._is_switch_on(switch_obj.id)
            side_fans = [f for f in self._fans if f.side_idx == side_idx]
            if on:
                for fan_obj in side_fans:
                    if fan_obj.joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_obj.id,
                            jointIndex=fan_obj.joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=self.fan_spin_velocity,
                            force=20.0,
                            physicsClientId=self._physics_client_id,
                        )
                self._apply_fan_force_to_ball(fan_obj.id, self._ball.id)
            else:
                for fan_obj in side_fans:
                    if fan_obj.joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_obj.id,
                            jointIndex=fan_obj.joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=0.0,
                            force=20.0,
                            physicsClientId=self._physics_client_id,
                        )

    def _apply_fan_force_to_ball(self, fan_id: int, ball_id: int) -> None:
        """Compute the direction the fan blows (+X in fan local frame) and
        apply force."""
        pos_fan, orn_fan = p.getBasePositionAndOrientation(
            fan_id, self._physics_client_id)

        if CFG.fan_fans_blow_opposite_direction:
            local_dir = np.array([-1.0, 0.0, 0.0])
        else:
            local_dir = np.array([1.0, 0.0, 0.0])  # +X is "forward"
        rmat = np.array(p.getMatrixFromQuaternion(orn_fan)).reshape((3, 3))
        world_dir = rmat.dot(local_dir)
        pos_ball, _ = p.getBasePositionAndOrientation(ball_id,
                                                      self._physics_client_id)
        force_vec = self.wind_force_magnitude * world_dir
        p.applyExternalForce(
            objectUniqueId=ball_id,
            linkIndex=-1,
            forceObj=force_vec.tolist(),
            posObj=pos_ball,
            flags=p.WORLD_FRAME,
            physicsClientId=self._physics_client_id,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _is_switch_on(self, switch_id: int) -> bool:
        """Check if a switch's joint is above the threshold."""
        joint_id = self._get_joint_id(switch_id, "joint_0")
        if joint_id < 0:
            return False
        j_pos, _, _, _ = p.getJointState(
            switch_id, joint_id, physicsClientId=self._physics_client_id)
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_switch_on(self, switch_id: int, power_on: bool) -> None:
        """Programmatically toggle a switch on/off."""
        joint_id = self._get_joint_id(switch_id, "joint_0")
        if joint_id < 0:
            return
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = j_max if power_on else j_min
        p.resetJointState(
            switch_id,
            joint_id,
            target_val * self.switch_joint_scale,
            physicsClientId=self._physics_client_id,
        )

    def _is_ball_close_to_position(self, bx: float, by: float, tx: float,
                                 ty: float) -> bool:
        """Check if the ball is close to the target."""
        dist = np.sqrt((bx - tx)**2 + (by - ty)**2)
        return dist < 0.02

    # -------------------------------------------------------------------------
    # Predicates
    # -------------------------------------------------------------------------
    @staticmethod
    def _FanOn_holds(state: State, objects: Sequence[Object]) -> bool:
        """(FanOn fan).

        True if the controlling switch is on.
        """
        (fan, ) = objects
        return state.get(fan, "is_on") > 0.5

    def _BallAtTarget_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        ball, target = objects
        return self._is_ball_close_to_position(state.get(ball, "x"),
                                             state.get(ball, "y"),
                                             state.get(target, "x"),
                                             state.get(target, "y"))

    def _BallAtPos_holds(self, state: State, objects: Sequence[Object]) -> bool:
        ball, pos = objects
        return self._is_ball_close_to_position(state.get(ball, "x"),
                                             state.get(ball, "y"),
                                             state.get(pos, "xx"),
                                             state.get(pos, "yy"))
    
    def _LeftOf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        pos1, pos2 = objects
        return self._is_ball_close_to_position(
                                            state.get(pos1, "xx") + self.pos_gap,
                                            state.get(pos1, "yy"),
                                            state.get(pos2, "xx"),
                                            state.get(pos2, "yy")
                                            )

    def _RightOf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        pos1, pos2 = objects
        return self._is_ball_close_to_position(
                                            state.get(pos1, "xx") - self.pos_gap,
                                            state.get(pos1, "yy"),
                                            state.get(pos2, "xx"),
                                            state.get(pos2, "yy")
                                            )

    def _UpOf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        pos1, pos2 = objects
        return self._is_ball_close_to_position(
                                            state.get(pos1, "xx"),
                                            state.get(pos1, "yy") - self.pos_gap,
                                            state.get(pos2, "xx"),
                                            state.get(pos2, "yy")
                                            )

    def _DownOf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """(DownOf pos1 pos2)."""
        pos1, pos2 = objects
        return self._is_ball_close_to_position(
                                            state.get(pos1, "xx"),
                                            state.get(pos1, "yy") + self.pos_gap,
                                            state.get(pos2, "xx"),
                                            state.get(pos2, "yy"))
    
    def _ClearPos_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """If the position is clear of walls."""

        pos, = objects
        pos_x, pos_y = state.get(pos, "xx"), state.get(pos, "yy")
        for obj in state.get_objects(self._wall_type):
            wx, wy = state.get(obj, "x"), state.get(obj, "y")
            if self._is_ball_close_to_position(pos_x, pos_y, wx, wy):
                return False
        return True
    
    def _LeftFanSwitch_holds(self, state: State, objects: Sequence[Object]
                             ) -> bool:
        switch, = objects
        return state.get(switch, "side") == 0
    
    def _RightFanSwitch_holds(self, state: State, objects: Sequence[Object]
                                ) -> bool:
        switch, = objects
        return state.get(switch, "side") == 1
    
    def _FrontFanSwitch_holds(self, state: State, objects: Sequence[Object]
                                ) -> bool:
        switch, = objects
        return state.get(switch, "side") == 3
    
    def _BackFanSwitch_holds(self, state: State, objects: Sequence[Object]
                                ) -> bool:
        switch, = objects
        return state.get(switch, "side") == 2
                    

    # -------------------------------------------------------------------------
    # Task Generation
    # -------------------------------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int,
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        # Example only; not fully updated. You can adapt as needed.
        # Make a tuple of tuple where each tuple is a x,y position
        # Starting from 
        fan_y_lb, fan_y_ub = 1.35, 1.50
        fan_x_lb, fan_x_ub = 0.40, 1.10 # 0.7
        left_coords = np.linspace(fan_y_lb, fan_y_ub, self.num_left_fans)
        right_coords = np.linspace(fan_y_lb, fan_y_ub, self.num_right_fans)
        front_coords = np.linspace(fan_x_lb, fan_x_ub, self.num_front_fans)
        back_coords = np.linspace(fan_x_lb, fan_x_ub, self.num_back_fans)


        # Draw a debug line mark on each of the positions
        for pos_obj in self._positions:
            pos = self.pos_dict[pos_obj]
            p.addUserDebugLine([pos["xx"], pos["yy"], self.table_height],
                               [pos["xx"], pos["yy"], self.table_height + 0.2],
                               [1, 0, 0],
                               parentObjectUniqueId=-1,
                               parentLinkIndex=-1)

        tasks = []
        for _ in range(num_tasks):
            num_walls = 3
            available_pos = self.grid_pos.copy()
            # Robot
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }

            # Target
            tar_pos = tuple(rng.choice(available_pos))
            available_pos.remove(tar_pos)
            target_dict = {
                "x": tar_pos[0],
                "y": tar_pos[1],
                "z": self.table_height,
                "rot": 0.0,
                "is_hit": 0.0,
            }

            init_dict = {}
            init_dict[self._robot] = robot_dict
            init_dict[self._target] = target_dict

            for fan_obj in self._fans:
                # pick the position from the real environment:
                # we can read it out of the environment if we want:
                # but for random tasks, you might randomize it.
                # We'll store placeholders:
                side_idx = fan_obj.side_idx
                # We'll replicate the rough positions from the real creation:
                if side_idx == 2:  # front
                    i = len([
                        f for f in self._fans
                        if f.side_idx == 2 and f.name < fan_obj.name
                    ])
                    # i.e. the nth front fan
                    px = front_coords[i]
                    py = self.y_lb + 0.14
                    rot = np.pi / 2
                elif side_idx == 3:  # back
                    i = len([
                        f for f in self._fans
                        if f.side_idx == 3 and f.name < fan_obj.name
                    ])
                    px = back_coords[i]
                    py = self.y_ub + self.fan_x_len / 2
                    rot = -np.pi / 2
                elif side_idx == 0:  # left
                    i = len([
                        f for f in self._fans
                        if f.side_idx == 0 and f.name < fan_obj.name
                    ])
                    px = self.x_lb - self.fan_x_len * 5
                    py = left_coords[i]
                    rot = 0.0
                else:  # right
                    i = len([
                        f for f in self._fans
                        if f.side_idx == 1 and f.name < fan_obj.name
                    ])
                    px = self.x_ub + self.fan_x_len * 5
                    py = right_coords[i]
                    rot = np.pi
                fan_dict = {
                    "x": px,
                    "y": py,
                    "z": self.table_height + self.fan_z_len / 2,
                    "rot": rot,
                    "side": float(side_idx),
                    "is_on": 0.0
                }
                init_dict[fan_obj] = fan_dict

            # Switches default off
            for switch_obj in self._switches:
                init_dict[switch_obj] = {
                    "x": 0.60 + 0.08 * switch_obj.side_idx,
                    "y": self.robot_init_y - 0.2,
                    "z": self.table_height,
                    "rot": np.pi / 2,
                    "side": float(switch_obj.side_idx),
                    "is_on": 0.0,
                }

            # Walls
            for i in range(num_walls):
                wall_pos = tuple(rng.choice(available_pos))
                available_pos.remove(wall_pos)
                init_dict[self._walls[i]] = {
                            "x": wall_pos[0],
                            "y": wall_pos[1],
                            "z": self.table_height,
                            "rot": rng.uniform(-np.pi/2, np.pi/2)}
                            
            # Ball
            ball_pos = tuple(rng.choice(available_pos))
            available_pos.remove(ball_pos)
            ball_dict = {"x": ball_pos[0], 
                         "y": ball_pos[1], 
                         "z": self.table_height + 0.05}
            init_dict[self._ball] = ball_dict

            init_dict.update(self.pos_dict)


            init_state = utils.create_state_from_dict(init_dict)

            # The positions that has the same coord as the target
            tx, ty = init_state.get(self._target, "x"), \
                    init_state.get(self._target, "y")
            for pos_obj in self._positions:
                px, py = init_state.get(pos_obj, "xx"), \
                    init_state.get(pos_obj, "yy")
                if np.isclose(px, tx, atol=0.01) and \
                    np.isclose(py, ty, atol=0.01):
                    target_pos_obj = pos_obj
                    break
            goal_atoms = {
                GroundAtom(self._BallAtPos, [self._ball, target_pos_obj]),
                # GroundAtom(self._BallAtTarget, [self._ball, self._target])
            }
            tasks.append(EnvironmentTask(init_state, goal_atoms))
        return self._add_pybullet_state_to_tasks(tasks)

if __name__ == "__main__":
    import time
    CFG.seed = 0
    CFG.env = "pybullet_fan"
    CFG.pybullet_sim_steps_per_action = 1
    CFG.fan_fans_blow_opposite_direction = True
    env = PyBulletFanEnv(use_gui=True)
    rng = np.random.default_rng(CFG.seed)
    tasks = env._make_tasks(1, rng)

    for task in tasks:
        env._reset_state(task.init)
        for _ in range(100000):
            action = Action(np.array(env._pybullet_robot.initial_joint_positions))
            env.step(action)
            time.sleep(0.01)
