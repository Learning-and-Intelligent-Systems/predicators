import logging
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
    z_ub: ClassVar[float] = table_height + 0.3
    init_padding: float = 0.05

    # -------------------------------------------------------------------------
    # Robot config
    # -------------------------------------------------------------------------
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.72, 0.0)
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
    NUM_FRONT_FANS: ClassVar[int] = 5
    NUM_BACK_FANS: ClassVar[int] = 5
    NUM_LEFT_FANS: ClassVar[int] = 2
    NUM_RIGHT_FANS: ClassVar[int] = 2

    # -------------------------------------------------------------------------
    # URDF scale or references
    # -------------------------------------------------------------------------
    fan_scale: ClassVar[float] = 0.08
    switch_joint_scale: ClassVar[float] = 0.1
    switch_on_threshold: ClassVar[float] = 0.5  # fraction of the joint range
    fan_spin_velocity: ClassVar[float] = 100.0  # velocity for joint_0
    wind_force_magnitude: ClassVar[float] = 0.1  # force on the ball
    switch_x_len: ClassVar[float] = 0.10  # length of the switch
    fan_x_len: ClassVar[float] = 0.2 * fan_scale  # length of the fan blades
    fan_z_len: ClassVar[float] = 1.5 * fan_scale  # height of the fan base

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
        ])
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
        ])
    _wall_type = Type("wall", ["x", "y", "z", "rot", "length"])
    _ball_type = Type("ball", ["x", "y", "z"])
    _target_type = Type("target", ["x", "y", "z", "rot", "is_hit"])

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
        self._wall1 = Object("wall1", self._wall_type)
        self._wall2 = Object("wall2", self._wall_type)

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

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_fan"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._FanOn, self._BallAtTarget}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._fan_type, self._switch_type,
            self._wall_type, self._ball_type, self._target_type
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
        for _ in range(cls.NUM_LEFT_FANS):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            left_fan_ids.append(fid)

        right_fan_ids = []
        for _ in range(cls.NUM_RIGHT_FANS):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            right_fan_ids.append(fid)

        back_fan_ids = []
        for _ in range(cls.NUM_BACK_FANS):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            back_fan_ids.append(fid)

        front_fan_ids = []
        for _ in range(cls.NUM_FRONT_FANS):
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
        wall_x_len, wall_y_len, wall_z_len = 0.05, 0.02, 0.02
        wall_id_1 = create_pybullet_block(
            color=(0.5, 0.5, 0.5, 1.0),
            half_extents=(wall_x_len, wall_y_len, wall_z_len),
            mass=0.0,
            friction=0.5,
            position=(0.75, 1.28, cls.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        bodies["wall_id_1"] = wall_id_1

        wall_id_2 = create_pybullet_block(
            color=(0.5, 0.5, 0.5, 1.0),
            half_extents=(wall_x_len, wall_y_len, wall_z_len),
            mass=0.0,
            friction=0.5,
            position=(0.75, 1.42, cls.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        bodies["wall_id_2"] = wall_id_2

        # ---------------------------------------------------------------------
        # Create the ball
        # ---------------------------------------------------------------------
        ball_id = create_pybullet_sphere(
            color=(1.0, 0.0, 0.0, 1),
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

        self._wall1.id = pybullet_bodies["wall_id_1"]
        self._wall2.id = pybullet_bodies["wall_id_2"]
        self._ball.id = pybullet_bodies["ball_id"]
        self._target.id = pybullet_bodies["target_id"]

    # -------------------------------------------------------------------------
    # Read state from PyBullet
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        return []

    def _get_state(self) -> State:
        """Construct a State from the current PyBullet simulation."""
        state_dict: Dict[Object, Dict[str, float]] = {}

        # Robot
        rx, ry, rz, qx, qy, qz, qw, rf = self._pybullet_robot.get_state()
        _, tilt, wrist = p.getEulerFromQuaternion([qx, qy, qz, qw])
        state_dict[self._robot] = {
            "x": rx,
            "y": ry,
            "z": rz,
            "fingers": self._fingers_joint_to_state(self._pybullet_robot, rf),
            "tilt": tilt,
            "wrist": wrist,
        }

        # Fans
        for fan_obj in self._fans:
            (fx, fy, fz), forn = p.getBasePositionAndOrientation(
                fan_obj.id, self._physics_client_id)
            euler_f = p.getEulerFromQuaternion(forn)
            # side_idx in [0=left,1=right,2=back,3=front]
            # Check if the controlling switch is on:
            controlling_switch = self._switches[fan_obj.side_idx]
            is_on_val = float(self._is_switch_on(controlling_switch.id))

            state_dict[fan_obj] = {
                "x": fx,
                "y": fy,
                "z": fz,
                "rot": euler_f[2],
                "side": float(fan_obj.side_idx),
                "is_on": is_on_val
            }

        # Switches
        for switch_obj in self._switches:
            (sx, sy, sz), sorn = p.getBasePositionAndOrientation(
                switch_obj.id, self._physics_client_id)
            euler_s = p.getEulerFromQuaternion(sorn)
            is_on_val = float(self._is_switch_on(switch_obj.id))
            state_dict[switch_obj] = {
                "x": sx,
                "y": sy,
                "z": sz,
                "rot": euler_s[2],
                "side": float(switch_obj.side_idx),
                "is_on": is_on_val,
            }

        # Walls
        for wall_obj in [self._wall1, self._wall2]:
            (wx, wy, wz), worn = p.getBasePositionAndOrientation(
                wall_obj.id, self._physics_client_id)
            euler_w = p.getEulerFromQuaternion(worn)
            wall_len = 0.3
            state_dict[wall_obj] = {
                "x": wx,
                "y": wy,
                "z": wz,
                "rot": euler_w[2],
                "length": wall_len,
            }

        # Ball
        (bx, by,
         bz), _ = p.getBasePositionAndOrientation(self._ball.id,
                                                  self._physics_client_id)
        state_dict[self._ball] = {"x": bx, "y": by, "z": bz}

        # Target
        (tx, ty,
         tz), torn = p.getBasePositionAndOrientation(self._target.id,
                                                     self._physics_client_id)
        euler_t = p.getEulerFromQuaternion(torn)
        is_hit_val = float(self._is_ball_close_to_target(bx, by, tx, ty))
        state_dict[self._target] = {
            "x": tx,
            "y": ty,
            "z": tz,
            "rot": euler_t[2],
            "is_hit": is_hit_val,
        }

        pyb_state = utils.create_state_from_dict(state_dict)
        joint_positions = self._pybullet_robot.get_joints()
        full_state = utils.PyBulletState(
            pyb_state.data,
            simulator_state={"joint_positions": joint_positions},
        )
        return full_state

    # -------------------------------------------------------------------------
    # Reset state
    # -------------------------------------------------------------------------
    def _reset_state(self, state: State) -> None:
        """Reset simulation from a given state."""
        super()._reset_state(state)  # resets robot

        # Rebuild object list
        self._objects = [
            self._robot, *self._fans, *self._switches, self._wall1,
            self._wall2, self._ball, self._target
        ]

        # Fans
        for fan_obj in self._fans:
            fx = state.get(fan_obj, "x")
            fy = state.get(fan_obj, "y")
            fz = state.get(fan_obj, "z")
            frot = state.get(fan_obj, "rot")
            update_object(
                fan_obj.id,
                position=(fx, fy, fz),
                orientation=p.getQuaternionFromEuler([0, 0, frot]),
                physics_client_id=self._physics_client_id,
            )

        # Switches
        for switch_obj in self._switches:
            sx = state.get(switch_obj, "x")
            sy = state.get(switch_obj, "y")
            sz = state.get(switch_obj, "z")
            srot = state.get(switch_obj, "rot")
            is_on_val = state.get(switch_obj, "is_on")
            update_object(
                switch_obj.id,
                position=(sx, sy, sz),
                orientation=p.getQuaternionFromEuler([0, 0, srot]),
                physics_client_id=self._physics_client_id,
            )
            self._set_switch_on(switch_obj.id, bool(is_on_val > 0.5))

        # Walls
        for wall_obj in [self._wall1, self._wall2]:
            wx = state.get(wall_obj, "x")
            wy = state.get(wall_obj, "y")
            wz = state.get(wall_obj, "z")
            wrot = state.get(wall_obj, "rot")
            update_object(
                wall_obj.id,
                position=(wx, wy, wz),
                orientation=p.getQuaternionFromEuler([0, 0, wrot]),
                physics_client_id=self._physics_client_id,
            )

        # Ball
        bx = state.get(self._ball, "x")
        by = state.get(self._ball, "y")
        bz = state.get(self._ball, "z")
        update_object(
            self._ball.id,
            position=(bx, by, bz),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=self._physics_client_id,
        )

        # Target
        tx = state.get(self._target, "x")
        ty = state.get(self._target, "y")
        tz = state.get(self._target, "z")
        trot = state.get(self._target, "rot")
        update_object(
            self._target.id,
            position=(tx, ty, tz),
            orientation=p.getQuaternionFromEuler([0, 0, trot]),
            physics_client_id=self._physics_client_id,
        )

        # Check reconstruction
        reconstructed = self._get_state()
        if not reconstructed.allclose(state):
            logging.warning("Could not reconstruct state exactly!")

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------
    def step(self, action: Action, render_obs: bool = False) -> State:
        """Execute a low-level action, then spin fans & blow the ball."""
        next_state = super().step(action, render_obs=render_obs)
        self._simulate_fans()
        final_state = self._get_state()
        self._current_observation = final_state
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

    def _is_ball_close_to_target(self, bx: float, by: float, tx: float,
                                 ty: float) -> bool:
        dist = np.hypot(bx - tx, by - ty)
        return dist < 0.05

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
        return self._is_ball_close_to_target(state.get(ball, "x"),
                                             state.get(ball, "y"),
                                             state.get(target, "x"),
                                             state.get(target, "y"))

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
        tasks = []
        for _ in range(num_tasks):
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
            target_dict = {
                "x": 0.95,
                "y": 1.45,
                "z": self.table_height,
                "rot": 0.0,
                "is_hit": 0.0,
            }

            init_dict = {}
            init_dict[self._robot] = robot_dict
            init_dict[self._target] = target_dict

            front_coords = np.linspace(0.40, 1.10, self.NUM_FRONT_FANS)
            back_coords = np.linspace(0.40, 1.10, self.NUM_BACK_FANS)
            side_y_lb, side_y_ub = 1.35, 1.50
            left_coords = np.linspace(side_y_lb, side_y_ub, self.NUM_LEFT_FANS)
            right_coords = np.linspace(side_y_lb, side_y_ub,
                                       self.NUM_RIGHT_FANS)

            for fan_obj in self._fans:
                # pick the position from the real environment:
                # we can read it out of the environment if we want:
                # but for random tasks, you might randomize it.
                fid = fan_obj.id  # doesn't help in a pure code snippet
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
            wall1_dict = {
                "x": 0.85,
                "y": 1.38,
                "z": self.table_height,
                "rot": np.pi / 2,
                "length": 0.3
            }
            wall2_dict = {
                "x": 0.65,
                "y": 1.42,
                "z": self.table_height,
                "rot": 0.0,
                "length": 0.3
            }
            init_dict[self._wall1] = wall1_dict
            init_dict[self._wall2] = wall2_dict

            # Ball
            ball_dict = {"x": 0.75, "y": 1.35, "z": self.table_height + 0.05}
            init_dict[self._ball] = ball_dict

            init_state = utils.create_state_from_dict(init_dict)
            goal_atoms = {
                GroundAtom(self._BallAtTarget, [self._ball, self._target])
            }
            tasks.append(EnvironmentTask(init_state, goal_atoms))
        return self._add_pybullet_state_to_tasks(tasks)


if __name__ == "__main__":
    import time
    CFG.seed = 0
    CFG.pybullet_sim_steps_per_action = 1
    env = PyBulletFanEnv(use_gui=True)
    rng = np.random.default_rng(CFG.seed)
    task = env._make_tasks(1, rng)[0]
    env._reset_state(task.init)

    while True:
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))
        env.step(action)
        time.sleep(0.01)
