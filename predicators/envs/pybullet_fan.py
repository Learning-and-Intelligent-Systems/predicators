import logging
from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block,\
    create_pybullet_sphere
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletFanEnv(PyBulletEnv):
    """A PyBullet environment where a ball is blown around by fans in a maze.

    There are four fans, each controlled by a corresponding switch. The ball
    starts in the maze. When a fan is on, it spins at joint_0 and blows the ball.
    The goal is for the ball to reach some target position.
    """

    # -------------------------------------------------------------------------
    # Table / workspace / Maze
    # (Adjust these as needed—here we reuse some values from the laser example.)
    # -------------------------------------------------------------------------
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2.0)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])

    # Bounds of the "maze" region (just for reference)
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = table_height + 0.3
    init_padding: float = 0.05

    # -------------------------------------------------------------------------
    # Robot config
    # (Same idea as your Laser environment.)
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
    # URDF scale or references (adjust to taste)
    # -------------------------------------------------------------------------
    switch_joint_scale: ClassVar[float] = 0.1
    switch_on_threshold: ClassVar[float] = 0.5  # fraction of the joint range
    fan_spin_velocity: ClassVar[float] = 100.0   # (example) velocity for joint_0
    wind_force_magnitude: ClassVar[float] = 0.1 # (example) force on the ball
    switch_x_len: ClassVar[float] = 0.10  # length of the switch
    fan_x_len: ClassVar[float] = 0.02  # length of the fan blades
    fan_z_len: ClassVar[float] = 0.15  # height of the fan base

    # -------------------------------------------------------------------------
    # Types
    # 
    # We'll define them analogously to the laser environment. 
    # The user specifically wants: fan, switch, wall, ball, robot, plus a target. 
    # We'll keep them simple. 
    # -------------------------------------------------------------------------
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _fan_type = Type("fan", ["x", "y", "z", "rot",  # fan base pose
                             "switch_x", "switch_y", "switch_z", "switch_rot",
                             "is_on"])  # same features as requested
    _wall_type = Type("wall", ["x", "y", "z", "rot", "length"])
    _ball_type = Type("ball", ["x", "y", "z"])
    _target_type = Type("target", ["x", "y", "z", "rot", "is_hit"])

    # -------------------------------------------------------------------------
    # Environment initialization
    # -------------------------------------------------------------------------
    def __init__(self, use_gui: bool = True) -> None:
        # Create environment objects (logic-level).
        self._robot = Object("robot", self._robot_type)

        # Four fans and their switches (we model them as one object each in code,
        # but logically you might separate them into two objects, or keep them combined).
        self._fan1 = Object("fan1", self._fan_type)
        self._fan2 = Object("fan2", self._fan_type)
        self._fan3 = Object("fan3", self._fan_type)
        self._fan4 = Object("fan4", self._fan_type)

        # Maze walls
        self._wall1 = Object("wall1", self._wall_type)
        self._wall2 = Object("wall2", self._wall_type)
        # Add as many walls as you want
        # ...

        # Ball
        self._ball = Object("ball", self._ball_type)

        # Target
        self._target = Object("target", self._target_type)

        super().__init__(use_gui=use_gui)

        # Define new predicates if desired, e.g. "FanOn", "BallAtTarget", etc.
        # For example:
        self._FanOn = Predicate("FanOn", [self._fan_type], self._FanOn_holds)
        self._BallAtTarget = Predicate("BallAtTarget", [self._ball_type, self._target_type],
                                       self._BallAtTarget_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_fan"

    @property
    def predicates(self) -> Set[Predicate]:
        """Return any environment-specific predicates (for planning, etc.)."""
        return {self._FanOn, self._BallAtTarget}

    @property
    def types(self) -> Set[Type]:
        """Return all custom types for this environment."""
        return {
            self._robot_type, self._fan_type, self._wall_type,
            self._ball_type, self._target_type
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Goal might be: get the ball to the target."""
        return {self._BallAtTarget}

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    # -------------------------------------------------------------------------
    @classmethod
    def initialize_pybullet(
        cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(using_gui)

        # Create a table or ground plane
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id

        # -------------------
        # Create the 4 fans + 4 switches
        # Each fan: "urdf/partnet_mobility/fan/101450/mobility.urdf"
        # Each switch: "urdf/partnet_mobility/switch/102812/mobility.urdf"
        # 
        # Adjust the positions/orientations so they face inward toward the ball.
        # Example arrangement around the (x_mid, y_mid) of the table:
        # -------------------
        # For brevity, we just do placeholders:
        fan_urdf = "urdf/partnet_mobility/fan/101450/mobility.urdf"
        switch_urdf = "urdf/partnet_mobility/switch/102812/switch.urdf"

        fan_ids = []
        # e.g., four corners around the ball
        fan_positions = [
            (0.75, 1.20, cls.table_height),
            (1.00, 1.35, cls.table_height),
            (0.75, 1.50, cls.table_height),
            (0.50, 1.35, cls.table_height),
        ]
        fan_orientations = [
            p.getQuaternionFromEuler([0, 0,  0]),
            p.getQuaternionFromEuler([0, 0,  np.pi/2]),
            p.getQuaternionFromEuler([0, 0,  np.pi]),
            p.getQuaternionFromEuler([0, 0, -np.pi/2]),
        ]

        for i in range(4):
            fid = create_object(
                asset_path=fan_urdf,
                position=fan_positions[i],
                orientation=fan_orientations[i],
                scale=0.1,
                use_fixed_base=True,
                physics_client_id=physics_client_id
            )
            fan_ids.append(fid)

        switch_ids = []
        switch_positions = [
            (cls.robot_init_x + 0.0, cls.robot_init_y - 0.10, cls.table_height),
            (cls.robot_init_x + 0.1, cls.robot_init_y - 0.10, cls.table_height),
            (cls.robot_init_x - 0.1, cls.robot_init_y - 0.10, cls.table_height),
            (cls.robot_init_x + 0.0, cls.robot_init_y - 0.20, cls.table_height),
        ]
        for i in range(4):
            sid = create_object(
                asset_path=switch_urdf,
                position=switch_positions[i],
                orientation=p.getQuaternionFromEuler([0, 0, 0]),
                scale=1,
                use_fixed_base=True,
                physics_client_id=physics_client_id
            )
            switch_ids.append(sid)

        bodies["fan_ids"] = fan_ids
        bodies["switch_ids"] = switch_ids

        # -------------------
        # Maze walls
        # Adjust URDF or create them in your own style
        # -------------------
        # Example: a single wall or a few walls forming a corridor
        wall_x_len, wall_y_len, wall_z_len = 0.05, 0.02, 0.02
        wall_id_1 = create_pybullet_block(
            color=(0.5, 0.5, 0.5, 1.0),
            half_extents=(wall_x_len, wall_y_len, wall_z_len),
            mass=0.0,
            friction=0.5,
            position=(0.75, 1.28, cls.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id
        )
        bodies["wall_id_1"] = wall_id_1

        wall_id_2 = create_pybullet_block(
            color=(0.5, 0.5, 0.5, 1.0),
            half_extents=(wall_x_len, wall_y_len, wall_z_len),
            mass=0.0,
            friction=0.5,
            position=(0.75, 1.42, cls.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id
        )
        bodies["wall_id_2"] = wall_id_2

        # -------------------
        # Create the ball
        # -------------------
        ball_id = create_pybullet_sphere(
            color=(1.0, 0.0, 0.0, 1),  # red color
            radius=0.05,  # adjust radius as needed
            mass=0.01,  # adjust mass as needed
            friction=10,  # adjust friction as needed
            position=(0.75, 1.35, cls.table_height + 0.05),  # center of fans
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id
        )
        p.changeDynamics(ball_id, -1,
                 linearDamping=10.0,    # try bigger if you want it to stop faster
                 angularDamping=10.0,
                 physicsClientId=physics_client_id)
        bodies["ball_id"] = ball_id

        # -------------------
        # Create the target
        # -------------------
        target_id = create_pybullet_block(
            color=(0, 1, 0, 1.0),
            half_extents=(0.03, 0.03, 0.0001),
            mass=0.0,
            friction=0.5,
            position=(0, 0, cls.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id
        )
        bodies["target_id"] = target_id

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _get_joint_id(obj_id: int, joint_name: str) -> int:
        """Helper: get the PyBullet joint ID given the joint name."""
        num_joints = p.getNumJoints(obj_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to all PyBullet object IDs and their relevant joints."""
        # Robot
        # (already stored by the parent class)

        # Fans
        self._fan1.id = pybullet_bodies["fan_ids"][0]
        self._fan2.id = pybullet_bodies["fan_ids"][1]
        self._fan3.id = pybullet_bodies["fan_ids"][2]
        self._fan4.id = pybullet_bodies["fan_ids"][3]

        # Switches
        # You might store a reference to the relevant joint for each switch
        self._fan1.switch_id = pybullet_bodies["switch_ids"][0]
        self._fan2.switch_id = pybullet_bodies["switch_ids"][1]
        self._fan3.switch_id = pybullet_bodies["switch_ids"][2]
        self._fan4.switch_id = pybullet_bodies["switch_ids"][3]

        # For each fan, get the spinning joint_0
        self._fan1.joint_id = self._get_joint_id(self._fan1.id, "joint_0")
        self._fan2.joint_id = self._get_joint_id(self._fan2.id, "joint_0")
        self._fan3.joint_id = self._get_joint_id(self._fan3.id, "joint_0")
        self._fan4.joint_id = self._get_joint_id(self._fan4.id, "joint_0")

        # Maze walls
        self._wall1.id = pybullet_bodies["wall_id_1"]
        self._wall2.id = pybullet_bodies["wall_id_2"]

        # Ball + target
        self._ball.id = pybullet_bodies["ball_id"]
        self._target.id = pybullet_bodies["target_id"]

    # -------------------------------------------------------------------------
    # Read state from PyBullet
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return a list of object IDs to check for contact with the robot."""
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
        # We'll read each fan's base position/orientation,
        # plus the corresponding switch position and orientation,
        # plus is_on (boolean: whether the switch is toggled beyond threshold).
        for fan_obj in [self._fan1, self._fan2, self._fan3, self._fan4]:
            # fan base
            (fx, fy, fz), forn = p.getBasePositionAndOrientation(
                fan_obj.id, self._physics_client_id)
            euler_f = p.getEulerFromQuaternion(forn)

            # switch
            (sx, sy, sz), sorn = p.getBasePositionAndOrientation(
                fan_obj.switch_id, self._physics_client_id)
            euler_s = p.getEulerFromQuaternion(sorn)

            # is_on
            is_on_val = float(self._is_switch_on(fan_obj.switch_id))

            state_dict[fan_obj] = {
                "x": fx,
                "y": fy,
                "z": fz,
                "rot": euler_f[2],
                "switch_x": sx,
                "switch_y": sy,
                "switch_z": sz,
                "switch_rot": euler_s[2],
                "is_on": is_on_val
            }

        # Walls
        for wall_obj in [self._wall1, self._wall2]:
            (wx, wy, wz), worn = p.getBasePositionAndOrientation(
                wall_obj.id, self._physics_client_id)
            euler_w = p.getEulerFromQuaternion(worn)
            # If you want to store length from the URDF or from a known constant:
            wall_len = 0.3  # example
            state_dict[wall_obj] = {
                "x": wx,
                "y": wy,
                "z": wz,
                "rot": euler_w[2],
                "length": wall_len,
            }

        # Ball
        (bx, by, bz), _ = p.getBasePositionAndOrientation(
            self._ball.id, self._physics_client_id)
        state_dict[self._ball] = {"x": bx, "y": by, "z": bz}

        # Target
        (tx, ty, tz), torn = p.getBasePositionAndOrientation(
            self._target.id, self._physics_client_id)
        euler_t = p.getEulerFromQuaternion(torn)
        # is_hit might represent "ball is close enough" or "some sensor"
        # We'll store 1.0 or 0.0 here if we want
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

        self._objects = [
            self._robot,
            self._fan1, self._fan2, self._fan3, self._fan4,
            self._wall1, self._wall2,
            self._ball,
            self._target
        ]

        # Fans + Switch
        for fan_obj in [self._fan1, self._fan2, self._fan3, self._fan4]:
            fx = state.get(fan_obj, "x")
            fy = state.get(fan_obj, "y")
            fz = state.get(fan_obj, "z")
            frot = state.get(fan_obj, "rot")
            update_object(
                fan_obj.id,
                position=(fx, fy, fz),
                orientation=p.getQuaternionFromEuler([0, 0, frot]),
                physics_client_id=self._physics_client_id
            )
            sx = state.get(fan_obj, "switch_x")
            sy = state.get(fan_obj, "switch_y")
            sz = state.get(fan_obj, "switch_z")
            srot = state.get(fan_obj, "switch_rot")
            update_object(
                fan_obj.switch_id,
                position=(sx, sy, sz),
                orientation=p.getQuaternionFromEuler([0, 0, srot]),
                physics_client_id=self._physics_client_id
            )
            # Set switch state if you need:
            self._set_switch_on(fan_obj.switch_id, bool(state.get(fan_obj, "is_on") > 0.5))

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
                physics_client_id=self._physics_client_id
            )

        # Ball
        bx = state.get(self._ball, "x")
        by = state.get(self._ball, "y")
        bz = state.get(self._ball, "z")
        update_object(
            self._ball.id,
            position=(bx, by, bz),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=self._physics_client_id
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
            physics_client_id=self._physics_client_id
        )

        # Check reconstruction
        reconstructed = self._get_state()
        if not reconstructed.allclose(state):
            logging.warning("Could not reconstruct state exactly!")

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------
    def step(self, action: Action, render_obs: bool = False) -> State:
        """Execute a low-level action (e.g. robot motion) and apply fan forces."""
        next_state = super().step(action, render_obs=render_obs)

        # After the robot action, handle spinning fans & apply wind to the ball
        self._simulate_fans()

        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    # -------------------------------------------------------------------------
    # Fan Simulation
    # -------------------------------------------------------------------------
    def _simulate_fans(self) -> None:
        """For each fan that is on, set joint velocity and blow the ball."""
        for fan_obj in [self._fan1, self._fan2, self._fan3, self._fan4]:
            on = self._is_switch_on(fan_obj.switch_id)
            if on:
                # Spin the joint
                if hasattr(fan_obj, "joint_id") and fan_obj.joint_id >= 0:
                    p.setJointMotorControl2(
                        bodyUniqueId=fan_obj.id,
                        jointIndex=fan_obj.joint_id,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=self.fan_spin_velocity,
                        force=10.0,  # max torque
                        physicsClientId=self._physics_client_id
                    )
                # Apply wind force to the ball
                self._apply_fan_force_to_ball(fan_obj.id, self._ball.id)
            else:
                # If the fan is off, stop spinning
                if hasattr(fan_obj, "joint_id") and fan_obj.joint_id >= 0:
                    p.setJointMotorControl2(
                        bodyUniqueId=fan_obj.id,
                        jointIndex=fan_obj.joint_id,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=0.0,
                        force=10.0,
                        physicsClientId=self._physics_client_id
                    )

    def _apply_fan_force_to_ball(self, fan_id: int, ball_id: int) -> None:
        """Compute the direction the fan is facing, apply a force to the ball."""
        pos_fan, orn_fan = p.getBasePositionAndOrientation(
            fan_id, self._physics_client_id)
        # The local direction that the fan "blows" could be +X or +Y in URDF space, etc.
        # Suppose it is +X in the fan's local frame, for example:
        local_dir = np.array([1.0, 0.0, 0.0])  # adjust as needed
        rmat = np.array(p.getMatrixFromQuaternion(orn_fan)).reshape((3, 3))
        world_dir = rmat.dot(local_dir)

        # Get ball's current position (to ensure the ball is in front, or just always apply)
        pos_ball, _ = p.getBasePositionAndOrientation(ball_id, self._physics_client_id)
        # You can check distance if you want a falloff, etc.

        force_vec = self.wind_force_magnitude * world_dir
        # Apply a force at the ball's center of mass
        p.applyExternalForce(
            objectUniqueId=ball_id,
            linkIndex=-1,
            forceObj=force_vec.tolist(),
            posObj=pos_ball,
            flags=p.WORLD_FRAME,
            physicsClientId=self._physics_client_id
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _is_switch_on(self, switch_id: int) -> bool:
        """Check if a switch's joint (joint_0) is above threshold."""
        # We assume there's exactly 1 relevant joint in the switch URDF, named "joint_0".
        joint_id = self._get_joint_id(switch_id, "joint_0")
        if joint_id < 0:
            return False
        j_pos, _, _, _ = p.getJointState(
            switch_id,
            joint_id,
            physicsClientId=self._physics_client_id)
        # Retrieve the joint limits
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_switch_on(self, switch_id: int, power_on: bool) -> None:
        """If we need to programmatically toggle a switch on/off."""
        joint_id = self._get_joint_id(switch_id, "joint_0")
        if joint_id < 0:
            return
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = j_max if power_on else j_min
        p.resetJointState(switch_id,
                          joint_id,
                          target_val * self.switch_joint_scale,
                          physicsClientId=self._physics_client_id)

    def _is_ball_close_to_target(self, bx: float, by: float,
                                 tx: float, ty: float) -> bool:
        """Check if ball is 'close enough' to the target."""
        dist = np.hypot(bx - tx, by - ty)
        return (dist < 0.05)  # tune threshold

    # -------------------------------------------------------------------------
    # Predicates
    # -------------------------------------------------------------------------
    @staticmethod
    def _FanOn_holds(state: State, objects: Sequence[Object]) -> bool:
        """Example predicate: (FanOn fan)."""
        (fan, ) = objects
        return state.get(fan, "is_on") > 0.5

    def _BallAtTarget_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Example predicate: (BallAtTarget ball target)."""
        ball, target = objects
        return self._is_ball_close_to_target(
            state.get(ball, "x"), state.get(ball, "y"),
            state.get(target, "x"), state.get(target, "y")
        )

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
            # Fans (positions are mostly static in this example, but you can randomize).
            fan_dicts = []
            # Each fan faces toward the ball in the center.
            mid_x = self.x_lb + (self.x_ub - self.x_lb) / 2 
            bottom_fan_y = self.y_lb + 0.14
            top_fan_y = self.y_ub + self.fan_x_len/2
            mid_y = (bottom_fan_y + top_fan_y) / 2
            fan_positions_orientations = [
                (self.x_lb - self.fan_x_len*5, mid_y, 0.0),        # left  fan (faces +X)
                (self.x_ub + self.fan_x_len*5, mid_y, np.pi),      # right fan (faces -X)
                (mid_x, top_fan_y,  -np.pi / 2), # back  fan (faces -Y)
                (mid_x, bottom_fan_y, np.pi / 2), # front fan (faces +Y)
            ]

            fan_dicts = []
            for i, (px, py, rot) in enumerate(fan_positions_orientations):
                fan_dicts.append({
                    "x": px,
                    "y": py,
                    "z": self.table_height + self.fan_z_len / 2,
                    "rot": rot,
                    # Example switch positions near the robot (customize as needed):
                    "switch_x": self.robot_init_x + (i * 0.1) - 0.2,
                    "switch_y": self.y_lb + 0.05,
                    "switch_z": self.table_height,
                    "switch_rot": np.pi / 2,  
                    "is_on": 0.0,  # off by default
                })

            # Walls
            wall1_dict = {
                "x": 0.85,
                "y": 1.38,
                "z": self.table_height,
                "rot": np.pi/2,
                "length": 0.3
            }
            wall2_dict = {
                "x": 0.65,
                "y": 1.42,
                "z": self.table_height,
                "rot": 0.0,
                "length": 0.3
            }
            # Ball
            ball_dict = {
                "x": 0.75,
                "y": 1.35,
                "z": self.table_height + 0.05
            }
            # Target
            target_dict = {
                "x": 0.95,
                "y": 1.45,
                "z": self.table_height,
                "rot": 0.0,
                "is_hit": 0.0,
            }

            init_dict = {
                self._robot: robot_dict,
                self._fan1: fan_dicts[0],
                self._fan2: fan_dicts[1],
                self._fan3: fan_dicts[2],
                self._fan4: fan_dicts[3],
                self._wall1: wall1_dict,
                self._wall2: wall2_dict,
                self._ball: ball_dict,
                self._target: target_dict,
            }
            init_state = utils.create_state_from_dict(init_dict)

            # Example goal: Ball is at the target
            goal_atoms = {GroundAtom(self._BallAtTarget, [self._ball, self._target])}

            tasks.append(EnvironmentTask(init_state, goal_atoms))

        return self._add_pybullet_state_to_tasks(tasks)

if __name__ == "__main__":
    """Run a simple simulation to test the environment."""
    import time

    # Make a task
    CFG.seed = 0
    CFG.pybullet_sim_steps_per_action = 1
    env = PyBulletFanEnv(use_gui=True)
    rng = np.random.default_rng(CFG.seed)
    task = env._make_tasks(1, rng)[0]
    env._reset_state(task.init)

    while True:
        # Robot does nothing
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))

        env.step(action)
        time.sleep(0.01)
