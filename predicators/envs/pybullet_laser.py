"""Laser env

python predicators/main.py --approach oracle --env pybullet_laser \
--seed 0 --num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --debug

python predicators/main.py --approach oracle --env pybullet_laser \
--seed 0 --num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --debug \
--pybullet_ik_validate False --sesame_check_expected_atoms False \
--video_not_break_on_exception
"""
import time
import logging
from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletLaserEnv(PyBulletEnv):
    """A PyBullet environment that simulates a laser station, mirrors, and
    targets on a table.

    Turning on the station emits a laser beam that can reflect off
    mirrors or partially pass through split mirrors, and stops when a
    target is hit.
    """

    _out_of_view_xy: ClassVar[Sequence[float]] = [10.0, 10.0]
    # -------------
    # Table / workspace bounds (adjust as you wish)
    # -------------
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

    # -------------
    # Robot config
    # -------------
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])
    robot_init_tilt: ClassVar[float] = np.pi / 2.0
    robot_init_wrist: ClassVar[float] = -np.pi / 2.0

    # -------------
    # Camera
    # -------------
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.25, 0.42)

    # -------------
    # URDF scale or references
    # -------------
    piece_width: ClassVar[float] = 0.08
    piece_height: ClassVar[float] = 0.11
    light_height: ClassVar[float] = piece_height * 2 / 3
    station_height: ClassVar[float] = piece_height * 3
    station_joint_scale: ClassVar[float] = 0.1
    station_on_threshold: ClassVar[float] = 0.5  # fraction of the joint range
    mirror_rot_offset: ClassVar[float] = -np.pi / 4

    # Laser
    _laser_color: ClassVar[Tuple[float, float, float]] = (1.0, 0.2, 0.2)
    _laser_width: ClassVar[float] = 10
    laser_life_time: ClassVar[float] = 0.5

    # -------------
    num_targets: ClassVar[int] = 2
    num_split_mirrors: ClassVar[int] = 1
    num_standard_mirrors: ClassVar[int] = 2

    # -------------
    # Types
    # -------------
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _station_type = Type("station", ["x", "y", "z", "rot", "is_on"],
                         sim_features=["id", "joint_id"])
    _mirror_type = Type("mirror",
                        ["x", "y", "z", "rot", "split_mirror", "is_held"])
    _target_type = Type("target", ["x", "y", "z", "rot", "is_hit"])

    def __init__(self, use_gui: bool = True) -> None:
        # Create environment objects (logic-level)
        self._robot = Object("robot", self._robot_type)
        self._station = Object("station", self._station_type)

        self._split_mirrors = [Object(f"split_mirror{i}", self._mirror_type)
                                for i in range(self.num_split_mirrors)]
        self._normal_mirrors = [Object(f"mirror{i}", self._mirror_type)
                                for i in range(self.num_standard_mirrors)]
        self._targets = [Object(f"target{i}", self._target_type)
                                for i in range(self.num_targets)]
        self._laser_ids = []

        # Initialize PyBullet
        super().__init__(use_gui=use_gui)

        # Define predicates
        # Example: "StationOn" checks whether the station is toggled on
        self._StationOn = Predicate("StationOn", [self._station_type],
                                    self._StationOn_holds)
        # Perhaps you want a "TargetHit" predicate
        self._TargetHit = Predicate("TargetHit", [self._target_type],
                                    self._TargetHit_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._mirror_type],
                                  self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._IsSplitMirror = Predicate(
            "IsSplitMirror", [self._mirror_type],
            lambda s, o: s.get(o[0], "split_mirror") > 0.5)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_laser"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._StationOn,
            self._TargetHit,
            self._Holding,
            self._HandEmpty,
            self._IsSplitMirror,
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type,
            self._station_type,
            self._mirror_type,
            self._target_type,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # Example: require that at least one target is hit
        return {self._TargetHit}

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

        # Laser station
        station_id = create_object(
            asset_path="urdf/partnet_mobility/switch/102812/" +
            "laser_station_switch.urdf",
            physics_client_id=physics_client_id,
            scale=1.0,
            use_fixed_base=True,
        )
        bodies["station_id"] = station_id

        # Mirrors
        normal_mirror_ids = []
        for _ in range(cls.num_standard_mirrors):
            mirror_id = create_object(
                asset_path="urdf/laser_mirror1.urdf",
                physics_client_id=physics_client_id,
                scale=1.0,
                use_fixed_base=False,
            )
            normal_mirror_ids.append(mirror_id)
        bodies["normal_mirror_ids"] = normal_mirror_ids

        split_mirror_ids = []
        for _ in range(cls.num_split_mirrors):
            mirror_id = create_object(
                asset_path="urdf/laser_mirror2.urdf",
                physics_client_id=physics_client_id,
                scale=1.0,
                use_fixed_base=False,
            )
            split_mirror_ids.append(mirror_id)
        bodies["split_mirror_ids"] = split_mirror_ids

        # Targets
        target_ids = []
        for _ in range(cls.num_targets):
            target_id = create_object(
                asset_path="urdf/laser_target.urdf",
                physics_client_id=physics_client_id,
                scale=1.0,
                use_fixed_base=False,
            )
            target_ids.append(target_id)
        bodies["target_ids"] = target_ids

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
        """Store references to the relevant PyBullet IDs."""
        self._station.id = pybullet_bodies["station_id"]
        self._station.joint_id = self._get_joint_id(self._station.id,
                                                    "joint_0")
        for mirror, mirror_id in zip(self._normal_mirrors, 
                                     pybullet_bodies["normal_mirror_ids"]):
            mirror.id = mirror_id
        for mirror, mirror_id in zip(self._split_mirrors,
                                     pybullet_bodies["split_mirror_ids"]):
            mirror.id = mirror_id
        for target, target_id in zip(self._targets, 
                                     pybullet_bodies["target_ids"]):
            target.id = target_id

    # -------------------------------------------------------------------------
    # State Reading/Writing
    # -------------------------------------------------------------------------
    def _create_task_specific_objects(self, state: State) -> None:
        pass

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return IDs of wires (assuming the robot can pick them up)."""
        return [m.id for m in self._normal_mirrors + self._split_mirrors]

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._station_type:
            if feature == "is_on":
                return 1.0 if self._station_powered_on() else 0.0
        elif obj.type == self._mirror_type:
            if feature == "split_mirror":
                return 1.0 if "split_mirror" in obj.name else 0.0
        elif obj.type == self._target_type:
            if feature == "is_hit":
                return 1.0 if self._is_target_hit(obj) else 0.0
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _reset_custom_env_state(self, state: State) -> None:
        oov_x, oov_y = self._out_of_view_xy

        for beam_id, _ in self._laser_ids:
            p.removeBody(beam_id, physicsClientId=self._physics_client_id)
        self._laser_ids = []

        # Move targets out of view if needed
        target_objs = state.get_objects(self._target_type)
        for i in range(len(target_objs), len(self._targets)):
            update_object(self._targets[i].id, position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)

        # Move split mirrors out of view if needed
        split_mirror_objs = [m for m in state.get_objects(
                                        self._mirror_type)
                                    if state.get(m, "split_mirror") > 0.5]
        for i in range(len(split_mirror_objs), len(self._split_mirrors)):
            update_object(self._split_mirrors[i].id, position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)

        # Move normal mirrors out of view if needed
        normal_mirror_objs = [m for m in state.get_objects(
                                        self._mirror_type)
                                    if state.get(m, "split_mirror") < 0.5]
        for i in range(len(normal_mirror_objs), len(self._normal_mirrors)):
            update_object(self._normal_mirrors[i].id, position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)
        
        switch_on = state.get(self._station, "is_on") > 0.5
        self._set_station_powered_on(switch_on)
    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------
    def step(self, action: Action, render_obs: bool = False) -> State:
        next_state = super().step(action, render_obs=render_obs)

        # After any motion, we simulate the laser
        self._simulate_laser()
        
        lasers_copy = self._laser_ids.copy()
        for beam_id, creation_time in lasers_copy:
            logging.info(f"beam_id: {beam_id}")
            if time.time() - creation_time > self.laser_life_time:
                p.removeBody(beam_id, 
                                 physicsClientId=self._physics_client_id)
                # Remove the beam from the list
                self._laser_ids.remove((beam_id, creation_time))
        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    # -------------------------------------------------------------------------
    # Laser Simulation
    # -------------------------------------------------------------------------
    def _simulate_laser(self) -> None:
        """Fire the laser if station is on, reflecting or splitting at mirrors
        and stopping if it hits a target.

        Updates the 'is_hit' feature on targets. We also draw red debug
        lines to visualize the laser beam.
        """
        # 1) Check if station is on
        if not self._station_powered_on():
            # Clear old hits
            self._clear_target_hits()
            return

        # 2) Build a basic ray from station outward
        station_pos, station_orn = p.getBasePositionAndOrientation(
            self._station.id, self._physics_client_id)
        station_pos = (station_pos[0], station_pos[1],
                       self.table_height + self.light_height)
        # Example beam direction: facing station_orn z-axis
        beam_dir = np.array([0.0, 1.0, 0.0])
        # Rotate beam_dir by the station's orientation
        rmat = np.array(p.getMatrixFromQuaternion(station_orn)).reshape(3, 3)
        beam_dir = rmat.dot(beam_dir)

        # 3) Recursively trace the beam
        start_pt = np.array(station_pos)
        max_depth = 5  # allow up to 5 mirror interactions
        self._clear_target_hits()
        self._trace_beam(start_pt, beam_dir, max_depth)

    def _trace_beam(self, start: np.ndarray, direction: np.ndarray,
                    depth: int):
        """Recursively move a line forward until it hits a mirror or target."""
        if depth <= 0:
            return

        # Cast a ray forward
        ray_len = 2.0  # you can adjust
        end_pt = start + direction * ray_len
        hits = p.rayTest(list(start),
                         list(end_pt),
                         physicsClientId=self._physics_client_id)
        # hits is a list, but for a single rayTest() there's typically 1 item.

        best_hit = None
        best_fraction = 1.1
        for h in hits:
            object_id = h[0]  # hitObjectUniqueId
            link_index = h[1]  # hitLinkIndex
            hit_fraction = h[2]  # fraction along the ray
            hit_position = h[3]  # (x, y, z) of the collision
            hit_normal = h[4]  # normal at collision

            # Check for a valid object and whether this hit is closer
            if object_id >= 0 and hit_fraction < best_fraction:
                best_hit = h
                best_fraction = hit_fraction

        if not best_hit:
            # No intersection => beam goes off into nowhere.
            # Draw a debug line all the way to end_pt.
            if CFG.laser_use_debug_line_for_beams:
                p.addUserDebugLine(
                    lineFromXYZ=start.tolist(),
                    lineToXYZ=end_pt.tolist(),
                    lineColorRGB=self._laser_color,  # red
                    lineWidth=self._laser_width,
                    lifeTime=self.laser_life_time,  # short lifetime so each step refreshes
                )
            else:
                laser_id = create_laser_cylinder(
                        start.tolist(), 
                        end_pt.tolist(),
                    )
                self._laser_ids.append((laser_id, time.time()))
            return

        # Unpack the best hit
        hit_id = best_hit[0]
        hit_fraction = best_hit[2]
        hit_point = np.array(best_hit[3])  # 3D position

        # Draw a debug line from start up to the hit point
        if CFG.laser_use_debug_line_for_beams:
            p.addUserDebugLine(
                lineFromXYZ=start.tolist(),
                lineToXYZ=hit_point.tolist(),
                lineColorRGB=self._laser_color,
                lineWidth=self._laser_width,
                lifeTime=self.laser_life_time,
            )
        else:
            laser_id = create_laser_cylinder(
                    start.tolist(), 
                    hit_point.tolist(),
                )
            self._laser_ids.append((laser_id, time.time()))

        # Check if it's a target
        for target in self._targets:
            if hit_id == target.id:
                self._set_target_hit(target, True)
                # Laser stops here
                return

        for mirror in self._normal_mirrors + self._split_mirrors:
            if hit_id == mirror.id:
                is_split = "split" in mirror.name
                if is_split:
                    # 1) Reflect path
                    reflect_dir = self._mirror_reflection(hit_id, direction)
                    self._trace_beam(hit_point + reflect_dir * 1e-3, reflect_dir,
                                     depth - 1)
                    # 2) Pass-through path
                    pass_dir = direction
                    self._trace_beam(hit_point + pass_dir * 1e-3, pass_dir,
                                     depth - 1)
                else:
                    # Normal mirror => reflect only
                    reflect_dir = self._mirror_reflection(hit_id, direction)
                    self._trace_beam(hit_point + reflect_dir * 1e-3, 
                                     reflect_dir,
                                     depth - 1)
        # Otherwise, it might have hit the station/table => stop
        return

    def _mirror_reflection(self, mirror_id: int,
                           incoming_dir: np.ndarray) -> np.ndarray:
        """Compute the approximate reflection of the incoming beam on a
        mirror's orientation."""
        # For simplicity, reflect across the mirror's local y-axis.
        # In a real environment you’d do actual local normal calculations.
        pos, orn = p.getBasePositionAndOrientation(mirror_id,
                                                   self._physics_client_id)
        # Convert the quaternion to Euler angles
        euler = p.getEulerFromQuaternion(orn)
        euler = list(euler)
        euler[2] -= np.pi / 4
        orn = p.getQuaternionFromEuler(euler)
        rmat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        # Suppose the mirror's local normal is the x-axis in URDF => mirror
        # reflection around that.
        local_normal = rmat[:, 0]  # pick an axis consistent with mirror shape
        incoming_norm = incoming_dir / (np.linalg.norm(incoming_dir) + 1e-9)
        # reflection = dir - 2*(dir · normal)*normal
        if CFG.laser_zero_reflection_angle:
            reflect = local_normal
        else:
            reflect = incoming_norm - 2 * (incoming_norm @ local_normal) * \
                    local_normal
        return reflect / (np.linalg.norm(reflect) + 1e-9)

    def _clear_target_hits(self):
        """Set all targets to not hit."""
        for target in self._targets:
            self._set_target_hit(target, False)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _station_powered_on(self) -> bool:
        """Check if station's switch is above threshold."""
        if not hasattr(self._station, "joint_id"):
            return False
        j_pos, _, _, _ = p.getJointState(
            self._station.id,
            self._station.joint_id,
            physicsClientId=self._physics_client_id)
        # get the joint limits
        info = p.getJointInfo(self._station.id,
                              self._station.joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        # Convert to fraction
        frac = (j_pos / self.station_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.station_on_threshold)

    def _set_station_powered_on(self, power_on: bool) -> None:
        """If you need to programmatically turn the station on/off."""
        if not hasattr(self._station, "joint_id"):
            return
        info = p.getJointInfo(self._station.id,
                              self._station.joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        mid_val = 0.5 * (j_min + j_max)
        target_val = j_max if power_on else j_min
        p.resetJointState(self._station.id,
                          self._station.joint_id,
                          target_val * self.station_joint_scale,
                          physicsClientId=self._physics_client_id)

    def _is_target_hit(self, target_obj: Object) -> bool:
        return False  # By default, determined after `_simulate_laser()`

    def _set_target_hit(self, target_obj: Object, val: bool) -> None:
        """If you want to show visual changes on the target, do that here."""
        pass  # e.g., change color if needed

    # -------------------------------------------------------------------------
    # Predicates
    # -------------------------------------------------------------------------
    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, wire = objects
        return state.get(wire, "is_held") > 0.5

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.03

    @staticmethod
    def _StationOn_holds(state: State, objects: Sequence[Object]) -> bool:
        (station, ) = objects
        return state.get(station, "is_on") > 0.5

    @staticmethod
    def _TargetHit_holds(state: State, objects: Sequence[Object]) -> bool:
        (target, ) = objects
        return state.get(target, "is_hit") > 0.5

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
        for task_idx in range(num_tasks):
            num_targets = 0
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }

            # Example layout: station near bottom, mirrors in middle, targets
            # near top
            station_x = (self.x_lb + self.x_ub) / 2
            station_y = self.y_lb + 2 * self.piece_width
            station_dict = {
                "x": station_x,
                "y": station_y,
                "z": self.table_height,
                "rot": 0,
                "is_on": 0.0,  # off initially
            }
            sm_x = station_x
            sm_y = station_y + 2 * self.piece_width
            split_mirror_dict = {
                "x": sm_x - 2 * self.piece_width,  # for demo
                "y": sm_y,
                "z": self.table_height,
                "rot": 0.0,
                "split_mirror": 1.0,
                "is_held": 0.0,
            }

            m1_x = sm_x + 2 * self.piece_width
            m1_y = sm_y
            mirror1_dict = {
                "x": m1_x,
                "y": m1_y,
                "z": self.table_height,
                "rot": 0.0,
                "split_mirror": 0.0,
                "is_held": 0.0,
            }

            t1_x = sm_x
            t1_y = sm_y + 2 * self.piece_width
            target1_dict = {
                "x": t1_x,
                "y": t1_y,
                "z": self.table_height,
                "rot": 0.0,
                "is_hit": 0.0,
            }
            num_targets += 1

            if task_idx == 1:
                m2_x = m1_x
                m2_y = m1_y + 2 * self.piece_width
                mirror2_dict = {
                    "x": m2_x,
                    "y": m2_y,
                    "z": self.table_height,
                    "rot": 0.0,
                    "split_mirror": 0.0,
                    "is_held": 0.0,
                }

                t2_x = m2_x + 2 * self.piece_width
                t2_y = m2_y
                target2_dict = {
                    "x": t2_x,
                    "y": t2_y,
                    "z": self.table_height,
                    "rot": 0.0,
                    "is_hit": 0.0,
                }
                num_targets += 1
            else:
                mirror2_dict = {
                    "x": m1_x,
                    "y": station_y,
                    "z": self.table_height,
                    "rot": 0.0,
                    "split_mirror": 0.0,
                    "is_held": 0.0,
                }
                # t2_x = m1_x
                # t2_y = m1_y + 2 * self.piece_width
                # target2_dict = {
                #     "x": t2_x,
                #     "y": t2_y,
                #     "z": self.table_height,
                #     "rot": 0.0,
                #     "is_hit": 0.0,
                # }

            init_dict = {
                self._robot: robot_dict,
                self._station: station_dict,
                self._normal_mirrors[0]: mirror1_dict,
                self._normal_mirrors[1]: mirror2_dict,
                self._split_mirrors[0]: split_mirror_dict,
                self._targets[0]: target1_dict,
                # self._targets[1]: target2_dict,
            }
            if task_idx == 1:
                init_dict[self._targets[1]] = target2_dict
            init_state = utils.create_state_from_dict(init_dict)

            goal_atoms = {
                *[GroundAtom(self._TargetHit, [self._targets[tid]]) for tid in 
                  range(num_targets)],
                GroundAtom(self._StationOn, [self._station]),
            }

            tasks.append(EnvironmentTask(init_state, goal_atoms))

        return self._add_pybullet_state_to_tasks(tasks)

def create_laser_cylinder(start, end, color=(1, 0, 0, 1),
                          radius=0.001) -> int:
    """Create a thin cylinder from start -> end, visible in getCameraImage."""
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    seg = end - start
    length = np.linalg.norm(seg)

    # Midpoint of the segment
    mid = (start + end) / 2.0

    # Direction (normalized)
    direction = seg / length

    # Cylinder in PyBullet is aligned along the local Z axis by default,
    # so we need a rotation that takes the Z-axis to "direction".
    # One way is to compute a quaternion that aligns z=[0,0,1] to `direction`.
    z_axis = np.array([0, 0, 1], dtype=float)
    # Cross product for the rotation axis
    rot_axis = np.cross(z_axis, direction)
    rot_axis_len = np.linalg.norm(rot_axis)
    if rot_axis_len < 1e-12:
        # direction is basically [0,0,1] or [0,0,-1]
        # handle it with a simple condition
        if direction[2] < 0:
            # 180-degree flip around X (or any perpendicular axis)
            orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        else:
            # no rotation needed
            orientation = [0, 0, 0, 1]
    else:
        rot_axis = rot_axis / rot_axis_len
        # Angle between z-axis and our vector
        angle = np.arccos(np.dot(z_axis, direction))
        orientation = p.getQuaternionFromAxisAngle(rot_axis, angle)

    # Create a visual shape for the cylinder
    vis_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=color,  # e.g. (1,0,0,1) for red
    )

    # Create a collision shape if you want it to be collidable
    # or -1 if you want no collisions:
    col_id = -1

    # Create the actual body
    body_id = p.createMultiBody(
        baseMass=0,                   # mass=0 => static object
        baseInertialFramePosition=[0,0,0],
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=mid.tolist(),
        baseOrientation=orientation,
    )

    # If you want this beam to vanish after `lifetime` seconds,
    # you can schedule a removal in your main loop, or store
    # (body_id, time_created) and remove once enough time passes:
    #   p.removeBody(body_id)

    logging.debug(f"created laser beam {body_id}")
    return body_id

if __name__ == "__main__":
    """Run a simple simulation to test the environment."""
    import time

    # Make a task
    CFG.seed = 0
    CFG.env = "pybullet_laser"
    env = PyBulletLaserEnv(use_gui=True)
    task = env._make_tasks(1, CFG.seed)[0]
    env._reset_state(task.init)

    while True:
        # Robot does nothing
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))

        env.step(action)
        time.sleep(0.01)
