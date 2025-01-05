"""The goal is to turn the lightbulb on.

In the simplest case, the lightbulb is automatically turned on when the
light is connected to both the positive and negative terminals of the
battery. The lightbulb and the battery are fixed, the wire is moveable.

python predicators/main.py --approach oracle --env pybullet_circuit \
--seed 0 --num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --debug
"""

import logging
from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion, Pose
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type, ConceptPredicate


class PyBulletCircuitEnv(PyBulletEnv):
    """A PyBullet environment involving a battery, a light bulb socket, and two
    wire connectors.

    When the battery is connected to the socket via connectors (the
    'Connected' predicate is true), the bulb color changes to yellow,
    indicating it's 'on'.
    """

    # Workspace / table bounds (adjust as you wish).
    connected_angle_tol: ClassVar[float] = 1e-1
    connected_pos_tol: ClassVar[float] = 1e-2
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = 0.2
    z_ub: ClassVar[float] = 0.75
    init_padding = 0.05

    # Table config
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, 0.0)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0., 0., np.pi / 2])

    # Robot config
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2
    max_angular_vel: ClassVar[float] = np.pi / 4

    # Some helpful color specs
    _bulb_on_color: ClassVar[Tuple[float, float, float,
                                   float]] = (1.0, 1.0, 0.0, 1.0)  # yellow
    _bulb_off_color: ClassVar[Tuple[float, float, float,
                                    float]] = (0.8, 0.8, 0.8, 1.0)  # white

    # Connector dimensions
    snap_width: ClassVar[float] = 0.05
    snap_height: ClassVar[float] = 0.05
    wire_snap_length: ClassVar[float] = 0.4
    battery_snap_length: ClassVar[float] = 0.2
    bulb_snap_length: ClassVar[float] = 0.2

    # Camera parameters
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # Types
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _wire_type = Type("wire", ["x", "y", "z", "rot", "is_held"])
    _battery_type = Type("battery", ["x", "y", "z", "rot"])
    _light_type = Type("light", ["x", "y", "z", "rot", "is_on"])

    def __init__(self, use_gui: bool = True) -> None:

        # Objects
        self._robot = Object("robot", self._robot_type)
        self._wire1 = Object("wire1", self._wire_type)
        self._wire2 = Object("wire2", self._wire_type)
        self._battery = Object("battery", self._battery_type)
        self._light = Object("light", self._light_type)

        super().__init__(use_gui)

        # Predicates
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._wire_type],
                                  self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._ConnectedToLight = Predicate("ConnectedToLight",
                                           [self._wire_type, self._light_type],
                                           self._ConnectedToLight_holds)
        self._ConnectedToBattery = Predicate(
            "ConnectedToBattery", [self._wire_type, self._battery_type],
            self._ConnectedToBattery_holds)
        # Ultimatly, we probably want a predicate for Connected(Light, 
        # BatteryPositiveTerminal) and Connected(Light, BatteryNegativeTerminal)
        # which will be evaluated recursively by checking if light is directly
        # connected to the battery or if it is connected to a wire that is 
        # connected to the battery.

        # Normal version used in the simulator
        self._CircuitClosed = Predicate("CircuitClosed", 
                                        [self._light_type, self._battery_type],
                                        self._CircuitClosed_holds)
        # self._CircuitClosed_abs = ConceptPredicate("CircuitClosed",
        #                             [self._wire_type, self._wire_type], 
        #                             self._CircuitClosed_CP_holds)
        self._LightOn = Predicate("LightOn", [self._light_type],
                                  self._LightOn_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_circuit"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            # If you want to define self._Connected, re-add it here
            self._Holding,
            self._HandEmpty,
            self._LightOn,
            self._ConnectedToLight,
            self._ConnectedToBattery,
            self._CircuitClosed,
            # self._CircuitClosed_abs,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # In many tasks, you might want the goal to be that the light is on.
        return {self._LightOn}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type,
            self._wire_type,
            self._battery_type,
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

        # Create the battery
        battery_id = create_object(
            asset_path="urdf/partnet_mobility/switch/102812/"+
                        "battery_switch_snap.urdf",
            physics_client_id=physics_client_id,
            scale=1,
            use_fixed_base=True,
        )
        bodies["battery_id"] = battery_id

        # Create the light socket (with a bulb)
        light_id = create_object(
            asset_path="urdf/bulb_box_snap.urdf",
            physics_client_id=physics_client_id,
            scale=1,
            use_fixed_base=True,
        )
        bodies["light_id"] = light_id

        # Create two wire connectors
        wire_ids = []
        for _ in range(2):
            wire_id = create_object(asset_path="urdf/snap_connector4.urdf",
                                    physics_client_id=physics_client_id,
                                    scale=1)
            wire_ids.append(wire_id)
        bodies["wire_ids"] = wire_ids

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _get_joint_id(obj_id: int, joint_name: str) -> int:
        """Get the joint ID for a joint with a given name.
        """
        num_joints = p.getNumJoints(obj_id)
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(obj_id, joint_index)
            if joint_info[1].decode('utf-8') == joint_name:
                return joint_index

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to PyBullet IDs for environment assets."""
        self._battery.id = pybullet_bodies["battery_id"]
        self._battery.joint_id = self._get_joint_id(self._battery.id, "joint_0")
        self._battery.joint_scale = 0.1
        self._light.id = pybullet_bodies["light_id"]
        self._wire1.id = pybullet_bodies["wire_ids"][0]
        self._wire2.id = pybullet_bodies["wire_ids"][1]

    # -------------------------------------------------------------------------
    # State Management
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return IDs of wires (assuming the robot can pick them up)."""
        return [self._wire1.id, self._wire2.id]

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

        # Battery
        (bx, by, bz), born = p.getBasePositionAndOrientation(
            self._battery.id, physicsClientId=self._physics_client_id)
        # --- CHANGED / ADDED ---
        # Convert orientation to Euler angles and store the yaw in "rot".
        b_euler = p.getEulerFromQuaternion(born)
        state_dict[self._battery] = {
            "x": bx,
            "y": by,
            "z": bz,
            "rot": b_euler[2],  # battery yaw
        }

        # Light (with bulb)
        (lx, ly, lz), lorn = p.getBasePositionAndOrientation(
            self._light.id, physicsClientId=self._physics_client_id)
        # --- CHANGED / ADDED ---
        # Convert orientation to Euler angles and store the yaw in "rot".
        l_euler = p.getEulerFromQuaternion(lorn)
        # We'll store an is_on feature: 1.0 means on, 0.0 means off
        state_dict[self._light] = {
            "x": lx,
            "y": ly,
            "z": lz,
            "rot": l_euler[2],  # light yaw
            "is_on": int(self._is_bulb_on(self._light.id)),
        }

        # Wires
        for wire_obj in [self._wire1, self._wire2]:
            (wx, wy, wz), orn = p.getBasePositionAndOrientation(
                wire_obj.id, physicsClientId=self._physics_client_id)
            is_held_val = 1.0 if wire_obj.id == self._held_obj_id else 0.0
            state_dict[wire_obj] = {
                "x": wx,
                "y": wy,
                "z": wz,
                "rot": p.getEulerFromQuaternion(orn)[2],
                "is_held": is_held_val
            }

        # Convert dictionary to a PyBulletState
        state = utils.create_state_from_dict(state_dict)
        joint_positions = self._pybullet_robot.get_joints()
        pyb_state = utils.PyBulletState(
            state.data, simulator_state={"joint_positions": joint_positions})
        return pyb_state

    def _reset_state(self, state: State) -> None:
        """Reset from a given state."""
        super()._reset_state(state)  # Clears constraints, resets robot
        self._objects = [
            self._robot, self._wire1, self._wire2, self._battery, self._light
        ]

        # Update battery
        bx = state.get(self._battery, "x")
        by = state.get(self._battery, "y")
        bz = state.get(self._battery, "z")
        brot = state.get(self._battery, "rot")
        update_object(self._battery.id,
                      position=(bx, by, bz),
                      orientation=p.getQuaternionFromEuler([0, 0, brot]),
                      physics_client_id=self._physics_client_id)

        # Update light socket
        lx = state.get(self._light, "x")
        ly = state.get(self._light, "y")
        lz = state.get(self._light, "z")
        # --- CHANGED / ADDED ---
        # Retrieve light rot
        lrot = state.get(self._light, "rot")
        update_object(self._light.id,
                      position=(lx, ly, lz),
                      orientation=p.getQuaternionFromEuler([0, 0, lrot]),
                      physics_client_id=self._physics_client_id)
        # Optionally set color here if you want to reflect the on/off state visually.

        # Update wires
        for wire_obj in [self._wire1, self._wire2]:
            wx = state.get(wire_obj, "x")
            wy = state.get(wire_obj, "y")
            wz = state.get(wire_obj, "z")
            rot = state.get(wire_obj, "rot")
            update_object(wire_obj.id,
                          position=(wx, wy, wz),
                          orientation=p.getQuaternionFromEuler([0, 0, rot]),
                          physics_client_id=self._physics_client_id)
            if state.get(wire_obj, "is_held") > 0.5:
                # TODO: create constraint between snap and robot
                self._attach(wire_obj.id, self._pybullet_robot)
                self._held_obj_id = wire_obj.id

        # Check if re-creation matches
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.warning("Could not reconstruct state exactly!")

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Process a single action step.

        If the battery is connected to the light, turn the bulb on.
        """
        next_state = super().step(action, render_obs=render_obs)

        _ = self._SwitchOn_holds(next_state, [self._battery])
        # Check if the CircuitClosed predicate is satisfied => turn the light on
        if self._CircuitClosed_holds(next_state, [self._light, self._battery])\
            and self._SwitchOn_holds(next_state, [self._battery]):
            self._turn_bulb_on()
        else:
            self._turn_bulb_off()

        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    # -------------------------------------------------------------------------
    # Predicates
    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, wire = objects
        return state.get(wire, "is_held") > 0.5

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.2

    @staticmethod
    def _ConnectedToLight_holds(state: State,
                                objects: Sequence[Object]) -> bool:
        (wire, light) = objects

        # Check if the wire is connected to the light based on their poses.
        wx = state.get(wire, "x")
        wy = state.get(wire, "y")
        wr = state.get(wire, "rot")
        lx = state.get(light, "x")
        ly = state.get(light, "y")
        lr = state.get(light, "rot")

        # Should be pi/2 rot apart
        target_angle = np.pi / 2
        min_angle = target_angle - PyBulletCircuitEnv.connected_angle_tol
        max_angle = target_angle + PyBulletCircuitEnv.connected_angle_tol
        angle_diff = abs(wr - lr)
        if angle_diff < min_angle or angle_diff > max_angle:
            return False

        # Correct x and y differences for connection
        target_x_diff = PyBulletCircuitEnv.wire_snap_length / 2 - \
                        PyBulletCircuitEnv.snap_width / 2
        target_y_diff = PyBulletCircuitEnv.bulb_snap_length / 2 + \
                        PyBulletCircuitEnv.snap_width / 2

        # Calculate how much deviation is allowed
        min_x_diff = target_x_diff - PyBulletCircuitEnv.connected_pos_tol
        max_x_diff = target_x_diff + PyBulletCircuitEnv.connected_pos_tol
        min_y_diff = target_y_diff - PyBulletCircuitEnv.connected_pos_tol
        max_y_diff = target_y_diff + PyBulletCircuitEnv.connected_pos_tol

        # Compute actual differences
        x_diff = abs(wx - lx)
        y_diff = abs(wy - ly)

        # Check whether the differences are out of the allowed tolerance
        if (x_diff < min_x_diff or x_diff > max_x_diff or y_diff < min_y_diff
                or y_diff > max_y_diff):
            return False
        return True

    @staticmethod
    def _ConnectedToBattery_holds(state: State,
                                  objects: Sequence[Object]) -> bool:
        return PyBulletCircuitEnv._ConnectedToLight_holds(state, objects)

    @staticmethod
    def _LightOn_holds(state: State, objects: Sequence[Object]) -> bool:
        (light, ) = objects
        return state.get(light, "is_on") > 0.5

    @staticmethod
    def _CircuitClosed_holds(state: State, objects: Sequence[Object]) -> bool:
        """Placeholder logic for checking if circuit is closed."""
        light, battery = objects
        wires = state.get_objects(PyBulletCircuitEnv._wire_type)

        for wire in wires:
            if not PyBulletCircuitEnv._ConnectedToLight_holds(
                    state, [wire, light]):
                return False
            if not PyBulletCircuitEnv._ConnectedToBattery_holds(
                    state, [wire, battery]):
                return False
        return True

    def _SwitchOn_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if the battery is switched on.
        """
        del state  # unused
        battery, = objects
        joint_state = p.getJointState(battery.id, battery.joint_id, 
                                physicsClientId=self._physics_client_id)[0] /\
                                    self._battery.joint_scale
        joint_min = p.getJointInfo(battery.id, battery.joint_id, 
                                physicsClientId=self._physics_client_id)[8]
        joint_max = p.getJointInfo(battery.id, battery.joint_id, 
                                physicsClientId=self._physics_client_id)[9]
        joint_state = np.clip((joint_state - joint_min) / 
                              (joint_max - joint_min), 0, 1)
        return bool(joint_state > 0.5)
    
    @staticmethod
    def _CircuitClosed_CP_holds(atoms: Set[GroundAtom], 
                                objects: Sequence[Object]) -> bool:
        wire1, wire2 = objects
        if wire1 == wire2:
            return False
        # Check both wires are connected to the light and battery
        w1_connected_to_light = False
        w1_connected_to_battery = False
        w2_connected_to_light = False
        w2_connected_to_battery = False
        for atom in atoms:
            if atom.predicate == "ConnectedToLight":
                if atom.args[0] == wire1 and atom.args[1] == objects[1]:
                    w1_connected_to_light = True
                elif atom.args[0] == wire2 and atom.args[1] == objects[1]:
                    w2_connected_to_light = True
            elif atom.predicate == "ConnectedToBattery":
                if atom.args[0] == wire1 and atom.args[1] == objects[0]:
                    w1_connected_to_battery = True
                elif atom.args[0] == wire2 and atom.args[1] == objects[0]:
                    w2_connected_to_battery = True
        return w1_connected_to_light and w1_connected_to_battery and \
                w2_connected_to_light and w2_connected_to_battery 

    # -------------------------------------------------------------------------
    # Turning the bulb on/off visually
    def _turn_bulb_on(self) -> None:
        if self._light.id is not None:
            p.changeVisualShape(
                self._light.id,
                3,  # all link indices
                rgbaColor=self._bulb_on_color,
                physicsClientId=self._physics_client_id)

    def _turn_bulb_off(self) -> None:
        if self._light.id is not None:
            p.changeVisualShape(
                self._light.id,
                3,  # all link indices
                rgbaColor=self._bulb_off_color,
                physicsClientId=self._physics_client_id)
    
    def _is_bulb_on(self, light_id) -> bool:
        """Check if the bulb is on."""
        color = p.getVisualShapeData(light_id, 
                        physicsClientId=self._physics_client_id)[3][-1]
        bulb_color_on_dist = sum(np.subtract(color, self._bulb_on_color)**2)
        bulb_color_off_dist = sum(np.subtract(color, self._bulb_off_color)**2)
        return bulb_color_on_dist < bulb_color_off_dist

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
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }

            # Battery near the lower region
            battery_x = self.x_lb + 2 * self.init_padding
            # For randomization, tweak or keep rot=0.0 as needed
            battery_dict = {
                "x": battery_x,
                "y": 1.3,
                "z": self.z_lb + self.snap_height / 2,
                "rot": np.pi / 2,
            }

            # Wires
            wire1_dict = {
                "x": 0.75,
                "y": 1.15,  # lower region
                "z": self.z_lb + self.snap_height / 2,
                "rot": 0.0,
                "is_held": 0.0,
            }
            wire2_dict = {
                "x": 0.75,
                "y": self.y_ub - self.init_padding * 3,  # upper region
                "z": self.z_lb + self.snap_height / 2,
                "rot": 0.0,
                "is_held": 0.0,
            }

            # Light near upper region
            bulb_x = battery_x + self.wire_snap_length - self.snap_width
            # For randomization, tweak or keep rot=0.0 as needed
            light_dict = {
                "x": bulb_x,
                "y": 1.3,
                "z": self.z_lb + self.snap_height / 2,
                "rot": -np.pi / 2,
                "is_on": 0.0,
            }

            init_dict = {
                self._robot: robot_dict,
                self._battery: battery_dict,
                self._light: light_dict,
                self._wire1: wire1_dict,
                self._wire2: wire2_dict,
            }
            init_state = utils.create_state_from_dict(init_dict)

            # The goal can be that the light is on.
            goal_atoms = {
                # GroundAtom(self._LightOn, [self._light]),
                GroundAtom(self._CircuitClosed, [self._light, self._battery]),
                # GroundAtom(self._ConnectedToBattery, [self._wire1, self._battery]),
            }
            tasks.append(EnvironmentTask(init_state, goal_atoms))

        return self._add_pybullet_state_to_tasks(tasks)

if __name__ == "__main__":
    """Run a simple simulation to test the environment."""
    import time

    # Make a task
    CFG.seed = 0
    env = PyBulletCircuitEnv(use_gui=True)
    task = env._make_tasks(1, CFG.seed)[0]
    env._reset_state(task.init)

    while True:
        # could potentially add a noop action
        robot = env._pybullet_robot
        ee_action = Pose((env.robot_init_x, env.robot_init_y, env.robot_init_z),
                         p.getQuaternionFromEuler([0, 
                                                    env.robot_init_tilt,
                                                    env.robot_init_wrist]))
        action = Action(np.array(robot.inverse_kinematics(ee_action, 
                                        validate=False, 
                                        set_joints=True)))
        env.step(action)
        p.stepSimulation(env._physics_client_id)
        time.sleep(0.01)
