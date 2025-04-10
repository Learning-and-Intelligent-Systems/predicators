"""A PyBullet version of CoffeeEnv. python predicators/main.py --env
pybullet_coffee --approach oracle --seed 0 \

--coffee_rotated_jug_ratio 0.5 \
--sesame_check_expected_atoms False --coffee_jug_pickable_pred True \
--pybullet_control_mode "reset" --coffee_twist_sampler False

To generate video demos:
python predicators/main.py --env pybullet_coffee --approach oracle --seed 0 \
--coffee_rotated_jug_ratio 0.5 \
--sesame_check_expected_atoms False --coffee_jug_pickable_pred True \
--pybullet_control_mode "reset" --coffee_twist_sampler False \
--make_test_videos --num_test_tasks 1 --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900

Needs pluged in:
python predicators/main.py --env pybullet_coffee --approach oracle --seed 0 \
--coffee_rotated_jug_ratio 0.5 \
--sesame_check_expected_atoms False --coffee_jug_pickable_pred True \
--coffee_twist_sampler False \
--make_test_videos --num_test_tasks 1 --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 \
--coffee_render_grid_world False --coffee_simple_tasks True \
--coffee_machine_have_light_bar False \
--coffee_move_back_after_place_and_push True \
--coffee_machine_has_plug True --sesame_max_skeletons_optimized 1 \
--make_failure_videos \
--debug --option_model_terminate_on_repeat False \
--coffee_use_pixelated_jug True

With the simplified tasks, both pixelated jug and old jug should work.
With the full tasks, the old jug should work.
"""
import logging
import random
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple
import copy

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.coffee import CoffeeEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, Object, \
    Predicate, State, Observation

class PyBulletCoffeeEnv(PyBulletEnv, CoffeeEnv):
    """PyBullet Coffee domain.

    x: cup <-> jug,
    y: robot <-> machine
    z: up <-> down
    """

    # Need to override a number of settings to conform to the actual dimensions
    # of the robots, table, etc.
    grasp_finger_tol: ClassVar[float] = 1e-2
    grasp_position_tol: ClassVar[float] = 1e-2
    dispense_tol: ClassVar[float] = 1e-2
    plugged_in_tol: ClassVar[float] = 1e-2
    pour_angle_tol: ClassVar[float] = 1e-1
    pour_pos_tol: ClassVar[float] = 1.0
    init_padding: ClassVar[float] = 0.05
    pick_jug_y_padding: ClassVar[float] = 0.05
    pick_jug_rot_tol: ClassVar[float] = 0.1
    safe_z_tol: ClassVar[float] = 1e-2
    place_jug_in_machine_tol: ClassVar[float] = 1e-3 / 2
    jug_twist_offset: ClassVar[float] = 0.025
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = 0.2
    z_ub: ClassVar[float] = 0.75
    robot_init_x: ClassVar[float] = (x_ub + x_lb) / 2.0
    robot_init_y: ClassVar[float] = (y_ub + y_lb) / 2.0
    # robot_rest_y: ClassVar[float] = ((y_ub + y_lb) / 2.0) - 0.1
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2
    tilt_lb: ClassVar[float] = robot_init_tilt
    tilt_ub: ClassVar[float] = tilt_lb - np.pi / 4
    # Machine settings.
    machine_x_len: ClassVar[float] = 0.2 * (x_ub - x_lb)
    machine_y_len: ClassVar[float] = 0.15 * (y_ub - y_lb)
    machine_z_len: ClassVar[float] = 0.5 * (z_ub - z_lb)
    machine_top_y_len: ClassVar[float] = 1.3 * machine_y_len
    machine_x: ClassVar[float] = x_ub - machine_x_len / 2 - init_padding
    machine_y: ClassVar[float] = y_ub - machine_y_len / 2 - init_padding
    button_radius: ClassVar[float] = 0.6 * machine_y_len
    button_height = button_radius / 4
    button_x: ClassVar[float] = machine_x
    button_y: ClassVar[float] =\
        machine_y - machine_y_len / 2 - machine_top_y_len - button_height/2
    button_z: ClassVar[float] = z_lb + machine_z_len - button_radius
    button_press_threshold: ClassVar[float] = 1e-2
    machine_color: ClassVar[Tuple[float, float, float, float]] =\
        (0.1, 0.1, 0.1, 1) # Black
    button_color_on: ClassVar[Tuple[float, float, float,
                                    float]] = (0.2, 0.5, 0.2, 1.0)
    plate_color_on: ClassVar[Tuple[float, float, float, float]] = machine_color
    button_color_off: ClassVar[Tuple[float, float, float,
                                     float]] = (0.5, 0.2, 0.2, 1.0)
    button_color_power_off: ClassVar[Tuple[float, float, float,
                                           float]] = (.25, .25, .25, 1.0)
    plate_color_off: ClassVar[Tuple[float, float, float,
                                    float]] = machine_color
    # Jug setting
    jug_radius: ClassVar[float] = 0.3 * machine_y_len
    jug_old_height: ClassVar[float] = 0.19 * (z_ub - z_lb)  # kettle urdf
    jug_new_height: ClassVar[float] = 0.12  #0.1 * (z_ub - z_lb)  # new cup

    @classmethod
    def jug_height(cls) -> float:
        """use class method to allow for dynamic changes."""
        if CFG.coffee_use_pixelated_jug:
            return cls.jug_new_height
        return cls.jug_old_height

    jug_init_x_lb: ClassVar[
        float] = machine_x - machine_x_len / 2 + init_padding
    jug_init_x_ub: ClassVar[
        float] = machine_x + machine_x_len / 2 - init_padding
    jug_init_y_lb: ClassVar[float] = y_lb + 3 * jug_radius + init_padding
    jug_init_y_ub: ClassVar[
        float] = machine_y - machine_y_len - 4 * jug_radius - init_padding
    jug_init_y_ub_og: ClassVar[
        float] = machine_y - machine_y_len - 3 * jug_radius - init_padding
    jug_handle_offset: ClassVar[float] = 3 * jug_radius  # kettle urdf
    jug_old_handle_height: ClassVar[float] = jug_old_height  # old kettle
    jug_new_handle_height: ClassVar[float] = 0.1  # new jug

    @classmethod
    def jug_handle_height(cls) -> float:
        """use class method to allow for dynamic changes."""
        if CFG.coffee_use_pixelated_jug:
            return cls.jug_new_handle_height
        return cls.jug_old_handle_height

    jug_init_rot_lb: ClassVar[float] = -2 * np.pi / 3
    jug_init_rot_ub: ClassVar[float] = 2 * np.pi / 3
    # jug_color: ClassVar[Tuple[float, float, float, float]] =\
    #     (0.5,1,0,0.5) # Green
    jug_color: ClassVar[Tuple[float, float, float, float]] =\
                (1,1,1,0.5) # White
    # Dispense area settings.
    dispense_area_x: ClassVar[float] = machine_x
    dispense_area_y: ClassVar[float] = machine_y - 5 * jug_radius
    dispense_radius = 2 * jug_radius
    dispense_height = 0.0001
    # Cup settings.
    cup_radius: ClassVar[float] = 0.6 * jug_radius
    cup_init_x_lb: ClassVar[float] = x_lb + cup_radius + init_padding
    cup_init_x_ub: ClassVar[
        float] = machine_x - machine_x_len / 2 - cup_radius - init_padding
    cup_init_y_lb: ClassVar[float] = jug_init_y_lb + init_padding
    cup_init_y_ub: ClassVar[float] = jug_init_y_ub_og
    cup_capacity_lb: ClassVar[float] = 0.075 * (z_ub - z_lb)
    cup_capacity_ub: ClassVar[float] = 0.15 * (z_ub - z_lb)
    cup_target_frac: ClassVar[float] = 0.75  # fraction of the capacity
    cup_colors: ClassVar[List[Tuple[float, float, float, float]]] = [
        (244 / 255, 27 / 255, 63 / 255, 1.),
        (121 / 255, 37 / 255, 117 / 255, 1.),
        (35 / 255, 100 / 255, 54 / 255, 1.),
    ]
    # Powercord / Plug settings.
    num_cord_links = 10
    cord_link_length = 0.02
    cord_segment_gap = 0.00
    cord_start_x = machine_x - machine_x_len / 2 - 4 * cord_link_length
    cord_start_y = machine_y - machine_y_len
    cord_start_z = z_lb + cord_link_length / 2
    plug_x = cord_start_x - (num_cord_links - 1) * cord_link_length -\
             cord_segment_gap * (num_cord_links - 1)
    plug_y = cord_start_y
    plug_z = cord_start_z
    # Socket settings.
    socket_height: ClassVar[float] = 0.1
    socket_width: ClassVar[float] = 0.05
    socket_depth: ClassVar[float] = 0.01
    socket_x: ClassVar[float] = (x_lb + x_ub) / 2
    socket_y: ClassVar[float] = machine_y
    socket_z: ClassVar[float] = z_lb + socket_height * 2
    # Pour settings.
    pour_x_offset: ClassVar[float] = cup_radius
    pour_y_offset: ClassVar[float] = -3 * (cup_radius + jug_radius)
    # pour_z_offset: ClassVar[float] = 2.5 * (cup_capacity_ub + \
    #                                  jug_old_height - jug_old_handle_height)
    @classmethod
    def pour_z_offset(cls) -> float:
        return 2.5 * (cls.cup_capacity_ub + cls.jug_height() -\
                      cls.jug_handle_height())

    pour_velocity: ClassVar[float] = cup_capacity_ub / 10.0
    # Table settings.
    table_pose: ClassVar[Pose3D] = (0.75, 1.35, 0.0)
    table_orientation: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    # Camera font view parameters.
    _camera_distance: ClassVar[float]
    _camera_fov: ClassVar[float]
    _camera_yaw: ClassVar[float]
    _camera_pitch: ClassVar[float]
    _camera_target: ClassVar[Pose3D]

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # if CFG.coffee_render_grid_world:
        #     # Camera parameters for grid world
        #     PyBulletCoffeeEnv._camera_distance = 3
        #     PyBulletCoffeeEnv._camera_fov = 8
        #     PyBulletCoffeeEnv._camera_yaw = 90
        #     PyBulletCoffeeEnv._camera_pitch = 0  # lower
        #     PyBulletCoffeeEnv._camera_target = (0.75, 1.33, 0.3)
        # else:
        #     # Camera parameters -- standard
        #     PyBulletCoffeeEnv._camera_distance = 1.3
        #     if CFG.coffee_machine_has_plug:
        #         PyBulletCoffeeEnv._camera_yaw = -60
        #         # self._camera_yaw: ClassVar[float] = -90
        #         # self._camera_yaw: ClassVar[float] = -180
        #     else:
        #         PyBulletCoffeeEnv._camera_yaw = 70
        #     PyBulletCoffeeEnv._camera_pitch = -38  # lower
        #     # PyBulletCoffeeEnv._camera_pitch = 0  # even lower
        #     PyBulletCoffeeEnv._camera_target = (0.75, 1.25, 0.42)
        # if CFG.pybullet_coffee_update_camera:
            # self.update_camera(CFG.pybullet_coffee_camera_distance, 
            #                    CFG.pybullet_coffee_camera_yaw, 
            #                    CFG.pybullet_coffee_camera_pitch, 
            #                    CFG.pybullet_coffee_camera_target)
        PyBulletCoffeeEnv._camera_fov = 8
        PyBulletCoffeeEnv._camera_distance = CFG.pybullet_coffee_camera_distance
        PyBulletCoffeeEnv._camera_yaw = CFG.pybullet_coffee_camera_yaw
        PyBulletCoffeeEnv._camera_pitch = CFG.pybullet_coffee_camera_pitch
        PyBulletCoffeeEnv._camera_target = CFG.pybullet_coffee_camera_target
        self.update_camera(CFG.pybullet_coffee_camera_distance, 
                            CFG.pybullet_coffee_camera_yaw, 
                            CFG.pybullet_coffee_camera_pitch, 
                               CFG.pybullet_coffee_camera_target)

        # Create the cups lazily because they can change size and color.
        self._cup_id_to_cup: Dict[int, Object] = {}
        self._cup_to_liquid_id: Dict[Object, Optional[int]] = {}
        self._cup_to_capacity: Dict[Object, float] = {}
        # The status of the jug is not modeled inside PyBullet.
        self._jug_filled = False
        self._jug_liquid_id: Optional[int] = None
        self._obj_id_to_obj: Dict[int, Object] = {}

        self._cord_ids: Optional[List[int]] = None
        self._plug_id: Optional[int] = None
        self._machine_plugged_in_id: Optional[int] = None

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._CupFilled,
            self._JugInMachine,
            self._Holding,
            self._MachineOn,
            self._OnTable,
            self._HandEmpty,
            self._JugFilled,
            self._RobotAboveCup,
            self._JugAboveCup,
            self._NotAboveCup,
            self._PressingButton,
            self._Twisting,
            self._NotSameCup,
            self._JugPickable,
            self._PluggedIn,
        }
    
    @property
    def agent_goal_predicates(self) -> Set[Predicate]:
        return self.goal_predicates

    @property
    def oracle_proposed_predicates(self) -> Set[Predicate]:
        """Return the predicates that the oracle can propose."""
        # Useful predicates when
        return {
            # Precondition to actions
            self._CupFilled,  # goal predicate
            self._Holding,  # Pour, Place # yes
            self._JugInMachine,  # TurnMachineOn # yes
            self._JugPickable,  # PickJug
            self._JugFilled,  # Pour,
            self._OnTable,  # Pick,
            self._MachineOn,  # Not needed in syPred's success # yes
            self._HandEmpty,  # Not needed in syPred's success; Pick # yes
        }

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle coffee-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        cls._add_pybullet_debug_lines(physics_client_id)

        table_id = cls._add_pybullet_table(physics_client_id)
        bodies["table_id"] = table_id

        machine_id = cls._add_pybullet_coffee_machine(physics_client_id)
        bodies["machine_id"] = machine_id

        dispense_area_id = cls._add_pybullet_dispense_area(physics_client_id)
        bodies["dispense_area_id"] = dispense_area_id

        button_id = cls._add_pybullet_machine_button(physics_client_id)
        bodies["button_id"] = button_id

        jug_id = cls._add_pybullet_jug(physics_client_id)
        bodies["jug_id"] = jug_id

        if CFG.coffee_machine_has_plug:
            cord_ids = cls._add_pybullet_cord(physics_client_id)
            bodies["cord_ids"] = cord_ids
            bodies["plug_id"] = cord_ids[-1]

            socket_id = cls._add_pybullet_socket(physics_client_id)
            bodies["socket_id"] = socket_id

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._jug_id = pybullet_bodies["jug_id"]
        self._machine_id = pybullet_bodies["machine_id"]
        self._dispense_area_id = pybullet_bodies["dispense_area_id"]
        self._button_id = pybullet_bodies["button_id"]
        if CFG.coffee_machine_has_plug:
            self._cord_ids = pybullet_bodies["cord_ids"]
            self._plug_id = pybullet_bodies["plug_id"]
            self._socket_id = pybullet_bodies["socket_id"]

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((cls.robot_init_x, cls.robot_init_y, cls.robot_init_z),
                       robot_ee_orn)
        base_pose = Pose(cls.robot_base_pos, cls.robot_base_orn)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home,
                                                base_pose)

    def _extract_robot_state(self, state: State) -> Array:
        qx, qy, qz, qw = self._state_to_gripper_orn(state)
        f = state.get(self._robot, "fingers")
        f = self.fingers_state_to_joint(self._pybullet_robot, f)
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        z = state.get(self._robot, "z")
        return np.array([x, y, z, qx, qy, qz, qw, f], dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_coffee"

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle coffee-specific resetting."""
        super()._reset_state(state)

        # Remove the old cups.
        for old_cup_id in self._cup_id_to_cup:
            p.removeBody(old_cup_id, physicsClientId=self._physics_client_id)
        self._obj_id_to_obj = {}
        self._obj_id_to_obj[self._pybullet_robot.robot_id] = self._robot
        self._obj_id_to_obj[self._table_id] = self._table
        self._obj_id_to_obj[self._jug_id] = self._jug
        self._obj_id_to_obj[self._machine_id] = self._machine

        # Reset cups based on the state.
        cup_objs = state.get_objects(self._cup_type)
        # Make new cups.
        self._cup_id_to_cup = {}
        self._cup_to_capacity = {}
        for _, cup_obj in enumerate(cup_objs):
            cup_cap = state.get(cup_obj, "capacity_liquid")
            cup_height = cup_cap
            cx = state.get(cup_obj, "x")
            cy = state.get(cup_obj, "y")
            cz = self.z_lb + cup_height / 2
            global_scale = 0.5 * cup_cap / self.cup_capacity_ub
            self._cup_to_capacity[cup_obj] = cup_cap

            cup_id = p.loadURDF(utils.get_env_asset_path("urdf/cup.urdf"),
                                useFixedBase=True,
                                globalScaling=global_scale,
                                physicsClientId=self._physics_client_id)
            # Rotate so handles face robot.
            cup_orn = p.getQuaternionFromEuler([np.pi, np.pi, 0.0])
            p.resetBasePositionAndOrientation(
                cup_id, (cx, cy, cz),
                cup_orn,
                physicsClientId=self._physics_client_id)

            # Create the visual_shape.
            # color = self.cup_colors[cup_idx % len(self.cup_colors)]
            color = random.choice(self.cup_colors)
            p.changeVisualShape(cup_id,
                                -1,
                                rgbaColor=color,
                                physicsClientId=self._physics_client_id)

            self._cup_id_to_cup[cup_id] = cup_obj
            self._obj_id_to_obj[cup_id] = cup_obj

        # Create liquid in cups.
        for liquid_id in self._cup_to_liquid_id.values():
            if liquid_id is not None:
                p.removeBody(liquid_id,
                             physicsClientId=self._physics_client_id)
        self._cup_to_liquid_id.clear()

        for cup in state.get_objects(self._cup_type):
            liquid_id = self._create_pybullet_liquid_for_cup(cup, state)
            self._cup_to_liquid_id[cup] = liquid_id

        # reset the empty jug
        p.changeVisualShape(self._jug_id,
                            0,
                            rgbaColor=self.jug_color,
                            physicsClientId=self._physics_client_id)
        self._jug_filled = bool(state.get(self._jug, "is_filled") > 0.5)
        if self._jug_liquid_id is not None:
            p.removeBody(self._jug_liquid_id,
                         physicsClientId=self._physics_client_id)
            self._jug_liquid_id = None
            if self._jug_filled:
                self._jug_liquid_id = self._create_pybullet_liquid_for_jug()

        # NOTE: if the jug is held, the parent class should take care of it.
        if not self._Holding_holds(state, [self._robot, self._jug]):
            assert self._held_obj_to_base_link is None
            jx = state.get(self._jug, "x")
            jy = state.get(self._jug, "y")
            jz = self._get_jug_z(state, self._jug) + self.jug_height() / 2
            rot = state.get(self._jug, "rot")
            jug_orientation = p.getQuaternionFromEuler(
                [0.0, 0.0, rot - np.pi / 2])
            p.resetBasePositionAndOrientation(
                self._jug_id, [jx, jy, jz],
                jug_orientation,
                physicsClientId=self._physics_client_id)

        # Update the button color.
        if self._MachineOn_holds(state, [self._machine]) and \
            self._JugInMachine_holds(state, [self._jug, self._machine]):
            button_color = self.button_color_on
            plate_color = self.plate_color_on
        else:
            if CFG.coffee_machine_has_plug and \
                self._PluggedIn_holds(state, [self._plug]):
                button_color = self.button_color_off
                plate_color = self.plate_color_off
            else:
                button_color = self.button_color_power_off
                plate_color = self.plate_color_off

        p.changeVisualShape(self._button_id,
                            -1,
                            rgbaColor=button_color,
                            physicsClientId=self._physics_client_id)
        p.changeVisualShape(self._button_id,
                            0,
                            rgbaColor=button_color,
                            physicsClientId=self._physics_client_id)
        p.changeVisualShape(self._dispense_area_id,
                            -1,
                            rgbaColor=plate_color,
                            physicsClientId=self._physics_client_id)

        # Reset plug
        if CFG.coffee_machine_has_plug:
            if self._cord_ids is not None:
                for obj_id in self._cord_ids:
                    p.removeBody(obj_id, self._physics_client_id)
            if self._machine_plugged_in_id is not None:
                p.removeConstraint(self._machine_plugged_in_id,
                                   physicsClientId=self._physics_client_id)
                self._machine_plugged_in_id = None
            self._cord_ids = self._add_pybullet_cord(self._physics_client_id)
            self._plug_id = self._cord_ids[-1]

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.debug("Desired state:")
            logging.debug(state.pretty_str())
            logging.debug("Reconstructed state:")
            logging.debug(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

    def _get_state(self, render_obs: bool = False) -> State:
        """Create a State instance based on the current PyBullet state."""
        state_dict = {}

        # Get robot state.
        rx, ry, rz, qx, qy, qz, qw, rf = self._pybullet_robot.get_state()
        tilt, wrist = self._gripper_orn_to_tilt_wrist((qx, qy, qz, qw))
        fingers = self._fingers_joint_to_state(self._pybullet_robot, rf)
        state_dict[self._robot] = {
            "x": rx,
            "y": ry,
            "z": rz,
            "tilt": tilt,
            "wrist": utils.wrap_angle(wrist),
            "fingers": fingers
        }
        joint_positions = self._pybullet_robot.get_joints()

        # Get cup states.
        for cup_id, cup in self._cup_id_to_cup.items():

            (x, y, z), _ = p.getBasePositionAndOrientation(
                cup_id, physicsClientId=self._physics_client_id)

            capacity = self._cup_to_capacity[cup]
            target_liquid = capacity * self.cup_target_frac

            # No liquid object is created if the current liquid is 0.
            if self._cup_to_liquid_id.get(cup, None) is not None:
                liquid_id = self._cup_to_liquid_id[cup]
                liquid_height = p.getVisualShapeData(
                    liquid_id,
                    physicsClientId=self._physics_client_id,
                )[0][3][0]
                current_liquid = self._cup_liquid_height_to_liquid(
                    liquid_height, capacity)
            else:
                current_liquid = 0.0

            state_dict[cup] = {
                "x": x,
                "y": y,
                "z": z,
                "capacity_liquid": capacity,
                "target_liquid": target_liquid,
                "current_liquid": current_liquid,
            }

        # Get jug state.
        (x, y, z), quat = p.getBasePositionAndOrientation(
            self._jug_id, physicsClientId=self._physics_client_id)
        rot = utils.wrap_angle(p.getEulerFromQuaternion(quat)[2] + np.pi / 2)
        # rot = p.getEulerFromQuaternion(quat)[2] + np.pi/2
        held = (self._jug_id == self._held_obj_id)
        filled = float(self._jug_filled)
        state_dict[self._jug] = {
            "x": x,
            "y": y,
            "z": z,
            "rot": rot,
            "is_held": held,
            "is_filled": filled,
        }
        state_dict[self._table] = {}

        # Get plug state.
        if CFG.coffee_machine_has_plug:
            (x, y, z), _ = p.getBasePositionAndOrientation(
                self._plug_id, physicsClientId=self._physics_client_id)
            state_dict[self._plug] = {
                "x": x,
                "y": y,
                "z": z,
                "plugged_in": float(self._machine_plugged_in_id is not None),
            }
        # Get machine state.
        button_color = p.getVisualShapeData(
            self._button_id, physicsClientId=self._physics_client_id)[0][-1]
        button_color_on_dist = sum(
            np.subtract(button_color, self.button_color_on)**2)
        button_color_off_dist = sum(
            np.subtract(button_color, self.button_color_off)**2)
        machine_on = float(button_color_on_dist < button_color_off_dist)
        state_dict[self._machine] = {
            "is_on": machine_on,
        }

        state = utils.create_state_from_dict(state_dict)
        sim_state = {"joint_positions": joint_positions}
        state = utils.PyBulletState(state.data, simulator_state=sim_state)
        if render_obs:
            # add unlabeled image, masks and labelled image
            state.add_images_and_masks(*self.render_segmented_obj())
            # import pdb; pdb.set_trace()

        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def step(self, action: Action, render_obs: bool = False) -> State:
        # import pdb; pdb.set_trace()
        # What's the previous robot state?
        # logging.debug("[env] start simulation step")
        current_ee_rpy = self._pybullet_robot.forward_kinematics(
            self._pybullet_robot.get_joints()).rpy
        state = super().step(action, render_obs=render_obs)
        # logging.debug("[env] set robot state")
        # logging.debug(f"tracking state: {state.pretty_str()}")

        # Move the liquid inside
        if self._jug_filled:
            pos, quat = p.getBasePositionAndOrientation(
                self._jug_id, physicsClientId=self._physics_client_id)
            p.resetBasePositionAndOrientation(
                self._jug_liquid_id,
                pos,
                quat,
                physicsClientId=self._physics_client_id)

        if CFG.coffee_machine_has_plug and \
            self._PluggedIn_holds(state, [self._plug]) and \
            self._machine_plugged_in_id is None:
            # logging.debug("[env] plug in the machine")
            # Create a constraint between plug and socket
            self._machine_plugged_in_id = p.createConstraint(
                parentBodyUniqueId=self._socket_id,
                parentLinkIndex=-1,
                childBodyUniqueId=self._plug_id,
                childLinkIndex=-1,
                jointAxis=[0, 0, 0],
                jointType=p.JOINT_FIXED,
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeVisualShape(self._button_id,
                                -1,
                                rgbaColor=self.button_color_off,
                                physicsClientId=self._physics_client_id)
            p.changeVisualShape(self._button_id,
                                0,
                                rgbaColor=self.button_color_off,
                                physicsClientId=self._physics_client_id)

        # If the robot is sufficiently close to the button, turn on the machine
        # and update the status of the jug.
        if self._PressingButton_holds(state, [self._robot, self._machine]):
            p.changeVisualShape(self._button_id,
                                -1,
                                rgbaColor=self.button_color_on,
                                physicsClientId=self._physics_client_id)
            p.changeVisualShape(self._button_id,
                                0,
                                rgbaColor=self.button_color_on,
                                physicsClientId=self._physics_client_id)

            # the jug is only filled if it's in the machine
            # and when the machine requires to plug in, the plug is in the
            # socket
            if self._JugInMachine_holds(state, [self._jug, self._machine]) and\
                    (not CFG.coffee_machine_has_plug or
                    self._machine_plugged_in_id is not None):
                if not self._jug_filled:
                    self._jug_liquid_id = self._create_pybullet_liquid_for_jug(
                    )
                self._jug_filled = True
            self._current_observation = self._get_state(render_obs)
            state = self._current_observation.copy()
        # If the robot is pouring into a cup, raise the liquid in it.
        if abs(state.get(self._robot, "tilt") -
               self.tilt_ub) < self.pour_angle_tol:
            # If the jug is empty, noop.
            if not self._jug_filled:
                return state
            # Find the cup to pour into, if any.
            cup = self._get_cup_to_pour(state)
            # If pouring into nothing, noop.
            if cup is None:
                return state
            # Increase the liquid in the cup.
            current_liquid = state.get(cup, "current_liquid")
            new_liquid = current_liquid + self.pour_velocity
            state.set(cup, "current_liquid", new_liquid)
            old_liquid_id = self._cup_to_liquid_id[cup]
            if old_liquid_id is not None:
                p.removeBody(old_liquid_id,
                             physicsClientId=self._physics_client_id)
            self._cup_to_liquid_id[cup] = self._create_pybullet_liquid_for_cup(
                cup, state)
            self._current_observation = self._get_state(render_obs)
            state = self._current_observation.copy()
        # Handle twisting
        elif self._Twisting_holds(state, [self._robot, self._jug]):
            gripper_pose = self._pybullet_robot.forward_kinematics(
                action.arr.tolist())
            d_roll = gripper_pose.rpy[0] - current_ee_rpy[0]
            d_yaw = gripper_pose.rpy[2] - current_ee_rpy[2]
            if np.abs(d_yaw) > 0.2:
                # changed sign
                print("flip roll")
                if d_yaw < 0:
                    d_roll -= np.pi
                if d_yaw > 0:
                    d_roll += np.pi

            (jx, jy, _), orn = p.getBasePositionAndOrientation(
                self._jug_id, physicsClientId=self._physics_client_id)
            jug_yaw = p.getEulerFromQuaternion(orn)[2]
            new_jug_yaw = jug_yaw - d_roll
            new_jug_yaw = utils.wrap_angle(new_jug_yaw)
            jug_orientation = p.getQuaternionFromEuler([0.0, 0.0, new_jug_yaw])
            p.resetBasePositionAndOrientation(
                self._jug_id, [jx, jy, self.z_lb + self.jug_height() / 2],
                jug_orientation,
                physicsClientId=self._physics_client_id)

            self._current_observation = self._get_state(render_obs)
            state = self._current_observation.copy()

        return state

    def _get_tasks(self,
                   num: int,
                   num_cups_lst: List[int],
                   rng: np.random.Generator,
                   is_train: bool = False) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num, num_cups_lst, rng, is_train)
        return self._add_pybullet_state_to_tasks(tasks)

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        if CFG.coffee_machine_has_plug:
            assert self._plug_id is not None
            return [self._jug_id, self._plug_id]
        return [self._jug_id]

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        if CFG.pybullet_robot == "fetch":
            # gripper parallel to y-axis
            normal = np.array([0., 1., 0.], dtype=np.float32)
        else:  # pragma: no cover
            # Shouldn't happen unless we introduce a new robot.
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        return {
            self._pybullet_robot.left_finger_id: normal,
            self._pybullet_robot.right_finger_id: -1 * normal,
        }

    def _state_to_gripper_orn(self, state: State) -> Quaternion:
        wrist = state.get(self._robot, "wrist")
        tilt = state.get(self._robot, "tilt")
        return self.tilt_wrist_to_gripper_orn(tilt, wrist)

    @classmethod
    def tilt_wrist_to_gripper_orn(cls, tilt: float,
                                  wrist: float) -> Quaternion:
        """Public for oracle options."""
        return p.getQuaternionFromEuler([0.0, tilt, wrist])

    def _gripper_orn_to_tilt_wrist(self,
                                   orn: Quaternion) -> Tuple[float, float]:
        _, tilt, wrist = p.getEulerFromQuaternion(orn)
        return (tilt, wrist)

    @classmethod
    def fingers_state_to_joint(cls, pybullet_robot: SingleArmPyBulletRobot,
                               finger_state: float) -> float:
        """Map the fingers in the given State to joint values for PyBullet."""
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

    def _cup_liquid_to_liquid_height(self, liquid: float,
                                     capacity: float) -> float:
        scale = 0.5 * np.sqrt(capacity / self.cup_capacity_ub)
        return liquid * scale

    def _cup_liquid_height_to_liquid(self, height: float,
                                     capacity: float) -> float:
        scale = 0.5 * np.sqrt(capacity / self.cup_capacity_ub)
        return height / scale

    def _cup_to_liquid_radius(self, capacity: float) -> float:
        scale = 1.5 * np.sqrt(capacity / self.cup_capacity_ub)
        return self.cup_radius * scale

    def _create_pybullet_liquid_for_cup(self, cup: Object,
                                        state: State) -> Optional[int]:
        current_liquid = state.get(cup, "current_liquid")
        cup_cap = state.get(cup, "capacity_liquid")
        liquid_height = self._cup_liquid_to_liquid_height(
            current_liquid, cup_cap)
        liquid_radius = self._cup_to_liquid_radius(cup_cap)
        if current_liquid == 0:
            return None
        cx = state.get(cup, "x")
        cy = state.get(cup, "y")
        cz = self.z_lb + current_liquid / 2 + 0.025

        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=liquid_radius,
            height=liquid_height,
            physicsClientId=self._physics_client_id)

        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=liquid_radius,
            length=liquid_height,
            rgbaColor=(0.35, 0.1, 0.0, 1.0),
            physicsClientId=self._physics_client_id)

        pose = (cx, cy, cz)
        orientation = self._default_orn
        return p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=collision_id,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=pose,
                                 baseOrientation=orientation,
                                 physicsClientId=self._physics_client_id)

    def _create_pybullet_liquid_for_jug(self) -> int:
        if CFG.coffee_use_pixelated_jug:
            liquid_height = self.jug_height() * 0.8
            liquid_radius = self.jug_radius * 1.3
        else:
            liquid_height = self.jug_height() * 0.6
            liquid_radius = self.jug_radius

        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[liquid_radius, liquid_radius, liquid_height / 2],
            physicsClientId=self._physics_client_id)

        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[liquid_radius, liquid_radius, liquid_height / 2],
            rgbaColor=(0.35, 0.1, 0.0, 1.0),
            physicsClientId=self._physics_client_id)

        pose, orientation = p.getBasePositionAndOrientation(
            self._jug_id, physicsClientId=self._physics_client_id)
        return p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=collision_id,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=pose,
                                 baseOrientation=orientation,
                                 physicsClientId=self._physics_client_id)

    @classmethod
    def _add_pybullet_coffee_machine(cls, physics_client_id: int) -> int:
        # Create the first box (main body base)
        half_extents_base = (
            cls.machine_x_len,
            cls.machine_y_len / 2,
            cls.machine_z_len / 2,
        )
        collision_id_base = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents_base,
            physicsClientId=physics_client_id)
        visual_id_base = p.createVisualShape(p.GEOM_BOX,
                                             halfExtents=half_extents_base,
                                             rgbaColor=cls.machine_color,
                                             physicsClientId=physics_client_id)
        pose_base = (
            cls.machine_x,
            cls.machine_y,
            cls.z_lb + cls.machine_z_len / 2,  # z
        )
        orientation_base = [0, 0, 0, 1]

        # Create the second box (top)
        half_extents_top = (
            cls.machine_x_len * 5 / 6,
            cls.machine_top_y_len / 2,
            cls.machine_z_len / 6,
        )
        collision_id_top = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents_top,
            physicsClientId=physics_client_id)
        visual_id_top = p.createVisualShape(p.GEOM_BOX,
                                            halfExtents=half_extents_top,
                                            rgbaColor=cls.machine_color,
                                            physicsClientId=physics_client_id)
        pose_top = (
            -cls.machine_x_len / 6,  # x relative to base
            -cls.machine_y_len / 2 -
            cls.machine_top_y_len / 2,  # y relative to base
            cls.machine_z_len / 3)
        orientation_top = cls._default_orn

        # Create the dispense area -- base.
        # Define the dimensions for the dispense area
        half_extents_dispense_base = (cls.machine_x_len,
                                      1.1 * cls.dispense_radius +
                                      cls.jug_radius + 0.003,
                                      cls.dispense_height)
        collision_id_dispense_base = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents_dispense_base,
            physicsClientId=physics_client_id)
        visual_id_dispense_base = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_dispense_base,
            rgbaColor=cls.machine_color,
            physicsClientId=physics_client_id)
        # the relative position for the dispense area
        pose_dispense_base = (
            0,
            -cls.machine_y_len - cls.dispense_radius + 0.01,
            -cls.machine_z_len / 2,
        )
        orientation_dispense_base = cls._default_orn

        # Create the multibody with a fixed link
        link_mass = 0
        link_inertial_frame_position = [0, 0, 0]
        link_inertial_frame_orientation = [0, 0, 0, 1]

        machine_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id_base,
            baseVisualShapeIndex=visual_id_base,
            basePosition=pose_base,
            baseOrientation=orientation_base,
            linkMasses=[link_mass, link_mass],
            linkCollisionShapeIndices=[
                collision_id_top, collision_id_dispense_base
            ],
            linkVisualShapeIndices=[visual_id_top, visual_id_dispense_base],
            linkPositions=[pose_top, pose_dispense_base],
            linkOrientations=[orientation_top, orientation_dispense_base],
            linkInertialFramePositions=[
                link_inertial_frame_position, link_inertial_frame_position
            ],
            linkInertialFrameOrientations=[
                link_inertial_frame_orientation,
                link_inertial_frame_orientation
            ],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0], [0, 0, 0]],
            physicsClientId=physics_client_id)

        return machine_id

    @classmethod
    def _add_pybullet_dispense_area(cls, physics_client_id: int) -> int:
        ## Create the dispense area -- base.
        pose = (
            cls.dispense_area_x,
            cls.dispense_area_y,
            # cls.z_lb + dispense_height)
            cls.z_lb)
        orientation = cls._default_orn

        # Dispense area circle
        # Create the collision shape.
        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=cls.dispense_radius,
            height=cls.dispense_height,
            physicsClientId=physics_client_id)

        # Create the visual_shape.
        visual_id = p.createVisualShape(p.GEOM_CYLINDER,
                                        radius=cls.dispense_radius +
                                        0.8 * cls.jug_radius,
                                        length=cls.dispense_height,
                                        rgbaColor=cls.plate_color_off,
                                        physicsClientId=physics_client_id)

        # Create the body.
        dispense_area_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=pose,
            baseOrientation=orientation,
            physicsClientId=physics_client_id)
        return dispense_area_id

    @classmethod
    def _add_pybullet_machine_button(cls, physics_client_id: int) -> int:
        # Add a button. Could do this as a link on the machine, but since
        # both never move, it doesn't matter.
        button_position = (cls.button_x, cls.button_y, cls.button_z)
        button_orientation = p.getQuaternionFromEuler(
            [0.0, np.pi / 2, np.pi / 2])

        # Create button shapes
        collision_id_button = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                cls.button_radius, cls.button_radius, cls.button_height / 2
            ],
            physicsClientId=physics_client_id)
        visual_id_button = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[cls.button_radius, cls.button_radius,
                         cls.button_height / 2],
            rgbaColor=cls.button_color_power_off if \
                CFG.coffee_machine_has_plug else cls.button_color_off,
            physicsClientId=physics_client_id)

        if CFG.coffee_machine_have_light_bar:
            # Create light bar shapes
            half_extents_bar = (
                cls.machine_z_len / 6 - 0.01,  # z
                cls.machine_x_len * 5 / 6,  # x
                cls.machine_top_y_len / 2)  # y
            collision_id_light_bar = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents_bar,
                physicsClientId=physics_client_id)
            visual_id_light_bar = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents_bar,
                rgbaColor=cls.button_color_off,
                physicsClientId=physics_client_id)

            # Link properties relative to the button
            link_positions = [[
                cls.machine_z_len / 6 - 0.017,  # larger is down
                cls.machine_x_len / 6 - 0.001,
                cls.machine_top_y_len / 2 - 0.001
            ]]
            link_orientations = [[0, 0, 0,
                                  1]]  # same orientation as the button

            button_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_id_button,
                baseVisualShapeIndex=visual_id_button,
                basePosition=button_position,
                baseOrientation=button_orientation,
                linkMasses=[0],
                linkCollisionShapeIndices=[collision_id_light_bar],
                linkVisualShapeIndices=[visual_id_light_bar],
                linkPositions=link_positions,
                linkOrientations=link_orientations,
                linkInertialFramePositions=[[0, 0, 0]],
                linkInertialFrameOrientations=[[0, 0, 0, 1]],
                linkParentIndices=[0],
                linkJointTypes=[p.JOINT_FIXED],
                linkJointAxis=[[0, 0, 0]],
                physicsClientId=physics_client_id)
        else:
            button_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_id_button,
                baseVisualShapeIndex=visual_id_button,
                basePosition=button_position,
                baseOrientation=button_orientation,
                physicsClientId=physics_client_id)
        return button_id

    @classmethod
    def _add_pybullet_jug(cls, physics_client_id: int) -> int:
        # Load coffee jug.

        # This pose doesn't matter because it gets overwritten in reset.
        jug_loc = ((0, 0, 0))
        rot = 0
        jug_orientation = p.getQuaternionFromEuler([0.0, 0.0, rot - np.pi / 2])

        # Old jug
        if CFG.coffee_use_pixelated_jug:
            jug_id = p.loadURDF(
                utils.get_env_asset_path("urdf/jug-pixel.urdf"),
                globalScaling=0.2,  # enlarged jug
                useFixedBase=False,
                physicsClientId=physics_client_id)

        else:
            jug_id = p.loadURDF(
                utils.get_env_asset_path("urdf/kettle.urdf"),
                globalScaling=0.09,  # enlarged jug
                useFixedBase=False,
                physicsClientId=physics_client_id)
            p.changeVisualShape(jug_id,
                                0,
                                rgbaColor=cls.jug_color,
                                physicsClientId=physics_client_id)
            # remove the lid
            p.changeVisualShape(jug_id,
                                1,
                                rgbaColor=[1, 1, 1, 0],
                                physicsClientId=physics_client_id)
        p.changeDynamics(
            bodyUniqueId=jug_id,
            linkIndex=-1,  # -1 for the base link
            mass=0.1,
            physicsClientId=physics_client_id)

        p.resetBasePositionAndOrientation(jug_id,
                                          jug_loc,
                                          jug_orientation,
                                          physicsClientId=physics_client_id)

        return jug_id

    @classmethod
    def _add_pybullet_table(cls, physics_client_id: int) -> int:
        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          cls.table_pose,
                                          cls.table_orientation,
                                          physicsClientId=physics_client_id)
        return table_id

    @classmethod
    def _add_pybullet_cord(cls, physics_client_id: int) -> List[int]:
        """First segment connects the machine, last connects to the wall."""
        # Rope parameters
        base_position = [cls.cord_start_x, cls.cord_start_y, cls.cord_start_z]
        segments = []

        # Create rope segments
        for i in range(cls.num_cord_links):

            # Position each segment along an arc with curvature
            x_pos = base_position[0] - i * (cls.cord_link_length +
                                            cls.cord_segment_gap)
            y_pos = base_position[1]  #+ curvature_amplitude *\
            # math.sin(i * math.pi / (cls.num_cord_links - 1))
            z_pos = base_position[2]  # Maintain height
            link_pos = [x_pos, y_pos, z_pos]

            # Set color: Red for the first link, Blue for the last link, and
            # Black for others
            if i == 0:
                color = [0, 0, 0, 1]  # Black
            elif i == cls.num_cord_links - 1:
                # color = [0, 0, 1, 1]  # Blue
                color = [1, 0, 0, 1]  # Red
            else:
                color = [0, 0, 0, 1]  # Black

            # Create collision and visual shapes
            segment = p.createCollisionShape(p.GEOM_BOX,
                                             halfExtents=[
                                                 cls.cord_link_length / 2,
                                                 cls.cord_link_length / 2,
                                                 cls.cord_link_length / 2
                                             ],
                                             physicsClientId=physics_client_id)
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    cls.cord_link_length / 2, cls.cord_link_length / 2,
                    cls.cord_link_length / 2
                ],
                rgbaColor=color,
                physicsClientId=physics_client_id)
            base_mass = 0 if i == 0 else 0.001
            segment_id = p.createMultiBody(baseMass=base_mass,
                                           baseCollisionShapeIndex=segment,
                                           baseVisualShapeIndex=visual_shape,
                                           basePosition=link_pos,
                                           physicsClientId=physics_client_id)
            segments.append(segment_id)

        # Connect segments with joints
        half_gap = cls.cord_segment_gap / 2
        for i in range(len(segments) - 1):
            p.createConstraint(
                parentBodyUniqueId=segments[i],
                parentLinkIndex=-1,
                childBodyUniqueId=segments[i + 1],
                childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 0],
                parentFramePosition=[
                    -cls.cord_link_length / 2 - half_gap, 0, 0
                ],
                childFramePosition=[cls.cord_link_length / 2 + half_gap, 0, 0])
            # Adjust constraint parameters for softness
            # p.changeConstraint(
            #     constraint_id,
            #     maxForce=0.1,    # Lower max force for flexibility
            #     erp=0.1         # Adjust error reduction parameter
            # )
        return segments

    @classmethod
    def _add_pybullet_socket(cls, physics_client_id: int) -> None:
        # Add the blue socket block

        socket_position = [cls.socket_x, cls.socket_y, cls.socket_z]
        socket_collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                cls.socket_width / 2, cls.socket_depth / 2,
                cls.socket_height / 2
            ],
            physicsClientId=physics_client_id)
        socket_visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                cls.socket_width / 2, cls.socket_depth / 2,
                cls.socket_height / 2
            ],
            rgbaColor=[0, 0, 1, 1],  # Blue color
            physicsClientId=physics_client_id)
        socket_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=socket_collision_shape,
            baseVisualShapeIndex=socket_visual_shape,
            basePosition=socket_position,
            physicsClientId=physics_client_id)

        # # Connect the last segment to the socket
        # p.createConstraint(
        #     parentBodyUniqueId=segments[-1],
        #     parentLinkIndex=-1,
        #     childBodyUniqueId=socket_id,
        #     childLinkIndex=-1,
        #     jointType=p.JOINT_FIXED,
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=[-cls.cord_link_length / 2, 0, 0],
        #     childFramePosition=[socket_size / 2, 0, 0],
        #     physicsClientId=physics_client_id
        # )
        return socket_id

    @classmethod
    def _add_pybullet_debug_lines(cls, physics_client_id: int) -> None:
        # Draw the workspace on the table for clarity.
        for z in [cls.z_lb, cls.z_ub]:
            p.addUserDebugLine([cls.x_lb, cls.y_lb, z],
                               [cls.x_ub, cls.y_lb, z], [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_ub, z],
                               [cls.x_ub, cls.y_ub, z], [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_lb, z],
                               [cls.x_lb, cls.y_ub, z], [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_ub, cls.y_lb, z],
                               [cls.x_ub, cls.y_ub, z], [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
        # Draw different sampling regions for reference.
        p.addUserDebugLine([cls.jug_init_x_lb, cls.jug_init_y_lb, cls.z_lb],
                           [cls.jug_init_x_ub, cls.jug_init_y_lb, cls.z_lb],
                           [0.0, 0.0, 1.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([cls.jug_init_x_lb, cls.jug_init_y_ub, cls.z_lb],
                           [cls.jug_init_x_ub, cls.jug_init_y_ub, cls.z_lb],
                           [0.0, 0.0, 1.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([cls.jug_init_x_lb, cls.jug_init_y_lb, cls.z_lb],
                           [cls.jug_init_x_lb, cls.jug_init_y_ub, cls.z_lb],
                           [0.0, 0.0, 1.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([cls.jug_init_x_ub, cls.jug_init_y_lb, cls.z_lb],
                           [cls.jug_init_x_ub, cls.jug_init_y_ub, cls.z_lb],
                           [0.0, 0.0, 1.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([cls.cup_init_x_lb, cls.cup_init_y_lb, cls.z_lb],
                           [cls.cup_init_x_ub, cls.cup_init_y_lb, cls.z_lb],
                           [0.0, 0.0, 1.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([cls.cup_init_x_lb, cls.cup_init_y_ub, cls.z_lb],
                           [cls.cup_init_x_ub, cls.cup_init_y_ub, cls.z_lb],
                           [0.0, 0.0, 1.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([cls.cup_init_x_lb, cls.cup_init_y_lb, cls.z_lb],
                           [cls.cup_init_x_lb, cls.cup_init_y_ub, cls.z_lb],
                           [0.0, 0.0, 1.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([cls.cup_init_x_ub, cls.cup_init_y_lb, cls.z_lb],
                           [cls.cup_init_x_ub, cls.cup_init_y_ub, cls.z_lb],
                           [0.0, 0.0, 1.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        # Draw coordinate frame labels for reference.
        p.addUserDebugLine([0, 0, 0], [0.25, 0, 0], [1.0, 0.0, 0.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0],
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0.0, 0.25, 0], [1.0, 0.0, 0.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0],
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([0, 0, 0], [0.0, 0, 0.25], [1.0, 0.0, 0.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugText("z", [0, 0, 0.25], [0.0, 0.0, 0.0],
                           physicsClientId=physics_client_id)

    def update_camera(self, distance, yaw, pitch, target):
        """Update the PyBullet camera view."""
        p.resetDebugVisualizerCamera(distance, yaw, pitch, target)
        # self.update_camera(0.4, -60, -30, (0.75, 1.25, 0.42))

    def _copy_observation(self, obs: Observation) -> Observation:
        return copy.deepcopy(obs)

# if CFG.coffee_render_grid_world:
#             # Camera parameters for grid world
#             PyBulletCoffeeEnv._camera_distance = 3
#             PyBulletCoffeeEnv._camera_fov = 8
#             PyBulletCoffeeEnv._camera_yaw = 90
#             PyBulletCoffeeEnv._camera_pitch = 0  # lower
#             PyBulletCoffeeEnv._camera_target = (0.75, 1.33, 0.3)
#         else:
#             # Camera parameters -- standard
#             PyBulletCoffeeEnv._camera_distance = 1.3
#             if CFG.coffee_machine_has_plug:
#                 PyBulletCoffeeEnv._camera_yaw = -60
#                 # self._camera_yaw: ClassVar[float] = -90
#                 # self._camera_yaw: ClassVar[float] = -180
#             else:
#                 PyBulletCoffeeEnv._camera_yaw = 70
#             PyBulletCoffeeEnv._camera_pitch = -38  # lower
#             # PyBulletCoffeeEnv._camera_pitch = 0  # even lower
#             PyBulletCoffeeEnv._camera_target = (0.75, 1.25, 0.42)