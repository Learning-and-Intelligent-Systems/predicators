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
--num_train_tasks 0 --num_test_tasks 1 --use_gui \
--coffee_rotated_jug_ratio 0 \
--sesame_check_expected_atoms False --coffee_jug_pickable_pred True \
--coffee_twist_sampler False \
--make_test_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 \
--coffee_machine_have_light_bar False \
--coffee_move_back_after_place_and_push True \
--coffee_machine_has_plug True --sesame_max_skeletons_optimized 1 \
--make_failure_videos \
--debug --option_model_terminate_on_repeat False \
--coffee_use_pixelated_jug True --pybullet_ik_validate False

With the simplified tasks, both pixelated jug and old jug should work.
With the full tasks, the old jug should work.
"""
import logging
import random
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.coffee import CoffeeEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, Predicate, \
    State


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
    _finger_action_tol: ClassVar[float] = 1e-3
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

    # Table / workspace config
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0., 0., np.pi / 2])

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2

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
                (1,1,1,1) # White
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
    # Camera font view parameters.
    _camera_distance: ClassVar[float]
    _camera_fov: ClassVar[float]
    _camera_yaw: ClassVar[float]
    _camera_pitch: ClassVar[float]
    _camera_target: ClassVar[Pose3D]

    def __init__(self, use_gui: bool = True) -> None:
        if CFG.coffee_render_grid_world:
            # Camera parameters for grid world
            PyBulletCoffeeEnv._camera_distance = 3
            PyBulletCoffeeEnv._camera_fov = 8
            PyBulletCoffeeEnv._camera_yaw = 90
            PyBulletCoffeeEnv._camera_pitch = 0  # lower
            PyBulletCoffeeEnv._camera_target = (0.75, 1.33, 0.3)
        else:
            # Camera parameters -- standard
            PyBulletCoffeeEnv._camera_distance = 1.3
            if CFG.coffee_machine_has_plug:
                PyBulletCoffeeEnv._camera_yaw = -60
                # self._camera_yaw: ClassVar[float] = -90
                # self._camera_yaw: ClassVar[float] = -180
            else:
                PyBulletCoffeeEnv._camera_yaw = 70
            PyBulletCoffeeEnv._camera_pitch = -38  # lower
            # PyBulletCoffeeEnv._camera_pitch = 0  # even lower
            PyBulletCoffeeEnv._camera_target = (0.75, 1.25, 0.42)

        super().__init__(use_gui)

        # Create the cups lazily because they can change size and color.
        # self._cup_id_to_cup: Dict[int, Object] = {}
        self._cup_to_liquid_id: Dict[Object, Optional[int]] = {}
        self._cup_to_capacity: Dict[Object, float] = {}
        # The status of the jug is not modeled inside PyBullet.
        self._jug_filled = False
        self._jug_liquid_id: Optional[int] = None

        self._cord_ids: Optional[List[int]] = None
        self._machine_plugged_in_id: Optional[int] = None

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

        # Coffee Machine
        machine_id = cls._add_pybullet_coffee_machine(physics_client_id)
        dispense_area_id = cls._add_pybullet_dispense_area(physics_client_id)
        button_id = cls._add_pybullet_machine_button(physics_client_id)
        bodies["machine_id"] = machine_id
        bodies["dispense_area_id"] = dispense_area_id
        bodies["button_id"] = button_id

        jug_id = cls._add_pybullet_jug(physics_client_id)
        bodies["jug_id"] = jug_id

        if CFG.coffee_machine_has_plug:
            socket_id = cls._add_pybullet_socket(physics_client_id)
            bodies["socket_id"] = socket_id

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table.id = pybullet_bodies["table_id"]
        self._jug.id = pybullet_bodies["jug_id"]
        self._machine.id = pybullet_bodies["machine_id"]
        self._robot.id = self._pybullet_robot.robot_id
        self._dispense_area_id = pybullet_bodies["dispense_area_id"]
        self._button_id = pybullet_bodies["button_id"]
        if CFG.coffee_machine_has_plug:
            self._socket_id = pybullet_bodies["socket_id"]

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_coffee"

    def _create_task_specific_objects(self, state: State) -> None:
        """Remove/rebuild cups, liquids, and cords so each new task can have
        different cups and states."""
        self._remake_cups(state)
        self._remake_cup_liquids(state)
        self._remake_jug_liquid(state)
        self._remake_cord()

    def _remake_cups(self, state: State) -> None:
        """Re-load cup URDFs with appropriate scaling and color for each new
        cup."""
        # for old_cup_id in self._cup_id_to_cup:
        #     p.removeBody(old_cup_id, physicsClientId=self._physics_client_id)
        for cup in self._cups:
            if cup.id is not None:
                p.removeBody(cup.id, physicsClientId=self._physics_client_id)
        # self._cup_id_to_cup.clear()

        cup_objs = state.get_objects(self._cup_type)
        self._cup_to_capacity.clear()
        for i, cup_obj in enumerate(cup_objs):
            cup_cap = state.get(cup_obj, "capacity_liquid")
            global_scale = 0.5 * cup_cap / self.cup_capacity_ub
            color = self._obj_colors[self._train_rng.choice(
                len(self._obj_colors))]
            if CFG.coffee_use_pixelated_jug:
                file = "urdf/pot-pixel.urdf"
                global_scale *= 0.5
            else:
                file = "urdf/cup.urdf"
            cup_id = create_object(file,
                                   color=color,
                                   scale=global_scale,
                                   use_fixed_base=True,
                                   physics_client_id=self._physics_client_id)
            # self._cup_id_to_cup[cup_id] = cup_obj
            self._cup_to_capacity[cup_obj] = cup_cap
            cup_obj.id = cup_id

    def _remake_cup_liquids(self, state: State) -> None:
        """Re-create the visual liquid objects for the new cups."""
        for liquid_id in self._cup_to_liquid_id.values():
            if liquid_id is not None:
                p.removeBody(liquid_id,
                             physicsClientId=self._physics_client_id)
        self._cup_to_liquid_id.clear()

        cup_objs = state.get_objects(self._cup_type)
        for cup in cup_objs:
            new_liquid_id = self._create_pybullet_liquid_for_cup(cup, state)
            self._cup_to_liquid_id[cup] = new_liquid_id

    def _remake_jug_liquid(self, state: State) -> None:
        """Check jug's is_filled status and re-create liquid object if needed.

        Remove old jug liquid if jug is now empty.
        """
        self._jug_filled = bool(state.get(self._jug, "is_filled") > 0.5)
        if self._jug_liquid_id is not None:
            p.removeBody(self._jug_liquid_id,
                         physicsClientId=self._physics_client_id)
            self._jug_liquid_id = None
        if self._jug_filled:
            self._jug_liquid_id = self._create_pybullet_liquid_for_jug()

    def _remake_cord(self) -> None:
        """If the machine uses a plug, rebuild the cord bodies and
        constraints."""
        if CFG.coffee_machine_has_plug:
            if self._cord_ids is not None:
                # Remove old cord pieces
                for part_id in self._cord_ids:
                    p.removeBody(part_id,
                                 physicsClientId=self._physics_client_id)
            if self._machine_plugged_in_id is not None:
                p.removeConstraint(self._machine_plugged_in_id,
                                   physicsClientId=self._physics_client_id)
                self._machine_plugged_in_id = None
            # Rebuild the cord chain
            self._cord_ids, self._cord_constraints = self._add_pybullet_cord(
                self._physics_client_id)
            self._plug.id = self._cord_ids[-1]

    def _reset_custom_env_state(self, state: State) -> None:
        """Handles extra coffee-specific reset steps: spawning cups from
        scratch, adding liquid visuals, adjusting jug fill color, toggling the
        machine button, etc.

        The base `_reset_state` has already done the standard
        position/orientation resets for objects in `_get_all_objects()`.
        """
        # Machine button color
        #    Check if the machine is on and the jug is in place:
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

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._jug_type:
            if feature == "is_filled":
                return float(self._jug_filled)
        elif obj.type == self._machine_type:
            if feature == "is_on":
                button_color = p.getVisualShapeData(
                    self._button_id,
                    physicsClientId=self._physics_client_id)[0][-1]
                button_color_on_dist = sum(
                    np.subtract(button_color, self.button_color_on)**2)
                button_color_off_dist = sum(
                    np.subtract(button_color, self.button_color_off)**2)
                return float(button_color_on_dist < button_color_off_dist)
        elif obj.type == self._cup_type:
            if feature == "capacity_liquid":
                return self._cup_to_capacity[obj]
            elif feature == "current_liquid":
                liquid_id = self._cup_to_liquid_id.get(obj, None)
                if liquid_id is not None:
                    liquid_height = p.getVisualShapeData(
                        liquid_id,
                        physicsClientId=self._physics_client_id,
                    )[0][3][0]
                    return self._cup_liquid_height_to_liquid(
                        liquid_height, self._cup_to_capacity[obj])
                else:
                    return 0.0
            elif feature == "target_liquid":
                return self._cup_to_capacity[obj] * self.cup_target_frac
        elif obj.type == self._plug_type:
            if feature == "plugged_in":
                return float(self._machine_plugged_in_id is not None)
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def step(self, action: Action, render_obs: bool = False) -> State:
        # Save current end-effector roll-pitch-yaw for later comparison
        current_ee_rpy = self._pybullet_robot.forward_kinematics(
            self._pybullet_robot.get_joints()).rpy
        state = super().step(action, render_obs=render_obs)
        self._update_jug_liquid_position()
        if CFG.coffee_machine_has_plug:
            self._check_and_apply_plug_in_constraint(state)
        self._handle_machine_on_and_jug_filling(state)
        self._handle_pouring(state)
        self._handle_twisting(state, current_ee_rpy, action)
        state = self._current_observation.copy()

        return state

    def _update_jug_liquid_position(self) -> None:
        """If the jug is filled, move its liquid to match the jug's pose."""
        if self._jug_filled and self._jug_liquid_id is not None:
            pos, quat = p.getBasePositionAndOrientation(
                self._jug.id, physicsClientId=self._physics_client_id)
            p.resetBasePositionAndOrientation(
                self._jug_liquid_id,
                pos,
                quat,
                physicsClientId=self._physics_client_id)

    def _check_and_apply_plug_in_constraint(self, state: State) -> None:
        """If the machine uses a plug and the plug is 'plugged_in' in the
        state, create (or maintain) a fixed constraint between the plug and the
        socket."""
        if self._PluggedIn_holds(state, [self._plug]) and \
                self._machine_plugged_in_id is None:
            # Create a constraint between plug and socket
            self._machine_plugged_in_id = p.createConstraint(
                parentBodyUniqueId=self._socket_id,
                parentLinkIndex=-1,
                childBodyUniqueId=self._plug.id,
                childLinkIndex=-1,
                jointAxis=[0, 0, 0],
                jointType=p.JOINT_FIXED,
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            # Update button color to "off" (but machine has power)
            p.changeVisualShape(self._button_id,
                                -1,
                                rgbaColor=self.button_color_off,
                                physicsClientId=self._physics_client_id)
            if CFG.coffee_plug_break_after_plugged_in:
                p.removeConstraint(self._cord_constraints[2],
                                   physicsClientId=self._physics_client_id)

    def _handle_machine_on_and_jug_filling(self, state: State) -> None:
        """If the robot is pressing the machine button, turn on the machine and
        fill the jug if it's placed in the machine and (optionally) plugged
        in."""
        if self._PressingButton_holds(state, [self._robot, self._machine]):
            # Change the machine button color to "on"
            p.changeVisualShape(self._button_id,
                                -1,
                                rgbaColor=self.button_color_on,
                                physicsClientId=self._physics_client_id)
            # Fill jug if in machine & (plugged in if required)
            if (self._JugInMachine_holds(state, [self._jug, self._machine])
                    and (not CFG.coffee_machine_has_plug
                         or self._machine_plugged_in_id is not None)):
                if not self._jug_filled:
                    self._jug_liquid_id = self._create_pybullet_liquid_for_jug(
                    )
                self._jug_filled = True
            # Refresh current observation
            self._current_observation = self._get_state(render_obs=False)

    def _handle_pouring(self, state: State) -> None:
        """If the robot is tilted sufficiently to pour, increase liquid in the
        appropriate cup.

        If the jug is empty or there's no target cup, do nothing.
        """
        if abs(state.get(self._robot, "tilt") -
               self.tilt_ub) < self.pour_angle_tol:
            # If the jug is empty, do nothing
            if not self._jug_filled:
                return
            # Identify which cup (if any) is being poured into
            cup = self._get_cup_to_pour(state)
            if cup is None:
                return

            # Increase the liquid in the cup
            current_liquid = state.get(cup, "current_liquid")
            new_liquid = current_liquid + self.pour_velocity
            state.set(cup, "current_liquid", new_liquid)

            # Remove the old liquid body in PyBullet
            old_liquid_id = self._cup_to_liquid_id.get(cup, None)
            if old_liquid_id is not None:
                p.removeBody(old_liquid_id,
                             physicsClientId=self._physics_client_id)

            # Create a new one with updated height
            self._cup_to_liquid_id[cup] = self._create_pybullet_liquid_for_cup(
                cup, state)
            # Refresh current observation
            self._current_observation = self._get_state(render_obs=False)

    def _handle_twisting(self, state: State,
                         current_ee_rpy: Tuple[float, float,
                                               float], action: Action) -> None:
        """If the robot is twisting the jug, update the jug's yaw accordingly.

        Accounts for flipping if the sign of yaw changes drastically.
        """
        if self._Twisting_holds(state, [self._robot, self._jug]):
            gripper_pose = self._pybullet_robot.forward_kinematics(
                action.arr.tolist())
            d_roll = gripper_pose.rpy[0] - current_ee_rpy[0]
            d_yaw = gripper_pose.rpy[2] - current_ee_rpy[2]

            # Handle wrap-around or flipping
            if abs(d_yaw) > 0.2:
                if d_yaw < 0:
                    d_roll -= np.pi
                else:
                    d_roll += np.pi

            (jx, jy, _), jug_quat = p.getBasePositionAndOrientation(
                self._jug.id, physicsClientId=self._physics_client_id)
            jug_yaw = p.getEulerFromQuaternion(jug_quat)[2]
            new_jug_yaw = utils.wrap_angle(jug_yaw - d_roll)
            new_jug_quat = p.getQuaternionFromEuler([0.0, 0.0, new_jug_yaw])
            p.resetBasePositionAndOrientation(
                self._jug.id, [jx, jy, self.z_lb + self.jug_height() / 2],
                new_jug_quat,
                physicsClientId=self._physics_client_id)
            # Refresh current observation
            self._current_observation = self._get_state(render_obs=False)

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
            assert self._plug.id is not None
            return [self._jug.id, self._plug.id]
        return [self._jug.id]

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
            self._jug.id, physicsClientId=self._physics_client_id)
        return p.createMultiBody(baseMass=0.001,
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
            halfExtents=(0, 0, 0),
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
        pose = (cls.dispense_area_x, cls.dispense_area_y, cls.z_lb)
        orientation = cls._default_orn

        # Dispense area circle
        # Create the collision shape.
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(0, 0, 0),
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
        jug_orientation = p.getQuaternionFromEuler([0.0, 0.0, rot])

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
                                          cls.table_pos,
                                          cls.table_orn,
                                          physicsClientId=physics_client_id)
        return table_id

    @classmethod
    def _add_pybullet_cord(cls, physics_client_id: int) -> List[int]:
        """First segment connects the machine, last connects to the wall."""
        # Rope parameters
        base_position = [cls.cord_start_x, cls.cord_start_y, cls.cord_start_z]
        segments = []
        constraint_ids = []

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
                color = [1, 0, 0, 1]  # Red
            else:
                color = [0.5, 0, 0, 1]  # Black

            # Create collision and visual shapes
            if i == cls.num_cord_links - 1:
                col_x = cls.cord_link_length / 2
                col_y = cls.cord_link_length / 2
                col_z = cls.cord_link_length / 2
            else:
                col_x = cls.cord_link_length / 4
                col_y = cls.cord_link_length / 4
                col_z = cls.cord_link_length / 4
            segment = p.createCollisionShape(p.GEOM_BOX,
                                             halfExtents=[col_x, col_y, col_z],
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
            constraint_id = p.createConstraint(
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
            constraint_ids.append(constraint_id)
            # Adjust constraint parameters for softness
            # p.changeConstraint(
            #     constraint_id,
            #     maxForce=0.1,    # Lower max force for flexibility
            #     erp=0.1         # Adjust error reduction parameter
            # )
        return segments, constraint_ids

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

    @classmethod
    def _get_jug_handle_grasp(cls, state: State,
                              jug: Object) -> Tuple[float, float, float]:
        # Orient pointing down.
        rot = state.get(jug, "rot")
        target_x = state.get(jug, "x") + np.cos(rot) * cls.jug_handle_offset
        target_y = state.get(jug,
                             "y") + np.sin(rot) * cls.jug_handle_offset - 0.02
        if not CFG.coffee_use_pixelated_jug:
            target_y += 0.02
        target_z = cls.z_lb + cls.jug_handle_height()
        return (target_x, target_y, target_z)


if __name__ == "__main__":
    """Run a simple simulation to test the environment."""
    import time

    # Make a task
    CFG.seed = 1
    CFG.pybullet_sim_steps_per_action = 1
    env = PyBulletCoffeeEnv(use_gui=True)
    rng = np.random.default_rng(CFG.seed)
    task = env._make_tasks(1, rng)[0]
    env._reset_state(task.init)

    while True:
        # Robot does nothing
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))

        env.step(action)
        time.sleep(0.01)
