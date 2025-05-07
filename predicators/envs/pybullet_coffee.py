"""A PyBullet version of CoffeeEnv."""

import logging
import random
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

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
    Predicate, State, Type
from predicators.utils import NSPredicate, RawState, VLMQuery


class PyBulletCoffeeEnv(PyBulletEnv, CoffeeEnv):
    """PyBullet Coffee domain."""

    # Need to override a number of settings to conform to the actual dimensions
    # of the robots, table, etc.
    grasp_finger_tol: ClassVar[float] = 1e-2
    grasp_position_tol: ClassVar[float] = 1e-2
    dispense_tol: ClassVar[float] = 1e-2
    pour_angle_tol: ClassVar[float] = 1e-1
    pour_pos_tol: ClassVar[float] = 1.0
    init_padding: ClassVar[float] = 0.05
    pick_jug_y_padding: ClassVar[float] = 0.05
    pick_jug_rot_tol: ClassVar[float] = np.pi / 3
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
    machine_top_y_len: ClassVar[float] = 1.5 * machine_y_len
    machine_x: ClassVar[float] = x_ub - machine_x_len / 2 - init_padding
    machine_y: ClassVar[float] = y_ub - machine_y_len / 2 - init_padding
    button_radius: ClassVar[float] = 0.2 * machine_y_len
    button_x: ClassVar[float] = machine_x
    button_y: ClassVar[
        float] = machine_y - machine_y_len / 2 - machine_top_y_len
    # button_z: ClassVar[float] = z_lb + 3 * machine_z_len / 4
    button_z: ClassVar[float] = z_lb + machine_z_len - button_radius
    button_press_threshold: ClassVar[float] = 1e-3
    machine_color: ClassVar[Tuple[float, float, float, float]] =\
        (0.1, 0.1, 0.1, 1) # Black
    # (0.75, 0.75, 0.75, 1.0) # Grey
    button_color_on: ClassVar[Tuple[float, float, float,
                                    float]] = (0.2, 0.5, 0.2, 1.0)
    plate_color_on: ClassVar[Tuple[float, float, float, float]] = machine_color
    #    float]] = (0.9, 0.3, 0.0, 0.7)
    button_color_off: ClassVar[Tuple[float, float, float,
                                     float]] = (0.5, 0.2, 0.2, 1.0)
    plate_color_off: ClassVar[Tuple[float, float, float,
                                    float]] = machine_color
    # float]] = (0.6, 0.6, 0.6, 0.5)
    jug_color: ClassVar[Tuple[float, float, float, float]] =\
        (0.5,1,0,0.5) # Green
    # (1.0, 1.0, 1.0, 1.0) # White
    # jug_color: ClassVar[Tuple[float, float, float, float]] = (1.0, 1.0, 1.0, 1.0)
    jug_radius: ClassVar[float] = 0.3 * machine_y_len
    jug_height: ClassVar[float] = 0.15 * (z_ub - z_lb)  # kettle urdf
    # jug_height: ClassVar[float] = 0.2 * (z_ub - z_lb) # cup urdf
    jug_init_x_lb: ClassVar[
        float] = machine_x - machine_x_len / 2 + init_padding
    jug_init_x_ub: ClassVar[
        float] = machine_x + machine_x_len / 2 - init_padding
    # adding 1 extra padding
    jug_init_y_lb: ClassVar[float] = y_lb + 3 * jug_radius + init_padding
    jug_init_y_ub: ClassVar[
        float] = machine_y - machine_y_len - 4 * jug_radius - init_padding
    jug_init_y_ub_og: ClassVar[
        float] = machine_y - machine_y_len - 3 * jug_radius - init_padding
    jug_handle_offset: ClassVar[float] = 3 * jug_radius  # kettle urdf
    # jug_handle_offset: ClassVar[float] = 2.2 * jug_radius # cup urdf
    jug_handle_height: ClassVar[float] = jug_height
    jug_init_rot_lb: ClassVar[float] = -2 * np.pi / 3
    jug_init_rot_ub: ClassVar[float] = 2 * np.pi / 3
    # Dispense area settings.
    dispense_area_x: ClassVar[float] = machine_x
    dispense_area_y: ClassVar[float] = machine_y - 5 * jug_radius
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
    pour_x_offset: ClassVar[float] = cup_radius
    pour_y_offset: ClassVar[float] = -3 * (cup_radius + jug_radius)
    pour_z_offset: ClassVar[float] = 2.5 * (cup_capacity_ub + jug_height - \
                                            jug_handle_height)
    pour_velocity: ClassVar[float] = cup_capacity_ub / 10.0
    # Table settings.
    table_pose: ClassVar[Pose3D] = (0.75, 1.35, 0.0)
    table_orientation: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    # Camera parameters.
    _camera_distance: ClassVar[float] = 1.3  #0.8
    # _camera_yaw: ClassVar[float] = 70
    # _camera_pitch: ClassVar[float] = -48
    # _camera_target: ClassVar[Pose3D] = (0.75, 1.35, 0.42)
    # Yichao
    # _camera_yaw: ClassVar[float] = -70
    _camera_yaw: ClassVar[float] = 70
    # _camera_yaw: ClassVar[float] = 80
    _camera_pitch: ClassVar[float] = -38  # lower
    # _camera_pitch: ClassVar[float] = 0
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # Camera font view parameters.
    _camera_distance_front: ClassVar[float] = 1
    _camera_yaw_front: ClassVar[float] = 180
    _camera_pitch_front: ClassVar[float] = -24

    # _camera_target_front: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        bbox_features = ["bbox_left", "bbox_right", "bbox_upper", "bbox_lower"]
        self._table_type = Type(
            "table",
            [] + (bbox_features if CFG.env_include_bbox_features else []))
        self._robot_type = Type(
            "robot", ["x", "y", "z", "tilt", "wrist", "fingers"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._jug_type = Type(
            "jug", ["x", "y", "z", "rot", "is_held", "is_filled"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._machine_type = Type(
            "coffee_machine", ["is_on"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._cup_type = Type("cup", [
            "x", "y", "z", "capacity_liquid", "target_liquid", "current_liquid"
        ] + (bbox_features if CFG.env_include_bbox_features else []))

        # Create the cups lazily because they can change size and color.
        self._cup_id_to_cup: Dict[int, Object] = {}
        self._cup_to_liquid_id: Dict[Object, Optional[int]] = {}
        self._cup_to_capacity: Dict[Object, float] = {}
        # The status of the jug is not modeled inside PyBullet.
        self._jug_filled = False
        self._jug_liquid_id = None
        self._obj_id_to_obj: Dict[int, Object] = {}

        self.ns_to_sym_predicates: Dict[Tuple[str], Predicate] = {
            ("JugInMachine"): self._JugInMachine,
            ("RobotHoldingJug"): self._Holding,
            ("GripperOpen"): self._HandEmpty,
            ("JugHasCoffee"): self._JugFilled,
            ("JugSideToGripper"): self._JugPickable,
            ("MachineOn"): self._MachineOn,
        }

        # Predicates
        self._CupFilled_NSP = NSPredicate("CupFilled", [self._cup_type],
                                          self._CupFilled_NSP_holds)
        self._Holding_NSP = Predicate("Holding",
                                      [self._robot_type, self._jug_type],
                                      self._Holding_NSP_holds)
        self._JugInMachine_NSP = Predicate(
            "JugInMachine", [self._jug_type, self._machine_type],
            self._JugInMachine_NSP_holds)
        self._JugPickable_NSP = Predicate("JugPickable", [self._jug_type],
                                          self._JugPickable_NSP_holds)
        self._JugFilled_NSP = Predicate("JugFilled", [self._jug_type],
                                        self._JugFilled_NSP_holds)
        self._OnTable_NSP = Predicate("OnTable", [self._jug_type],
                                      self._OnTable_NSP_holds)
        self._MachineOn_NSP = Predicate("MachineOn", [self._machine_type],
                                        self._MachineOn_NSP_holds)

    @property
    def ns_predicates(self) -> Set[NSPredicate]:
        return {
            self._CupFilled_NSP,
            self._JugFilled_NSP,
            self._Holding_NSP,
            self._HandEmpty,
            self._JugInMachine_NSP,
            self._JugPickable_NSP,
            self._OnTable_NSP,
            self._MachineOn_NSP,
        }

    def _MachineOn_NSP_holds(self, state: RawState,
                             objects: Sequence[Object]) -> bool:
        machine, = objects
        machine_name = machine.id_name
        attention_image = state.crop_to_objects([machine])
        return state.evaluate_simple_assertion(f"{machine_name} is on",
                                               attention_image)

    def _CupFilled_NSP_holds(self, state: RawState,
                             objects: Sequence[Object]) -> bool:
        """Determine if the cup is filled coffee."""
        cup, = objects
        cup_name = cup.id_name
        attention_image = state.crop_to_objects([cup])
        return state.evaluate_simple_assertion(f"{cup_name} has coffee in it",
                                               attention_image)

    def _Holding_NSP_holds(self, state: RawState,
                           objects: Sequence[Object]) -> bool:
        """Determine if the robot is holding the jug."""
        robot, jug = objects

        # The block can't be held if the robot's hand is open.
        finger_state = state.get(robot, "fingers")
        if finger_state == 0.4:
            return False

        robot_name = robot.id_name
        jug_name = jug.id_name
        attention_image = state.crop_to_objects([robot, jug])
        return state.evaluate_simple_assertion(
            f"{robot_name} is holding {jug_name}", attention_image)

    def _JugInMachine_NSP_holds(
        self,
        state: RawState,
        objects: Sequence[Object],
    ) -> bool:
        """Determine if the jug is inside the machine."""
        jug, machine = objects
        jug_name = jug.id_name
        machine_name = machine.id_name

        # If the jug is far from the machine, it's not in the machine.
        if state.get(jug, "bbox_left") > state.get(machine, "bbox_right"):
            return False

        attention_image = state.crop_to_objects([jug, machine])
        return state.evaluate_simple_assertion(
            f"{jug_name} is in the coffee machine {machine_name}",
            attention_image)

    def _JugPickable_NSP_holds(self, state: RawState,
                               objects: Sequence[Object]) -> bool:
        """Determine if the jug is pickable."""
        jug, = objects
        jug_rot = state.get(jug, "rot")

        if self.get_name == "coffee":
            pick_jug_rot_tol = self.pick_jug_rot_tol
        else:
            # in pb-coffee determine by grasp check
            # pick_jug_rot_tol = 1/5 * np.pi
            pick_jug_rot_tol = 0.5
            # pick_jug_rot_tol = - np.pi / 4

        return abs(jug_rot) <= pick_jug_rot_tol
        # jug, = objects
        # jug_name = jug.id_name
        # attention_image = state.crop_to_objects([jug])
        # return state.evaluate_simple_assertion(
        #     f"{jug_name}'s handle is pointing to the robot so it can " +
        #     "be directly picked up", attention_image)

    def _JugFilled_NSP_holds(self, state: RawState,
                             objects: Sequence[Object]) -> bool:
        """Determine if the jug is filled with coffee."""
        jug, = objects
        jug_name = jug.id_name
        attention_image = state.crop_to_objects([jug])
        return state.evaluate_simple_assertion(
            f"glass {jug_name} is filled with coffee", attention_image)

    def _OnTable_NSP_holds(self, state: RawState,
                           objects: Sequence[Object]) -> bool:
        """Determine if the jug is on the table."""
        jug, = objects
        jug_name = jug.id_name

        # We know there is only one table in this environment.
        table = state.get_objects(self._table_type)[0]
        table_name = table.id_name
        # Crop the image to the smallest bounding box that include both objects.

        attention_image = state.crop_to_objects([jug, table])
        return state.evaluate_simple_assertion(
            f"glass {jug_name} is directly resting on {table_name}'s surface.",
            attention_image)

    @property
    def oracle_proposed_predicates(self) -> Set[Predicate]:
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
            # Should add: in correct rotation

            # self._Twisting,
            # self._RobotAboveCup,
            # self._JugAboveCup,
            # self._NotAboveCup,
            # self._PressingButton,
            # self._NotSameCup,
        }

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle coffee-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

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

        # Load table.
        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          cls.table_pose,
                                          cls.table_orientation,
                                          physicsClientId=physics_client_id)
        # Uncomment to make the grey table black.
        # p.changeVisualShape(table_id, -1, rgbaColor=[0.1, 0.1, 0.1, 1],
        #                     physicsClientId=physics_client_id)
        bodies["table_id"] = table_id

        ## Load coffee machine.

        # Create the collision shape.
        # half_extents = (
        #     cls.machine_x_len,
        #     cls.machine_y_len / 2,
        #     cls.machine_z_len / 2,
        # )
        # collision_id = p.createCollisionShape(
        #     p.GEOM_BOX,
        #     halfExtents=half_extents,
        #     physicsClientId=physics_client_id)

        # # Create the visual_shape.
        # visual_id = p.createVisualShape(p.GEOM_BOX,
        #                                 halfExtents=half_extents,
        #                                 rgbaColor=(0.2, 0.2, 0.2, 1.0),
        #                                 physicsClientId=physics_client_id)

        # # Create the body.
        # pose = (
        #     cls.machine_x,
        #     cls.machine_y,
        #     cls.z_lb + cls.machine_z_len / 2,
        # )
        # orientation = cls._default_orn
        # machine_id = p.createMultiBody(baseMass=0,
        #                                baseCollisionShapeIndex=collision_id,
        #                                baseVisualShapeIndex=visual_id,
        #                                basePosition=pose,
        #                                baseOrientation=orientation,
        #                                physicsClientId=physics_client_id)

        # bodies["machine_id"] = machine_id

        # # Dispense top
        # top_half_extents = (
        #                     # cls.machine_x_len / 2,
        #                     # cls.machine_x_len / 3,
        #                     cls.machine_x_len * 5 / 6,
        #                     cls.machine_top_y_len / 2,
        #                     cls.machine_z_len / 6)
        # collision_id = p.createCollisionShape(
        #     p.GEOM_BOX,
        #     halfExtents=top_half_extents,
        #     physicsClientId=physics_client_id)

        # # Create the visual_shape.
        # visual_id = p.createVisualShape(p.GEOM_BOX,
        #                                 halfExtents=top_half_extents,
        #                                 rgbaColor=(0.2, 0.2, 0.2, 1.0),
        #                                 physicsClientId=physics_client_id)

        # # Create the body.
        # pose = (
        #     # cls.machine_x - cls.machine_x_len / 2,
        #     # cls.machine_x - cls.machine_x_len / 6,
        #     cls.machine_x - cls.machine_x_len / 6,
        #     # cls.machine_y - cls.machine_y_len/2 - cls.machine_top_y_len / 2,
        #     cls.machine_y - cls.machine_y_len/2 - cls.machine_top_y_len / 2,
        #     cls.z_lb + cls.machine_z_len / 2 + cls.machine_z_len / 3,
        # )
        # orientation = cls._default_orn
        # p.createMultiBody(baseMass=0,
        #                     baseCollisionShapeIndex=collision_id,
        #                     baseVisualShapeIndex=visual_id,
        #                     basePosition=pose,
        #                     baseOrientation=orientation,
        #                     physicsClientId=physics_client_id)

        # new body and top
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
        visual_id_base = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_base,
            # rgbaColor=(0.2, 0.2, 0.2, 1.0),
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
            # cls.machine_x_len * 2 / 3,
            cls.machine_top_y_len / 2,
            cls.machine_z_len / 6,
        )
        collision_id_top = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents_top,
            physicsClientId=physics_client_id)
        visual_id_top = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_top,
            # rgbaColor=(0.2, 0.2, 0.2, 1.0),
            rgbaColor=cls.machine_color,
            physicsClientId=physics_client_id)
        pose_top = (
            -cls.machine_x_len / 6,  # x relative to base
            # -cls.machine_x_len / 3,
            -cls.machine_y_len / 2 -
            cls.machine_top_y_len / 2,  # y relative to base
            # - cls.machine_top_y_len ,
            # cls.machine_z_len / 2 + cls.machine_z_len / 3,  # z relative to base
            cls.machine_z_len / 3)
        orientation_top = cls._default_orn

        # Create the dispense area -- base.
        # Define the dimensions for the dispense area
        dispense_radius = 2 * cls.jug_radius
        dispense_height = 0.005
        half_extents_dispense_base = (cls.machine_x_len,
                                      1.1 * dispense_radius + cls.jug_radius +
                                      0.003, dispense_height)
        collision_id_dispense_base = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents_dispense_base,
            physicsClientId=physics_client_id)
        visual_id_dispense_base = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_dispense_base,
            # rgbaColor=(0.2, 0.2, 0.2, 1.0),
            rgbaColor=cls.machine_color,
            physicsClientId=physics_client_id)
        # the relative position for the dispense area
        pose_dispense_base = (
            0,
            -cls.machine_y_len - dispense_radius + 0.01,
            -cls.machine_z_len / 2,
        )
        orientation_dispense_base = cls._default_orn

        # Create the multibody with a fixed link
        link_mass = 0
        link_collision_id = collision_id_top
        link_visual_id = visual_id_top
        link_position = pose_top
        link_orientation = orientation_top
        link_inertial_frame_position = [0, 0, 0]
        link_inertial_frame_orientation = [0, 0, 0, 1]
        link_parent_frame_position = [0, 0, 0]
        link_parent_frame_orientation = [0, 0, 0, 1]

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

        bodies["machine_id"] = machine_id
        # new end

        ## Create the dispense area -- base.
        # dispense_radius = 2 * cls.jug_radius
        # dispense_height = 0.005
        pose = (
            cls.dispense_area_x,
            cls.dispense_area_y,
            # cls.z_lb + dispense_height)
            cls.z_lb)
        orientation = cls._default_orn
        # half_extents = (cls.machine_x_len,
        #                 1.1 * dispense_radius + cls.jug_radius + 0.003,
        # # half_extents = (1.1 * dispense_radius, 1.1 * dispense_radius,
        #                 dispense_height)

        # # Dispense area square.
        # collision_id = p.createCollisionShape(
        #     p.GEOM_BOX,
        #     halfExtents=half_extents,
        #     physicsClientId=physics_client_id)

        # # Create the visual_shape.
        # visual_id = p.createVisualShape(p.GEOM_BOX,
        #                                 halfExtents=half_extents,
        #                                 rgbaColor=(0.2, 0.2, 0.2, 1.0),
        #                                 physicsClientId=physics_client_id)

        # # Create the body.
        # p.createMultiBody(baseMass=0,
        #                   baseCollisionShapeIndex=collision_id,
        #                   baseVisualShapeIndex=visual_id,
        #                   basePosition=np.add(pose, (0, 0, -dispense_height)),
        #                   baseOrientation=orientation,
        #                   physicsClientId=physics_client_id)

        # Dispense area circle
        # Create the collision shape.
        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=dispense_radius,
            height=dispense_height,
            physicsClientId=physics_client_id)

        # Create the visual_shape.
        visual_id = p.createVisualShape(p.GEOM_CYLINDER,
                                        radius=dispense_radius +
                                        0.8 * cls.jug_radius,
                                        length=dispense_height,
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

        bodies["dispense_area_id"] = dispense_area_id

        # Add a button. Could do this as a link on the machine, but since
        # both never move, it doesn't matter.
        # button_height = cls.button_radius / 2
        # collision_id = p.createCollisionShape(
        #     p.GEOM_CYLINDER,
        #     radius=cls.button_radius,
        #     height=button_height,
        #     physicsClientId=physics_client_id)

        # # Create the visual_shape.
        # visual_id = p.createVisualShape(p.GEOM_CYLINDER,
        #                                 radius=cls.button_radius,
        #                                 length=button_height,
        #                                 rgbaColor=(0.5, 0.2, 0.2, 1.0),
        #                                 physicsClientId=physics_client_id)

        # # Create the body.
        # pose = (
        #     cls.button_x,
        #     cls.button_y,
        #     cls.button_z,
        # )

        # # Facing outward.
        # orientation = p.getQuaternionFromEuler([0.0, np.pi / 2, np.pi / 2])
        # button_id = p.createMultiBody(baseMass=0,
        #                               baseCollisionShapeIndex=collision_id,
        #                               baseVisualShapeIndex=visual_id,
        #                               basePosition=pose,
        #                               baseOrientation=orientation,
        #                               physicsClientId=physics_client_id)
        # Add a button. Could do this as a link on the machine, but since
        # both never move, it doesn't matter.
        button_height = cls.button_radius / 2
        collision_id_button = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=cls.button_radius,
            height=button_height,
            physicsClientId=physics_client_id)

        # Create the visual_shape for the button.
        visual_id_button = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=cls.button_radius,
            length=button_height,
            rgbaColor=cls.button_color_off,
            physicsClientId=physics_client_id)

        # Create the body.
        pose_button = (
            cls.button_x,
            cls.button_y,
            cls.button_z,
        )

        # Facing outward.
        orientation_button = p.getQuaternionFromEuler(
            [0.0, np.pi / 2, np.pi / 2])
        # orientation_button = [0, 0, 0, 1]

        # Add a light bar at the same position as the button.
        half_extents_bar = (
            cls.machine_z_len / 6 - 0.01,  # z
            cls.machine_x_len * 5 / 6,  # x
            cls.machine_top_y_len / 2  # y
        )
        collision_id_light_bar = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents_bar,
            physicsClientId=physics_client_id)
        visual_id_light_bar = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents_bar,
            # rgbaColor=(0.2, 0.2, 0.2, 1.0),
            rgbaColor=cls.button_color_off,
            physicsClientId=physics_client_id)

        # The light bar will have the same pose and orientation as the button.
        link_mass = 0
        link_collision_ids = [collision_id_light_bar]
        link_visual_ids = [visual_id_light_bar]
        # relative position to the button
        link_positions = [[
            cls.machine_z_len / 6 - 0.017,  # larger is down
            cls.machine_x_len / 6 - 0.001,
            cls.machine_top_y_len / 2 - 0.001
        ]]
        link_orientations = [[0, 0, 0, 1]]  # same orientation as the button

        button_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id_button,
            baseVisualShapeIndex=visual_id_button,
            basePosition=pose_button,
            baseOrientation=orientation_button,
            linkMasses=[link_mass],
            linkCollisionShapeIndices=link_collision_ids,
            linkVisualShapeIndices=link_visual_ids,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]],
            physicsClientId=physics_client_id)

        # bodies["combined_button_light_bar_id"] = combined_button_light_bar_id

        bodies["button_id"] = button_id

        # Load coffee jug.

        # Create the body.
        # This pose doesn't matter because it gets overwritten in reset.
        jug_pose = ((cls.jug_init_x_lb + cls.jug_init_x_ub) / 2,
                    (cls.jug_init_y_lb + cls.jug_init_y_ub) / 2,
                    cls.z_lb + cls.jug_height / 2)
        # The jug orientation updates based on the rotation of the state.
        rot = (cls.jug_init_rot_lb + cls.jug_init_rot_ub) / 2
        jug_orientation = p.getQuaternionFromEuler([0.0, 0.0, rot - np.pi / 2])

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # jug_id = p.loadURDF(utils.get_env_asset_path("urdf/cup.urdf"),
        #                     useFixedBase=True,
        #                     globalScaling=0.45,
        #                     physicsClientId=physics_client_id)
        jug_id = p.loadURDF(
            utils.get_env_asset_path("urdf/kettle.urdf"),
            useFixedBase=True,
            # globalScaling=0.075, # original jug
            globalScaling=0.09,  # enlarged jug
            physicsClientId=physics_client_id)

        # Assuming jug_id and physics_client_id are already defined
        # for link_index in range(p.getNumJoints(jug_id,
        #                             physicsClientId=physics_client_id)):
        #     # 0 for body, 1 for lid
        #     p.changeVisualShape(jug_id, link_index,
        #                         rgbaColor=[1, 1, 1, 0.3],
        #                         physicsClientId=physics_client_id)
        # for link_index in range(p.getNumJoints(jug_id,
        #                             physicsClientId=physics_client_id)):
        #     # 0 for body, 1 for lid
        #     p.changeVisualShape(jug_id, link_index,
        #                         rgbaColor=[0.3, 0.3, 0.3, 1],
        #                         physicsClientId=physics_client_id)
        #     p.changeVisualShape(jug_id, link_index,
        #                         rgbaColor=[0.3, 0.3, 0.3, 1],
        #                         physicsClientId=physics_client_id)
        # Make the jug transparent
        # p.changeVisualShape(jug_id, 0, rgbaColor=[1,1,1,0.9],
        p.changeVisualShape(jug_id,
                            0,
                            rgbaColor=cls.jug_color,
                            physicsClientId=physics_client_id)
        # # remove the lid
        p.changeVisualShape(jug_id,
                            1,
                            rgbaColor=[1, 1, 1, 0],
                            physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(jug_id,
                                          jug_pose,
                                          jug_orientation,
                                          physicsClientId=physics_client_id)
        bodies["jug_id"] = jug_id

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._jug_id = pybullet_bodies["jug_id"]
        self._machine_id = pybullet_bodies["machine_id"]
        self._dispense_area_id = pybullet_bodies["dispense_area_id"]
        self._button_id = pybullet_bodies["button_id"]

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
        # self._obj_id_to_obj[self._dispense_area_id] = self._dispense_area
        # self._obj_di_to_obj[self._button_id] = self._button

        # Reset cups based on the state.
        cup_objs = state.get_objects(self._cup_type)
        # Make new cups.
        self._cup_id_to_cup = {}
        self._cup_to_capacity = {}
        for cup_idx, cup_obj in enumerate(cup_objs):
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

        # Remove the liquid in jug
        # for link_index in range(p.getNumJoints(self._jug_id,
        #                             physicsClientId=self._physics_client_id)):
        #     # 0 for body, 1 for lid
        #     p.changeVisualShape(self._jug_id, link_index,
        #                         rgbaColor=[1, 1, 1, 1],
        #                         physicsClientId=self._physics_client_id)
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
            jz = self._get_jug_z(state, self._jug) + self.jug_height / 2
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
            button_color = self.button_color_off
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

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.debug("Desired state:")
            logging.debug(state.pretty_str())
            logging.debug("Reconstructed state:")
            logging.debug(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state."""
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
            "wrist": wrist,
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
        state = utils.PyBulletState(state.data,
                                    simulator_state=joint_positions)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def step(self, action: Action) -> State:
        # What's the previous robot state?
        current_ee_rpy = self._pybullet_robot.forward_kinematics(
            self._pybullet_robot.get_joints()).rpy
        state = super().step(action)
        # If the robot is sufficiently close to the button, turn on the machine
        # and update the status of the jug.
        if self._jug_filled:
            # Move the liquid inside
            pos, quat = p.getBasePositionAndOrientation(
                self._jug_id, physicsClientId=self._physics_client_id)
            p.resetBasePositionAndOrientation(
                self._jug_liquid_id,
                pos,
                quat,
                physicsClientId=self._physics_client_id)

        if self._PressingButton_holds(state, [self._robot, self._machine]):
            if CFG.coffee_mac_requires_jug_to_turn_on:
                if self._JugInMachine_holds(state, [self._jug, self._machine]):
                    p.changeVisualShape(
                        self._button_id,
                        -1,
                        rgbaColor=self.button_color_on,
                        physicsClientId=self._physics_client_id)
                    p.changeVisualShape(
                        self._button_id,
                        0,
                        rgbaColor=self.button_color_on,
                        physicsClientId=self._physics_client_id)
            else:
                p.changeVisualShape(self._button_id,
                                    -1,
                                    rgbaColor=self.button_color_on,
                                    physicsClientId=self._physics_client_id)
                p.changeVisualShape(self._button_id,
                                    0,
                                    rgbaColor=self.button_color_on,
                                    physicsClientId=self._physics_client_id)
            # the jug is only filled if it's in the machine
            if self._JugInMachine_holds(state, [self._jug, self._machine]):
                if not self._jug_filled:
                    self._jug_liquid_id = \
                        self._create_pybullet_liquid_for_jug()
                self._jug_filled = True
            self._current_observation = self._get_state()
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
            self._current_observation = self._get_state()
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
            # print(f"[in step] cur ee rpy {current_ee_rpy}"\
            #     " -- should equal to the ``current ee rpy above``")
            # print(f"[in step] ee rpy {gripper_pose.rpy} -- should equal the ``new ee``"\
            #       " rpy in the policy above")
            # print(f"[in step] ee d_roll {-d_roll:.3f} -- should equal the d_roll in "\
            #       " the policy above")
            if d_roll > 2 * np.pi / 3:
                d_roll

            (jx, jy, jz), orn = p.getBasePositionAndOrientation(
                self._jug_id, physicsClientId=self._physics_client_id)
            jug_yaw = p.getEulerFromQuaternion(orn)[2]
            new_jug_yaw = jug_yaw - d_roll
            new_jug_yaw = utils.wrap_angle(new_jug_yaw)
            # print(f"[in step] jug yaw {jug_yaw + np.pi/2:.3f}")
            jug_orientation = p.getQuaternionFromEuler([0.0, 0.0, new_jug_yaw])
            # print(f"[in step] jug new yaw {jug_yaw-d_roll+np.pi/2:.3f}")
            p.resetBasePositionAndOrientation(
                self._jug_id, [jx, jy, self.z_lb + self.jug_height / 2],
                jug_orientation,
                physicsClientId=self._physics_client_id)

            self._current_observation = self._get_state()
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

    def _create_pybullet_liquid_for_jug(self) -> Optional[int]:
        # current_liquid = state.get(cup, "current_liquid")
        # cup_cap = state.get(cup, "capacity_liquid")
        liquid_height = self.jug_height * 0.7
        liquid_radius = self.jug_radius * 1.5
        # cx = state.get(jug, "x")
        # cy = state.get(jug, "y")
        # cz = self.z_lb
        # for link_index in range(p.getNumJoints(self._jug_id,
        #                             physicsClientId=self._physics_client_id)):
        #     # 0 for body, 1 for lid

        # add color to jug
        # p.changeVisualShape(self._jug_id, 0,
        #                         rgbaColor=[0.2, 0.05, 0.0, 1],
        #                         physicsClientId=self._physics_client_id)

        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=liquid_radius,
            height=liquid_height,
            physicsClientId=self._physics_client_id)

        visual_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=liquid_radius,
            length=liquid_height,
            rgbaColor=(0.2 * 1.5, 0.05 * 1.5, 0.0, 1.0),
            physicsClientId=self._physics_client_id)

        pose, orientation = p.getBasePositionAndOrientation(
            self._jug_id, physicsClientId=self._physics_client_id)
        return p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=collision_id,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=pose,
                                 baseOrientation=orientation,
                                 physicsClientId=self._physics_client_id)
