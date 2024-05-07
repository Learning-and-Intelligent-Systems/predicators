"""A PyBullet version of CoffeeEnv."""

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.coffee import CoffeeEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Array, EnvironmentTask, Object, State


class PyBulletCoffeeEnv(PyBulletEnv, CoffeeEnv):
    """PyBullet Coffee domain."""

    # Need to override a number of settings to conform to the actual dimensions
    # of the robots, table, etc.
    grasp_finger_tol: ClassVar[float] = 1e-2
    grasp_position_tol: ClassVar[float] = 1e-2
    dispense_tol: ClassVar[float] = 1e-2
    pour_angle_tol: ClassVar[float] = 1e-1
    pour_pos_tol: ClassVar[float] = 1e-2
    init_padding: ClassVar[float] = 0.05
    pick_jug_y_padding: ClassVar[float] = 0.05
    pick_jug_rot_tol: ClassVar[float] = np.pi / 3
    safe_z_tol: ClassVar[float] = 1e-2
    place_jug_in_machine_tol: ClassVar[float] = 1e-3
    jug_twist_offset: ClassVar[float] = 0.025
    init_padding: ClassVar[float] = 0.05
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = 0.2
    z_ub: ClassVar[float] = 0.75
    robot_init_x: ClassVar[float] = (x_ub + x_lb) / 2.0
    robot_init_y: ClassVar[float] = (y_ub + y_lb) / 2.0
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.7441, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2])
    # Machine settings.
    machine_x_len: ClassVar[float] = 0.2 * (x_ub - x_lb)
    machine_y_len: ClassVar[float] = 0.15 * (y_ub - y_lb)
    machine_z_len: ClassVar[float] = 0.5 * (z_ub - z_lb)
    machine_x: ClassVar[float] = x_ub - machine_x_len / 2 - init_padding
    machine_y: ClassVar[float] = y_ub - machine_y_len / 2 - init_padding
    button_x: ClassVar[float] = machine_x
    button_y: ClassVar[float] = machine_y - machine_y_len / 2
    button_z: ClassVar[float] = z_lb + 3 * machine_z_len / 4
    button_radius: ClassVar[float] = 0.2 * machine_y_len
    button_color_on: ClassVar[Tuple[float, float, float,
                                    float]] = (0.2, 0.5, 0.2, 1.0)
    plate_color_on: ClassVar[Tuple[float, float, float,
                                   float]] = (0.9, 0.3, 0.0, 0.7)
    button_color_off: ClassVar[Tuple[float, float, float,
                                     float]] = (0.5, 0.2, 0.2, 1.0)
    plate_color_off: ClassVar[Tuple[float, float, float,
                                    float]] = (0.6, 0.6, 0.6, 0.5)
    # Jug settings.
    jug_radius: ClassVar[float] = 0.3 * machine_y_len
    jug_height: ClassVar[float] = 0.15 * (z_ub - z_lb)
    jug_init_x_lb: ClassVar[float] = machine_x - machine_x_len / 2 + init_padding
    jug_init_x_ub: ClassVar[float] = machine_x + machine_x_len / 2 - init_padding
    jug_init_y_lb: ClassVar[float] = y_lb + 3 * jug_radius + init_padding
    jug_init_y_ub: ClassVar[
        float] = machine_y - machine_y_len - 3 * jug_radius - init_padding
    jug_handle_offset: ClassVar[float] = 0.8 * jug_radius
    jug_handle_height: ClassVar[float] = z_lb + jug_height
    jug_handle_radius: ClassVar[float] = jug_handle_height / 3  # for rendering
    # TODO
    jug_init_rot_lb: ClassVar[float] = -1e-5
    jug_init_rot_ub: ClassVar[float] = 1e-5
    # Dispense area settings.
    dispense_area_x: ClassVar[float] = machine_x
    dispense_area_y: ClassVar[float] = machine_y - 3.5 * jug_radius
    # Cup settings.
    cup_radius: ClassVar[float] = 0.6 * jug_radius
    cup_init_x_lb: ClassVar[float] = x_lb + cup_radius + init_padding
    cup_init_x_ub: ClassVar[
        float] = machine_x - machine_x_len / 2 - cup_radius - init_padding
    cup_init_y_lb: ClassVar[float] = jug_init_y_lb
    cup_init_y_ub: ClassVar[float] = jug_init_y_ub
    cup_capacity_lb: ClassVar[float] = 0.075 * (z_ub - z_lb)
    cup_capacity_ub: ClassVar[float] = 0.15 * (z_ub - z_lb)
    cup_target_frac: ClassVar[float] = 0.75  # fraction of the capacity
    cup_colors: ClassVar[List[Tuple[float, float, float, float]]] = [
        (244 / 255, 27 / 255, 63 / 255, 1.),
        (121 / 255, 37 / 255, 117 / 255, 1.),
        (35 / 255, 100 / 255, 54 / 255, 1.),
    ]
    # Table settings.
    table_pose: ClassVar[Pose3D] = (0.75, 1.35, 0.0)
    table_orientation: ClassVar[Quaternion] = p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2])
    # Camera parameters.
    _camera_distance: ClassVar[float] = 0.8
    _camera_yaw: ClassVar[float] = 100.0
    _camera_pitch: ClassVar[float] = -24
    _camera_target: ClassVar[Pose3D] = (0.75, 1.65, 0.42)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Create the cups lazily because they can change size and color.
        self._cup_id_to_cup: Dict[int, Object] = {}
        self._cup_to_liquid_id: Dict[Object, int] = {}
        self._cup_to_capacity: Dict[Object, float] = {}

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
                                [cls.x_ub, cls.y_lb, z],
                                [1.0, 0.0, 0.0],
                                lineWidth=5.0,
                                physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_ub, z],
                                [cls.x_ub, cls.y_ub, z],
                                [1.0, 0.0, 0.0],
                                lineWidth=5.0,
                                physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_lb, z],
                                [cls.x_lb, cls.y_ub, z],
                                [1.0, 0.0, 0.0],
                                lineWidth=5.0,
                                physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_ub, cls.y_lb, z],
                                [cls.x_ub, cls.y_ub, z],
                                [1.0, 0.0, 0.0],
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
        p.addUserDebugLine([0, 0, 0],
                            [0.25, 0, 0],
                            [1.0, 0.0, 0.0],
                            lineWidth=5.0,
                            physicsClientId=physics_client_id)
        p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0],
                            physicsClientId=physics_client_id)
        p.addUserDebugLine([0, 0, 0],
                            [0.0, 0.25, 0],
                            [1.0, 0.0, 0.0],
                            lineWidth=5.0,
                            physicsClientId=physics_client_id)
        p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0],
                            physicsClientId=physics_client_id)
        p.addUserDebugLine([0, 0, 0],
                            [0.0, 0, 0.25],
                            [1.0, 0.0, 0.0],
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
        bodies["table_id"] = table_id


        ## Load coffee machine.

        # Create the collision shape.
        half_extents = (
            cls.machine_x_len / 2,
            cls.machine_y_len / 2,
            cls.machine_z_len / 2,
        )
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=physics_client_id)

        # Create the visual_shape.
        visual_id = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=half_extents,
                                        rgbaColor=(0.7, 0.7, 0.7, 1.0),
                                        physicsClientId=physics_client_id)

        # Create the body.
        pose = (
            cls.machine_x,
            cls.machine_y,
            cls.z_lb + cls.machine_z_len / 2,
        )
        orientation = cls._default_orn
        machine_id = p.createMultiBody(baseMass=0,
                                       baseCollisionShapeIndex=collision_id,
                                       baseVisualShapeIndex=visual_id,
                                       basePosition=pose,
                                       baseOrientation=orientation,
                                       physicsClientId=physics_client_id)

        bodies["machine_id"] = machine_id

        ## Create the dispense area.
        dispense_radius = 2 * cls.jug_radius
        dispense_height = 0.005
        pose = (cls.dispense_area_x, cls.dispense_area_y,
                cls.z_lb + dispense_height)
        orientation = cls._default_orn
        half_extents = [
            1.1 * dispense_radius, 1.1 * dispense_radius, dispense_height
        ]

        # Create a square beneath the dispense area for visual niceness.
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=physics_client_id)

        # Create the visual_shape.
        visual_id = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=half_extents,
                                        rgbaColor=(0.3, 0.3, 0.3, 1.0),
                                        physicsClientId=physics_client_id)

        # Create the body.
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=collision_id,
                          baseVisualShapeIndex=visual_id,
                          basePosition=np.add(pose, (0, 0, -dispense_height)),
                          baseOrientation=orientation,
                          physicsClientId=physics_client_id)

        # Create the collision shape.
        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=dispense_radius,
            height=dispense_height,
            physicsClientId=physics_client_id)

        # Create the visual_shape.
        visual_id = p.createVisualShape(p.GEOM_CYLINDER,
                                        radius=dispense_radius,
                                        length=dispense_height,
                                        rgbaColor=(0.6, 0.6, 0.6, 0.8),
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
        button_height = cls.button_radius / 2
        collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=cls.button_radius,
            height=button_height,
            physicsClientId=physics_client_id)

        # Create the visual_shape.
        visual_id = p.createVisualShape(p.GEOM_CYLINDER,
                                        radius=cls.button_radius,
                                        length=button_height,
                                        rgbaColor=(0.5, 0.2, 0.2, 1.0),
                                        physicsClientId=physics_client_id)

        # Create the body.
        pose = (
            cls.button_x,
            cls.button_y,
            cls.button_z,
        )

        # Facing outward.
        orientation = p.getQuaternionFromEuler([0.0, np.pi / 2, np.pi / 2])
        button_id = p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=collision_id,
                                      baseVisualShapeIndex=visual_id,
                                      basePosition=pose,
                                      baseOrientation=orientation,
                                      physicsClientId=physics_client_id)

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

        jug_id = p.loadURDF(utils.get_env_asset_path("urdf/kettle.urdf"),
                            useFixedBase=True,
                            globalScaling=0.075,
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
        f = self._fingers_state_to_joint(self._pybullet_robot, f)
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

        # Reset cups based on the state.
        cup_objs = state.get_objects(self._cup_type)

        # Remove the old cups.
        for old_cup_id in self._cup_id_to_cup:
            p.removeBody(old_cup_id, physicsClientId=self._physics_client_id)
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
            color = self.cup_colors[cup_idx % len(self.cup_colors)]
            p.changeVisualShape(cup_id,
                                -1,
                                rgbaColor=color,
                                physicsClientId=self._physics_client_id)

            self._cup_id_to_cup[cup_id] = cup_obj

        # Create liquid in cups.
        for liquid_id in self._cup_to_liquid_id.values():
            p.removeBody(liquid_id, physicsClientId=self._physics_client_id)
        self._cup_to_liquid_id.clear()

        for cup in state.get_objects(self._cup_type):
            current_liquid = state.get(cup, "current_liquid")
            cup_cap = state.get(cup_obj, "capacity_liquid")
            liquid_height = self._cup_liquid_to_liquid_height(
                current_liquid, cup_cap)
            liquid_radius = self._cup_to_liquid_radius(cup_cap)
            if current_liquid == 0:
                continue
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
            liquid_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=pose,
                baseOrientation=orientation,
                physicsClientId=self._physics_client_id)
            self._cup_to_liquid_id[cup] = liquid_id

        # NOTE: if the jug is held, the parent class should take care of it.
        if not self._Holding_holds(state, [self._robot, self._jug]):
            assert self._held_obj_to_base_link is None
            jx = state.get(self._jug, "x")
            jy = state.get(self._jug, "y")
            jz = self._get_jug_z(state, self._jug) + self.jug_height / 2
            rot = state.get(self._jug, "rot")
            jug_orientation = p.getQuaternionFromEuler([0.0, 0.0, rot - np.pi / 2])
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
        p.changeVisualShape(self._dispense_area_id,
                            -1,
                            rgbaColor=plate_color,
                            physicsClientId=self._physics_client_id)
        
        # TODO remove
        # while True:
        #     p.stepSimulation(physicsClientId=self._physics_client_id)
        #     import time
        #     time.sleep(0.01)

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

            (x, y, _), _ = p.getBasePositionAndOrientation(
                cup_id, physicsClientId=self._physics_client_id)

            capacity = self._cup_to_capacity[cup]
            target_liquid = capacity * self.cup_target_frac

            # No liquid object is created if the current liquid is 0.
            if cup in self._cup_to_liquid_id:
                liquid_id = self._cup_to_liquid_id[cup]
                liquid_height = p.getVisualShapeData(
                    liquid_id,
                    physicsClientId=self._physics_client_id,
                )[0][3][2]
                current_liquid = self._cup_liquid_height_to_liquid(
                    liquid_height)
            else:
                current_liquid = 0.0

            state_dict[cup] = {
                "x": x,
                "y": y,
                "capacity_liquid": capacity,
                "target_liquid": target_liquid,
                "current_liquid": current_liquid,
            }

        # Get jug state.
        (x, y, _), quat = p.getBasePositionAndOrientation(
            self._jug_id, physicsClientId=self._physics_client_id)
        rot = utils.wrap_angle(p.getEulerFromQuaternion(quat)[2] + np.pi / 2)
        held = (self._jug_id == self._held_obj_id)
        filled = 0.0  # TODO!! need to change color or something when 'full'
        state_dict[self._jug] = {
            "x": x,
            "y": y,
            "rot": rot,
            "is_held": held,
            "is_filled": filled,
        }

        # Get machine state.
        button_color = p.getVisualShapeData(
            self._button_id, physicsClientId=self._physics_client_id)[0][-2]
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

    def _get_tasks(self, num: int, num_cups_lst: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num, num_cups_lst, rng)
        return self._add_pybullet_state_to_tasks(tasks)

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        return {self._jug_id}

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
        return self.tilt_wrist_to_gripper_orn(
            tilt, wrist
        )
    
    @classmethod
    def tilt_wrist_to_gripper_orn(cls, tilt: float, wrist: float) -> Quaternion:
        """Public for oracle options."""
        if abs(tilt - cls.robot_init_tilt) > cls.pour_angle_tol:
            return p.getQuaternionFromEuler(
                [0.0, np.pi / 2 + tilt, -np.pi / 2])
        return p.getQuaternionFromEuler([0.0, np.pi / 2, wrist - np.pi / 2])

    def _gripper_orn_to_tilt_wrist(self,
                                   orn: Quaternion) -> Tuple[float, float]:
        _, offset_tilt, offset_wrist = p.getEulerFromQuaternion(orn)
        tilt = utils.wrap_angle(offset_tilt - np.pi / 2)
        wrist = utils.wrap_angle(offset_wrist + np.pi / 2)
        return (tilt, wrist)

    @classmethod
    def _fingers_state_to_joint(cls, pybullet_robot: SingleArmPyBulletRobot,
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
        scale = 1.5 * np.sqrt(capacity / self.cup_capacity_ub)
        return liquid * scale

    def _cup_to_liquid_radius(self, capacity: float) -> float:
        scale = 1.5 * np.sqrt(capacity / self.cup_capacity_ub)
        return self.cup_radius * scale

    def _cup_liquid_height_to_liquid(self, height: float) -> float:
        import ipdb
        ipdb.set_trace()
