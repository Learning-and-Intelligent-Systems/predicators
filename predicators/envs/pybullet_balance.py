"""Making a demo video: 
python predicators/main.py --approach oracle --env pybullet_balance --seed 1 \
--num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --make_test_videos \
--sesame_task_planning_heuristic "goal_count" \
--excluded_predicates "Balanced,OnPlate" --sesame_max_skeletons_optimized 100 \
--sesame_check_expected_atoms False
"""
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, \
    Union

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, ConceptPredicate, \
    EnvironmentTask, GroundAtom, NSPredicate, Object, Predicate, State, Type
from predicators.utils import RawState, VLMQuery


class PyBulletBalanceEnv(PyBulletEnv):
    """PyBullet Balance domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Table parameters.
    _table_height: ClassVar[float] = 0.4
    _table2_pose: ClassVar[Pose3D] = (1.35, 0.75, _table_height / 2)
    _table_x, _table2_y, _table_z = _table2_pose
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)
    _table_mid_w = 0.1
    _table_side_w = 0.3
    _table_gap = 0.05
    _table_mid_half_extents = [0.1, _table_mid_w / 2, _table_height / 2]

    # Plate
    _plate_height: ClassVar[float] = 0.02
    _plate_z = _table_height - _plate_height * 3
    _plate1_pose: ClassVar[Pose3D] = (_table_x, _table2_y - _table_mid_w / 2 -
                                      _table_side_w / 2 - _table_gap, _plate_z)
    _plate3_pose: ClassVar[Pose3D] = (_table_x, _table2_y + _table_mid_w / 2 +
                                      _table_side_w / 2 + _table_gap, _plate_z)
    _plate_half_extents = (0.25, _table_side_w / 2, _plate_height)
    # Under plate beams
    _beam1_pose: ClassVar[Pose3D] = (_table_x,
                                     (_plate1_pose[1] + _table2_pose[1]) / 2,
                                     _plate_z - 4 * _plate_height)
    _beam2_pose: ClassVar[Pose3D] = (_table_x,
                                     (_plate3_pose[1] + _table2_pose[1]) / 2,
                                     _plate_z - 4 * _plate_height)
    _beam_half_extents = [0.01, 0.15, _plate_height]

    # Button on table
    _button_radius = 0.04
    _button_color_off = [1, 0, 0, 1]
    _button_color_on = [0, 1, 0, 1]
    button_x, button_y, button_z = _table_x, _table2_y, _table_height
    button_press_threshold = 1e-3

    # Workspace parameters
    x_lb: ClassVar[float] = 1.325
    x_ub: ClassVar[float] = 1.375
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    z_lb: ClassVar[float] = _table_height
    z_ub: ClassVar[float] = 0.75 + _table_height / 2
    y_plate1_ub: ClassVar[float] = _plate1_pose[1] + _table_side_w / 2 - 0.1
    y_plate3_lb: ClassVar[float] = _plate3_pose[1] - _table_side_w / 2 + 0.1

    # Robot parameters
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = z_ub - 0.1
    held_tol: ClassVar[float] = 0.5
    on_tol: ClassVar[float] = 0.01
    collision_padding: ClassVar[float] = 2.0

    _camera_target: ClassVar[Pose3D] = (1.65, 0.75, 0.52)

    _block_mass: ClassVar[float] = 1
    _block_size = CFG.balance_block_size
    _num_blocks_train = CFG.balance_num_blocks_train
    _num_blocks_test = CFG.balance_num_blocks_test

    def __init__(self, use_gui: bool = True) -> None:
        # Types
        bbox_features = ["bbox_left", "bbox_right", "bbox_upper", "bbox_lower"]
        self._block_type = Type("block", [
            "pose_x", "pose_y", "pose_z", "held", "color_r", "color_g",
            "color_b"
        ] + (bbox_features if CFG.env_include_bbox_features else []))
        self._robot_type = Type(
            "robot", ["pose_x", "pose_y", "pose_z", "fingers"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._plate_type = Type(
            "plate", ["pose_z"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._machine_type = Type(
            "machine", ["is_on"] +
            (bbox_features if CFG.env_include_bbox_features else []))

        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._plate1 = Object("plate1", self._plate_type)
        # self._table2 = Object("table2", self._plate_type)
        self._plate3 = Object("plate3", self._plate_type)
        self._machine = Object("mac", self._machine_type)

        super().__init__(use_gui)

        # Predicates
        self._DirectlyOn = Predicate(
            "DirectlyOn",
            [self._block_type, self._block_type],
            self._DirectlyOn_holds,
            # lambda objs:
            # f"{objs[0]} is directly on top of {objs[1]} with no blocks in between."
        )
        self._DirectlyOnPlate = Predicate(
            "DirectlyOnPlate",
            [self._block_type, self._plate_type],
            self._DirectlyOnPlate_holds,
        )
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding", [self._block_type],
                                  self._Holding_holds)
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)
        self._MachineOn = Predicate("MachineOn",
                                    [self._machine_type, self._robot_type],
                                    self._MachineOn_holds)
        self._Balanced = Predicate("Balanced",
                                   [self._plate_type, self._plate_type],
                                   self._Balanced_holds)
        self._ClearPlate = Predicate("ClearPlate", [self._plate_type],
                                     self._ClearPlate_holds)

        self._OnPlate_abs = ConceptPredicate(
            "OnPlate",
            [self._block_type, self._plate_type],
            self._OnPlate_CP_holds,
        )
        self._Balanced_abs = ConceptPredicate(
            "Balanced",
            [self._plate_type, self._plate_type],
            # self._EqualBlocksOnPlates_CP_holds,
            self._Balanced_CP_holds,
            untransformed_predicate=self._Balanced)

        self._DirectlyOn_NSP = NSPredicate(
            "DirectlyOn", [self._block_type, self._block_type],
            self._DirectlyOn_NSP_holds)
        self._DirectlyOnPlate_NSP = NSPredicate(
            "DirectlyOnPlate", [self._block_type],
            self._DirectlyOnPlate_NSP_holds)
        self._Holding_NSP = NSPredicate("Holding", [self._block_type],
                                        self._Holding_NSP_holds)
        self._GripperOpen_NSP = NSPredicate("GripperOpen", [self._robot_type],
                                            self._GripperOpen_NSP_holds)
        self._Clear_NSP = NSPredicate("Clear", [self._block_type],
                                      self._Clear_NSP_holds)

        # We track the correspondence between PyBullet object IDs and Object
        # instances for blocks. This correspondence changes with the task.
        self._block_id_to_block: Dict[int, Object] = {}

        self.ns_to_sym_predicates: Dict[Tuple[str], Predicate] = {
            ("GripperOpen"): self._GripperOpen,
            ("Holding"): self._Holding,
            ("Clear"): self._Clear,
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._DirectlyOn, self._DirectlyOnPlate, self._GripperOpen,
            self._Holding, self._Clear, self._MachineOn, self._ClearPlate,
            self._Balanced_abs, self._OnPlate_abs
        }

    @property
    def concept_predicates(self) -> Set[ConceptPredicate]:
        return {self._Balanced_abs}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        if CFG.balance_holding_goals:
            return {self._Holding}
        return {self._DirectlyOn, self._DirectlyOnPlate}

    @property
    def types(self) -> Set[Type]:
        return {
            self._block_type, self._robot_type, self._plate_type,
            self._machine_type
        }

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle blocks-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        table2_id = create_pybullet_block(
            (.9, .9, .9, 1),
            cls._table_mid_half_extents,
            0.0,  # mass
            1.0,  # friction
            cls._table2_pose,
            cls._table_orientation,
            physics_client_id,
        )

        plate3_id = create_pybullet_block(
            (.9, .9, .9, 1),
            cls._plate_half_extents,
            1.0,
            1.0,
            cls._plate3_pose,
            cls._table_orientation,
            physics_client_id,
        )

        plate1_id = create_pybullet_block(
            (.9, .9, .9, 1),
            cls._plate_half_extents,
            1.0,
            1.0,
            cls._plate1_pose,
            cls._table_orientation,
            physics_client_id,
        )
        bodies["table_ids"] = [plate1_id, plate3_id, table2_id]

        beam1_id = create_pybullet_block(
            (0.9, 0.9, 0.9, 1),
            cls._beam_half_extents,
            1.0,
            1.0,
            cls._beam1_pose,
            cls._table_orientation,
            physics_client_id,
        )
        beam2_id = create_pybullet_block(
            (0.9, 0.9, 0.9, 1),
            cls._beam_half_extents,
            1.0,
            1.0,
            cls._beam2_pose,
            cls._table_orientation,
            physics_client_id,
        )
        bodies["beam_ids"] = [beam1_id, beam2_id]
        cls.fix_plates_and_beams_in_place(physics_client_id, table2_id, plate1_id, 
                                          plate3_id, beam1_id, beam2_id)


        button_id = create_pybullet_block(
            cls._button_color_off,
            [cls._button_radius] * 3,
            1.0,
            1.0,
            (cls.button_x, cls.button_y, cls.button_z),
            cls._table_orientation,
            physics_client_id,
        )
        bodies["button_id"] = button_id

        # Create blocks. Note that we create the maximum number once, and then
        # later on, in reset_state(), we will remove blocks from the workspace
        # (teleporting them far away) based on which ones are in the state.
        num_blocks = max(max(CFG.blocks_num_blocks_train),
                         max(CFG.blocks_num_blocks_test))
        block_ids = []
        block_size = CFG.blocks_block_size
        for i in range(num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            half_extents = (block_size / 2.0, block_size / 2.0,
                            block_size / 2.0)
            block_ids.append(
                create_pybullet_block(color,
                                      half_extents,
                                      cls._block_mass,
                                      cls._obj_friction,
                                      physics_client_id=physics_client_id))
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def fix_plates_and_beams_in_place(physics_client_id, table_id, plate1_id, 
                                      plate3_id, beam1_id, beam2_id):
    # Doesn't work for some reason
        for child_id in [plate1_id, plate3_id, beam1_id, beam2_id]:
            parent_pos, parent_orn = p.getBasePositionAndOrientation(table_id, 
                                            physicsClientId=physics_client_id)
            child_pos, child_orn = p.getBasePositionAndOrientation(child_id, 
                                            physicsClientId=physics_client_id)
            rel_pos, rel_orn = p.multiplyTransforms(
                p.invertTransform(parent_pos, parent_orn)[0],
                p.invertTransform(parent_pos, parent_orn)[1],
                child_pos,
                child_orn
            )
            p.createConstraint(
                parentBodyUniqueId=table_id,
                parentLinkIndex=-1,
                childBodyUniqueId=child_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=rel_pos,
                parentFrameOrientation=rel_orn,
                childFramePosition=(0, 0, 0),
                childFrameOrientation=(0, 0, 0),
                physicsClientId=physics_client_id
            )

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._plate1.id = pybullet_bodies["table_ids"][0]
        self._plate3.id = pybullet_bodies["table_ids"][1]
        self._table_id = pybullet_bodies["table_ids"][2]
        self._machine.id = pybullet_bodies["button_id"]
        self._robot.id = self._pybullet_robot.robot_id
        self._block_ids = pybullet_bodies["block_ids"]
        self._beam_ids = pybullet_bodies["beam_ids"]

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_balance"

    # -------------------------------------------------------------------------
    # State Management: Get, (Re)Set, Step

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state.

        Note that in addition to the state inside PyBullet itself, this
        uses self._block_id_to_block and self._held_obj_id. As long as
        the PyBullet internal state is only modified through reset() and
        step(), these all should remain in sync.
        """
        state_dict = {}

        # Get robot state.
        rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(self._pybullet_robot, rf)
        state_dict[self._robot] = np.array([rx, ry, rz, fingers],
                                           dtype=np.float32)
        joint_positions = self._pybullet_robot.get_joints()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            held = (block_id == self._held_obj_id)
            visual_data = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0]
            r, g, b, _ = visual_data[7]
            # pose_x, pose_y, pose_z, held
            state_dict[block] = np.array([bx, by, bz, held, r, g, b],
                                         dtype=np.float32)

        # Get machine state.
        button_color = p.getVisualShapeData(
            self._machine.id, physicsClientId=self._physics_client_id)[0][-1]
        button_color_on_dist = sum(
            np.subtract(button_color, self._button_color_on)**2)
        button_color_off_dist = sum(
            np.subtract(button_color, self._button_color_off)**2)
        machine_on = float(button_color_on_dist < button_color_off_dist)
        state_dict[self._machine] = np.array([machine_on], dtype=np.float32)

        # Get table state.
        plate1_pos, _ = p.getBasePositionAndOrientation(
            self._plate1.id, physicsClientId=self._physics_client_id)
        plate1_z = plate1_pos[2]
        state_dict[self._plate1] = np.array([plate1_z], dtype=np.float32)

        plate3_pos, _ = p.getBasePositionAndOrientation(
            self._plate3.id, physicsClientId=self._physics_client_id)
        plate3_z = plate3_pos[2]
        state_dict[self._plate3] = np.array([plate3_z], dtype=np.float32)

        state = utils.PyBulletState(state_dict,
                                    simulator_state=joint_positions)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def step(self, action: Action, render_obs: bool = False) -> State:
        state = super().step(action, render_obs=render_obs)

        self._update_balance_beam(state)
        self.fix_plates_and_beams_in_place(self._physics_client_id, self._table_id, 
                                          self._plate1.id, self._plate3.id, 
                                          self._beam_ids[0], self._beam_ids[1])

        # Turn machine on
        if self._PressingButton_holds(state, [self._robot, self._machine]):
            if self._Balanced_holds(state, [self._plate1, self._plate3]):
                p.changeVisualShape(self._machine.id,
                                    -1,
                                    rgbaColor=self._button_color_on,
                                    physicsClientId=self._physics_client_id)
            self._current_observation = self._get_state()
            state = self._current_observation.copy()

        return state


    def _reset_state(self, state: State) -> None:
        """Run super(), then handle blocks-specific resetting."""
        super()._reset_state(state)

        # Reset blocks based on the state.
        plate1_z = state.get(self._plate1, "pose_z")
        p.resetBasePositionAndOrientation(
            self._plate1.id,
            [self._plate1_pose[0], self._plate1_pose[1], plate1_z],
            self._table_orientation,
            physicsClientId=self._physics_client_id)

        # --- ADDED: reset plate3 pose_z from the state
        plate3_z = state.get(self._plate3, "pose_z")
        p.resetBasePositionAndOrientation(
            self._plate3.id,
            [self._plate3_pose[0], self._plate3_pose[1], plate3_z],
            self._table_orientation,
            physicsClientId=self._physics_client_id)

        block_objs = state.get_objects(self._block_type)
        self._block_id_to_block = {}
        self._objects = [
            self._robot, self._plate1, self._plate3, self._machine
        ]
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            self._block_id_to_block[block_id] = block_obj
            block_obj.id = block_id
            self._objects.append(block_obj)
            bx = state.get(block_obj, "pose_x")
            by = state.get(block_obj, "pose_y")
            bz = state.get(block_obj, "pose_z")
            p.resetBasePositionAndOrientation(
                block_id, [bx, by, bz],
                self._default_orn,
                physicsClientId=self._physics_client_id)
            # Update the block color. RGB values are between 0 and 1.
            r = state.get(block_obj, "color_r")
            g = state.get(block_obj, "color_g")
            b = state.get(block_obj, "color_b")
            color = (r, g, b, 1.0)  # alpha = 1.0
            p.changeVisualShape(block_id,
                                linkIndex=-1,
                                rgbaColor=color,
                                physicsClientId=self._physics_client_id)

        # Check if we're holding some block.
        held_block = self._get_held_block(state)
        if held_block is not None:
            self._force_grasp_object(held_block)

        # For any blocks not involved, put them out of view.
        h = self._block_size
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]
            assert block_id not in self._block_id_to_block
            p.resetBasePositionAndOrientation(
                block_id, [oov_x, oov_y, i * h],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # Reset the difference to zero on environment reset
        self._prev_diff = 0
        # Also do one beam update to make sure the initial positions match
        self._update_balance_beam(state)
        self.fix_plates_and_beams_in_place(self._physics_client_id, self._table_id, 
                                          self._plate1.id, self._plate3.id, 
                                          self._beam_ids[0], self._beam_ids[1])

        # Update the button color
        if self._MachineOn_holds(state, [self._machine, self._robot]):
            button_color = self._button_color_on
        else:
            button_color = self._button_color_off
        p.changeVisualShape(self._machine.id,
                            -1,
                            rgbaColor=button_color,
                            physicsClientId=self._physics_client_id)

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            # logging.debug("Desired state:")
            # logging.debug(state.pretty_str())
            # logging.debug("Reconstructed state:")
            # logging.debug(reconstructed_state.pretty_str())
            logging.warning("Could not reconstruct state exactly!")

    def _update_balance_beam(self, state: State) -> None:
        """Shift the plates, beams, *and blocks on them* to simulate a
        balance, ensuring rising sides move blocks first then plate,
        and dropping sides move plate first then blocks."""
        left_count = self.count_num_blocks(state, self._plate1)
        right_count = self.count_num_blocks(state, self._plate3)
        diff = left_count - right_count
        if diff == self._prev_diff:
            return

        shift_per_block = 0.01
        shift_amount = abs(diff) * shift_per_block
        block_objs = state.get_objects(self._block_type)
        left_dropping = diff > 0
        right_dropping = diff < 0

        def shift_blocks(is_left: bool, dropping: bool):
            """Shift blocks for one side, dropping or rising."""
            sign = -1 if dropping else 1
            midpoint_y = self._table2_y
            for block_obj in block_objs:
                # Skip out-of-view or held
                if state.get(block_obj, "pose_z") < 0 or \
                   self._held_obj_id == block_obj.id:
                    continue
                by = state.get(block_obj, "pose_y")
                belongs_to_side = (by < midpoint_y) if is_left else (by > midpoint_y)
                if belongs_to_side:
                    old_z = state.get(block_obj, "pose_z")
                    padding = 0
                    new_z = old_z + (sign * shift_amount) + (sign * padding)
                    block_pos, block_orn = p.getBasePositionAndOrientation(
                        block_obj.id, physicsClientId=self._physics_client_id)
                    p.resetBasePositionAndOrientation(
                        block_obj.id, [block_pos[0], block_pos[1], new_z],
                        block_orn, physicsClientId=self._physics_client_id)

        def shift_plate(is_left: bool, dropping: bool):
            """Shift plate & beam, dropping or rising."""
            sign = -1 if dropping else 1
            if is_left:
                plate_id, beam_id = self._plate1.id, self._beam_ids[0]
                base_plate_z, base_beam_z = self._plate1_pose[2], self._beam1_pose[2]
            else:
                plate_id, beam_id = self._plate3.id, self._beam_ids[1]
                base_plate_z, base_beam_z = self._plate3_pose[2], self._beam2_pose[2]

            new_plate_z = base_plate_z + (sign * shift_amount)
            new_beam_z = base_beam_z + (sign * shift_amount)

            plate_pos, plate_orn = p.getBasePositionAndOrientation(
                plate_id, physicsClientId=self._physics_client_id)
            p.resetBasePositionAndOrientation(
                plate_id, [plate_pos[0], plate_pos[1], new_plate_z],
                plate_orn, physicsClientId=self._physics_client_id)

            beam_pos, beam_orn = p.getBasePositionAndOrientation(
                beam_id, physicsClientId=self._physics_client_id)
            p.resetBasePositionAndOrientation(
                beam_id, [beam_pos[0], beam_pos[1], new_beam_z],
                beam_orn, physicsClientId=self._physics_client_id)

        # Left side update
        if left_dropping:
            # Drop left plate
            shift_plate(True, True)
            # Drop left blocks
            # Rise right blocks
            shift_blocks(False, False)
            # Rise right plate
            shift_plate(False, False)
        else:
            shift_blocks(True, False)
            shift_plate(True, False)
            shift_plate(False, True)
            shift_blocks(False, True)

        # Right side update

        self._prev_diff = diff


    # -------------------------------------------------------------------------
    # Predicates
    def _OnPlate_CP_holds(self, atoms: Set[GroundAtom],
                          objects: Sequence[Object]) -> bool:
        x, y = objects
        for atom in atoms:
            if atom.predicate == self._DirectlyOnPlate and\
               atom.objects == [x, y]:
                return True
        other_blocks = {
            a.objects[0]
            for a in atoms if a.predicate == self._DirectlyOn
            or a.predicate == self._OnPlate_abs
        }

        for other_block in other_blocks:
            holds1 = False
            for atom in atoms:
                if atom.predicate == self._DirectlyOn and\
                   atom.objects == [x, other_block]:
                    holds1 = True
                    break
            if holds1 and self._OnPlate_CP_holds(atoms, [other_block, y]):
                return True
        return False

    def _ClearPlate_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        plate, = objects
        for block in state.get_objects(self._block_type):
            if self._DirectlyOnPlate_holds(state, [block, plate]):
                return False
        return True

    def _MachineOn_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        machine, _ = objects
        return state.get(machine, "is_on") > 0.5

    def _PressingButton_holds(self, state: State,
                              objects: Sequence[Object]) -> bool:
        robot, _ = objects
        button_pos = (self.button_x, self.button_y,
                      self.button_z + self._button_radius)
        x = state.get(robot, "pose_x")
        y = state.get(robot, "pose_y")
        z = state.get(robot, "pose_z")
        sq_dist_to_button = np.sum(np.subtract(button_pos, (x, y, z))**2)
        return bool(sq_dist_to_button < self.button_press_threshold)

    # Function to count the number of blocks in the tower
    def count_num_blocks(self, state: State, table: Object) -> int:

        def count_recursive(base_obj, count):
            for block in state.get_objects(self._block_type):
                if base_obj.type == self._block_type and\
                        self._DirectlyOn_holds(state, [block, base_obj]):
                    count = count_recursive(block, count + 1)
                elif base_obj.type == self._plate_type and\
                        self._DirectlyOnPlate_holds(state, [block,
                                                            base_obj]):
                    count = count_recursive(block, count + 1)
            return count

        return count_recursive(table, 0)

    def _Balanced_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if the blocks are balanced on the table."""
        plate1, plate2 = objects
        if plate1 == plate2:
            return False

        # Get the height of the blocks using recursion
        height1 = self.count_num_blocks(state, plate1)
        height2 = self.count_num_blocks(state, plate2)

        return height1 == height2

    def _EqualBlocksOnPlates_CP_holds(self, atoms: Set[GroundAtom],
                                      objects: Sequence[Object]) -> bool:
        left_plate, right_plate = objects
        if left_plate == right_plate:
            return False
        left_count = 0
        right_count = 0
        for atom in atoms:
            if atom.predicate == self._OnPlate_abs and\
               atom.objects[1] == left_plate:
                left_count += 1
            if atom.predicate == self._OnPlate_abs and\
               atom.objects[1] == right_plate:
                right_count += 1
        # logging.debug(f"left: {left_count}, right: {right_count}")
        return left_count == right_count

    def _Balanced_CP_holds(self, atoms: Set[GroundAtom],
                           objects: Sequence[Object]) -> bool:
        """Check if the blocks are balanced on the table."""
        plate1, table2 = objects
        if plate1 == table2:
            return False
        # Function to count the number of blocks in the tower
        def count_num_blocks(table):

            def count_recursive(base_obj, count):
                for atom in atoms:
                    if atom.predicate == self._DirectlyOn and\
                            atom.objects[1] == base_obj:
                        count = count_recursive(atom.objects[0], count + 1)
                    elif atom.predicate == self._DirectlyOnPlate and\
                            atom.objects[1] == base_obj:
                        count = count_recursive(atom.objects[0], count + 1)
                return count

            return count_recursive(table, 0)

        # Get the height of the blocks using recursion
        height1 = count_num_blocks(plate1)
        height2 = count_num_blocks(table2)

        return height1 == height2

    def _DirectlyOn_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        if state.get(block1, "held") >= self.held_tol or \
           state.get(block2, "held") >= self.held_tol:
            return False
        x1 = state.get(block1, "pose_x")
        y1 = state.get(block1, "pose_y")
        z1 = state.get(block1, "pose_z")
        x2 = state.get(block2, "pose_x")
        y2 = state.get(block2, "pose_y")
        z2 = state.get(block2, "pose_z")
        return np.allclose([x1, y1, z1], [x2, y2, z2 + self._block_size],
                           atol=self.on_tol)

    def _DirectlyOnPlate_holds(self, state: State,
                               objects: Sequence[Object]) -> bool:
        block, table = objects
        y = state.get(block, "pose_y")
        z = state.get(block, "pose_z")
        table_z = state.get(table, "pose_z") + self._plate_height / 2
        desired_z = table_z + self._block_size * 0.5

        if (state.get(block, "held") < self.held_tol) and \
                (desired_z-self.on_tol < z < desired_z+self.on_tol):
            if table.name == "plate1":
                return y < self._table2_y
            elif table.name == "plate3":
                return y > self._table2_y
            else:
                raise ValueError("Invalid table name")
        else:
            return False

    @staticmethod
    def _GripperOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        rf = state.get(robot, "fingers")
        assert rf in (0.0, 1.0)
        return rf == 1.0

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return self._get_held_block(state) == block

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        if self._Holding_holds(state, objects):
            return False
        block, = objects
        for other_block in state:
            if other_block.type != self._block_type:
                continue
            if self._DirectlyOn_holds(state, [other_block, block]):
                return False
        return True

    def _get_held_block(self, state: State) -> Optional[Object]:
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            if state.get(block, "held") >= self.held_tol:
                return block
        return None

    def _Clear_NSP_holds(self, state: RawState, objects: Sequence[Object]) -> \
            Union[bool, VLMQuery]:
        """Is there no block on top of the block."""
        block, = objects
        for other_block in state:
            if other_block.type != self._block_type:
                continue
            if self._DirectlyOn_holds(state, [other_block, block]):
                return False
        return True

    def _Holding_NSP_holds(self, state: RawState, objects: Sequence[Object]) ->\
            bool:
        """Is the robot holding the block."""
        block, = objects

        # The block can't be held if the robot's hand is open.
        # We know there is only one robot in this environment.
        robot = state.get_objects(self._robot_type)[0]
        if self._GripperOpen_NSP_holds(state, [robot]):
            return False

        # Using simple heuristics to check if they have overlap
        block_bbox = state.get_obj_bbox(block)
        robot_bbox = state.get_obj_bbox(robot)
        if block_bbox.right < robot_bbox.left or \
            block_bbox.left > robot_bbox.right or\
            block_bbox.upper < robot_bbox.lower or\
            block_bbox.lower > robot_bbox.upper:
            return False

        block_name = block.id_name
        attention_image = state.crop_to_objects([block, robot])
        return state.evaluate_simple_assertion(
            f"{block_name} is held by the robot", attention_image)

    def _GripperOpen_NSP_holds(self, state: RawState, objects: Sequence[Object]) ->\
            bool:
        """Is the robots gripper open."""
        robot, = objects
        finger_state = state.get(robot, "fingers")
        assert finger_state in (0.0, 1.0)
        return finger_state == 1.0


    def _DirectlyOnPlate_NSP_holds(state: RawState, objects:Sequence[Object]) ->\
            bool:
        """Determine if the block in objects is directly resting on the table's
        surface in the scene image."""
        block, = objects
        block_name = block.id_name

        # We know there is only one table in this environment.
        plate = state.get_objects(self._plate_type)[0]
        plate_name = plate.id_name
        # Crop the image to the smallest bounding box that include both objects.
        attention_image = state.crop_to_objects([block, plate])

        return state.evaluate_simple_assertion(
            f"{block_name} is directly resting on {plate_name}'s surface.",
            attention_image)

    def _DirectlyOn_NSP_holds(state: RawState,
                              objects: Sequence[Object]) -> bool:
        """Determine if the first block in objects is directly on top of the
        second block with no blocks in between in the scene image, by using a
        combination of rules and VLMs."""

        block1, block2 = objects
        block1_name, block2_name = block1.id_name, block2.id_name

        # We know a block can't be on top of itself.
        if block1_name == block2_name:
            return False

        # Situations where we're certain that block1 won't be above block2
        if state.get(block1, "bbox_lower") < state.get(block2, "bbox_lower") or\
           state.get(block1, "bbox_left") > state.get(block2, "bbox_right") or\
           state.get(block1, "bbox_right") < state.get(block2, "bbox_left") or\
           state.get(block1, "bbox_upper") < state.get(block2, "bbox_upper") or\
           state.get(block1, "pose_z") < state.get(block2, "pose_z"):
            return False

        # Use a VLM query to handle to reminder cases
        # Crop the scene image to the smallest bounding box that include both
        # objects.
        attention_image = state.crop_to_objects([block1, block2])
        return state.evaluate_simple_assertion(
            f"{block1_name} is directly on top of {block2_name} with no " +
            "blocks in between.", attention_image)

    # -------------------------------------------------------------------------
    # Task Generation
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                possible_num_blocks=self._num_blocks_train,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                possible_num_blocks=self._num_blocks_test,
                                rng=self._test_rng)

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _force_grasp_object(self, block: Object) -> None:
        block_to_block_id = {b: i for i, b in self._block_id_to_block.items()}
        block_id = block_to_block_id[block]
        # The block should already be held. Otherwise, the position of the
        # block was wrong in the state.
        held_obj_id = self._detect_held_object()
        assert block_id == held_obj_id
        # Create the grasp constraint.
        self._held_obj_id = block_id
        self._create_grasp_constraint()

    def _make_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for idx in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks, p=[0.3, 0.7])
            piles = self._sample_initial_piles(num_blocks, rng)
            init_state = self._sample_state_from_piles(piles, rng)
            goal = {
                GroundAtom(self._MachineOn, [self._machine, self._robot]),
                # GroundAtom(self._DirectlyOn, [piles[1][3], piles[0][1]]),
                }
                # }
            # while True:  # repeat until goal is not satisfied
            #     goal = self._sample_goal_from_piles(num_blocks, piles, rng)
            #     if not all(goal_atom.holds(init_state) for goal_atom in goal):
            #         break
            # if idx == 0:
            # }
            # while True:  # repeat until goal is not satisfied
            #     goal = self._sample_goal_from_piles(num_blocks, piles, rng)
            #     if not all(goal_atom.holds(init_state) for goal_atom in goal):
            #         break
            # if idx == 0:
            tasks.append(EnvironmentTask(init_state, goal))
        return self._add_pybullet_state_to_tasks(tasks)

    def _sample_initial_piles(self, num_blocks: int,
                              rng: np.random.Generator) -> List[List[Object]]:
        n_piles = 0
        piles: List[List[Object]] = []
        for block_num in range(num_blocks):
            block = Object(f"block{block_num}", self._block_type)
            # If coin flip, start new pile
            # if (block_num == 0 or rng.uniform() < 0.2) and n_piles < 2:
            # increase the chance of starting a new pile
            # if (block_num == 0 or rng.uniform() < 0.4) and n_piles < 2:
            #     n_piles += 1
            #     piles.append([])
            # For generating a 1:5 pile
            if (block_num == 0 or block_num == 1):
                n_piles += 1
                piles.append([])
            # Add block to pile
            piles[-1].append(block)
        return piles

    def _sample_state_from_piles(self, piles: List[List[Object]],
                                 rng: np.random.Generator) -> State:
        data: Dict[Object, Array] = {}
        # Create objects
        block_to_pile_idx = {}
        for i, pile in enumerate(piles):
            for j, block in enumerate(pile):
                assert block not in block_to_pile_idx
                block_to_pile_idx[block] = (i, j)
        # Sample pile (x, y)s
        pile_to_xy: Dict[int, Tuple[float, float]] = {}
        for i in range(len(piles)):
            pile_to_xy[i] = self._sample_initial_pile_xy(
                rng, set(pile_to_xy.values()))
        # Create block states
        for block, pile_idx in block_to_pile_idx.items():
            pile_i, pile_j = pile_idx
            x, y = pile_to_xy[pile_i]
            # Example: 0.2 + 0.045 * 0.5
            z = self._plate_z + self._plate_height + \
                    self._block_size * (0.5 + pile_j)
            r, g, b = rng.uniform(size=3)
            if "clear" in self._block_type.feature_names:
                # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b,
                # clear]
                # Block is clear iff it is at the top of a pile
                clear = pile_j == len(piles[pile_i]) - 1
                data[block] = np.array([x, y, z, 0.0, r, g, b, clear])
            else:
                # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b]
                data[block] = np.array([x, y, z, 0.0, r, g, b])
        # [pose_x, pose_y, pose_z, fingers]
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = 1.0  # fingers start out open
        data[self._robot] = np.array([rx, ry, rz, rf], dtype=np.float32)
        data[self._plate1] = np.array([self._plate1_pose[2]], dtype=np.float32)
        # data[self._table2] = np.array([], dtype=np.float32)
        data[self._plate3] = np.array([self._plate3_pose[2]], dtype=np.float32)
        data[self._machine] = np.array([0.0], dtype=np.float32)
        return State(data)

    def _sample_goal_from_piles(self, num_blocks: int,
                                piles: List[List[Object]],
                                rng: np.random.Generator) -> Set[GroundAtom]:
        # Sample a goal that involves holding a block that's on the top of
        # the pile. This is useful for isolating the learning of picking and
        # unstacking. (For just picking, use num_blocks 1).
        if CFG.balance_holding_goals:
            pile_idx = rng.choice(len(piles))
            top_block = piles[pile_idx][-1]
            return {GroundAtom(self._Holding, [top_block])}
        # Sample goal pile that is different from initial
        while True:
            goal_piles = self._sample_initial_piles(num_blocks, rng)
            if goal_piles != piles:
                break
        # Create goal from piles
        goal_atoms = set()
        for pile in goal_piles:
            goal_atoms.add(GroundAtom(self._DirectlyOnPlate, [pile[0]]))
            if len(pile) == 1:
                continue
            for block1, block2 in zip(pile[1:], pile[:-1]):
                goal_atoms.add(GroundAtom(self._DirectlyOn, [block1, block2]))
        return goal_atoms

    def _sample_initial_pile_xy(
            self, rng: np.random.Generator,
            existing_xys: Set[Tuple[float, float]]) -> Tuple[float, float]:
        while True:
            x = rng.uniform(self.x_lb, self.x_ub)
            if rng.uniform(0, 1) < 0.5:
                # Table 1
                y = rng.uniform(self.y_lb, self.y_plate1_ub)
            else:
                # Table 3
                y = rng.uniform(self.y_plate3_lb, self.y_ub)

            if self._table_xy_is_clear(x, y, existing_xys):
                return (x, y)

    def _table_xy_is_clear(self, x: float, y: float,
                           existing_xys: Set[Tuple[float, float]]) -> bool:
        if all(
                abs(x - other_x) > self.collision_padding * self._block_size
                for other_x, _ in existing_xys):
            return True
        if all(
                abs(y - other_y) > self.collision_padding * self._block_size
                for _, other_y in existing_xys):
            return True
        return False


if __name__ == "__main__":
    """Run a simple simulation to test the environment."""
    import time

    # Make a task
    CFG.seed = 1
    CFG.num_train_tasks = 0
    CFG.num_test_tasks = 1
    env = PyBulletBalanceEnv(use_gui=True)
    task = env._generate_test_tasks()[0]
    env._reset_state(task.init)

    while True:
        # Robot does nothing
        action = Action(np.array(env._pybullet_robot.get_joints()))

        env.step(action)
        time.sleep(0.01)
