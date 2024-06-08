"""A PyBullet version of Blocks."""

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple, Sequence, Set, Union

import numpy as np
import pybullet as p
from PIL import Image

from predicators import utils
from predicators.envs.blocks import BlocksEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Array, EnvironmentTask, Object, State, Type,\
    Predicate
from predicators.utils import NSPredicate, RawState, BoundingBox,\
    evaluate_simple_assertion, VLMQuery, _MemoizedClassifier
from predicators.image_patch_wrapper import ImagePatch

class PyBulletBlocksEnv(PyBulletEnv, BlocksEnv):
    """PyBullet Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    # Repeat for LLM predicates parsing
    # Types
    _block_type = Type("block", ["pose_x", "pose_y", "pose_z", "held", 
                                "color_r", "color_g", "color_b"])
    _robot_type = Type("robot", ["pose_x", "pose_y", "pose_z", "fingers"])
    _table_type = Type("table", [])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Repeat for LLM predicates parsing
        # Predicates
        self._On = Predicate("On", [self._block_type, self._block_type],
                             self._On_holds)
        self._OnTable = Predicate("OnTable", [self._block_type],
                                  self._OnTable_holds)
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding", [self._block_type],
                                  self._Holding_holds)
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)

        # Neuro-Symbolic Predicates
        self._On_NSP = NSPredicate("On", [self._block_type, self._block_type],
                            _MemoizedClassifier(self._On_NSP_holds))
        self._OnTable_NSP = NSPredicate("OnTable", [self._block_type],
                            _MemoizedClassifier(self._OnTable_NSP_holds))
        self._GripperOpen_NSP = NSPredicate("GripperOpen", [self._robot_type],
                            _MemoizedClassifier(self._GripperOpen_NSP_holds))
        self._Holding_NSP = NSPredicate("Holding", [self._block_type],
                            _MemoizedClassifier(self._Holding_NSP_holds))
        self._Clear_NSP = NSPredicate("Clear", [self._block_type],
                            _MemoizedClassifier(self._Clear_NSP_holds))

        # We track the correspondence between PyBullet object IDs and Object
        # instances for blocks. This correspondence changes with the task.
        self._block_id_to_block: Dict[int, Object] = {}
        # Mapping from pybullet object id to Object instances
        # which can be used to get the segmented image for the object
        self._obj_id_to_obj: Dict[int, Object] = {}
        self.vlm = utils.create_vlm_by_name(CFG.vlm_model_name)


    @property
    def ns_predicates(self) -> Set[NSPredicate]:
        return {
            self._On_NSP, 
            self._OnTable_NSP, 
            self._GripperOpen_NSP,
            self._Holding_NSP, 
            self._Clear_NSP
            }
    
    @property
    def ns_predicates_to_predicates(self) -> Dict[NSPredicate, Predicate]:
        return {self._On_NSP: self._On, self._OnTable_NSP: self._OnTable,
                self._GripperOpen_NSP: self._GripperOpen,
                self._Holding_NSP: self._Holding, self._Clear_NSP: self._Clear}

    # @staticmethod
    def _Clear_NSP_holds(self, state: RawState, objects: Sequence[Object]) -> \
            Union[bool, VLMQuery]:
        '''
        Is there no block on top of the block
        '''
        block, = objects

        # Label the object in the scene image.
        block_name = block.id_name
        state_ip = ImagePatch(state)

        # We only need to look at the object and the space on top of it to
        # determine if it's clear.
        attention_image = state_ip.crop_to_objects([block], top_margin=20, 
                                                   lower_margin=5)
        if CFG.save_nsp_image_patch_before_query:
            attention_image.save(f"{CFG.image_dir}/clear({block_name}).png")
            # return False
        return evaluate_simple_assertion(
            f"there is no block directly on top of {block_name}.",
            attention_image)

    def _Holding_NSP_holds(self, state: RawState, objects: Sequence[Object]) ->\
            bool:
        '''
        Is the robot holding the block
        '''
        block, = objects

        # The block can't be held if the robot's hand is open.
        # We know there is only one robot in this environment.
        robot = state.get_objects(self._robot_type)[0]
        if self._GripperOpen_NSP_holds(state, [robot]):
            return False

        block_name = block.id_name
        state_ip = ImagePatch(state)
        attention_image = state_ip.crop_to_objects([block, robot])

        if CFG.save_nsp_image_patch_before_query:
            attention_image.save(f"{CFG.image_dir}/holding({block_name}).png")

        return evaluate_simple_assertion(f"{block_name} is held by the robot",
            attention_image)

    def _GripperOpen_NSP_holds(self, state: RawState, objects: Sequence[Object]) ->\
            bool:
        '''
        Is the robots gripper open
        '''
        robot, = objects
        finger_state = state.get(robot, "fingers")
        assert finger_state in (0.0, 1.0)
        return finger_state == 1.0
    
    def _OnTable_NSP_holds(self, state: RawState, objects:Sequence[Object]) ->\
            bool:
        '''Determine if the block in objects is on the table in the scene image.
        Attributes:
        -----------
        state : RawState
        objects : Sequence[Object]
            The block whose relationship with the table is to be determined.
        '''
        block, = objects
        block_name = block.id_name
        state_ip = ImagePatch(state)

        # Crop the scene image to the smallest bounding box that include both
        # objects.
        # We know there is only one table in this environment.
        table = state.get_objects(self._table_type)[0]
        table_name = table.id_name
        attention_image = state_ip.crop_to_objects([block, table])

        if CFG.save_nsp_image_patch_before_query:
            attention_image.save(f"{CFG.image_dir}/OnTable({block_name}).png")

        return evaluate_simple_assertion(
            f"{block_name} is directly resting on {table_name}'s surface.",
            attention_image)

    def _On_NSP_holds(self, state: RawState, objects: Sequence[Object]) -> bool:
        '''Determine if the first block in objects is on the second block in the 
        scene image.
        Attributes:
        -----------
        state : RawState
        objects : Sequence[Object]
            The two blocks whose relationship is to be determined.
        '''
        state_ip = ImagePatch(state)
        block1, block2 = objects
        block1_name, block2_name = block1.id_name, block2.id_name

        # Heuristics: we know a block can't be on top of itself.
        if block1_name == block2_name:
            return False

        # Using simple heuristics to check if they are far away
        block1_bbox = state.get_obj_bbox(block1) 
        block2_bbox = state.get_obj_bbox(block2)
        if (block1_bbox.lower < block2_bbox.lower) or \
           (block1_bbox.left > block2_bbox.right) or \
           (block1_bbox.right < block2_bbox.left) or \
           (block1_bbox.upper < block2_bbox.upper):
            return False

        # Crop the scene image to the smallest bounding box that include both
        # objects.
        attention_image = state_ip.crop_to_objects([block1, block2])

        if CFG.save_nsp_image_patch_before_query:
            attention_image.save(f"{CFG.image_dir}/On({block1_name}, "+
                                 f"{block2_name}).png")

        return evaluate_simple_assertion(
            f"{block1_name} is directly on top of {block2_name} with no blocks"+
             " in between.", attention_image)

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle blocks-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)
        bodies["table_id"] = table_id

        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            # Rendering now works in non-GUI version
            # assert using_gui, \
            #     "using_gui must be True to use pybullet_draw_debug."
            # Draw the workspace on the table for clarity.
            p.addUserDebugLine([cls.x_lb, cls.y_lb, cls.table_height],
                               [cls.x_ub, cls.y_lb, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_ub, cls.table_height],
                               [cls.x_ub, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_lb, cls.table_height],
                               [cls.x_lb, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_ub, cls.y_lb, cls.table_height],
                               [cls.x_ub, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            # Draw coordinate frame labels for reference.
            p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            p.addUserDebugText("z", [0, 0, 0.25], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            # Draw the pick z location at the x/y midpoint.
            mid_x = (cls.x_ub + cls.x_lb) / 2
            mid_y = (cls.y_ub + cls.y_lb) / 2
            p.addUserDebugText("*", [mid_x, mid_y, cls.pick_z],
                               [1.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)

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
                create_pybullet_block(color, half_extents, cls._obj_mass,
                                      cls._obj_friction, cls._default_orn,
                                      physics_client_id))
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._block_ids = pybullet_bodies["block_ids"]

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((cls.robot_init_x, cls.robot_init_y, cls.robot_init_z),
                       robot_ee_orn)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home)

    def _extract_robot_state(self, state: State) -> Array:
        # The orientation is fixed in this environment.
        qx, qy, qz, qw = self.get_robot_ee_home_orn()
        f = self.fingers_state_to_joint(self._pybullet_robot,
                                        state.get(self._robot, "fingers"))
        return np.array([
            state.get(self._robot, "pose_x"),
            state.get(self._robot, "pose_y"),
            state.get(self._robot, "pose_z"), qx, qy, qz, qw, f
        ],
                        dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_blocks"

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle blocks-specific resetting."""
        super()._reset_state(state)

        # Reset blocks based on the state.
        block_objs = state.get_objects(self._block_type)
        self._block_id_to_block = {}
        self._obj_id_to_obj = {}
        self._obj_id_to_obj[self._pybullet_robot.robot_id] = self._robot
        self._obj_id_to_obj[self._table_id] = self._table
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            self._block_id_to_block[block_id] = block_obj
            self._obj_id_to_obj[block_id] = block_obj
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

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.debug("Desired state:")
            logging.debug(state.pretty_str())
            logging.debug("Reconstructed state:")
            logging.debug(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

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
        fingers = self._fingers_joint_to_state(rf)
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
        
        # Get table state.
        state_dict[self._table] = np.array([], dtype=np.float32)

        state = utils.PyBulletState(state_dict,
                                    simulator_state=joint_positions)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num_tasks, possible_num_blocks, rng)
        return self._add_pybullet_state_to_tasks(tasks)

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        if CFG.pybullet_robot == "panda":
            # gripper rotated 90deg so parallel to x-axis
            normal = np.array([1., 0., 0.], dtype=np.float32)
        elif CFG.pybullet_robot == "fetch":
            # gripper parallel to y-axis
            normal = np.array([0., 1., 0.], dtype=np.float32)
        else:  # pragma: no cover
            # Shouldn't happen unless we introduce a new robot.
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        return {
            self._pybullet_robot.left_finger_id: normal,
            self._pybullet_robot.right_finger_id: -1 * normal,
        }

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

    @classmethod
    def fingers_state_to_joint(cls, pybullet_robot: SingleArmPyBulletRobot,
                               fingers_state: float) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either 0 or 1. Transform them to be
        either pybullet_robot.closed_fingers or
        pybullet_robot.open_fingers.
        """
        assert fingers_state in (0.0, 1.0)
        open_f = pybullet_robot.open_fingers
        closed_f = pybullet_robot.closed_fingers
        return closed_f if fingers_state == 0.0 else open_f

    def _fingers_joint_to_state(self, fingers_joint: float) -> float:
        """Convert the finger joint values in PyBullet to values for the State.

        The joint values given as input are the ones coming out of
        self._pybullet_robot.get_state().
        """
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
        # Fingers in the State should be either 0 or 1.
        return int(fingers_joint > (open_f + closed_f) / 2)
