"""Behavior2D domain.

This environment is a simple 2D projection of 100 BEHAVIOR tasks. 
The initial states are read from a set of precomputed 3D BEHAVIOR 
scenes. There is no simulation of motion, but actions are simply
teleport-style.

In 2D, there is no distinction between inside/ontop/under, so we
use inside for all of them. Note that PlaceInside might sometimes
require IsOpen, so we make sure to carry that into the 2D version
"""

import glob
import json
import logging
import os
from typing import ClassVar, Dict, List, Optional, Sequence, Set

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from bddl.object_taxonomy import ObjectTaxonomy

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    Object, ParameterizedOption, Predicate, State, Type



class Behavior2DEnv(BaseEnv):
    """Behavior2D domains."""

    robot_radius: ClassVar[float] = 0.3     # TODO: probably get this from scale in init_state
    gripper_length: ClassVar[float] = 2     # TODO: is this ridiculously long?
    _robot_color: ClassVar[str] = "gray"

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._rng = np.random.default_rng(self._seed)

        self.behavior_task = CFG.behavior_task_name
        # Sort to ensure determinism
        self.task_init_state_filenames = sorted(self._get_init_state_filenames())
        assert CFG.num_train_tasks + CFG.num_test_tasks <= len(self.task_init_state_filenames)
        shuffled_task_filenames = self._rng.permutation(self.task_init_state_filenames)
        self._train_task_filenames = shuffled_task_filenames[:CFG.num_train_tasks]
        self._test_task_filenames = shuffled_task_filenames[CFG.num_train_tasks:CFG.num_train_tasks+CFG.num_test_tasks]
        # print(self._test_task_filenames)
        # exit()
        # Types
        self.toggleable_types = set()
        self.insideable_types = {'basket', 'bottom_cabinet_no_top', 'cheese', 'stove', 'teapot', 'dishwasher', 
            'top_cabinet', 'burner', 'car', 'coffee_table', 'desk', 'sheet', 'trash_can', 'countertop', 'backpack', 
            'sink', 'carton', 'shelf', 'bucket', 'casserole', 'room_floor_living_room', 'bed', 'pedestal_table', 
            'sofa', 'fridge', 'plate', 'jar', 'bottom_cabinet', 'stocking', 'paper_bag', 'breakfast_table'}
        # self._need_pair_of_not_inside = {('spaghetti_sauce', 'fridge'), ('puree', 'fridge'), ('bagel', 'fridge'), 
        #     ('olive', 'fridge'), ('sushi', 'fridge'), ('hamburger', 'carton'), ('muffin', 'fridge'), 
        #     ('sugar_jar', 'fridge'), ('cheese', 'fridge'), ('pop', 'fridge'), ('salad', 'carton'), ('egg', 'fridge'), 
        #     ('vinegar', 'fridge'), ('lasagna', 'fridge'), ('jam', 'fridge'), ('parsley', 'fridge'), ('potato', 'fridge'), 
        #     ('rag', 'bottom_cabinet_no_top'), ('ginger', 'fridge'), ('basil', 'fridge'), ('candy_cane', 'fridge'), 
        #     ('catsup', 'fridge'), ('salt', 'fridge'), ('notebook', 'bottom_cabinet'), ('olive_oil', 'bottom_cabinet'), 
        #     ('umbrella', 'bottom_cabinet'), ('martini', 'fridge'), ('hardback', 'bottom_cabinet'), ('beer', 'fridge'), 
        #     ('plate', 'top_cabinet'), ('guacamole', 'fridge'), ('milk', 'fridge'), ('hamburger', 'fridge'), 
        #     ('juice', 'fridge'), ('soup', 'fridge'), ('plate', 'bottom_cabinet_no_top'), ('cinnamon', 'fridge'), 
        #     ('coffee_cup', 'fridge'), ('canned_food', 'fridge'), ('rag', 'bottom_cabinet'), ('salad', 'fridge'), 
        #     ('flour', 'fridge'), ('lollipop', 'fridge'), ('baguette', 'fridge'), ('plate', 'bottom_cabinet'), 
        #     ('pretzel', 'fridge'), ('tea_bag', 'fridge'), ('clove', 'fridge'), ('yogurt', 'fridge'), 
        #     ('mayonnaise', 'fridge'), ('chip', 'fridge'), ('olive_oil', 'top_cabinet'), ('butter', 'fridge'), 
        #     ('water', 'fridge'), ('sweet_corn', 'fridge'), ('oatmeal', 'fridge'), ('olive_oil', 'bottom_cabinet_no_top'), 
        #     ('cream', 'fridge')}
        self._need_pair_of_not_inside = set()
        self.openable_types = set()
        self.cookable_types = set()
        self.burnable_types = set()
        self.freezable_types = set()
        self.soakable_types = set()
        self.dustyable_types = set()
        self.stainable_types = set()
        self.sliceable_types = set()
        self.cooker_types = set()
        self.freezer_types = set()
        self.soaker_types = set()
        self.cleaner_types = set()
        self.slicer_types = set()

        self.all_features = ["pose_x", "pose_y", "yaw", "width", "height", "held",
                             "open", "toggled_on", "dusty", "stained", "sliced",
                             "cooked", "burnt", "frozen", "soaked"]
        self._parent_object_type = Type("object", self.all_features)
        self._object_types = self._get_all_object_types()
        self._object_types_set = set(self._object_types.values())
        self._robot_type = Type("robot", ["pose_x", "pose_y", "yaw", "gripper_free"])

        # Predicates
        self._predicates = self._create_behavior_predicates()
        self._predicates_set = set(self._predicates.values())

        # Static objects (always exist no matter the settings)
        self._robot = Object("robby", self._robot_type)


    def _get_all_object_types(self):
        type_names = set()
        for file in self.task_init_state_filenames:
            with open(file, "r") as f:
                init_state = json.load(f)
            for obj in init_state:
                if obj not in {"walls", "floors", "ceilings", "BRBody_1"}:
                    name = self._get_object_typename(obj)
                    type_names.add(name)
                    if init_state[obj]["toggleable"]:
                        self.toggleable_types.add(name)
                    if init_state[obj]["openable"]:
                        self.openable_types.add(name)
                    if init_state[obj]["cookable"]:
                        self.cookable_types.add(name)
                    if init_state[obj]["burnable"]:
                        self.burnable_types.add(name)
                    if init_state[obj]["freezable"]:
                        self.freezable_types.add(name)
                    if init_state[obj]["soakable"]:
                        self.soakable_types.add(name)
                    if init_state[obj]["dustyable"]:
                        self.dustyable_types.add(name)
                    if init_state[obj]["stainable"]:
                        self.stainable_types.add(name)
                    if init_state[obj]["sliceable"]:
                        self.sliceable_types.add(name)
                    if init_state[obj]["cooker"]:
                        self.cooker_types.add(name)
                    if init_state[obj]["freezer"]:
                        self.freezer_types.add(name)
                    if init_state[obj]["soaker"]:
                        self.soaker_types.add(name)
                    if init_state[obj]["cleaner"]:
                        self.cleaner_types.add(name)
                    if init_state[obj]["slicer"]:
                        self.slicer_types.add(name)

        types = {}
        for name in type_names:
            if name not in types:
                types[name] = Type(name, self.all_features, parent=self._parent_object_type)
        return types

    def _create_behavior_predicates(self):
        predicate_specs = [
            # Geometry
            ("inside", self._inside_classifier, 2),
            ("nextto", self._nextto_classifier, 2),
            ("touching", self._touching_classifier, 2),
            ("onfloor", self._onfloor_classifier, 1),
            # Symbolic
            ("open", self._open_classifier, 1), 
            ("toggled_on", self._toggled_on_classifier, 1),
            ("cooked", self._cooked_classifier, 1),
            ("burnt", self._burnt_classifier, 1),
            ("frozen", self._frozen_classifier, 1),
            ("soaked", self._soaked_classifier, 1),
            ("dusty", self._dusty_classifier, 1),
            ("stained", self._stained_classifier, 1),
            ("sliced", self._sliced_classifier, 1),
            # Affordances, receptor
            ("insideable", self._insideable_classifier, 1),
            ("openable", self._openable_classifier, 1),
            ("toggleable", self._toggleable_classifier, 1),
            ("not-toggleable", self._not_toggleable_classifier, 1),
            ("cookable", self._cookable_classifier, 1),
            ("burnable", self._burnable_classifier, 1),
            ("freezable", self._freezable_classifier, 1),
            ("soakable", self._soakable_classifier, 1),
            ("dustyable", self._dustyable_classifier, 1),
            ("stainable", self._stainable_classifier, 1),
            ("sliceable", self._sliceable_classifier, 1),
            # Affordances, effector
            ("cooker", self._cooker_classifier, 1),
            ("freezer", self._freezer_classifier, 1),
            ("soaker", self._soaker_classifier, 1),
            ("cleaner", self._cleaner_classifier, 1),
            ("slicer", self._slicer_classifier, 1),
            # Robot-oriented,
            ("reachable", self._reachable_classifier, 1),
            ("handempty", self._handempty_classifier, 0),
            ("holding", self._holding_classifier, 1),            
            # Negations
            ("not-inside", self._not_inside_classifier, 2),
            ("closed", self._closed_classifier, 1),
            ("toggled_off", self._toggled_off_classifier, 1),
            ("not-dusty", self._not_dusty_classifier, 1),
            ("not-stained", self._not_stained_classifier, 1),
            ("not-openable", self._not_openable_classifier, 1),
            ("reachable-nothing", self._reachable_nothing_classifier, 0),
            ("inside-nothing", self._inside_nothing_classifier, 1),
            ("not-holding", self._not_holding_classifier, 1),
        ]

        predicates = {}
        for name, classifier, arity in predicate_specs:
            predicates[name] = Predicate(name, [self._parent_object_type] * arity, classifier)
        return predicates

    @classmethod
    def get_name(cls) -> str:
        return "behavior2d"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        arr = action.arr
        if arr[-1] < 0.5:
            transition_fn = self._transition_navigate_to
        elif arr[-1] < 1.5:
            transition_fn = self._transition_grasp
        elif arr[-1] < 2.5:
            transition_fn = self._transition_place_inside
        elif arr[-1] < 3.5:
            transition_fn = self._transition_place_nextto
        elif arr[-1] < 4.5:
            transition_fn = self._transition_open
        elif arr[-1] < 5.5:
            transition_fn = self._transition_close
        elif arr[-1] < 6.5:
            transition_fn = self._transition_toggleon
        elif arr[-1] <= 7.5:
            transition_fn = self._transition_cook
        elif arr[-1] <= 8.5:
            transition_fn = self._transition_freeze
        elif arr[-1] <= 9.5:
            transition_fn = self._transition_soak
        elif arr[-1] <= 10.5:
            transition_fn = self._transition_clean_dusty
        elif arr[-1] <= 11.5:
            transition_fn = self._transition_clean_stained
        elif arr[-1] <= 12.0:
            transition_fn = self._transition_slice
        return transition_fn(state, action)


    def _transition_navigate_to(self, state: State, action: Action) -> State:
        pos_x, pos_y, yaw = action.arr[:3]
        next_state = state.copy()
        
        robby = self._robot
        robby_geom = utils.Circle(pos_x, pos_y, self.robot_radius)
        ignore_objects = {robby}
        held_obj = self._get_held_object(state)
        if held_obj is not None:
            ignore_objects.add(held_obj)
        if self.detect_collision(state, robby_geom, ignore_objects):
            return next_state

        next_state.set(robby, "pose_x", pos_x)
        next_state.set(robby, "pose_y", pos_y)
        next_state.set(robby, "yaw", yaw)
        return next_state

    def _transition_grasp(self, state: State, action: Action) -> State:
        yaw, offset_gripper, obj_id = action.arr[2:5]
        next_state = state.copy()
        held_obj = self._get_held_object(state)
        if held_obj is not None:
            # print("failed bc no object held")
            return next_state

        robby = self._robot
        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        gripper_free =  state.get(robby, "gripper_free")
        if gripper_free != 1.0:
            # print("failed bc gripper is not free")
            return next_state

        tip_x = robby_x + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.sin(robby_yaw)

        pick_obj = self.get_object_by_id(state, int(obj_id))
        obj_x = state.get(pick_obj, "pose_x")
        obj_y = state.get(pick_obj, "pose_y")
        obj_w = state.get(pick_obj, "width")
        obj_h = state.get(pick_obj, "height")
        obj_yaw = state.get(pick_obj, "yaw")
        while obj_yaw > np.pi:
            obj_yaw -= (2 * np.pi)
        while obj_yaw < -np.pi:
            obj_yaw += (2 * np.pi)
        obj_rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
        # If gripper didn't reach object, fail
        if not obj_rect.contains_point(tip_x, tip_y):
            # print("failed bc gripper doesnt contain tip")
            return next_state
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        ignore_objects = {pick_obj, robby}

        # Execute pick
        obj_rect = obj_rect.rotate_about_point(tip_x, tip_y, yaw)
        obj_rect = obj_rect.rotate_about_point(tip_x, tip_y, -robby_yaw)
        rot_rel_x = obj_rect.x - tip_x
        rot_rel_y = obj_rect.y - tip_y
        rot_rel_yaw = obj_rect.theta
        next_state.set(pick_obj, "held", 1.0)
        next_state.set(pick_obj, "pose_x", rot_rel_x)
        next_state.set(pick_obj, "pose_y", rot_rel_y)
        next_state.set(pick_obj, "yaw", rot_rel_yaw)
        next_state.set(robby, "gripper_free", 0.0)

        inside_picked_list = self._get_objects_inside(state, pick_obj)
        # print(f"attempting to bring along: {inside_picked_list}")
        for obj in inside_picked_list:
            obj_x = state.get(obj, "pose_x")
            obj_y = state.get(obj, "pose_y")
            obj_w = state.get(obj, "width")
            obj_h = state.get(obj, "height")
            obj_yaw = state.get(obj, "yaw")
            while obj_yaw > np.pi:
                obj_yaw -= (2 * np.pi)
            while obj_yaw < -np.pi:
                obj_yaw += (2 * np.pi)
            obj_rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
            obj_rect = obj_rect.rotate_about_point(tip_x, tip_y, yaw)
            obj_rect = obj_rect.rotate_about_point(tip_x, tip_y, -robby_yaw)
            rot_rel_x = obj_rect.x - tip_x
            rot_rel_y = obj_rect.y - tip_y
            rot_rel_yaw = obj_rect.theta
            next_state.set(obj, "held", -1.0)
            next_state.set(obj, "pose_x", rot_rel_x)
            next_state.set(obj, "pose_y", rot_rel_y)
            next_state.set(obj, "yaw", rot_rel_yaw)
            assert self._inside_classifier(next_state, [obj, pick_obj])
        return next_state

    def _transition_place_inside(self, state: State, action: Action) -> State:
        offset_gripper, inside_obj_id = action.arr[3:5]
        next_state = state.copy()

        obj = self._get_held_object(state)
        if obj is None:
            # print("no object held")
            return next_state
        rot_rel_x = state.get(obj, "pose_x")
        rot_rel_y = state.get(obj, "pose_y")
        rot_rel_yaw = state.get(obj, "yaw")
        obj_w = state.get(obj, "width")
        obj_h = state.get(obj, "height")

        robby = self._robot
        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            # print("gripper is free")
            return next_state

        inside_obj = self.get_object_by_id(state, int(inside_obj_id))
        if inside_obj.type.name not in self.insideable_types:
            return next_state
        if inside_obj.type.name in self.openable_types and not self._open_classifier(state, [inside_obj]):
            # print("inside object is openable and is not open")
            return next_state

        inside_obj_x = state.get(inside_obj, "pose_x")
        inside_obj_y = state.get(inside_obj, "pose_y")
        inside_obj_w = state.get(inside_obj, "width")
        inside_obj_h = state.get(inside_obj, "height")
        inside_obj_yaw = state.get(inside_obj, "yaw")

        # print(f"Big sides: ({inside_obj_w}, {inside_obj_h}); small sides: ({obj_w}, {obj_h})")
        tip_x = robby_x + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.sin(robby_yaw)

        obj_rect = utils.Rectangle(rot_rel_x + tip_x, rot_rel_y + tip_y, obj_w, obj_h, rot_rel_yaw)
        obj_rect = obj_rect.rotate_about_point(tip_x, tip_y, robby_yaw)
        place_x = obj_rect.x
        place_y = obj_rect.y
        place_yaw = obj_rect.theta

        # Check whether the objs center-of-mass is within the inside_obj bounds
        # and that there are no other collisions with the obj or gripper
        inside_obj_rect = utils.Rectangle(inside_obj_x, inside_obj_y, inside_obj_w, inside_obj_h,
                                     inside_obj_yaw)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        ignore_objects = {obj, robby, inside_obj}
        for other_obj in state:
            if other_obj == obj or other_obj == inside_obj or other_obj == self._robot:
                continue
            held = state.get(other_obj, "held")
            if held == 1.0 or held == -1.0:
                continue
            x = state.get(other_obj, "pose_x")
            y = state.get(other_obj, "pose_y")
            yaw = state.get(other_obj, "yaw")
            width = state.get(other_obj, "width")
            height = state.get(other_obj, "height")
            while yaw > np.pi:
                yaw -= (2 * np.pi)
            while yaw < -np.pi:
                yaw += (2 * np.pi)
            other_obj_rect = utils.Rectangle(x, y, width, height, yaw)
            if other_obj_rect.contains_point(*(inside_obj_rect.center)):
                ignore_objects.add(other_obj)

        # TODO: figure out what to do about collision checking
        if not inside_obj_rect.contains_point(*(obj_rect.center)):# or \
            # self.detect_collision(state, obj_rect, ignore_objects): #or \
            # self.detect_collision(state, gripper_line, ignore_objects):
            # print("object not inside")
            return next_state

        next_state.set(obj, "held", 0.0)
        next_state.set(obj, "pose_x", place_x)
        next_state.set(obj, "pose_y", place_y)
        next_state.set(obj, "yaw", place_yaw)
        next_state.set(robby, "gripper_free", 1.0)

        # Loop over objects inside held_obj and place them too
        for other_obj in state:
            if other_obj == self._robot or other_obj == obj:
                continue
            obj_inside_held = state.get(other_obj, "held")
            if obj_inside_held != -1.0:
                continue
            rot_rel_x = state.get(other_obj, "pose_x")
            rot_rel_y = state.get(other_obj, "pose_y")
            rot_rel_yaw = state.get(other_obj, "yaw")
            obj_w = state.get(other_obj, "width")
            obj_h = state.get(other_obj, "height")
            obj_rect = utils.Rectangle(rot_rel_x + tip_x, rot_rel_y + tip_y, obj_w, obj_h, rot_rel_yaw)
            obj_rect = obj_rect.rotate_about_point(tip_x, tip_y, robby_yaw)
            place_x = obj_rect.x
            place_y = obj_rect.y
            place_yaw = obj_rect.theta

            next_state.set(other_obj, "held", 0.0)
            next_state.set(other_obj, "pose_x", place_x)
            next_state.set(other_obj, "pose_y", place_y)
            next_state.set(other_obj, "yaw", place_yaw)
            assert self._inside_classifier(next_state, [other_obj, obj])
        # print("success")
        return next_state

    def _transition_place_nextto(self, state: State, action: Action) -> State:
        offset_gripper = action.arr[3]
        next_state = state.copy()

        obj = self._get_held_object(state)
        if obj is None:
            # print("failed due to no held object")
            return next_state
        rot_rel_x = state.get(obj, "pose_x")
        rot_rel_y = state.get(obj, "pose_y")
        rot_rel_yaw = state.get(obj, "yaw")
        obj_w = state.get(obj, "width")
        obj_h = state.get(obj, "height")

        robby = self._robot
        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            # print("failed due to gripper free")
            return next_state

        tip_x = robby_x + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.sin(robby_yaw)

        obj_rect = utils.Rectangle(rot_rel_x + tip_x, rot_rel_y + tip_y, obj_w, obj_h, rot_rel_yaw)
        obj_rect = obj_rect.rotate_about_point(tip_x, tip_y, robby_yaw)
        place_x = obj_rect.x
        place_y = obj_rect.y
        place_yaw = obj_rect.theta

        # Check whether the objs center-of-mass is within the inside_obj bounds
        # and that there are no other collisions with the obj or gripper
        obj_rect = utils.Rectangle(place_x, place_y, obj_w, obj_h,
                                    place_yaw)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        ignore_objects = {obj, robby}
        # TODO: figure out what to do about collision checking in the end
        # if self.detect_collision(state, obj_rect, ignore_objects): #or \
        #     # self.detect_collision(state, gripper_line, ignore_objects):
        #     print("failed due to collision")
        #     return next_state

        next_state.set(obj, "held", 0.0)
        next_state.set(obj, "pose_x", place_x)
        next_state.set(obj, "pose_y", place_y)
        next_state.set(obj, "yaw", place_yaw)
        next_state.set(robby, "gripper_free", 1.0)

        # Loop over objects inside held_obj and place them too
        for other_obj in state:
            if other_obj == self._robot or other_obj == obj:
                continue
            obj_inside_held = state.get(other_obj, "held")
            if obj_inside_held != -1.0:
                continue
            rot_rel_x = state.get(other_obj, "pose_x")
            rot_rel_y = state.get(other_obj, "pose_y")
            rot_rel_yaw = state.get(other_obj, "yaw")
            obj_w = state.get(other_obj, "width")
            obj_h = state.get(other_obj, "height")
            obj_rect = utils.Rectangle(rot_rel_x + tip_x, rot_rel_y + tip_y, obj_w, obj_h, rot_rel_yaw)
            obj_rect = obj_rect.rotate_about_point(tip_x, tip_y, robby_yaw)
            place_x = obj_rect.x
            place_y = obj_rect.y
            place_yaw = obj_rect.theta

            next_state.set(other_obj, "held", 0.0)
            next_state.set(other_obj, "pose_x", place_x)
            next_state.set(other_obj, "pose_y", place_y)
            next_state.set(other_obj, "yaw", place_yaw)
            assert self._inside_classifier(next_state, [other_obj, obj])

        assert self._reachable_classifier(next_state, [obj])
        return next_state

    # The actions below are implemented as "magic words". For this
    # to be somewhat reasonable, the simulator has to enforce the 
    # preconditions of reachability and such.

    def _transition_open(self, state: State, action: Action) -> State:
        target_obj_id = action.arr[4]
        next_state = state.copy()

        robby = self._robot
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 1.0:
            return next_state

        target_obj = self.get_object_by_id(state, int(target_obj_id))
        if not self._reachable_classifier(state, [target_obj]):
            return next_state

        if not target_obj.type.name in self.openable_types:
            return next_state

        next_state.set(target_obj, "open", 1.0)
        return next_state

    def _transition_close(self, state: State, action: Action) -> State:
        target_obj_id = action.arr[4]
        next_state = state.copy()

        robby = self._robot
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 1.0:
            return next_state

        target_obj = self.get_object_by_id(state, int(target_obj_id))
        if not self._reachable_classifier(state, [target_obj]):
            return next_state

        if not target_obj.type.name in self.openable_types:
            return next_state

        next_state.set(target_obj, "open", 0.0)
        return next_state

    def _transition_toggleon(self, state: State, action: Action) -> State:
        target_obj_id = action.arr[4]
        next_state = state.copy()

        robby = self._robot
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 1.0:
            return next_state

        target_obj = self.get_object_by_id(state, int(target_obj_id))
        if not self._reachable_classifier(state, [target_obj]):
            return next_state

        if not target_obj.type.name in self.toggleable_types:
            return next_state

        next_state.set(target_obj, "toggled_on", 1.0)
        return next_state

    def _transition_cook(self, state: State, action: Action) -> State:
        cooker_obj_id = action.arr[4]
        next_state = state.copy()

        robby = self._robot
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            return next_state

        cooker_obj = self.get_object_by_id(state, int(cooker_obj_id))
        if not cooker_obj.type.name in self.cooker_types or not self._reachable_classifier(state, [cooker_obj]):
            return next_state

        target_obj = self._get_held_object(state)
        if not target_obj.type.name in self.cookable_types:
            return next_state

        if cooker_obj.type.name in self.toggleable_types and not self._toggled_on_classifier(state, [cooker_obj]):
            return next_state

        next_state.set(target_obj, "cooked", 1.0)
        return next_state

    def _transition_freeze(self, state: State, action: Action) -> State:
        freezer_obj_id = action.arr[4]
        next_state = state.copy()

        robby = self._robot
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            return next_state

        freezer_obj = self.get_object_by_id(state, int(freezer_obj_id))
        if not freezer_obj.type.name in self.freezer_types or not self._reachable_classifier(state, [freezer_obj]):
            return next_state

        target_obj = self._get_held_object(state)
        if not target_obj.type.name in self.freezable_types:
            return next_state

        if freezer_obj.type.name in self.toggleable_types and not self._toggled_on_classifier(state, [freezer_obj]):
            return next_state

        next_state.set(target_obj, "frozen", 1.0)
        return next_state

    def _transition_soak(self, state: State, action: Action) -> State:
        soaker_obj_id = action.arr[4]
        next_state = state.copy()

        robby = self._robot
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            return next_state

        soaker_obj = self.get_object_by_id(state, int(soaker_obj_id))
        if not soaker_obj.type.name in self.soaker_types or not self._reachable_classifier(state, [soaker_obj]):
            return next_state

        target_obj = self._get_held_object(state)
        if not target_obj.type.name in self.soakable_types:
            return next_state

        if soaker_obj.type.name in self.toggleable_types and not self._toggled_on_classifier(state, [soaker_obj]):
            return next_state

        next_state.set(target_obj, "soaked", 1.0)
        return next_state

    def _transition_clean_dusty(self, state: State, action: Action) -> State:
        target_obj_id = action.arr[4]
        next_state = state.copy()

        robby = self._robot
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            return next_state

        target_obj = self.get_object_by_id(state, int(target_obj_id))
        if not target_obj.type.name in self.dustyable_types or not self._reachable_classifier(state, [target_obj]):
            return next_state

        cleaner_obj = self._get_held_object(state)
        if not cleaner_obj.type.name in self.cleaner_types:
            return next_state

        next_state.set(target_obj, "dusty", 0.0)
        return next_state

    def _transition_clean_stained(self, state: State, action: Action) -> State:
        target_obj_id = action.arr[4]
        next_state = state.copy()

        robby = self._robot
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            return next_state

        target_obj = self.get_object_by_id(state, int(target_obj_id))
        if not target_obj.type.name in self.stainable_types or not self._reachable_classifier(state, [target_obj]):
            return next_state

        cleaner_obj = self._get_held_object(state)
        if not cleaner_obj.type.name in self.cleaner_types or not self._soaked_classifier(state, [cleaner_obj]):
            return next_state

        next_state.set(target_obj, "stained", 0.0)
        return next_state

    def _transition_slice(self, state: State, action: Action) -> State:
        target_obj_id = action.arr[4]
        next_state = state.copy()

        robby = self._robot
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            return next_state

        target_obj = self.get_object_by_id(state, int(target_obj_id))
        if not target_obj.type.name in self.sliceable_types or not self._reachable_classifier(state, [target_obj]):
            return next_state

        slicer_obj = self._get_held_object(state)
        if not slicer_obj.type.name in self.slicer_types:
            return next_state

        next_state.set(target_obj, "sliced", 1.0)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(self._train_task_filenames)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(self._test_task_filenames)

    @property
    def predicates(self) -> Set[Predicate]:
        return self._predicates_set

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self.predicates

    @property
    def types(self) -> Set[Type]:
        return self._object_types_set | {self._robot_type} | {self._parent_object_type}

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, yaw, offset_gripper, target_obj_id, action_id]
        lowers = np.array([-np.inf, -np.inf, -np.pi, 0.0, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([np.inf, np.inf, np.pi, 1.0, np.inf, 12.0], dtype=np.float32)
        return Box(lowers, uppers)

    def render_state_plt(
        self,
        state: State,
        task: EnvironmentTask,
        action: Optional[Action] = None,
        caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        objs = [obj for obj in state if not obj.is_instance(self._robot_type)]
        if task is not None:
            goal_objs = {obj for atom in task.goal for obj in atom.objects}
        else:
            goal_objs = objs
        robby = self._robot

        # Draw robot
        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        gripper_free = state.get(robby, "gripper_free") == 1.0
        circ = utils.Circle(robby_x, robby_y, self.robot_radius)
        circ.plot(ax, facecolor=self._robot_color)
        offset_gripper = 0  # if gripper_free else 1
        tip_x = robby_x + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.sin(robby_yaw)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        gripper_line.plot(ax, color="white")

        # Draw objs
        for b in sorted(objs):
            x = state.get(b, "pose_x")
            y = state.get(b, "pose_y")
            w = state.get(b, "width")
            h = state.get(b, "height")
            yaw = state.get(b, "yaw")
            holding = state.get(b, "held") == 1.0
            fc = "blue" if b in goal_objs else "gray"
            ec = "red" if holding else "gray"
            if holding:
                aux_x = tip_x + x * np.sin(robby_yaw) + y * np.cos(robby_yaw)
                aux_y = tip_y + y * np.sin(robby_yaw) - x * np.cos(robby_yaw)
                x = aux_x
                y = aux_y
                yaw += robby_yaw
            while yaw > np.pi:
                yaw -= (2 * np.pi)
            while yaw < -np.pi:
                yaw += (2 * np.pi)
            rect = utils.Rectangle(x, y, w, h, yaw)
            rect.plot(ax, facecolor=fc, edgecolor=ec, alpha=0.3)
        plt.suptitle(caption, fontsize=12, wrap=True)
        plt.tight_layout()
        plt.axis("off")
        return fig

    def _get_tasks(self, task_init_state_filenames: Sequence[str]) -> List[EnvironmentTask]:
        # Create all objects from files
        # Create list of obj id"s
        # Parse goal from bddl somehow

        tasks = []
        for file in task_init_state_filenames:
            with open(file, "r") as f:
                init_state_3d = json.load(f)
            goal_file = file.replace("state.json", "goal.json")
            with open(goal_file, "r") as f:
                goal_3d = json.load(f)

            obj_name_to_obj = {}
            data: Dict[Object, Array] = {}
            # objects_file = os.path.join('/'.join(file.split('/')[:-1]), 'objects.json')
            objects_file = file.replace("state.json", "objects.json")
            for obj_name in init_state_3d:
                if obj_name not in {"walls", "floors", "ceilings", "BRBody_1"}:
                    type_name = self._get_object_typename(obj_name)
                    obj = Object(obj_name, self._object_types[type_name])
                    obj_name_to_obj[obj_name] = obj
                    data_obj = []

                    # 2D position
                    pos = init_state_3d[obj_name]["pos"]
                    data_obj.append(pos[0]) 
                    data_obj.append(pos[1]) 

                    # Yaw
                    orn = init_state_3d[obj_name]["orn"]
                    yaw = orn[2]
                    while yaw > np.pi:
                        yaw -= (2 * np.pi)
                    while yaw < -np.pi:
                        yaw += (2 * np.pi)
                    data_obj.append(yaw)

                    # Width and height
                    bbox = init_state_3d[obj_name]["bbox"]
                    data_obj.append(bbox[0])
                    data_obj.append(bbox[1])

                    # Held
                    data_obj.append(False)

                    # Object-specific properties
                    data_obj.append(init_state_3d[obj_name]["open"])
                    data_obj.append(init_state_3d[obj_name]["toggled-on"])
                    data_obj.append(init_state_3d[obj_name]["dusty"])
                    data_obj.append(init_state_3d[obj_name]["stained"])
                    data_obj.append(init_state_3d[obj_name]["sliced"])
                    data_obj.append(init_state_3d[obj_name]["cooked"])
                    data_obj.append(init_state_3d[obj_name]["burnt"])
                    data_obj.append(init_state_3d[obj_name]["frozen"])
                    data_obj.append(init_state_3d[obj_name]["soaked"])

                    data[obj] = np.array(data_obj)

            data_robot = []

            # Robot 2D position
            robot_pos = init_state_3d["BRBody_1"]["pos"]
            data_robot.append(robot_pos[0])
            data_robot.append(robot_pos[1])

            # Robot yaw
            robot_orn = init_state_3d["BRBody_1"]["orn"]
            yaw = robot_orn[2]
            while yaw > np.pi:
                yaw -= (2 * np.pi)
            while yaw < -np.pi:
                yaw += (2 * np.pi)
            data_robot.append(yaw)

            # Gripper free
            data_robot.append(1.0)

            data[self._robot] = np.array(data_robot)
            init_state = State(data)

            goal = self._get_goal_from_file(goal_3d, obj_name_to_obj)

            with open(objects_file, 'r') as f:
                objects = json.load(f)
            objects = [o for o in objects if not (o.startswith('floor') or o.startswith('agent'))]

            taxonomy = ObjectTaxonomy()

            goal_objects = {o for atom in goal for o in atom.objects}
            other_objects = set(init_state.data.keys()) - goal_objects
            other_objects = {o for o in other_objects if not (o.type.name.startswith('floor') or o.type.name.startswith('robot'))}
            task_relevant_objects = set()
            for o1 in objects:
                print(o1)
                o1_class = '_'.join(o1.split('_')[:-1])
                for o2 in goal_objects:
                    # if o1.split('.')[0] == o2.type.name:
                    print(o2.type.name, taxonomy.get_subtree_igibson_categories(o1_class))
                    if o2.type.name in taxonomy.get_subtree_igibson_categories(o1_class):
                        task_relevant_objects.add(o2)
                        goal_objects.remove(o2)
                        break
                else:
                    for o2 in other_objects:
                        # if o1.split('.')[0] == o2.type.name:
                        if o2.type.name in taxonomy.get_subtree_igibson_categories(o1_class):
                            task_relevant_objects.add(o2)
                            other_objects.remove(o2)
                            break
                    else:
                        for o2 in other_objects:
                            if o2.type.name in taxonomy.get_subtree_igibson_categories(o1_class):
                                task_relevant_objects.add(o2)
                                other_objects.remove(o2)
                                break
            assert len(task_relevant_objects) == len(objects), f'{len(task_relevant_objects), len(objects)}\n\t{task_relevant_objects}\n\t{objects}'

            if CFG.behavior_only_relevant_objects:
                init_state = State({k: v for k, v in init_state.data.items() if k in task_relevant_objects or k == self._robot})
            tasks.append(EnvironmentTask(init_state, goal))
        return tasks

    def _get_goal_from_file(self, goal_3d, obj_name_to_obj):
        goal = set()

        for head_expr in goal_3d:
            head_expr = head_expr[0]
            if head_expr[0] == "not":
                if head_expr[1] == "open":
                    bddl_name = "closed"
                elif head_expr[1] == "dusty":
                    bddl_name = "not-dusty"
                elif head_expr[1] == "stained":
                    bddl_name = "not-stained"
                elif head_expr[1] == "inside":
                    bddl_name = "not-inside"
                else:
                    raise ValueError
                obj_start_idx = 2
            else:
                bddl_name = head_expr[0]
                obj_start_idx = 1
            if bddl_name in {"ontop", "under"}:
                bddl_name = "inside"
            try:
                objects = [obj_name_to_obj[name] for name in head_expr[obj_start_idx:]]
                pred_name = bddl_name #+ "-" + "-".join(obj.type.name for obj in objects)
                pred = self._predicates[pred_name]
                goal.add(GroundAtom(pred, objects))
            except KeyError:
                # TODO: I'm skipping objects that aren't in the object_names, which should be
                # floors and walls only
                logging.warn(f"Skipping goal atom {bddl_name}: {head_expr[obj_start_idx:]}")
        return goal

    def _get_init_state_filenames(self) -> List[str]:
        pattern = os.path.join(CFG.behavior_init_states_path, 
                            self.behavior_task, "*state.json")
        fnames = []
        for name in glob.glob(pattern):
            if os.path.basename(name).startswith('operator_count_'):
                count = int(os.path.basename(name).lstrip('operator_count_').split('_')[0])
                if count > 1:
                    continue
            fnames.append(name)
        return fnames

    @staticmethod
    def _get_object_typename(obj_name):
        obj_name = obj_name.replace("_multiplexer", "")
        split_by_hyphen = obj_name.split("-")
        if len(split_by_hyphen) > 1:
            if split_by_hyphen[0] == "t" and split_by_hyphen[1] == "shirt":
                type_name = "tshirt"
            else:
                type_name = split_by_hyphen[0]
        else:
            type_name = "_".join(obj_name.split("_")[:-1])
        return type_name


    def detect_collision(self, state: State, geom: utils._Geom2D,
                        ignore_objects: Optional[Set[Object]] = None) -> bool:
        if ignore_objects is None:
            ignore_objects = set()
        for obj in state.data:
            if obj in ignore_objects:
                continue
            if obj.is_instance(self._robot_type):
                x = state.get(obj, "pose_x")
                y = state.get(obj, "pose_y")
                obj_geom = utils.Circle(x, y, self.robot_radius)
            else:
                # All non-robot objects are rectangles
                held = state.get(obj, "held")
                if held == 1.0 or held == -1.0:
                    continue
                x = state.get(obj, "pose_x")
                y = state.get(obj, "pose_y")
                yaw = state.get(obj, "yaw")
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                while yaw > np.pi:
                    yaw -= (2 * np.pi)
                while yaw < -np.pi:
                    yaw += (2 * np.pi)
                obj_geom = utils.Rectangle(x, y, width, height, yaw)
            if utils.geom2ds_intersect(geom, obj_geom):
                return True
        return False

    def _inside_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, inside_obj = objects
        if obj is inside_obj:
            return False

        obj_x = state.get(obj, "pose_x")
        obj_y = state.get(obj, "pose_y")
        obj_w = state.get(obj, "width")
        obj_h = state.get(obj, "height")
        obj_yaw = state.get(obj, "yaw")
        obj_held = state.get(obj, "held")

        inside_obj_x = state.get(inside_obj, "pose_x")
        inside_obj_y = state.get(inside_obj, "pose_y")
        inside_obj_w = state.get(inside_obj, "width")
        inside_obj_h = state.get(inside_obj, "height")
        inside_obj_yaw = state.get(inside_obj, "yaw")
        inside_obj_held = state.get(inside_obj, "held")

        # There are 9 cases for the logic of "inside" flags (1,0,-1)x(1,0,-1)
        # If the inside object is held, the object can only be inside it if we flagged it as -1
        if inside_obj_held == 1.0:
            return obj_held == -1.0

        # If the inside object is inside a held object, the object can only be inside it if we flagged it as -1
        # AND its relative coordinates are inside the inside object's relative coordinates
        if inside_obj_held == -1.0 and obj_held != -1.0:
            return False

        # If the inside object is not held, then the object can only be inside it if it is also not held
        if inside_obj_held == 0.0 and obj_held != 0.0:
            return False            

        # The remaining cases are both are not held or both are inside held. The geometry is equivalent for both cases

        while obj_yaw > np.pi:
            obj_yaw -= (2 * np.pi)
        while obj_yaw < -np.pi:
            obj_yaw += (2 * np.pi)
        obj_rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
        inside_obj_rect = utils.Rectangle(inside_obj_x, inside_obj_y, inside_obj_w, inside_obj_h,
                                     inside_obj_yaw)
        # return inside_obj_rect.contains_point(*(obj_rect.center))
        return inside_obj_rect.contains_point(*(obj_rect.vertices[0])) and \
                inside_obj_rect.contains_point(*(obj_rect.vertices[1])) and \
                inside_obj_rect.contains_point(*(obj_rect.vertices[2])) and \
                inside_obj_rect.contains_point(*(obj_rect.vertices[3]))

    def _nextto_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj_a, obj_b = objects
        if obj_a is obj_b:
            return False

        obj_a_x = state.get(obj_a, "pose_x")
        obj_a_y = state.get(obj_a, "pose_y")
        obj_a_w = state.get(obj_a, "width")
        obj_a_h = state.get(obj_a, "height")
        obj_a_yaw = state.get(obj_a, "yaw")
        obj_a_held = state.get(obj_a, "held")

        obj_b_x = state.get(obj_b, "pose_x")
        obj_b_y = state.get(obj_b, "pose_y")
        obj_b_w = state.get(obj_b, "width")
        obj_b_h = state.get(obj_b, "height")
        obj_b_yaw = state.get(obj_b, "yaw")
        obj_b_held = state.get(obj_b, "held")

        # There are 9 cases for the logic of "inside" flags (1,0,-1)x(1,0,-1)
        # If either object is held, it cannot have any nexttos
        if obj_a_held == 1.0 or obj_b_held == 1.0:
            return False

        # If either object is inside held, the other must also be inside held
        if (obj_a_held == -1.0 and obj_b_held != -1.0) or (obj_b_held == -1.0 and obj_a_held != -1.0):
            return False

        # The remaining cases are both are not held or both are inside held. The geometry is equivalent for both cases

        while obj_a_yaw > np.pi:
            obj_a_yaw -= (2 * np.pi)
        while obj_a_yaw < -np.pi:
            obj_a_yaw += (2 * np.pi)
        obj_a_rect = utils.Rectangle(obj_a_x, obj_a_y, obj_a_w, obj_a_h, obj_a_yaw)
        obj_b_rect = utils.Rectangle(obj_b_x, obj_b_y, obj_b_w, obj_b_h,
                                     obj_b_yaw)

        avg_aabb_length = np.mean([obj_a_w + obj_b_w, obj_a_h + obj_b_h])
        distance = obj_a_rect.distance_to(obj_b_rect)

        # TODO: iGibson uses distance only as one check, but then computes
        # an adjacency list and checks whether the two objects are in the
        # adjacency list of each other.

        return distance <= avg_aabb_length / 6 and \
            not self._touching_classifier(state, [obj_a, obj_b]) and \
            not self._inside_classifier(state, [obj_a, obj_b]) and \
            not self._inside_classifier(state, [obj_b, obj_a])
                # not obj_a_rect.contains_point(*(obj_b_rect.center)) and \
                # not obj_b_rect.contains_point(*(obj_a_rect.center))

    def _touching_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj_a, obj_b = objects
        if obj_a is obj_b:
            return False

        obj_a_x = state.get(obj_a, "pose_x")
        obj_a_y = state.get(obj_a, "pose_y")
        obj_a_w = state.get(obj_a, "width")
        obj_a_h = state.get(obj_a, "height")
        obj_a_yaw = state.get(obj_a, "yaw")
        obj_a_held = state.get(obj_a, "held")

        obj_b_x = state.get(obj_b, "pose_x")
        obj_b_y = state.get(obj_b, "pose_y")
        obj_b_w = state.get(obj_b, "width")
        obj_b_h = state.get(obj_b, "height")
        obj_b_yaw = state.get(obj_b, "yaw")
        obj_b_held = state.get(obj_b, "held")

        # There are 9 cases for the logic of "inside" flags (1,0,-1)x(1,0,-1)
        # If either object is held, it cannot have any touching's
        if obj_a_held == 1.0 or obj_b_held == 1.0:
            return False

        # If either object is inside held, the other must also be inside held
        if (obj_a_held == -1.0 and obj_b_held != -1.0) or (obj_b_held == -1.0 and obj_a_held != -1.0):
            return False

        # The remaining cases are both are not held or both are inside held. The geometry is equivalent for both cases

        while obj_a_yaw > np.pi:
            obj_a_yaw -= (2 * np.pi)
        while obj_a_yaw < -np.pi:
            obj_a_yaw += (2 * np.pi)
        obj_a_rect = utils.Rectangle(obj_a_x, obj_a_y, obj_a_w, obj_a_h, obj_a_yaw)
        obj_b_rect = utils.Rectangle(obj_b_x, obj_b_y, obj_b_w, obj_b_h,
                                     obj_b_yaw)

        return utils.rectangles_intersect(obj_a_rect, obj_b_rect) and \
                not self._inside_classifier(state, [obj_a, obj_b]) and \
                not self._inside_classifier(state, [obj_b, obj_a])
                # not obj_a_rect.contains_point(*(obj_b_rect.center)) and \
                # not obj_b_rect.contains_point(*(obj_a_rect.center))

    def _onfloor_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO: I am skipping all floor stuff for now. Decide later if we want to add it in
        return False
        # raise NotImplementedError("I don't know how to go about this one")

    def _open_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.openable_types:
            return state.get(obj,"open")
        return False

    def _toggled_on_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.toggleable_types:
            return state.get(obj,"toggled_on")
        return False

    def _cooked_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.cookable_types:
            return state.get(obj,"cooked")
        return False

    def _burnt_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.burnable_types:
            return state.get(obj,"burnt")
        return False

    def _frozen_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.freezable_types:
            return state.get(obj,"frozen")
        return False

    def _soaked_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.soakable_types:
            return state.get(obj,"soaked")
        return False

    def _dusty_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.dustyable_types:
            return state.get(obj,"dusty")
        return False

    def _stained_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.stainable_types:
            return state.get(obj,"stained")
        return False

    def _sliced_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.sliceable_types:
            return state.get(obj,"sliced")
        return False

    def _insideable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.insideable_types

    def _openable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.openable_types

    def _toggleable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.toggleable_types

    def _cookable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.cookable_types

    def _burnable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.burnable_types

    def _freezable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.freezable_types

    def _soakable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.soakable_types

    def _dustyable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.dustyable_types

    def _stainable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.stainable_types

    def _sliceable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.sliceable_types

    def _cooker_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.cooker_types

    def _freezer_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.freezer_types

    def _soaker_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.soaker_types    

    def _cleaner_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.cleaner_types

    def _slicer_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return obj.type.name in self.slicer_types

    def _reachable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        robby = self._robot

        # Held object is not reachable
        if state.get(obj, "held") == 1.0:
            return False
        obj_x = state.get(obj, "pose_x")
        obj_y = state.get(obj, "pose_y")
        obj_w = state.get(obj, "width")
        obj_h = state.get(obj, "height")
        obj_yaw = state.get(obj, "yaw")
        while obj_yaw > np.pi:
            obj_yaw -= (2 * np.pi)
        while obj_yaw < -np.pi:
            obj_yaw += (2 * np.pi)
        obj_rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)

        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        tip_x = robby_x + (self.robot_radius + self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + self.gripper_length) * np.sin(robby_yaw)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        return utils.line_segment_intersects_rectangle(gripper_line, obj_rect)

    def _handempty_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        robby = self._robot
        return state.get(robby, "gripper_free") == 1.0

    def _holding_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "held") == 1.0

    def _not_inside_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj_a, obj_b = objects
        if obj_a is obj_b:
            return False
        if (obj_a.type.name, obj_b.type.name) in self._need_pair_of_not_inside:
            return not self._inside_classifier(state, objects)
        return False

    def _closed_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.openable_types:
            return not state.get(obj,"open")
        return False

    def _toggled_off_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.toggleable_types:
            return not state.get(obj,"toggled_on")
        return False

    def _not_dusty_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.dustyable_types:
            return not state.get(obj,"dusty")
        return False

    def _not_stained_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if obj.type.name in self.stainable_types:
            return not state.get(obj,"stained")
        return False

    def _not_openable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return not (obj.type.name in self.openable_types)

    def _not_toggleable_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return not (obj.type.name in self.toggleable_types)

    def _reachable_nothing_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        for obj in state:
            if obj == self._robot:
                continue
            if self._reachable_classifier(state, [obj]):
                return False
        return True

    def _inside_nothing_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        for inside_obj in state:
            if inside_obj == self._robot:
                continue
            if self._inside_classifier(state, [obj, inside_obj]):
                return False
        return True

    def _not_holding_classifier(self, state: State, objects: Sequence[Object]) -> bool:
        return not self._holding_classifier(state, objects)

    def _get_pickable_object(self, state: State, tip_x: float, tip_y: float):
        pick_obj = None
        obj_rect = None
        for obj in state.data:
            if not obj.is_instance(self._robot_type):
                obj_x = state.get(obj, "pose_x")
                obj_y = state.get(obj, "pose_y")
                obj_w = state.get(obj, "width")
                obj_h = state.get(obj, "height")
                obj_yaw = state.get(obj, "yaw")
                while obj_yaw > np.pi:
                    obj_yaw -= (2 * np.pi)
                while obj_yaw < -np.pi:
                    obj_yaw += (2 * np.pi)
                obj_rect = utils.Rectangle(obj_x, obj_y, obj_w, obj_h, obj_yaw)
                if obj_rect.contains_point(tip_x, tip_y):
                    pick_obj = obj
                    break
        return pick_obj, obj_rect

    def _get_held_object(self, state):
        for obj in state:
            if not obj.is_instance(self._robot_type) and state.get(obj,
                                                              "held") == 1.0:
                return obj
        return None

    def _get_objects_inside(self, state: State, inside_obj: Object) -> List[Object]:
        obj_list: List[Object] = []
        for obj in state:
            if obj == self._robot:
                continue
            if self._inside_classifier(state, [obj, inside_obj]):
                obj_list.append(obj)

        return obj_list

    @staticmethod
    def get_id_by_object(state: State, obj: Object) -> int:
        sorted_objects = sorted([o for o in state.data], key=lambda x: x.name)
        return sorted_objects.index(obj)

    @staticmethod
    def get_object_by_id(state: State, idx: int) -> Object:
        sorted_objects = sorted([o for o in state.data], key=lambda x: x.name)
        return sorted_objects[idx]