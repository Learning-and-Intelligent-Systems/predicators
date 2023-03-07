"""Cozmo environment for testing lifelong learning."""

import logging
from typing import Callable, ClassVar, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import MultiDiscrete

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import CozmoAction, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, Dict, Array

import cozmo
import asyncio

class CozmoEnv(BaseEnv):
    """An environment for Anki Cozmo Robot."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._robot_type = Type("robot", ["x", "y", "z", "rx", "ry", "rz"])
        self._cube_type = Type("cube", ["x", "y", "z", "rx", "ry", "rz", "touched", "color", "id"])
        self._dock_type = Type("dock", ["x", "y", "z"])
        # Predicates
        self._Reachable = Predicate("Reachable",
                                  [self._robot_type, self._cube_type],
                                  self._NextTo_holds)
        self._NextTo = Predicate("NextTo",
                                  [self._cube_type, self._cube_type],
                                  self._NextTo_holds)
        self._Touched = Predicate("Touched",
                                  [self._cube_type],
                                  self._Touched_holds)
        self._IsRed = Predicate("IsRed",
                                  [self._cube_type],
                                  self._IsRed_holds)
        self._IsBlue = Predicate("IsBlue",
                                  [self._cube_type],
                                  self._IsBlue_holds)
        self._IsGreen = Predicate("IsGreen",
                                  [self._cube_type],
                                  self._IsGreen_holds)
        self._OnTop = Predicate("OnTop",
                                  [self._cube_type, self._cube_type],
                                  self._OnTop_holds)
        self._Under = Predicate("Under",
                                  [self._cube_type, self._cube_type],
                                  self._Under_holds)
        # self._IsRightSideUp = Predicate("IsRightSideUp",
        #                           [self._cube_type],
        #                           self._IsRightSideUp_holds)
        # self._IsUpSideDown = Predicate("IsUpSideDown",
        #                           [self._cube_type],
        #                           self._IsUpSideDown_holds)
        # self._IsOnSide = Predicate("IsRightSideUp",
        #                           [self._cube_type],
        #                           self._IsOnSide_holds)

        # Static objects (always exist no matter the settings).
        self._robot = Object("cozmo", self._robot_type)
        self._cube_0 = Object("cube_0", self._cube_type)
        self._cube_1 = Object("cube_1", self._cube_type)
        self._cube_2 = Object("cube_2", self._cube_type)
        self._dock_0 = Object("dock_0", self._dock_type)

        # Last State
        self._last_state: Optional(State) = None

    @classmethod
    def get_name(cls) -> str:
        return "cozmo"

    def get_state(self, last_state: State = None, state_info: str = "all") -> State:
        def cozmo_program(robot: cozmo.robot.Robot):
            assert state_info == "all" or state_info == "robot"
            state = {}
            state[self._robot] = {
                                    "x": robot.pose.position.x,
                                    "y": robot.pose.position.y,
                                    "z": robot.pose.position.z,
                                    "rx": robot.pose.rotation.euler_angles[0],
                                    "ry": robot.pose.rotation.euler_angles[1],
                                    "rz": robot.pose.rotation.euler_angles[2]
                                }
            if state_info == "all":
                lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
                cubes = robot.world.wait_until_observe_num_objects(num=3, object_type=cozmo.objects.LightCube, timeout=30)
                dock = robot.world.wait_for_observed_charger(timeout=30)
                lookaround.stop()
                name_to_object = {"cube_0": self._cube_0, "cube_1": self._cube_1, "cube_2": self._cube_2, "dock_0": self._dock_0}
                for i, cube in enumerate(cubes):
                    state[name_to_object[f"cube_{i}"]] = {
                                                            "x": cube.pose.position.x,
                                                            "y": cube.pose.position.y,
                                                            "z": cube.pose.position.z,
                                                            "rx": cube.pose.rotation.euler_angles[0],
                                                            "ry": cube.pose.rotation.euler_angles[1],
                                                            "rz": cube.pose.rotation.euler_angles[2],
                                                            "touched": False,
                                                            "color": 0,
                                                            "id": cube.object_id
                                                        }
                if dock is not None:
                    state[self._dock_0] = {
                                            "x": dock.pose.position.x,
                                            "y": dock.pose.position.y,
                                            "z": dock.pose.position.z,
                                            "rx": dock.pose.rotation.euler_angles[0],
                                            "ry": dock.pose.rotation.euler_angles[1],
                                            "rz": dock.pose.rotation.euler_angles[2]
                                        }
                if last_state is None:
                    self._last_state = utils.create_state_from_dict(state)
                    return
            for obj, features in state.items():
                for feature_name, feature_val in features.items():
                    last_state.set(obj, feature_name, feature_val)
                    self._last_state = last_state
        cozmo.run_program(cozmo_program)
        assert self._last_state is not None
        return self._last_state

    def simulate(self, state: State, action: CozmoAction) -> State:
        action_id = int(action.arr[0])
        object_id1 = int(action.arr[1])
        object_id2 = int(action.arr[2])

        # (0) MoveTo
        # (1) Touch
        # (2) Paint
        # (3) PlaceOntop

        object_id_to_obj = {"r": self._robot,
                            state.get(self._cube_0, "id"): self._cube_0,
                            state.get(self._cube_1, "id"): self._cube_1,
                            state.get(self._cube_2, "id"): self._cube_2,
                            "d":self._dock_0}

        next_state = state.copy()
        if action_id == 0:
            target_x = state.get(object_id_to_obj[object_id2], "x")
            target_y = state.get(object_id_to_obj[object_id2], "y")
            next_state.set(self._robot, "x", target_x)
            next_state.set(self._robot, "y", target_y)
        elif action_id == 1:
            next_state.set(object_id_to_obj[object_id1], "touched", True)
        elif action_id == 2:
            next_state.set(object_id_to_obj[object_id2], "color", object_id1)
        elif action_id == 3:
            target_x = state.get(object_id_to_obj[object_id2], "x")
            target_y = state.get(object_id_to_obj[object_id2], "y")
            target_z = state.get(object_id_to_obj[object_id2], "z") + 1
            # for obj in object_id_to_obj.values():
            #     if self._OnTop_holds(state, [obj, object_id_to_obj[object_id1]]):
            #         next_state.set(obj, "x", target_x)
            #         next_state.set(obj, "y", target_y)
            #         next_state.set(obj, "z", target_z + 1)
            next_state.set(object_id_to_obj[object_id1], "x", target_x)
            next_state.set(object_id_to_obj[object_id1], "y", target_y)
            next_state.set(object_id_to_obj[object_id1], "z", target_z)
        return next_state

    def execute_on_robot(self, state: State, action: CozmoAction) -> State:
        action_id = int(action.arr[0])
        # TODO Run Cozmo Action
        cozmo_program = action.run
        cozmo.run_program(cozmo_program)
        # TODO Don't use cozmo as simulator
        if action_id in [0, 1, 2, 3]:
            next_state = self.get_state(last_state=state, state_info="robot")
            next_state = self.simulate(state, action)
        else:
            next_state = self.get_state(last_state=state, state_info="all")
        return next_state

    def step(self, action: CozmoAction) -> State:
        self._current_state = self.execute_on_robot(self._current_state, action)
        # Copy to prevent external changes to the environment's state.
        return self._current_state.copy()

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Reachable, self._NextTo, self._Touched, self._IsRed, self._IsBlue, self._IsGreen, self._OnTop, self._Under}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Reachable, self._NextTo, self._Touched, self._IsRed, self._IsBlue, self._IsGreen, self._OnTop, self._Under}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._cube_type, self._dock_type}

    @property
    def options(self) -> Set[ParameterizedOption]:  # pragma: no cover
        raise NotImplementedError(
            "This base class method will be deprecated soon!")

    @property
    def action_space(self) -> MultiDiscrete:
        # Discrete actions and objects.
        return MultiDiscrete([4, 5, 5])

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[CozmoAction] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        def cozmo_program(robot: cozmo.robot.Robot):
            # Turn on image receiving by the camera
            robot.camera.image_stream_enabled = True
            latest_image = robot.world.latest_image.raw_image
            latest_image.show()
        # latest_image.save("latest_cozmo_image.jpg")
        cozmo.run_program(cozmo_program)
        return fig

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        # There is only one goal in this environment.
        goal_atom1 = GroundAtom(self._Reachable, [self._robot, self._cube_0])
        goal_atom2 = GroundAtom(self._Touched, [self._cube_0])
        goal_atom3 = GroundAtom(self._IsRed, [self._cube_0])
        goal_atom4 = GroundAtom(self._OnTop, [self._cube_0, self._cube_1])
        goal_atom5 = GroundAtom(self._Under, [self._cube_1, self._cube_0])
        goal_atom6 = GroundAtom(self._IsBlue, [self._cube_1])
        goal_atom7 = GroundAtom(self._IsGreen, [self._cube_2])
        goal_atom8 = GroundAtom(self._OnTop, [self._cube_1, self._cube_2])
        goal = {goal_atom3, goal_atom6, goal_atom7, goal_atom4}
        # goal = {goal_atom4, goal_atom8}
        # The initial positions of the robot and dot vary. The only constraint
        # is that the initial positions should be far enough away that the goal
        # is not initially satisfied.
        tasks: List[Task] = []
        while len(tasks) < num:
            #state = self.get_state() # TODO hardcode
            state = State(data={
                            self._robot: np.array([0., 0., 0., 1.57079633, 1.57334067, 0.]),
                            self._cube_0: np.array([ 3.36951080e+02, -5.80603256e+01, -5.41569901e+00,  1.57079631e+00, -1.49432387e+00,  7.64963000e-08,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]),
                            self._cube_1: np.array([ 3.39060669e+02,  8.83256760e+01, -1.72424316e-03,  1.57079564e+00, -1.89550696e+00, -7.80109033e-07,  0.00000000e+00,  0.00000000e+00, 2.00000000e+00]),
                            self._cube_2: np.array([ 3.37060669e+02,  1.83256760e+01, -2.72424316e-00,  1.57079594e+00, -1.69550696e+00, -7.80109033e-07,  0.00000000e+00,  0.00000000e+00, 3.00000000e+00]),
                            self._dock_0: np.array([-187.11340332,   -8.03570366,   -1.11306763])}, simulator_state=None)
            # Make sure goal is not satisfied.
            if not all([goal_atom.holds(state) for goal_atom in goal]):
                tasks.append(Task(state, goal))
        return tasks

    def _NextTo_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        x1 = state.get(obj1, "x")
        y1 = state.get(obj1, "y")
        x2 = state.get(obj2, "x")
        y2 = state.get(obj2, "y")
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return dist < 200

    def _Touched_holds(self, state: State, objects: Sequence[Object]) -> bool:
        assert len(objects) == 1
        cube = objects[0]
        return state.get(cube, "touched")
        
    def _IsRed_holds(self, state: State, objects: Sequence[Object]) -> bool:
        assert len(objects) == 1
        cube = objects[0]
        return int(state.get(cube, "color")) == 1

    def _IsBlue_holds(self, state: State, objects: Sequence[Object]) -> bool:
        assert len(objects) == 1
        cube = objects[0]
        return int(state.get(cube, "color")) == 2

    def _IsGreen_holds(self, state: State, objects: Sequence[Object]) -> bool:
        assert len(objects) == 1
        cube = objects[0]
        return int(state.get(cube, "color")) == 3

    def _OnTop_holds(self, state: State, objects: Sequence[Object]) -> bool:
        cube1, cube2 = objects
        x1 = state.get(cube1, "x")
        y1 = state.get(cube1, "y")
        z1 = state.get(cube1, "z")
        x2 = state.get(cube2, "x")
        y2 = state.get(cube2, "y")
        z2 = state.get(cube2, "z")
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        zdist = (z1 - z2)
        return (dist < 10) and (zdist > 0)

    def _Under_holds(self, state: State, objects: Sequence[Object]) -> bool:
        assert len(objects) == 2
        return self._OnTop_holds(state, [objects[1], objects[0]])