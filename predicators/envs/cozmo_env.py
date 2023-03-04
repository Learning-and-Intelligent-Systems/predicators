"""Cozmo environment for testing lifelong learning."""

import logging
from typing import Callable, ClassVar, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Discrete

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
        self._cube_type = Type("cube", ["x", "y", "z", "rx", "ry", "rz"])
        self._dock_type = Type("dock", ["x", "y", "z"])
        # Predicates
        self._NextTo = Predicate("NextTo",
                                  [self._robot_type, self._cube_type],
                                  self._NextTo_holds)
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

    def get_state(self) -> State:
        def cozmo_program(robot: cozmo.robot.Robot):
            lookaround = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
            cubes = robot.world.wait_until_observe_num_objects(num=3, object_type=cozmo.objects.LightCube, timeout=30)
            #charger = robot.world.wait_for_observed_charger(timeout=30)
            lookaround.stop()
            state = {}
            state[self._robot] = {
                                    "x": robot.pose.position.x,
                                    "y": robot.pose.position.y,
                                    "z": robot.pose.position.z,
                                    "rx": robot.pose.rotation.euler_angles[0],
                                    "ry": robot.pose.rotation.euler_angles[1],
                                    "rz": robot.pose.rotation.euler_angles[2]
                                }
            name_to_object = {"cube_0": self._cube_0, "cube_1": self._cube_1, "cube_2": self._cube_2, "dock_0": self._dock_0}
            for i, cube in enumerate(cubes):
                state[name_to_object[f"cube_{i}"]] = {
                                                        "x": cube.pose.position.x,
                                                        "y": cube.pose.position.y,
                                                        "z": cube.pose.position.z,
                                                        "rx": cube.pose.rotation.euler_angles[0],
                                                        "ry": cube.pose.rotation.euler_angles[1],
                                                        "rz": cube.pose.rotation.euler_angles[2]
                                                    }
            # if charger:
            #     obj_name_to_obj[f"charger_0"] = charger
            #     state["charger_0"] = {"pose": charger.pose}
            # For ontop, nextto, under, sees
            if self._last_state is None:
                self._last_state = utils.create_state_from_dict(state)
            else:
                for obj, features in state.items():
                    if obj == self._robot:
                        for feature_name, feature_val in features.items():
                            self._last_state.set(obj, feature_name, feature_val)
        cozmo.run_program(cozmo_program)
        assert self._last_state is not None
        return self._last_state

    def simulate(self, state: State, action: CozmoAction) -> State:
        # TODO Run Cozmo Action
        cozmo_program = action.run
        cozmo.run_program(cozmo_program)
        # TODO Don't use cozmo as simulator
        next_state = self.get_state()
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._NextTo}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._NextTo}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._cube_type, self._dock_type}

    @property
    def options(self) -> Set[ParameterizedOption]:  # pragma: no cover
        raise NotImplementedError(
            "This base class method will be deprecated soon!")

    @property
    def action_space(self) -> Discrete:
        # Discrete actions and objects.
        return Discrete(0)

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[CozmoAction] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        robot_color = "red"
        target_color = "blue"
        rad = (self.touch_multiplier * self.action_magnitude) / 2
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        target_x = state.get(self._cube_0, "x")
        target_y = state.get(self._cube_0, "y")
        robot_circ = plt.Circle((robot_x, robot_y), rad, color=robot_color)
        target_circ = plt.Circle((target_x, target_y), rad, color=target_color)
        ax.add_patch(robot_circ)
        ax.add_patch(target_circ)
        ax.set_xlim(self.x_lb - rad, self.x_ub + rad)
        ax.set_ylim(self.y_lb - rad, self.y_ub + rad)
        title = f"{robot_color} = robot, {target_color} = target"
        if caption is not None:
            title += f";\n{caption}"
        plt.suptitle(title, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        # There is only one goal in this environment.
        goal_atom = GroundAtom(self._NextTo, [self._robot, self._cube_0])
        goal = {goal_atom}
        # The initial positions of the robot and dot vary. The only constraint
        # is that the initial positions should be far enough away that the goal
        # is not initially satisfied.
        tasks: List[Task] = []
        while len(tasks) < num:
            state = self.get_state()
            # Make sure goal is not satisfied.
            if not goal_atom.holds(state):
                tasks.append(Task(state, goal))
        return tasks

    def _NextTo_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, target = objects
        rx = state.get(robot, "x")
        ry = state.get(robot, "y")
        cx = state.get(target, "x")
        cy = state.get(target, "y")
        dist = np.sqrt((rx - cx)**2 + (ry - cy)**2)
        print("Dist", dist)
        return dist < 200