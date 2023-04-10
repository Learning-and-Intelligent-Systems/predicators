from typing import Dict, List, Optional, Sequence, Set
import logging

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators.envs.base_env import BaseEnv
from predicators.envs.ballbin import BallbinEnv
from predicators.envs.bookshelf import BookshelfEnv
from predicators.envs.boxtray import BoxtrayEnv
from predicators.envs.cupboard import CupboardEnv
from predicators.envs.stickbasket import StickbasketEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, GroundAtom, Object, ParameterizedOption, Predicate, State, Task, Type
from predicators import utils

class PlanarBehaviorEnv(BaseEnv):
    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._env1 = BallbinEnv(use_gui)
        self._env2 = BookshelfEnv(use_gui)
        self._env3 = BoxtrayEnv(use_gui)
        self._env4 = CupboardEnv(use_gui)
        self._env5 = StickbasketEnv(use_gui)
        self._env_list = [self._env1, self._env2, self._env3, self._env4, self._env5]

        # Types
        # TODO: create better hierarchical typing with pickable-object and placeable-object
        self._shared_object_type = Type("object-shared", ["pose_x", "pose_y", "width", "height", "yaw", "held"])
        self._shared_pickable_type = Type("pickable-shared", ["pose_x", "pose_y", "width", "height", "yaw", "held"], self._shared_object_type)
        self._shared_placeable_type = Type("placeable-shared", ["pose_x", "pose_y", "width", "height", "yaw", "held"], self._shared_object_type)
        self._shared_robot_type = Type("robot-shared", ["pose_x", "pose_y", "yaw", "gripper_free"])

        for env in self._env_list:
            env._object_type.set_parent(self._shared_object_type)
            env._pickable_type.set_parent(self._shared_pickable_type)
            env._placeable_type.set_parent(self._shared_placeable_type)
            env._robot_type.set_parent(self._shared_robot_type)
            for t in env.types:
                t.set_name(env.get_name() + '-' + t.name)
        # A type to hold the env type
        self._dummy_type = Type("dummy", ["indicator"])
        self._indicator_to_env_map = {1: self._env1, 2: self._env2, 3: self._env3, 4: self._env4, 5: self._env5}
        if CFG.planar_behavior_task_order == 'fixed':
            logging.info("Leaving planar behavior tasks in fixed order")
        elif CFG.planar_behavior_task_order == 'shuffled':
            logging.info("Shuffling planar behavior tasks")
            indicator_order = self._train_rng.permutation(list(self._indicator_to_env_map.keys()))
            self._indicator_to_env_map = {indicator: self._indicator_to_env_map[indicator] for indicator in indicator_order}
        if CFG.planar_behavior_task_order != 'interleaved':
            logging.info(f"Env order: {[env.get_name() for env in self._indicator_to_env_map.values()]}")

        # Predicates
        self._On = Predicate("On", [self._shared_pickable_type, self._shared_placeable_type], self._On_holds)
        self._CanReach = Predicate("CanReach", [self._shared_object_type, self._shared_robot_type], self._CanReach_holds)
        self._Holding = Predicate("Holding", [self._shared_pickable_type], self._Holding_holds)
        self._GripperFree = Predicate("GripperFree", [self._shared_robot_type], self._GripperFree_holds)

        # Options
        lowers = np.min([self._env1._NavigateTo.params_space.low, self._env2._NavigateTo.params_space.low, self._env3._NavigateTo.params_space.low, self._env4._NavigateTo.params_space.low, self._env5._NavigateTo.params_space.low], axis=0)
        uppers = np.min([self._env1._NavigateTo.params_space.high, self._env2._NavigateTo.params_space.high, self._env3._NavigateTo.params_space.high, self._env4._NavigateTo.params_space.high, self._env5._NavigateTo.params_space.high], axis=0)
        self._NavigateTo = utils.SingletonParameterizedOption(
            # variables: [robot, object to navigate to]
            # params: [offser_x, offset_y]
            "NavigateTo",
            self._NavigateTo_policy,
            types=[self._shared_robot_type, self._shared_object_type],
            params_space=Box(lowers, uppers))

        lowers = np.min([self._env1._PickBall.params_space.low, self._env2._PickBook.params_space.low, self._env3._PickBox.params_space.low, self._env4._PickCup.params_space.low, self._env5._PickStick.params_space.low], axis=0)
        uppers = np.min([self._env1._PickBall.params_space.high, self._env2._PickBook.params_space.high, self._env3._PickBox.params_space.high, self._env4._PickCup.params_space.high, self._env5._PickStick.params_space.high], axis=0)
        self._Pick = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick]
            # params: [offset_gripper, book_yaw]
            "Pick",
            self._Pick_policy,
            types=[self._shared_robot_type, self._shared_object_type],
            params_space=Box(lowers, uppers))

        lowers = np.min([self._env1._PlaceBallOnBin.params_space.low, self._env2._PlaceBookOnShelf.params_space.low, self._env3._PlaceBoxOnTray.params_space.low, self._env4._PlaceCupOnCupboard.params_space.low, self._env5._PlaceStickOnBasket.params_space.low], axis=0)
        uppers = np.min([self._env1._PlaceBallOnBin.params_space.high, self._env2._PlaceBookOnShelf.params_space.high, self._env3._PlaceBoxOnTray.params_space.high, self._env4._PlaceCupOnCupboard.params_space.high, self._env5._PlaceStickOnBasket.params_space.high], axis=0)
        self._Place = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick, object to place]
            # params: [offse_gripper]
            "Place",
            self._Place_policy,
            types=[self._shared_robot_type, self._shared_object_type, self._shared_object_type],
            params_space=Box(lowers, uppers))

        # Static dummy object
        self._dummy = Object("dummy", self._dummy_type)

    @classmethod
    def get_name(cls) -> str:
        return "planar_behavior"

    def simulate(self, state: State, action: Action) -> State:
        env_indicator = state.get(self._dummy, "indicator")
        env = self._indicator_to_env_map[env_indicator]
        return env.simulate(state, action)

    def _generate_train_tasks(self) -> List[Task]:
        tasks = []
        for indicator, env in self._indicator_to_env_map.items():
            env_tasks = env._generate_train_tasks()
            new_tasks = []
            for task in env_tasks:
                data = task.init.data
                data[self._dummy] = np.array([indicator])
                new_goal = set()
                for atom in task.goal:
                    if atom.predicate.name.startswith('On'):
                        new_goal.add(GroundAtom(self._On, atom.objects))
                    elif atom.predicate.name == 'CanReach':
                        new_goal.add(GroundAtom(self._CanReach, atom.objects))
                state = State(data)
                new_tasks.append(Task(state, new_goal))
            tasks += new_tasks
        if CFG.planar_behavior_task_order == 'interleaved':
            task_order = self._train_rng.permutation(len(tasks))
            tasks = [tasks[t] for t in task_order]
        return tasks

    def _generate_test_tasks(self) -> List[Task]:
        tasks = []
        for indicator, env in self._indicator_to_env_map.items():
            env_tasks = env._generate_test_tasks()
            new_tasks = []
            for task in env_tasks:
                data = task.init.data
                data[self._dummy] = np.array([indicator])
                new_goal = set()
                for atom in task.goal:
                    if atom.predicate.name.startswith('On'):
                        new_goal.add(GroundAtom(self._On, atom.objects))
                    elif atom.predicate.name == 'CanReach':
                        new_goal.add(GroundAtom(self._CanReach, atom.objects))
                state = State(data)
                new_tasks.append(Task(state, new_goal))
            tasks += new_tasks
        if CFG.planar_behavior_task_order == 'interleaved':
            task_order = self._test_rng.permutation(len(tasks))
            tasks = [tasks[t] for t in task_order]
        return tasks

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._On, self._CanReach, self._Holding, self._GripperFree}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._On}

    @property
    def types(self) -> Set[Type]:
        types = {self._shared_robot_type, self._shared_object_type, self._shared_pickable_type, self._shared_placeable_type, self._dummy_type}
        for env in self._env_list:
            types |= env.types
        return types

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._NavigateTo, self._Pick, self._Place}

    @property
    def action_space(self) -> Box:
        lowers = np.min([self._env1.action_space.low, self._env2.action_space.low, self._env3.action_space.low, self._env4.action_space.low, self._env5.action_space.low], axis=0)
        uppers = np.max([self._env1.action_space.high, self._env2.action_space.high, self._env3.action_space.high, self._env4.action_space.high, self._env5.action_space.high], axis=0)
        return Box(lowers, uppers)

    def render_state_plt(self,
        state: State,
        task: Task,
        action: Optional[Action] = None,
        caption: Optional[str] = None) -> matplotlib.figure.Figure:
        env_indicator = state.get(self._dummy, "indicator")
        env = self._indicator_to_env_map[env_indicator]
        return env.simulate(state, task, action, caption)

    def _Pick_policy(self, state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
        env_indicator = state.get(self._dummy, "indicator")
        env = self._indicator_to_env_map[env_indicator]
        if env.get_name() == 'ballbin':
            return env._PickBall_policy(state, memory, objects, params)
        elif env.get_name() == 'bookshelf':
            return env._PickBook_policy(state, memory, objects, params)
        elif env.get_name() == 'boxtray':
            return env._PickBox_policy(state, memory, objects, params)
        elif env.get_name() == 'cupboard':
            return env._PickCup_policy(state, memory, objects, params)
        elif env.get_name() == 'stickbasket':
            return env._PickStick_policy(state, memory, objects, params)

    def _Place_policy(self, state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
        env_indicator = state.get(self._dummy, "indicator")
        env = self._indicator_to_env_map[env_indicator]
        if env.get_name() == 'ballbin':
            return env._PlaceBallOnBin_policy(state, memory, objects, params)
        elif env.get_name() == 'bookshelf':
            return env._PlaceBookOnShelf_policy(state, memory, objects, params)
        elif env.get_name() == 'boxtray':
            return env._PlaceBoxOnTray_policy(state, memory, objects, params)
        elif env.get_name() == 'cupboard':
            return env._PlaceCupOnCupboard_policy(state, memory, objects, params)
        elif env.get_name() == 'stickbasket':
            return env._PlaceStickOnBasket_policy(state, memory, objects, params)

    def _NavigateTo_policy(self, state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
        env_indicator = state.get(self._dummy, "indicator")
        env = self._indicator_to_env_map[env_indicator]
        return env._NavigateTo_policy(state, memory, objects, params)

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        env_indicator = state.get(self._dummy, "indicator")
        env = self._indicator_to_env_map[env_indicator]
        return env._Holding_holds(state, objects)

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        env_indicator = state.get(self._dummy, "indicator")
        env = self._indicator_to_env_map[env_indicator]
        if env.get_name() == 'ballbin':
            return env._OnBin_holds(state, objects)
        elif env.get_name() == 'bookshelf':
            return env._OnShelf_holds(state, objects)
        elif env.get_name() == 'boxtray':
            return env._OnTray_holds(state, objects)
        elif env.get_name() == 'cupboard':
            return env._OnCupboard_holds(state, objects)
        elif env.get_name() == 'stickbasket':
            return env._OnBasket_holds(state, objects)

    def _GripperFree_holds(self, state: State, objects: Sequence[Object]) -> bool:
        env_indicator = state.get(self._dummy, "indicator")
        env = self._indicator_to_env_map[env_indicator]
        return env._GripperFree_holds(state, objects)

    def _CanReach_holds(self, state: State, objects: Sequence[Object]) -> bool:
        env_indicator = state.get(self._dummy, "indicator")
        env = self._indicator_to_env_map[env_indicator]
        return env._CanReach_holds(state, objects)