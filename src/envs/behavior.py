"""Behavior (iGibson) environment.
"""

import functools
import os
from typing import List, Set, Sequence, Dict, Tuple, Optional
import numpy as np
from gym.spaces import Box
try:
    import bddl
    import igibson
    from igibson.envs import behavior_env
    from igibson.objects.articulated_object import URDFObject
    from igibson.object_states.on_floor import RoomFloor
    from igibson.utils.checkpoint_utils import \
        save_internal_states, load_internal_states
    _BEHAVIOR_IMPORTED = True
    bddl.set_backend("iGibson")
except ModuleNotFoundError:
    _BEHAVIOR_IMPORTED = False
from predicators.src.envs import BaseEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG
from predicators.src import utils


class BehaviorEnv(BaseEnv):
    """Behavior (iGibson) environment.
    """
    def __init__(self) -> None:
        if not _BEHAVIOR_IMPORTED:
            raise ModuleNotFoundError("Behavior is not installed.")
        config_file = os.path.join(igibson.root_path,
                                   CFG.behavior_config_file)
        self._env = behavior_env.BehaviorEnv(
            config_file=config_file,
            mode=CFG.behavior_mode,
            action_timestep=CFG.behavior_action_timestep,
            physics_timestep=CFG.behavior_physics_timestep
        )
        self._type_name_to_type = {}
        super().__init__()

    def simulate(self, state: State, action: Action) -> State:
        assert state.simulator_state is not None
        # TODO test that this works as expected
        load_internal_states(self._env.simulator, state.simulator_state)
        self._env.step(action.arr)
        next_state = self._current_ig_state_to_state()
        return next_state

    def get_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               rng=self._train_rng)

    def get_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               rng=self._test_rng)

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        # TODO: figure out how to use rng here
        for _ in range(num):
            self._env.reset()
            init_state = self._current_ig_state_to_state()
            # TODO: get goal for task in predicates
            goal = {GroundAtom(self._Dummy, [])}
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    @property
    def predicates(self) -> Set[Predicate]:
        # TODO
        self._Dummy = Predicate("Dummy", [], lambda s, o: False)
        return {self._Dummy}

    @property
    def types(self) -> Set[Type]:
        self._env.reset()  # TODO is this required?
        for ig_obj in self._get_task_relevant_objects():
            # Create type
            type_name, _ = ig_obj.bddl_object_scope.rsplit("_", 1)
            if type_name in self._type_name_to_type:
                continue
            # TODO: get type-specific features
            obj_type = Type(type_name, ["pos_x", "pos_y", "pos_z",
                                        "orn_0", "orn_1", "orn_2", "orn_3"])
            self._type_name_to_type[type_name] = obj_type
        return set(self._type_name_to_type.values())

    @property
    def options(self) -> Set[ParameterizedOption]:
        # TODO
        return set()

    @property
    def action_space(self) -> Box:
        # 11-dimensional, between -1 and 1
        return self._env.action_space

    def render(self, state: State, task: Task,
               action: Optional[Action] = None) -> Image:
        # TODO
        import ipdb; ipdb.set_trace()

    def _get_task_relevant_objects(self):
        # https://github.com/Learning-and-Intelligent-Systems/iGibson/blob/f21102347be7f3cef2cc39b943b1cf3166a428f4/igibson/envs/behavior_mp_env.py#L104
        return [item for item in self._env.task.object_scope.values()
                if isinstance(item, URDFObject) or isinstance(item, RoomFloor)
        ]

    @functools.lru_cache(maxsize=None)
    def _ig_object_to_object(self, ig_obj):
        type_name, _ = ig_obj.bddl_object_scope.rsplit("_", 1)
        obj_type = self._type_name_to_type[type_name]
        return Object(ig_obj.bddl_object_scope, obj_type)

    def _current_ig_state_to_state(self):
        state_data = {}
        for ig_obj in self._get_task_relevant_objects():
            obj = self._ig_object_to_object(ig_obj)
            # Get object features
            # TODO: generalize this!
            obj_state = np.concatenate([
                ig_obj.get_position(),
                ig_obj.get_orientation(),
            ])
            state_data[obj] = obj_state
        simulator_state = save_internal_states(self._env.simulator)
        return State(state_data, simulator_state)
