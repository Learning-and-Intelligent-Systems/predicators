"""Behavior (iGibson) environment.
"""

import functools
import itertools
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
    from igibson.activity.bddl_backend import SUPPORTED_PREDICATES, \
        ObjectStateUnaryPredicate, ObjectStateBinaryPredicate
    from bddl.condition_evaluation import get_predicate_for_token    
    _BEHAVIOR_IMPORTED = True
    bddl.set_backend("iGibson")
except ModuleNotFoundError:
    _BEHAVIOR_IMPORTED = False
from predicators.src.envs import BaseEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG
from predicators.src import utils


# TODO move this to settings
_BDDL_PREDICATE_NAMES = {
    "inside",
    "nextto",
    "ontop",
    "under",
    "touching",
    "onfloor",
    "cooked",
    "burnt",
    "soaked",
    "open",
    "dusty",
    "stained",
    "sliced",
    "toggled_on",
    "frozen",
}


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
            goal = self._get_task_goal()
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    def _get_task_goal(self) -> Set[GroundAtom]:
        # TODO: figure out a more general, less hacky way to do this.
        # Currently assumes that the goal is a single AND of
        # ground atoms (this is also assumed by the planner).
        goal = set()
        assert len(self._env.task.ground_goal_state_options) == 1
        for head_expr in self._env.task.ground_goal_state_options[0]:
            bddl_name = head_expr.terms[0]  # untyped
            ig_objs = [self._name_to_ig_object(t) for t in head_expr.terms[1:]]
            objects = [self._ig_object_to_object(i) for i in ig_objs]
            pred_name = _create_predicate_name(bddl_name,
                                               [o.type for o in objects])
            pred = self._name_to_predicate(pred_name)
            atom = GroundAtom(pred, objects)
            goal.add(atom)
        return goal

    @property
    def predicates(self) -> Set[Predicate]:
        predicates = set()
        types_lst = sorted(self.types)  # for determinism
        for bddl_name in _BDDL_PREDICATE_NAMES:
            bddl_predicate = SUPPORTED_PREDICATES[bddl_name]
            # We will create one predicate for every combination of types.
            # TODO: filter out implausible type combinations per predicate.
            arity = _bddl_predicate_arity(bddl_predicate)
            for type_combo in itertools.product(types_lst, repeat=arity):
                pred_name = _create_predicate_name(bddl_name, type_combo)
                _classifier = self._create_classifier_from_bddl(bddl_predicate)
                pred = Predicate(pred_name, list(type_combo), _classifier)
                predicates.add(pred)
        return predicates

    @property
    def types(self) -> Set[Type]:
        for ig_obj in self._get_task_relevant_objects():
            # Create type
            type_name = _ig_object_to_type_name(ig_obj)
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
                if isinstance(item, URDFObject) or isinstance(item, RoomFloor)]

    @functools.lru_cache(maxsize=None)
    def _ig_object_to_object(self, ig_obj):
        type_name = _ig_object_to_type_name(ig_obj)
        obj_type = self._type_name_to_type[type_name]
        return Object(ig_obj.bddl_object_scope, obj_type)

    @functools.lru_cache(maxsize=None)
    def _object_to_ig_object(self, obj):
        return self._name_to_ig_object(obj.name)

    @functools.lru_cache(maxsize=None)
    def _name_to_ig_object(self, name):
        for ig_obj in self._get_task_relevant_objects():
            if ig_obj.bddl_object_scope == name:
                return ig_obj
        raise ValueError(f"No IG object found for name {name}.")

    @functools.lru_cache(maxsize=None)
    def _name_to_predicate(self, name):
        for pred in self.predicates:
            if name == pred.name:
                return pred
        raise ValueError(f"No predicate found for name {name}.")

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

    def _create_classifier_from_bddl(self, bddl_predicate):
        def _classifier(s, o):
            # Behavior's predicates store the current object states
            # internally and use them to classify groundings of the
            # predicate. Because of this, we will assume that whenever
            # a predicate classifier is called, the internal simulator
            # state is equal to the state input to the classifier.

            # TODO: assert that this assumption holds. Need to implement allclose.
            # assert s.allclose(self._current_ig_state_to_state)
            
            # TODO: can we and should we do some sort of caching here?
            
            arity = _bddl_predicate_arity(bddl_predicate)

            if arity == 1:
                assert len(o) == 1
                ig_obj = self._object_to_ig_object(o[0])
                bddl_ground_atom = bddl_predicate.STATE_CLASS(ig_obj)
                bddl_ground_atom.initialize(self._env.simulator)

                try:
                    return bddl_ground_atom.get_value()
                except KeyError:  # TODO investigate more
                    return False
            
            if arity == 2:
                assert len(o) == 2
                ig_obj = self._object_to_ig_object(o[0])
                other_ig_obj = self._object_to_ig_object(o[1])
                bddl_partial_ground_atom = bddl_predicate.STATE_CLASS(ig_obj)
                bddl_partial_ground_atom.initialize(self._env.simulator)
                try:
                    return bddl_partial_ground_atom.get_value(other_ig_obj)
                except KeyError:  # TODO investigate more
                    return False
            
            raise ValueError("BDDL predicate has unexpected arity.")
        return _classifier


def _ig_object_to_type_name(ig_obj):
    if isinstance(ig_obj, RoomFloor):
        assert ":" in ig_obj.bddl_object_scope
        type_name = ig_obj.bddl_object_scope.split(":")[0]
        return type_name.rsplit("_", 1)[0]
    assert isinstance(ig_obj, URDFObject)
    return ig_obj.bddl_object_scope.rsplit("_", 1)[0]


def _bddl_predicate_arity(bddl_predicate):
    # isinstance does not work here, maybe because of the way
    # that these bddl_predicate classes are created?
    if ObjectStateUnaryPredicate in bddl_predicate.__bases__:
        return 1
    if ObjectStateBinaryPredicate in bddl_predicate.__bases__:
        return 2
    raise ValueError("BDDL predicate has unexpected arity.")


def _create_predicate_name(bddl_name, type_combo):
    type_names = "-".join(t.name for t in type_combo)
    return f"{bddl_name}-{type_names}"

