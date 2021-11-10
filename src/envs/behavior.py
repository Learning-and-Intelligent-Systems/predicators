"""Behavior (iGibson) environment.
"""
# pylint:disable=import-error

import functools
import itertools
import os
from typing import List, Set, Optional
import numpy as np
from gym.spaces import Box
from predicators.src.envs import BaseEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image
from predicators.src.settings import CFG


_BEHAVIOR_IMPORTED = False

def _import_behavior():
    """Lazy imports for iGibson/Pybullet etc.

    These imports are not that slow, but they're nonzero
    time, so we import lazily to keep unit tests fast.
    """
    global _BEHAVIOR_IMPORTED
    try:
        import bddl
        import igibson
        from igibson.envs import behavior_env
        from igibson.objects.articulated_object import URDFObject
        from igibson.object_states.on_floor import RoomFloor
        from igibson.robots.behavior_robot import BRBody
        from igibson.utils.checkpoint_utils import \
            save_internal_states, load_internal_states
        from igibson.activity.bddl_backend import SUPPORTED_PREDICATES, \
            ObjectStateUnaryPredicate, ObjectStateBinaryPredicate
        _BEHAVIOR_IMPORTED = True
        bddl.set_backend("iGibson")  # pylint: disable=no-member
    except ModuleNotFoundError:
        _BEHAVIOR_IMPORTED = False


class BehaviorEnv(BaseEnv):
    """Behavior (iGibson) environment.
    """
    def __init__(self) -> None:
        if not _BEHAVIOR_IMPORTED:
            _import_behavior()  # Lazy imports
        if not _BEHAVIOR_IMPORTED:  # If still False, not installed
            raise ModuleNotFoundError("Behavior is not installed.")
        config_file = os.path.join(igibson.root_path,
                                   CFG.behavior_config_file)
        self._env = behavior_env.BehaviorEnv(
            config_file=config_file,
            mode=CFG.behavior_mode,
            action_timestep=CFG.behavior_action_timestep,
            physics_timestep=CFG.behavior_physics_timestep,
            action_filter="mobile_manipulation",
        )
        self._env.robots[0].initial_z_offset = 0.7

        self._type_name_to_type = {}

        super().__init__()

    def simulate(self, state: State, action: Action) -> State:
        assert state.simulator_state is not None
        # TODO: test that this works as expected
        load_internal_states(self._env.simulator, state.simulator_state)
        a = action.arr
        self._env.step(a)
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
        # Currently assumes that the goal is a single AND of
        # ground atoms (this is also assumed by the planner).
        goal = set()
        assert len(self._env.task.ground_goal_state_options) == 1
        for head_expr in self._env.task.ground_goal_state_options[0]:
            bddl_name = head_expr.terms[0]  # untyped
            ig_objs = [self._name_to_ig_object(t) for t in head_expr.terms[1:]]
            objects = [self._ig_object_to_object(i) for i in ig_objs]
            pred_name = self._create_type_combo_name(
                bddl_name, [o.type for o in objects])
            pred = self._name_to_predicate(pred_name)
            atom = GroundAtom(pred, objects)
            goal.add(atom)
        return goal

    @property
    def predicates(self) -> Set[Predicate]:
        predicates = set()
        types_lst = sorted(self.types)  # for determinism
        # First, extract predicates from iGibson
        for bddl_name in [
                "inside",
                "nextto",
                "ontop",
                "under",
                "touching",
                # NOTE: OnFloor(robot, floor) does not evaluate to true
                # even though it's in the initial BDDL state, because
                # it uses geometry, and the behaviorbot actually floats
                # and doesn't touch the floor. But it doesn't matter.
                "onfloor",
                "cooked",
                "burnt",
                "soaked",
                "open",
                "dusty",
                "stained",
                "sliced",
                "toggled_on",
                "frozen"]:
            bddl_predicate = SUPPORTED_PREDICATES[bddl_name]
            # We will create one predicate for every combination of types.
            # Ideally, we would filter out implausible type combinations
            # per predicate, but this should happen automatically when we
            # go to collect data and do operator learning.
            arity = self._bddl_predicate_arity(bddl_predicate)
            for type_combo in itertools.product(types_lst, repeat=arity):
                pred_name = self._create_type_combo_name(bddl_name, type_combo)
                classifier = self._create_classifier_from_bddl(bddl_predicate)
                pred = Predicate(pred_name, list(type_combo), classifier)
                predicates.add(pred)
        # Second, add in custom predicates
        custom_predicate_specs = [
            ("handempty", self._handempty_classifier, 0),
            ("holding", self._holding_classifier, 1),
            ("nextto-nothing", self._nextto_nothing_classifier, 1),
        ]
        for name, classifier, arity in custom_predicate_specs:
            for type_combo in itertools.product(types_lst, repeat=arity):
                pred_name = self._create_type_combo_name(name, type_combo)
                pred = Predicate(pred_name, list(type_combo), classifier)
                predicates.add(pred)
        return predicates

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self.predicates

    @property
    def types(self) -> Set[Type]:
        for ig_obj in self._get_task_relevant_objects():
            # Create type
            type_name = self._ig_object_to_type_name(ig_obj)
            if type_name in self._type_name_to_type:
                continue
            # In the future, we may need other object attributes,
            # but for the moment, we just need position and orientation.
            obj_type = Type(type_name, ["pos_x", "pos_y", "pos_z",
                                        "orn_0", "orn_1", "orn_2", "orn_3"])
            self._type_name_to_type[type_name] = obj_type
        return set(self._type_name_to_type.values())

    @property
    def options(self) -> Set[ParameterizedOption]:
        return set()

    @property
    def action_space(self) -> Box:
        # 17-dimensional, between -1 and 1
        assert self._env.action_space.shape == (17,)
        assert np.all(self._env.action_space.low == -1)
        assert np.all(self._env.action_space.high == 1)
        return self._env.action_space

    def render(self, state: State, task: Task,
               action: Optional[Action] = None) -> Image:
        raise Exception("Cannot make videos for behavior env, change "
                        "behavior_mode in settings.py instead")

    def _get_task_relevant_objects(self):
        return list(self._env.task.object_scope.values())

    @functools.lru_cache(maxsize=None)
    def _ig_object_to_object(self, ig_obj):
        type_name = self._ig_object_to_type_name(ig_obj)
        obj_type = self._type_name_to_type[type_name]
        ig_obj_name = self._ig_object_name(ig_obj)
        return Object(ig_obj_name, obj_type)

    @functools.lru_cache(maxsize=None)
    def _object_to_ig_object(self, obj):
        return self._name_to_ig_object(obj.name)

    @functools.lru_cache(maxsize=None)
    def _name_to_ig_object(self, name):
        for ig_obj in self._get_task_relevant_objects():
            if self._ig_object_name(ig_obj) == name:
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
            # In the future, we may need other object attributes,
            # but for the moment, we just need position and orientation.
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
            # predicate. Because of this, we will assert that whenever
            # a predicate classifier is called, the internal simulator
            # state is equal to the state input to the classifier.
            assert s.allclose(self._current_ig_state_to_state())
            arity = self._bddl_predicate_arity(bddl_predicate)
            if arity == 1:
                assert len(o) == 1
                ig_obj = self._object_to_ig_object(o[0])
                bddl_ground_atom = bddl_predicate.STATE_CLASS(ig_obj)
                bddl_ground_atom.initialize(self._env.simulator)
                # TODO: figure out why this is crashing with KeyError
                return bddl_ground_atom.get_value()
            if arity == 2:
                assert len(o) == 2
                ig_obj = self._object_to_ig_object(o[0])
                other_ig_obj = self._object_to_ig_object(o[1])
                bddl_partial_ground_atom = bddl_predicate.STATE_CLASS(ig_obj)
                bddl_partial_ground_atom.initialize(self._env.simulator)
                return bddl_partial_ground_atom.get_value(other_ig_obj)
            raise ValueError("BDDL predicate has unexpected arity.")
        return _classifier

    def _get_grasped_objects(self, state):
        grasped_objs = set()
        for obj in state:
            ig_obj = self._object_to_ig_object(obj)
            if any(self._env.robots[0].is_grasping(ig_obj)):
                grasped_objs.add(obj)
        return grasped_objs

    def _handempty_classifier(self, state, objs):
        # Check allclose() here for uniformity with _create_classifier_from_bddl
        assert state.allclose(self._current_ig_state_to_state())
        assert len(objs) == 0
        grasped_objs = self._get_grasped_objects(state)
        return len(grasped_objs) == 0

    def _holding_classifier(self, state, objs):
        # Check allclose() here for uniformity with _create_classifier_from_bddl
        assert state.allclose(self._current_ig_state_to_state())
        assert len(objs) == 1
        grasped_objs = self._get_grasped_objects(state)
        return objs[0] in grasped_objs

    def _nextto_nothing_classifier(self, state, objs):
        # Check allclose() here for uniformity with _create_classifier_from_bddl
        assert state.allclose(self._current_ig_state_to_state())
        assert len(objs) == 1
        ig_obj = self._object_to_ig_object(objs[0])
        bddl_predicate = SUPPORTED_PREDICATES["nextto"]
        for obj in state:
            other_ig_obj = self._object_to_ig_object(obj)
            bddl_ground_atom = bddl_predicate.STATE_CLASS(ig_obj)
            bddl_ground_atom.initialize(self._env.simulator)
            if bddl_ground_atom.get_value(other_ig_obj):
                return False
        return True

    @staticmethod
    def _ig_object_name(ig_obj):
        if isinstance(ig_obj, (URDFObject, RoomFloor)):
            return ig_obj.bddl_object_scope
        # Robot does not have a field "bddl_object_scope", so we define
        # its name manually.
        assert isinstance(ig_obj, BRBody)
        return "agent.n.01_1"

    @staticmethod
    def _ig_object_to_type_name(ig_obj):
        ig_obj_name = BehaviorEnv._ig_object_name(ig_obj)
        if isinstance(ig_obj, RoomFloor):
            assert ":" in ig_obj_name
            type_name = ig_obj_name.split(":")[0]
            return type_name.rsplit("_", 1)[0]
        # Object is either URDFObject or robot.
        assert ":" not in ig_obj_name
        return ig_obj_name.rsplit("_", 1)[0]

    @staticmethod
    def _bddl_predicate_arity(bddl_predicate):
        # NOTE: isinstance does not work here, maybe because of the
        # way that these bddl_predicate classes are created?
        if ObjectStateUnaryPredicate in bddl_predicate.__bases__:
            return 1
        if ObjectStateBinaryPredicate in bddl_predicate.__bases__:
            return 2
        raise ValueError("BDDL predicate has unexpected arity.")

    @staticmethod
    def _create_type_combo_name(original_name, type_combo):
        type_names = "-".join(t.name for t in type_combo)
        return f"{original_name}-{type_names}"
