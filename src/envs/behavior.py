"""Behavior (iGibson) environment."""
# pylint: disable=import-error

import functools
import itertools
import os
from typing import List, Set, Optional, Dict, Callable, Sequence, \
    Iterator, Union, Tuple
import numpy as np
from numpy.random._generator import Generator

try:
    import pybullet as pyb
    import bddl
    import igibson
    from igibson.simulator import Simulator  # pylint: disable=unused-import
    from igibson.envs import behavior_env
    from igibson.objects.articulated_object import (  # pylint: disable=unused-import
        ArticulatedObject, )
    from igibson.objects.articulated_object import URDFObject
    from igibson.object_states.on_floor import RoomFloor
    from igibson.robots.behavior_robot import BRBody
    from igibson.activity.bddl_backend import SUPPORTED_PREDICATES, \
        ObjectStateUnaryPredicate,ObjectStateBinaryPredicate
    from igibson.utils.checkpoint_utils import save_checkpoint, load_checkpoint

    _BEHAVIOR_IMPORTED = True
    bddl.set_backend("iGibson")  # pylint: disable=no-member
except ModuleNotFoundError as e:
    print(e)
    _BEHAVIOR_IMPORTED = False
from gym.spaces import Box
from predicators.src.envs.behavior_options import navigate_to_obj_pos, \
        grasp_obj_at_pos, place_ontop_obj_pos, create_navigate_policy, \
            create_grasp_policy, create_place_policy, \
                create_navigate_option_model, create_grasp_option_model, \
                    create_place_option_model
from predicators.src.envs import BaseEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG


class BehaviorEnv(BaseEnv):
    """Behavior (iGibson) environment."""

    def __init__(self) -> None:
        if not _BEHAVIOR_IMPORTED:
            raise ModuleNotFoundError("Behavior is not installed.")
        config_file = os.path.join(igibson.root_path, CFG.behavior_config_file)
        super().__init__()  # To ensure self._seed is defined.
        self._rng = np.random.default_rng(self._seed)
        self.igibson_behavior_env = behavior_env.BehaviorEnv(
            config_file=config_file,
            mode=CFG.behavior_mode,
            action_timestep=CFG.behavior_action_timestep,
            physics_timestep=CFG.behavior_physics_timestep,
            action_filter="mobile_manipulation",
            rng=self._rng,
        )
        self.igibson_behavior_env.robots[0].initial_z_offset = 0.7
        self._type_name_to_type: Dict[str, Type] = {}

        planner_fns: List[Callable[[
            "behavior_env.BehaviorEnv", Union[
                "URDFObject", "RoomFloor"], Array, Optional[Generator]
        ], Optional[Tuple[List[List[float]], List[List[float]]]]]] = [
            navigate_to_obj_pos, grasp_obj_at_pos, place_ontop_obj_pos
        ]
        option_policy_fns: List[
            Callable[[List[List[float]], List[List[float]]],
                     Callable[[State, "behavior_env.BehaviorEnv"],
                              Tuple[Array, bool]]]] = [
                                  create_navigate_policy, create_grasp_policy,
                                  create_place_policy
                              ]
        option_model_fns: List[
            Callable[[List[List[float]], List[List[float]]],
                     Callable[[State, "behavior_env.BehaviorEnv"], None]]] = [
                         create_navigate_option_model,
                         create_grasp_option_model, create_place_option_model
                     ]

        # name, planner_fn, option_policy_fn, option_model_fn,
        # param_dim, arity, parameter upper and lower bounds
        option_elems = [
            ("NavigateTo", planner_fns[0], option_policy_fns[0],
             option_model_fns[0], 2, 1, (-5.0, 5.0)),
            ("Grasp", planner_fns[1], option_policy_fns[2],
             option_model_fns[1], 3, 1, (-np.pi, np.pi)),
            ("PlaceOnTop", planner_fns[2], option_policy_fns[2],
             option_model_fns[2], 3, 1, (-1.0, 1.0)),
        ]
        self._options: Set[ParameterizedOption] = set()
        for (name, planner_fn, policy_fn, option_model_fn, param_dim, num_args,
             parameter_limits) in option_elems:
            # Create a different option for each type combo
            for types in itertools.product(self.types, repeat=num_args):
                option_name = self._create_type_combo_name(name, types)
                option = make_behavior_option(
                    option_name,
                    types=list(types),
                    params_space=Box(parameter_limits[0], parameter_limits[1],
                                     (param_dim, )),
                    env=self.igibson_behavior_env,
                    planner_fn=planner_fn,
                    policy_fn=policy_fn,
                    option_model_fn=option_model_fn,
                    object_to_ig_object=self.object_to_ig_object,
                    rng=self._rng,
                )
                self._options.add(option)

    def simulate(self, state: State, action: Action) -> State:
        assert state.simulator_state is not None
        load_checkpoint(
            self.igibson_behavior_env.simulator,
            "tmp_behavior_states/" + state.simulator_state.split("+")[1] + "/",
            int(state.simulator_state.split("+")[0]))

        a = action.arr
        self.igibson_behavior_env.step(a)
        # a[16] is used to indicate whether to grasp or release the currently-
        # held object. 1.0 indicates that the object should be grasped, and
        # -1.0 indicates it should be released
        if a[16] == 1.0:
            assisted_grasp_action = np.zeros(28, dtype=float)
            # We now need to create a 28-dimensional action to pass to
            # the assisted grasping code. Here, the 26th dimension dictates
            # whether to close the hand or not (1.0 indicates that the
            # hand should be closed)
            assisted_grasp_action[26] = 1.0
            _ = (self.igibson_behavior_env.robots[0].parts["right_hand"].
                 handle_assisted_grasping(assisted_grasp_action))
        elif a[16] == -1.0:
            released_obj = self.igibson_behavior_env.scene.get_objects()[
                self.igibson_behavior_env.robots[0].parts["right_hand"].
                object_in_hand]
            # force release object to avoid dealing with stateful assisted
            # grasping release mechanism
            self.igibson_behavior_env.robots[0].parts[
                "right_hand"].force_release_obj()
            # reset the released object to zero velocity
            pyb.resetBaseVelocity(
                released_obj.get_body_id(),
                linearVelocity=[0, 0, 0],
                angularVelocity=[0, 0, 0],
            )
        next_state = self.current_ig_state_to_state()
        return next_state

    def train_tasks_generator(self) -> Iterator[List[Task]]:
        yield self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def get_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        tasks = []
        for _ in range(num):
            # Behavior uses np.random everywhere. This is a somewhat
            # hacky workaround for that.
            np.random.seed(rng.integers(0, (2**32) - 1))
            self.igibson_behavior_env.reset()
            init_state = self.current_ig_state_to_state()
            goal = self._get_task_goal()
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    def _get_task_goal(self) -> Set[GroundAtom]:
        # Currently assumes that the goal is a single AND of
        # ground atoms (this is also assumed by the planner).
        goal = set()
        assert len(
            self.igibson_behavior_env.task.ground_goal_state_options) == 1
        for head_expr in self.igibson_behavior_env.task.\
            ground_goal_state_options[0]:
            bddl_name = head_expr.terms[0]  # untyped
            ig_objs = [self._name_to_ig_object(t) for t in head_expr.terms[1:]]
            objects = [self._ig_object_to_object(i) for i in ig_objs]
            pred_name = self._create_type_combo_name(bddl_name,
                                                     [o.type for o in objects])
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
                # "inside",
                # "nextto",
                "ontop",
                # "under",
                # "touching",
                # NOTE: OnFloor(robot, floor) does not evaluate to true
                # even though it's in the initial BDDL state, because
                # it uses geometry, and the behaviorbot actually floats
                # and doesn't touch the floor. But it doesn't matter.
                # "onfloor",
                # "cooked",
                # "burnt",
                # "frozen",
                # "soaked",
                # "open",
                # "dusty",
                # "stained",
                # "sliced",
                # "toggled_on",
        ]:
            bddl_predicate = SUPPORTED_PREDICATES[bddl_name]
            # We will create one predicate for every combination of types.
            # Ideally, we would filter out implausible type combinations
            # per predicate, but this should happen automatically when we
            # go to collect data and do NSRT learning.
            arity = self._bddl_predicate_arity(bddl_predicate)
            for type_combo in itertools.product(types_lst, repeat=arity):
                pred_name = self._create_type_combo_name(bddl_name, type_combo)
                classifier = self._create_classifier_from_bddl(bddl_predicate)
                pred = Predicate(pred_name, list(type_combo), classifier)
                predicates.add(pred)

        # Second, add in custom predicates except reachable-nothing
        custom_predicate_specs = [
            ("handempty", self._handempty_classifier, 0),
            ("holding", self._holding_classifier, 1),
            ("reachable", self._reachable_classifier, 2),
        ]

        for name, classifier, arity in custom_predicate_specs:
            for type_combo in itertools.product(types_lst, repeat=arity):
                pred_name = self._create_type_combo_name(name, type_combo)
                pred = Predicate(pred_name, list(type_combo), classifier)
                predicates.add(pred)

        # Finally, add the reachable-nothing predicate, which only applies
        # to the 'agent' type
        for i in range(len(types_lst)):
            if types_lst[i].name == "agent.n.01":
                pred_name = self._create_type_combo_name(
                    "reachable-nothing", (types_lst[i], ))
                pred = Predicate(
                    pred_name,
                    [types_lst[i]],
                    self._reachable_nothing_classifier,
                )
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
            obj_type = Type(
                type_name,
                [
                    "pos_x", "pos_y", "pos_z", "orn_0", "orn_1", "orn_2",
                    "orn_3"
                ],
            )
            self._type_name_to_type[type_name] = obj_type
        return set(self._type_name_to_type.values())

    @property
    def options(self) -> Set[ParameterizedOption]:
        return self._options

    @property
    def action_space(self) -> Box:
        # 17-dimensional, between -1 and 1
        assert self.igibson_behavior_env.action_space.shape == (17, )
        assert np.all(self.igibson_behavior_env.action_space.low == -1)
        assert np.all(self.igibson_behavior_env.action_space.high == 1)
        return self.igibson_behavior_env.action_space

    def render(self,
               state: State,
               task: Task,
               action: Optional[Action] = None) -> List[Image]:
        raise Exception("Cannot make videos for behavior env, change "
                        "behavior_mode in settings.py instead")

    def _get_task_relevant_objects(self) -> List["ArticulatedObject"]:
        return list(self.igibson_behavior_env.task.object_scope.values())

    @functools.lru_cache(maxsize=None)
    def _ig_object_to_object(self, ig_obj: "ArticulatedObject") -> Object:
        type_name = self._ig_object_to_type_name(ig_obj)
        obj_type = self._type_name_to_type[type_name]
        ig_obj_name = self._ig_object_name(ig_obj)
        return Object(ig_obj_name, obj_type)

    @functools.lru_cache(maxsize=None)
    def object_to_ig_object(self, obj: Object) -> "ArticulatedObject":
        """Maintains a mapping of objects to underlying igibson objects."""
        return self._name_to_ig_object(obj.name)

    @functools.lru_cache(maxsize=None)
    def _name_to_ig_object(self, name: str) -> "ArticulatedObject":
        for ig_obj in self._get_task_relevant_objects():
            if self._ig_object_name(ig_obj) == name:
                return ig_obj
        raise ValueError(f"No IG object found for name {name}.")

    @functools.lru_cache(maxsize=None)
    def _name_to_predicate(self, name: str) -> Predicate:
        for pred in self.predicates:
            if name == pred.name:
                return pred
        raise ValueError(f"No predicate found for name {name}.")

    def current_ig_state_to_state(self) -> State:
        """Function to create a predicators State from the current underlying
        iGibson simulator state."""
        state_data = {}
        for ig_obj in self._get_task_relevant_objects():
            obj = self._ig_object_to_object(ig_obj)
            # In the future, we may need other object attributes,
            # but for the moment, we just need position and orientation.
            obj_state = np.hstack([
                ig_obj.get_position(),
                ig_obj.get_orientation(),
            ])
            state_data[obj] = obj_state

        def save_unique_checkpoint(sim: "Simulator",
                                   save_dir: str) -> Tuple[str, str]:
            unique_id = int(np.random.randint(size=1, low=0, high=10**6))
            path = save_dir + str(unique_id) + "/"
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            simulator_state = save_checkpoint(sim, path)
            return str(simulator_state), str(unique_id)

        simulator_state, unique_id = save_unique_checkpoint(
            self.igibson_behavior_env.simulator, "tmp_behavior_states/")

        return State(state_data, simulator_state + "+" + unique_id)

    def _create_classifier_from_bddl(
        self,
        bddl_predicate: "bddl.AtomicFormula",
    ) -> Callable[[State, Sequence[Object]], bool]:

        def _classifier(s: State, o: Sequence[Object]) -> bool:
            # Behavior's predicates store the current object states
            # internally and use them to classify groundings of the
            # predicate. Because of this, we will assert that whenever
            # a predicate classifier is called, the internal simulator
            # state is equal to the state input to the classifier.
            try:
                assert s.allclose(self.current_ig_state_to_state())
            except AssertionError:
                import ipdb
                ipdb.set_trace()

            arity = self._bddl_predicate_arity(bddl_predicate)
            if arity == 1:
                assert len(o) == 1
                ig_obj = self.object_to_ig_object(o[0])
                bddl_ground_atom = bddl_predicate.STATE_CLASS(ig_obj)
                bddl_ground_atom.initialize(
                    self.igibson_behavior_env.simulator)
                return bddl_ground_atom.get_value()
            if arity == 2:
                assert len(o) == 2
                ig_obj = self.object_to_ig_object(o[0])
                other_ig_obj = self.object_to_ig_object(o[1])
                bddl_partial_ground_atom = bddl_predicate.STATE_CLASS(ig_obj)
                bddl_partial_ground_atom.initialize(
                    self.igibson_behavior_env.simulator)
                return bddl_partial_ground_atom.get_value(other_ig_obj)

            raise ValueError("BDDL predicate has unexpected arity.")

        return _classifier

    def _reachable_classifier(self, state: State,
                              objs: Sequence[Object]) -> bool:
        # Check allclose() here for uniformity with
        # _create_classifier_from_bddl
        try:
            assert state.allclose(self.current_ig_state_to_state())
        except AssertionError:
            import ipdb
            ipdb.set_trace()

        assert len(objs) == 2
        ig_obj = self.object_to_ig_object(objs[0])
        ig_other_obj = self.object_to_ig_object(objs[1])
        return (np.linalg.norm(  # type: ignore
            np.array(ig_obj.get_position()) -
            np.array(ig_other_obj.get_position())) < 2)

    def _reachable_nothing_classifier(self, state: State,
                                      objs: Sequence[Object]) -> bool:
        # Check allclose() here for uniformity with _create_classifier_from_bddl
        assert state.allclose(self.current_ig_state_to_state())
        assert len(objs) == 1
        for obj in state:
            if self._reachable_classifier(
                    state=state, objs=[obj, objs[0]]) and (obj != objs[0]):
                return False
        return True

    def _get_grasped_objects(self, state: State) -> Set[Object]:
        grasped_objs = set()
        for obj in state:
            ig_obj = self.object_to_ig_object(obj)

            # NOTE: The below block is necessary because somehow the body_id
            # is sometimes a 1-element list...
            if isinstance(ig_obj.body_id, list):
                assert len(ig_obj.body_id) == 1
                ig_obj.body_id = ig_obj.body_id[0]

            if np.any(self.igibson_behavior_env.robots[0].is_grasping(
                    ig_obj.body_id)):
                grasped_objs.add(obj)

        return grasped_objs

    def _handempty_classifier(self, state: State,
                              objs: Sequence[Object]) -> bool:
        # Check allclose() here for uniformity with
        # _create_classifier_from_bddl
        assert state.allclose(self.current_ig_state_to_state())
        assert len(objs) == 0
        grasped_objs = self._get_grasped_objects(state)
        return len(grasped_objs) == 0

    def _holding_classifier(self, state: State,
                            objs: Sequence[Object]) -> bool:
        # Check allclose() here for uniformity with
        # _create_classifier_from_bddl
        assert state.allclose(self.current_ig_state_to_state())
        assert len(objs) == 1
        grasped_objs = self._get_grasped_objects(state)
        return objs[0] in grasped_objs

    @staticmethod
    def _ig_object_name(ig_obj: "ArticulatedObject") -> str:
        if isinstance(ig_obj, (URDFObject, RoomFloor)):
            return ig_obj.bddl_object_scope
        # Robot does not have a field "bddl_object_scope", so we define
        # its name manually.
        assert isinstance(ig_obj, BRBody)
        return "agent.n.01_1"

    @staticmethod
    def _ig_object_to_type_name(ig_obj: "ArticulatedObject") -> str:
        ig_obj_name = BehaviorEnv._ig_object_name(ig_obj)
        if isinstance(ig_obj, RoomFloor):
            assert ":" in ig_obj_name
            type_name = ig_obj_name.split(":")[0]
            return type_name.rsplit("_", 1)[0]
        # Object is either URDFObject or robot.
        assert ":" not in ig_obj_name
        return ig_obj_name.rsplit("_", 1)[0]

    @staticmethod
    def _bddl_predicate_arity(bddl_predicate: "bddl.AtomicFormula") -> int:
        # NOTE: isinstance does not work here, maybe because of the
        # way that these bddl_predicate classes are created?
        if ObjectStateUnaryPredicate in bddl_predicate.__bases__:
            return 1
        if ObjectStateBinaryPredicate in bddl_predicate.__bases__:
            return 2
        raise ValueError("BDDL predicate has unexpected arity.")

    @staticmethod
    def _create_type_combo_name(original_name: str,
                                type_combo: Sequence[Type]) -> str:
        type_names = "-".join(t.name for t in type_combo)
        return f"{original_name}-{type_names}"


def make_behavior_option(
        name: str, types: Sequence[Type], params_space: Box,
        env: "behavior_env.BehaviorEnv", planner_fn: Callable[[
            "behavior_env.BehaviorEnv", Union[
                "URDFObject", "RoomFloor"], Array, Optional[Generator]
        ], Optional[Tuple[List[List[float]], List[List[float]]]]],
        policy_fn: Callable[[List[List[float]], List[List[float]]],
                            Callable[[State, "behavior_env.BehaviorEnv"],
                                     Tuple[Array, bool]]],
        option_model_fn: Callable[[List[List[float]], List[List[float]]],
                                  Callable[[State, "behavior_env.BehaviorEnv"],
                                           None]],
        object_to_ig_object: Callable[[Object], "ArticulatedObject"],
        rng: Generator) -> ParameterizedOption:
    """Makes an option for a BEHAVIOR env using custom implemented
    controller_fn."""

    def _policy(state: State, memory: Dict, _objects: Sequence[Object],
                _params: Array) -> Action:
        assert "has_terminated" in memory
        # must call initiable() first, and it must return True
        assert memory.get("rrt_plan") is not None
        assert memory.get("policy_controller") is not None
        assert not memory["has_terminated"]
        action_arr, memory["has_terminated"] = memory["policy_controller"](
            state, env)
        return Action(action_arr)

    def _initiable(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        igo = [object_to_ig_object(o) for o in objects]
        assert len(igo) == 1
        
        # Load the checkpoint associated with state.simulator_state
        # to make sure that we run RRT from the intended state.
        if state.simulator_state is not None:
            load_checkpoint(
                env.simulator, "tmp_behavior_states/" +
                state.simulator_state.split("+")[1] + "/",
                int(state.simulator_state.split("+")[0]))
            env.step(np.zeros(17))

        if memory.get("planner_result") is not None:
            # In this case, an rrt_plan has already been found for this
            # option (most likely, this will occur when executing a
            # series of options after having planned).
            return True

        # NOTE: the below type ignore comment is necessary because mypy
        # doesn't like that rng is being passed by keyword (seems to be
        # an issue with mypy: https://github.com/python/mypy/issues/1655)
        planner_result = planner_fn(env, igo[0], params,
                                    rng=rng)  # type: ignore
        if planner_result is not None:
            # We can unpack the planner result into the rrt_plan and the
            # original orientation of the robot or hand.
            memory["planner_result"] = planner_result
            # We know planner_result[0] is the rrt_plan and planner_result[1]
            # is the original orientation
            memory["policy_controller"] = policy_fn(
                memory["planner_result"][0], memory["planner_result"][1])
            memory["model_controller"] = option_model_fn(
                memory["planner_result"][0], memory["planner_result"][1])
        return planner_result is not None

    def _terminal(_state: State, memory: Dict, _objects: Sequence[Object],
                  _params: Array) -> bool:
        assert "has_terminated" in memory
        return memory["has_terminated"]

    return ParameterizedOption(
        name,
        types=types,
        params_space=params_space,
        _policy=_policy,
        _initiable=_initiable,
        _terminal=_terminal,
    )
