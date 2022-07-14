"""Behavior (iGibson) environment."""
# pylint: disable=import-error

import functools
import itertools
import os
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import matplotlib
import numpy as np
from numpy.random._generator import Generator

try:
    import bddl
    import igibson
    import pybullet as pyb
    from igibson.activity.bddl_backend import SUPPORTED_PREDICATES, \
        ObjectStateBinaryPredicate, ObjectStateUnaryPredicate
    from igibson.envs import behavior_env
    from igibson.object_states.on_floor import RoomFloor
    from igibson.objects.articulated_object import \
        ArticulatedObject  # pylint: disable=unused-import
    from igibson.objects.articulated_object import \
        URDFObject  # pylint: disable=unused-import
    from igibson.robots.behavior_robot import BRBody
    from igibson.simulator import Simulator  # pylint: disable=unused-import
    from igibson.utils.checkpoint_utils import load_checkpoint, save_checkpoint
    from igibson.utils.utils import modify_config_file

    _BEHAVIOR_IMPORTED = True
    bddl.set_backend("iGibson")  # pylint: disable=no-member
    if not os.path.exists("tmp_behavior_states/"):
        os.makedirs("tmp_behavior_states/")
except (ImportError, ModuleNotFoundError) as e:
    _BEHAVIOR_IMPORTED = False
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.envs.behavior_options import create_grasp_option_model, \
    create_grasp_policy, create_navigate_option_model, \
    create_navigate_policy, create_place_option_model, create_place_policy, \
    grasp_obj_at_pos, navigate_to_obj_pos, place_ontop_obj_pos
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, Video


class BehaviorEnv(BaseEnv):
    """BEHAVIOR (iGibson) environment."""

    def __init__(self) -> None:
        if not _BEHAVIOR_IMPORTED:
            raise ModuleNotFoundError("BEHAVIOR is not installed.")
        # behavior_randomize_init_state will always be False in this
        # config_file because we are not using their scene samplers.
        # We are loading pre-computed scenes.
        self._config_file = modify_config_file(
            os.path.join(igibson.root_path, CFG.behavior_config_file),
            CFG.behavior_task_name, CFG.behavior_scene_name, False)

        super().__init__()  # To ensure self._seed is defined.
        self._rng = np.random.default_rng(self._seed)
        self.task_num = 0  # unique id to differentiate tasks
        self.task_instance_id = 0  # id used for scene
        self.set_igibson_behavior_env(task_instance_id=self.task_instance_id,
                                      seed=self._seed)
        self._type_name_to_type: Dict[str, Type] = {}
        # a map between task nums and the snapshot id for saving/loading
        # purposes
        self.task_num_task_instance_id_to_igibson_seed: Dict[Tuple[int, int],
                                                             int] = {}
        # Everytime we load a new scene in BEHAVIOR we also need to set
        # the valid options again, because there might be new type combos.
        self.set_options()

    def set_options(self) -> None:
        """Sets the underlying options for this particular task using the
        current type list."""
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
            Callable[[List[List[float]], List[List[float]], "URDFObject"],
                     Callable[[State, "behavior_env.BehaviorEnv"], None]]] = [
                         create_navigate_option_model,
                         create_grasp_option_model, create_place_option_model
                     ]

        # name, planner_fn, option_policy_fn, option_model_fn,
        # param_dim, arity, parameter upper and lower bounds
        option_elems = [
            ("NavigateTo", planner_fns[0], option_policy_fns[0],
             option_model_fns[0], 2, 1, (-5.0, 5.0)),
            ("Grasp", planner_fns[1], option_policy_fns[1],
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
                    env=self,
                    planner_fn=planner_fn,
                    policy_fn=policy_fn,
                    option_model_fn=option_model_fn,
                    object_to_ig_object=self.object_to_ig_object,
                    rng=self._rng,
                )
                self._options.add(option)

    @classmethod
    def get_name(cls) -> str:
        return "behavior"

    def simulate(self, state: State, action: Action) -> State:
        assert isinstance(state.simulator_state, str)
        self.task_num = int(state.simulator_state.split("-")[0])
        self.task_instance_id = int(state.simulator_state.split("-")[1])
        if not state.allclose(
                self.current_ig_state_to_state(save_state=False)):
            load_checkpoint_state(state, self)

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

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               rng=self._test_rng,
                               testing=True)

    def _get_tasks(self,
                   num: int,
                   rng: np.random.Generator,
                   testing: bool = False) -> List[Task]:
        tasks = []
        for _ in range(num):
            # BEHAVIOR uses np.random everywhere. This is a somewhat
            # hacky workaround for that.
            curr_env_seed = rng.integers(0, (2**32) - 1)
            # ID used to generate scene in BEHAVIOR default scene is 0
            self.task_instance_id = 0
            if CFG.behavior_randomize_init_state:
                # Get random scene for BEHAVIOR between O-9 and 10-20
                # if train or test, respectively.
                if testing:
                    self.task_instance_id = rng.integers(10, 20)
                else:
                    self.task_instance_id = rng.integers(0, 10)
                self.set_igibson_behavior_env(
                    task_instance_id=self.task_instance_id, seed=curr_env_seed)
                self.set_options()
            self.igibson_behavior_env.reset()
            self.task_num_task_instance_id_to_igibson_seed[(
                self.task_num, self.task_instance_id)] = curr_env_seed
            os.makedirs(f"tmp_behavior_states/{CFG.behavior_scene_name}__" +
                        f"{CFG.behavior_task_name}__{self.task_num}__" +
                        f"{self.task_instance_id}",
                        exist_ok=True)
            init_state = self.current_ig_state_to_state()
            goal = self._get_task_goal()
            task = Task(init_state, goal)
            tasks.append(task)
            self.task_num += 1

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

        # Second, add in custom predicates.
        custom_predicate_specs = [
            ("reachable-nothing", self._reachable_nothing_classifier, 0),
            ("handempty", self._handempty_classifier, 0),
            ("holding", self._holding_classifier, 1),
            ("reachable", self._reachable_classifier, 1),
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
            type_name = ig_obj.category
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

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise Exception("Cannot make videos for behavior env, change "
                        "behavior_mode in settings.py instead")

    def _get_task_relevant_objects(self) -> List["ArticulatedObject"]:
        return list(self.igibson_behavior_env.task.object_scope.values())

    def set_igibson_behavior_env(self, task_instance_id: int,
                                 seed: int) -> None:
        """Sets/resets the igibson_behavior_env."""
        np.random.seed(seed)
        env_creation_attempts = 0
        # NOTE: this while loop is necessary because in some cases
        # when CFG.randomize_init_state is True, creating a new
        # iGibson env may fail and we need to keep trying until
        # ig_objs_bddl_scope doesn't contain any None's
        while True:
            self.igibson_behavior_env = behavior_env.BehaviorEnv(
                config_file=self._config_file,
                mode=CFG.behavior_mode,
                action_timestep=CFG.behavior_action_timestep,
                physics_timestep=CFG.behavior_physics_timestep,
                action_filter="mobile_manipulation",
                instance_id=task_instance_id,
                rng=self._rng,
            )
            self.igibson_behavior_env.step(
                np.zeros(self.igibson_behavior_env.action_space.shape))
            ig_objs_bddl_scope = [
                self._ig_object_name(obj)
                for obj in self._get_task_relevant_objects()
            ]
            if None not in ig_objs_bddl_scope or env_creation_attempts > 9:
                break
            env_creation_attempts += 1

        if env_creation_attempts > 9:
            raise RuntimeError("ERROR: Failed to sample iGibson BEHAVIOR "
                               "environment that meets bddl initial "
                               "conditions!")
        self.igibson_behavior_env.robots[0].initial_z_offset = 0.7
        self.igibson_behavior_env.use_rrt = CFG.behavior_option_model_rrt

    # Do not add @functools.lru_cache(maxsize=None) here this will
    # lead to wrong mappings when we load a different scene
    def _ig_object_to_object(self, ig_obj: "ArticulatedObject") -> Object:
        type_name = ig_obj.category
        obj_type = self._type_name_to_type[type_name]
        ig_obj_name = self._ig_object_name(ig_obj)
        return Object(ig_obj_name, obj_type)

    # Do not add @functools.lru_cache(maxsize=None) here this will
    # lead to wrong mappings when we load a different scene
    def object_to_ig_object(self, obj: Object) -> "ArticulatedObject":
        """Maintains a mapping of objects to underlying igibson objects."""
        return self._name_to_ig_object(obj.name)

    # Do not add @functools.lru_cache(maxsize=None) here this will
    # lead to wrong mappings when we load a different scene
    def _name_to_ig_object(self, name: str) -> "ArticulatedObject":
        for ig_obj in self._get_task_relevant_objects():
            # Name is extended with sub-type in some behavior tasks
            if self._ig_object_name(ig_obj).startswith(name):
                return ig_obj
        raise ValueError(f"No IG object found for name {name}.")

    @functools.lru_cache(maxsize=None)
    def _name_to_predicate(self, name: str) -> Predicate:
        for pred in self.predicates:
            if name == pred.name:
                return pred
        raise ValueError(f"No predicate found for name {name}.")

    def current_ig_state_to_state(self, save_state: bool = True) -> State:
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

        # NOTE: we set simulator state to none as a 'dummy' value.
        # we should never load a simulator state that was saved when
        # save_state was set to False!
        simulator_state = None
        if save_state:
            simulator_state = save_checkpoint(
                self.igibson_behavior_env.simulator,
                f"tmp_behavior_states/{CFG.behavior_scene_name}__" +
                f"{CFG.behavior_task_name}__{self.task_num}__" +
                f"{self.task_instance_id}/")
        return utils.BehaviorState(
            state_data,
            f"{self.task_num}-{self.task_instance_id}-{simulator_state}")

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
            if not s.allclose(
                    self.current_ig_state_to_state(save_state=False)):
                load_checkpoint_state(s, self)

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
        if not state.allclose(
                self.current_ig_state_to_state(save_state=False)):
            load_checkpoint_state(state, self)
        assert len(objs) == 1
        ig_obj = self.object_to_ig_object(objs[0])
        # We assume we're running BEHAVIOR with only 1 agent
        # in the scene.
        assert len(self.igibson_behavior_env.robots) == 1
        robot_obj = self.igibson_behavior_env.robots[0]
        # If the two objects are the same (i.e reachable(agent, agent)),
        # we always want to return False so that when we learn
        # operators, such predicates don't needlessly appear in preconditions.
        if self._holding_classifier(state=state, objs=[objs[0]]):
            return False
        # We also always want reachable-agent to be False so it doesn't
        # appear in any preconditions.
        if ig_obj.name == "agent":
            return False
        return (np.linalg.norm(  # type: ignore
            np.array(robot_obj.get_position()) -
            np.array(ig_obj.get_position())) < 2)

    def _reachable_nothing_classifier(self, state: State,
                                      objs: Sequence[Object]) -> bool:
        if not state.allclose(
                self.current_ig_state_to_state(save_state=False)):
            load_checkpoint_state(state, self)
        assert len(objs) == 0
        for obj in state:
            if self._reachable_classifier(state=state, objs=[obj]):
                return False
        return True

    def _get_grasped_objects(self, state: State) -> Set[Object]:
        grasped_objs = set()
        for obj in state:
            ig_obj = self.object_to_ig_object(obj)

            # NOTE: The below block is necessary because somehow the body_id
            # is sometimes a 1-element list...
            if isinstance(ig_obj.body_id, list):
                # For some reason sofa is a 4 part body, so we use the
                # first body in the list as the sofa's obj body
                ig_obj.body_id = ig_obj.body_id[0]

            if np.any(self.igibson_behavior_env.robots[0].is_grasping(
                    ig_obj.body_id)):
                grasped_objs.add(obj)

        return grasped_objs

    def _handempty_classifier(self, state: State,
                              objs: Sequence[Object]) -> bool:
        if not state.allclose(
                self.current_ig_state_to_state(save_state=False)):
            load_checkpoint_state(state, self)
        assert len(objs) == 0
        grasped_objs = self._get_grasped_objects(state)
        return len(grasped_objs) == 0

    def _holding_classifier(self, state: State,
                            objs: Sequence[Object]) -> bool:
        if not state.allclose(
                self.current_ig_state_to_state(save_state=False)):
            load_checkpoint_state(state, self)
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
        return "agent"

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
        if len(type_combo) == 0:
            return original_name
        type_names = "-".join(t.name for t in type_combo)
        return f"{original_name}-{type_names}"


def load_checkpoint_state(s: State, env: BehaviorEnv) -> None:
    """Sets the underlying iGibson environment to a particular saved state."""
    assert s.simulator_state is not None
    # Get the new_task_num_task_instance_id associated with this state
    # from s.simulator_state.
    new_task_num_task_instance_id = (int(s.simulator_state.split("-")[0]),
                                     int(s.simulator_state.split("-")[1]))
    # If the new_task_num_task_instance_id is new, then we need to load
    # a new iGibson behavior env with our random seed saved in
    # env.new_task_num_task_instance_id_to_igibson_seed. Otherwise
    # we're already in the correct environment and can just load the
    # checkpoint. Also note that we overwrite the task.init saved checkpoint
    # so that it's compatible with the new environment!
    env.task_num = new_task_num_task_instance_id[0]
    if new_task_num_task_instance_id != (
            env.task_num,
            env.task_instance_id) and CFG.behavior_randomize_init_state:
        env.task_instance_id = new_task_num_task_instance_id[1]
        env.set_igibson_behavior_env(
            task_instance_id=new_task_num_task_instance_id[1],
            seed=env.task_num_task_instance_id_to_igibson_seed[
                new_task_num_task_instance_id])
        env.set_options()
        env.current_ig_state_to_state(
        )  # overwrite the old task_init checkpoint file!
    load_checkpoint(
        env.igibson_behavior_env.simulator,
        f"tmp_behavior_states/{CFG.behavior_scene_name}__" +
        f"{CFG.behavior_task_name}__{env.task_num}__{env.task_instance_id}",
        int(s.simulator_state.split("-")[2]))
    # We step the environment to update the visuals of where the robot is!
    env.igibson_behavior_env.step(
        np.zeros(env.igibson_behavior_env.action_space.shape))


def make_behavior_option(
        name: str, types: Sequence[Type], params_space: Box, env: BehaviorEnv,
        planner_fn: Callable[[
            "behavior_env.BehaviorEnv", Union[
                "URDFObject", "RoomFloor"], Array, Optional[Generator]
        ], Optional[Tuple[List[List[float]], List[List[float]]]]],
        policy_fn: Callable[[List[List[float]], List[List[float]]],
                            Callable[[State, "behavior_env.BehaviorEnv"],
                                     Tuple[Array, bool]]],
        option_model_fn: Callable[
            [List[List[float]], List[List[float]], "URDFObject"],
            Callable[[State, "behavior_env.BehaviorEnv"], None]],
        object_to_ig_object: Callable[[Object], "ArticulatedObject"],
        rng: Generator) -> ParameterizedOption:
    """Makes an option for a BEHAVIOR env using custom implemented
    controller_fn."""

    def policy(state: State, memory: Dict, _objects: Sequence[Object],
               _params: Array) -> Action:
        assert "has_terminated" in memory
        # must call initiable() first, and it must return True
        assert memory.get("policy_controller") is not None
        assert not memory["has_terminated"]
        action_arr, memory["has_terminated"] = memory["policy_controller"](
            state, env.igibson_behavior_env)
        return Action(action_arr)

    def initiable(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        igo = [object_to_ig_object(o) for o in objects]
        assert len(igo) == 1

        # Load the checkpoint associated with state.simulator_state
        # to make sure that we run RRT from the intended state.
        load_checkpoint_state(state, env)

        if memory.get("planner_result") is not None:
            # In this case, an rrt_plan has already been found for this
            # option (most likely, this will occur when executing a
            # series of options after having planned).
            return True

        # NOTE: the below type ignore comment is necessary because mypy
        # doesn't like that rng is being passed by keyword (seems to be
        # an issue with mypy: https://github.com/python/mypy/issues/1655)
        planner_result = planner_fn(env.igibson_behavior_env,
                                    igo[0],
                                    params,
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
                memory["planner_result"][0], memory["planner_result"][1],
                igo[0])
            memory["has_terminated"] = False
        return planner_result is not None

    def terminal(_state: State, memory: Dict, _objects: Sequence[Object],
                 _params: Array) -> bool:
        assert "has_terminated" in memory
        return memory["has_terminated"]

    return ParameterizedOption(
        name,
        types=types,
        params_space=params_space,
        policy=policy,
        initiable=initiable,
        terminal=terminal,
    )
