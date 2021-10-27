"""Behavior (iGibson) environment.

TODO next:
- start integrating options from Nishanth and Willie (still WIP)

Other notes
* OnFloor(robot, floor) does not evaluate to true now,
  even though it's in the initial BDDL state, because
  it uses geometry, and the behaviorbot actually floats
  and doesn't touch the floor. But it doesn't matter.
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
    import pybullet as p
    from igibson.envs import behavior_env
    from igibson.objects.articulated_object import URDFObject
    from igibson.object_states.on_floor import RoomFloor
    from igibson.utils.checkpoint_utils import \
        save_internal_states, load_internal_states
    from igibson.activity.bddl_backend import SUPPORTED_PREDICATES, \
        ObjectStateUnaryPredicate, ObjectStateBinaryPredicate
    from igibson.external.pybullet_tools.utils import CIRCULAR_LIMITS, get_base_difference_fn
    from igibson.utils.behavior_robot_planning_utils import plan_base_motion_br
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

# TODO: remove
np.random.seed(0)


# TODO move this to settings
# Note that we also define custom predicates in the env
# which are not defined in behavior.
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
            physics_timestep=CFG.behavior_physics_timestep,
            action_filter="mobile_manipulation",
        )
        self._env.robots[0].initial_z_offset = 0.7

        self._type_name_to_type = {}

        super().__init__()

    def simulate(self, state: State, action: Action) -> State:
        assert state.simulator_state is not None
        # TODO test that this works as expected
        load_internal_states(self._env.simulator, state.simulator_state)

        a = action.arr

        # # TEMPORARY TESTING
        if not hasattr(self, "_temp_option"):
            obj = sorted(state)[2]
            print("ATTEMPTING TO NAVIGATE TO ", obj)
            options = sorted(self.options, key=lambda o: o.name)
            nav = options[1]
            assert nav.name == 'NavigateTo-book.n.02'
            self._temp_option = nav.ground([obj], np.array([-0.6, 0.6]))
            print("CREATED OPTION:", self._temp_option)
        action = self._temp_option.policy(state)
        a = action.arr
        print("STEPPING ACTION:", a)
        # # END TEMPORARY TESTING

        # # TEMPORARY TESTING
        # if not hasattr(self, "_temp_plan"):
        #     obj = sorted(state)[2]
        #     ig_obj = self._object_to_ig_object(obj)
        #     print("ATTEMPTING TO NAVIGATE TO ", obj, ig_obj)
        #     self._temp_plan, _ = navigate_to_obj_pos(self._env, ig_obj,
        #         np.array([-0.6, 0.6]))
        #     print("FOUND PLAN:", self._temp_plan.shape)
        #     print("PLAN:", self._temp_plan)
        #     import ipdb; ipdb.set_trace()
        #     self._temp_plan = list(self._temp_plan.T)
        # a = self._temp_plan.pop(0)
        # print("STEPPING ACTION:", a)
        # # END TEMPORARY TESTING

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
        # TODO: figure out a more general, less hacky way to do this.
        # Currently assumes that the goal is a single AND of
        # ground atoms (this is also assumed by the planner).
        goal = set()
        assert len(self._env.task.ground_goal_state_options) == 1
        for head_expr in self._env.task.ground_goal_state_options[0]:
            bddl_name = head_expr.terms[0]  # untyped
            ig_objs = [self._name_to_ig_object(t) for t in head_expr.terms[1:]]
            objects = [self._ig_object_to_object(i) for i in ig_objs]
            pred_name = _create_type_combo_name(bddl_name,
                                               [o.type for o in objects])
            pred = self._name_to_predicate(pred_name)
            atom = GroundAtom(pred, objects)
            goal.add(atom)
        return goal

    @property
    def predicates(self) -> Set[Predicate]:
        predicates = set()
        types_lst = sorted(self.types)  # for determinism
        # Extract predicates from behavior/igibson 
        for bddl_name in _BDDL_PREDICATE_NAMES:
            bddl_predicate = SUPPORTED_PREDICATES[bddl_name]
            # We will create one predicate for every combination of types.
            # TODO: filter out implausible type combinations per predicate.
            arity = _bddl_predicate_arity(bddl_predicate)
            for type_combo in itertools.product(types_lst, repeat=arity):
                pred_name = _create_type_combo_name(bddl_name, type_combo)
                _classifier = self._create_classifier_from_bddl(bddl_predicate)
                pred = Predicate(pred_name, list(type_combo), _classifier)
                predicates.add(pred)
        # Add custom predicates
        custom_predicate_specs = [
            ("handempty", self._handempty_classifier, 0),
            ("holding", self._holding_classifier, 1),
            ("nextto-nothing", self._nextto_nothing_classifier, 1),
        ]
        for name, classifier, arity in custom_predicate_specs:
            for type_combo in itertools.product(types_lst, repeat=arity):
                pred_name = _create_type_combo_name(name, type_combo)
                pred = Predicate(pred_name, list(type_combo), classifier)
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
        # name, controller_fn, param_dim, arity
        controllers = [
            ("NavigateTo", navigate_to_obj_pos_controller, 2, 1),
            ("Pick", pick_controller, 0, 1),
            ("PlaceOnTop", place_on_top_controller, 0, 2),
        ]

        options = set()

        for name, controller_fn, param_dim, num_args in controllers:
            # Create a different option for each type combo
            for type_combo in itertools.product(self.types,
                                                repeat=num_args):
                option_name = _create_type_combo_name(name, type_combo)
                controller = _StatefulController(controller_fn, self._env,
                    self._object_to_ig_object)
                option = ParameterizedOption(option_name,
                    types=list(type_combo),
                    params_space=Box(-1, 1, (param_dim,)),
                    _policy=controller.get_action,
                    _initiable=lambda s, o, p: True,
                    _terminal=controller.has_terminated)

                options.add(option)

        return options

    @property
    def action_space(self) -> Box:
        # 17-dimensional, between -1 and 1
        assert self._env.action_space.shape == (17,)
        return self._env.action_space

    def render(self, state: State, task: Task,
               action: Optional[Action] = None) -> Image:
        # TODO
        import ipdb; ipdb.set_trace()

    def _get_task_relevant_objects(self):
        return list(self._env.task.object_scope.values())

    @functools.lru_cache(maxsize=None)
    def _ig_object_to_object(self, ig_obj):
        type_name = _ig_object_to_type_name(ig_obj)
        obj_type = self._type_name_to_type[type_name]
        ig_obj_name = _ig_object_name(ig_obj)
        return Object(ig_obj_name, obj_type)

    @functools.lru_cache(maxsize=None)
    def _object_to_ig_object(self, obj):
        return self._name_to_ig_object(obj.name)

    @functools.lru_cache(maxsize=None)
    def _name_to_ig_object(self, name):
        for ig_obj in self._get_task_relevant_objects():
            if _ig_object_name(ig_obj) == name:
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

    def _get_grasped_objects(self, state):
        grasped_objs = set()
        for obj in state:
            ig_obj = self._object_to_ig_object(obj)
            if any(self._env.robots[0].is_grasping(ig_obj)):
                grasped_objs.add(obj)
        return grasped_objs

    def _handempty_classifier(self, state, objs):
        assert len(objs) == 0
        # See disclaimers in _create_classifier_from_bddl
        grasped_objs = self._get_grasped_objects(state)
        return len(grasped_objs) == 0

    def _holding_classifier(self, state, objs):
        assert len(objs) == 1
        # See disclaimers in _create_classifier_from_bddl
        grasped_objs = self._get_grasped_objects(state)
        return objs[0] in grasped_objs

    def _nextto_nothing_classifier(self, state, objs):
        assert len(objs) == 1
        ig_obj = self._object_to_ig_object(objs[0])
        # See disclaimers in _create_classifier_from_bddl
        bddl_predicate = SUPPORTED_PREDICATES["nextto"]
        for obj in state:
            # TODO maybe refactor out common code
            other_ig_obj = self._object_to_ig_object(obj)
            bddl_ground_atom = bddl_predicate.STATE_CLASS(ig_obj)
            bddl_ground_atom.initialize(self._env.simulator)
            try:
                if bddl_ground_atom.get_value(other_ig_obj):
                    return False
            except KeyError:
                continue
        return True


def _ig_object_name(ig_obj):
    if isinstance(ig_obj, (URDFObject, RoomFloor)):
        return ig_obj.bddl_object_scope
    # Robot is special
    assert "robot" in str(ig_obj)
    return "agent.n.01_1"


def _ig_object_to_type_name(ig_obj):
    ig_obj_name = _ig_object_name(ig_obj)
    if isinstance(ig_obj, RoomFloor):
        assert ":" in ig_obj_name
        type_name = ig_obj_name.split(":")[0]
        return type_name.rsplit("_", 1)[0]
    # Object is either URDFObject or robot
    return ig_obj_name.rsplit("_", 1)[0]


def _bddl_predicate_arity(bddl_predicate):
    # isinstance does not work here, maybe because of the way
    # that these bddl_predicate classes are created?
    if ObjectStateUnaryPredicate in bddl_predicate.__bases__:
        return 1
    if ObjectStateBinaryPredicate in bddl_predicate.__bases__:
        return 2
    raise ValueError("BDDL predicate has unexpected arity.")


def _create_type_combo_name(original_name, type_combo):
    type_names = "-".join(t.name for t in type_combo)
    return f"{original_name}-{type_names}"


### Option definitions ###

class _StatefulController:
    """Temporary glue for options and controllers.
    """
    def __init__(self, controller_fn, env, object_to_ig_object):
        self._controller_fn = controller_fn
        self._env = env
        self._object_to_ig_object = object_to_ig_object
        self._plan = None

    def get_action(self, s, o, p):
        if self._plan is None:
            igo = [self._object_to_ig_object(i) for i in o]
            # TODO: assert s is current state of self._env
            plan, success = self._controller_fn(self._env, igo, p)
            self._plan = list(plan.T)
            if not success:
                raise Exception("Failed to find a plan in controller.")
            print("FOUND PLAN OF LEN:", len(self._plan))
        assert len(self._plan) > 0
        action = self._plan.pop(0)
        return Action(action)

    def has_terminated(self, s, o, p):
        return len(self._plan) == 0


def get_body_ids(env, include_self=False):
    ids = []
    for object in env.scene.get_objects():
        if isinstance(object, URDFObject):
            ids.extend(object.body_ids)

    if include_self:
        ids.append(env.robots[0].parts["left_hand"].get_body_id())
        ids.append(env.robots[0].parts["body"].get_body_id())

    return ids


def detect_collision(bodyA, object_in_hand=None):
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id == bodyA or body_id == object_in_hand:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision


def detect_robot_collision(robot):
    object_in_hand = robot.parts["right_hand"].object_in_hand
    return (
        detect_collision(robot.parts["body"].body_id)
        or detect_collision(robot.parts["left_hand"].body_id)
        or detect_collision(robot.parts["right_hand"].body_id, object_in_hand)
    )


def sample_fn(env):
    random_point = env.scene.get_random_point()
    x, y = random_point[1][:2]
    # TODO: unfortunate that this is not deterministic...
    theta = np.random.uniform(*CIRCULAR_LIMITS)
    return (x, y, theta)


def navigate_to_obj_pos_controller(env, objs, pos_offset):
    """
    Parameterized controller for navigation.
    Runs motion planning to find a feasible trajectory to a certain x,y position offset from obj 
    and selects an orientation such that the robot is facing the object. If the navigation is infeasible,
    returns an indication to this effect.

    :param obj: an object to navigate toward
    :param to_pos: a length-2 numpy array (x, y) containing a position to navigate to

    :return: ret_actions: an np.array of shape (17,n), where, n is the number of low-level 
        action commands the controller has commanded in order to get to pos_offset.
    :return: exec_status: a boolean indicating the execution status.
    """
    assert len(objs) == 1
    obj = objs[0]

    # test agent positions around an obj
    # try to place the agent near the object, and rotate it to the object
    valid_position = None  # ((x,y,z),(roll, pitch, yaw))
    original_position = env.robots[0].get_position()
    original_orientation = env.robots[0].get_orientation()
    base_diff_fn = get_base_difference_fn()

    if isinstance(obj, URDFObject): # must be a URDFObject so we can get its position!
        obj_pos = obj.get_position()
        pos = [pos_offset[0] + obj_pos[0], pos_offset[1] + obj_pos[1], env.robots[0].initial_z_offset] 
        yaw_angle = np.arctan2(pos_offset[1], pos_offset[0])
        orn = [0, 0, yaw_angle]
        env.robots[0].set_position_orientation(pos, p.getQuaternionFromEuler(orn))
        print(pos)
        print(p.getQuaternionFromEuler(orn))
        eye_pos = env.robots[0].parts["eye"].get_position()
        ray_test_res = p.rayTest(eye_pos, obj_pos)
        # Test to see if the robot is obstructed by some object, but make sure that object 
        # is not either the robot's body or the object we want to pick up!
        blocked = len(ray_test_res) > 0 and (ray_test_res[0][0] not in (env.robots[0].parts["body"].get_body_id(), obj.get_body_id()))
        if not detect_robot_collision(env.robots[0]) and not blocked:
            valid_position = (pos, orn)
    else:
        print("ERROR! Object to navigate to is not valid (not an instance of URDFObject).")
        env.robots[0].set_position_orientation(original_position, original_orientation)
        return np.zeros((17,1)), False

    if valid_position is not None:
        env.robots[0].set_position_orientation(original_position, original_orientation)
        plan = plan_base_motion_br(
            robot=env.robots[0],
            end_conf=[valid_position[0][0], valid_position[0][1], valid_position[1][2]],
            base_limits=(),
            obstacles=get_body_ids(env),
            override_sample_fn=lambda : sample_fn(env),
        )

        plan_num_steps = len(plan)
        ret_actions = np.zeros((17, plan_num_steps - 1))
        plan_arr = np.array(plan)

        for i in range(1, plan_num_steps):
            # First compute the delta x,y and rotation in the world frame
            curr_delta_xyrot_W = base_diff_fn(plan_arr[i, :], plan_arr[i-1, :])
            curr_delta_xy_W = curr_delta_xyrot_W[0:2]
            curr_delta_rot_W = curr_delta_xyrot_W[2]
            curr_delta_xyz_W = np.array([curr_delta_xy_W[0], curr_delta_xy_W[1], 0.0])
            curr_delta_quat_W = p.getQuaternionFromEuler(np.array([0.0, 0.0, curr_delta_rot_W]))
            
            # Next, grab the current position and orientation in the world frame
            curr_xyz_W = np.array([plan_arr[i-1,0], plan_arr[i-1,1], env.robots[0].initial_z_offset])
            curr_rot_W = plan_arr[i-1,2]
            curr_quat_W = p.getQuaternionFromEuler(np.array([original_orientation[0], original_orientation[1], curr_rot_W]))

            # Use these to compute the delta pose in the frame of the current pose
            curr_xyz_O, curr_quat_O = p.invertTransform(curr_xyz_W, curr_quat_W) # Invert the current position in the world frame
            curr_xyrot_O = np.array([curr_xyz_O[0], curr_xyz_O[1], p.getEulerFromQuaternion(curr_quat_O)[2]]) # Turn this into x,y,z_rot
            # Multiplying the current pose inverse just computed by the delta pose in the world frame computed above gives 
            # the new pose in the object frame
            new_xyz_O, curr_delta_quat_O = p.multiplyTransforms(curr_xyz_O, curr_quat_O, curr_delta_xyz_W, curr_delta_quat_W)
            new_rot_O = p.getEulerFromQuaternion(curr_delta_quat_O)
            # Get the velocity in the object frame by subtracting this new pose from the old one
            delta_xyzrot_O = base_diff_fn(np.array([new_xyz_O[0], new_xyz_O[1], new_rot_O[2]]), curr_xyrot_O)
            
            # Finally, use this delta pose to compute the action (x,y,z_rot) to be returned 
            ret_actions[0:3, i-1] = delta_xyzrot_O

        env.robots[0].set_position_orientation(original_position, original_orientation)        
        return ret_actions, True
        
    else:
        print("Position commanded is in collision!")
        env.robots[0].set_position_orientation(original_position, original_orientation)
        return np.zeros((17, 1)), False

def pick_controller(env, objs, params):
    # TODO
    return np.reshape(env.action_space.sample(), (17, 1)), True


def place_on_top_controller(env, objs, params):
    # TODO
    return np.reshape(env.action_space.sample(), (17, 1)), True

### End option definitions ###
