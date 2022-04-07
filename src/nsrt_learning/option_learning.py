"""Definitions of option learning strategies."""

from __future__ import annotations

import abc
import logging
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from predicators.src.envs import get_or_create_env
from predicators.src.envs.blocks import BlocksEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, Box, Datastore, Object, \
    OptionSpec, ParameterizedOption, Segment, State, STRIPSOperator, \
    Variable
from predicators.src.torch_models import ImplicitMLPRegressor, MLPRegressor, \
    Regressor
from predicators.src.utils import OptionExecutionFailure


def create_option_learner(action_space: Box) -> _OptionLearnerBase:
    """Create an option learner given its name."""
    if CFG.option_learner == "no_learning":
        return _KnownOptionsOptionLearner()
    if CFG.option_learner == "oracle":
        return _OracleOptionLearner()
    if CFG.option_learner == "direct_bc":
        return _DirectBehaviorCloningOptionLearner(action_space)
    if CFG.option_learner == "implicit_bc":
        return _ImplicitBehaviorCloningOptionLearner(action_space)
    raise NotImplementedError(f"Unknown option_learner: {CFG.option_learner}")


class _OptionLearnerBase(abc.ABC):
    """Struct defining an option learner, which has an abstract method for
    learning option specs and an abstract method for annotating data segments
    with options."""

    @abc.abstractmethod
    def learn_option_specs(self, strips_ops: List[STRIPSOperator],
                           datastores: List[Datastore]) -> List[OptionSpec]:
        """Given datastores and STRIPS operators that were fit on them, learn
        option specs, which are tuples of (ParameterizedOption,
        Sequence[Variable]).

        The returned option specs should be one-to-one with the given
        strips_ops / datastores (which are already one-to-one with each
        other).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def update_segment_from_option_spec(self, segment: Segment,
                                        option_spec: OptionSpec) -> None:
        """Figure out which option was executed within the given segment.
        Modify the segment in-place to include this option, via
        segment.set_option().

        At this point, we know which ParameterizedOption was used, and
        we know the option_vars. This information is included in the
        given option_spec. But we don't know what parameters were used
        in the option, which this method should figure out.
        """
        raise NotImplementedError("Override me!")


class _KnownOptionsOptionLearner(_OptionLearnerBase):
    """The "option learner" that's used when we're in the code path where
    CFG.option_learner is "no_learning"."""

    def learn_option_specs(self, strips_ops: List[STRIPSOperator],
                           datastores: List[Datastore]) -> List[OptionSpec]:
        # Since we're not actually doing option learning, the data already
        # contains the options. So, we just extract option specs from the data.
        option_specs = []
        for datastore in datastores:
            param_option = None
            option_vars = []
            for i, (segment, var_to_obj) in enumerate(datastore):
                option = segment.actions[0].get_option()
                if i == 0:
                    obj_to_var = {o: v for v, o in var_to_obj.items()}
                    param_option = option.parent
                    option_vars = [obj_to_var[o] for o in option.objects]
                else:
                    assert param_option == option.parent
                    option_args = [var_to_obj[v] for v in option_vars]
                    assert option_args == option.objects
                # Make sure the option is consistent within a trajectory.
                for a in segment.actions:
                    option_a = a.get_option()
                    assert param_option == option_a.parent
                    option_args = [var_to_obj[v] for v in option_vars]
                    assert option_args == option_a.objects
            assert param_option is not None and option_vars is not None, \
                "No data in this datastore?"
            option_specs.append((param_option, option_vars))
        return option_specs

    def update_segment_from_option_spec(self, segment: Segment,
                                        option_spec: OptionSpec) -> None:
        # If we're not doing option learning, the segments will already have
        # the options, so there is nothing to do here.
        pass


class _OracleOptionLearner(_OptionLearnerBase):
    """The option learner that just cheats by looking up ground truth options
    from the environment.

    Useful for testing.
    """

    def learn_option_specs(self, strips_ops: List[STRIPSOperator],
                           datastores: List[Datastore]) -> List[OptionSpec]:
        env = get_or_create_env(CFG.env)
        option_specs: List[OptionSpec] = []
        if CFG.env == "cover":
            assert len(strips_ops) == 3
            PickPlace = [
                option for option in env.options if option.name == "PickPlace"
            ][0]
            # All strips operators use the same PickPlace option,
            # which has no parameters.
            for _ in strips_ops:
                option_specs.append((PickPlace, []))
        elif CFG.env == "blocks":
            assert len(strips_ops) == 4
            Pick = [option for option in env.options
                    if option.name == "Pick"][0]
            Stack = [
                option for option in env.options if option.name == "Stack"
            ][0]
            PutOnTable = [
                option for option in env.options if option.name == "PutOnTable"
            ][0]
            for op in strips_ops:
                if {atom.predicate.name for atom in op.preconditions} in \
                   ({"GripperOpen", "Clear", "OnTable"},
                    {"GripperOpen", "Clear", "On"}):
                    # PickFromTable or Unstack operators
                    gripper_open_atom = [
                        atom for atom in op.preconditions
                        if atom.predicate.name == "GripperOpen"
                    ][0]
                    robot = gripper_open_atom.variables[0]
                    clear_atom = [
                        atom for atom in op.preconditions
                        if atom.predicate.name == "Clear"
                    ][0]
                    block = clear_atom.variables[0]
                    option_specs.append((Pick, [robot, block]))
                elif {atom.predicate.name for atom in op.preconditions} == \
                     {"Clear", "Holding"}:
                    # Stack operator
                    gripper_open_atom = [
                        atom for atom in op.add_effects
                        if atom.predicate.name == "GripperOpen"
                    ][0]
                    robot = gripper_open_atom.variables[0]
                    clear_atom = [
                        atom for atom in op.preconditions
                        if atom.predicate.name == "Clear"
                    ][0]
                    otherblock = clear_atom.variables[0]
                    option_specs.append((Stack, [robot, otherblock]))
                elif {atom.predicate.name for atom in op.preconditions} == \
                     {"Holding"}:
                    # PutOnTable operator
                    gripper_open_atom = [
                        atom for atom in op.add_effects
                        if atom.predicate.name == "GripperOpen"
                    ][0]
                    robot = gripper_open_atom.variables[0]
                    option_specs.append((PutOnTable, [robot]))
        return option_specs

    def update_segment_from_option_spec(self, segment: Segment,
                                        option_spec: OptionSpec) -> None:
        if CFG.env == "cover":
            param_opt, opt_vars = option_spec
            assert not opt_vars
            assert len(segment.actions) == 1
            # In the cover env, the action is itself the option parameter.
            params = segment.actions[0].arr
            option = param_opt.ground([], params)
            segment.set_option(option)
        if CFG.env == "blocks":
            param_opt, opt_vars = option_spec
            assert len(segment.actions) == 1
            act = segment.actions[0].arr
            robby = [obj for obj in segment.states[1]
                     if obj.name == "robby"][0]
            # Transform action array back into parameters.
            if param_opt.name == "Pick":
                assert len(opt_vars) == 2
                picked_blocks = [
                    obj for obj in segment.states[1]
                    if obj.type.name == "block"
                    and segment.states[0].get(obj, "held") < 0.5 <
                    segment.states[1].get(obj, "held")
                ]
                assert len(picked_blocks) == 1
                block = picked_blocks[0]
                params = np.zeros(0, dtype=np.float32)
                option = param_opt.ground([robby, block], params)
                segment.set_option(option)
            elif param_opt.name == "PutOnTable":
                x, y, _, _ = act
                params = np.array([
                    (x - BlocksEnv.x_lb) / (BlocksEnv.x_ub - BlocksEnv.x_lb),
                    (y - BlocksEnv.y_lb) / (BlocksEnv.y_ub - BlocksEnv.y_lb)
                ])
                option = param_opt.ground([robby], params)
                segment.set_option(option)
            elif param_opt.name == "Stack":
                assert len(opt_vars) == 2
                dropped_blocks = [
                    obj for obj in segment.states[1]
                    if obj.type.name == "block"
                    and segment.states[1].get(obj, "held") < 0.5 <
                    segment.states[0].get(obj, "held")
                ]
                assert len(dropped_blocks) == 1
                block = dropped_blocks[0]
                params = np.zeros(0, dtype=np.float32)
                option = param_opt.ground([robby, block], params)
                segment.set_option(option)


class _LearnedNeuralParameterizedOption(ParameterizedOption):
    """A parameterized option that "implements" an operator.

    * The option objects correspond to the operator parameters.
    * The option initiable corresponds to the operator preconditions.
    * The option terminal corresponds to the operator effects.
    * The option policy is implemented as a neural regressor that takes in the
      states of the option objects and the input continuous parameters (see
      below), all concatenated into a 1D vector, and outputs an action.

    The continuous parameters of the option are the main thing to note: they
    correspond to a desired change in state for a subset of the option objects.
    The objects that are changing correspond to changing_parameters; all other
    objects are assumed to have no change in their state.

    Note that the option terminal, which corresponds to the operator effects,
    already describes how we want the objects to change. But these effects only
    characterize a *set* of desired state changes. For example, a Pick operator
    with Holding(?x) in the add effects says that we want to end up in *some*
    state where we are holding ?x, but there are in general an infinite number
    of such states. The role of the continuous parameters is to identify one
    specific state in this set.

    The hope is that by sampling different continuous parameters, the option
    will be able to navigate to different states in the effect set, giving the
    diversity of transition samples that we need to do bilevel planning.

    Also note that the parameters correspond to a state *change*, rather than
    an absolute state. This distinction is useful for learning; relative changes
    lead to better data efficiency / generalization than absolute ones. However,
    it's also important to note that the change here is a delta between the
    initial state that the option is executed from and the final state where the
    option is terminated. In the middle of executing the option, the desired
    delta between the current state and the final state is different from what
    it was in the initial state. To handle this, we save the *absolute* desired
    state in the memory of the option during initialization, and compute the
    updated delta on each call to the policy.
    """

    def __init__(self, name: str, operator: STRIPSOperator,
                 regressor: Regressor, changing_parameters: Sequence[Variable],
                 action_space: Box) -> None:
        assert set(changing_parameters).issubset(set(operator.parameters))
        changing_parameter_idxs = [
            i for i, v in enumerate(operator.parameters)
            if v in changing_parameters
        ]
        types = [v.type for v in operator.parameters]
        option_param_dim = sum(v.type.dim for v in changing_parameters)
        params_space = Box(low=-np.inf,
                           high=np.inf,
                           shape=(option_param_dim, ),
                           dtype=np.float32)
        self._operator = operator
        self._regressor = regressor
        self._changing_parameter_idxs = changing_parameter_idxs
        self._action_space = action_space
        super().__init__(name,
                         types,
                         params_space,
                         policy=self._regressor_based_policy,
                         initiable=self._precondition_based_initiable,
                         terminal=self._effect_based_terminal)

    def _precondition_based_initiable(self, state: State, memory: Dict,
                                      objects: Sequence[Object],
                                      params: Array) -> bool:
        # The memory here is used to store the absolute params, based on
        # the relative params and the object states.
        memory["params"] = params  # store for sanity checking in policy
        changing_objects = [objects[i] for i in self._changing_parameter_idxs]
        memory["absolute_params"] = state.vec(changing_objects) + params
        # Check if initiable based on preconditions.
        grounded_op = self._operator.ground(tuple(objects))
        return all(pre.holds(state) for pre in grounded_op.preconditions)

    def _regressor_based_policy(self, state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
        # Compute the updated relative goal.
        assert np.allclose(params, memory["params"])
        changing_objects = [objects[i] for i in self._changing_parameter_idxs]
        relative_goal_vec = memory["absolute_params"] - state.vec(
            changing_objects)
        x = np.hstack(([1.0], state.vec(objects), relative_goal_vec))
        action_arr = self._regressor.predict(x)
        if np.isnan(action_arr).any():
            raise OptionExecutionFailure("Option policy returned nan.")
        action_arr = np.clip(action_arr, self._action_space.low,
                             self._action_space.high)
        return Action(np.array(action_arr, dtype=np.float32))

    def _effect_based_terminal(self, state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> bool:
        assert np.allclose(params, memory["params"])
        # The hope is that we terminate in the effects.
        grounded_op = self._operator.ground(tuple(objects))
        if all(e.holds(state) for e in grounded_op.add_effects) and \
            not any(e.holds(state) for e in grounded_op.delete_effects):
            return True
        # Optimization: remember the most recent state and terminate early if
        # the state is repeated, since this option will never get unstuck.
        if "last_state" in memory and memory["last_state"].allclose(state):
            return True
        memory["last_state"] = state
        # Not yet done.
        return False


class _BehaviorCloningOptionLearner(_OptionLearnerBase):
    """Learn _LearnedNeuralParameterizedOption objects by behavior cloning.

    See the docstring for _LearnedNeuralParameterizedOption for a description
    of the option structure.

    In this paradigm, the option initiable and termination are determined from
    the operators, so the main thing that needs to be learned is the option
    policy. We learn this policy by behavior cloning (fitting a regressor
    via supervised learning) in learn_option_specs().
    """

    def __init__(self, action_space: Box) -> None:
        super().__init__()
        # Actions are clipped to stay within the action space.
        self._action_space = action_space
        # While learning the policy, we record the map from each segment to
        # the option parameterization, so we don't need to recompute it in
        # update_segment_from_option_spec.
        self._segment_to_grounding: Dict[Segment, Tuple[Sequence[Object],
                                                        Array]] = {}

    @abc.abstractmethod
    def _create_regressor(self) -> Regressor:
        raise NotImplementedError("Override me!")

    def learn_option_specs(self, strips_ops: List[STRIPSOperator],
                           datastores: List[Datastore]) -> List[OptionSpec]:
        option_specs: List[Tuple[ParameterizedOption, List[Variable]]] = []

        assert len(strips_ops) == len(datastores)

        for op, datastore in zip(strips_ops, datastores):
            logging.info(f"\nLearning option for NSRT {op.name}")

            X_regressor: List[Array] = []
            Y_regressor = []

            # The superset of all objects whose states we may include in the
            # option params is op.parameters. But we only want to include
            # those objects that have state changes (in at least some data).
            # We do this for learning efficiency; including all objects would
            # likely work too, but may require more data for the model to
            # realize that those objects' parameters can be ignored.
            changing_parameter_set = self._get_changing_parameters(datastore)
            # Just to avoid confusion, we will insist that the order of the
            # changing parameters is consistent with the order of the original.
            changing_parameters = sorted(changing_parameter_set,
                                         key=op.parameters.index)
            del changing_parameter_set  # not used after this
            assert changing_parameters == [
                v for v in op.parameters if v in changing_parameters
            ]

            for segment, var_to_obj in datastore:
                all_objects_in_operator = [
                    var_to_obj[v] for v in op.parameters
                ]

                # First, determine the absolute goal vector for this segment.
                changing_objects = [var_to_obj[v] for v in changing_parameters]
                initial_state = segment.states[0]
                final_state = segment.states[-1]
                absolute_params = final_state.vec(changing_objects)
                option_param = absolute_params - initial_state.vec(
                    changing_objects)

                # Store the option parameterization for this segment so we can
                # use it in update_segment_from_option_spec.
                self._segment_to_grounding[segment] = (all_objects_in_operator,
                                                       option_param)
                del option_param  # not used after this

                # Next, create the input vectors from all object states named
                # in the operator parameters (not just the changing ones).
                # Note that each segment contributions multiple data points,
                # one per action.
                assert len(segment.states) == len(segment.actions) + 1
                for state, action in zip(segment.states, segment.actions):
                    state_features = state.vec(all_objects_in_operator)
                    # Compute the relative goal vector for this segment.
                    relative_goal_vec = absolute_params - state.vec(
                        changing_objects)
                    # Add a bias term for regression.
                    x = np.hstack(([1.0], state_features, relative_goal_vec))
                    X_regressor.append(x)
                    Y_regressor.append(action.arr)

            X_arr_regressor = np.array(X_regressor, dtype=np.float32)
            Y_arr_regressor = np.array(Y_regressor, dtype=np.float32)
            regressor = self._create_regressor()
            logging.info("Fitting regressor with X shape: "
                         f"{X_arr_regressor.shape}, Y shape: "
                         f"{Y_arr_regressor.shape}.")
            regressor.fit(X_arr_regressor, Y_arr_regressor)

            # Construct the ParameterizedOption for this operator.
            name = f"{op.name}LearnedOption"
            parameterized_option = _LearnedNeuralParameterizedOption(
                name, op, regressor, changing_parameters, self._action_space)
            option_specs.append((parameterized_option, list(op.parameters)))

        return option_specs

    @staticmethod
    def _get_changing_parameters(datastore: Datastore) -> Set[Variable]:
        all_changing_variables = set()
        for segment, var_to_obj in datastore:
            start = segment.states[0]
            end = segment.states[-1]
            for v, o in var_to_obj.items():
                if not np.array_equal(start[o], end[o]):
                    all_changing_variables.add(v)
        return all_changing_variables

    def update_segment_from_option_spec(self, segment: Segment,
                                        option_spec: OptionSpec) -> None:
        objects, params = self._segment_to_grounding[segment]
        param_opt, opt_vars = option_spec
        assert all(o.type == v.type for o, v in zip(objects, opt_vars))
        option = param_opt.ground(objects, params)
        segment.set_option(option)


class _DirectBehaviorCloningOptionLearner(_BehaviorCloningOptionLearner):
    """Use an MLPRegressor for regression."""

    def _create_regressor(self) -> Regressor:
        return MLPRegressor(seed=CFG.seed,
                            hid_sizes=CFG.mlp_regressor_hid_sizes,
                            max_train_iters=CFG.mlp_regressor_max_itr,
                            clip_gradients=CFG.mlp_regressor_clip_gradients,
                            clip_value=CFG.mlp_regressor_gradient_clip_value,
                            learning_rate=CFG.learning_rate)


class _ImplicitBehaviorCloningOptionLearner(_BehaviorCloningOptionLearner):
    """Use an ImplicitMLPRegressor for regression."""

    def _create_regressor(self) -> Regressor:
        num_neg = CFG.implicit_mlp_regressor_num_negative_data_per_input
        num_sam = CFG.implicit_mlp_regressor_num_samples_per_inference
        return ImplicitMLPRegressor(
            seed=CFG.seed,
            hid_sizes=CFG.mlp_regressor_hid_sizes,
            max_train_iters=CFG.implicit_mlp_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate,
            num_negative_data_per_input=num_neg,
            num_samples_per_inference=num_sam,
            temperature=CFG.implicit_mlp_regressor_temperature)
