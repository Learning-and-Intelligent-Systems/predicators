"""Definitions of option learning strategies."""

from __future__ import annotations
import abc
from dataclasses import dataclass
from functools import cached_property
from typing import List, Sequence, Dict, Tuple, Set
import numpy as np
from predicators.src.structs import STRIPSOperator, OptionSpec, Datastore, \
    Segment, Array, ParameterizedOption, Object, Variable, Box, Predicate, \
    State, Action
from predicators.src.settings import CFG
from predicators.src.torch_models import MLPRegressor
from predicators.src.envs import create_env, BlocksEnv
from predicators.src import utils


def create_option_learner() -> _OptionLearnerBase:
    """Create an option learner given its name."""
    if CFG.option_learner == "no_learning":
        return _KnownOptionsOptionLearner()
    if CFG.option_learner == "oracle":
        return _OracleOptionLearner()
    if CFG.option_learner == "simple":
        return _SimpleOptionLearner()
    raise NotImplementedError(f"Unknown option_learner: {CFG.option_learner}")


class _OptionLearnerBase:
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
            option_vars = None
            for i, (segment, sub) in enumerate(datastore):
                option = segment.actions[0].get_option()
                if i == 0:
                    param_option = option.parent
                    option_vars = [sub[o] for o in option.objects]
                else:
                    assert param_option == option.parent
                    assert option_vars == [sub[o] for o in option.objects]
                # Make sure the option is consistent within a trajectory.
                for a in segment.actions:
                    option_a = a.get_option()
                    assert param_option == option_a.parent
                    assert option_vars == [sub[o] for o in option_a.objects]
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
        env = create_env(CFG.env)
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
                params = np.zeros(3, dtype=np.float32)
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
                params = np.array([0, 0, BlocksEnv.block_size],
                                  dtype=np.float32)
                option = param_opt.ground([robby, block], params)
                segment.set_option(option)


@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSimpleParameterizedOption(ParameterizedOption):
    """A convenience class for holding a learned parameterized option.

    Prefer to use this because it is pickleable.
    """
    _operator: STRIPSOperator
    _regressor: MLPRegressor
    _changing_parameter_idxs: Sequence[int]

    def __init__(self, name: str, operator: STRIPSOperator,
                 regressor: MLPRegressor,
                 changing_parameters: Sequence[Variable]) -> None:
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
        # Dataclass is frozen, so we have to do these hacks.
        object.__setattr__(self, "_operator", operator)
        object.__setattr__(self, "_regressor", regressor)
        object.__setattr__(self, "_changing_parameter_idxs",
                           changing_parameter_idxs)
        super().__init__(name,
                         types,
                         params_space,
                         _policy=self._regressor_based_policy,
                         _initiable=self._precondition_based_initiable,
                         _terminal=self._effect_based_terminal)

    @cached_property
    def _predicates(self) -> Set[Predicate]:
        """Helper for initiable and terminal."""
        return {a.predicate for a in self._operator.preconditions | \
                self._operator.add_effects | self._operator.delete_effects}

    def _precondition_based_initiable(self, state: State, memory: Dict,
                                      objects: Sequence[Object],
                                      params: Array) -> bool:
        # The memory here is used to store the absolute params, based on
        # the relative params and the object states.
        memory["params"] = params  # store for sanity checking in policy
        changing_objects = [objects[i] for i in self._changing_parameter_idxs]
        # TODO remove before merging. This is just because the current oracle
        # is parameterized absolutely.
        if CFG.sampler_learner == "oracle":
            memory["absolute_goal_vec"] = params
        else:
            memory["absolute_goal_vec"] = state.vec(changing_objects) + params
        # Check if initiable based on preconditions.
        grounded_op = self._operator.ground(tuple(objects))
        preconditions = grounded_op.preconditions
        abs_state = utils.abstract(state, self._predicates)
        return preconditions.issubset(abs_state)

    def _regressor_based_policy(self, state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
        # Compute the updated relative goal.
        assert np.allclose(params, memory["params"])
        changing_objects = [objects[i] for i in self._changing_parameter_idxs]
        relative_goal_vec = memory["absolute_goal_vec"] - state.vec(
            changing_objects)
        x = np.hstack(([1.0], state.vec(objects), relative_goal_vec))
        action_arr = self._regressor.predict(x)
        assert not np.isnan(action_arr).any()
        return Action(np.array(action_arr, dtype=np.float32))

    def _effect_based_terminal(self, state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> bool:
        del memory, params  # unused
        # The hope is that we terminate in the effects.
        grounded_op = self._operator.ground(tuple(objects))
        abs_state = utils.abstract(state, self._predicates)
        if grounded_op.add_effects.issubset(abs_state) and \
            not grounded_op.delete_effects & abs_state:
            return True
        # If we fall outside of the space where the preconditions hold, also
        # terminate. The assumption is that the option should take us from one
        # abstract state to another without hitting any others in between.
        if not grounded_op.preconditions.issubset(abs_state):
            return True
        # Not yet done.
        return False


class _SimpleOptionLearner(_OptionLearnerBase):
    """The option learner that learns options from fixed-length input vectors.

    That is, the the input data only includes objects whose state
    changes from the first state in the segment to the last state in the
    segment. The input data does not include data from other objects in
    the scene, where the number of other objects would differ across
    different instiantiations of the env.
    """

    def __init__(self) -> None:
        super().__init__()
        # While learning the policy, we record the map from each segment to
        # the option parameterization, so we don't need to recompute it in
        # update_segment_from_option_spec.
        self._segment_to_grounding: Dict[Segment, Tuple[Sequence[Object],
                                                        Array]] = {}

    def learn_option_specs(self, strips_ops: List[STRIPSOperator],
                           datastores: List[Datastore]) -> List[OptionSpec]:
        # In this paradigm, the option initiable and termination are determined
        # from the operators, so the main thing that needs to be learned is the
        # option policy.
        option_specs: List[Tuple[ParameterizedOption, List[Variable]]] = []

        assert len(strips_ops) == len(datastores)

        for op, datastore in zip(strips_ops, datastores):
            print(f"\nLearning option for NSRT {op.name}")

            X_regressor: List[Array] = []
            Y_regressor = []

            # The superset of all objects whose states we may include in the
            # option params is op.parameters. But we only want to include
            # those objects that have state changes (in at least some data).
            changing_parameter_set = self._get_changing_parameters(datastore)
            # Just to avoid confusion, we will insist that the order of the
            # changing parameters is consistent with the order of the original.
            changing_parameters = sorted(changing_parameter_set,
                                         key=op.parameters.index)
            del changing_parameter_set  # not used after this
            assert changing_parameters == [
                v for v in op.parameters if v in changing_parameters
            ]

            for segment, sub in datastore:
                inv_sub = {v: o for o, v in sub.items()}
                all_objects_in_operator = [inv_sub[v] for v in op.parameters]

                # First, determine the absolute goal vector for this segment.
                changing_objects = [inv_sub[v] for v in changing_parameters]
                initial_state = segment.states[0]
                final_state = segment.states[-1]
                absolute_goal_vec = final_state.vec(changing_objects)
                option_param = absolute_goal_vec - initial_state.vec(
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
                    # Compute the relative goal vector this segment.
                    relative_goal_vec = absolute_goal_vec - state.vec(
                        changing_objects)
                    # Add a bias term for regression.
                    x = np.hstack(([1.0], state_features, relative_goal_vec))
                    X_regressor.append(x)
                    Y_regressor.append(action.arr)

            X_arr_regressor = np.array(X_regressor, dtype=np.float32)
            Y_arr_regressor = np.array(Y_regressor, dtype=np.float32)
            regressor = MLPRegressor()
            print(f"Fitting regressor with X shape: {X_arr_regressor.shape}, "
                  f"Y shape: {Y_arr_regressor.shape}.")
            regressor.fit(X_arr_regressor, Y_arr_regressor)

            # Construct the ParameterizedOption for this operator.
            name = f"{op.name}LearnedOption"
            parameterized_option = _LearnedSimpleParameterizedOption(
                name, op, regressor, changing_parameters)
            option_specs.append((parameterized_option, list(op.parameters)))

        return option_specs

    @staticmethod
    def _get_changing_parameters(datastore: Datastore) -> Set[Variable]:
        all_changing_variables = set()
        for segment, sub in datastore:
            start = segment.states[0]
            end = segment.states[-1]
            for o, v in sub.items():
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
