"""Definitions of option learning strategies."""

from __future__ import annotations
import abc
from typing import List, Sequence, Dict, Tuple
import numpy as np
from predicators.src.structs import STRIPSOperator, OptionSpec, Datastore, \
    Segment, Array, ParameterizedOption, Object
from predicators.src.settings import CFG
from predicators.src.torch_models import BasicMLP
from predicators.src.envs import create_env, BlocksEnv


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


class _SimpleOptionLearner(_OptionLearnerBase):
    """The option learner that learns options from fixed-length input vectors.
    That is, the the input data only includes objects whose state changes from
    the first state in the segment to the last state in the segment. The input
    data does not include data from other objects in the scene, where the number
    of other objects would differ across different instiantiations of the env.
    """

    def __init__(self) -> None:
        super().__init__()
        # While learning the policy, we record the map from each segment to
        # the option parameterization, so we don't need to recompute it in
        # update_segment_from_option_spec.
        self._segment_to_grounding: Dict[Segment, Tuple[Sequence[Object], Array]] = {}

    def learn_option_specs(self, strips_ops: List[STRIPSOperator],
                           datastores: List[Datastore]) -> List[OptionSpec]:
        # In this paradigm, the option initiable and termination are determined
        # from the operators, so the main thing that needs to be learned is the
        # option policy.
        option_specs = []

        assert len(strips_ops) == len(datastores)

        for op, datastore in zip(strips_ops, datastores):
            print(f"\nLearning option for NSRT {op.name}")

            X_regressor: List[List[Array]] = []
            Y_regressor = []

            # The superset of all objects whose states we may include in the
            # option params is op.parameters. But we only want to include
            # those objects that have state changes (in at least some data).
            changing_parameter_set = self._get_changing_parameters(datastore)
            option_param_dim = sum(v.type.dim for v in changing_parameter_set)
            # Just to avoid confusion, we will insist that the order of the
            # changing parameters is consistent with the order of the original.
            changing_parameters = sorted(changing_parameter_set, key=op.parameters.index)
            del changing_parameter_set  # not used after this
            assert changing_parameters == [v for v in op.parameters if v in changing_parameters]

            for segment, sub in datastore:
                inv_sub = {v: o for o, v in sub.items()}

                # First, determine the continuous option parameterization for
                # this segment.
                changing_objects = [inv_sub[v] for v in changing_parameters]
                option_param = self._get_option_parameterization(segment, changing_objects)
                assert len(option_param) == option_param_dim
                # Store the option parameterization for this segment so we can
                # use it in update_segment_from_option_spec.
                self._segment_to_grounding[segment] = (changing_objects, option_param)
                del changing_objects  # not used after this

                # Next, create the input vectors from all object states named
                # in the operator parameters (not just the changing ones).
                all_objects_in_operator = [inv_sub[v] for v in op.parameters]
                # Note that each segment contributions multiple data points,
                # one per action.
                assert len(segment.states) == len(segment.actions) + 1
                for state, action in zip(segment.states, segment.actions):
                    state_features = state.vec(all_objects_in_operator)
                    # Add a bias term for regression.
                    x = np.hstack(([1.0], state_features, option_param))
                    X_regressor.append(x)
                    Y_regressor.append(action.arr)

            X_arr_regressor = np.array(X_regressor, dtype=np.float32)
            Y_arr_regressor = np.array(Y_regressor, dtype=np.float32)
            regressor = BasicMLP()
            print(f"Fitting regressor with X shape: {X_arr_regressor.shape}, Y shape: {X_arr_regressor.shape}.")
            regressor.fit(X_arr_regressor, Y_arr_regressor)

            # Construct the ParameterizedOption for this operator.
            parameterized_option = self._construct_parameterized_option(name, op, regressor,
                option_param_dim)
            option_specs.append((parameterized_option, operator.parameters))

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

    @staticmethod
    def _get_option_parameterization(segment: Segment, changing_objects: Sequence[Object]) -> Array:
        # Parameterization is RELATIVE change in object states.
        start = segment.states[0]
        end = segment.states[-1]
        return end.vec(changing_objects) - start.vec(changing_objects)

    @staticmethod
    def _construct_parameterized_option(name: str, op: STRIPSOperator, regressor: BasicMLP,
        option_param_dim: int) -> ParameterizedOption:
        import ipdb; ipdb.set_trace()

    def update_segment_from_option_spec(
            self, segment: Segment, option_spec: OptionSpec) -> None:
        objects, params = self._segment_to_grounding[segment]
        param_opt, opt_vars = option_spec
        option = param_opt.ground(objects, params)
        segment.set_option(option)
