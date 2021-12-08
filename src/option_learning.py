"""Definitions of option learning strategies.
"""

from __future__ import annotations
import abc
from typing import List
import numpy as np
from predicators.src.structs import STRIPSOperator, OptionSpec, Datastore, \
    Segment, Partition, Variable, ParameterizedOption, Action
from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.envs import create_env, BlocksEnv
from predicators.src.torch_models import NeuralGaussianRegressor, BasicMLP
from gym.spaces import Box


def create_option_learner() -> _OptionLearnerBase:
    """Create an option learner given its name.
    """
    if CFG.option_learner == "no_learning":
        return _KnownOptionsOptionLearner()
    if CFG.option_learner == "oracle":
        return _OracleOptionLearner()
    if CFG.option_learner == "simple":
        return _SimpleOptionLearner()
    raise NotImplementedError(f"Unknown option learner: {CFG.option_learner}")


class _OptionLearnerBase(abc.ABC):
    """Struct defining an option learner, which has an abstract method for
    learning option specs and an abstract method for annotating data segments
    with options.
    """
    @abc.abstractmethod
    def learn_option_specs(
            self, strips_ops: List[STRIPSOperator],
            datastores: List[Datastore]) -> List[OptionSpec]:
        """Given datastores and STRIPS operators that were fit on them,
        learn option specs, which are tuples of (ParameterizedOption,
        Sequence[Variable]). The returned option specs should be one-to-one
        with the given strips_ops / datastores (which are already one-to-one
        with each other).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def update_segment_from_option_spec(
            self, segment: Segment, option_spec: OptionSpec) -> None:
        """Figure out which option was executed within the given segment.
        Modify the segment in-place to include this option, via
        segment.set_option().

        At this point, we know which ParameterizedOption was used, and we know
        the option_vars. This information is included in the given option_spec.
        But we don't know what parameters were used in the option,
        which this method should figure out.
        """
        raise NotImplementedError("Override me!")


class _SimpleOptionLearner(_OptionLearnerBase):
    """The option learner that learns options from fixed-length input vectors.
    That is, the the input data only includes objects whose state changes from
    the first state in the segment to the last state in the segment. The input
    data does not include data from other objects in the scene, where the number
    of other objects would differ across different instiantiations of the env.
    """

    def __init__(self) -> None:
        super().__init__()
        self.segment_to_grounding = {}

    def learn_option_specs(
            self, strips_ops: List[STRIPSOperator],
            partitions: List[Partition]) -> List[OptionSpec]:

        option_specs = []

        for idx, p in enumerate(partitions):
            X_regressor: List[List[Array]] = []
            Y_regressor = []
            op = strips_ops[idx]
            print(f"\nLearning option for NSRT {op.name}")
            param_ub, param_lb = [], []
            types = []
            variables = []
            # temp = [len(s.actions) for (s, _) in p]
            # print("TEMP: ", temp)
            # print("NUMBER OF data points: ", sum(a for a in temp))
            # Get objects involved in effects and sort them.
            for segment, _ in p:
                objects = sorted({o for atom in segment.add_effects | \
                                 segment.delete_effects for o in atom.objects})
                types = [o.type for o in objects]
                # print("TYPES: ", types)

                param = []
                all_objects = list(segment.states[0])
                objects_that_changed = []
                for o in all_objects:
                    start = segment.states[0]
                    end = segment.states[-1]
                    if not np.array_equal(start[o], end[o]):
                        objects_that_changed.append(o)
                # Keep track of max and min for each dimension of parameters
                # that come from each object.
                if len(param_ub) == 0:
                    param_ub = [[-np.inf]*len(segment.states[0][o]) \
                                for o in objects_that_changed]
                    param_lb = [[np.inf]*len(segment.states[0][o]) \
                                for o in objects_that_changed]

                for j, o in enumerate(objects_that_changed):
                    object_param = segment.states[0][o] - segment.states[-1][o]
                    for k in range(len(object_param)):
                        if object_param[k] > param_ub[j][k]:
                            param_ub[j][k] = object_param[k]
                        if object_param[k] < param_lb[j][k]:
                            param_lb[j][k] = object_param[k]
                    param.extend(object_param)
                self.segment_to_grounding[segment] = (objects, param)

                variables = [Variable(f"?x{i}", o.type) \
                             for i, o in enumerate(objects)]

                # Construct input. Input is a bias, state features of objects
                # involved in effects, and the delta of each low-level feature
                # of each object involved in effects between the first and last
                # state of the segment.
                for i in range(len(segment.states) - 1):
                    s, s_prime = segment.states[i], segment.states[i+1]
                    X_regressor.append([np.array(1.0)])
                    for o in objects:
                        X_regressor[-1].extend(s[o])
                    X_regressor[-1].extend(param)
                    Y_regressor.append(segment.actions[i].arr)

            X_arr_regressor = np.array(X_regressor)
            Y_arr_regressor = np.array(Y_regressor)
            print("X SHAPE: ", X_arr_regressor.shape)
            print("Y SHAPE: ", Y_arr_regressor.shape)
            regressor = BasicMLP()
            regressor.fit(X_arr_regressor, Y_arr_regressor)

            # Construct the ParameterizedOption for this partition.
            name = str(idx)
            high = np.array([dim for o in param_ub for dim in o])
            low = np.array([dim for o in param_lb for dim in o])
            # high = np.array([np.inf for o in param_ub for dim in o])
            # low = np.array([-np.inf for o in param_lb for dim in o])
            params_space = Box(low, high, dtype=np.float32)
            _initiable = self._create_initiable_from_op(op)
            # _initiable = (lambda op:
            #                 lambda s, m, o, p:
            #                     (lambda preconditions: preconditions.issubset(
            #                         utils.abstract(
            #                         s, {a.predicate for a in preconditions})))
            #                     (op.ground(tuple(o_)).preconditions))
            #              (op)
            _terminal = self._create_terminal_from_op(op)
            _policy = self._create_policy_from_regressor(regressor)
            option = ParameterizedOption(name=name, types=types, \
                                         params_space=params_space, \
                                         _policy=_policy, \
                                         _initiable=_initiable, \
                                         _terminal=_terminal)
            option_specs.append((option, variables))

        return option_specs

    def update_segment_from_option_spec(
            self, segment: Segment, option_spec: OptionSpec) -> None:
            objects, params = self.segment_to_grounding[segment]
            param_opt, opt_vars = option_spec
            option = param_opt.ground(objects, params)
            segment.set_option(option)

    @staticmethod
    def _create_initiable_from_op(op
        ) -> Callable[[State, Dict, Sequence[Object], Array], bool]:
        preds = {a.predicate for a in op.preconditions}
        def _initiable(s_: State, m: Dict, o_: Sequence[Object],
                       p_: Array) -> bool:
            del m, p_  # unused
            grounded_op = op.ground(tuple(o_))
            preconditions = grounded_op.preconditions
            abs_state = utils.abstract(s_, preds)
            return preconditions.issubset(abs_state)
        return _initiable

    @staticmethod
    def _create_terminal_from_op(op: STRIPSOperator
        ) -> Callable[[State, Dict, Sequence[Object], Array], bool]:
        preds = {a.predicate for a in op.add_effects | op.delete_effects}
        def _terminal(s_: State, m: Dict, o_: Sequence[Object],
                      p_: Array) -> bool:
            del m, p_  # unused
            grounded_op = op.ground(tuple(o_))
            abs_state = utils.abstract(s_, preds)
            return grounded_op.add_effects.issubset(abs_state) and \
                not (grounded_op.delete_effects & abs_state)
        return _terminal

    @staticmethod
    def _create_policy_from_regressor(regressor: TemporaryMLP
        ) -> Callable[[State, Dict, Sequence[Object], Array], Action]:
        def _policy(s_: State, m: Dict, o_: Sequence[Object],
                    p_: Array) -> bool:
            del m  # unused
            x_lst : List[Any] = [1.0]  # start with bias term
            for obj in o_:
                x_lst.extend(s_[obj])
            x_lst.extend(p_)
            x = np.array(x_lst)
            action = regressor(x)
            return Action(np.array(action.detach(), dtype=np.float32))
        return _policy


class _KnownOptionsOptionLearner(_OptionLearnerBase):
    """The "option learner" that's used when we're in the code path where
    CFG.option_learner is "no_learning".
    """
    def learn_option_specs(
            self, strips_ops: List[STRIPSOperator],
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

    def update_segment_from_option_spec(
            self, segment: Segment, option_spec: OptionSpec) -> None:
        # If we're not doing option learning, the segments will already have
        # the options, so there is nothing to do here.
        pass


class _OracleOptionLearner(_OptionLearnerBase):
    """The option learner that just cheats by looking up ground truth options
    from the environment. Useful for testing.
    """
    def learn_option_specs(
            self, strips_ops: List[STRIPSOperator],
            datastores: List[Datastore]) -> List[OptionSpec]:
        env = create_env(CFG.env)
        option_specs: List[OptionSpec] = []
        if CFG.env == "cover":
            assert len(strips_ops) == 3
            PickPlace = [option for option in env.options
                         if option.name == "PickPlace"][0]
            # All strips operators use the same PickPlace option,
            # which has no parameters.
            for _ in strips_ops:
                option_specs.append((PickPlace, []))
        elif CFG.env == "blocks":
            assert len(strips_ops) == 4
            Pick = [option for option in env.options
                    if option.name == "Pick"][0]
            Stack = [option for option in env.options
                     if option.name == "Stack"][0]
            PutOnTable = [option for option in env.options
                          if option.name == "PutOnTable"][0]
            for op in strips_ops:
                if {atom.predicate.name for atom in op.preconditions} in \
                   ({"GripperOpen", "Clear", "OnTable"},
                    {"GripperOpen", "Clear", "On"}):
                    # PickFromTable or Unstack operators
                    gripper_open_atom = [
                        atom for atom in op.preconditions
                        if atom.predicate.name == "GripperOpen"][0]
                    robot = gripper_open_atom.variables[0]
                    clear_atom = [
                        atom for atom in op.preconditions
                        if atom.predicate.name == "Clear"][0]
                    block = clear_atom.variables[0]
                    option_specs.append((Pick, [robot, block]))
                elif {atom.predicate.name for atom in op.preconditions} == \
                     {"Clear", "Holding"}:
                    # Stack operator
                    gripper_open_atom = [
                        atom for atom in op.add_effects
                        if atom.predicate.name == "GripperOpen"][0]
                    robot = gripper_open_atom.variables[0]
                    clear_atom = [
                        atom for atom in op.preconditions
                        if atom.predicate.name == "Clear"][0]
                    otherblock = clear_atom.variables[0]
                    option_specs.append((Stack, [robot, otherblock]))
                elif {atom.predicate.name for atom in op.preconditions} == \
                     {"Holding"}:
                    # PutOnTable operator
                    gripper_open_atom = [
                        atom for atom in op.add_effects
                        if atom.predicate.name == "GripperOpen"][0]
                    robot = gripper_open_atom.variables[0]
                    option_specs.append((PutOnTable, [robot]))
        return option_specs

    def update_segment_from_option_spec(
            self, segment: Segment, option_spec: OptionSpec) -> None:
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
                picked_blocks = [obj for obj in segment.states[1]
                                 if obj.type.name == "block" and
                                 segment.states[0].get(obj, "held") < 0.5 <
                                 segment.states[1].get(obj, "held")]
                assert len(picked_blocks) == 1
                block = picked_blocks[0]
                params = np.zeros(3, dtype=np.float32)
                option = param_opt.ground([robby, block], params)
                segment.set_option(option)
            elif param_opt.name == "PutOnTable":
                x, y, _, _ = act
                params = np.array([
                    (x - BlocksEnv.x_lb) / (BlocksEnv.x_ub - BlocksEnv.x_lb),
                    (y - BlocksEnv.y_lb) / (BlocksEnv.y_ub - BlocksEnv.y_lb)])
                option = param_opt.ground([robby], params)
                segment.set_option(option)
            elif param_opt.name == "Stack":
                assert len(opt_vars) == 2
                dropped_blocks = [obj for obj in segment.states[1]
                                  if obj.type.name == "block" and
                                  segment.states[1].get(obj, "held") < 0.5 <
                                  segment.states[0].get(obj, "held")]
                assert len(dropped_blocks) == 1
                block = dropped_blocks[0]
                params = np.array([0, 0, BlocksEnv.block_size],
                                  dtype=np.float32)
                option = param_opt.ground([robby, block], params)
                segment.set_option(option)
