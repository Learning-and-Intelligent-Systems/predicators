"""Definitions of option learning strategies.
"""

from __future__ import annotations
import abc
from typing import List
import numpy as np
from predicators.src.structs import STRIPSOperator, OptionSpec, Partition, \
    Segment
from predicators.src.settings import CFG
from predicators.src.envs import create_env, BlocksEnv


def create_option_learner() -> _OptionLearnerBase:
    """Create an option learner given its name.
    """
    if not CFG.do_option_learning:
        return _KnownOptionsOptionLearner()
    if CFG.option_learner == "oracle":
        return _OracleOptionLearner()
    raise NotImplementedError(f"Unknown option learner: {CFG.option_learner}")


class _OptionLearnerBase:
    """Struct defining an option learner, which has an abstract method for
    learning option specs and an abstract method for annotating data segments
    with options.
    """
    @abc.abstractmethod
    def learn_option_specs(
            self, strips_ops: List[STRIPSOperator],
            partitions: List[Partition]) -> List[OptionSpec]:
        """Given partitioned data and some STRIPS operators fit on that data,
        learn option specs, which are tuples of (ParameterizedOption,
        Sequence[Variable]). The returned option specs should be one-to-one
        with the given strips_ops / partitions (which are already one-to-one
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


class _KnownOptionsOptionLearner(_OptionLearnerBase):
    """The "option learner" that's used when we're in the code path where
    CFG.do_option_learning is False.
    """
    def learn_option_specs(
            self, strips_ops: List[STRIPSOperator],
            partitions: List[Partition]) -> List[OptionSpec]:
        # Since we're not actually doing option learning, the data already
        # contains the options. So, we just extract option specs from the data.
        option_specs = []
        for partition in partitions:
            param_option = None
            option_vars = None
            for i, (segment, sub) in enumerate(partition):
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
                "No data for this partition?"
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
            partitions: List[Partition]) -> List[OptionSpec]:
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
