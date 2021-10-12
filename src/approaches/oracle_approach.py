"""A TAMP approach that uses hand-specified operators.

The approach is aware of the initial predicates and options.
Predicates that are not in the initial predicates are excluded from
the ground truth operators. If an operator's option is not included,
that operator will not be generated at all.
"""

from typing import List, Sequence, Set
import numpy as np
from predicators.src.approaches import TAMPApproach
from predicators.src.envs import create_env
from predicators.src.structs import Operator, Predicate, State, \
    ParameterizedOption, Variable, Type, LiftedAtom, Object, Array
from predicators.src.settings import CFG


class OracleApproach(TAMPApproach):
    """A TAMP approach that uses hand-specified operators.
    """
    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_operators(self) -> Set[Operator]:
        return get_gt_ops(self._initial_predicates,
                           self._initial_options)


def get_gt_ops(predicates: Set[Predicate],
               options: Set[ParameterizedOption]) -> Set[Operator]:
    """Create ground truth operators for an env.
    """
    if CFG.env == "cover":
        ops = _get_cover_gt_ops(options_are_typed=False)
    elif CFG.env == "cover_typed":
        ops = _get_cover_gt_ops(options_are_typed=True)
    else:
        raise NotImplementedError("Ground truth operators not implemented")
    # Filter out excluded predicates/options
    final_ops = set()
    for op in ops:
        if op.option not in options:
            continue
        op = op.filter_predicates(predicates)
        final_ops.add(op)
    return final_ops


def _get_from_env_by_names(env_name: str, names: Sequence[str],
                           env_attr: str) -> List:
    """Helper for loading types, predicates, and options by name.
    """
    env = create_env(env_name)
    name_to_env_obj = {}
    for o in getattr(env, env_attr):
        name_to_env_obj[o.name] = o
    assert set(name_to_env_obj).issuperset(set(names))
    return [name_to_env_obj[name] for name in names]


def _get_types_by_names(env_name: str,
                        names: Sequence[str]) -> List[Type]:
    """Load types from an env given their names.
    """
    return _get_from_env_by_names(env_name, names, "types")


def _get_predicates_by_names(env_name: str,
                             names: Sequence[str]) -> List[Predicate]:
    """Load predicates from an env given their names.
    """
    return _get_from_env_by_names(env_name, names, "predicates")


def _get_options_by_names(env_name: str,
                          names: Sequence[str]) -> List[ParameterizedOption]:
    """Load parameterized options from an env given their names.
    """
    return _get_from_env_by_names(env_name, names, "options")


def _get_cover_gt_ops(options_are_typed: bool) -> Set[Operator]:
    """Create ground truth operators for CoverEnv.
    """
    block_type, target_type = _get_types_by_names("cover", ["block", "target"])

    IsBlock, IsTarget, Covers, HandEmpty, Holding = \
        _get_predicates_by_names("cover", ["IsBlock", "IsTarget", "Covers",
                                           "HandEmpty", "Holding"])

    if options_are_typed:
        Pick, Place = _get_options_by_names("cover_typed", ["Pick", "Place"])
    else:
        PickPlace, = _get_options_by_names("cover", ["PickPlace"])

    operators = set()

    # Pick
    block = Variable("?block", block_type)
    parameters = [block]
    if options_are_typed:
        option_vars = [block]
        option = Pick
    else:
        option_vars = []
        option = PickPlace
    preconditions = {LiftedAtom(IsBlock, [block]), LiftedAtom(HandEmpty, [])}
    add_effects = {LiftedAtom(Holding, [block])}
    delete_effects = {LiftedAtom(HandEmpty, [])}
    def pick_sampler(state: State, rng: np.random.Generator,
                     objs: Sequence[Object]) -> Array:
        assert len(objs) == 1
        b = objs[0]
        assert b.type == block_type
        if options_are_typed:
            lb = float(-state.get(b, "width")/2)  # relative positioning only
            ub = float(state.get(b, "width")/2)  # relative positioning only
        else:
            lb = float(state.get(b, "pose") - state.get(b, "width")/2)
            lb = max(lb, 0.0)
            ub = float(state.get(b, "pose") + state.get(b, "width")/2)
            ub = min(ub, 1.0)
        return np.array(rng.uniform(lb, ub, size=(1,)), dtype=np.float32)
    pick_operator = Operator("Pick", parameters, preconditions,
                             add_effects, delete_effects, option,
                             option_vars, pick_sampler)
    operators.add(pick_operator)

    # Place
    target = Variable("?target", target_type)
    parameters = [block, target]
    if options_are_typed:
        option_vars = [target]
        option = Place
    else:
        option_vars = []
        option = PickPlace
    preconditions = {LiftedAtom(IsBlock, [block]),
                     LiftedAtom(IsTarget, [target]),
                     LiftedAtom(Holding, [block])}
    add_effects = {LiftedAtom(HandEmpty, []),
                   LiftedAtom(Covers, [block, target])}
    delete_effects = {LiftedAtom(Holding, [block])}
    def place_sampler(state: State, rng: np.random.Generator,
                      objs: Sequence[Object]) -> Array:
        assert len(objs) == 2
        t = objs[1]
        assert t.type == target_type
        lb = float(state.get(t, "pose") - state.get(t, "width")/10)
        lb = max(lb, 0.0)
        ub = float(state.get(t, "pose") + state.get(t, "width")/10)
        ub = min(ub, 1.0)
        return np.array(rng.uniform(lb, ub, size=(1,)), dtype=np.float32)
    place_operator = Operator("Place", parameters, preconditions,
                              add_effects, delete_effects, option,
                              option_vars, place_sampler)
    operators.add(place_operator)

    return operators
