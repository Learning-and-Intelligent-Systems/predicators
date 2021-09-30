"""A TAMP approach that uses hand-specified operators.

The approach is aware of the initial predicates and options.
Predicates that are not in the initial predicates are excluded from
the ground truth operators. If an operator's option is not included,
that operator will not be generated at all.
"""
from typing import List, Sequence, Set
from absl import flags
from predicators.src.approaches import TAMPApproach
from predicators.src.envs import create_env
from predicators.src.structs import Operator, Predicate, \
    ParameterizedOption, Variable, Type, LiftedAtom


class OracleApproach(TAMPApproach):
    """A TAMP approach that uses hand-specified operators.
    """
    def _get_current_operators(self) -> Set[Operator]:
        """Get the current set of operators.
        """
        return _get_gt_ops(self._initial_predicates,
                           self._initial_options)


def _get_gt_ops(predicates: Set[Predicate],
                options: Set[ParameterizedOption]) -> Set[Operator]:
    """Create ground-truth operators for an env.
    """
    if flags.env.name == "Cover":
        ops = _get_cover_gt_ops()
    else:
        raise NotImplementedError("Groundtruth operators not implemented")
    # Filter out excluded predicates/options
    final_ops = set()
    for op in ops:
        if op.option not in options:
            continue
        op = _remove_excluded_predicates_from_op(op, predicates)
        final_ops.add(op)
    return final_ops


def _remove_excluded_predicates_from_op(operator: Operator,
    included_predicates: Set[Predicate]) -> Operator:
    """Remove excluded predicates from everywhere in an operator.
    """
    preconditions = {a for a in operator.preconditions \
                     if a.predicate in included_predicates}
    add_effects = {a for a in operator.add_effects \
                   if a.predicate in included_predicates}
    delete_effects = {a for a in operator.delete_effects \
                      if a.predicate in included_predicates}
    # Note that the parameters must stay the same for the sake
    # of the sampler input arguments
    return Operator(operator.name, operator.parameters,
                    preconditions, add_effects, delete_effects,
                    operator.option, operator._sampler)


def _get_from_env_by_names(env_name: str, names: Sequence[str],
                           env_attr: str) -> List:
    """Helper for loading predicates and options by name.
    """
    env = create_env(env_name)
    name_to_env_obj = {}
    for o in getattr(env, env_attr):
        name_to_env_obj[o.name] = o
    assert set(name_to_env_obj.keys()).issuperset(set(names))
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


def _get_cover_gt_ops() -> Set[Operator]:
    """Create ground-truth operators for CoverEnv.
    """
    block_type, target_type = _get_types_by_names("Cover", ["block", "target"])

    IsBlock, IsTarget, Covers, HandEmpty, Holding = \
        _get_predicates_by_names("Cover", ["IsBlock", "IsTarget", "Covers",
                                           "HandEmpty", "Holding"])

    PickPlace, = _get_options_by_names("Cover", ["PickPlace"])

    operators = set()

    # Pick
    block = Variable("?block", block_type)
    parameters = [block]
    preconditions = {LiftedAtom(IsBlock, [block]), LiftedAtom(HandEmpty, [])}
    add_effects = {LiftedAtom(Holding, [block])}
    delete_effects = {LiftedAtom(HandEmpty, [])}
    def pick_sampler(state, rng, objs):
        assert len(objs) == 1
        b = objs[0]
        assert b.type == block_type
        lb = state.get(b, "pose") - state.get(b, "width")/2
        ub = state.get(b, "pose") + state.get(b, "width")/2
        return rng.uniform(lb, ub, size=(1,))
    pick_operator = Operator("Pick", parameters, preconditions,
                             add_effects, delete_effects, PickPlace,
                             pick_sampler)
    operators.add(pick_operator)

    # Place
    target = Variable("?target", target_type)
    parameters = [block, target]
    preconditions = {LiftedAtom(IsBlock, [block]),
                     LiftedAtom(IsTarget, [target]),
                     LiftedAtom(Holding, [block])}
    add_effects = {LiftedAtom(HandEmpty, []),
                   LiftedAtom(Covers, [block, target])}
    delete_effects = {LiftedAtom(Holding, [block])}
    def place_sampler(state, rng, objs):
        assert len(objs) == 2
        t = objs[1]
        assert t.type == target_type
        lb = state.get(t, "pose") - state.get(t, "width")/10
        ub = state.get(t, "pose") + state.get(t, "width")/10
        return rng.uniform(lb, ub, size=(1,))
    place_operator = Operator("Place", parameters, preconditions,
                              add_effects, delete_effects, PickPlace,
                              place_sampler)
    operators.add(place_operator)

    return operators
