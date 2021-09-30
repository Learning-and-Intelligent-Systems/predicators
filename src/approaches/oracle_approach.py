"""A TAMP approach that uses hand-specified operators.

The approach is aware of the initial predicates and options.
Predicates that are not in the initial predicates are excluded from
the ground truth operators. If an operator's option is not included,
that operator will not be generated at all.
"""
from absl import flags
from typing import List, Sequence, Set
from predicators.src.approaches import TAMPApproach
from predicators.src.envs import create_env
from predicators.src.structs import Operator, Predicate, \
    ParameterizedOption, Type


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
        return _get_cover_gt_ops(predicates, options)
    raise NotImplementedError("Groundtruth operators not implemented")


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


def _get_cover_gt_ops(predicates: Set[Predicate],
                      options: Set[ParameterizedOption]) -> Set[Operator]:
    """Create ground-truth operators for CoverEnv.
    """
    block_type, target_type = _get_types_by_names("Cover", ["block", "target"])

    IsBlock, IsTarget, Covers, HandEmpty, Holding = \
        _get_predicates_by_names("Cover", ["IsBlock", "IsTarget", "Covers",
                                           "HandEmpty", "Holding"])

    PickPlace, = _get_options_by_names("Cover", ["PickPlace"])

    operators = set()

    # Pick
    block = block_type("?block")
    parameters = [block]
    preconditions = {IsBlock([block]), HandEmpty([])}
    add_effects = {Holding([block])}
    delete_effects = {HandEmpty([])}
    def sampler(rng, state, objs):
        assert len(objs) == 1
        b = objs[0]
        assert b.type == block_type
        lb = state.get(b, "pose") - state.get(b, "width")/2
        ub = state.get(b, "pose") + state.get(b, "width")/2
        return rng.uniform(lb, ub)
    pick_operator = Operator("Pick", parameters, preconditions,
                             add_effects, delete_effects, PickPlace, sampler)
    operators.add(pick_operator)

    # Place
    target = target_type("?target")
    parameters = [block, target]
    preconditions = {IsBlock([block]), IsTarget([target]), Holding([block])}
    add_effects = {HandEmpty([]), Covers([block, target])}
    delete_effects = {Holding([block])}
    def sampler(rng, state, objs):
        assert len(objs) == 2
        t = objs[1]
        assert t.type == target_type
        lb = state.get(t, "pose") - state.get(t, "width")/10
        ub = state.get(t, "pose") + state.get(t, "width")/10
        return rng.uniform(lb, ub)
    place_operator = Operator("Place", parameters, preconditions,
                              add_effects, delete_effects, PickPlace, sampler)
    operators.add(place_operator)

    return operators    
