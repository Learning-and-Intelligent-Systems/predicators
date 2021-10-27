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
        return get_gt_ops(self._initial_predicates, self._initial_options)


def get_gt_ops(predicates: Set[Predicate],
               options: Set[ParameterizedOption]) -> Set[Operator]:
    """Create ground truth operators for an env.
    """
    if CFG.env == "cover":
        ops = _get_cover_gt_ops(options_are_typed=False)
    elif CFG.env == "cover_typed":
        ops = _get_cover_gt_ops(options_are_typed=True)
    elif CFG.env == "augmented_cover":
        ops = _get_cover_augmented_gt_ops()
    elif CFG.env == "cluttered_table":
        ops = _get_cluttered_table_gt_ops()
    elif CFG.env == "blocks":
        ops = _get_blocks_gt_ops()
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

def _get_cover_augmented_gt_ops() -> Set[Operator]:
    pass

def _get_cluttered_table_gt_ops() -> Set[Operator]:
    """Create ground truth operators for ClutteredTableEnv.
    """
    can_type, = _get_types_by_names("cluttered_table", ["can"])

    HandEmpty, Holding = _get_predicates_by_names(
        "cluttered_table", ["HandEmpty", "Holding"])

    Grasp, Dump = _get_options_by_names("cluttered_table", ["Grasp", "Dump"])

    operators = set()

    # Grasp
    can = Variable("?can", can_type)
    parameters = [can]
    option_vars = [can]
    option = Grasp
    preconditions = {LiftedAtom(HandEmpty, [])}
    add_effects = {LiftedAtom(Holding, [can])}
    delete_effects = {LiftedAtom(HandEmpty, [])}
    def grasp_sampler(state: State, rng: np.random.Generator,
                      objs: Sequence[Object]) -> Array:
        assert len(objs) == 1
        can = objs[0]
        end_x = state.get(can, "pose_x")
        end_y = state.get(can, "pose_y")
        start_x, start_y = rng.uniform(0.0, 1.0, size=2)  # start from anywhere
        return np.array([start_x, start_y, end_x, end_y], dtype=np.float32)
    grasp_operator = Operator("Grasp", parameters, preconditions,
                              add_effects, delete_effects, option,
                              option_vars, grasp_sampler)
    operators.add(grasp_operator)

    # Dump
    can = Variable("?can", can_type)
    parameters = [can]
    option_vars = []
    option = Dump
    preconditions = {LiftedAtom(Holding, [can])}
    add_effects = {LiftedAtom(HandEmpty, [])}
    delete_effects = {LiftedAtom(Holding, [can])}
    dump_operator = Operator("Dump", parameters, preconditions, add_effects,
                             delete_effects, option, option_vars,
                             lambda s, r, o: np.array([], dtype=np.float32))
    operators.add(dump_operator)

    return operators


def _get_blocks_gt_ops() -> Set[Operator]:
    """Create ground truth operators for BlocksEnv.
    """
    block_type, robot_type = _get_types_by_names("blocks", ["block", "robot"])

    On, OnTable, GripperOpen, Holding, Clear = _get_predicates_by_names(
        "blocks", ["On", "OnTable", "GripperOpen", "Holding", "Clear"])

    Pick, Stack, PutOnTable = _get_options_by_names(
        "blocks", ["Pick", "Stack", "PutOnTable"])

    operators = set()

    # PickFromTable
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, robot]
    option_vars = [block]
    option = Pick
    preconditions = {LiftedAtom(OnTable, [block]),
                     LiftedAtom(Clear, [block]),
                     LiftedAtom(GripperOpen, [robot])}
    add_effects = {LiftedAtom(Holding, [block])}
    delete_effects = {LiftedAtom(OnTable, [block]),
                      LiftedAtom(Clear, [block]),
                      LiftedAtom(GripperOpen, [robot])}
    def pick_sampler(state: State, rng: np.random.Generator,
                     objs: Sequence[Object]) -> Array:
        del state, rng, objs  # unused
        return np.zeros(3, dtype=np.float32)
    pickfromtable_operator = Operator(
        "PickFromTable", parameters, preconditions, add_effects,
        delete_effects, option, option_vars, pick_sampler)
    operators.add(pickfromtable_operator)

    # Unstack
    block = Variable("?block", block_type)
    otherblock = Variable("?otherblock", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, otherblock, robot]
    option_vars = [block]
    option = Pick
    preconditions = {LiftedAtom(On, [block, otherblock]),
                     LiftedAtom(Clear, [block]),
                     LiftedAtom(GripperOpen, [robot])}
    add_effects = {LiftedAtom(Holding, [block]),
                   LiftedAtom(Clear, [otherblock])}
    delete_effects = {LiftedAtom(On, [block, otherblock]),
                      LiftedAtom(Clear, [block]),
                      LiftedAtom(GripperOpen, [robot])}
    unstack_operator = Operator(
        "Unstack", parameters, preconditions, add_effects,
        delete_effects, option, option_vars, pick_sampler)
    operators.add(unstack_operator)

    # Stack
    block = Variable("?block", block_type)
    otherblock = Variable("?otherblock", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, otherblock, robot]
    option_vars = [otherblock]
    option = Stack
    preconditions = {LiftedAtom(Holding, [block]),
                     LiftedAtom(Clear, [otherblock])}
    add_effects = {LiftedAtom(On, [block, otherblock]),
                   LiftedAtom(Clear, [block]),
                   LiftedAtom(GripperOpen, [robot])}
    delete_effects = {LiftedAtom(Holding, [block]),
                      LiftedAtom(Clear, [otherblock])}
    def stack_sampler(state: State, rng: np.random.Generator,
                      objs: Sequence[Object]) -> Array:
        del state, rng, objs  # unused
        return np.array([0, 0, CFG.blocks_block_size], dtype=np.float32)
    stack_operator = Operator(
        "Stack", parameters, preconditions, add_effects,
        delete_effects, option, option_vars, stack_sampler)
    operators.add(stack_operator)

    # PutOnTable
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, robot]
    option_vars = []
    option = PutOnTable
    preconditions = {LiftedAtom(Holding, [block])}
    add_effects = {LiftedAtom(OnTable, [block]),
                   LiftedAtom(Clear, [block]),
                   LiftedAtom(GripperOpen, [robot])}
    delete_effects = {LiftedAtom(Holding, [block])}
    def putontable_sampler(state: State, rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del state, objs  # unused
        x = rng.uniform()
        y = rng.uniform()
        return np.array([x, y], dtype=np.float32)
    putontable_operator = Operator(
        "PutOnTable", parameters, preconditions, add_effects,
        delete_effects, option, option_vars, putontable_sampler)
    operators.add(putontable_operator)

    return operators
