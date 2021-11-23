"""A TAMP approach that uses hand-specified NSRTs.

The approach is aware of the initial predicates and options.
Predicates that are not in the initial predicates are excluded from
the ground truth NSRTs. If an NSRT's option is not included,
that NSRT will not be generated at all.
"""

from typing import List, Sequence, Set
import itertools
import numpy as np
from predicators.src.approaches import TAMPApproach
from predicators.src.envs import create_env, BlocksEnv
from predicators.src.structs import NSRT, Predicate, State, \
    ParameterizedOption, Variable, Type, LiftedAtom, Object, Array
from predicators.src.settings import CFG
from predicators.src.envs.behavior_options import navigate_to_param_sampler, \
    grasp_obj_param_sampler, place_ontop_obj_pos_sampler
from predicators.src.envs import get_env_instance

class OracleApproach(TAMPApproach):
    """A TAMP approach that uses hand-specified NSRTs.
    """
    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_nsrts(self) -> Set[NSRT]:
        return get_gt_nsrts(self._initial_predicates, self._initial_options)


def get_gt_nsrts(predicates: Set[Predicate],
                 options: Set[ParameterizedOption]) -> Set[NSRT]:
    """Create ground truth NSRTs for an env.
    """
    if CFG.env in ("cover", "cover_hierarchical_types"):
        nsrts = _get_cover_gt_nsrts(options_are_typed=False)
    elif CFG.env == "cover_typed_options":
        nsrts = _get_cover_gt_nsrts(options_are_typed=True)
    elif CFG.env == "cover_multistep_options":
        nsrts = _get_cover_gt_nsrts(options_are_typed=True,
                                    place_sampler_relative=True)
    elif CFG.env == "cluttered_table":
        nsrts = _get_cluttered_table_gt_nsrts()
    elif CFG.env == "blocks":
        nsrts = _get_blocks_gt_nsrts()
    elif CFG.env == "behavior":
        nsrts = _get_behavior_gt_nsrts()
    else:
        raise NotImplementedError("Ground truth NSRTs not implemented")
    # Filter out excluded predicates/options
    final_nsrts = set()
    for nsrt in nsrts:
        if nsrt.option not in options:
            continue
        nsrt = nsrt.filter_predicates(predicates)
        final_nsrts.add(nsrt)
    return final_nsrts


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


def _get_cover_gt_nsrts(options_are_typed: bool,
                        place_sampler_relative: bool = False) -> Set[NSRT]:
    """Create ground truth NSRTs for CoverEnv.
    """
    block_type, target_type = _get_types_by_names(CFG.env, ["block", "target"])

    IsBlock, IsTarget, Covers, HandEmpty, Holding = \
        _get_predicates_by_names(CFG.env, ["IsBlock", "IsTarget", "Covers",
                                           "HandEmpty", "Holding"])

    if options_are_typed:
        Pick, Place = _get_options_by_names(CFG.env, ["Pick", "Place"])
    else:
        PickPlace, = _get_options_by_names(CFG.env, ["PickPlace"])

    nsrts = set()

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
        assert b.is_instance(block_type)
        if options_are_typed:
            lb = float(-state.get(b, "width")/2)  # relative positioning only
            ub = float(state.get(b, "width")/2)  # relative positioning only
        else:
            lb = float(state.get(b, "pose") - state.get(b, "width")/2)
            lb = max(lb, 0.0)
            ub = float(state.get(b, "pose") + state.get(b, "width")/2)
            ub = min(ub, 1.0)
        return np.array(rng.uniform(lb, ub, size=(1,)), dtype=np.float32)
    pick_nsrt = NSRT("Pick", parameters, preconditions,
                     add_effects, delete_effects, option,
                     option_vars, pick_sampler)
    nsrts.add(pick_nsrt)

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
        assert t.is_instance(target_type)
        if place_sampler_relative:
            lb = float(-state.get(t, "width")/2)  # relative positioning only
            ub = float(state.get(t, "width")/2)  # relative positioning only
        else:
            lb = float(state.get(t, "pose") - state.get(t, "width")/10)
            lb = max(lb, 0.0)
            ub = float(state.get(t, "pose") + state.get(t, "width")/10)
            ub = min(ub, 1.0)
        return np.array(rng.uniform(lb, ub, size=(1,)), dtype=np.float32)
    place_nsrt = NSRT("Place", parameters, preconditions,
                      add_effects, delete_effects, option,
                      option_vars, place_sampler)
    nsrts.add(place_nsrt)

    return nsrts


def _get_cluttered_table_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for ClutteredTableEnv.
    """
    can_type, = _get_types_by_names("cluttered_table", ["can"])

    HandEmpty, Holding = _get_predicates_by_names(
        "cluttered_table", ["HandEmpty", "Holding"])

    Grasp, Dump = _get_options_by_names("cluttered_table", ["Grasp", "Dump"])

    nsrts = set()

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
    grasp_nsrt = NSRT("Grasp", parameters, preconditions,
                      add_effects, delete_effects, option,
                      option_vars, grasp_sampler)
    nsrts.add(grasp_nsrt)

    # Dump
    can = Variable("?can", can_type)
    parameters = [can]
    option_vars = []
    option = Dump
    preconditions = {LiftedAtom(Holding, [can])}
    add_effects = {LiftedAtom(HandEmpty, [])}
    delete_effects = {LiftedAtom(Holding, [can])}
    dump_nsrt = NSRT("Dump", parameters, preconditions, add_effects,
                     delete_effects, option, option_vars,
                     lambda s, r, o: np.array([], dtype=np.float32))
    nsrts.add(dump_nsrt)

    return nsrts


def _get_blocks_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for BlocksEnv.
    """
    block_type, robot_type = _get_types_by_names("blocks", ["block", "robot"])

    On, OnTable, GripperOpen, Holding, Clear = _get_predicates_by_names(
        "blocks", ["On", "OnTable", "GripperOpen", "Holding", "Clear"])

    Pick, Stack, PutOnTable = _get_options_by_names(
        "blocks", ["Pick", "Stack", "PutOnTable"])

    nsrts = set()

    # PickFromTable
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, robot]
    option_vars = [robot, block]
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
    pickfromtable_nsrt = NSRT(
        "PickFromTable", parameters, preconditions, add_effects,
        delete_effects, option, option_vars, pick_sampler)
    nsrts.add(pickfromtable_nsrt)

    # Unstack
    block = Variable("?block", block_type)
    otherblock = Variable("?otherblock", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, otherblock, robot]
    option_vars = [robot, block]
    option = Pick
    preconditions = {LiftedAtom(On, [block, otherblock]),
                     LiftedAtom(Clear, [block]),
                     LiftedAtom(GripperOpen, [robot])}
    add_effects = {LiftedAtom(Holding, [block]),
                   LiftedAtom(Clear, [otherblock])}
    delete_effects = {LiftedAtom(On, [block, otherblock]),
                      LiftedAtom(Clear, [block]),
                      LiftedAtom(GripperOpen, [robot])}
    unstack_nsrt = NSRT(
        "Unstack", parameters, preconditions, add_effects,
        delete_effects, option, option_vars, pick_sampler)
    nsrts.add(unstack_nsrt)

    # Stack
    block = Variable("?block", block_type)
    otherblock = Variable("?otherblock", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, otherblock, robot]
    option_vars = [robot, otherblock]
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
        return np.array([0, 0, BlocksEnv.block_size], dtype=np.float32)
    stack_nsrt = NSRT(
        "Stack", parameters, preconditions, add_effects,
        delete_effects, option, option_vars, stack_sampler)
    nsrts.add(stack_nsrt)

    # PutOnTable
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, robot]
    option_vars = [robot]
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
    putontable_nsrt = NSRT(
        "PutOnTable", parameters, preconditions, add_effects,
        delete_effects, option, option_vars, putontable_sampler)
    nsrts.add(putontable_nsrt)

    return nsrts


def _get_behavior_gt_nsrts() -> Set[NSRT]:
    """Create ground truth nsrts for BehaviorEnv.
    """
    env = get_env_instance("behavior")

    type_name_to_type = {t.name : t for t in env.types}
    pred_name_to_pred = {p.name : p for p in env.predicates}

    def _get_lifted_atom(base_pred_name, objects):
        type_names = "-".join(o.type.name for o in objects)
        pred_name = f"{base_pred_name}-{type_names}"
        pred = pred_name_to_pred[pred_name]
        return LiftedAtom(pred, objects)

    agent_type = type_name_to_type["agent.n.01"]
    agent_obj = Variable("?agent", agent_type)

    nsrts = set()
    op_name_count = itertools.count()

    for option in env.options:
        split_name = option.name.split("-")
        base_option_name = split_name[0]
        option_arg_type_names = split_name[1:]

        if base_option_name == "NavigateTo":
            assert len(option_arg_type_names) == 1
            target_obj_type_name = option_arg_type_names[0]
            target_obj_type = type_name_to_type[target_obj_type_name]
            target_obj = Variable("?targ", target_obj_type)

            # Navigate to from nextto nothing
            nextto_nothing = _get_lifted_atom("nextto-nothing", [agent_obj])
            parameters = [target_obj, agent_obj]
            option_vars = [target_obj]
            preconditions = {nextto_nothing}
            add_effects = {_get_lifted_atom("nextto", [target_obj, agent_obj])}
            delete_effects = {nextto_nothing}
            nsrt = NSRT(f"{option.name}-{next(op_name_count)}",
                                parameters, preconditions, add_effects,
                                delete_effects, option, option_vars,
                                lambda s, r, o: navigate_to_param_sampler(r))
            nsrts.add(nsrt)

            # Navigate to while nextto something
            for origin_obj_type in sorted(env.types):
                origin_obj = Variable("?origin", origin_obj_type)
                origin_next_to = _get_lifted_atom("nextto",
                    [origin_obj, agent_obj])
                targ_next_to = _get_lifted_atom("nextto",
                    [target_obj, agent_obj])
                parameters = [origin_obj, agent_obj, target_obj]
                option_vars = [target_obj]
                preconditions = {origin_next_to}
                add_effects = {targ_next_to}
                delete_effects = {origin_next_to}
                nsrt = NSRT(f"{option.name}-{next(op_name_count)}",
                                    parameters, preconditions, add_effects,
                                    delete_effects, option, option_vars,
                                    lambda s, r, o: navigate_to_param_sampler(r))
                nsrts.add(nsrt)

        elif base_option_name == "Grasp":
            assert len(option_arg_type_names) == 1
            target_obj_type_name = option_arg_type_names[0]
            target_obj_type = type_name_to_type[target_obj_type_name]
            target_obj = Variable("?targ", target_obj_type)

            # Pick from ontop something
            for surf_obj_type in sorted(env.types):
                surf_obj = Variable("?surf", surf_obj_type)
                parameters = [target_obj, agent_obj, surf_obj]
                option_vars = [target_obj]
                handempty = _get_lifted_atom("handempty", [])
                targ_next_to = _get_lifted_atom("nextto",
                    [target_obj, agent_obj])
                targ_holding = _get_lifted_atom("holding", [target_obj])
                preconditions = {handempty, targ_next_to}
                add_effects = {targ_holding}
                delete_effects = {handempty}
                nsrt = NSRT(f"{option.name}-{next(op_name_count)}",
                                    parameters, preconditions, add_effects,
                                    delete_effects, option, option_vars,
                                    lambda s, r, o: grasp_obj_param_sampler(r))
                nsrts.add(nsrt)

        elif base_option_name == "PlaceOnTop":
            assert len(option_arg_type_names) == 1
            surf_obj_type_name = option_arg_type_names[0]
            surf_obj_type = type_name_to_type[surf_obj_type_name]
            surf_obj = Variable("?surf", surf_obj_type)

            # We need to place the object we're holding!
            for held_obj_types in sorted(env.types):
                held_obj = Variable("?held", held_obj_types)
                parameters = [held_obj, agent_obj, surf_obj]
                option_vars = [held_obj, surf_obj]
                handempty = _get_lifted_atom("handempty", [])
                held_holding = _get_lifted_atom("holding", [held_obj])
                surf_next_to = _get_lifted_atom("nextto", [surf_obj, agent_obj])
                ontop = _get_lifted_atom("ontop", [held_obj, surf_obj])
                preconditions = {held_holding, surf_next_to}
                add_effects = {ontop, handempty}
                delete_effects = {held_holding}
                nsrt = NSRT(f"{option.name}-{next(op_name_count)}",
                                    parameters, preconditions, add_effects,
                                    delete_effects, option, option_vars,
                                    lambda s, r, o: place_ontop_obj_pos_sampler(env, o, rng=r))
                nsrts.add(nsrt)
        
        else:
            raise ValueError(
                f"Unexpected base option name: {base_option_name}")

    return nsrts