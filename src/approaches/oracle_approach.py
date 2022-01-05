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
from predicators.src.envs import create_env, BlocksEnv, PaintingEnv, PlayroomEnv
from predicators.src.structs import NSRT, Predicate, State, \
    ParameterizedOption, Variable, Type, LiftedAtom, Object, Array
from predicators.src.settings import CFG
from predicators.src.envs.behavior_options import (
    navigate_to_param_sampler,
    grasp_obj_param_sampler,
    place_ontop_obj_pos_sampler,
)
from predicators.src.envs import get_env_instance


class OracleApproach(TAMPApproach):
    """A TAMP approach that uses hand-specified NSRTs."""

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_nsrts(self) -> Set[NSRT]:
        return get_gt_nsrts(self._initial_predicates, self._initial_options)


def get_gt_nsrts(
    predicates: Set[Predicate], options: Set[ParameterizedOption]
) -> Set[NSRT]:
    """Create ground truth NSRTs for an env."""
    if CFG.env in ("cover", "cover_hierarchical_types"):
        nsrts = _get_cover_gt_nsrts(options_are_typed=False)
    elif CFG.env == "cover_typed_options":
        nsrts = _get_cover_gt_nsrts(options_are_typed=True)
    elif CFG.env == "cover_multistep_options":
        nsrts = _get_cover_gt_nsrts(
            options_are_typed=True, place_sampler_relative=True
        )
        nsrts = _get_cover_gt_nsrts(options_are_typed=True,
                                    include_robot_in_holding=True,
                                    place_sampler_relative=True)
    elif CFG.env == "cluttered_table":
        nsrts = _get_cluttered_table_gt_nsrts()
    elif CFG.env == "blocks":
        nsrts = _get_blocks_gt_nsrts()
    elif CFG.env == "behavior":
        nsrts = _get_behavior_gt_nsrts()
    elif CFG.env == "painting":
        nsrts = _get_painting_gt_nsrts()
    elif CFG.env == "playroom":
        nsrts = _get_playroom_gt_nsrts()
    elif CFG.env == "repeated_nextto":
        nsrts = _get_repeated_nextto_gt_nsrts()
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


def _get_from_env_by_names(
    env_name: str, names: Sequence[str], env_attr: str
) -> List:
    """Helper for loading types, predicates, and options by name."""
    env = create_env(env_name)
    name_to_env_obj = {}
    for o in getattr(env, env_attr):
        name_to_env_obj[o.name] = o
    assert set(name_to_env_obj).issuperset(set(names))
    return [name_to_env_obj[name] for name in names]


def _get_types_by_names(env_name: str, names: Sequence[str]) -> List[Type]:
    """Load types from an env given their names."""
    return _get_from_env_by_names(env_name, names, "types")


def _get_predicates_by_names(
    env_name: str, names: Sequence[str]
) -> List[Predicate]:
    """Load predicates from an env given their names."""
    return _get_from_env_by_names(env_name, names, "predicates")


def _get_options_by_names(
    env_name: str, names: Sequence[str]
) -> List[ParameterizedOption]:
    """Load parameterized options from an env given their names."""
    return _get_from_env_by_names(env_name, names, "options")


def _get_cover_gt_nsrts(options_are_typed: bool,
                        include_robot_in_holding: bool = False,
                        place_sampler_relative: bool = False) -> Set[NSRT]:
    """Create ground truth NSRTs for CoverEnv.
    """
    block_type, target_type, robot_type = _get_types_by_names(
        CFG.env, ["block", "target", "robot"])

    IsBlock, IsTarget, Covers, HandEmpty, Holding = _get_predicates_by_names(
        CFG.env, ["IsBlock", "IsTarget", "Covers", "HandEmpty", "Holding"]
    )

    if options_are_typed:
        Pick, Place = _get_options_by_names(CFG.env, ["Pick", "Place"])
    else:
        (PickPlace,) = _get_options_by_names(CFG.env, ["PickPlace"])

    nsrts = set()

    # Pick
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, robot] if include_robot_in_holding else [block]
    if options_are_typed:
        option_vars = [block]
        option = Pick
    else:
        option_vars = []
        option = PickPlace
    preconditions = {LiftedAtom(IsBlock, [block]), LiftedAtom(HandEmpty, [])}
    if include_robot_in_holding:
        add_effects = {LiftedAtom(Holding, [block, robot])}
    else:
        add_effects = {LiftedAtom(Holding, [block])}
    delete_effects = {LiftedAtom(HandEmpty, [])}

    def pick_sampler(state: State, rng: np.random.Generator,
                     objs: Sequence[Object]) -> Array:
        assert len(objs) == 2 if include_robot_in_holding else len(objs) == 1
        b = objs[0]
        assert b.is_instance(block_type)
        if options_are_typed:
            lb = float(-state.get(b, "width") / 2)  # relative positioning only
            ub = float(state.get(b, "width") / 2)  # relative positioning only
        else:
            lb = float(state.get(b, "pose") - state.get(b, "width") / 2)
            lb = max(lb, 0.0)
            ub = float(state.get(b, "pose") + state.get(b, "width") / 2)
            ub = min(ub, 1.0)
        return np.array(rng.uniform(lb, ub, size=(1,)), dtype=np.float32)

    pick_nsrt = NSRT("Pick", parameters, preconditions,
                     add_effects, delete_effects, set(), option,
                     option_vars, pick_sampler)
    nsrts.add(pick_nsrt)

    # Place
    target = Variable("?target", target_type)
    parameters = [block, target, robot] if include_robot_in_holding \
        else [block, target]
    if options_are_typed:
        option_vars = [target]
        option = Place
    else:
        option_vars = []
        option = PickPlace

    add_effects = {LiftedAtom(HandEmpty, []),
                   LiftedAtom(Covers, [block, target])}
    if include_robot_in_holding:
        preconditions = {LiftedAtom(IsBlock, [block]),
                         LiftedAtom(IsTarget, [target]),
                         LiftedAtom(Holding, [block, robot])}
        delete_effects = {LiftedAtom(Holding, [block, robot])}
    else:
        preconditions = {LiftedAtom(IsBlock, [block]),
                         LiftedAtom(IsTarget, [target]),
                         LiftedAtom(Holding, [block])}
        delete_effects = {LiftedAtom(Holding, [block])}
    def place_sampler(state: State, rng: np.random.Generator,
                      objs: Sequence[Object]) -> Array:
        assert len(objs) == 3 if include_robot_in_holding else len(objs) == 2
        t = objs[1]
        assert t.is_instance(target_type)
        if place_sampler_relative:
            lb = float(-state.get(t, "width") / 2)  # relative positioning only
            ub = float(state.get(t, "width") / 2)  # relative positioning only
        else:
            lb = float(state.get(t, "pose") - state.get(t, "width") / 10)
            lb = max(lb, 0.0)
            ub = float(state.get(t, "pose") + state.get(t, "width") / 10)
            ub = min(ub, 1.0)
        return np.array(rng.uniform(lb, ub, size=(1,)), dtype=np.float32)

    place_nsrt = NSRT("Place", parameters, preconditions,
                      add_effects, delete_effects, set(), option,
                      option_vars, place_sampler)
    nsrts.add(place_nsrt)

    return nsrts


def _get_cluttered_table_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for ClutteredTableEnv."""
    (can_type,) = _get_types_by_names("cluttered_table", ["can"])

    HandEmpty, Holding, Untrashed = _get_predicates_by_names(
        "cluttered_table", ["HandEmpty", "Holding", "Untrashed"])

    Grasp, Dump = _get_options_by_names("cluttered_table", ["Grasp", "Dump"])

    nsrts = set()

    # Grasp
    can = Variable("?can", can_type)
    parameters = [can]
    option_vars = [can]
    option = Grasp
    preconditions = {LiftedAtom(HandEmpty, []), LiftedAtom(Untrashed, [can])}
    add_effects = {LiftedAtom(Holding, [can])}
    delete_effects = {LiftedAtom(HandEmpty, [])}

    def grasp_sampler(
        state: State, rng: np.random.Generator, objs: Sequence[Object]
    ) -> Array:
        assert len(objs) == 1
        can = objs[0]
        # Need a max here in case the can is trashed already, in which case
        # both pose_x and pose_y will be -999.
        end_x = max(0.0, state.get(can, "pose_x"))
        end_y = max(0.0, state.get(can, "pose_y"))
        start_x, start_y = rng.uniform(0.0, 1.0, size=2)  # start from anywhere
        return np.array([start_x, start_y, end_x, end_y], dtype=np.float32)

    grasp_nsrt = NSRT("Grasp", parameters, preconditions,
                      add_effects, delete_effects, set(), option,
                      option_vars, grasp_sampler)
    nsrts.add(grasp_nsrt)

    # Dump
    can = Variable("?can", can_type)
    parameters = [can]
    option_vars = []
    option = Dump
    preconditions = {LiftedAtom(Holding, [can]), LiftedAtom(Untrashed, [can])}
    add_effects = {LiftedAtom(HandEmpty, [])}
    delete_effects = {LiftedAtom(Holding, [can]), LiftedAtom(Untrashed, [can])}
    dump_nsrt = NSRT("Dump", parameters, preconditions, add_effects,
                     delete_effects, set(), option, option_vars,
                     lambda s, r, o: np.array([], dtype=np.float32))
    nsrts.add(dump_nsrt)

    return nsrts


def _get_blocks_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for BlocksEnv."""
    block_type, robot_type = _get_types_by_names("blocks", ["block", "robot"])

    On, OnTable, GripperOpen, Holding, Clear = _get_predicates_by_names(
        "blocks", ["On", "OnTable", "GripperOpen", "Holding", "Clear"]
    )

    Pick, Stack, PutOnTable = _get_options_by_names(
        "blocks", ["Pick", "Stack", "PutOnTable"]
    )

    nsrts = set()

    # PickFromTable
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, robot]
    option_vars = [robot, block]
    option = Pick
    preconditions = {
        LiftedAtom(OnTable, [block]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot]),
    }
    add_effects = {LiftedAtom(Holding, [block])}
    delete_effects = {
        LiftedAtom(OnTable, [block]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot]),
    }

    def pick_sampler(
        state: State, rng: np.random.Generator, objs: Sequence[Object]
    ) -> Array:
        del state, rng, objs  # unused
        return np.zeros(3, dtype=np.float32)

    pickfromtable_nsrt = NSRT(
        "PickFromTable", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, pick_sampler)
    nsrts.add(pickfromtable_nsrt)

    # Unstack
    block = Variable("?block", block_type)
    otherblock = Variable("?otherblock", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, otherblock, robot]
    option_vars = [robot, block]
    option = Pick
    preconditions = {
        LiftedAtom(On, [block, otherblock]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot]),
    }
    add_effects = {
        LiftedAtom(Holding, [block]),
        LiftedAtom(Clear, [otherblock]),
    }
    delete_effects = {
        LiftedAtom(On, [block, otherblock]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot]),
    }
    unstack_nsrt = NSRT(
        "Unstack", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, pick_sampler)
    nsrts.add(unstack_nsrt)

    # Stack
    block = Variable("?block", block_type)
    otherblock = Variable("?otherblock", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, otherblock, robot]
    option_vars = [robot, otherblock]
    option = Stack
    preconditions = {
        LiftedAtom(Holding, [block]),
        LiftedAtom(Clear, [otherblock]),
    }
    add_effects = {
        LiftedAtom(On, [block, otherblock]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot]),
    }
    delete_effects = {
        LiftedAtom(Holding, [block]),
        LiftedAtom(Clear, [otherblock]),
    }

    def stack_sampler(
        state: State, rng: np.random.Generator, objs: Sequence[Object]
    ) -> Array:
        del state, rng, objs  # unused
        return np.array([0, 0, BlocksEnv.block_size], dtype=np.float32)

    stack_nsrt = NSRT(
        "Stack", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, stack_sampler)
    nsrts.add(stack_nsrt)

    # PutOnTable
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, robot]
    option_vars = [robot]
    option = PutOnTable
    preconditions = {LiftedAtom(Holding, [block])}
    add_effects = {
        LiftedAtom(OnTable, [block]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot]),
    }
    delete_effects = {LiftedAtom(Holding, [block])}

    def putontable_sampler(
        state: State, rng: np.random.Generator, objs: Sequence[Object]
    ) -> Array:
        del state, objs  # unused
        x = rng.uniform()
        y = rng.uniform()
        return np.array([x, y], dtype=np.float32)

    putontable_nsrt = NSRT(
        "PutOnTable", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, putontable_sampler)
    nsrts.add(putontable_nsrt)

    return nsrts


def _get_painting_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for PaintingEnv.
    """
    obj_type, box_type, lid_type, shelf_type, robot_type = \
        _get_types_by_names("painting", ["obj", "box", "lid", "shelf", "robot"])

    (InBox, InShelf, IsBoxColor, IsShelfColor, GripperOpen, OnTable,
     HoldingTop, HoldingSide, Holding, IsWet, IsDry, IsDirty, IsClean) = \
         _get_predicates_by_names(
             "painting", ["InBox", "InShelf", "IsBoxColor", "IsShelfColor",
                          "GripperOpen", "OnTable", "HoldingTop", "HoldingSide",
                          "Holding", "IsWet", "IsDry", "IsDirty", "IsClean"])

    Pick, Wash, Dry, Paint, Place, OpenLid = _get_options_by_names(
        "painting", ["Pick", "Wash", "Dry", "Paint", "Place", "OpenLid"])

    nsrts = set()

    # PickFromTop
    obj = Variable("?obj", obj_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, robot]
    option_vars = [robot, obj]
    option = Pick
    preconditions = {LiftedAtom(GripperOpen, [robot]),
                     LiftedAtom(OnTable, [obj])}
    add_effects = {LiftedAtom(Holding, [obj]),
                   LiftedAtom(HoldingTop, [robot])}
    delete_effects = {LiftedAtom(GripperOpen, [robot])}
    def pickfromtop_sampler(state: State, rng: np.random.Generator,
                            objs: Sequence[Object]) -> Array:
        del state, rng, objs  # unused
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    pickfromtop_nsrt = NSRT(
        "PickFromTop", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, pickfromtop_sampler)
    nsrts.add(pickfromtop_nsrt)

    # PickFromSide
    obj = Variable("?obj", obj_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, robot]
    option_vars = [robot, obj]
    option = Pick
    preconditions = {LiftedAtom(GripperOpen, [robot]),
                     LiftedAtom(OnTable, [obj])}
    add_effects = {LiftedAtom(Holding, [obj]),
                   LiftedAtom(HoldingSide, [robot])}
    delete_effects = {LiftedAtom(GripperOpen, [robot])}
    def pickfromside_sampler(state: State, rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
        del state, rng, objs  # unused
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    pickfromside_nsrt = NSRT(
        "PickFromSide", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, pickfromside_sampler)
    nsrts.add(pickfromside_nsrt)

    # Wash
    obj = Variable("?obj", obj_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, robot]
    option_vars = [robot]
    option = Wash
    preconditions = {LiftedAtom(Holding, [obj]),
                     LiftedAtom(IsDry, [obj]),
                     LiftedAtom(IsDirty, [obj])}
    add_effects = {LiftedAtom(IsWet, [obj]),
                   LiftedAtom(IsClean, [obj])}
    delete_effects = {LiftedAtom(IsDry, [obj]),
                      LiftedAtom(IsDirty, [obj])}
    def wash_sampler(state: State, rng: np.random.Generator,
                     objs: Sequence[Object]) -> Array:
        del state, rng, objs  # unused
        return np.array([1.0], dtype=np.float32)
    wash_nsrt = NSRT(
        "Wash", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, wash_sampler)
    nsrts.add(wash_nsrt)

    # Dry
    obj = Variable("?obj", obj_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, robot]
    option_vars = [robot]
    option = Dry
    preconditions = {LiftedAtom(Holding, [obj]),
                     LiftedAtom(IsWet, [obj])}
    add_effects = {LiftedAtom(IsDry, [obj])}
    delete_effects = {LiftedAtom(IsWet, [obj])}
    def dry_sampler(state: State, rng: np.random.Generator,
                    objs: Sequence[Object]) -> Array:
        del state, rng, objs  # unused
        return np.array([1.0], dtype=np.float32)
    dry_nsrt = NSRT(
        "Dry", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, dry_sampler)
    nsrts.add(dry_nsrt)

    # PaintToBox
    obj = Variable("?obj", obj_type)
    box = Variable("?box", box_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, box, robot]
    option_vars = [robot]
    option = Paint
    preconditions = {LiftedAtom(Holding, [obj]),
                     LiftedAtom(IsDry, [obj]),
                     LiftedAtom(IsClean, [obj])}
    add_effects = {LiftedAtom(IsBoxColor, [obj, box])}
    delete_effects = set()
    def painttobox_sampler(state: State, rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del rng  # unused
        box_color = state.get(objs[1], "color")
        return np.array([box_color], dtype=np.float32)
    painttobox_nsrt = NSRT(
        "PaintToBox", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, painttobox_sampler)
    nsrts.add(painttobox_nsrt)

    # PaintToShelf
    obj = Variable("?obj", obj_type)
    shelf = Variable("?shelf", shelf_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, shelf, robot]
    option_vars = [robot]
    option = Paint
    preconditions = {LiftedAtom(Holding, [obj]),
                     LiftedAtom(IsDry, [obj]),
                     LiftedAtom(IsClean, [obj])}
    add_effects = {LiftedAtom(IsShelfColor, [obj, shelf])}
    delete_effects = set()
    def painttoshelf_sampler(state: State, rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
        del rng  # unused
        shelf_color = state.get(objs[1], "color")
        return np.array([shelf_color], dtype=np.float32)
    painttoshelf_nsrt = NSRT(
        "PaintToShelf", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, painttoshelf_sampler)
    nsrts.add(painttoshelf_nsrt)

    # PlaceInBox
    obj = Variable("?obj", obj_type)
    box = Variable("?box", box_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, box, robot]
    option_vars = [robot]
    option = Place
    preconditions = {LiftedAtom(Holding, [obj]),
                     LiftedAtom(HoldingTop, [robot])}
    add_effects = {LiftedAtom(InBox, [obj, box]),
                   LiftedAtom(GripperOpen, [robot])}
    delete_effects = {LiftedAtom(HoldingTop, [robot]),
                      LiftedAtom(Holding, [obj]),
                      LiftedAtom(OnTable, [obj])}
    def placeinbox_sampler(state: State, rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        x = state.get(objs[0], "pose_x")
        y = rng.uniform(PaintingEnv.box_lb, PaintingEnv.box_ub)
        z = state.get(objs[0], "pose_z")
        return np.array([x, y, z], dtype=np.float32)
    placeinbox_nsrt = NSRT(
        "PlaceInBox", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, placeinbox_sampler)
    nsrts.add(placeinbox_nsrt)

    # PlaceInShelf
    obj = Variable("?obj", obj_type)
    shelf = Variable("?shelf", shelf_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, shelf, robot]
    option_vars = [robot]
    option = Place
    preconditions = {LiftedAtom(Holding, [obj]),
                     LiftedAtom(HoldingSide, [robot])}
    add_effects = {LiftedAtom(InShelf, [obj, shelf]),
                   LiftedAtom(GripperOpen, [robot])}
    delete_effects = {LiftedAtom(HoldingSide, [robot]),
                      LiftedAtom(Holding, [obj]),
                      LiftedAtom(OnTable, [obj])}
    def placeinshelf_sampler(state: State, rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
        x = state.get(objs[0], "pose_x")
        y = rng.uniform(PaintingEnv.shelf_lb, PaintingEnv.shelf_ub)
        z = state.get(objs[0], "pose_z")
        return np.array([x, y, z], dtype=np.float32)
    placeinshelf_nsrt = NSRT(
        "PlaceInShelf", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, placeinshelf_sampler)
    nsrts.add(placeinshelf_nsrt)

    # OpenLid
    lid = Variable("?lid", lid_type)
    robot = Variable("?robot", robot_type)
    parameters = [lid, robot]
    option_vars = [robot, lid]
    option = OpenLid
    preconditions = {LiftedAtom(GripperOpen, [robot])}
    add_effects = set()
    delete_effects = set()
    def openlid_sampler(state: State, rng: np.random.Generator,
                        objs: Sequence[Object]) -> Array:
        del state, rng, objs  # unused
        return np.array([], dtype=np.float32)
    openlid_nsrt = NSRT(
        "OpenLid", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, openlid_sampler)
    nsrts.add(openlid_nsrt)

    return nsrts

def _get_playroom_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for Playroom Env.
    """
    block_type, robot_type, door_type, dial_type, region_type = \
        _get_types_by_names(CFG.env,
            ["block", "robot", "door", "dial", "region"])

    On, OnTable, GripperOpen, Holding, Clear, NextToTable, NextToDoor, \
        NextToDial, InRegion, Borders, Connects, IsBoringRoom, IsPlayroom, \
        IsBoringRoomDoor, IsPlayroomDoor, DoorOpen, DoorClosed, LightOn, \
        LightOff = \
            _get_predicates_by_names(
            "playroom", ["On", "OnTable", "GripperOpen", "Holding", "Clear",
            "NextToTable", "NextToDoor", "NextToDial", "InRegion", "Borders",
            "Connects", "IsBoringRoom", "IsPlayroom", "IsBoringRoomDoor",
            "IsPlayroomDoor", "DoorOpen", "DoorClosed", "LightOn", "LightOff"])

    Pick, Stack, PutOnTable, Move, OpenDoor, CloseDoor, TurnOnDial, \
        TurnOffDial = _get_options_by_names("playroom",
        ["Pick", "Stack", "PutOnTable", "Move", "OpenDoor", "CloseDoor",
        "TurnOnDial", "TurnOffDial"])

    nsrts = set()

    # PickFromTable
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [robot, block]
    option_vars = [robot, block]
    option = Pick
    preconditions = {LiftedAtom(OnTable, [block]),
                     LiftedAtom(Clear, [block]),
                     LiftedAtom(GripperOpen, [robot]),
                     LiftedAtom(NextToTable, [robot])}
    add_effects = {LiftedAtom(Holding, [block])}
    delete_effects = {LiftedAtom(OnTable, [block]),
                      LiftedAtom(Clear, [block]),
                      LiftedAtom(GripperOpen, [robot])}
    def pickfromtable_sampler(state: State, rng: np.random.Generator,
                              objs: Sequence[Object]) -> Array:
        del rng  # unused
        assert len(objs) == 2
        _, block = objs
        assert block.is_instance(block_type)
        # find rotation of robot that faces the table
        x, y = state.get(block, "pose_x"), state.get(block, "pose_y")
        cls = PlayroomEnv
        table_x = (cls.table_x_lb+cls.table_x_ub)/2
        table_y = (cls.table_y_lb+cls.table_y_ub)/2
        rotation = np.arctan2(table_y-y, table_x-x) / np.pi
        return np.array([0, 0, 0, rotation], dtype=np.float32)
    pickfromtable_nsrt = NSRT(
        "PickFromTable", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, pickfromtable_sampler)
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
                     LiftedAtom(GripperOpen, [robot]),
                     LiftedAtom(NextToTable, [robot])}
    add_effects = {LiftedAtom(Holding, [block]),
                   LiftedAtom(Clear, [otherblock])}
    delete_effects = {LiftedAtom(On, [block, otherblock]),
                      LiftedAtom(Clear, [block]),
                      LiftedAtom(GripperOpen, [robot])}
    def unstack_sampler(state: State, rng: np.random.Generator,
                        objs: Sequence[Object]) -> Array:
        del rng  # unused
        assert len(objs) == 3
        block, _, _ = objs
        assert block.is_instance(block_type)
        # find rotation of robot that faces the table
        x, y = state.get(block, "pose_x"), state.get(block, "pose_y")
        cls = PlayroomEnv
        table_x = (cls.table_x_lb+cls.table_x_ub)/2
        table_y = (cls.table_y_lb+cls.table_y_ub)/2
        rotation = np.arctan2(table_y-y, table_x-x) / np.pi
        return np.array([0, 0, 0, rotation], dtype=np.float32)
    unstack_nsrt = NSRT(
        "Unstack", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, unstack_sampler)
    nsrts.add(unstack_nsrt)

    # Stack
    block = Variable("?block", block_type)
    otherblock = Variable("?otherblock", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, otherblock, robot]
    option_vars = [robot, otherblock]
    option = Stack
    preconditions = {LiftedAtom(Holding, [block]),
                     LiftedAtom(Clear, [otherblock]),
                     LiftedAtom(NextToTable, [robot])}
    add_effects = {LiftedAtom(On, [block, otherblock]),
                   LiftedAtom(Clear, [block]),
                   LiftedAtom(GripperOpen, [robot])}
    delete_effects = {LiftedAtom(Holding, [block]),
                      LiftedAtom(Clear, [otherblock])}
    def stack_sampler(state: State, rng: np.random.Generator,
                      objs: Sequence[Object]) -> Array:
        del rng  # unused
        assert len(objs) == 3
        _, otherblock, _ = objs
        assert otherblock.is_instance(block_type)
        # find rotation of robot that faces the table
        x, y = state.get(otherblock, "pose_x"), state.get(otherblock, "pose_y")
        cls = PlayroomEnv
        table_x = (cls.table_x_lb+cls.table_x_ub)/2
        table_y = (cls.table_y_lb+cls.table_y_ub)/2
        rotation = np.arctan2(table_y-y, table_x-x) / np.pi
        return np.array([0, 0, PlayroomEnv.block_size, rotation],
                        dtype=np.float32)
    stack_nsrt = NSRT(
        "Stack", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, stack_sampler)
    nsrts.add(stack_nsrt)

    # PutOnTable
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, robot]
    option_vars = [robot]
    option = PutOnTable
    preconditions = {LiftedAtom(Holding, [block]),
                     LiftedAtom(NextToTable, [robot])}
    add_effects = {LiftedAtom(OnTable, [block]),
                   LiftedAtom(Clear, [block]),
                   LiftedAtom(GripperOpen, [robot])}
    delete_effects = {LiftedAtom(Holding, [block])}
    def putontable_sampler(state: State, rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del state, objs  # unused
        x = rng.uniform()
        y = rng.uniform()
        # find rotation of robot that faces the table
        cls = PlayroomEnv
        table_x = (cls.table_x_lb+cls.table_x_ub)/2
        table_y = (cls.table_y_lb+cls.table_y_ub)/2
        rotation = np.arctan2(table_y-y, table_x-x) / np.pi
        return np.array([x, y, rotation], dtype=np.float32)
    putontable_nsrt = NSRT(
        "PutOnTable", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, putontable_sampler)
    nsrts.add(putontable_nsrt)

    # AdvanceThroughDoor
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    from_region = Variable("?from", region_type)
    to_region = Variable("?to", region_type)
    parameters = [robot, door, from_region, to_region]
    option_vars = [robot]
    option = Move
    preconditions = {LiftedAtom(InRegion, [robot, from_region]),
                     LiftedAtom(Connects, [door, from_region, to_region]),
                     LiftedAtom(DoorOpen, [door]),
                     LiftedAtom(NextToDoor, [robot, door])}
    add_effects = {LiftedAtom(InRegion, [robot, to_region])}
    delete_effects = {LiftedAtom(InRegion, [robot, from_region])}
    def advancethroughdoor_sampler(state: State, rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
        del rng  # unused
        assert len(objs) == 4
        robot, door, _, _ = objs
        assert robot.is_instance(robot_type)
        assert door.is_instance(door_type)
        x = state.get(robot, "pose_x")
        door_x, door_y = state.get(door, "pose_x"), state.get(door, "pose_y")
        final_x = door_x+0.2 if x < door_x else door_x-0.2
        rotation = 0.0 if x < door_x else -1.0
        return np.array([final_x, door_y, rotation], dtype=np.float32)
    advancethroughdoor_nsrt = NSRT(
        "AdvanceThroughDoor", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, advancethroughdoor_sampler)
    nsrts.add(advancethroughdoor_nsrt)

    # MoveTableToDoor
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    region = Variable("?region", region_type)
    parameters = [robot, door, region]
    option_vars = [robot]
    option = Move
    preconditions = {LiftedAtom(NextToTable, [robot]),
                     LiftedAtom(IsBoringRoomDoor, [door])}
    add_effects = {LiftedAtom(NextToDoor, [robot, door])}
    delete_effects = {LiftedAtom(NextToTable, [robot])}
    def movetabletodoor_sampler(state: State, rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
        del rng  # unused
        assert len(objs) == 3
        _, door, _ = objs
        assert door.is_instance(door_type)
        x, y = state.get(door, "pose_x")-0.2, state.get(door, "pose_y")
        return np.array([x, y, 0.0], dtype=np.float32)
    movetabletodoor_nsrt = NSRT(
        "MoveTableToDoor", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, movetabletodoor_sampler)
    nsrts.add(movetabletodoor_nsrt)

    # MoveDoorToTable
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    region = Variable("?region", region_type)
    parameters = [robot, door, region]
    option_vars = [robot]
    option = Move
    preconditions = {LiftedAtom(IsBoringRoom, [region]),
                     LiftedAtom(InRegion, [robot, region]),
                     LiftedAtom(NextToDoor, [robot, door]),
                     LiftedAtom(IsBoringRoomDoor, [door])}
    add_effects = {LiftedAtom(NextToTable, [robot])}
    delete_effects = {LiftedAtom(NextToDoor, [robot, door])}
    def movedoortotable_sampler(state: State, rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
        del state, rng, objs  # unused
        x, y = PlayroomEnv.table_x_ub, PlayroomEnv.table_y_ub
        return np.array([x, y, -0.75], dtype=np.float32)
    movedoortotable_nsrt = NSRT(
        "MoveDoorToTable", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, movedoortotable_sampler)
    nsrts.add(movedoortotable_nsrt)

    # MoveDoorToDoor
    robot = Variable("?robot", robot_type)
    fromdoor = Variable("?fromdoor", door_type)
    todoor = Variable("?todoor", door_type)
    region = Variable("?region", region_type)
    parameters = [robot, fromdoor, todoor, region]
    option_vars = [robot]
    option = Move
    preconditions = {LiftedAtom(Borders, [fromdoor, region, todoor]),
                     LiftedAtom(InRegion, [robot, region]),
                     LiftedAtom(NextToDoor, [robot, fromdoor])}
    add_effects = {LiftedAtom(NextToDoor, [robot, todoor])}
    delete_effects = {LiftedAtom(NextToDoor, [robot, fromdoor])}
    def movedoortodoor_sampler(state: State, rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
        del rng  # unused
        assert len(objs) == 4
        _, fromdoor, todoor, _ = objs
        assert fromdoor.is_instance(door_type)
        assert todoor.is_instance(door_type)
        to_x, to_y = state.get(todoor, "pose_x"), state.get(todoor, "pose_y")
        from_x = state.get(fromdoor, "pose_x")
        rotation = 0.0 if from_x < to_x else -1.0
        x = to_x-0.1 if from_x < to_x else to_x+0.1
        return np.array([x, to_y, rotation], dtype=np.float32)
    movedoortodoor_nsrt = NSRT(
        "MoveDoorToDoor", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, movedoortodoor_sampler)
    nsrts.add(movedoortodoor_nsrt)

    # MoveDoorToDial
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    dial = Variable("?dial", dial_type)
    region = Variable("?region", region_type)
    parameters = [robot, door, dial, region]
    option_vars = [robot]
    option = Move
    preconditions = {LiftedAtom(IsPlayroom, [region]),
                     LiftedAtom(InRegion, [robot, region]),
                     LiftedAtom(IsPlayroomDoor, [door]),
                     LiftedAtom(NextToDoor, [robot, door])}
    add_effects = {LiftedAtom(NextToDial, [robot, dial])}
    delete_effects = {LiftedAtom(NextToDoor, [robot, door])}
    def movedoortodial_sampler(state: State, rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
        del rng  # unused
        assert len(objs) == 4
        _, _, dial, _ = objs
        assert dial.is_instance(dial_type)
        dial_x, dial_y = state.get(dial, "pose_x"), state.get(dial, "pose_y")
        return np.array([dial_x-0.2, dial_y, -1.0], dtype=np.float32)
    movedoortodial_nsrt = NSRT(
        "MoveDoorToDial", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, movedoortodial_sampler)
    nsrts.add(movedoortodial_nsrt)

    # MoveDialToDoor
    robot = Variable("?robot", robot_type)
    dial = Variable("?dial", dial_type)
    door = Variable("?door", door_type)
    region = Variable("?region", region_type)
    parameters = [robot, dial, door, region]
    option_vars = [robot]
    option = Move
    preconditions = {LiftedAtom(IsPlayroomDoor, [door]),
                     LiftedAtom(NextToDial, [robot, dial])}
    add_effects = {LiftedAtom(NextToDoor, [robot, door])}
    delete_effects = {LiftedAtom(NextToDial, [robot, dial])}
    def movedialtodoor_sampler(state: State, rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
        del rng  # unused
        assert len(objs) == 4
        _, _, door, _ = objs
        assert door.is_instance(door_type)
        x, y = state.get(door, "pose_x"), state.get(door, "pose_y")
        return np.array([x+0.1, y, -1.0], dtype=np.float32)
    movedialtodoor_nsrt = NSRT(
        "MoveDialToDoor", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, movedialtodoor_sampler)
    nsrts.add(movedialtodoor_nsrt)

    # OpenDoor
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    parameters = [robot, door]
    option_vars = [door]
    option = OpenDoor
    preconditions = {LiftedAtom(NextToDoor, [robot, door]),
                     LiftedAtom(DoorClosed, [door]),
                     LiftedAtom(GripperOpen, [robot])}
    add_effects = {LiftedAtom(DoorOpen, [door])}
    delete_effects = {LiftedAtom(DoorClosed, [door])}
    def toggledoor_sampler(state: State, rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del rng  # unused
        assert len(objs) == 2
        robot, door = objs
        assert robot.is_instance(robot_type)
        assert door.is_instance(door_type)
        x, door_x = state.get(robot, "pose_x"), state.get(door, "pose_x")
        rotation = 0.0 if x < door_x else -1.0
        dx = -0.2 if x < door_x else 0.2
        return np.array([dx, 0, 0, rotation], dtype=np.float32)
    opendoor_nsrt = NSRT(
        "OpenDoor", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, toggledoor_sampler)
    nsrts.add(opendoor_nsrt)

    # CloseDoor
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    parameters = [robot, door]
    option_vars = [door]
    option = CloseDoor
    preconditions = {LiftedAtom(NextToDoor, [robot, door]),
                     LiftedAtom(DoorOpen, [door]),
                     LiftedAtom(GripperOpen, [robot])}
    add_effects = {LiftedAtom(DoorClosed, [door])}
    delete_effects = {LiftedAtom(DoorOpen, [door])}
    closedoor_nsrt = NSRT(
        "CloseDoor", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, toggledoor_sampler)
    nsrts.add(closedoor_nsrt)

    # TurnOnDial
    robot = Variable("?robot", robot_type)
    dial = Variable("?dial", dial_type)
    parameters = [robot, dial]
    option_vars = [dial]
    option = TurnOnDial
    preconditions = {LiftedAtom(NextToDial, [robot, dial]),
                     LiftedAtom(LightOff, [dial]),
                     LiftedAtom(GripperOpen, [robot])}
    add_effects = {LiftedAtom(LightOn, [dial])}
    delete_effects = {LiftedAtom(LightOff, [dial])}
    def toggledial_sampler(state: State, rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del state, rng, objs  # unused
        return np.array([-0.2, 0, 0, 0.0], dtype=np.float32)
    turnondial_nsrt = NSRT(
        "TurnOnDial", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, toggledial_sampler)
    nsrts.add(turnondial_nsrt)

    # TurnOffDial
    robot = Variable("?robot", robot_type)
    dial = Variable("?dial", dial_type)
    parameters = [robot, dial]
    option_vars = [dial]
    option = TurnOffDial
    preconditions = {LiftedAtom(NextToDial, [robot, dial]),
                     LiftedAtom(LightOn, [dial]),
                     LiftedAtom(GripperOpen, [robot])}
    add_effects = {LiftedAtom(LightOff, [dial])}
    delete_effects = {LiftedAtom(LightOn, [dial])}
    turnoffdial_nsrt = NSRT(
        "TurnOffDial", parameters, preconditions, add_effects,
        delete_effects, set(), option, option_vars, toggledial_sampler)
    nsrts.add(turnoffdial_nsrt)

    return nsrts

def _get_repeated_nextto_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for RepeatedNextToEnv.
    """
    robot_type, dot_type = _get_types_by_names(CFG.env, ["robot", "dot"])

    NextTo, NextToNothing, Grasped = _get_predicates_by_names(
        CFG.env, ["NextTo", "NextToNothing", "Grasped"])

    Move, Grasp = _get_options_by_names(CFG.env, ["Move", "Grasp"])

    nsrts = set()

    # Move
    robot = Variable("?robot", robot_type)
    targetdot = Variable("?targetdot", dot_type)
    parameters = [robot, targetdot]
    option_vars = [robot, targetdot]
    option = Move
    preconditions : Set[LiftedAtom] = set()
    add_effects = {LiftedAtom(NextTo, [robot, targetdot])}
    delete_effects : Set[LiftedAtom] = set()
    # Moving could have us end up NextTo other objects. It could also
    # include NextToNothing as a delete effect.
    side_predicates = {NextTo, NextToNothing}
    move_nsrt = NSRT("Move", parameters, preconditions,
        add_effects, delete_effects, side_predicates, option, option_vars,
        lambda s, rng, o: np.zeros(1, dtype=np.float32))
    nsrts.add(move_nsrt)

    # Grasp
    robot = Variable("?robot", robot_type)
    targetdot = Variable("?targetdot", dot_type)
    parameters = [robot, targetdot]
    option_vars = [robot, targetdot]
    option = Grasp
    preconditions = {LiftedAtom(NextTo, [robot, targetdot])}
    add_effects = {LiftedAtom(Grasped, [robot, targetdot])}
    delete_effects = {LiftedAtom(NextTo, [robot, targetdot])}
    # After grasping, it's possible that you could end up NextToNothing,
    # but it's also possible that you remain next to something else.
    # Note that NextTo isn't a side predicate here because it's not
    # something we'd be unsure about for any object. For every object we
    # are NextTo but did not grasp, we will stay NextTo it.
    side_predicates = {NextToNothing}
    grasp_nsrt = NSRT("Grasp", parameters, preconditions,
        add_effects, delete_effects, side_predicates, option, option_vars,
        lambda s, rng, o: np.zeros(0, dtype=np.float32))
    nsrts.add(grasp_nsrt)

    return nsrts


def _get_behavior_gt_nsrts() -> Set[NSRT]:
    """Create ground truth nsrts for BehaviorEnv."""
    env = get_env_instance("behavior")

    type_name_to_type = {t.name: t for t in env.types}
    pred_name_to_pred = {p.name: p for p in env.predicates}

    def _get_lifted_atom(
        base_pred_name: str, objects: Sequence[Variable]
    ) -> LiftedAtom:
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
            nextto_nothing = _get_lifted_atom("reachable-nothing", [agent_obj])
            parameters = [agent_obj, target_obj]
            option_vars = [target_obj]
            preconditions = {nextto_nothing}
            add_effects = {
                _get_lifted_atom("reachable", [target_obj, agent_obj])
            }
            delete_effects = {nextto_nothing}
            nsrt = NSRT(
                f"{option.name}-{next(op_name_count)}",
                parameters,
                preconditions,
                add_effects,
                delete_effects,
                option,
                option_vars,
                lambda s, r, o: navigate_to_param_sampler(r,\
                    [env.object_to_ig_object(o_i) for o_i in o], # type: ignore
                )
            )
            nsrts.add(nsrt)

            # Navigate to while nextto something
            for origin_obj_type in sorted(env.types):
                origin_obj = Variable("?origin", origin_obj_type)
                origin_next_to = _get_lifted_atom(
                    "reachable", [origin_obj, agent_obj]
                )
                targ_next_to = _get_lifted_atom(
                    "reachable", [target_obj, agent_obj]
                )
                parameters = [origin_obj, agent_obj, target_obj]
                option_vars = [target_obj]
                preconditions = {origin_next_to}
                add_effects = {targ_next_to}
                delete_effects = {origin_next_to}
                nsrt = NSRT(
                    f"{option.name}-{next(op_name_count)}",
                    parameters,
                    preconditions,
                    add_effects,
                    delete_effects,
                    option,
                    option_vars,
                    lambda s, r, o: navigate_to_param_sampler(r,\
                    [env.object_to_ig_object(o_i) for o_i in o] # type: ignore
                    )
                )
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
                targ_next_to = _get_lifted_atom(
                    "reachable", [target_obj, agent_obj]
                )
                # targ_graspable = _get_lifted_atom(
                #    "graspable", [target_obj]
                # )
                targ_holding = _get_lifted_atom("holding", [target_obj])
                # preconditions = {handempty, targ_next_to, targ_graspable}
                preconditions = {handempty, targ_next_to}
                add_effects = {targ_holding}
                delete_effects = {handempty}
                nsrt = NSRT(
                    f"{option.name}-{next(op_name_count)}",
                    parameters,
                    preconditions,
                    add_effects,
                    delete_effects,
                    option,
                    option_vars,
                    lambda s, r, o: grasp_obj_param_sampler(r),
                )
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
                option_vars = [surf_obj]
                handempty = _get_lifted_atom("handempty", [])
                held_holding = _get_lifted_atom("holding", [held_obj])
                surf_next_to = _get_lifted_atom(
                    "reachable", [surf_obj, agent_obj]
                )
                ontop = _get_lifted_atom("ontop", [held_obj, surf_obj])
                preconditions = {held_holding, surf_next_to}
                add_effects = {ontop, handempty}
                delete_effects = {held_holding}
                nsrt = NSRT(
                  f"{option.name}-{next(op_name_count)}",
                  parameters,
                  preconditions,
                  add_effects,
                  delete_effects,
                  option,
                  option_vars,
                  lambda s, r, o: place_ontop_obj_pos_sampler(  # type: ignore
                    env.behavior_env,  # type: ignore
                    [env.object_to_ig_object(o_i) for o_i in o], # type: ignore
                    rng=r,
                  ),
                )
                nsrts.add(nsrt)

        else:
            raise ValueError(f"Unexpected base option name: {base_option_name}")

    return nsrts
