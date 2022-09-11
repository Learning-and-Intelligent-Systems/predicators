"""Definitions of ground truth NSRTs for all environments."""

import itertools
import logging
from typing import List, Sequence, Set, Union, cast

import numpy as np
from numpy.random._generator import Generator

from predicators.behavior_utils.behavior_utils import OPENABLE_OBJECT_TYPES, \
    PICK_PLACE_OBJECT_TYPES, PLACE_INTO_SURFACE_OBJECT_TYPES, \
    PLACE_ONTOP_SURFACE_OBJECT_TYPES, check_hand_end_pose, \
    check_nav_end_pose, load_checkpoint_state
from predicators.envs import get_or_create_env
from predicators.envs.behavior import BehaviorEnv
from predicators.envs.doors import DoorsEnv
from predicators.envs.painting import PaintingEnv
from predicators.envs.pddl_env import _PDDLEnv
from predicators.envs.playroom import PlayroomEnv
from predicators.envs.repeated_nextto_painting import RepeatedNextToPaintingEnv
from predicators.envs.satellites import SatellitesEnv
from predicators.envs.tools import ToolsEnv
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler

try:  # pragma: no cover
    from igibson.external.pybullet_tools.utils import get_aabb, get_aabb_extent
    from igibson.object_states.on_floor import \
        RoomFloor  # pylint: disable=unused-import
    from igibson.objects.articulated_object import \
        URDFObject  # pylint: disable=unused-import
    from igibson.utils import sampling_utils
except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
    pass


def get_gt_nsrts(predicates: Set[Predicate],
                 options: Set[ParameterizedOption]) -> Set[NSRT]:
    """Create ground truth NSRTs for an env."""
    if CFG.env in ("cover", "cover_hierarchical_types", "cover_typed_options",
                   "cover_regrasp", "cover_multistep_options",
                   "pybullet_cover"):
        nsrts = _get_cover_gt_nsrts()
    elif CFG.env == "cluttered_table":
        nsrts = _get_cluttered_table_gt_nsrts()
    elif CFG.env == "cluttered_table_place":
        nsrts = _get_cluttered_table_gt_nsrts(with_place=True)
    elif CFG.env in ("blocks", "pybullet_blocks"):
        nsrts = _get_blocks_gt_nsrts()
    elif CFG.env == "behavior":
        nsrts = _get_behavior_gt_nsrts()  # pragma: no cover
    elif CFG.env in ("painting", "repeated_nextto_painting"):
        nsrts = _get_painting_gt_nsrts()
    elif CFG.env == "tools":
        nsrts = _get_tools_gt_nsrts()
    elif CFG.env == "playroom":
        nsrts = _get_playroom_gt_nsrts()
    elif CFG.env == "repeated_nextto":
        nsrts = _get_repeated_nextto_gt_nsrts(CFG.env)
    elif CFG.env == "repeated_nextto_single_option":
        nsrts = _get_repeated_nextto_single_option_gt_nsrts()
    elif CFG.env == "screws":
        nsrts = _get_screws_gt_nsrts()
    elif CFG.env.startswith("pddl_"):
        nsrts = _get_pddl_env_gt_nsrts(CFG.env)
    elif CFG.env == "touch_point":
        nsrts = _get_touch_point_gt_nsrts()
    elif CFG.env == "stick_button":
        nsrts = _get_stick_button_gt_nsrts()
    elif CFG.env == "doors":
        nsrts = _get_doors_gt_nsrts()
    elif CFG.env == "coffee":
        nsrts = _get_coffee_gt_nsrts()
    elif CFG.env in ("satellites", "satellites_simple"):
        nsrts = _get_satellites_gt_nsrts()
    else:
        raise NotImplementedError("Ground truth NSRTs not implemented")
    # Filter out excluded predicates from NSRTs, and filter out NSRTs whose
    # options are excluded.
    final_nsrts = set()
    for nsrt in nsrts:
        if nsrt.option not in options:
            continue
        nsrt = nsrt.filter_predicates(predicates)
        final_nsrts.add(nsrt)
    return final_nsrts


def _get_from_env_by_names(env_name: str, names: Sequence[str],
                           env_attr: str) -> List:
    """Helper for loading types, predicates, and options by name."""
    env = get_or_create_env(env_name)
    name_to_env_obj = {}
    for o in getattr(env, env_attr):
        name_to_env_obj[o.name] = o
    assert set(name_to_env_obj).issuperset(set(names))
    return [name_to_env_obj[name] for name in names]


def _get_types_by_names(env_name: str, names: Sequence[str]) -> List[Type]:
    """Load types from an env given their names."""
    return _get_from_env_by_names(env_name, names, "types")


def _get_predicates_by_names(env_name: str,
                             names: Sequence[str]) -> List[Predicate]:
    """Load predicates from an env given their names."""
    return _get_from_env_by_names(env_name, names, "predicates")


def _get_options_by_names(env_name: str,
                          names: Sequence[str]) -> List[ParameterizedOption]:
    """Load parameterized options from an env given their names."""
    return _get_from_env_by_names(env_name, names, "options")


def _get_cover_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for CoverEnv or environments that inherit from
    CoverEnv."""
    # Types
    block_type, target_type, robot_type = _get_types_by_names(
        CFG.env, ["block", "target", "robot"])

    # Objects
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    target = Variable("?target", target_type)

    # Predicates
    IsBlock, IsTarget, Covers, HandEmpty, Holding = \
        _get_predicates_by_names(CFG.env, ["IsBlock", "IsTarget", "Covers",
                                           "HandEmpty", "Holding"])

    # Options
    if CFG.env in ("cover", "pybullet_cover", "cover_hierarchical_types",
                   "cover_regrasp"):
        PickPlace, = _get_options_by_names(CFG.env, ["PickPlace"])
    elif CFG.env in ("cover_typed_options", "cover_multistep_options"):
        Pick, Place = _get_options_by_names(CFG.env, ["Pick", "Place"])

    nsrts = set()

    # Pick
    parameters = [block]
    holding_predicate_args = [block]
    if CFG.env == "cover_multistep_options":
        parameters.append(robot)
        holding_predicate_args.append(robot)
    preconditions = {LiftedAtom(IsBlock, [block]), LiftedAtom(HandEmpty, [])}
    add_effects = {LiftedAtom(Holding, holding_predicate_args)}
    delete_effects = {LiftedAtom(HandEmpty, [])}

    if CFG.env in ("cover", "pybullet_cover", "cover_hierarchical_types",
                   "cover_regrasp"):
        option = PickPlace
        option_vars = []
    elif CFG.env == "cover_typed_options":
        option = Pick
        option_vars = [block]
    elif CFG.env == "cover_multistep_options":
        option = Pick
        option_vars = [block, robot]

    if CFG.env == "cover_multistep_options":

        def pick_sampler(state: State, goal: Set[GroundAtom],
                         rng: np.random.Generator,
                         objs: Sequence[Object]) -> Array:
            # The only things that change are the block's grasp, and the
            # robot's grip, holding, x, and y.
            assert len(objs) == 2
            block, robot = objs
            assert block.is_instance(block_type)
            assert robot.is_instance(robot_type)
            bx, by = state.get(block, "x"), state.get(block, "y")
            rx, ry = state.get(robot, "x"), state.get(robot, "y")
            bw = state.get(block, "width")

            if CFG.cover_multistep_goal_conditioned_sampling:
                # Goal conditioned sampling currently assumes one goal.
                assert len(goal) == 1
                goal_atom = next(iter(goal))
                t = goal_atom.objects[1]
                tx, tw = state.get(t, "x"), state.get(t, "width")
                thr_found = False  # target hand region
                # Loop over objects in state to find target hand region,
                # whose center should overlap with the target.
                for obj in state.data:
                    if obj.type.name == "target_hand_region":
                        tlb = state.get(obj, "lb")
                        tub = state.get(obj, "ub")
                        tm = (tlb + tub) / 2  # midpoint of hand region
                        if tx - tw / 2 < tm < tx + tw / 2:
                            thr_found = True
                            break
                assert thr_found

            if CFG.cover_multistep_degenerate_oracle_samplers:
                desired_x = float(bx)
            elif CFG.cover_multistep_goal_conditioned_sampling:
                # Block position adjusted by target/ thr offset
                desired_x = bx + (tm - tx)
            else:
                desired_x = rng.uniform(bx - bw / 2, bx + bw / 2)
            # This option changes the grasp for the block from -1.0 to 1.0, so
            # the delta is 1.0 - (-1.0) = 2.0
            block_param = [2.0]
            # The grip changes from -1.0 to 1.0.
            # The holding changes from -1.0 to 1.0.
            # x, y, grip, holding
            robot_param = [desired_x - rx, by - ry, 2.0, 2.0]
            param = block_param + robot_param
            return np.array(param, dtype=np.float32)
    else:

        def pick_sampler(state: State, goal: Set[GroundAtom],
                         rng: np.random.Generator,
                         objs: Sequence[Object]) -> Array:
            del goal  # unused
            assert len(objs) == 1
            b = objs[0]
            assert b.is_instance(block_type)
            if CFG.env == "cover_typed_options":
                lb = float(-state.get(b, "width") / 2)
                ub = float(state.get(b, "width") / 2)
            elif CFG.env in ("cover", "pybullet_cover",
                             "cover_hierarchical_types", "cover_regrasp"):
                lb = float(state.get(b, "pose") - state.get(b, "width") / 2)
                lb = max(lb, 0.0)
                ub = float(state.get(b, "pose") + state.get(b, "width") / 2)
                ub = min(ub, 1.0)
            return np.array(rng.uniform(lb, ub, size=(1, )), dtype=np.float32)

    pick_nsrt = NSRT("Pick", parameters, preconditions, add_effects,
                     delete_effects, set(), option, option_vars, pick_sampler)
    nsrts.add(pick_nsrt)

    # Place (to Cover)
    parameters = [block, target]
    holding_predicate_args = [block]
    if CFG.env == "cover_multistep_options":
        parameters = [block, robot, target]
        holding_predicate_args.append(robot)
    preconditions = {
        LiftedAtom(IsBlock, [block]),
        LiftedAtom(IsTarget, [target]),
        LiftedAtom(Holding, holding_predicate_args)
    }
    add_effects = {
        LiftedAtom(HandEmpty, []),
        LiftedAtom(Covers, [block, target])
    }
    delete_effects = {LiftedAtom(Holding, holding_predicate_args)}
    if CFG.env == "cover_regrasp":
        Clear, = _get_predicates_by_names("cover_regrasp", ["Clear"])
        preconditions.add(LiftedAtom(Clear, [target]))
        delete_effects.add(LiftedAtom(Clear, [target]))

    if CFG.env in ("cover", "pybullet_cover", "cover_hierarchical_types",
                   "cover_regrasp"):
        option = PickPlace
        option_vars = []
    elif CFG.env == "cover_typed_options":
        option = Place
        option_vars = [target]
    elif CFG.env == "cover_multistep_options":
        option = Place
        option_vars = [block, robot, target]

    if CFG.env == "cover_multistep_options":

        def place_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:

            if CFG.cover_multistep_goal_conditioned_sampling:
                # Goal conditioned sampling currently assumes one goal.
                assert len(goal) == 1
                goal_atom = next(iter(goal))
                t = goal_atom.objects[1]
                tx, tw = state.get(t, "x"), state.get(t, "width")
                thr_found = False  # target hand region
                # Loop over objects in state to find target hand region,
                # whose center should overlap with the target.
                for obj in state.data:
                    if obj.type.name == "target_hand_region":
                        lb = state.get(obj, "lb")
                        ub = state.get(obj, "ub")
                        m = (lb + ub) / 2  # midpoint of hand region
                        if tx - tw / 2 < m < tx + tw / 2:
                            thr_found = True
                            break
                assert thr_found

            assert len(objs) == 3
            block, robot, target = objs
            assert block.is_instance(block_type)
            assert robot.is_instance(robot_type)
            assert target.is_instance(target_type)
            rx = state.get(robot, "x")
            tx, tw = state.get(target, "x"), state.get(target, "width")
            if CFG.cover_multistep_degenerate_oracle_samplers:
                desired_x = float(tx)
            elif CFG.cover_multistep_goal_conditioned_sampling:
                desired_x = m  # midpoint of hand region
            else:
                desired_x = rng.uniform(tx - tw / 2, tx + tw / 2)
            delta_x = desired_x - rx
            # This option changes the grasp for the block from 1.0 to -1.0, so
            # the delta is -1.0 - 1.0 = -2.0.
            # x, grasp
            block_param = [delta_x, -2.0]
            # The grip changes from 1.0 to -1.0.
            # The holding changes from 1.0 to -1.0.
            # x, grip, holding
            robot_param = [delta_x, -2.0, -2.0]
            param = block_param + robot_param
            return np.array(param, dtype=np.float32)
    else:

        def place_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del goal  # unused
            assert len(objs) == 2
            t = objs[-1]
            assert t.is_instance(target_type)
            lb = float(state.get(t, "pose") - state.get(t, "width") / 10)
            lb = max(lb, 0.0)
            ub = float(state.get(t, "pose") + state.get(t, "width") / 10)
            ub = min(ub, 1.0)
            return np.array(rng.uniform(lb, ub, size=(1, )), dtype=np.float32)

    place_nsrt = NSRT("Place",
                      parameters, preconditions, add_effects, delete_effects,
                      set(), option, option_vars, place_sampler)
    nsrts.add(place_nsrt)

    # Place (not on any target)
    if CFG.env == "cover_regrasp":
        parameters = [block]
        preconditions = {
            LiftedAtom(IsBlock, [block]),
            LiftedAtom(Holding, [block])
        }
        add_effects = {
            LiftedAtom(HandEmpty, []),
        }
        delete_effects = {LiftedAtom(Holding, [block])}
        option = PickPlace
        option_vars = []

        def place_on_table_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
            # Always at the current location.
            del goal, rng  # this sampler is deterministic
            assert len(objs) == 1
            held_obj = objs[0]
            x = state.get(held_obj, "pose") + state.get(held_obj, "grasp")
            return np.array([x], dtype=np.float32)

        place_on_table_nsrt = NSRT("PlaceOnTable", parameters,
                                   preconditions, add_effects, delete_effects,
                                   set(), option, option_vars,
                                   place_on_table_sampler)
        nsrts.add(place_on_table_nsrt)

    return nsrts


def _get_cluttered_table_gt_nsrts(with_place: bool = False) -> Set[NSRT]:
    """Create ground truth NSRTs for ClutteredTableEnv."""
    can_type, = _get_types_by_names("cluttered_table", ["can"])

    HandEmpty, Holding, Untrashed = _get_predicates_by_names(
        "cluttered_table", ["HandEmpty", "Holding", "Untrashed"])

    if with_place:
        Grasp, Place = _get_options_by_names("cluttered_table_place",
                                             ["Grasp", "Place"])
    else:
        Grasp, Dump = _get_options_by_names("cluttered_table",
                                            ["Grasp", "Dump"])

    nsrts = set()

    # Grasp
    can = Variable("?can", can_type)
    parameters = [can]
    option_vars = [can]
    option = Grasp
    preconditions = {LiftedAtom(HandEmpty, []), LiftedAtom(Untrashed, [can])}
    add_effects = {LiftedAtom(Holding, [can])}
    delete_effects = {LiftedAtom(HandEmpty, [])}

    def grasp_sampler(state: State, goal: Set[GroundAtom],
                      rng: np.random.Generator,
                      objs: Sequence[Object]) -> Array:
        del goal  # unused
        assert len(objs) == 1
        can = objs[0]
        # Need a max here in case the can is trashed already, in which case
        # both pose_x and pose_y will be -999.
        end_x = max(0.0, state.get(can, "pose_x"))
        end_y = max(0.0, state.get(can, "pose_y"))
        if with_place:
            start_x, start_y = 0.2, 0.1
        else:
            start_x, start_y = rng.uniform(0.0, 1.0,
                                           size=2)  # start from anywhere
        return np.array([start_x, start_y, end_x, end_y], dtype=np.float32)

    grasp_nsrt = NSRT("Grasp",
                      parameters, preconditions, add_effects, delete_effects,
                      set(), option, option_vars, grasp_sampler)
    nsrts.add(grasp_nsrt)

    if not with_place:
        # Dump
        can = Variable("?can", can_type)
        parameters = [can]
        option_vars = []
        option = Dump
        preconditions = {
            LiftedAtom(Holding, [can]),
            LiftedAtom(Untrashed, [can])
        }
        add_effects = {LiftedAtom(HandEmpty, [])}
        delete_effects = {
            LiftedAtom(Holding, [can]),
            LiftedAtom(Untrashed, [can])
        }
        dump_nsrt = NSRT("Dump", parameters, preconditions, add_effects,
                         delete_effects, set(), option, option_vars,
                         null_sampler)
        nsrts.add(dump_nsrt)

    else:
        # Place
        can = Variable("?can", can_type)
        parameters = [can]
        option_vars = [can]
        option = Place
        preconditions = {
            LiftedAtom(Holding, [can]),
            LiftedAtom(Untrashed, [can])
        }
        add_effects = {LiftedAtom(HandEmpty, [])}
        delete_effects = {LiftedAtom(Holding, [can])}

        def place_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            start_x, start_y = 0.2, 0.1
            # Goal-conditioned sampling
            if CFG.cluttered_table_place_goal_conditioned_sampling:
                # Get the pose of the goal object
                assert len(goal) == 1
                goal_atom = next(iter(goal))
                assert goal_atom.predicate == Holding
                goal_obj = goal_atom.objects[0]
                goal_x = state.get(goal_obj, "pose_x")
                goal_y = state.get(goal_obj, "pose_y")
                # Place up w.r.t the goal, and to some distance left
                # or right such that we're not going out of x bounds
                # 0 to 0.4.
                end_y = goal_y * 1.1
                end_x = goal_x + 0.2
                if end_x > 0.4:
                    end_x = goal_x - 0.2
                return np.array([start_x, start_y, end_x, end_y],
                                dtype=np.float32)
            # Non-goal-conditioned sampling
            del state, goal, objs
            return np.array(
                [start_x, start_y,
                 rng.uniform(0, 0.4),
                 rng.uniform(0, 1.0)],
                dtype=np.float32)

        place_nsrt = NSRT("Place", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          place_sampler)
        nsrts.add(place_nsrt)

    return nsrts


def _get_blocks_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for BlocksEnv."""
    block_type, robot_type = _get_types_by_names(CFG.env, ["block", "robot"])

    On, OnTable, GripperOpen, Holding, Clear = _get_predicates_by_names(
        CFG.env, ["On", "OnTable", "GripperOpen", "Holding", "Clear"])

    Pick, Stack, PutOnTable = _get_options_by_names(
        CFG.env, ["Pick", "Stack", "PutOnTable"])

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
        LiftedAtom(GripperOpen, [robot])
    }
    add_effects = {LiftedAtom(Holding, [block])}
    delete_effects = {
        LiftedAtom(OnTable, [block]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot])
    }

    pickfromtable_nsrt = NSRT("PickFromTable", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, null_sampler)
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
        LiftedAtom(GripperOpen, [robot])
    }
    add_effects = {
        LiftedAtom(Holding, [block]),
        LiftedAtom(Clear, [otherblock])
    }
    delete_effects = {
        LiftedAtom(On, [block, otherblock]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot])
    }
    unstack_nsrt = NSRT("Unstack",
                        parameters, preconditions, add_effects, delete_effects,
                        set(), option, option_vars, null_sampler)
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
        LiftedAtom(Clear, [otherblock])
    }
    add_effects = {
        LiftedAtom(On, [block, otherblock]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot])
    }
    delete_effects = {
        LiftedAtom(Holding, [block]),
        LiftedAtom(Clear, [otherblock])
    }

    stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                      delete_effects, set(), option, option_vars, null_sampler)
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
        LiftedAtom(GripperOpen, [robot])
    }
    delete_effects = {LiftedAtom(Holding, [block])}

    def putontable_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del state, goal, objs  # unused
        # Note: normalized coordinates w.r.t. workspace.
        x = rng.uniform()
        y = rng.uniform()
        return np.array([x, y], dtype=np.float32)

    putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                           add_effects, delete_effects, set(), option,
                           option_vars, putontable_sampler)
    nsrts.add(putontable_nsrt)

    return nsrts


def _get_painting_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for PaintingEnv."""
    obj_type, box_type, lid_type, shelf_type, robot_type = \
        _get_types_by_names(CFG.env, ["obj", "box", "lid", "shelf", "robot"])

    (InBox, InShelf, IsBoxColor, IsShelfColor, GripperOpen, OnTable, \
        NotOnTable, HoldingTop, HoldingSide, Holding, IsWet, IsDry, IsDirty, \
        IsClean) = \
        _get_predicates_by_names(
            CFG.env, ["InBox", "InShelf", "IsBoxColor", "IsShelfColor",
                        "GripperOpen", "OnTable", "NotOnTable", "HoldingTop",
                        "HoldingSide", "Holding", "IsWet", "IsDry", "IsDirty",
                        "IsClean"])

    Pick, Wash, Dry, Paint, Place, OpenLid = _get_options_by_names(
        CFG.env, ["Pick", "Wash", "Dry", "Paint", "Place", "OpenLid"])

    if CFG.env == "repeated_nextto_painting":
        (NextTo, NextToBox, NextToShelf, NextToTable) = \
         _get_predicates_by_names(
             CFG.env, ["NextTo", "NextToBox", "NextToShelf", "NextToTable"])
        MoveToObj, MoveToBox, MoveToShelf = _get_options_by_names(
            CFG.env, ["MoveToObj", "MoveToBox", "MoveToShelf"])

    nsrts = set()

    # PickFromTop
    obj = Variable("?obj", obj_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, robot]
    option_vars = [robot, obj]
    option = Pick
    preconditions = {
        LiftedAtom(GripperOpen, [robot]),
        LiftedAtom(OnTable, [obj])
    }
    if CFG.env == "repeated_nextto_painting":
        preconditions.add(LiftedAtom(NextTo, [robot, obj]))
    add_effects = {LiftedAtom(Holding, [obj]), LiftedAtom(HoldingTop, [obj])}
    delete_effects = {LiftedAtom(GripperOpen, [robot])}

    def pickfromtop_sampler(state: State, goal: Set[GroundAtom],
                            rng: np.random.Generator,
                            objs: Sequence[Object]) -> Array:
        del state, goal, rng, objs  # unused
        return np.array([1.0], dtype=np.float32)

    pickfromtop_nsrt = NSRT("PickFromTop", parameters, preconditions,
                            add_effects, delete_effects, set(), option,
                            option_vars, pickfromtop_sampler)
    nsrts.add(pickfromtop_nsrt)

    # PickFromSide
    obj = Variable("?obj", obj_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, robot]
    option_vars = [robot, obj]
    option = Pick
    preconditions = {
        LiftedAtom(GripperOpen, [robot]),
        LiftedAtom(OnTable, [obj])
    }
    if CFG.env == "repeated_nextto_painting":
        preconditions.add(LiftedAtom(NextTo, [robot, obj]))
    add_effects = {LiftedAtom(Holding, [obj]), LiftedAtom(HoldingSide, [obj])}
    delete_effects = {LiftedAtom(GripperOpen, [robot])}

    def pickfromside_sampler(state: State, goal: Set[GroundAtom],
                             rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
        del state, goal, rng, objs  # unused
        return np.array([0.0], dtype=np.float32)

    pickfromside_nsrt = NSRT("PickFromSide", parameters, preconditions,
                             add_effects, delete_effects, set(), option,
                             option_vars, pickfromside_sampler)
    nsrts.add(pickfromside_nsrt)

    # Wash
    obj = Variable("?obj", obj_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, robot]
    option_vars = [robot]
    option = Wash
    preconditions = {
        LiftedAtom(Holding, [obj]),
        LiftedAtom(IsDry, [obj]),
        LiftedAtom(IsDirty, [obj])
    }
    if CFG.env == "repeated_nextto_painting":
        preconditions.add(LiftedAtom(NextTo, [robot, obj]))
    add_effects = {LiftedAtom(IsWet, [obj]), LiftedAtom(IsClean, [obj])}
    delete_effects = {LiftedAtom(IsDry, [obj]), LiftedAtom(IsDirty, [obj])}

    wash_nsrt = NSRT("Wash", parameters, preconditions, add_effects,
                     delete_effects, set(), option, option_vars, null_sampler)
    nsrts.add(wash_nsrt)

    # Dry
    obj = Variable("?obj", obj_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, robot]
    option_vars = [robot]
    option = Dry
    preconditions = {
        LiftedAtom(Holding, [obj]),
        LiftedAtom(IsWet, [obj]),
    }
    if CFG.env == "repeated_nextto_painting":
        preconditions.add(LiftedAtom(NextTo, [robot, obj]))
    add_effects = {LiftedAtom(IsDry, [obj])}
    delete_effects = {LiftedAtom(IsWet, [obj])}

    dry_nsrt = NSRT("Dry", parameters, preconditions, add_effects,
                    delete_effects, set(), option, option_vars, null_sampler)
    nsrts.add(dry_nsrt)

    # PaintToBox
    obj = Variable("?obj", obj_type)
    box = Variable("?box", box_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, box, robot]
    option_vars = [robot]
    option = Paint
    preconditions = {
        LiftedAtom(Holding, [obj]),
        LiftedAtom(IsDry, [obj]),
        LiftedAtom(IsClean, [obj])
    }
    if CFG.env == "repeated_nextto_painting":
        preconditions.add(LiftedAtom(NextTo, [robot, obj]))
    add_effects = {LiftedAtom(IsBoxColor, [obj, box])}
    delete_effects = set()

    def painttobox_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del goal, rng  # unused
        box_color = state.get(objs[1], "color")
        return np.array([box_color], dtype=np.float32)

    painttobox_nsrt = NSRT("PaintToBox", parameters, preconditions,
                           add_effects, delete_effects, set(), option,
                           option_vars, painttobox_sampler)
    nsrts.add(painttobox_nsrt)

    # PaintToShelf
    obj = Variable("?obj", obj_type)
    shelf = Variable("?shelf", shelf_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, shelf, robot]
    option_vars = [robot]
    option = Paint
    preconditions = {
        LiftedAtom(Holding, [obj]),
        LiftedAtom(IsDry, [obj]),
        LiftedAtom(IsClean, [obj])
    }
    if CFG.env == "repeated_nextto_painting":
        preconditions.add(LiftedAtom(NextTo, [robot, obj]))
    add_effects = {LiftedAtom(IsShelfColor, [obj, shelf])}
    delete_effects = set()

    def painttoshelf_sampler(state: State, goal: Set[GroundAtom],
                             rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
        del goal, rng  # unused
        shelf_color = state.get(objs[1], "color")
        return np.array([shelf_color], dtype=np.float32)

    painttoshelf_nsrt = NSRT("PaintToShelf", parameters, preconditions,
                             add_effects, delete_effects, set(), option,
                             option_vars, painttoshelf_sampler)
    nsrts.add(painttoshelf_nsrt)

    # PlaceInBox
    obj = Variable("?obj", obj_type)
    box = Variable("?box", box_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, box, robot]
    option_vars = [robot]
    option = Place
    preconditions = {
        LiftedAtom(Holding, [obj]),
        LiftedAtom(HoldingTop, [obj]),
    }
    if CFG.env == "repeated_nextto_painting":
        preconditions.add(LiftedAtom(NextToBox, [robot, box]))
        preconditions.add(LiftedAtom(NextTo, [robot, obj]))
    add_effects = {
        LiftedAtom(InBox, [obj, box]),
        LiftedAtom(GripperOpen, [robot]),
        LiftedAtom(NotOnTable, [obj]),
    }
    delete_effects = {
        LiftedAtom(HoldingTop, [obj]),
        LiftedAtom(Holding, [obj]),
        LiftedAtom(OnTable, [obj]),
    }
    if CFG.env == "repeated_nextto_painting":
        # (Not)OnTable is affected by moving, not placing, in rnt_painting.
        # So we remove it from the add and delete effects here.
        add_effects.remove(LiftedAtom(NotOnTable, [obj]))
        delete_effects.remove(LiftedAtom(OnTable, [obj]))

    def placeinbox_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del goal  # unused
        x = state.get(objs[0], "pose_x")
        if CFG.env == "painting":
            y = rng.uniform(PaintingEnv.box_lb, PaintingEnv.box_ub)
            z = state.get(objs[0], "pose_z")
        elif CFG.env == "repeated_nextto_painting":
            y = rng.uniform(RepeatedNextToPaintingEnv.box_lb,
                            RepeatedNextToPaintingEnv.box_ub)
            z = RepeatedNextToPaintingEnv.obj_z
        return np.array([x, y, z], dtype=np.float32)

    placeinbox_nsrt = NSRT("PlaceInBox", parameters, preconditions,
                           add_effects, delete_effects, set(), option,
                           option_vars, placeinbox_sampler)
    nsrts.add(placeinbox_nsrt)

    # PlaceInShelf
    obj = Variable("?obj", obj_type)
    shelf = Variable("?shelf", shelf_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, shelf, robot]
    option_vars = [robot]
    option = Place
    preconditions = {
        LiftedAtom(Holding, [obj]),
        LiftedAtom(HoldingSide, [obj]),
    }
    if CFG.env == "repeated_nextto_painting":
        preconditions.add(LiftedAtom(NextToShelf, [robot, shelf]))
        preconditions.add(LiftedAtom(NextTo, [robot, obj]))
    add_effects = {
        LiftedAtom(InShelf, [obj, shelf]),
        LiftedAtom(GripperOpen, [robot]),
        LiftedAtom(NotOnTable, [obj]),
    }
    delete_effects = {
        LiftedAtom(HoldingSide, [obj]),
        LiftedAtom(Holding, [obj]),
        LiftedAtom(OnTable, [obj]),
    }
    if CFG.env == "repeated_nextto_painting":
        # (Not)OnTable is affected by moving, not placing, in rnt_painting.
        # So we remove it from the add and delete effects here.
        add_effects.remove(LiftedAtom(NotOnTable, [obj]))
        delete_effects.remove(LiftedAtom(OnTable, [obj]))

    def placeinshelf_sampler(state: State, goal: Set[GroundAtom],
                             rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
        del goal  # unused
        x = state.get(objs[0], "pose_x")
        if CFG.env == "painting":
            y = rng.uniform(PaintingEnv.shelf_lb, PaintingEnv.shelf_ub)
            z = state.get(objs[0], "pose_z")
        elif CFG.env == "repeated_nextto_painting":
            y = rng.uniform(RepeatedNextToPaintingEnv.shelf_lb,
                            RepeatedNextToPaintingEnv.shelf_ub)
            z = RepeatedNextToPaintingEnv.obj_z
        return np.array([x, y, z], dtype=np.float32)

    placeinshelf_nsrt = NSRT("PlaceInShelf", parameters, preconditions,
                             add_effects, delete_effects, set(), option,
                             option_vars, placeinshelf_sampler)
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

    openlid_nsrt = NSRT("OpenLid",
                        parameters, preconditions, add_effects, delete_effects,
                        set(), option, option_vars, null_sampler)
    nsrts.add(openlid_nsrt)

    # PlaceOnTable
    obj = Variable("?obj", obj_type)
    robot = Variable("?robot", robot_type)
    parameters = [obj, robot]
    option_vars = [robot]
    option = Place
    if CFG.env == "painting":
        # The environment is a little weird: the object is technically
        # already OnTable when we go to place it on the table, because
        # of how the classifier is implemented.
        preconditions = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(OnTable, [obj]),
        }
        add_effects = {
            LiftedAtom(GripperOpen, [robot]),
        }
        delete_effects = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(HoldingTop, [obj]),
            LiftedAtom(HoldingSide, [obj]),
        }
    elif CFG.env == "repeated_nextto_painting":
        preconditions = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(NextTo, [robot, obj]),
            LiftedAtom(NextToTable, [robot]),
        }
        add_effects = {
            LiftedAtom(GripperOpen, [robot]),
            LiftedAtom(OnTable, [obj]),
        }
        delete_effects = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(HoldingTop, [obj]),
            LiftedAtom(HoldingSide, [obj]),
            LiftedAtom(NotOnTable, [obj]),
        }

    def placeontable_sampler(state: State, goal: Set[GroundAtom],
                             rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
        del goal  # unused
        x = state.get(objs[0], "pose_x")
        if CFG.env == "painting":
            # Always release the object where it is, to avoid the
            # possibility of collisions with other objects.
            y = state.get(objs[0], "pose_y")
            z = state.get(objs[0], "pose_z")
        elif CFG.env == "repeated_nextto_painting":
            # Release the object at a randomly-chosen position on the table
            # such that it is NextTo the robot.
            robot_y = state.get(objs[1], "pose_y")
            table_lb = RepeatedNextToPaintingEnv.table_lb
            table_ub = RepeatedNextToPaintingEnv.table_ub
            assert table_lb < robot_y < table_ub
            nextto_thresh = RepeatedNextToPaintingEnv.nextto_thresh
            y_sample_lb = max(table_lb, robot_y - nextto_thresh)
            y_sample_ub = min(table_ub, robot_y + nextto_thresh)
            y = rng.uniform(y_sample_lb, y_sample_ub)
            z = RepeatedNextToPaintingEnv.obj_z
        return np.array([x, y, z], dtype=np.float32)

    placeontable_nsrt = NSRT("PlaceOnTable", parameters, preconditions,
                             add_effects, delete_effects, set(), option,
                             option_vars, placeontable_sampler)
    nsrts.add(placeontable_nsrt)

    if CFG.env == "repeated_nextto_painting":

        def moveto_sampler(state: State, goal: Set[GroundAtom],
                           _rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
            del goal  # unused
            y = state.get(objs[1], "pose_y")
            return np.array([y], dtype=np.float32)

        # MoveToObj
        robot = Variable("?robot", robot_type)
        targetobj = Variable("?targetobj", obj_type)
        parameters = [robot, targetobj]
        option_vars = [robot, targetobj]
        option = MoveToObj
        preconditions = {
            LiftedAtom(GripperOpen, [robot]),
            LiftedAtom(OnTable, [targetobj]),
        }
        add_effects = {
            LiftedAtom(NextTo, [robot, targetobj]),
            LiftedAtom(NextToTable, [robot])
        }
        delete_effects = set()
        # Moving could have us end up NextTo other objects, and
        # can turn off being next to the box or the shelf.
        ignore_effects = {NextTo, NextToBox, NextToShelf}

        movetoobj_nsrt = NSRT("MoveToObj", parameters, preconditions,
                              add_effects, delete_effects, ignore_effects,
                              option, option_vars, moveto_sampler)
        nsrts.add(movetoobj_nsrt)

        # MoveToBox
        robot = Variable("?robot", robot_type)
        targetbox = Variable("?targetbox", box_type)
        obj = Variable("?obj", obj_type)
        parameters = [robot, targetbox, obj]
        option_vars = [robot, targetbox]
        option = MoveToBox
        preconditions = {
            LiftedAtom(NextTo, [robot, obj]),
            LiftedAtom(Holding, [obj]),
        }
        add_effects = {
            LiftedAtom(NextToBox, [robot, targetbox]),
            LiftedAtom(NextTo, [robot, obj]),
            LiftedAtom(NotOnTable, [obj])
        }
        delete_effects = {
            LiftedAtom(NextToTable, [robot]),
            LiftedAtom(OnTable, [obj])
        }
        # Moving could have us end up NextTo other objects.
        ignore_effects = {NextTo}
        movetobox_nsrt = NSRT("MoveToBox", parameters, preconditions,
                              add_effects, delete_effects, ignore_effects,
                              option, option_vars, moveto_sampler)
        nsrts.add(movetobox_nsrt)

        # MoveToShelf
        robot = Variable("?robot", robot_type)
        targetshelf = Variable("?targetshelf", shelf_type)
        obj = Variable("?obj", obj_type)
        parameters = [robot, targetshelf, obj]
        option_vars = [robot, targetshelf]
        option = MoveToShelf
        preconditions = {
            LiftedAtom(NextTo, [robot, obj]),
            LiftedAtom(Holding, [obj]),
        }
        add_effects = {
            LiftedAtom(NextToShelf, [robot, targetshelf]),
            LiftedAtom(NextTo, [robot, obj]),
            LiftedAtom(NotOnTable, [obj])
        }
        delete_effects = {
            LiftedAtom(NextToTable, [robot]),
            LiftedAtom(OnTable, [obj])
        }
        # Moving could have us end up NextTo other objects.
        ignore_effects = {NextTo}
        movetoshelf_nsrt = NSRT("MoveToShelf", parameters, preconditions,
                                add_effects, delete_effects, ignore_effects,
                                option, option_vars, moveto_sampler)
        nsrts.add(movetoshelf_nsrt)

    return nsrts


def _get_tools_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for ToolsEnv."""
    robot_type, screw_type, screwdriver_type, nail_type, hammer_type, \
        bolt_type, wrench_type, contraption_type = _get_types_by_names(
            "tools", ["robot", "screw", "screwdriver", "nail", "hammer",
                      "bolt", "wrench", "contraption"])

    HandEmpty, HoldingScrew, HoldingScrewdriver, HoldingNail, HoldingHammer, \
        HoldingBolt, HoldingWrench, ScrewPlaced, NailPlaced, BoltPlaced, \
        ScrewFastened, NailFastened, BoltFastened, ScrewdriverGraspable, \
        HammerGraspable = _get_predicates_by_names(
            "tools", ["HandEmpty", "HoldingScrew", "HoldingScrewdriver",
                      "HoldingNail", "HoldingHammer", "HoldingBolt",
                      "HoldingWrench", "ScrewPlaced", "NailPlaced",
                      "BoltPlaced", "ScrewFastened", "NailFastened",
                      "BoltFastened", "ScrewdriverGraspable",
                      "HammerGraspable"])

    PickScrew, PickScrewdriver, PickNail, PickHammer, PickBolt, PickWrench, \
        Place, FastenScrewWithScrewdriver, FastenScrewByHand, \
        FastenNailWithHammer, FastenBoltWithWrench = _get_options_by_names(
            "tools", ["PickScrew", "PickScrewdriver", "PickNail", "PickHammer",
                      "PickBolt", "PickWrench", "Place",
                      "FastenScrewWithScrewdriver", "FastenScrewByHand",
                      "FastenNailWithHammer", "FastenBoltWithWrench"])

    def placeback_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
        # Sampler for placing an item back in its initial spot.
        del goal, rng  # unused
        _, item = objs
        pose_x = state.get(item, "pose_x")
        pose_y = state.get(item, "pose_y")
        return np.array([pose_x, pose_y], dtype=np.float32)

    def placeoncontraption_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
        # Sampler for placing an item on a contraption.
        del goal  # unused
        _, _, contraption = objs
        pose_lx = state.get(contraption, "pose_lx")
        pose_ly = state.get(contraption, "pose_ly")
        pose_ux = pose_lx + ToolsEnv.contraption_size
        pose_uy = pose_ly + ToolsEnv.contraption_size
        # Note: Here we just use the average (plus noise), to make sampler
        # learning easier. We found that it's harder to learn to imitate the
        # more preferable sampler, which uses rng.uniform over the bounds.
        pose_x = pose_lx + (pose_ux - pose_lx) / 2.0 + rng.uniform() * 0.01
        pose_y = pose_ly + (pose_uy - pose_ly) / 2.0 + rng.uniform() * 0.01
        return np.array([pose_x, pose_y], dtype=np.float32)

    nsrts = set()

    # PickScrew
    robot = Variable("?robot", robot_type)
    screw = Variable("?screw", screw_type)
    parameters = [robot, screw]
    option_vars = [robot, screw]
    option = PickScrew
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
    }
    add_effects = {LiftedAtom(HoldingScrew, [screw])}
    delete_effects = {
        LiftedAtom(HandEmpty, [robot]),
    }
    nsrts.add(
        NSRT("PickScrew", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, null_sampler))

    # PickScrewdriver
    robot = Variable("?robot", robot_type)
    screwdriver = Variable("?screwdriver", screwdriver_type)
    parameters = [robot, screwdriver]
    option_vars = [robot, screwdriver]
    option = PickScrewdriver
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(ScrewdriverGraspable, [screwdriver])
    }
    add_effects = {LiftedAtom(HoldingScrewdriver, [screwdriver])}
    delete_effects = {LiftedAtom(HandEmpty, [robot])}
    nsrts.add(
        NSRT("PickScrewDriver", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, null_sampler))

    # PickNail
    robot = Variable("?robot", robot_type)
    nail = Variable("?nail", nail_type)
    parameters = [robot, nail]
    option_vars = [robot, nail]
    option = PickNail
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
    }
    add_effects = {LiftedAtom(HoldingNail, [nail])}
    delete_effects = {
        LiftedAtom(HandEmpty, [robot]),
    }
    nsrts.add(
        NSRT("PickNail", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, null_sampler))

    # PickHammer
    robot = Variable("?robot", robot_type)
    hammer = Variable("?hammer", hammer_type)
    parameters = [robot, hammer]
    option_vars = [robot, hammer]
    option = PickHammer
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(HammerGraspable, [hammer])
    }
    add_effects = {LiftedAtom(HoldingHammer, [hammer])}
    delete_effects = {LiftedAtom(HandEmpty, [robot])}
    nsrts.add(
        NSRT("PickHammer", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, null_sampler))

    # PickBolt
    robot = Variable("?robot", robot_type)
    bolt = Variable("?bolt", bolt_type)
    parameters = [robot, bolt]
    option_vars = [robot, bolt]
    option = PickBolt
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
    }
    add_effects = {LiftedAtom(HoldingBolt, [bolt])}
    delete_effects = {
        LiftedAtom(HandEmpty, [robot]),
    }
    nsrts.add(
        NSRT("PickBolt", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, null_sampler))

    # PickWrench
    robot = Variable("?robot", robot_type)
    wrench = Variable("?wrench", wrench_type)
    parameters = [robot, wrench]
    option_vars = [robot, wrench]
    option = PickWrench
    preconditions = {LiftedAtom(HandEmpty, [robot])}
    add_effects = {LiftedAtom(HoldingWrench, [wrench])}
    delete_effects = {LiftedAtom(HandEmpty, [robot])}
    nsrts.add(
        NSRT("PickWrench", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, null_sampler))

    # PlaceScrewdriverBack
    robot = Variable("?robot", robot_type)
    screwdriver = Variable("?screwdriver", screwdriver_type)
    parameters = [robot, screwdriver]
    option_vars = [robot]
    option = Place
    preconditions = {
        LiftedAtom(HoldingScrewdriver, [screwdriver]),
        LiftedAtom(ScrewdriverGraspable, [screwdriver])
    }
    add_effects = {LiftedAtom(HandEmpty, [robot])}
    delete_effects = {LiftedAtom(HoldingScrewdriver, [screwdriver])}
    nsrts.add(
        NSRT("PlaceScrewdriverBack", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, placeback_sampler))

    # PlaceHammerBack
    robot = Variable("?robot", robot_type)
    hammer = Variable("?hammer", hammer_type)
    parameters = [robot, hammer]
    option_vars = [robot]
    option = Place
    preconditions = {
        LiftedAtom(HoldingHammer, [hammer]),
        LiftedAtom(HammerGraspable, [hammer])
    }
    add_effects = {LiftedAtom(HandEmpty, [robot])}
    delete_effects = {LiftedAtom(HoldingHammer, [hammer])}
    nsrts.add(
        NSRT("PlaceHammerBack", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, placeback_sampler))

    # PlaceWrenchBack
    robot = Variable("?robot", robot_type)
    wrench = Variable("?wrench", wrench_type)
    parameters = [robot, wrench]
    option_vars = [robot]
    option = Place
    preconditions = {LiftedAtom(HoldingWrench, [wrench])}
    add_effects = {LiftedAtom(HandEmpty, [robot])}
    delete_effects = {LiftedAtom(HoldingWrench, [wrench])}
    nsrts.add(
        NSRT("PlaceWrenchBack", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, placeback_sampler))

    # PlaceScrewOnContraption
    robot = Variable("?robot", robot_type)
    screw = Variable("?screw", screw_type)
    contraption = Variable("?contraption", contraption_type)
    parameters = [robot, screw, contraption]
    option_vars = [robot]
    option = Place
    preconditions = {LiftedAtom(HoldingScrew, [screw])}
    add_effects = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(ScrewPlaced, [screw, contraption])
    }
    delete_effects = {LiftedAtom(HoldingScrew, [screw])}
    nsrts.add(
        NSRT("PlaceScrewOnContraption", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars,
             placeoncontraption_sampler))

    # PlaceNailOnContraption
    robot = Variable("?robot", robot_type)
    nail = Variable("?nail", nail_type)
    contraption = Variable("?contraption", contraption_type)
    parameters = [robot, nail, contraption]
    option_vars = [robot]
    option = Place
    preconditions = {LiftedAtom(HoldingNail, [nail])}
    add_effects = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(NailPlaced, [nail, contraption])
    }
    delete_effects = {LiftedAtom(HoldingNail, [nail])}
    nsrts.add(
        NSRT("PlaceNailOnContraption", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars,
             placeoncontraption_sampler))

    # PlaceBoltOnContraption
    robot = Variable("?robot", robot_type)
    bolt = Variable("?bolt", bolt_type)
    contraption = Variable("?contraption", contraption_type)
    parameters = [robot, bolt, contraption]
    option_vars = [robot]
    option = Place
    preconditions = {LiftedAtom(HoldingBolt, [bolt])}
    add_effects = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(BoltPlaced, [bolt, contraption])
    }
    delete_effects = {LiftedAtom(HoldingBolt, [bolt])}
    nsrts.add(
        NSRT("PlaceBoltOnContraption", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars,
             placeoncontraption_sampler))

    # FastenScrewWithScrewdriver
    robot = Variable("?robot", robot_type)
    screw = Variable("?screw", screw_type)
    screwdriver = Variable("?screwdriver", screwdriver_type)
    contraption = Variable("?contraption", contraption_type)
    parameters = [robot, screw, screwdriver, contraption]
    option_vars = [robot, screw, screwdriver, contraption]
    option = FastenScrewWithScrewdriver
    preconditions = {
        LiftedAtom(HoldingScrewdriver, [screwdriver]),
        LiftedAtom(ScrewPlaced, [screw, contraption])
    }
    add_effects = {LiftedAtom(ScrewFastened, [screw])}
    delete_effects = set()
    nsrts.add(
        NSRT("FastenScrewWithScrewdriver", parameters, preconditions,
             add_effects, delete_effects, set(), option, option_vars,
             null_sampler))

    # FastenScrewByHand
    robot = Variable("?robot", robot_type)
    screw = Variable("?screw", screw_type)
    contraption = Variable("?contraption", contraption_type)
    parameters = [robot, screw, contraption]
    option_vars = [robot, screw, contraption]
    option = FastenScrewByHand
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(ScrewPlaced, [screw, contraption])
    }
    add_effects = {LiftedAtom(ScrewFastened, [screw])}
    delete_effects = set()
    nsrts.add(
        NSRT("FastenScrewByHand", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, null_sampler))

    # FastenNailWithHammer
    robot = Variable("?robot", robot_type)
    nail = Variable("?nail", nail_type)
    hammer = Variable("?hammer", hammer_type)
    contraption = Variable("?contraption", contraption_type)
    parameters = [robot, nail, hammer, contraption]
    option_vars = [robot, nail, hammer, contraption]
    option = FastenNailWithHammer
    preconditions = {
        LiftedAtom(HoldingHammer, [hammer]),
        LiftedAtom(NailPlaced, [nail, contraption])
    }
    add_effects = {LiftedAtom(NailFastened, [nail])}
    delete_effects = set()
    nsrts.add(
        NSRT("FastenNailWithHammer", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, null_sampler))

    # FastenBoltWithWrench
    robot = Variable("?robot", robot_type)
    bolt = Variable("?bolt", bolt_type)
    wrench = Variable("?wrench", wrench_type)
    contraption = Variable("?contraption", contraption_type)
    parameters = [robot, bolt, wrench, contraption]
    option_vars = [robot, bolt, wrench, contraption]
    option = FastenBoltWithWrench
    preconditions = {
        LiftedAtom(HoldingWrench, [wrench]),
        LiftedAtom(BoltPlaced, [bolt, contraption])
    }
    add_effects = {LiftedAtom(BoltFastened, [bolt])}
    delete_effects = set()
    nsrts.add(
        NSRT("FastenBoltWithWrench", parameters, preconditions, add_effects,
             delete_effects, set(), option, option_vars, null_sampler))

    return nsrts


def _get_playroom_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for Playroom Env."""
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

    Pick, Stack, PutOnTable, MoveToDoor, MoveDoorToTable, \
        MoveDoorToDial, OpenDoor, CloseDoor, TurnOnDial, \
        TurnOffDial = _get_options_by_names("playroom",
        ["Pick", "Stack", "PutOnTable", "MoveToDoor",
         "MoveDoorToTable", "MoveDoorToDial", "OpenDoor", "CloseDoor",
         "TurnOnDial", "TurnOffDial"])

    nsrts = set()

    # PickFromTable
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [robot, block]
    option_vars = [robot, block]
    option = Pick
    preconditions = {
        LiftedAtom(OnTable, [block]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot]),
        LiftedAtom(NextToTable, [robot])
    }
    add_effects = {LiftedAtom(Holding, [block])}
    delete_effects = {
        LiftedAtom(OnTable, [block]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot])
    }

    def pickfromtable_sampler(state: State, goal: Set[GroundAtom],
                              rng: np.random.Generator,
                              objs: Sequence[Object]) -> Array:
        del goal, rng  # unused
        assert len(objs) == 2
        _, block = objs
        assert block.is_instance(block_type)
        # find rotation of robot that faces the table
        x, y = state.get(block, "pose_x"), state.get(block, "pose_y")
        cls = PlayroomEnv
        table_x = (cls.table_x_lb + cls.table_x_ub) / 2
        table_y = (cls.table_y_lb + cls.table_y_ub) / 2
        rotation = np.arctan2(table_y - y, table_x - x) / np.pi
        return np.array([rotation], dtype=np.float32)

    pickfromtable_nsrt = NSRT("PickFromTable", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, pickfromtable_sampler)
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
        LiftedAtom(NextToTable, [robot])
    }
    add_effects = {
        LiftedAtom(Holding, [block]),
        LiftedAtom(Clear, [otherblock])
    }
    delete_effects = {
        LiftedAtom(On, [block, otherblock]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot])
    }

    def unstack_sampler(state: State, goal: Set[GroundAtom],
                        rng: np.random.Generator,
                        objs: Sequence[Object]) -> Array:
        del goal, rng  # unused
        assert len(objs) == 3
        block, _, _ = objs
        assert block.is_instance(block_type)
        # find rotation of robot that faces the table
        x, y = state.get(block, "pose_x"), state.get(block, "pose_y")
        cls = PlayroomEnv
        table_x = (cls.table_x_lb + cls.table_x_ub) / 2
        table_y = (cls.table_y_lb + cls.table_y_ub) / 2
        rotation = np.arctan2(table_y - y, table_x - x) / np.pi
        return np.array([rotation], dtype=np.float32)

    unstack_nsrt = NSRT("Unstack",
                        parameters, preconditions, add_effects, delete_effects,
                        set(), option, option_vars, unstack_sampler)
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
        LiftedAtom(NextToTable, [robot])
    }
    add_effects = {
        LiftedAtom(On, [block, otherblock]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot])
    }
    delete_effects = {
        LiftedAtom(Holding, [block]),
        LiftedAtom(Clear, [otherblock])
    }

    def stack_sampler(state: State, goal: Set[GroundAtom],
                      rng: np.random.Generator,
                      objs: Sequence[Object]) -> Array:
        del goal, rng  # unused
        assert len(objs) == 3
        _, otherblock, _ = objs
        assert otherblock.is_instance(block_type)
        # find rotation of robot that faces the table
        x, y = state.get(otherblock, "pose_x"), state.get(otherblock, "pose_y")
        cls = PlayroomEnv
        table_x = (cls.table_x_lb + cls.table_x_ub) / 2
        table_y = (cls.table_y_lb + cls.table_y_ub) / 2
        rotation = np.arctan2(table_y - y, table_x - x) / np.pi
        return np.array([rotation], dtype=np.float32)

    stack_nsrt = NSRT("Stack",
                      parameters, preconditions, add_effects, delete_effects,
                      set(), option, option_vars, stack_sampler)
    nsrts.add(stack_nsrt)

    # PutOnTable
    block = Variable("?block", block_type)
    robot = Variable("?robot", robot_type)
    parameters = [block, robot]
    option_vars = [robot]
    option = PutOnTable
    preconditions = {
        LiftedAtom(Holding, [block]),
        LiftedAtom(NextToTable, [robot])
    }
    add_effects = {
        LiftedAtom(OnTable, [block]),
        LiftedAtom(Clear, [block]),
        LiftedAtom(GripperOpen, [robot])
    }
    delete_effects = {LiftedAtom(Holding, [block])}

    def putontable_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del state, goal, objs  # unused
        x = rng.uniform()
        y = rng.uniform()
        # find rotation of robot that faces the table
        cls = PlayroomEnv
        table_x = (cls.table_x_lb + cls.table_x_ub) / 2
        table_y = (cls.table_y_lb + cls.table_y_ub) / 2
        rotation = np.arctan2(table_y - y, table_x - x) / np.pi
        return np.array([x, y, rotation], dtype=np.float32)

    putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                           add_effects, delete_effects, set(), option,
                           option_vars, putontable_sampler)
    nsrts.add(putontable_nsrt)

    # AdvanceThroughDoor
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    from_region = Variable("?from", region_type)
    to_region = Variable("?to", region_type)
    parameters = [robot, door, from_region, to_region]
    option_vars = [robot, from_region, door]
    option = MoveToDoor
    preconditions = {
        LiftedAtom(InRegion, [robot, from_region]),
        LiftedAtom(Connects, [door, from_region, to_region]),
        LiftedAtom(DoorOpen, [door]),
        LiftedAtom(NextToDoor, [robot, door])
    }
    add_effects = {LiftedAtom(InRegion, [robot, to_region])}
    delete_effects = {LiftedAtom(InRegion, [robot, from_region])}

    def advancethroughdoor_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
        del goal, rng  # unused
        assert len(objs) == 4
        robot, door, _, _ = objs
        assert robot.is_instance(robot_type)
        assert door.is_instance(door_type)
        if state.get(robot, "pose_x") < state.get(door, "pose_x"):
            return np.array([0.2, 0.0, 0.0], dtype=np.float32)
        return np.array([-0.2, 0.0, -1.0], dtype=np.float32)

    advancethroughdoor_nsrt = NSRT("AdvanceThroughDoor", parameters,
                                   preconditions, add_effects, delete_effects,
                                   set(), option, option_vars,
                                   advancethroughdoor_sampler)
    nsrts.add(advancethroughdoor_nsrt)

    # MoveTableToDoor
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    region = Variable("?region", region_type)
    parameters = [robot, door, region]
    option_vars = [robot, region, door]
    option = MoveToDoor
    preconditions = {
        LiftedAtom(IsBoringRoom, [region]),
        LiftedAtom(InRegion, [robot, region]),
        LiftedAtom(NextToTable, [robot]),
        LiftedAtom(IsBoringRoomDoor, [door])
    }
    add_effects = {LiftedAtom(NextToDoor, [robot, door])}
    delete_effects = {LiftedAtom(NextToTable, [robot])}

    def movetabletodoor_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
        del state, goal, rng, objs  # unused
        return np.array([-0.2, 0.0, 0.0], dtype=np.float32)

    movetabletodoor_nsrt = NSRT("MoveTableToDoor", parameters,
                                preconditions, add_effects, delete_effects,
                                set(), option, option_vars,
                                movetabletodoor_sampler)
    nsrts.add(movetabletodoor_nsrt)

    # MoveDoorToTable
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    region = Variable("?region", region_type)
    parameters = [robot, door, region]
    option_vars = [robot, region]
    option = MoveDoorToTable
    preconditions = {
        LiftedAtom(IsBoringRoom, [region]),
        LiftedAtom(InRegion, [robot, region]),
        LiftedAtom(NextToDoor, [robot, door]),
        LiftedAtom(IsBoringRoomDoor, [door])
    }
    add_effects = {LiftedAtom(NextToTable, [robot])}
    delete_effects = {LiftedAtom(NextToDoor, [robot, door])}

    def movedoortotable_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
        del state, goal, rng, objs  # unused
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    movedoortotable_nsrt = NSRT("MoveDoorToTable", parameters,
                                preconditions, add_effects, delete_effects,
                                set(), option, option_vars,
                                movedoortotable_sampler)
    nsrts.add(movedoortotable_nsrt)

    # MoveDoorToDoor
    robot = Variable("?robot", robot_type)
    fromdoor = Variable("?fromdoor", door_type)
    todoor = Variable("?todoor", door_type)
    region = Variable("?region", region_type)
    parameters = [robot, fromdoor, todoor, region]
    option_vars = [robot, region, todoor]
    option = MoveToDoor
    preconditions = {
        LiftedAtom(Borders, [fromdoor, region, todoor]),
        LiftedAtom(InRegion, [robot, region]),
        LiftedAtom(NextToDoor, [robot, fromdoor])
    }
    add_effects = {LiftedAtom(NextToDoor, [robot, todoor])}
    delete_effects = {LiftedAtom(NextToDoor, [robot, fromdoor])}

    def movedoortodoor_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
        del goal, rng  # unused
        assert len(objs) == 4
        _, fromdoor, todoor, _ = objs
        assert fromdoor.is_instance(door_type)
        assert todoor.is_instance(door_type)
        if state.get(fromdoor, "pose_x") < state.get(todoor, "pose_x"):
            return np.array([-0.1, 0.0, 0.0], dtype=np.float32)
        return np.array([0.1, 0.0, -1.0], dtype=np.float32)

    movedoortodoor_nsrt = NSRT("MoveDoorToDoor", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, movedoortodoor_sampler)
    nsrts.add(movedoortodoor_nsrt)

    # MoveDoorToDial
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    dial = Variable("?dial", dial_type)
    region = Variable("?region", region_type)
    parameters = [robot, door, dial, region]
    option_vars = [robot, region, dial]
    option = MoveDoorToDial
    preconditions = {
        LiftedAtom(IsPlayroom, [region]),
        LiftedAtom(InRegion, [robot, region]),
        LiftedAtom(IsPlayroomDoor, [door]),
        LiftedAtom(NextToDoor, [robot, door])
    }
    add_effects = {LiftedAtom(NextToDial, [robot, dial])}
    delete_effects = {LiftedAtom(NextToDoor, [robot, door])}

    def movedoortodial_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
        del state, goal, rng, objs  # unused
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    movedoortodial_nsrt = NSRT("MoveDoorToDial", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, movedoortodial_sampler)
    nsrts.add(movedoortodial_nsrt)

    # MoveDialToDoor
    robot = Variable("?robot", robot_type)
    dial = Variable("?dial", dial_type)
    door = Variable("?door", door_type)
    region = Variable("?region", region_type)
    parameters = [robot, dial, door, region]
    option_vars = [robot, region, door]
    option = MoveToDoor
    preconditions = {
        LiftedAtom(IsPlayroom, [region]),
        LiftedAtom(InRegion, [robot, region]),
        LiftedAtom(IsPlayroomDoor, [door]),
        LiftedAtom(NextToDial, [robot, dial])
    }
    add_effects = {LiftedAtom(NextToDoor, [robot, door])}
    delete_effects = {LiftedAtom(NextToDial, [robot, dial])}

    def movedialtodoor_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
        del state, goal, rng, objs  # unused
        return np.array([0.1, 0.0, -1.0], dtype=np.float32)

    movedialtodoor_nsrt = NSRT("MoveDialToDoor", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, movedialtodoor_sampler)
    nsrts.add(movedialtodoor_nsrt)

    # OpenDoor
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    parameters = [robot, door]
    option_vars = [robot, door]
    option = OpenDoor
    preconditions = {
        LiftedAtom(NextToDoor, [robot, door]),
        LiftedAtom(DoorClosed, [door]),
        LiftedAtom(GripperOpen, [robot])
    }
    add_effects = {LiftedAtom(DoorOpen, [door])}
    delete_effects = {LiftedAtom(DoorClosed, [door])}

    def toggledoor_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del goal, rng  # unused
        assert len(objs) == 2
        robot, door = objs
        assert robot.is_instance(robot_type)
        assert door.is_instance(door_type)
        x, door_x = state.get(robot, "pose_x"), state.get(door, "pose_x")
        rotation = 0.0 if x < door_x else -1.0
        dx = -0.2 if x < door_x else 0.2
        return np.array([dx, 0, 0, rotation], dtype=np.float32)

    opendoor_nsrt = NSRT("OpenDoor", parameters, preconditions, add_effects,
                         delete_effects, set(), option, option_vars,
                         toggledoor_sampler)
    nsrts.add(opendoor_nsrt)

    # CloseDoor
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    parameters = [robot, door]
    option_vars = [robot, door]
    option = CloseDoor
    preconditions = {
        LiftedAtom(NextToDoor, [robot, door]),
        LiftedAtom(DoorOpen, [door]),
        LiftedAtom(GripperOpen, [robot])
    }
    add_effects = {LiftedAtom(DoorClosed, [door])}
    delete_effects = {LiftedAtom(DoorOpen, [door])}
    closedoor_nsrt = NSRT("CloseDoor", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          toggledoor_sampler)
    nsrts.add(closedoor_nsrt)

    # TurnOnDial
    robot = Variable("?robot", robot_type)
    dial = Variable("?dial", dial_type)
    parameters = [robot, dial]
    option_vars = [robot, dial]
    option = TurnOnDial
    preconditions = {
        LiftedAtom(NextToDial, [robot, dial]),
        LiftedAtom(LightOff, [dial]),
        LiftedAtom(GripperOpen, [robot])
    }
    add_effects = {LiftedAtom(LightOn, [dial])}
    delete_effects = {LiftedAtom(LightOff, [dial])}

    def toggledial_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del state, goal, rng, objs  # unused
        return np.array([-0.2, 0, 0, 0.0], dtype=np.float32)

    turnondial_nsrt = NSRT("TurnOnDial", parameters, preconditions,
                           add_effects, delete_effects, set(), option,
                           option_vars, toggledial_sampler)
    nsrts.add(turnondial_nsrt)

    # TurnOffDial
    robot = Variable("?robot", robot_type)
    dial = Variable("?dial", dial_type)
    parameters = [robot, dial]
    option_vars = [robot, dial]
    option = TurnOffDial
    preconditions = {
        LiftedAtom(NextToDial, [robot, dial]),
        LiftedAtom(LightOn, [dial]),
        LiftedAtom(GripperOpen, [robot])
    }
    add_effects = {LiftedAtom(LightOff, [dial])}
    delete_effects = {LiftedAtom(LightOn, [dial])}
    turnoffdial_nsrt = NSRT("TurnOffDial", parameters, preconditions,
                            add_effects, delete_effects, set(), option,
                            option_vars, toggledial_sampler)
    nsrts.add(turnoffdial_nsrt)

    return nsrts


def _get_repeated_nextto_gt_nsrts(env_name: str) -> Set[NSRT]:
    """Create ground truth NSRTs for RepeatedNextToEnv."""
    robot_type, dot_type = _get_types_by_names(env_name, ["robot", "dot"])

    NextTo, NextToNothing, Grasped = _get_predicates_by_names(
        env_name, ["NextTo", "NextToNothing", "Grasped"])

    Move, Grasp = _get_options_by_names(env_name, ["Move", "Grasp"])

    nsrts = set()

    # Move
    robot = Variable("?robot", robot_type)
    targetdot = Variable("?targetdot", dot_type)
    parameters = [robot, targetdot]
    option_vars = [robot, targetdot]
    option = Move
    preconditions: Set[LiftedAtom] = set()
    add_effects = {LiftedAtom(NextTo, [robot, targetdot])}
    delete_effects: Set[LiftedAtom] = set()
    # Moving could have us end up NextTo other objects. It could also
    # include NextToNothing as a delete effect.
    ignore_effects = {NextTo, NextToNothing}
    move_nsrt = NSRT("Move", parameters, preconditions, add_effects,
                     delete_effects, ignore_effects, option, option_vars,
                     lambda s, g, rng, o: np.zeros(1, dtype=np.float32))
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
    # Note that NextTo isn't an ignore effect here because it's not
    # something we'd be unsure about for any object. For every object we
    # are NextTo but did not grasp, we will stay NextTo it.
    ignore_effects = {NextToNothing}
    grasp_nsrt = NSRT("Grasp", parameters, preconditions, add_effects,
                      delete_effects, ignore_effects, option, option_vars,
                      null_sampler)
    nsrts.add(grasp_nsrt)

    return nsrts


def _get_repeated_nextto_single_option_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for RepeatedNextToSingleOptionEnv."""
    rn_grasp_nsrt, rn_move_nsrt = sorted(
        _get_repeated_nextto_gt_nsrts("repeated_nextto"),
        key=lambda nsrt: nsrt.name)
    assert rn_grasp_nsrt.name == "Grasp"
    assert rn_move_nsrt.name == "Move"

    MoveGrasp, = _get_options_by_names(CFG.env, ["MoveGrasp"])

    nsrts = set()

    # Move
    move_nsrt = NSRT(
        rn_move_nsrt.name, rn_move_nsrt.parameters, rn_move_nsrt.preconditions,
        rn_move_nsrt.add_effects, rn_move_nsrt.delete_effects,
        rn_move_nsrt.ignore_effects, MoveGrasp, rn_move_nsrt.option_vars,
        lambda s, g, rng, o: np.array([-1.0, 0.0], dtype=np.float32))
    nsrts.add(move_nsrt)

    # Grasp
    grasp_nsrt = NSRT(
        rn_grasp_nsrt.name, rn_grasp_nsrt.parameters,
        rn_grasp_nsrt.preconditions, rn_grasp_nsrt.add_effects,
        rn_grasp_nsrt.delete_effects, rn_grasp_nsrt.ignore_effects, MoveGrasp,
        rn_grasp_nsrt.option_vars,
        lambda s, g, rng, o: np.array([1.0, 0.0], dtype=np.float32))
    nsrts.add(grasp_nsrt)

    return nsrts


def _get_screws_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for ScrewsEnv."""
    screw_type, gripper_type, receptacle_type = _get_types_by_names(
        CFG.env, ["screw", "gripper", "receptacle"])
    GripperCanPickScrew, AboveReceptacle, HoldingScrew, ScrewInReceptacle = \
        _get_predicates_by_names(
        CFG.env, [
            "GripperCanPickScrew", "AboveReceptacle", "HoldingScrew",
            "ScrewInReceptacle"
        ])
    MoveToScrew, MoveToReceptacle, MagnetizeGripper, DemagnetizeGripper = \
        _get_options_by_names(
        CFG.env, [
            "MoveToScrew", "MoveToReceptacle", "MagnetizeGripper",
            "DemagnetizeGripper"
        ])

    nsrts: Set[NSRT] = set()

    # MoveToScrew
    robot = Variable("?robot", gripper_type)
    screw = Variable("?screw", screw_type)
    parameters = [robot, screw]
    option_vars = [robot, screw]
    option = MoveToScrew
    preconditions: Set[LiftedAtom] = set()
    add_effects = {LiftedAtom(GripperCanPickScrew, [robot, screw])}
    delete_effects: Set[LiftedAtom] = set()
    ignore_effects = {GripperCanPickScrew}
    move_to_screw_nsrt = NSRT("MoveToScrew", parameters, preconditions,
                              add_effects, delete_effects, ignore_effects,
                              option, option_vars, null_sampler)
    nsrts.add(move_to_screw_nsrt)

    # MoveToReceptacle
    robot = Variable("?robot", gripper_type)
    receptacle = Variable("?receptacle", receptacle_type)
    screw = Variable("?screw", screw_type)
    parameters = [robot, receptacle, screw]
    option_vars = [robot, receptacle, screw]
    option = MoveToReceptacle
    preconditions = {LiftedAtom(HoldingScrew, [robot, screw])}
    add_effects = {LiftedAtom(AboveReceptacle, [robot, receptacle])}
    ignore_effects = {GripperCanPickScrew}
    move_to_receptacle_nsrt = NSRT("MoveToReceptacle", parameters,
                                   preconditions, add_effects, delete_effects,
                                   ignore_effects, option, option_vars,
                                   null_sampler)
    nsrts.add(move_to_receptacle_nsrt)

    # MagnetizeGripper
    robot = Variable("?robot", gripper_type)
    screw = Variable("?screw", screw_type)
    parameters = [robot, screw]
    option_vars = [robot]
    option = MagnetizeGripper
    preconditions = {LiftedAtom(GripperCanPickScrew, [robot, screw])}
    add_effects = {LiftedAtom(HoldingScrew, [robot, screw])}
    ignore_effects = {HoldingScrew}
    magnetize_gripper_nsrt = NSRT("MagnetizeGripper", parameters,
                                  preconditions, add_effects, delete_effects,
                                  ignore_effects, option, option_vars,
                                  null_sampler)
    nsrts.add(magnetize_gripper_nsrt)

    # DemagnetizeGripper
    robot = Variable("?robot", gripper_type)
    screw = Variable("?screw", screw_type)
    receptacle = Variable("?receptacle", receptacle_type)
    parameters = [robot, screw, receptacle]
    option_vars = [robot]
    option = DemagnetizeGripper
    preconditions = {
        LiftedAtom(HoldingScrew, [robot, screw]),
        LiftedAtom(AboveReceptacle, [robot, receptacle])
    }
    add_effects = {LiftedAtom(ScrewInReceptacle, [screw, receptacle])}
    delete_effects = {LiftedAtom(HoldingScrew, [robot, screw])}
    ignore_effects = {HoldingScrew}
    demagnetize_gripper_nsrt = NSRT("DemagnetizeGripper", parameters,
                                    preconditions, add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)
    nsrts.add(demagnetize_gripper_nsrt)

    return nsrts


def _get_touch_point_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for TouchPointEnv."""
    robot_type, target_type = _get_types_by_names(CFG.env, ["robot", "target"])
    Touched, = _get_predicates_by_names(CFG.env, ["Touched"])
    MoveTo, = _get_options_by_names(CFG.env, ["MoveTo"])

    nsrts = set()

    # MoveTo
    robot = Variable("?robot", robot_type)
    target = Variable("?target", target_type)
    parameters = [robot, target]
    option_vars = [robot, target]
    option = MoveTo
    preconditions: Set[LiftedAtom] = set()
    add_effects = {LiftedAtom(Touched, [robot, target])}
    delete_effects: Set[LiftedAtom] = set()
    ignore_effects: Set[Predicate] = set()
    move_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                     delete_effects, ignore_effects, option, option_vars,
                     null_sampler)
    nsrts.add(move_nsrt)

    return nsrts


def _get_stick_button_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for StickButtonEnv."""
    robot_type, button_type, stick_type = _get_types_by_names(
        CFG.env, ["robot", "button", "stick"])
    Pressed, RobotAboveButton, StickAboveButton, \
        Grasped, HandEmpty, AboveNoButton = _get_predicates_by_names(
            CFG.env, ["Pressed", "RobotAboveButton",
            "StickAboveButton", "Grasped", "HandEmpty", "AboveNoButton"])
    RobotPressButton, PickStick, StickPressButton = _get_options_by_names(
        CFG.env, ["RobotPressButton", "PickStick", "StickPressButton"])

    nsrts = set()

    # RobotPressButtonFromNothing
    robot = Variable("?robot", robot_type)
    button = Variable("?button", button_type)
    parameters = [robot, button]
    option_vars = [robot, button]
    option = RobotPressButton
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(AboveNoButton, []),
    }
    add_effects = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(RobotAboveButton, [robot, button])
    }
    delete_effects = {LiftedAtom(AboveNoButton, [])}
    ignore_effects: Set[Predicate] = set()
    robot_press_button_nsrt = NSRT("RobotPressButtonFromNothing", parameters,
                                   preconditions, add_effects, delete_effects,
                                   ignore_effects, option, option_vars,
                                   null_sampler)
    nsrts.add(robot_press_button_nsrt)

    # RobotPressButtonFromButton
    robot = Variable("?robot", robot_type)
    button = Variable("?button", button_type)
    from_button = Variable("?from-button", button_type)
    parameters = [robot, button, from_button]
    option_vars = [robot, button]
    option = RobotPressButton
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(RobotAboveButton, [robot, from_button]),
    }
    add_effects = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(RobotAboveButton, [robot, button])
    }
    delete_effects = {LiftedAtom(RobotAboveButton, [robot, from_button])}
    ignore_effects = set()
    robot_press_button_nsrt = NSRT("RobotPressButtonFromButton", parameters,
                                   preconditions, add_effects, delete_effects,
                                   ignore_effects, option, option_vars,
                                   null_sampler)
    nsrts.add(robot_press_button_nsrt)

    # PickStickFromNothing
    robot = Variable("?robot", robot_type)
    stick = Variable("?stick", stick_type)
    parameters = [robot, stick]
    option_vars = [robot, stick]
    option = PickStick
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(AboveNoButton, []),
    }
    add_effects = {
        LiftedAtom(Grasped, [robot, stick]),
    }
    delete_effects = {LiftedAtom(HandEmpty, [robot])}
    ignore_effects = set()

    def pick_stick_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        del state, goal, objs  # unused
        # Normalized x position along the long dimension of the stick, in the
        # center of the short dimension.
        pick_pos = rng.uniform(0, 1)
        return np.array([pick_pos], dtype=np.float32)

    pick_stick_nsrt = NSRT("PickStickFromNothing", parameters, preconditions,
                           add_effects, delete_effects, ignore_effects, option,
                           option_vars, pick_stick_sampler)
    nsrts.add(pick_stick_nsrt)

    # PickStickFromButton
    robot = Variable("?robot", robot_type)
    stick = Variable("?stick", stick_type)
    button = Variable("?from-button", button_type)
    parameters = [robot, stick, button]
    option_vars = [robot, stick]
    option = PickStick
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(RobotAboveButton, [robot, button])
    }
    add_effects = {
        LiftedAtom(Grasped, [robot, stick]),
        LiftedAtom(AboveNoButton, [])
    }
    delete_effects = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(RobotAboveButton, [robot, button]),
    }
    ignore_effects = set()
    pick_stick_nsrt = NSRT("PickStickFromButton", parameters, preconditions,
                           add_effects, delete_effects, ignore_effects, option,
                           option_vars, pick_stick_sampler)
    nsrts.add(pick_stick_nsrt)

    # StickPressButtonFromNothing
    robot = Variable("?robot", robot_type)
    stick = Variable("?stick", stick_type)
    button = Variable("?button", button_type)
    parameters = [robot, stick, button]
    option_vars = [robot, stick, button]
    option = StickPressButton
    preconditions = {
        LiftedAtom(Grasped, [robot, stick]),
        LiftedAtom(AboveNoButton, []),
    }
    add_effects = {
        LiftedAtom(StickAboveButton, [stick, button]),
        LiftedAtom(Pressed, [button])
    }
    delete_effects = {LiftedAtom(AboveNoButton, [])}
    ignore_effects = set()
    stick_button_nsrt = NSRT("StickPressButtonFromNothing", parameters,
                             preconditions, add_effects, delete_effects,
                             ignore_effects, option, option_vars, null_sampler)
    nsrts.add(stick_button_nsrt)

    # StickPressButtonFromButton
    robot = Variable("?robot", robot_type)
    stick = Variable("?stick", stick_type)
    button = Variable("?button", button_type)
    from_button = Variable("?from-button", button_type)
    parameters = [robot, stick, button, from_button]
    option_vars = [robot, stick, button]
    option = StickPressButton
    preconditions = {
        LiftedAtom(Grasped, [robot, stick]),
        LiftedAtom(StickAboveButton, [stick, from_button])
    }
    add_effects = {
        LiftedAtom(StickAboveButton, [stick, button]),
        LiftedAtom(Pressed, [button])
    }
    delete_effects = {LiftedAtom(StickAboveButton, [stick, from_button])}
    ignore_effects = set()
    stick_button_nsrt = NSRT("StickPressButtonFromButton", parameters,
                             preconditions, add_effects, delete_effects,
                             ignore_effects, option, option_vars, null_sampler)
    nsrts.add(stick_button_nsrt)

    return nsrts


def _get_doors_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for DoorsEnv."""
    robot_type, door_type, room_type = _get_types_by_names(
        CFG.env, ["robot", "door", "room"])
    InRoom, InDoorway, InMainRoom, TouchingDoor, DoorIsOpen, DoorInRoom, \
        DoorsShareRoom = _get_predicates_by_names(CFG.env, ["InRoom",
            "InDoorway", "InMainRoom", "TouchingDoor", "DoorIsOpen",
            "DoorInRoom", "DoorsShareRoom"])
    MoveToDoor, OpenDoor, MoveThroughDoor = _get_options_by_names(
        CFG.env, ["MoveToDoor", "OpenDoor", "MoveThroughDoor"])

    nsrts = set()

    # MoveToDoorFromMainRoom
    # This operator should only be used on the first step of a plan.
    robot = Variable("?robot", robot_type)
    room = Variable("?room", room_type)
    door = Variable("?door", door_type)
    parameters = [robot, room, door]
    option_vars = [robot, door]
    option = MoveToDoor
    preconditions = {
        LiftedAtom(InRoom, [robot, room]),
        LiftedAtom(InMainRoom, [robot, room]),
        LiftedAtom(DoorInRoom, [door, room]),
    }
    add_effects = {
        LiftedAtom(TouchingDoor, [robot, door]),
        LiftedAtom(InDoorway, [robot, door])
    }
    delete_effects = {LiftedAtom(InMainRoom, [robot, room])}
    ignore_effects: Set[Predicate] = set()
    move_to_door_nsrt = NSRT("MoveToDoorFromMainRoom", parameters,
                             preconditions, add_effects, delete_effects,
                             ignore_effects, option, option_vars, null_sampler)
    nsrts.add(move_to_door_nsrt)

    # MoveToDoorFromDoorWay
    robot = Variable("?robot", robot_type)
    start_door = Variable("?start_door", door_type)
    end_door = Variable("?end_door", door_type)
    parameters = [robot, start_door, end_door]
    option_vars = [robot, end_door]
    option = MoveToDoor
    preconditions = {
        LiftedAtom(InDoorway, [robot, start_door]),
        LiftedAtom(DoorsShareRoom, [start_door, end_door]),
    }
    add_effects = {
        LiftedAtom(TouchingDoor, [robot, end_door]),
        LiftedAtom(InDoorway, [robot, end_door])
    }
    delete_effects = {LiftedAtom(InDoorway, [robot, start_door])}
    ignore_effects = set()
    move_to_door_nsrt = NSRT("MoveToDoorFromDoorWay", parameters,
                             preconditions, add_effects, delete_effects,
                             ignore_effects, option, option_vars, null_sampler)
    nsrts.add(move_to_door_nsrt)

    # OpenDoor
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    parameters = [door, robot]
    option_vars = [door, robot]
    option = OpenDoor
    preconditions = {
        LiftedAtom(TouchingDoor, [robot, door]),
        LiftedAtom(InDoorway, [robot, door]),
    }
    add_effects = {LiftedAtom(DoorIsOpen, [door])}
    delete_effects = {
        LiftedAtom(TouchingDoor, [robot, door]),
    }
    ignore_effects = set()

    # Allow protected access because this is an oracle. Used in the sampler.
    env = get_or_create_env(CFG.env)
    assert isinstance(env, DoorsEnv)
    get_open_door_target_value = env._get_open_door_target_value  # pylint: disable=protected-access

    # Even though this option does not need to be parameterized, we make it so,
    # because we want to match the parameter space of the option that will
    # get learned during option learning. This is useful for when we want
    # to use sampler_learner = "oracle" too.
    def open_door_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
        del rng, goal  # unused
        door, _ = objs
        assert door.is_instance(door_type)
        # Calculate the desired change in the doors "rotation" feature.
        # Allow protected access because this is an oracle.
        mass = state.get(door, "mass")
        friction = state.get(door, "friction")
        target_rot = state.get(door, "target_rot")
        target_val = get_open_door_target_value(mass=mass,
                                                friction=friction,
                                                target_rot=target_rot)
        current_val = state.get(door, "rot")
        delta_rot = target_val - current_val
        # The door always changes from closed to open.
        delta_open = 1.0
        return np.array([delta_rot, delta_open], dtype=np.float32)

    open_door_nsrt = NSRT("OpenDoor", parameters, preconditions, add_effects,
                          delete_effects, ignore_effects, option, option_vars,
                          open_door_sampler)
    nsrts.add(open_door_nsrt)

    # MoveThroughDoor
    robot = Variable("?robot", robot_type)
    start_room = Variable("?start_room", room_type)
    end_room = Variable("?end_room", room_type)
    door = Variable("?door", door_type)
    parameters = [robot, start_room, door, end_room]
    option_vars = [robot, door]
    option = MoveThroughDoor
    preconditions = {
        LiftedAtom(InRoom, [robot, start_room]),
        LiftedAtom(InDoorway, [robot, door]),
        LiftedAtom(DoorIsOpen, [door]),
        LiftedAtom(DoorInRoom, [door, start_room]),
        LiftedAtom(DoorInRoom, [door, end_room]),
    }
    add_effects = {
        LiftedAtom(InRoom, [robot, end_room]),
    }
    delete_effects = {
        LiftedAtom(InRoom, [robot, start_room]),
    }
    ignore_effects = set()
    move_through_door_nsrt = NSRT("MoveThroughDoor", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, null_sampler)
    nsrts.add(move_through_door_nsrt)

    return nsrts


def _get_coffee_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for CoffeeEnv."""
    robot_type, jug_type, cup_type, machine_type = _get_types_by_names(
        CFG.env, ["robot", "jug", "cup", "machine"])
    CupFilled, Holding, JugInMachine, MachineOn, OnTable, HandEmpty, \
        JugFilled, RobotAboveCup, JugAboveCup, NotAboveCup, PressingButton, \
        Twisting, NotSameCup = \
        _get_predicates_by_names(CFG.env, ["CupFilled",
            "Holding", "JugInMachine", "MachineOn", "OnTable", "HandEmpty",
            "JugFilled", "RobotAboveCup", "JugAboveCup", "NotAboveCup",
            "PressingButton", "Twisting", "NotSameCup"])
    MoveToTwistJug, TwistJug, PickJug, PlaceJugInMachine, TurnMachineOn, \
        Pour = _get_options_by_names(CFG.env, ["MoveToTwistJug", "TwistJug",
            "PickJug", "PlaceJugInMachine", "TurnMachineOn", "Pour"])

    nsrts = set()

    # MoveToTwistJug
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    parameters = [robot, jug]
    option_vars = [robot, jug]
    option = MoveToTwistJug
    preconditions = {
        LiftedAtom(OnTable, [jug]),
        LiftedAtom(HandEmpty, [robot]),
    }
    add_effects = {
        LiftedAtom(Twisting, [robot, jug]),
    }
    delete_effects = {
        LiftedAtom(HandEmpty, [robot]),
    }
    ignore_effects: Set[Predicate] = set()
    move_to_twist_jug_nsrt = NSRT("MoveToTwistJug", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, null_sampler)
    nsrts.add(move_to_twist_jug_nsrt)

    # TwistJug
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    parameters = [robot, jug]
    option_vars = [robot, jug]
    option = TwistJug
    preconditions = {
        LiftedAtom(OnTable, [jug]),
        LiftedAtom(Twisting, [robot, jug]),
    }
    add_effects = {
        LiftedAtom(HandEmpty, [robot]),
    }
    delete_effects = {
        LiftedAtom(Twisting, [robot, jug]),
    }
    ignore_effects = set()

    def twist_jug_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
        del state, goal, objs  # unused
        return np.array(rng.uniform(-1, 1, size=(1, )), dtype=np.float32)

    twist_jug_nsrt = NSRT("TwistJug", parameters, preconditions, add_effects,
                          delete_effects, ignore_effects, option, option_vars,
                          twist_jug_sampler)
    nsrts.add(twist_jug_nsrt)

    # PickJugFromTable
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    parameters = [robot, jug]
    option_vars = [robot, jug]
    option = PickJug
    preconditions = {
        LiftedAtom(OnTable, [jug]),
        LiftedAtom(HandEmpty, [robot])
    }
    add_effects = {
        LiftedAtom(Holding, [robot, jug]),
    }
    delete_effects = {
        LiftedAtom(OnTable, [jug]),
        LiftedAtom(HandEmpty, [robot])
    }
    ignore_effects = set()
    pick_jug_from_table_nsrt = NSRT("PickJugFromTable", parameters,
                                    preconditions, add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)
    nsrts.add(pick_jug_from_table_nsrt)

    # PlaceJugInMachine
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    machine = Variable("?machine", machine_type)
    parameters = [robot, jug, machine]
    option_vars = [robot, jug, machine]
    option = PlaceJugInMachine
    preconditions = {
        LiftedAtom(Holding, [robot, jug]),
    }
    add_effects = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(JugInMachine, [jug, machine]),
    }
    delete_effects = {
        LiftedAtom(Holding, [robot, jug]),
    }
    ignore_effects = set()
    place_jug_in_machine_nsrt = NSRT("PlaceJugInMachine", parameters,
                                     preconditions, add_effects,
                                     delete_effects, ignore_effects, option,
                                     option_vars, null_sampler)
    nsrts.add(place_jug_in_machine_nsrt)

    # TurnMachineOn
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    machine = Variable("?machine", machine_type)
    parameters = [robot, jug, machine]
    option_vars = [robot, machine]
    option = TurnMachineOn
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(JugInMachine, [jug, machine]),
    }
    add_effects = {
        LiftedAtom(JugFilled, [jug]),
        LiftedAtom(MachineOn, [machine]),
        LiftedAtom(PressingButton, [robot, machine]),
    }
    delete_effects = set()
    ignore_effects = set()
    turn_machine_on_nsrt = NSRT("TurnMachineOn", parameters, preconditions,
                                add_effects, delete_effects, ignore_effects,
                                option, option_vars, null_sampler)
    nsrts.add(turn_machine_on_nsrt)

    # PickJugFromMachine
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    machine = Variable("?machine", machine_type)
    parameters = [robot, jug, machine]
    option_vars = [robot, jug]
    option = PickJug
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(JugInMachine, [jug, machine]),
        LiftedAtom(PressingButton, [robot, machine]),
    }
    add_effects = {
        LiftedAtom(Holding, [robot, jug]),
    }
    delete_effects = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(JugInMachine, [jug, machine]),
        LiftedAtom(PressingButton, [robot, machine]),
    }
    ignore_effects = set()
    pick_jug_from_machine_nsrt = NSRT("PickJugFromMachine", parameters,
                                      preconditions, add_effects,
                                      delete_effects, ignore_effects, option,
                                      option_vars, null_sampler)
    nsrts.add(pick_jug_from_machine_nsrt)

    # PourFromNowhere
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    cup = Variable("?cup", cup_type)
    parameters = [robot, jug, cup]
    option_vars = [robot, jug, cup]
    option = Pour
    preconditions = {
        LiftedAtom(Holding, [robot, jug]),
        LiftedAtom(JugFilled, [jug]),
        LiftedAtom(NotAboveCup, [robot, jug]),
    }
    add_effects = {
        LiftedAtom(JugAboveCup, [jug, cup]),
        LiftedAtom(RobotAboveCup, [robot, cup]),
        LiftedAtom(CupFilled, [cup]),
    }
    delete_effects = {
        LiftedAtom(NotAboveCup, [robot, jug]),
    }
    ignore_effects = set()
    pour_from_nowhere_nsrt = NSRT("PourFromNowhere", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, null_sampler)
    nsrts.add(pour_from_nowhere_nsrt)

    # PourFromOtherCup
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    cup = Variable("?cup", cup_type)
    other_cup = Variable("?other_cup", cup_type)
    parameters = [robot, jug, cup, other_cup]
    option_vars = [robot, jug, cup]
    option = Pour
    preconditions = {
        LiftedAtom(Holding, [robot, jug]),
        LiftedAtom(JugFilled, [jug]),
        LiftedAtom(JugAboveCup, [jug, other_cup]),
        LiftedAtom(RobotAboveCup, [robot, other_cup]),
        LiftedAtom(NotSameCup, [cup, other_cup]),
    }
    add_effects = {
        LiftedAtom(JugAboveCup, [jug, cup]),
        LiftedAtom(RobotAboveCup, [robot, cup]),
        LiftedAtom(CupFilled, [cup]),
    }
    delete_effects = {
        LiftedAtom(JugAboveCup, [jug, other_cup]),
        LiftedAtom(RobotAboveCup, [robot, other_cup]),
    }
    ignore_effects = set()
    pour_from_other_cup_nsrt = NSRT("PourFromOtherCup", parameters,
                                    preconditions, add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)
    nsrts.add(pour_from_other_cup_nsrt)

    return nsrts


def _get_satellites_gt_nsrts() -> Set[NSRT]:
    """Create ground truth NSRTs for SatellitesEnv."""
    sat_type, obj_type = _get_types_by_names(CFG.env, ["satellite", "object"])
    Sees, CalibrationTarget, IsCalibrated, HasCamera, HasInfrared, HasGeiger, \
        ShootsChemX, ShootsChemY, HasChemX, HasChemY, CameraReadingTaken, \
        InfraredReadingTaken, GeigerReadingTaken = _get_predicates_by_names(
            CFG.env, ["Sees", "CalibrationTarget", "IsCalibrated",
                      "HasCamera", "HasInfrared", "HasGeiger",
                      "ShootsChemX", "ShootsChemY", "HasChemX", "HasChemY",
                      "CameraReadingTaken", "InfraredReadingTaken",
                      "GeigerReadingTaken"])
    MoveTo, Calibrate, ShootChemX, ShootChemY, UseInstrument = \
        _get_options_by_names(
            CFG.env, ["MoveTo", "Calibrate", "ShootChemX", "ShootChemY",
                      "UseInstrument"])

    nsrts = set()

    # MoveTo
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = MoveTo
    preconditions: Set[LiftedAtom] = set()
    add_effects = {
        LiftedAtom(Sees, [sat, obj]),
    }
    delete_effects: Set[LiftedAtom] = set()
    ignore_effects = {Sees}

    def moveto_sampler(state: State, goal: Set[GroundAtom],
                       rng: np.random.Generator,
                       objs: Sequence[Object]) -> Array:
        del goal  # unused
        _, obj = objs
        obj_x = state.get(obj, "x")
        obj_y = state.get(obj, "y")
        min_dist = SatellitesEnv.radius * 4
        max_dist = SatellitesEnv.fov_dist - SatellitesEnv.radius * 2
        dist = rng.uniform(min_dist, max_dist)
        angle = rng.uniform(-np.pi, np.pi)
        x = obj_x + dist * np.cos(angle)
        y = obj_y + dist * np.sin(angle)
        return np.array([x, y], dtype=np.float32)

    moveto_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                       delete_effects, ignore_effects, option, option_vars,
                       moveto_sampler)
    nsrts.add(moveto_nsrt)

    # Calibrate
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = Calibrate
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(CalibrationTarget, [sat, obj]),
    }
    add_effects = {
        LiftedAtom(IsCalibrated, [sat]),
    }
    delete_effects = set()
    ignore_effects = set()
    calibrate_nsrt = NSRT("Calibrate", parameters, preconditions, add_effects,
                          delete_effects, ignore_effects, option, option_vars,
                          null_sampler)
    nsrts.add(calibrate_nsrt)

    # ShootChemX
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = ShootChemX
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(ShootsChemX, [sat]),
    }
    add_effects = {
        LiftedAtom(HasChemX, [obj]),
    }
    delete_effects = set()
    ignore_effects = set()
    shoot_chem_x_nsrt = NSRT("ShootChemX", parameters, preconditions,
                             add_effects, delete_effects, ignore_effects,
                             option, option_vars, null_sampler)
    nsrts.add(shoot_chem_x_nsrt)

    # ShootChemY
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = ShootChemY
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(ShootsChemY, [sat]),
    }
    add_effects = {
        LiftedAtom(HasChemY, [obj]),
    }
    delete_effects = set()
    ignore_effects = set()
    shoot_chem_y_nsrt = NSRT("ShootChemY", parameters, preconditions,
                             add_effects, delete_effects, ignore_effects,
                             option, option_vars, null_sampler)
    nsrts.add(shoot_chem_y_nsrt)

    # TakeCameraReading
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = UseInstrument
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(IsCalibrated, [sat]),
        LiftedAtom(HasCamera, [sat]),
        # taking a camera reading requires Chemical X
        LiftedAtom(HasChemX, [obj]),
    }
    add_effects = {
        LiftedAtom(CameraReadingTaken, [sat, obj]),
    }
    delete_effects = set()
    ignore_effects = set()
    take_camera_reading_nsrt = NSRT("TakeCameraReading", parameters,
                                    preconditions, add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)
    nsrts.add(take_camera_reading_nsrt)

    # TakeInfraredReading
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = UseInstrument
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(IsCalibrated, [sat]),
        LiftedAtom(HasInfrared, [sat]),
        # taking an infrared reading requires Chemical Y
        LiftedAtom(HasChemY, [obj]),
    }
    add_effects = {
        LiftedAtom(InfraredReadingTaken, [sat, obj]),
    }
    delete_effects = set()
    ignore_effects = set()
    take_infrared_reading_nsrt = NSRT("TakeInfraredReading", parameters,
                                      preconditions, add_effects,
                                      delete_effects, ignore_effects, option,
                                      option_vars, null_sampler)
    nsrts.add(take_infrared_reading_nsrt)

    # TakeGeigerReading
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = UseInstrument
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(IsCalibrated, [sat]),
        LiftedAtom(HasGeiger, [sat]),
        # taking a Geiger reading doesn't require any chemical
    }
    add_effects = {
        LiftedAtom(GeigerReadingTaken, [sat, obj]),
    }
    delete_effects = set()
    ignore_effects = set()
    take_geiger_reading_nsrt = NSRT("TakeGeigerReading", parameters,
                                    preconditions, add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)
    nsrts.add(take_geiger_reading_nsrt)

    return nsrts


def _get_behavior_gt_nsrts() -> Set[NSRT]:  # pragma: no cover
    """Create ground truth nsrts for BehaviorEnv."""
    # Without this cast, mypy complains:
    #   "BaseEnv" has no attribute "object_to_ig_object"
    # and using isinstance(env, BehaviorEnv) instead does not work.
    env_base = get_or_create_env("behavior")
    env = cast(BehaviorEnv, env_base)

    # NOTE: These two methods below are necessary to help instantiate
    # all combinations of types for predicates (e.g. reachable(robot, book),
    # reachable(robot, fridge), etc). If we had support for hierarchical types
    # such that all types could inherit from 'object' we would not need
    # to perform this combinatorial enumeration.

    type_name_to_type = {t.name: t for t in env.types}
    pred_name_to_pred = {p.name: p for p in env.predicates}

    def _get_lifted_atom(base_pred_name: str,
                         objects: Sequence[Variable]) -> LiftedAtom:
        pred = _get_predicate(base_pred_name, [o.type for o in objects])
        return LiftedAtom(pred, objects)

    def _get_predicate(base_pred_name: str,
                       types: Sequence[Type]) -> Predicate:
        if len(types) == 0:
            return pred_name_to_pred[base_pred_name]
        type_names = "-".join(t.name for t in types)
        pred_name = f"{base_pred_name}-{type_names}"
        return pred_name_to_pred[pred_name]

    # We start by creating reachable predicates for the agent and
    # all possible other types. These predicates will
    # be used as ignore effects for navigateTo operators.
    reachable_predicates = set()
    for reachable_pred_type in env.types:
        # We don't care about the "reachable(agent)" predicate
        # since it will always be False.
        if reachable_pred_type.name == "agent":
            continue
        reachable_predicates.add(
            _get_predicate("reachable", [reachable_pred_type]))

    nsrts = set()
    op_name_count_nav = itertools.count()
    op_name_count_pick = itertools.count()
    op_name_count_place = itertools.count()
    op_name_count_open = itertools.count()
    op_name_count_close = itertools.count()
    op_name_count_place_inside = itertools.count()

    # Dummy sampler definition. Useful for open and close.
    def dummy_param_sampler(state: State, goal: Set[GroundAtom],
                            rng: Generator,
                            objects: Sequence["URDFObject"]) -> Array:
        """Dummy sampler."""
        del state, goal, rng, objects
        return np.array([0.0, 0.0, 0.0])

    # NavigateTo sampler definition.
    MAX_NAVIGATION_SAMPLES = 50

    def navigate_to_param_sampler(state: State, goal: Set[GroundAtom],
                                  rng: Generator,
                                  objects: Sequence["URDFObject"]) -> Array:
        """Sampler for navigateTo option.

        Loads the entire iGibson env to perform collision checking and
        generates a finite number of samples such that the returned
        sample is not in collision with any object in the environment.
        """
        del goal
        # Get the current env for collision checking.
        env = get_or_create_env("behavior")
        assert isinstance(env, BehaviorEnv)
        load_checkpoint_state(state, env)
        # The navigation nsrts are designed such that the target
        # obj is always last in the params list.
        obj_to_sample_near = objects[-1]
        closeness_limit = 2.00
        nearness_limit = 0.15
        distance = nearness_limit + (
            (closeness_limit - nearness_limit) * rng.random())
        yaw = rng.random() * (2 * np.pi) - np.pi
        x = distance * np.cos(yaw)
        y = distance * np.sin(yaw)
        sampler_output = np.array([x, y])
        # The below while loop avoids sampling values that would put the
        # robot in collision with some object in the environment. It may
        # not always succeed at this and will exit after a certain number
        # of tries.
        logging.info("Sampling params for navigation...")
        num_samples_tried = 0
        while (check_nav_end_pose(env.igibson_behavior_env, obj_to_sample_near,
                                  sampler_output) is None):
            distance = closeness_limit * rng.random()
            yaw = rng.random() * (2 * np.pi) - np.pi
            x = distance * np.cos(yaw)
            y = distance * np.sin(yaw)
            sampler_output = np.array([x, y])
            # NOTE: In many situations, it is impossible to find a good sample
            # no matter how many times we try. Thus, we break this loop after
            # a certain number of tries so the planner will backtrack.
            if num_samples_tried > MAX_NAVIGATION_SAMPLES:
                break
            num_samples_tried += 1

        return sampler_output

    # Grasp sampler definition.
    def grasp_obj_param_sampler(state: State, goal: Set[GroundAtom],
                                rng: Generator,
                                objects: Sequence["URDFObject"]) -> Array:
        """Sampler for grasp option."""
        del state, goal, objects
        x_offset = (rng.random() * 0.4) - 0.2
        y_offset = (rng.random() * 0.4) - 0.2
        z_offset = rng.random() * 0.2
        return np.array([x_offset, y_offset, z_offset])

    # Place OnTop sampler definition.
    MAX_PLACEONTOP_SAMPLES = 25

    def place_ontop_obj_pos_sampler(
            state: State, goal: Set[GroundAtom], rng: Generator,
            objects: Union["URDFObject", "RoomFloor"]) -> Array:
        """Sampler for placeOnTop option."""
        del goal
        assert rng is not None
        # objA is the object the robot is currently holding, and
        # objB is the surface that it must place onto.
        # The BEHAVIOR NSRT's are designed such that objA is the 0th
        # argument, and objB is the last.
        objA = objects[0]
        objB = objects[-1]

        params = {
            "max_angle_with_z_axis": 0.17,
            "bimodal_stdev_fraction": 1e-6,
            "bimodal_mean_fraction": 1.0,
            "max_sampling_attempts": 50,
            "aabb_offset": 0.01,
        }
        aabb = get_aabb(objA.get_body_id())
        aabb_extent = get_aabb_extent(aabb)

        random_seed_int = rng.integers(10000000)
        sampling_results = sampling_utils.sample_cuboid_on_object(
            objB,
            num_samples=1,
            cuboid_dimensions=aabb_extent,
            axis_probabilities=[0, 0, 1],
            refuse_downwards=True,
            random_seed_number=random_seed_int,
            **params,
        )

        if sampling_results[0] is None or sampling_results[0][0] is None:
            # If sampling fails, fall back onto custom-defined object-specific
            # samplers
            if "shelf" in objB.name:
                # Get the current env for collision checking.
                env = get_or_create_env("behavior")
                assert isinstance(env, BehaviorEnv)
                load_checkpoint_state(state, env)
                objB_sampling_bounds = objB.bounding_box / 2
                sample_params = np.array([
                    rng.uniform(-objB_sampling_bounds[0],
                                objB_sampling_bounds[0]),
                    rng.uniform(-objB_sampling_bounds[1],
                                objB_sampling_bounds[1]),
                    rng.uniform(-objB_sampling_bounds[2] + 0.3,
                                objB_sampling_bounds[1]) + 0.3
                ])
                logging.info("Sampling params for placeOnTop shelf...")
                num_samples_tried = 0
                while not check_hand_end_pose(env.igibson_behavior_env, objB,
                                              sample_params):
                    sample_params = np.array([
                        rng.uniform(-objB_sampling_bounds[0],
                                    objB_sampling_bounds[0]),
                        rng.uniform(-objB_sampling_bounds[1],
                                    objB_sampling_bounds[1]),
                        rng.uniform(-objB_sampling_bounds[2] + 0.3,
                                    objB_sampling_bounds[1]) + 0.3
                    ])
                    # NOTE: In many situations, it is impossible to find a
                    # good sample no matter how many times we try. Thus, we
                    # break this loop after a certain number of tries so the
                    # planner will backtrack.
                    if num_samples_tried > MAX_PLACEONTOP_SAMPLES:
                        break
                    num_samples_tried += 1
                return sample_params
            # If there's no object specific sampler, just return a
            # random sample.
            return np.array([
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.3, 1.0)
            ])

        rnd_params = np.subtract(sampling_results[0][0], objB.get_position())
        return rnd_params

    # Place Inside sampler definition.
    def place_inside_obj_pos_sampler(
            state: State, goal: Set[GroundAtom], rng: Generator,
            objects: Union["URDFObject", "RoomFloor"]) -> Array:
        """Sampler for placeOnTop option."""
        del state, goal
        assert rng is not None
        # objA is the object the robot is currently holding, and
        # objB is the surface that it must place onto.
        # The BEHAVIOR NSRT's are designed such that objA is the 0th
        # argument, and objB is the last.
        objA = objects[0]
        objB = objects[-1]

        params = {
            "max_angle_with_z_axis": 0.17,
            "bimodal_stdev_fraction": 0.4,
            "bimodal_mean_fraction": 0.5,
            "max_sampling_attempts": 100,
            "aabb_offset": -0.01,
        }
        aabb = get_aabb(objA.get_body_id())
        aabb_extent = get_aabb_extent(aabb)

        random_seed_int = rng.integers(10000000)
        sampling_results = sampling_utils.sample_cuboid_on_object(
            objB,
            num_samples=1,
            cuboid_dimensions=aabb_extent,
            axis_probabilities=[0, 0, 1],
            refuse_downwards=True,
            random_seed_number=random_seed_int,
            **params,
        )

        if sampling_results[0] is None or sampling_results[0][0] is None:
            # If sampling fails, returns a random set of params
            return np.array([
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.3, 1.0)
            ])

        rnd_params = np.subtract(sampling_results[0][0], objB.get_position())
        return rnd_params

    for option in env.options:
        split_name = option.name.split("-")
        base_option_name = split_name[0]
        option_arg_type_names = split_name[1:]
        # Edge case t-shirt gets parsed wrong with split("").
        if option_arg_type_names[0] == "t" and option_arg_type_names[
                1] == "shirt":
            option_arg_type_names = ["t-shirt"] + option_arg_type_names[2:]

        if base_option_name == "NavigateTo":
            assert len(option_arg_type_names) == 1
            target_obj_type_name = option_arg_type_names[0]
            target_obj_type = type_name_to_type[target_obj_type_name]
            target_obj = Variable("?targ", target_obj_type)
            # We don't need an NSRT to navigate to the agent or the room floor.
            if target_obj_type_name in ["agent", "room_floor"]:
                continue
            # Navigate To.
            parameters = [target_obj]
            option_vars = [target_obj]
            preconditions: Set[LiftedAtom] = set()
            add_effects = {_get_lifted_atom("reachable", [target_obj])}
            reachable_nothing = _get_lifted_atom("reachable-nothing", [])
            delete_effects = {reachable_nothing}
            nsrt = NSRT(
                f"{option.name}-{next(op_name_count_nav)}", parameters,
                preconditions, add_effects, delete_effects,
                reachable_predicates, option, option_vars,
                lambda s, g, r, o: navigate_to_param_sampler(
                    s,
                    g,
                    r,
                    [
                        env.object_to_ig_object(o_i)
                        if isinstance(o_i, Object) else o_i for o_i in o
                    ],
                ))
            nsrts.add(nsrt)

        elif base_option_name == "Grasp":
            assert len(option_arg_type_names) == 1
            target_obj_type_name = option_arg_type_names[0]
            target_obj_type = type_name_to_type[target_obj_type_name]
            target_obj = Variable("?targ", target_obj_type)
            # If the target object is not in these object types, we do not
            # have to make a NSRT with this type.
            if target_obj_type.name not in PICK_PLACE_OBJECT_TYPES:
                continue
            # Grasp an object from ontop some surface.
            for surf_obj_type in sorted(env.types):
                # If the surface object is not in these object types, we do not
                # have to make a NSRT with this type.
                if surf_obj_type.name not in PLACE_ONTOP_SURFACE_OBJECT_TYPES\
                    | PLACE_INTO_SURFACE_OBJECT_TYPES:
                    continue
                surf_obj = Variable("?surf", surf_obj_type)
                parameters = [target_obj, surf_obj]
                option_vars = [target_obj]
                handempty = _get_lifted_atom("handempty", [])
                targ_reachable = _get_lifted_atom("reachable", [target_obj])
                targ_holding = _get_lifted_atom("holding", [target_obj])
                ontop = _get_lifted_atom("ontop", [target_obj, surf_obj])
                inside = _get_lifted_atom("inside", [target_obj, surf_obj])
                preconditions_ontop = {handempty, targ_reachable, ontop}
                preconditions_inside = {handempty, targ_reachable, inside}
                add_effects = {targ_holding}
                delete_effects_ontop = {handempty, ontop, targ_reachable}
                delete_effects_inside = {handempty, inside, targ_reachable}
                # NSRT for grasping an object from ontop an object.
                nsrt = NSRT(
                    f"{option.name}-{next(op_name_count_pick)}",
                    parameters,
                    preconditions_ontop,
                    add_effects,
                    delete_effects_ontop,
                    set(),
                    option,
                    option_vars,
                    grasp_obj_param_sampler,
                )
                nsrts.add(nsrt)
                # NSRT for grasping an object from inside an object.
                nsrt = NSRT(
                    f"{option.name}-{next(op_name_count_pick)}",
                    parameters,
                    preconditions_inside,
                    add_effects,
                    delete_effects_inside,
                    set(),
                    option,
                    option_vars,
                    grasp_obj_param_sampler,
                )
                nsrts.add(nsrt)

        elif base_option_name == "PlaceOnTop":
            assert len(option_arg_type_names) == 1
            surf_obj_type_name = option_arg_type_names[0]
            surf_obj_type = type_name_to_type[surf_obj_type_name]
            surf_obj = Variable("?surf", surf_obj_type)
            # We don't need an NSRT to place objects on top of the
            # agent because this is never necessary.
            if surf_obj.type.name == "agent":
                continue
            # If the surface object is not in these object types, we do not
            # have to make a NSRT with this type.
            if surf_obj_type.name not in PLACE_ONTOP_SURFACE_OBJECT_TYPES:
                continue
            # We need to place the object we're holding!
            for held_obj_types in sorted(env.types):
                # If the held object is not in these object types, we do not
                # have to make a NSRT with this type.
                if held_obj_types.name not in PICK_PLACE_OBJECT_TYPES:
                    continue
                held_obj = Variable("?held", held_obj_types)
                parameters = [held_obj, surf_obj]
                option_vars = [surf_obj]
                handempty = _get_lifted_atom("handempty", [])
                held_holding = _get_lifted_atom("holding", [held_obj])
                surf_reachable = _get_lifted_atom("reachable", [surf_obj])
                held_reachable = _get_lifted_atom("reachable", [held_obj])
                ontop = _get_lifted_atom("ontop", [held_obj, surf_obj])
                preconditions = {held_holding, surf_reachable}
                add_effects = {ontop, handempty, held_reachable}
                delete_effects = {held_holding}
                nsrt = NSRT(
                    f"{option.name}-{next(op_name_count_place)}",
                    parameters,
                    preconditions,
                    add_effects,
                    delete_effects,
                    set(),
                    option,
                    option_vars,
                    lambda s, g, r, o: place_ontop_obj_pos_sampler(
                        s,
                        g,
                        objects=[
                            env.object_to_ig_object(o_i)
                            if isinstance(o_i, Object) else o_i for o_i in o
                        ],
                        rng=r,
                    ),
                )
                nsrts.add(nsrt)

        elif base_option_name == "Open":
            assert len(option_arg_type_names) == 1
            open_obj_type_name = option_arg_type_names[0]
            open_obj_type = type_name_to_type[open_obj_type_name]
            # We don't need an NSRT to open objects that are not
            # openable.
            if open_obj_type.name not in OPENABLE_OBJECT_TYPES:
                continue
            open_obj = Variable("?obj", open_obj_type)
            # We don't need an NSRT to open the agent.
            if open_obj_type_name == "agent":
                continue
            parameters = [open_obj]
            option_vars = [open_obj]
            openable_predicate = _get_lifted_atom("openable", [open_obj])
            closed_predicate = _get_lifted_atom("closed", [open_obj])
            preconditions = {
                _get_lifted_atom("reachable", [open_obj]), closed_predicate,
                openable_predicate
            }
            add_effects = {_get_lifted_atom("open", [open_obj])}
            delete_effects = {closed_predicate}
            nsrt = NSRT(
                f"{option.name}-{next(op_name_count_open)}", parameters,
                preconditions, add_effects, delete_effects, set(), option,
                option_vars, lambda s, g, r, o: dummy_param_sampler(
                    s,
                    g,
                    r,
                    [
                        env.object_to_ig_object(o_i)
                        if isinstance(o_i, Object) else o_i for o_i in o
                    ],
                ))
            nsrts.add(nsrt)

        elif base_option_name == "Close":
            assert len(option_arg_type_names) == 1
            close_obj_type_name = option_arg_type_names[0]
            close_obj_type = type_name_to_type[close_obj_type_name]
            close_obj = Variable("?obj", close_obj_type)
            # We don't need an NSRT to close objects that are not
            # openable.
            if close_obj_type.name not in OPENABLE_OBJECT_TYPES:
                continue
            parameters = [close_obj]
            option_vars = [close_obj]
            preconditions = {
                _get_lifted_atom("reachable", [close_obj]),
                _get_lifted_atom("open", [close_obj]),
                _get_lifted_atom("openable", [close_obj])
            }
            add_effects = {_get_lifted_atom("closed", [close_obj])}
            delete_effects = {_get_lifted_atom("open", [close_obj])}
            nsrt = NSRT(
                f"{option.name}-{next(op_name_count_close)}", parameters,
                preconditions, add_effects, delete_effects, set(), option,
                option_vars, lambda s, g, r, o: dummy_param_sampler(
                    s,
                    g,
                    r,
                    [
                        env.object_to_ig_object(o_i)
                        if isinstance(o_i, Object) else o_i for o_i in o
                    ],
                ))
            nsrts.add(nsrt)

        elif base_option_name == "PlaceInside":
            assert len(option_arg_type_names) == 1
            surf_obj_type_name = option_arg_type_names[0]
            surf_obj_type = type_name_to_type[surf_obj_type_name]
            surf_obj = Variable("?surf", surf_obj_type)
            # We don't need an NSRT to place objects inside the
            # agent or room floor because this is not possible.
            if surf_obj_type.name in ["agent", "room_floor"]:
                continue
            # If the surface object is not in these object types, we do not
            # have to make a NSRT with this type.
            if surf_obj_type.name not in PLACE_INTO_SURFACE_OBJECT_TYPES:
                continue
            # We need to place the object we're holding. Note that we create
            # two different place-inside NSRTs: one for when the object we are
            # placing into is `openable`, and one for other cases.
            for held_obj_types in sorted(env.types):
                # If the held object is not in these object types, we do not
                # have to make a NSRT with this type.
                if held_obj_types.name not in PICK_PLACE_OBJECT_TYPES or \
                    held_obj_types.name == surf_obj_type.name:
                    continue
                held_obj = Variable("?held", held_obj_types)
                parameters = [held_obj, surf_obj]
                option_vars = [surf_obj]
                handempty = _get_lifted_atom("handempty", [])
                held_holding = _get_lifted_atom("holding", [held_obj])
                surf_reachable = _get_lifted_atom("reachable", [surf_obj])
                held_reachable = _get_lifted_atom("reachable", [held_obj])
                inside = _get_lifted_atom("inside", [held_obj, surf_obj])
                openable_surf = _get_lifted_atom("openable", [surf_obj])
                not_openable_surf = _get_lifted_atom("not-openable",
                                                     [surf_obj])
                open_surf = _get_lifted_atom("open", [surf_obj])
                preconditions_openable = {
                    held_holding, surf_reachable, openable_surf, open_surf
                }
                preconditions_not_openable = {
                    held_holding, surf_reachable, not_openable_surf
                }
                add_effects = {inside, handempty, held_reachable}
                delete_effects = {held_holding}
                # NSRT for placing into an openable surface.
                openable_nsrt = NSRT(
                    f"{option.name}-{next(op_name_count_place_inside)}",
                    parameters,
                    preconditions_openable,
                    add_effects,
                    delete_effects,
                    set(),
                    option,
                    option_vars,
                    lambda s, g, r, o: place_inside_obj_pos_sampler(
                        s,
                        g,
                        objects=[
                            env.object_to_ig_object(o_i)
                            if isinstance(o_i, Object) else o_i for o_i in o
                        ],
                        rng=r,
                    ),
                )
                nsrts.add(openable_nsrt)
                # NSRT for placing into a not-openable surface.
                not_openable_nsrt = NSRT(
                    f"{option.name}-{next(op_name_count_place_inside)}",
                    parameters,
                    preconditions_not_openable,
                    add_effects,
                    delete_effects,
                    set(),
                    option,
                    option_vars,
                    lambda s, g, r, o: place_inside_obj_pos_sampler(
                        s,
                        g,
                        objects=[
                            env.object_to_ig_object(o_i)
                            if isinstance(o_i, Object) else o_i for o_i in o
                        ],
                        rng=r,
                    ),
                )
                nsrts.add(not_openable_nsrt)

        else:
            raise ValueError(
                f"Unexpected base option name: {base_option_name}")

    return nsrts


def _get_pddl_env_gt_nsrts(name: str) -> Set[NSRT]:
    env = get_or_create_env(name)
    assert isinstance(env, _PDDLEnv)

    nsrts = set()
    option_name_to_option = {o.name: o for o in env.options}

    for strips_op in env.strips_operators:
        option = option_name_to_option[strips_op.name]
        nsrt = strips_op.make_nsrt(
            option=option,
            option_vars=strips_op.parameters,
            sampler=null_sampler,
        )
        nsrts.add(nsrt)

    return nsrts
