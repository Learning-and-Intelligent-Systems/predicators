"""Test cases for the oracle approach class."""

from typing import Any, Dict, List, Set

import numpy as np
import pytest

from predicators import utils
from predicators.approaches.oracle_approach import OracleApproach
from predicators.envs.blocks import BlocksEnv
from predicators.envs.cluttered_table import ClutteredTableEnv, \
    ClutteredTablePlaceEnv
from predicators.envs.coffee import CoffeeEnv
from predicators.envs.cover import CoverEnv, CoverEnvHierarchicalTypes, \
    CoverEnvRegrasp, CoverEnvTypedOptions, CoverMultistepOptions
from predicators.envs.doors import DoorsEnv
from predicators.envs.painting import PaintingEnv
from predicators.envs.pddl_env import FixedTasksBlocksPDDLEnv, \
    ProceduralTasksBlocksPDDLEnv, ProceduralTasksDeliveryPDDLEnv, \
    ProceduralTasksEasyDeliveryPDDLEnv
from predicators.envs.playroom import PlayroomEnv
from predicators.envs.repeated_nextto import RepeatedNextToEnv, \
    RepeatedNextToSingleOptionEnv
from predicators.envs.repeated_nextto_painting import RepeatedNextToPaintingEnv
from predicators.envs.satellites import SatellitesEnv, SatellitesSimpleEnv
from predicators.envs.screws import ScrewsEnv
from predicators.envs.stick_button import StickButtonEnv
from predicators.envs.tools import ToolsEnv
from predicators.envs.touch_point import TouchPointEnv
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.option_model import _OracleOptionModel
from predicators.settings import CFG
from predicators.structs import Action, Variable

ENV_NAME_AND_CLS = [
    ("cover", CoverEnv), ("cover_typed_options", CoverEnvTypedOptions),
    ("cover_hierarchical_types", CoverEnvHierarchicalTypes),
    ("cover_regrasp", CoverEnvRegrasp),
    ("cover_multistep_options", CoverMultistepOptions),
    ("cluttered_table", ClutteredTableEnv),
    ("cluttered_table_place", ClutteredTablePlaceEnv), ("blocks", BlocksEnv),
    ("painting", PaintingEnv), ("tools", ToolsEnv), ("playroom", PlayroomEnv),
    ("repeated_nextto", RepeatedNextToEnv),
    ("repeated_nextto_single_option", RepeatedNextToSingleOptionEnv),
    ("satellites", SatellitesEnv), ("satellites_simple", SatellitesSimpleEnv),
    ("screws", ScrewsEnv),
    ("repeated_nextto_painting", RepeatedNextToPaintingEnv),
    ("pddl_blocks_fixed_tasks", FixedTasksBlocksPDDLEnv),
    ("pddl_blocks_procedural_tasks", ProceduralTasksBlocksPDDLEnv),
    ("pddl_delivery_procedural_tasks", ProceduralTasksDeliveryPDDLEnv),
    ("pddl_easy_delivery_procedural_tasks",
     ProceduralTasksEasyDeliveryPDDLEnv), ("touch_point", TouchPointEnv),
    ("stick_button", StickButtonEnv), ("doors", DoorsEnv),
    ("coffee", CoffeeEnv)
]

# For each environment name in ENV_NAME_AND_CLS, a list of additional
# configuration arguments to pass into reset_config() when running the
# oracle approach. Each element in this list defines an experiment.
# See the usage in test_oracle_approach().
EXTRA_ARGS_ORACLE_APPROACH: Dict[str, List[Dict[str, Any]]] = {
    name: [{}]
    for name, _ in ENV_NAME_AND_CLS
}
EXTRA_ARGS_ORACLE_APPROACH["cover_multistep_options"] = [
    {
        "cover_multistep_degenerate_oracle_samplers": False,
        "cover_multistep_thr_percent": 0.99,
        "cover_multistep_bhr_percent": 0.99
    },
    {
        "cover_multistep_degenerate_oracle_samplers": False,
        "cover_multistep_thr_percent": 0.99,
        "cover_multistep_bhr_percent": 0.99
    },
    {
        "cover_multistep_degenerate_oracle_samplers": True,
        "cover_multistep_thr_percent": 0.99,
        "cover_multistep_bhr_percent": 0.99,
        "num_train_tasks": 3,  # the third task has two goal atoms
        "num_test_tasks": 3,
    },
    {
        "cover_multistep_bimodal_goal": True,
        "cover_multistep_goal_conditioned_sampling": True,
        "cover_num_blocks": 1,
        "cover_num_targets": 1,
        "cover_block_widths": [0.12],
        "cover_target_widths": [0.1],
        "cover_multistep_thr_percent": 0.3,
        "cover_multistep_bhr_percent": 0.99,
        "sesame_max_skeletons_optimized": 1,
        "sesame_max_samples_per_step": 1
    },
]
EXTRA_ARGS_ORACLE_APPROACH["cluttered_table"] = [
    {
        "cluttered_table_num_cans_train": 3,
        "cluttered_table_num_cans_test": 3
    },
]
EXTRA_ARGS_ORACLE_APPROACH["cluttered_table_place"] = [
    {
        "cluttered_table_num_cans_train": 3,
        "cluttered_table_num_cans_test": 3
    },
]
EXTRA_ARGS_ORACLE_APPROACH["painting"] = [
    {
        "painting_initial_holding_prob": 1.0,
    },
]
EXTRA_ARGS_ORACLE_APPROACH["repeated_nextto_painting"] = [
    {
        "rnt_painting_num_objs_train": [1, 2],
        "rnt_painting_num_objs_test": [3, 4]
    },
]
EXTRA_ARGS_ORACLE_APPROACH["tools"] = [
    {
        "tools_num_items_train": [2],
        "tools_num_items_test": [2]
    },
]
EXTRA_ARGS_ORACLE_APPROACH["stick_button"] = [
    {
        "stick_button_num_buttons_train": [1],
        "stick_button_num_buttons_test": [2],
        "stick_button_disable_angles": False
    },
    {
        "stick_button_num_buttons_train": [1],
        "stick_button_num_buttons_test": [2],
        "stick_button_disable_angles": True
    },
]
EXTRA_ARGS_ORACLE_APPROACH["doors"] = [{
    "doors_room_map_size": 2,
    "doors_min_room_exists_frac": 1.0,
    "doors_max_room_exists_frac": 1.0,
    "doors_birrt_smooth_amt": 0,
    "doors_min_obstacles_per_room": 1,
    "doors_max_obstacles_per_room": 1,
}]
EXTRA_ARGS_ORACLE_APPROACH["pddl_delivery_procedural_tasks"] = [
    {
        "pddl_delivery_procedural_train_min_num_locs": 2,
        "pddl_delivery_procedural_train_max_num_locs": 3,
        "pddl_delivery_procedural_train_min_want_locs": 1,
        "pddl_delivery_procedural_train_max_want_locs": 1,
        "pddl_delivery_procedural_train_min_extra_newspapers": 0,
        "pddl_delivery_procedural_train_max_extra_newspapers": 1,
        "pddl_delivery_procedural_test_min_num_locs": 2,
        "pddl_delivery_procedural_test_max_num_locs": 3,
        "pddl_delivery_procedural_test_min_want_locs": 1,
        "pddl_delivery_procedural_test_max_want_locs": 1,
        "pddl_delivery_procedural_test_min_extra_newspapers": 0,
        "pddl_delivery_procedural_test_max_extra_newspapers": 1,
        "sesame_use_visited_state_set": True,
    },
]
EXTRA_ARGS_ORACLE_APPROACH["pddl_easy_delivery_procedural_tasks"] = [
    {
        "pddl_easy_delivery_procedural_train_min_num_locs": 2,
        "pddl_easy_delivery_procedural_train_max_num_locs": 3,
        "pddl_easy_delivery_procedural_train_min_want_locs": 1,
        "pddl_easy_delivery_procedural_train_max_want_locs": 1,
        "pddl_easy_delivery_procedural_train_min_extra_newspapers": 0,
        "pddl_easy_delivery_procedural_train_max_extra_newspapers": 1,
        "pddl_easy_delivery_procedural_test_min_num_locs": 2,
        "pddl_easy_delivery_procedural_test_max_num_locs": 3,
        "pddl_easy_delivery_procedural_test_min_want_locs": 1,
        "pddl_easy_delivery_procedural_test_max_want_locs": 1,
        "pddl_easy_delivery_procedural_test_min_extra_newspapers": 0,
        "pddl_easy_delivery_procedural_test_max_extra_newspapers": 1,
        "sesame_use_visited_state_set": True,
    },
]
EXTRA_ARGS_ORACLE_APPROACH["behavior_tasks"] = [
    {
        "option_model_name": "oracle_behavior",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "behavior_scene_name": "Pomaria_1_int",
        "behavior_task_list": "\"[sorting_books]\"",
        "offline_data_planning_timeout": 30000.0,
    },
]


def _policy_solves_task(policy, task, simulator):
    """Helper method used in this file."""
    traj = utils.run_policy_with_simulator(policy,
                                           simulator,
                                           task.init,
                                           task.goal_holds,
                                           max_num_steps=CFG.horizon)
    return task.goal_holds(traj.states[-1])


@pytest.mark.parametrize("env_name,env_cls", ENV_NAME_AND_CLS)
def test_oracle_approach(env_name, env_cls):
    """Tests for OracleApproach class with all environments."""
    for extra_args in EXTRA_ARGS_ORACLE_APPROACH[env_name]:
        args = {
            "env": env_name,
            **extra_args,
        }
        # Default to 2 train and test tasks, but allow them to be specified in
        # the extra args too.
        if "num_train_tasks" not in args:
            args["num_train_tasks"] = 2
        if "num_test_tasks" not in args:
            args["num_test_tasks"] = 2
        utils.reset_config(args)
        env = env_cls()
        train_tasks = env.get_train_tasks()
        approach = OracleApproach(env.predicates, env.options, env.types,
                                  env.action_space, train_tasks)
        assert not approach.is_learning_based
        for task in train_tasks:
            policy = approach.solve(task, timeout=500)
            assert _policy_solves_task(policy, task, env.simulate)
        for task in env.get_test_tasks():
            policy = approach.solve(task, timeout=500)
            assert _policy_solves_task(policy, task, env.simulate)
    # Tests if OracleApproach can load _OracleOptionModel
    assert isinstance(approach.get_option_model(), _OracleOptionModel)


def test_get_gt_nsrts():
    """Test get_gt_nsrts alone."""
    utils.reset_config({"env": "not a real environment"})
    with pytest.raises(NotImplementedError):
        get_gt_nsrts(set(), set())


@pytest.mark.parametrize("env_name,env_cls", ENV_NAME_AND_CLS)
def test_nsrt_parameters(env_name, env_cls):
    """Checks assumptions on the oracle operators for all environments."""
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })
    env = env_cls()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    for nsrt in nsrts:
        effects_vars: Set[Variable] = set()
        precond_vars: Set[Variable] = set()
        for lifted_atom in nsrt.add_effects:
            effects_vars |= set(lifted_atom.variables)
        for lifted_atom in nsrt.delete_effects:
            effects_vars |= set(lifted_atom.variables)
        for lifted_atom in nsrt.preconditions:
            precond_vars |= set(lifted_atom.variables)
        assert set(nsrt.option_vars).issubset(nsrt.parameters), \
            f"Option variables is not a subset of parameters in {nsrt.name}"
        for var in nsrt.parameters:
            assert var in nsrt.option_vars or var in effects_vars, \
                f"Variable {var} not found in effects or option of {nsrt.name}"
        assert set(nsrt.parameters) == (set(nsrt.option_vars) | precond_vars |
                                        effects_vars), \
            f"Set of parameters is not the union of option and operator " \
            f"variables in {nsrt.name}"


def test_cover_get_gt_nsrts():
    """Tests for get_gt_nsrts in CoverEnv."""
    utils.reset_config({
        "env": "cover",
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })
    # All predicates and options
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    assert len(nsrts) == 2
    pick_nsrt, place_nsrt = sorted(nsrts, key=lambda o: o.name)
    assert pick_nsrt.name == "Pick"
    assert place_nsrt.name == "Place"
    train_task = env.get_train_tasks()[0]
    state = train_task.init
    block0, _, _, target0, _ = list(state)
    assert block0.name == "block0"
    assert target0.name == "target0"
    pick0_nsrt = pick_nsrt.ground([block0])
    rng = np.random.default_rng(123)
    pick_option = pick0_nsrt.sample_option(state, train_task.goal, rng)
    pick_action = pick_option.policy(state)
    assert env.action_space.contains(pick_action.arr)
    state = env.simulate(state, pick_action)
    place0_nsrt = place_nsrt.ground([block0, target0])
    place_option = place0_nsrt.sample_option(state, train_task.goal, rng)
    place_action = place_option.policy(state)
    assert env.action_space.contains(place_action.arr)
    # Excluded option
    assert get_gt_nsrts(env.predicates, set()) == set()
    # Excluded predicate
    predicates = {p for p in env.predicates if p.name != "Holding"}
    nsrts = get_gt_nsrts(predicates, env.options)
    assert len(nsrts) == 2
    pick_nsrt, place_nsrt = sorted(nsrts, key=lambda o: o.name)
    for atom in pick_nsrt.preconditions:
        assert atom.predicate.name != "Holding"
    assert len(pick_nsrt.add_effects) == 0
    for atom in pick_nsrt.delete_effects:
        assert atom.predicate.name != "Holding"


@pytest.mark.parametrize("place_version", [True, False])
def test_cluttered_table_get_gt_nsrts(place_version):
    """Tests for get_gt_nsrts in ClutteredTableEnv."""
    if not place_version:
        utils.reset_config({
            "env": "cluttered_table",
            # Keep num_train_tasks high enough to ensure hitting the
            # EnvironmentFailure check below at least once
            "num_train_tasks": 5,
            "num_test_tasks": 2
        })
        # All predicates and options
        env = ClutteredTableEnv()
    else:
        utils.reset_config({
            "env": "cluttered_table_place",
            # Higher num of train tasks needed for full coverage.
            "num_train_tasks": 5,
            "num_test_tasks": 2
        })
        env = ClutteredTablePlaceEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    assert len(nsrts) == 2
    if not place_version:
        dump_nsrt, grasp_nsrt = sorted(nsrts, key=lambda o: o.name)
        assert dump_nsrt.name == "Dump"
        assert grasp_nsrt.name == "Grasp"
    else:
        grasp_nsrt, place_nsrt = sorted(nsrts, key=lambda o: o.name)
        assert grasp_nsrt.name == "Grasp"
        assert place_nsrt.name == "Place"
    train_tasks = env.get_train_tasks()
    for (i, task) in enumerate(train_tasks):
        if i < len(train_tasks) / 2:
            utils.reset_config(
                {"cluttered_table_place_goal_conditioned_sampling": False})
        else:
            utils.reset_config(
                {"cluttered_table_place_goal_conditioned_sampling": True})
        state = task.init
        if not place_version:
            can0, can1, _, can3, _ = list(state)
            assert can0.name == "can0"
            assert can3.name == "can3"
        else:
            can0, can1 = list(state)
            assert can0.name == "can0"
            assert can1.name == "can1"
        grasp0_nsrt = grasp_nsrt.ground([can0])
        with pytest.raises(AssertionError):
            grasp_nsrt.ground([])
        rng = np.random.default_rng(123)
        if i == 0 and place_version:
            # This case checks for exception when placing collides.
            grasp_action = Action(
                np.array([0.2, 0.1, 0.2, 0.6], dtype=np.float32))
        else:
            grasp_option = grasp0_nsrt.sample_option(state, task.goal, rng)
            grasp_action = grasp_option.policy(state)
        assert env.action_space.contains(grasp_action.arr)
        try:
            state = env.simulate(state, grasp_action)
        except utils.EnvironmentFailure as e:
            assert len(e.info["offending_objects"]) == 1
        if not place_version:
            dump0_nsrt = dump_nsrt.ground([can3])
            with pytest.raises(AssertionError):
                dump_nsrt.ground([can3, can1])
            dump_option = dump0_nsrt.sample_option(state, task.goal, rng)
            dump_action = dump_option.policy(state)
            assert env.action_space.contains(dump_action.arr)
            env.simulate(state, dump_action)  # never raises EnvironmentFailure
        else:
            place1_nsrt = place_nsrt.ground([can1])
            with pytest.raises(AssertionError):
                place_nsrt.ground([can0, can1])
            if i == 0:
                # This case checks for exception when placing collides.
                place_action = Action(
                    np.array([0.2, 0.1, 0.1, 0.85], dtype=np.float32))
                assert env.action_space.contains(place_action.arr)
            else:
                place_option = place1_nsrt.sample_option(state, task.goal, rng)
                place_action = place_option.policy(state)
                assert env.action_space.contains(place_action.arr)
            try:
                env.simulate(state, place_action)
            except utils.EnvironmentFailure as e:
                assert len(e.info["offending_objects"]) == 1


def test_repeated_nextto_painting_get_gt_nsrts():
    """Tests for the ground truth NSRTs in RepeatedNextToPaintingEnv."""
    # The OracleApproach test doesn't cover the PlaceOnTable or PlaceInShelf
    # samplers, so we test those here.
    utils.reset_config({
        "env": "repeated_nextto_painting",
        "num_test_tasks": 1,
    })
    env = RepeatedNextToPaintingEnv()
    init = env.get_train_tasks()[0].init
    obj0 = [obj for obj in list(init) if obj.name == "obj0"][0]
    shelf = [obj for obj in list(init) if obj.name == "receptacle_shelf"][0]
    robby = [obj for obj in list(init) if obj.name == "robby"][0]
    rng = np.random.default_rng(123)
    # Test PlaceOnTable
    nsrts = get_gt_nsrts(env.predicates, env.options)
    ptables = [nsrt for nsrt in nsrts if nsrt.name.startswith("PlaceOnTable")]
    assert len(ptables) == 1
    ptable = ptables[0]
    opt = ptable.ground([obj0, robby]).sample_option(init, set(), rng)
    assert opt.objects == [robby]
    assert RepeatedNextToPaintingEnv.table_lb < opt.params[1] < \
        RepeatedNextToPaintingEnv.table_ub
    # Test PlaceInShelf
    pshelves = [nsrt for nsrt in nsrts if nsrt.name.startswith("PlaceInShelf")]
    assert len(pshelves) == 1
    pshelf = pshelves[0]
    opt = pshelf.ground([obj0, shelf, robby]).sample_option(init, set(), rng)
    assert opt.objects == [robby]
    assert RepeatedNextToPaintingEnv.shelf_lb < opt.params[1] < \
        RepeatedNextToPaintingEnv.shelf_ub


def test_playroom_get_gt_nsrts():
    """Tests for the ground truth NSRTs in PlayroomEnv."""
    utils.reset_config({
        "env": "playroom",
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })
    env = PlayroomEnv()
    # Test MoveDialToDoor for coverage.
    nsrts = get_gt_nsrts(env.predicates, env.options)
    movedialtodoor = [nsrt for nsrt in nsrts \
                      if nsrt.name == "MoveDialToDoor"][0]
    train_tasks = env.get_train_tasks()
    train_task = train_tasks[0]
    state = train_task.init
    objs = list(state)
    robot, dial, door5, door6, region6, region7 = objs[17], objs[3], objs[
        8], objs[9], objs[15], objs[16]
    assert robot.name == "robby"
    assert dial.name == "dial"
    assert door5.name == "door5"
    assert door6.name == "door6"
    assert region6.name == "region6"
    assert region7.name == "region7"
    movedialtodoor_nsrt = movedialtodoor.ground([robot, dial, door6, region7])
    rng = np.random.default_rng(123)
    movetodoor_option = movedialtodoor_nsrt.sample_option(
        state, train_task.goal, rng)
    movetodoor_action = movetodoor_option.policy(state)
    assert env.action_space.contains(movetodoor_action.arr)
    assert np.all(movetodoor_action.arr == np.array([110.1, 15, 1, -1, 1],
                                                    dtype=np.float32))
    # Test MoveDoorToTable for coverage.
    movedoortotable = [nsrt for nsrt in nsrts \
                      if nsrt.name == "MoveDoorToTable"][0]
    movedoortotable_nsrt = movedoortotable.ground([robot, door6, region7])
    movedoortotable_option = movedoortotable_nsrt.sample_option(
        state, train_task.goal, rng)
    movedoortotable_action = movedoortotable_option.policy(state)
    assert env.action_space.contains(movedoortotable_action.arr)
    # Test AdvanceThroughDoor (moving left) for coverage.
    state.set(robot, "pose_x", 110.3)
    advancethroughdoor = [nsrt for nsrt in nsrts \
                      if nsrt.name == "AdvanceThroughDoor"][0]
    advancethroughdoor_nsrt = advancethroughdoor.ground(
        [robot, door6, region7, region6])
    movetodoor_option2 = advancethroughdoor_nsrt.sample_option(
        state, train_task.goal, rng)
    movetodoor_action2 = movetodoor_option2.policy(state)
    assert env.action_space.contains(movetodoor_action2.arr)
    # Test MoveDoorToDoor (moving left) for coverage.
    movedoortodoor = [nsrt for nsrt in nsrts \
                      if nsrt.name == "MoveDoorToDoor"][0]
    movedoortodoor_nsrt = movedoortodoor.ground([robot, door6, door5, region6])
    movedoortodoor_option = movedoortodoor_nsrt.sample_option(
        state, train_task.goal, rng)
    movedoortodoor_action = movedoortodoor_option.policy(state)
    assert env.action_space.contains(movedoortodoor_action.arr)
