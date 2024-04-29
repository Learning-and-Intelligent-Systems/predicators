"""Test cases for the oracle approach class."""
from typing import Any, Dict, List, Set
from unittest.mock import patch

import numpy as np
import pytest
from gym.spaces import Box

import predicators.envs.pddl_env
from predicators import utils
from predicators.approaches.base_approach import ApproachFailure, \
    ApproachTimeout
from predicators.approaches.oracle_approach import OracleApproach
from predicators.envs.blocks import BlocksEnv
from predicators.envs.cluttered_table import ClutteredTableEnv, \
    ClutteredTablePlaceEnv
from predicators.envs.coffee import CoffeeEnv
from predicators.envs.cover import BumpyCoverEnv, CoverEnv, \
    CoverEnvHierarchicalTypes, CoverEnvPlaceHard, CoverEnvRegrasp, \
    CoverEnvTypedOptions, CoverMultistepOptions, RegionalBumpyCoverEnv
from predicators.envs.doors import DoorsEnv
from predicators.envs.exit_garage import ExitGarageEnv
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.envs.painting import PaintingEnv
from predicators.envs.pddl_env import FixedTasksBlocksPDDLEnv, \
    ProceduralTasksBlocksPDDLEnv, ProceduralTasksDeliveryPDDLEnv, \
    ProceduralTasksEasyDeliveryPDDLEnv
from predicators.envs.playroom import PlayroomEnv, PlayroomSimpleEnv
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.repeated_nextto import RepeatedNextToAmbiguousEnv, \
    RepeatedNextToEnv, RepeatedNextToSimple, RepeatedNextToSingleOptionEnv
from predicators.envs.repeated_nextto_painting import RepeatedNextToPaintingEnv
from predicators.envs.sandwich import SandwichEnv
from predicators.envs.satellites import SatellitesEnv, SatellitesSimpleEnv
from predicators.envs.screws import ScrewsEnv
from predicators.envs.stick_button import StickButtonEnv, \
    StickButtonMovementEnv
from predicators.envs.tools import ToolsEnv
from predicators.envs.touch_point import TouchOpenEnv, TouchPointEnv, \
    TouchPointEnvParam
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.option_model import _OracleOptionModel
from predicators.settings import CFG
from predicators.structs import NSRT, Action, ParameterizedOption, Task, \
    Variable

_PDDL_ENV_MODULE_PATH = predicators.envs.pddl_env.__name__

ENV_NAME_AND_CLS = [
    ("cover", CoverEnv), ("cover_typed_options", CoverEnvTypedOptions),
    ("cover_place_hard", CoverEnvPlaceHard),
    ("cover_hierarchical_types", CoverEnvHierarchicalTypes),
    ("cover_regrasp", CoverEnvRegrasp), ("bumpy_cover", BumpyCoverEnv),
    ("cover_multistep_options", CoverMultistepOptions),
    ("regional_bumpy_cover", RegionalBumpyCoverEnv),
    ("cluttered_table", ClutteredTableEnv),
    ("cluttered_table_place", ClutteredTablePlaceEnv), ("blocks", BlocksEnv),
    ("exit_garage", ExitGarageEnv), ("narrow_passage", NarrowPassageEnv),
    ("painting", PaintingEnv), ("sandwich", SandwichEnv), ("tools", ToolsEnv),
    ("playroom", PlayroomEnv), ("repeated_nextto", RepeatedNextToEnv),
    ("repeated_nextto_single_option", RepeatedNextToSingleOptionEnv),
    ("repeated_nextto_ambiguous", RepeatedNextToAmbiguousEnv),
    ("repeated_nextto_simple", RepeatedNextToSimple),
    ("satellites", SatellitesEnv), ("satellites_simple", SatellitesSimpleEnv),
    ("screws", ScrewsEnv),
    ("repeated_nextto_painting", RepeatedNextToPaintingEnv),
    ("pddl_blocks_fixed_tasks", FixedTasksBlocksPDDLEnv),
    ("pddl_blocks_procedural_tasks", ProceduralTasksBlocksPDDLEnv),
    ("pddl_delivery_procedural_tasks", ProceduralTasksDeliveryPDDLEnv),
    ("pddl_easy_delivery_procedural_tasks",
     ProceduralTasksEasyDeliveryPDDLEnv), ("touch_point", TouchPointEnv),
    ("touch_point_param", TouchPointEnvParam), ("touch_open", TouchOpenEnv),
    ("stick_button", StickButtonEnv),
    ("stick_button_move", StickButtonMovementEnv), ("doors", DoorsEnv),
    ("coffee", CoffeeEnv), ("pybullet_blocks", PyBulletBlocksEnv)
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
EXTRA_ARGS_ORACLE_APPROACH["bumpy_cover"] = [
    {
        "bumpy_cover_right_targets": True,
        "sesame_max_samples_per_step": 100,
    },
    {
        "bumpy_cover_right_targets": False,
    },
]
EXTRA_ARGS_ORACLE_APPROACH["regional_bumpy_cover"] = [
    {
        "bumpy_cover_init_bumpy_prob": 1.0,
        "bumpy_cover_bumpy_region_start": 0.5
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
EXTRA_ARGS_ORACLE_APPROACH["exit_garage"] = [{
    "exit_garage_clear_refine_penalty":
    0,
    "exit_garage_min_num_obstacles":
    1,
    "exit_garage_max_num_obstacles":
    1,
    "exit_garage_rrt_num_control_samples":
    15,
    "exit_garage_rrt_sample_goal_eps":
    0.3,
}, {
    "exit_garage_clear_refine_penalty":
    0,
    "exit_garage_min_num_obstacles":
    3,
    "exit_garage_max_num_obstacles":
    3,
    "exit_garage_raise_environment_failure":
    True,
    "exit_garage_motion_planning_ignore_obstacles":
    True,
}]
EXTRA_ARGS_ORACLE_APPROACH["narrow_passage"] = [{
    "narrow_passage_open_door_refine_penalty":
    0,
    "narrow_passage_door_width_padding_lb":
    0.075,
    "narrow_passage_door_width_padding_ub":
    0.075,
    "narrow_passage_passage_width_padding_lb":
    0.075,
    "narrow_passage_passage_width_padding_ub":
    0.075,
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
EXTRA_ARGS_ORACLE_APPROACH["pybullet_blocks"] = [
    {
        "pybullet_robot": "panda",
        "option_model_name": "oracle",
        "option_model_terminate_on_repeat": False,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "blocks_num_blocks_train": [3],
        "blocks_num_blocks_test": [3],
    },
]
EXTRA_ARGS_ORACLE_APPROACH["blocks"] = [
    {
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "blocks_num_blocks_train": [3],
        "blocks_num_blocks_test": [3],
    },
    {
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "blocks_num_blocks_train": [1],
        "blocks_num_blocks_test": [1],
        "blocks_holding_goals": True,
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
        env = env_cls(use_gui=False)
        train_tasks = [t.task for t in env.get_train_tasks()]
        test_tasks = [t.task for t in env.get_test_tasks()]
        approach = OracleApproach(env.predicates,
                                  get_gt_options(env.get_name()), env.types,
                                  env.action_space, train_tasks)
        assert not approach.is_learning_based
        for task in train_tasks:
            policy = approach.solve(task, timeout=500)
            assert _policy_solves_task(policy, task, env.simulate)
        for task in test_tasks:
            policy = approach.solve(task, timeout=500)
            assert _policy_solves_task(policy, task, env.simulate)
    # Tests if OracleApproach can load _OracleOptionModel
    assert isinstance(approach.get_option_model(), _OracleOptionModel)


def test_planning_without_sim():
    """Tests the oracle approach in an environment with no simulator."""
    # Test planning in a PDDL environment, which should succeed without
    # simulation.
    utils.reset_config({
        "env": "pddl_blocks_procedural_tasks",
        "num_train_tasks": 0,
        "num_test_tasks": 2,
        "bilevel_plan_without_sim": True,
    })
    simulate_path_str = \
        f"{_PDDL_ENV_MODULE_PATH}.ProceduralTasksBlocksPDDLEnv.simulate"
    with patch(simulate_path_str) as mock_simulate:
        # Raise an error (and fail the test) if simulate is called.
        mock_simulate.side_effect = AssertionError("Simulate called.")
        env = ProceduralTasksBlocksPDDLEnv(use_gui=False)
        train_tasks = [t.task for t in env.get_train_tasks()]
        approach = OracleApproach(env.predicates,
                                  get_gt_options(env.get_name()), env.types,
                                  env.action_space, train_tasks)
    # Test the policy outside of patch() because _policy_solves_task uses the
    # simulator.
    task = env.get_test_tasks()[0].task
    policy = approach.solve(task, timeout=500)
    assert _policy_solves_task(policy, task, env.simulate)
    # Running the policy again should fail because the plan is empty.
    with pytest.raises(ApproachFailure) as e:
        _policy_solves_task(policy, task, env.simulate)
    assert "NSRT plan exhausted." in str(e)

    # Cover case where unknown task planner is used.
    utils.reset_config({
        "env": "cover",
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "bilevel_plan_without_sim": True,
        "sesame_task_planner": "not-a-real-planner"
    })
    with pytest.raises(ValueError):
        policy = approach.solve(task, timeout=500)

    # Test timeout.
    utils.reset_config({
        "env": "pddl_blocks_procedural_tasks",
        "num_train_tasks": 0,
        "num_test_tasks": 2,
        "bilevel_plan_without_sim": True,
    })
    with pytest.raises(ApproachTimeout) as e:
        approach.solve(task, timeout=0)

    # Test planning failure.
    objects = set(task.init)
    blocks = sorted(o for o in objects if o.type.name == "block")
    block0, block1 = blocks[:2]
    pred_name_to_pred = {p.name: p for p in env.predicates}
    on = pred_name_to_pred["on"]
    impossible_goal = {on([block0, block1]), on([block1, block0])}
    new_task = Task(task.init, impossible_goal)
    with pytest.raises(ApproachFailure) as e:
        approach.solve(new_task, timeout=500)

    # Cover case where the option is not initiable.
    utils.reset_config({
        "env": "cover",
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "bilevel_plan_without_sim": True,
    })
    env = CoverEnv(use_gui=False)
    train_tasks = [t.task for t in env.get_train_tasks()]

    # Force options to be non-initiable.
    options = get_gt_options(env.get_name())
    approach = OracleApproach(env.predicates, options, env.types,
                              env.action_space, train_tasks)

    assert len(options) == 1
    option = next(iter(options))
    new_option = ParameterizedOption(option.name, option.types,
                                     option.params_space, option.policy,
                                     lambda _1, _2, _3, _4: False,
                                     option.terminal)
    nsrts = approach._nsrts  # pylint: disable=protected-access
    new_nsrts = set()
    for nsrt in nsrts:
        new_nsrt = NSRT(
            nsrt.name,
            nsrt.parameters,
            nsrt.preconditions,
            nsrt.add_effects,
            nsrt.delete_effects,
            nsrt.ignore_effects,
            new_option,
            nsrt.option_vars,
            nsrt._sampler,  # pylint: disable=protected-access
        )
        new_nsrts.add(new_nsrt)
    approach._nsrts = new_nsrts  # pylint: disable=protected-access

    task = env.get_test_tasks()[0]

    policy = approach.solve(task, timeout=500)
    with pytest.raises(ApproachFailure) as e:
        policy(task.init)
    assert "Unsound option policy." in str(e)


def test_get_gt_nsrts():
    """Test get_gt_nsrts alone."""
    with pytest.raises(NotImplementedError):
        get_gt_nsrts("not a real environment", set(), set())


@pytest.mark.parametrize("env_name,env_cls", ENV_NAME_AND_CLS)
def test_nsrt_parameters(env_name, env_cls):
    """Checks assumptions on the oracle operators for all environments."""
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })
    env = env_cls(use_gui=False)
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
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
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    assert len(nsrts) == 2
    pick_nsrt, place_nsrt = sorted(nsrts, key=lambda o: o.name)
    assert pick_nsrt.name == "Pick"
    assert place_nsrt.name == "Place"
    train_task = env.get_train_tasks()[0].task
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
    assert get_gt_nsrts(env.get_name(), env.predicates, set()) == set()
    # Excluded predicate
    predicates = {p for p in env.predicates if p.name != "Holding"}
    nsrts = get_gt_nsrts(env.get_name(), predicates,
                         get_gt_options(env.get_name()))
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
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    assert len(nsrts) == 2
    if not place_version:
        dump_nsrt, grasp_nsrt = sorted(nsrts, key=lambda o: o.name)
        assert dump_nsrt.name == "Dump"
        assert grasp_nsrt.name == "Grasp"
    else:
        grasp_nsrt, place_nsrt = sorted(nsrts, key=lambda o: o.name)
        assert grasp_nsrt.name == "Grasp"
        assert place_nsrt.name == "Place"
    train_tasks = [t.task for t in env.get_train_tasks()]
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
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    ptables = [nsrt for nsrt in nsrts if nsrt.name.startswith("PlaceOnTable")]
    assert len(ptables) == 2
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


def test_playroom_simple_get_gt_nsrts():
    """Tests for the ground truth NSRTs in PlayroomSimpleEnv."""
    utils.reset_config({
        "env": "playroom_simple",
        "num_train_tasks": 1,
        "num_test_tasks": 1
    })
    env = PlayroomSimpleEnv()
    # Test MoveTableToDial for coverage.
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    movetabletodial = [nsrt for nsrt in nsrts \
                       if nsrt.name == "MoveTableToDial"][0]
    train_tasks = [t.task for t in env.get_train_tasks()]
    train_task = train_tasks[0]
    state = train_task.init
    objs = list(state)
    robot, dial = objs[-1], objs[-2]
    assert robot.name == "robby"
    assert dial.name == "dial"
    movetabletodial_nsrt = movetabletodial.ground([robot, dial])
    rng = np.random.default_rng(123)
    movetodial_option = movetabletodial_nsrt.sample_option(
        state, train_task.goal, rng)
    movetodial_action = movetodial_option.policy(state)
    assert env.action_space.contains(movetodial_action.arr)
    assert np.all(movetodial_action.arr == np.array([125, 15, 1, 0, 1],
                                                    dtype=np.float32))


def test_playroom_get_gt_nsrts():
    """Tests for the ground truth NSRTs in PlayroomEnv."""
    utils.reset_config({
        "env": "playroom",
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })
    env = PlayroomEnv()
    # Test MoveDialToDoor for coverage.
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    movedialtodoor = [nsrt for nsrt in nsrts \
                      if nsrt.name == "MoveDialToDoor"][0]
    train_tasks = [t.task for t in env.get_train_tasks()]
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


def test_external_oracle_approach():
    """Test that it's possible for an external user of predicators to define
    their own environment and NSRTs and use the oracle approach."""

    utils.reset_config({"num_train_tasks": 2, "num_test_tasks": 2})

    class _ExternalBlocksEnv(BlocksEnv):
        """To make sure that the test doesn't pass without using the new NSRTs,
        reverse the action space."""

        @classmethod
        def get_name(cls) -> str:
            return "external_blocks"

        @property
        def action_space(self) -> Box:
            original_space = super().action_space
            return Box(original_space.low[::-1],
                       original_space.high[::-1],
                       dtype=np.float32)

        def simulate(self, state, action):
            # Need to rewrite these lines here to avoid assertion in simulate
            # that uses action_space.
            x, y, z, fingers = action.arr[::-1]
            # Infer which transition function to follow
            if fingers < 0.5:
                return self._transition_pick(state, x, y, z)
            if z < self.table_height + self._block_size:
                return self._transition_putontable(state, x, y, z)
            return self._transition_stack(state, x, y, z)

    env = _ExternalBlocksEnv()
    assert env.get_name() == "external_blocks"

    # Create external options by modifying blocks options.
    options = set()
    old_option_to_new_option = {}

    def _reverse_policy(original_policy):

        def new_policy(state, memory, objects, params):
            action = original_policy(state, memory, objects, params)
            return Action(action.arr[::-1])

        return new_policy

    original_options = get_gt_options("blocks")
    for option in original_options:
        new_policy = _reverse_policy(option.policy)
        new_option = ParameterizedOption(f"external_{option.name}",
                                         option.types, option.params_space,
                                         new_policy, option.initiable,
                                         option.terminal)
        options.add(new_option)
        old_option_to_new_option[option] = new_option

    # Create the option model.
    option_model = _OracleOptionModel(options, env.simulate)

    # Create external NSRTs by just modifying blocks NSRTs.
    nsrts = set()
    for nsrt in get_gt_nsrts("blocks", env.predicates, original_options):
        nsrt_option = old_option_to_new_option[nsrt.option]
        sampler = nsrt._sampler  # pylint: disable=protected-access
        new_nsrt = NSRT(f"external_{nsrt.name}", nsrt.parameters,
                        nsrt.preconditions, nsrt.add_effects,
                        nsrt.delete_effects, nsrt.ignore_effects, nsrt_option,
                        nsrt.option_vars, sampler)
        nsrts.add(new_nsrt)

    # Create oracle approach.
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = OracleApproach(env.predicates,
                              options,
                              env.types,
                              env.action_space,
                              train_tasks,
                              nsrts=nsrts,
                              option_model=option_model)

    # Get a policy for the first task.
    task = train_tasks[0]
    policy = approach.solve(task, timeout=500)

    # Verify the policy.
    assert _policy_solves_task(policy, task, env.simulate)
