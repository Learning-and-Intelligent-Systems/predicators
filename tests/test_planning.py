"""Test cases for planning algorithms."""
import time
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.oracle_approach import OracleApproach
from predicators.envs.blocks import BlocksEnv
from predicators.envs.cluttered_table import ClutteredTableEnv
from predicators.envs.cover import CoverEnv
from predicators.envs.painting import PaintingEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.option_model import _OptionModelBase, _OracleOptionModel, \
    create_option_model
from predicators.planning import PlanningFailure, PlanningTimeout, \
    run_task_plan_once, sesame_plan, task_plan, task_plan_grounding
from predicators.settings import CFG
from predicators.structs import NSRT, Action, ParameterizedOption, Predicate, \
    State, STRIPSOperator, Task, Type, _GroundNSRT, _Option


@pytest.mark.parametrize(
    "sesame_check_expected_atoms,sesame_grounder,expectation,"
    "sesame_use_necessary_atoms",
    [(True, "naive", does_not_raise(), True),
     (True, "naive", does_not_raise(), False),
     (False, "naive", does_not_raise(), True),
     (True, "fd_translator", does_not_raise(), True),
     (True, "not a real grounder", pytest.raises(ValueError), True)])
def test_sesame_plan(sesame_check_expected_atoms, sesame_grounder, expectation,
                     sesame_use_necessary_atoms):
    """Tests for sesame_plan() with A*."""
    utils.reset_config({
        "env": "cover",
        "sesame_check_expected_atoms": sesame_check_expected_atoms,
        "sesame_use_necessary_atoms": sesame_use_necessary_atoms,
        "sesame_grounder": sesame_grounder,
        "num_test_tasks": 1,
        "sesame_task_planner": "astar",
    })
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    env_task = env.get_test_tasks()[0]
    task = env_task.task
    option_model = create_option_model(CFG.option_model_name)
    with expectation as e:
        plan, _, metrics = sesame_plan(
            task,
            option_model,
            nsrts,
            env.predicates,
            env.types,
            10,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
        )
    if e is None:
        assert len(plan) == 3
        assert all(isinstance(act, _Option) for act in plan)
        assert metrics["num_nodes_created"] >= metrics["num_nodes_expanded"]
    else:
        assert "Unrecognized sesame_grounder" in str(e)


def test_task_plan():
    """Tests for task_plan()."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    env_task = env.get_train_tasks()[0]
    task = env_task.task
    init_atoms = utils.abstract(task.init, env.predicates)
    objects = set(task.init)
    ground_nsrts, reachable_atoms = task_plan_grounding(
        init_atoms, objects, nsrts)
    heuristic = utils.create_task_planning_heuristic("hadd", init_atoms,
                                                     task.goal, ground_nsrts,
                                                     env.predicates, objects)
    task_plan_generator = task_plan(init_atoms,
                                    task.goal,
                                    ground_nsrts,
                                    reachable_atoms,
                                    heuristic,
                                    timeout=1,
                                    seed=123,
                                    max_skeletons_optimized=3)
    skeleton, _, metrics = next(task_plan_generator)
    initial_metrics = metrics
    assert len(skeleton) == 2
    assert isinstance(skeleton[0], _GroundNSRT)
    assert isinstance(skeleton[1], _GroundNSRT)
    total_num_skeletons = 1
    for _, _, metrics in task_plan_generator:
        total_num_skeletons += 1
        assert metrics["num_skeletons_optimized"] == total_num_skeletons
    assert total_num_skeletons == 3
    assert initial_metrics["num_skeletons_optimized"] == 1
    # Test timeout.
    with pytest.raises(PlanningTimeout):
        next(
            task_plan(init_atoms,
                      task.goal,
                      ground_nsrts,
                      reachable_atoms,
                      heuristic,
                      timeout=1e-6,
                      seed=123,
                      max_skeletons_optimized=3))


def test_sesame_plan_failures():
    """Tests for failures in the planner using the OracleApproach on CoverEnv
    and PaintingEnv."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    option_model = create_option_model(CFG.option_model_name)
    approach = OracleApproach(env.predicates, get_gt_options(env.get_name()),
                              env.types, env.action_space, train_tasks)
    env_task = train_tasks[0]
    task = env_task.task
    trivial_task = Task(task.init, set())
    policy = approach.solve(trivial_task, timeout=500)
    with pytest.raises(ApproachFailure):
        policy(task.init)  # plan should get exhausted immediately
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           trivial_task.init,
                                           trivial_task.goal_holds,
                                           max_num_steps=CFG.horizon)
    assert trivial_task.goal_holds(traj.states[-1])
    assert len(task.goal) == 1
    Covers = next(iter(task.goal)).predicate
    block0 = [obj for obj in task.init if obj.name == "block0"][0]
    target0 = [obj for obj in task.init if obj.name == "target0"][0]
    target1 = [obj for obj in task.init if obj.name == "target1"][0]
    impossible_task = Task(
        task.init, {Covers([block0, target0]),
                    Covers([block0, target1])})
    with pytest.raises(ApproachTimeout):
        approach.solve(impossible_task, timeout=0.1)  # times out
    with pytest.raises(ApproachTimeout):
        approach.solve(impossible_task, timeout=-100)  # times out
    utils.reset_config({"env": "cover", "sesame_grounder": "fd_translator"})
    with pytest.raises(ApproachTimeout):
        approach.solve(impossible_task, timeout=-100)  # times out
    utils.reset_config({"env": "cover", "sesame_grounder": "naive"})
    old_max_samples_per_step = CFG.sesame_max_samples_per_step
    old_max_skeletons = CFG.sesame_max_skeletons_optimized
    CFG.sesame_max_samples_per_step = 1
    CFG.sesame_max_skeletons_optimized = float("inf")
    approach = OracleApproach(env.predicates, get_gt_options(env.get_name()),
                              env.types, env.action_space, train_tasks)
    with pytest.raises(ApproachTimeout):
        approach.solve(impossible_task, timeout=1)  # backtracking occurs
    CFG.sesame_max_skeletons_optimized = old_max_skeletons
    approach = OracleApproach(env.predicates, get_gt_options(env.get_name()),
                              env.types, env.action_space, train_tasks)
    with pytest.raises(ApproachFailure):
        approach.solve(impossible_task, timeout=1)  # hits skeleton limit
    CFG.sesame_max_samples_per_step = old_max_samples_per_step
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    # Test that no plan is found when the horizon is too short.
    with pytest.raises(PlanningFailure):
        sesame_plan(
            task,
            option_model,
            nsrts,
            env.predicates,
            env.types,
            500,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=0)

    # Test that no plan is found when all of the options appear to terminate
    # immediately, under the option model.

    class _MockOptionModel(_OptionModelBase):
        """A mock option model that always predicts a noop."""

        def __init__(self, simulator):
            self._simulator = simulator

        def get_next_state_and_num_actions(self, state, option):
            return state, 0

    option_model = _MockOptionModel(env.simulate)
    with pytest.raises(PlanningFailure):
        sesame_plan(
            task,
            option_model,
            nsrts,
            env.predicates,
            env.types,
            500,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            max_skeletons_optimized=1,
            max_horizon=CFG.horizon)

    # Test for dr-reachability.
    nsrts = {nsrt for nsrt in nsrts if nsrt.name == "Place"}
    with pytest.raises(PlanningFailure):
        # Goal is not dr-reachable, should fail fast.
        sesame_plan(
            task,
            option_model,
            nsrts,
            env.predicates,
            env.types,
            500,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon)
    with pytest.raises(PlanningFailure):
        # Goal is not dr-reachable, but we disable that check.
        # Should run out of skeletons.
        sesame_plan(
            task,
            option_model,
            nsrts,
            env.predicates,
            env.types,
            500,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
            check_dr_reachable=False)
    utils.reset_config({"env": "painting", "painting_num_objs_train": [10]})
    env = PaintingEnv()
    train_tasks = env.get_train_tasks()
    env_task = train_tasks[0]
    task = env_task.task
    option_model = create_option_model(CFG.option_model_name)
    approach = OracleApproach(env.predicates, get_gt_options(env.get_name()),
                              env.types, env.action_space, train_tasks)
    with pytest.raises(ApproachTimeout):
        approach.solve(task, timeout=0.1)


def test_sesame_plan_uninitiable_option():
    """Tests planning in the presence of an option whose initiation set is
    nontrivial."""
    # pylint: disable=protected-access
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    option_model = create_option_model(CFG.option_model_name)
    initiable = lambda s, m, o, p: False
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    old_option = next(iter(get_gt_options(env.get_name())))
    new_option = ParameterizedOption(old_option.name, old_option.types,
                                     old_option.params_space,
                                     old_option.policy, initiable,
                                     old_option.terminal)
    new_nsrts = set()
    for nsrt in nsrts:
        new_nsrts.add(
            NSRT(nsrt.name + "UNINITIABLE", nsrt.parameters,
                 nsrt.preconditions, nsrt.add_effects, nsrt.delete_effects,
                 nsrt.ignore_effects, new_option, nsrt.option_vars,
                 nsrt._sampler))
    env_task = env.get_train_tasks()[0]
    task = env_task.task
    with pytest.raises(PlanningFailure) as e:
        # Planning should reach sesame_max_skeletons_optimized
        sesame_plan(
            task,
            option_model,
            new_nsrts,
            env.predicates,
            env.types,
            500,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon)
    assert "Planning reached max_skeletons_optimized!" in str(e.value)


def test_sesame_check_static_object_changes():
    """Tests for sesame_check_static_object_changes = True."""
    utils.reset_config({
        "env": "cover",
        "sesame_check_static_object_changes": True,
        "sesame_static_object_change_tol": 1e-3,
    })
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    env_task = env.get_test_tasks()[0]
    task = env_task.task
    option_model = create_option_model(CFG.option_model_name)
    # In Cover, the NSRTs do not contain the robot as an argument, so planning
    # is not possible if we are checking for static object changes.
    with pytest.raises(PlanningFailure) as e:
        # Planning should reach sesame_max_skeletons_optimized
        sesame_plan(
            task,
            option_model,
            nsrts,
            env.predicates,
            env.types,
            500,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon)
    assert "Planning reached max_skeletons_optimized!" in str(e.value)
    # In Blocks, the NSRTs are fully scoped, so planning should succeed even
    # when sesame_check_static_object_changes is True.
    utils.reset_config({
        "env": "blocks",
        "sesame_check_static_object_changes": True,
        "sesame_static_object_change_tol": 1e-3,
    })
    env = BlocksEnv()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    env_task = env.get_test_tasks()[0]
    task = env_task.task
    option_model = create_option_model(CFG.option_model_name)
    plan, _, _ = sesame_plan(
        task,
        option_model,
        nsrts,
        env.predicates,
        env.types,
        500,  # timeout
        123,  # seed
        CFG.sesame_task_planning_heuristic,
        CFG.sesame_max_skeletons_optimized,
        max_horizon=CFG.horizon)
    assert len(plan) > 0


def test_planning_determinism():
    """Tests that planning is deterministic when there are multiple ways of
    achieving a goal."""
    robot_type = Type("robot_type", ["asleep", "cried"])
    robot_var = robot_type("?robot")
    robby = robot_type("robby")
    robin = robot_type("robin")
    asleep = Predicate("Asleep", [robot_type], lambda s, o: s[o[0]][0])
    parameters = [robot_var]
    preconditions = set()
    add_effects = {asleep([robot_var])}
    delete_effects = set()
    ignore_effects = set()
    neg_params_space = Box(-0.75, -0.25, (1, ))
    sleep_option = utils.SingletonParameterizedOption(
        "Sleep",
        lambda s, m, o, p: Action(p - int(o[0] == robby)),
        types=[robot_type],
        params_space=neg_params_space)
    sleep_op = STRIPSOperator("Sleep", parameters, preconditions, add_effects,
                              delete_effects, ignore_effects)
    sleep_nsrt = sleep_op.make_nsrt(
        sleep_option, [robot_var],
        lambda s, g, rng, objs: neg_params_space.sample())
    cried = Predicate("Cried", [robot_type], lambda s, o: s[o[0]][1])
    parameters = [robot_var]
    preconditions = set()
    add_effects = {cried([robot_var])}
    delete_effects = set()
    ignore_effects = set()
    pos_params_space = Box(0.25, 0.75, (1, ))
    cry_option = utils.SingletonParameterizedOption(
        "Cry",
        lambda s, m, o, p: Action(p + int(o[0] == robby)),
        types=[robot_type],
        params_space=pos_params_space)
    cry_op = STRIPSOperator("Cry", parameters, preconditions, add_effects,
                            delete_effects, ignore_effects)
    cry_nsrt = cry_op.make_nsrt(
        cry_option, [robot_var],
        lambda s, g, rng, objs: pos_params_space.sample())

    goal = {asleep([robby]), asleep([robin]), cried([robby]), cried([robin])}
    task1 = Task(State({robby: [0, 0], robin: [0, 0]}), goal)
    task2 = Task(State({robin: [0, 0], robby: [0, 0]}), goal)
    options = {sleep_option, cry_option}

    class _MockEnv:

        @staticmethod
        def simulate(state, action):
            """A mock simulate method."""
            next_state = state.copy()
            if action.arr.item() < -1:
                next_state[robby][0] = 1
            elif action.arr.item() < 0:
                next_state[robin][0] = 1
            elif action.arr.item() < 1:
                next_state[robin][1] = 1
            else:
                next_state[robby][1] = 1
            return next_state

    env = _MockEnv()
    option_model = _OracleOptionModel(options, env.simulate)
    # Check that sesame_plan is deterministic, over both NSRTs and objects.
    plan1 = [
        (act.name, act.objects) for act in sesame_plan(
            task1,
            option_model,
            [sleep_nsrt, cry_nsrt],
            {asleep, cried},
            {robot_type},
            10,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
        )[0]
    ]
    plan2 = [
        (act.name, act.objects) for act in sesame_plan(
            task1,
            option_model,
            [cry_nsrt, sleep_nsrt],
            {asleep, cried},
            {robot_type},
            10,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
        )[0]
    ]
    plan3 = [
        (act.name, act.objects) for act in sesame_plan(
            task2,
            option_model,
            [sleep_nsrt, cry_nsrt],
            {asleep, cried},
            {robot_type},
            10,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
        )[0]
    ]
    plan4 = [
        (act.name, act.objects) for act in sesame_plan(
            task2,
            option_model,
            [cry_nsrt, sleep_nsrt],
            {asleep, cried},
            {robot_type},
            10,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
        )[0]
    ]
    assert plan1 == plan2 == plan3 == plan4
    # Check that task_plan is deterministic, over both NSRTs and objects.
    predicates = {asleep, cried}
    init_atoms = set()
    nsrt_orders = [[sleep_op, cry_op], [cry_op, sleep_op]]
    object_orders = [[robby, robin], [robin, robby]]

    all_plans = []
    for nsrts in nsrt_orders:
        for objects in object_orders:
            ground_nsrts, reachable_atoms = task_plan_grounding(
                init_atoms, objects, sorted(nsrts))
            heuristic = utils.create_task_planning_heuristic(
                "hadd", init_atoms, goal, ground_nsrts, predicates, objects)
            skeleton, _, _ = next(
                task_plan(init_atoms,
                          goal,
                          ground_nsrts,
                          reachable_atoms,
                          heuristic,
                          timeout=10,
                          seed=123,
                          max_skeletons_optimized=1))
            all_plans.append([(act.name, act.objects) for act in skeleton])

    assert len(all_plans) == 4
    for plan in all_plans[1:]:
        assert plan == all_plans[0]


def test_policy_guided_sesame():
    """Tests for sesame_plan() with an abstract policy used for guidance."""
    utils.reset_config({
        "env": "cover",
        "num_test_tasks": 1,
        "cover_initial_holding_prob": 0,
    })
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    env_task = env.get_test_tasks()[0]
    task = env_task.task
    option_model = create_option_model(CFG.option_model_name)
    # With a trivial policy, we would expect the number of nodes to be the
    # same as it would be if we planned with no policy.
    unguided_plan, _, unguided_metrics = sesame_plan(
        task,
        option_model,
        nsrts,
        env.predicates,
        env.types,
        1,  # timeout
        123,  # seed
        CFG.sesame_task_planning_heuristic,
        CFG.sesame_max_skeletons_optimized,
        max_horizon=CFG.horizon,
    )
    trivial_policy = lambda a, o, g: None
    guided_plan, _, guided_metrics = sesame_plan(
        task,
        option_model,
        nsrts,
        env.predicates,
        env.types,
        1,  # timeout
        123,  # seed
        CFG.sesame_task_planning_heuristic,
        CFG.sesame_max_skeletons_optimized,
        max_horizon=CFG.horizon,
        abstract_policy=trivial_policy,
        max_policy_guided_rollout=50)
    assert unguided_metrics["num_nodes_created"] == \
        guided_metrics["num_nodes_created"]
    assert unguided_metrics["num_nodes_expanded"] == \
        guided_metrics["num_nodes_expanded"]
    # Check that the plans are equal.
    assert len(unguided_plan) == len(guided_plan)
    for option1, option2 in zip(unguided_plan, guided_plan):
        assert option1.name == option2.name
        assert option1.objects == option2.objects
        assert np.allclose(option1.params, option2.params)
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    pick_nsrt = nsrt_name_to_nsrt["Pick"]
    place_nsrt = nsrt_name_to_nsrt["Place"]

    # When using a perfect policy, we should only expand the first node.
    def _abstract_policy(atoms, objs, goal):
        del objs  # unused
        assert all(a.predicate.name == "Covers" for a in goal)
        block_to_target = {a.objects[0]: a.objects[1] for a in goal}
        held_blocks = [
            a.objects[0] for a in atoms if a.predicate.name == "Holding"
        ]
        assert len(held_blocks) <= 1
        # If already holding an object, put it on the target.
        if held_blocks:
            block = held_blocks[0]
            target = block_to_target[block]
            return place_nsrt.ground([block, target])
        # Otherwise, pick up a block that's not yet at its goal.
        unrealized_goals = goal - atoms
        unrealized_blocks = sorted(a.objects[0] for a in unrealized_goals)
        block = unrealized_blocks[0]
        return pick_nsrt.ground([block])

    _, _, metrics = sesame_plan(
        task,
        option_model,
        nsrts,
        env.predicates,
        env.types,
        1,  # timeout
        123,  # seed
        CFG.sesame_task_planning_heuristic,
        CFG.sesame_max_skeletons_optimized,
        max_horizon=CFG.horizon,
        abstract_policy=_abstract_policy,
        max_policy_guided_rollout=50)

    assert metrics["num_nodes_expanded"] == 1.0

    # Test that a policy that outputs invalid NSRTs is not used.
    objects = set(task.init)
    ground_nsrts = [
        ground_nsrt for nsrt in nsrts
        for ground_nsrt in utils.all_ground_nsrts(nsrt, objects)
    ]

    def _invalid_policy(atoms, objs, goal):
        del objs, goal  # unused
        for ground_nsrt in ground_nsrts:
            if not ground_nsrt.preconditions.issubset(atoms):
                return ground_nsrt
        raise Exception("Should not happen.")  # pragma: no cover

    _, _, invalid_policy_metrics = sesame_plan(
        task,
        option_model,
        nsrts,
        env.predicates,
        env.types,
        1,  # timeout
        123,  # seed
        CFG.sesame_task_planning_heuristic,
        CFG.sesame_max_skeletons_optimized,
        max_horizon=CFG.horizon,
        abstract_policy=_invalid_policy,
        max_policy_guided_rollout=50)
    assert unguided_metrics["num_nodes_created"] == \
        invalid_policy_metrics["num_nodes_created"]
    assert unguided_metrics["num_nodes_expanded"] == \
        invalid_policy_metrics["num_nodes_expanded"]

    # Test timeout.
    def _slow_policy(atoms, objs, goal):
        time.sleep(0.51)
        return _abstract_policy(atoms, objs, goal)

    with pytest.raises(PlanningTimeout):
        sesame_plan(
            task,
            option_model,
            nsrts,
            env.predicates,
            env.types,
            0.5,  # timeout
            123,  # seed
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
            abstract_policy=_slow_policy,
            max_policy_guided_rollout=50)


def test_sesame_plan_fast_downward():
    """Tests for sesame_plan() with Fast Downward.

    We don't actually want to test Fast Downward, because we don't want
    to force people (and Github) to download and build it, so this test
    is written in a way that will pass whether you have Fast Downward
    installed or not.
    """
    for sesame_task_planner in ("fdopt", "fdsat", "not a real task planner"):
        utils.reset_config({
            "env": "cluttered_table",
            "num_test_tasks": 50,
            "sesame_task_planner": sesame_task_planner,
        })
        env = ClutteredTableEnv()
        nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                             get_gt_options(env.get_name()))
        env_task = env.get_test_tasks()[0]
        task = env_task.task
        option_model = create_option_model(CFG.option_model_name)
        try:
            plan, _, metrics = sesame_plan(
                task,
                option_model,
                nsrts,
                env.predicates,
                env.types,
                1,  # timeout
                123,  # seed
                CFG.sesame_task_planning_heuristic,
                CFG.sesame_max_skeletons_optimized,
                max_horizon=CFG.horizon,
            )
            # We only get to these lines if FD is installed.
            assert all(isinstance(act, _Option)
                       for act in plan)  # pragma: no cover
            assert metrics["num_nodes_created"] >= \
                metrics["num_nodes_expanded"]  # pragma: no cover
        except AssertionError as e:  # pragma: no cover
            # If the FD_EXEC_PATH environment variable is not set, we should
            # crash in the planner.
            assert "Please follow the instructions" in str(e)
        except ValueError as e:
            assert "Unrecognized sesame_task_planner" in str(e)


def test_task_planning_only():
    """Tests for the run_task_plan_once function."""
    utils.reset_config({
        "env": "cluttered_table",
        "num_test_tasks": 50,
    })
    env = ClutteredTableEnv()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    env_task = env.get_test_tasks()[0]
    task = env_task.task
    preds = env.predicates
    types = env.types
    with pytest.raises(PlanningFailure) as e:
        run_task_plan_once(task,
                           nsrts,
                           preds,
                           types,
                           100000.0,
                           0,
                           task_planning_heuristic="lmcut",
                           max_horizon=0.0)
    assert "exceeds horizon" in str(e)
