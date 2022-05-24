"""Test cases for planning algorithms."""

from contextlib import nullcontext as does_not_raise

import pytest
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.approaches.oracle_approach import OracleApproach
from predicators.src.envs.cover import CoverEnv
from predicators.src.envs.painting import PaintingEnv
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.option_model import _OptionModelBase, \
    _OracleOptionModel, create_option_model
from predicators.src.planning import PlanningFailure, PlanningTimeout, \
    sesame_plan, task_plan, task_plan_grounding
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Action, ParameterizedOption, \
    Predicate, State, STRIPSOperator, Task, Type, _GroundNSRT, _Option


@pytest.mark.parametrize(
    "sesame_check_expected_atoms,sesame_grounder,expectation",
    [(True, "naive", does_not_raise()), (False, "naive", does_not_raise()),
     (True, "fd_translator", does_not_raise()),
     (True, "not a real grounder", pytest.raises(ValueError))])
def test_sesame_plan(sesame_check_expected_atoms, sesame_grounder,
                     expectation):
    """Tests for sesame_plan()."""
    utils.reset_config({
        "env": "cover",
        "sesame_check_expected_atoms": sesame_check_expected_atoms,
        "sesame_grounder": sesame_grounder,
        "num_test_tasks": 1,
    })
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    task = env.get_test_tasks()[0]
    option_model = create_option_model(CFG.option_model_name)
    with expectation as e:
        plan, metrics = sesame_plan(
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
    nsrts = get_gt_nsrts(env.predicates, env.options)
    task = env.get_train_tasks()[0]
    init_atoms = utils.abstract(task.init, env.predicates)
    objects = set(task.init)
    strips_ops = []
    option_specs = []
    for nsrt in nsrts:
        strips_ops.append(
            STRIPSOperator(nsrt.name, nsrt.parameters, nsrt.preconditions,
                           nsrt.add_effects, nsrt.delete_effects,
                           nsrt.side_predicates))
        option_specs.append((nsrt.option, nsrt.option_vars))
    ground_nsrts, reachable_atoms = task_plan_grounding(
        init_atoms, objects, strips_ops, option_specs)
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
    approach = OracleApproach(env.predicates, env.options, env.types,
                              env.action_space, train_tasks)
    task = train_tasks[0]
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
    approach = OracleApproach(env.predicates, env.options, env.types,
                              env.action_space, train_tasks)
    with pytest.raises(ApproachTimeout):
        approach.solve(impossible_task, timeout=1)  # backtracking occurs
    CFG.sesame_max_skeletons_optimized = old_max_skeletons
    approach = OracleApproach(env.predicates, env.options, env.types,
                              env.action_space, train_tasks)
    with pytest.raises(ApproachFailure):
        approach.solve(impossible_task, timeout=1)  # hits skeleton limit
    CFG.sesame_max_samples_per_step = old_max_samples_per_step
    nsrts = get_gt_nsrts(env.predicates, env.options)
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
        """A mock option model that always predicts a no-op."""

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
    task = train_tasks[0]
    option_model = create_option_model(CFG.option_model_name)
    approach = OracleApproach(env.predicates, env.options, env.types,
                              env.action_space, train_tasks)
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
    nsrts = get_gt_nsrts(env.predicates, env.options)
    old_option = next(iter(env.options))
    new_option = ParameterizedOption(old_option.name, old_option.types,
                                     old_option.params_space,
                                     old_option.policy, initiable,
                                     old_option.terminal)
    new_nsrts = set()
    for nsrt in nsrts:
        new_nsrts.add(
            NSRT(nsrt.name + "UNINITIABLE", nsrt.parameters,
                 nsrt.preconditions, nsrt.add_effects, nsrt.delete_effects,
                 nsrt.side_predicates, new_option, nsrt.option_vars,
                 nsrt._sampler))
    task = env.get_train_tasks()[0]
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
    side_predicates = set()
    neg_params_space = Box(-0.75, -0.25, (1, ))
    sleep_option = utils.SingletonParameterizedOption(
        "Sleep",
        lambda s, m, o, p: Action(p - int(o[0] == robby)),
        types=[robot_type],
        params_space=neg_params_space)
    sleep_op = STRIPSOperator("Sleep", parameters, preconditions, add_effects,
                              delete_effects, side_predicates)
    sleep_nsrt = sleep_op.make_nsrt(
        sleep_option, [robot_var],
        lambda s, g, rng, objs: neg_params_space.sample())
    cried = Predicate("Cried", [robot_type], lambda s, o: s[o[0]][1])
    parameters = [robot_var]
    preconditions = set()
    add_effects = {cried([robot_var])}
    delete_effects = set()
    side_predicates = set()
    pos_params_space = Box(0.25, 0.75, (1, ))
    cry_option = utils.SingletonParameterizedOption(
        "Cry",
        lambda s, m, o, p: Action(p + int(o[0] == robby)),
        types=[robot_type],
        params_space=pos_params_space)
    cry_op = STRIPSOperator("Cry", parameters, preconditions, add_effects,
                            delete_effects, side_predicates)
    cry_nsrt = cry_op.make_nsrt(
        cry_option, [robot_var],
        lambda s, g, rng, objs: pos_params_space.sample())

    goal = {asleep([robby]), asleep([robin]), cried([robby]), cried([robin])}
    task1 = Task(State({robby: [0, 0], robin: [0, 0]}), goal)
    task2 = Task(State({robin: [0, 0], robby: [0, 0]}), goal)

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

        @property
        def options(self):
            """Mock options."""
            return {sleep_option, cry_option}

    env = _MockEnv()
    option_model = _OracleOptionModel(env)
    # Check that sesame_plan is deterministic, over both NSRTs and objects.
    plan1 = [
        (act.name, act.objects) for act in sesame_plan(
            task1,
            option_model,
            [sleep_nsrt, cry_nsrt],
            set(),
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
            set(),
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
            set(),
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
            set(),
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
    option_specs = [(sleep_nsrt.option, sleep_nsrt.option_vars),
                    (cry_nsrt.option, cry_nsrt.option_vars)]
    nsrt_orders = [[sleep_op, cry_op], [cry_op, sleep_op]]
    object_orders = [[robby, robin], [robin, robby]]

    all_plans = []
    for nsrts in nsrt_orders:
        for objects in object_orders:
            ground_nsrts, reachable_atoms = task_plan_grounding(
                init_atoms, objects, nsrts, option_specs)
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
