"""Test cases for planning algorithms."""

import pytest
from gym.spaces import Box
from predicators.src.approaches import OracleApproach
from predicators.src.approaches.oracle_approach import get_gt_nsrts
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.envs import CoverEnv
from predicators.src.planning import sesame_plan, task_plan
from predicators.src import utils
from predicators.src.structs import Task, NSRT, ParameterizedOption, _Option, \
    _GroundNSRT, STRIPSOperator, Predicate, State, Type, Action
from predicators.src.settings import CFG
from predicators.src.option_model import create_option_model


def test_sesame_plan():
    """Tests for sesame_plan()."""
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    task = next(env.train_tasks_generator())[0]
    option_model = create_option_model(CFG.option_model_name, env.simulate)
    plan, _ = sesame_plan(task,
                          option_model,
                          nsrts,
                          env.predicates,
                          timeout=1,
                          seed=123)
    assert len(plan) == 2
    assert isinstance(plan[0], _Option)
    assert isinstance(plan[1], _Option)


def test_task_plan():
    """Tests for task_plan()."""
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    task = next(env.train_tasks_generator())[0]
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
    skeleton, _, _ = task_plan(init_atoms,
                               objects,
                               task.goal,
                               strips_ops,
                               option_specs,
                               timeout=1,
                               seed=123)
    assert len(skeleton) == 2
    assert isinstance(skeleton[0], _GroundNSRT)
    assert isinstance(skeleton[1], _GroundNSRT)


def test_sesame_plan_failures():
    """Tests for failures in the planner using the OracleApproach on
    CoverEnv."""
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    env.seed(123)
    option_model = create_option_model(CFG.option_model_name, env.simulate)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    approach.seed(123)
    task = next(env.train_tasks_generator())[0]
    trivial_task = Task(task.init, set())
    policy = approach.solve(trivial_task, timeout=500)
    with pytest.raises(ApproachFailure):
        policy(task.init)  # plan should get exhausted immediately
    assert utils.policy_solves_task(policy, trivial_task, env.simulate,
                                    env.predicates)
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
    old_max_samples_per_step = CFG.max_samples_per_step
    old_max_skeletons = CFG.max_skeletons_optimized
    CFG.max_samples_per_step = 1
    CFG.max_skeletons_optimized = float("inf")
    with pytest.raises(ApproachTimeout):
        approach.solve(impossible_task, timeout=1)  # backtracking occurs
    CFG.max_skeletons_optimized = old_max_skeletons
    with pytest.raises(ApproachFailure):
        approach.solve(impossible_task, timeout=1)  # hits skeleton limit
    CFG.max_samples_per_step = old_max_samples_per_step
    nsrts = get_gt_nsrts(env.predicates, env.options)
    nsrts = {nsrt for nsrt in nsrts if nsrt.name == "Place"}
    with pytest.raises(ApproachFailure):
        # Goal is not dr-reachable, should fail fast.
        sesame_plan(task,
                    option_model,
                    nsrts,
                    env.predicates,
                    timeout=500,
                    seed=123)
    with pytest.raises(ApproachFailure):
        # Goal is not dr-reachable, but we disable that check.
        # Should run out of skeletons.
        sesame_plan(task,
                    option_model,
                    nsrts,
                    env.predicates,
                    timeout=500,
                    seed=123,
                    check_dr_reachable=False)


def test_sesame_plan_uninitiable_option():
    """Tests planning in the presence of an option whose initiation set is
    nontrivial."""
    # pylint: disable=protected-access
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    env.seed(123)
    option_model = create_option_model(CFG.option_model_name, env.simulate)
    initiable = lambda s, m, o, p: False
    nsrts = get_gt_nsrts(env.predicates, env.options)
    old_option = next(iter(env.options))
    new_option = ParameterizedOption(old_option.name, old_option.types,
                                     old_option.params_space,
                                     old_option._policy, initiable,
                                     old_option._terminal)
    new_nsrts = set()
    for nsrt in nsrts:
        new_nsrts.add(
            NSRT(nsrt.name + "UNINITIABLE", nsrt.parameters,
                 nsrt.preconditions, nsrt.add_effects, nsrt.delete_effects,
                 nsrt.side_predicates, new_option, nsrt.option_vars,
                 nsrt._sampler))
    task = next(env.train_tasks_generator())[0]
    with pytest.raises(ApproachFailure) as e:
        # Planning should reach max_skeletons_optimized
        sesame_plan(task,
                    option_model,
                    new_nsrts,
                    env.predicates,
                    timeout=500,
                    seed=123)
    assert "Planning reached max_skeletons_optimized!" in str(e.value)


def test_planning_determinism():
    """Tests that planning is deterministic when there are multiple ways of
    achieving a goal."""
    utils.update_config({"env": "cover"})
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
    sleep_option = ParameterizedOption(
        "Sleep", [robot_type], neg_params_space,
        lambda s, m, o, p: Action(p - int(o[0] == robby)),
        lambda s, m, o, p: True, lambda s, m, o, p: True)
    sleep_op = STRIPSOperator("Sleep", parameters, preconditions, add_effects,
                              delete_effects, side_predicates)
    sleep_nsrt = sleep_op.make_nsrt(
        sleep_option, [robot_var],
        lambda s, rng, objs: neg_params_space.sample())
    cried = Predicate("Cried", [robot_type], lambda s, o: s[o[0]][1])
    parameters = [robot_var]
    preconditions = set()
    add_effects = {cried([robot_var])}
    delete_effects = set()
    side_predicates = set()
    pos_params_space = Box(0.25, 0.75, (1, ))
    cry_option = ParameterizedOption(
        "Cry", [robot_type], pos_params_space,
        lambda s, m, o, p: Action(p + int(o[0] == robby)),
        lambda s, m, o, p: True, lambda s, m, o, p: True)
    cry_op = STRIPSOperator("Cry", parameters, preconditions, add_effects,
                            delete_effects, side_predicates)
    cry_nsrt = cry_op.make_nsrt(cry_option, [robot_var],
                                lambda s, rng, objs: pos_params_space.sample())

    def _simulator(s, a):
        ns = s.copy()
        if a.arr.item() < -1:
            ns[robby][0] = 1
        elif a.arr.item() < 0:
            ns[robin][0] = 1
        elif a.arr.item() < 1:
            ns[robin][1] = 1
        else:
            ns[robby][1] = 1
        return ns

    goal = {asleep([robby]), asleep([robin]), cried([robby]), cried([robin])}
    task1 = Task(State({robby: [0, 0], robin: [0, 0]}), goal)
    task2 = Task(State({robin: [0, 0], robby: [0, 0]}), goal)
    option_model = create_option_model("default", _simulator)
    # Check that sesame_plan is deterministic, over both NSRTs and objects.
    plan1 = [(act.name, act.objects)
             for act in sesame_plan(task1,
                                    option_model, [sleep_nsrt, cry_nsrt],
                                    set(),
                                    timeout=10,
                                    seed=123)[0]]
    plan2 = [(act.name, act.objects)
             for act in sesame_plan(task1,
                                    option_model, [cry_nsrt, sleep_nsrt],
                                    set(),
                                    timeout=10,
                                    seed=123)[0]]
    plan3 = [(act.name, act.objects)
             for act in sesame_plan(task2,
                                    option_model, [sleep_nsrt, cry_nsrt],
                                    set(),
                                    timeout=10,
                                    seed=123)[0]]
    plan4 = [(act.name, act.objects)
             for act in sesame_plan(task2,
                                    option_model, [cry_nsrt, sleep_nsrt],
                                    set(),
                                    timeout=10,
                                    seed=123)[0]]
    assert plan1 == plan2 == plan3 == plan4
    # Check that task_plan is deterministic, over both NSRTs and objects.
    option_specs = [(sleep_nsrt.option, sleep_nsrt.option_vars),
                    (cry_nsrt.option, cry_nsrt.option_vars)]
    plan1 = [(act.name, act.objects)
             for act in task_plan(set(), [robby, robin],
                                  goal, [sleep_op, cry_op],
                                  option_specs,
                                  seed=123,
                                  timeout=10)[0]]
    plan2 = [(act.name, act.objects)
             for act in task_plan(set(), [robby, robin],
                                  goal, [cry_op, sleep_op],
                                  option_specs,
                                  seed=123,
                                  timeout=10)[0]]
    plan3 = [(act.name, act.objects)
             for act in task_plan(set(), [robin, robby],
                                  goal, [sleep_op, cry_op],
                                  option_specs,
                                  seed=123,
                                  timeout=10)[0]]
    plan4 = [(act.name, act.objects)
             for act in task_plan(set(), [robin, robby],
                                  goal, [cry_op, sleep_op],
                                  option_specs,
                                  seed=123,
                                  timeout=10)[0]]
    assert plan1 == plan2 == plan3 == plan4
