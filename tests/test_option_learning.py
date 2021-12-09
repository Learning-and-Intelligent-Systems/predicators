"""Tests for option learning.
"""

import pytest
import numpy as np
from predicators.src.envs import create_env, EnvironmentFailure, BaseEnv
from predicators.src.datasets.demo_replay import create_demo_replay_data
from predicators.src.nsrt_learning import segment_trajectory, \
    learn_strips_operators
from predicators.src.option_learning import create_option_learner
from predicators.src import utils
from predicators.src.approaches import create_approach, ApproachTimeout, \
    ApproachFailure, BaseApproach
from predicators.src.datasets import create_dataset
from predicators.src.sampler_learning import learn_samplers
from predicators.src.planning import sesame_plan
import copy
from predicators.src.structs import NSRT

def test_known_options_option_learner():
    """Tests for _KnownOptionsOptionLearner.
    """
    env = create_env("cover")
    # We need to call update_config twice because the first call sets
    # some variables whose values we can then change in the second call.
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123})
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "num_train_tasks": 3,
                         "do_option_learning": False})
    train_tasks = next(env.train_tasks_generator())
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for _, actions, _ in ground_atom_dataset:
        for act in actions:
            assert act.has_option()
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]
    strips_ops, partitions = learn_strips_operators(segments)
    assert len(strips_ops) == len(partitions) == 4
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops) == 4
    assert len(env.options) == 1
    PickPlace = next(iter(env.options))
    assert option_specs == [(PickPlace, []) for _ in range(4)]
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            assert segment.has_option()
            option = segment.get_option()
            # This call should be a no-op when options are known.
            option_learner.update_segment_from_option_spec(segment, spec)
            assert segment.has_option()
            assert segment.get_option() == option
    # Reset configuration.
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "do_option_learning": False})


def test_oracle_option_learner_cover():
    """Tests for _OracleOptionLearner for the cover environment.
    """
    env = create_env("cover")
    # We need to call update_config twice because the first call sets
    # some variables whose values we can then change in the second call.
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123})
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "num_train_tasks": 3,
                         "do_option_learning": True,
                         "option_learner": "oracle"})
    train_tasks = next(env.train_tasks_generator())
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for _, actions, _ in ground_atom_dataset:
        for act in actions:
            assert not act.has_option()
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]
    strips_ops, partitions = learn_strips_operators(segments)
    assert len(strips_ops) == len(partitions) == 3
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops) == 3
    assert len(env.options) == 1
    PickPlace = next(iter(env.options))
    assert option_specs == [(PickPlace, []), (PickPlace, []), (PickPlace, [])]
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            assert not segment.has_option()
            # This call should add an option to the segment.
            option_learner.update_segment_from_option_spec(segment, spec)
            assert segment.has_option()
            option = segment.get_option()
            # In cover env, param == action array.
            assert option.parent == PickPlace
            assert np.allclose(option.params, segment.actions[0].arr)
    # Reset configuration.
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "do_option_learning": False})


def test_oracle_option_learner_blocks():
    """Tests for _OracleOptionLearner for the blocks environment.
    """
    env = create_env("blocks")
    # We need to call update_config twice because the first call sets
    # some variables whose values we can then change in the second call.
    utils.update_config({"env": "blocks",
                         "approach": "nsrt_learning",
                         "seed": 123})
    utils.update_config({"env": "blocks",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "num_train_tasks": 3,
                         "do_option_learning": True,
                         "option_learner": "oracle"})
    train_tasks = next(env.train_tasks_generator())
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for _, actions, _ in ground_atom_dataset:
        for act in actions:
            assert not act.has_option()
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]
    strips_ops, partitions = learn_strips_operators(segments)
    assert len(strips_ops) == len(partitions) == 4
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops) == 4
    assert len(env.options) == 3
    Pick = [option for option in env.options
            if option.name == "Pick"][0]
    Stack = [option for option in env.options
             if option.name == "Stack"][0]
    PutOnTable = [option for option in env.options
                  if option.name == "PutOnTable"][0]
    param_opts = [spec[0] for spec in option_specs]
    assert param_opts.count(Pick) == 2
    assert param_opts.count(Stack) == 1
    assert param_opts.count(PutOnTable) == 1
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            assert not segment.has_option()
            # This call should add an option to the segment.
            option_learner.update_segment_from_option_spec(segment, spec)
            assert segment.has_option()
            option = segment.get_option()
            assert option.parent in (Pick, Stack, PutOnTable)
            assert [obj.type for obj in option.objects] == option.parent.types
    # Reset configuration.
    utils.update_config({"env": "blocks",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "do_option_learning": False})


def test_simple_option_learner_cover_multistep_options():
    """Tests for _SimpleOptionLearner for the cover_multistep_options
    environment.
    """
    # First, construct ground truth samplers for every task. This is done by
    # getting the state trajectory for every task using the oracle approach, and
    # assigning sequences of this state trajectory to the corresponding
    # operators. For each state sequence, construct the ground truth parameter
    # by calculating the difference in low-level object features the same way it
    # is done in option_learning.py. For simplicity, restrict to tasks with only
    # 1 atom goal condition.
    relevant_tasks_indices = [0, 1, 3, 4, 6, 7, 9]
    utils.update_config({"env": "cover_multistep_options"})
    env = create_env("cover_multistep_options")
    env.seed(123)
    approach = create_approach("oracle", env.simulate, env.predicates,
                               env.options, env.types, env.action_space)
    assert not approach.is_learning_based
    approach.seed(123)
    task_states_per_op = [] #  A list of lists of state sequences
    tasks = []
    tasks = env._get_tasks(10, env._train_rng)
    samplers_per_task = [None]*len(tasks)  # These will be filled in later.

    for i, task in enumerate(tasks):
        # policy = approach.solve(task, timeout=500)
        # traj, _, reached = utils.run_policy_on_task(policy, task, env.simulate, env.predicates, 100)
        # assert reached

        if i not in relevant_tasks_indices:
            continue
        timeout = 10
        approach._num_calls += 1
        seed = approach._seed + approach._num_calls
        plan, metrics = sesame_plan(task, approach._option_model,
                                    approach._get_current_nsrts(),
                                    approach._get_current_predicates(),
                                    timeout, seed)
        # Define a new option_plan_to_policy that
        # also indicates when the first option is done and the second option
        # is starting. We need this because we need to assign states from the
        # generated trajectory to each of the two options. (Two options because
        # we limited the tasks to those solvable with two operators.)
        def new_option_plan_to_policy(plan):
            queue = list(plan)  # Don't modify plan, just in case
            initialized = False  # Special case first step
            switched = False
            def _policy(state):
                nonlocal initialized, switched
                # On the very first state, check initiation condition, and
                # take the action no matter what.
                if not initialized:
                    if not queue:
                        raise OptionPlanExhausted()
                    assert queue[0].initiable(state), "Unsound option plan"
                    initialized = True
                elif queue[0].terminal(state):
                    queue.pop(0)
                    switched = True
                    if not queue:
                        raise OptionPlanExhausted()
                    assert queue[0].initiable(state), "Unsound option plan"
                return queue[0].policy(state), switched
            return _policy
        # option_policy = utils.option_plan_to_policy(plan)
        option_policy = new_option_plan_to_policy(plan)

        def _policy(s):
            try:
                return option_policy(s)
                # return option_policy(s)[0]
            except utils.OptionPlanExhausted:
                raise ApproachFailure("Option plan exhausted.")

        # Define a new run_policy_on_task that also keeps track of
        # which states correspond to which option/operator.
        def new_run_policy_on_task(policy, task, simulator, predicates, max_steps):
            state = task.init
            atoms = utils.abstract(state, predicates)
            states = [state]
            states_per_op = [[state], []]
            idx = 0
            actions: List[Action] = []
            if task.goal.issubset(atoms):  # goal is already satisfied
                goal_reached = True
            else:
                goal_reached = False
                for _ in range(max_steps):
                    act, switched = policy(state)
                    state = simulator(state, act)
                    atoms = utils.abstract(state, predicates)
                    actions.append(act)
                    states.append(state)
                    states_per_op[idx].append(state)
                    if switched:
                        idx = 1
                        states_per_op[idx].append(state)
                    if task.goal.issubset(atoms):
                        goal_reached = True
                        break
            return (states, actions), [], goal_reached, states_per_op
        traj, _, reached, states_per_op = new_run_policy_on_task(_policy, task, env.simulate, env.predicates, 100)
        assert reached

        # Create "ground truth" samplers.
        task_states_per_op.append(states_per_op)
        objects = list(states_per_op[0][0])
        block = [b for b in objects if b.type == env._block_type][0]
        robot = [r for r in objects if r.type == env._robot_type][0]
        objects_that_change = [block, robot]
        samplers = []
        def make_sampler(seq, objects_that_change):
            def sampler(state, rng, objs):
                param = []
                for o in objects_that_change:
                    object_param = seq[0][o] - seq[-1][o]
                    param.extend(object_param)
                return param
            return sampler
        for state_seq in states_per_op:
            sampler = make_sampler(state_seq, objects_that_change)
            samplers.append(sampler)
        samplers_per_task[i] = samplers

    # Now we learn NSRTs.
    env = create_env("cover_multistep_options")
    # We need to call update_config twice because the first call sets
    # some variables whose values we can then change in the second call.
    utils.update_config({"env": "cover_multistep_options",
                         "approach": "nsrt_learning",
                         "seed": 123})
    utils.update_config({"env": "cover_multistep_options",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "num_train_tasks": 10,
                         "do_option_learning": True,
                         "option_learner": "simple"})
    dataset = create_dataset(env, tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for _, actions, _ in ground_atom_dataset:
        for act in actions:
            assert not act.has_option()
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]
    strips_ops, partitions = learn_strips_operators(segments)
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops)
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            # Modifies segment in-place.
            option_learner.update_segment_from_option_spec(segment, spec)
    samplers = learn_samplers(strips_ops, partitions, option_specs, False)
    assert len(samplers) == len(strips_ops)
    nsrts = []
    for op, option_spec, sampler in zip(strips_ops, option_specs, samplers):
        param_option, option_vars = option_spec
        nsrt = op.make_nsrt(param_option, option_vars, sampler)
        nsrts.append(nsrt)

    assert(len(nsrts)==2)

    # Now, make a plan with these learned NSRTs.
    num_solved = 0
    for i, task in enumerate(tasks):
        if i not in relevant_tasks_indices:
            continue
        print("i: ", i)
        # Get sequence of ground NSRTs?
        block = [b for b in list(task.init) if b.type == env._block_type][0]
        robot = [r for r in list(task.init) if r.type == env._robot_type][0]
        target = [t for t in list(task.init) if t.type == env._target_type][0]
        pick = strips_ops[0].make_nsrt(
            nsrts[0].option,
            nsrts[0].option_vars,
            samplers_per_task[i][0]
        )
        place = strips_ops[1].make_nsrt(
            nsrts[1].option,
            nsrts[1].option_vars,
            samplers_per_task[i][1]
        )
        # We know that the objects are in this order based on the type signature
        # of the learned operators. 
        pick_grounded = pick.ground([block, robot])
        place_grounded = place.ground([block, robot, target])

        plan = [pick_grounded.sample_option(task.init, env._train_rng),
                place_grounded.sample_option(task.init, env._train_rng)]
        # print("Sampler-substituted Learned NSRT plan: ", plan)

        # Now try solving with this plan
        def policy_maker(plan):
            option_policy = utils.option_plan_to_policy(plan)
            def _policy(s):
                try:
                    return option_policy(s)
                except utils.OptionPlanExhausted:
                    raise ApproachFailure("Option plan exhausted.")
            return _policy
        policy = policy_maker(plan)
        try:
            _, video, solved = utils.run_policy_on_task(
                policy, task, env.simulate, env.predicates,
                100, False)
        except EnvironmentFailure as e:
            print(f"Task {i+1} / {len(tasks)}: Environment failed "
                  f"with error: {e}")
            continue
        if solved:
            print(f"Task {i+1} / {len(tasks)}: SOLVED")
            num_solved += 1
        else:
            print(f"Task {i+1} / {len(tasks)}: Policy failed")

    print(f"Tasks solved: {num_solved} / {len(tasks)}")
    # Reset configuration.
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "do_option_learning": False})

def test_create_option_learner():
    """Tests for create_option_learner().
    """
    utils.update_config({"env": "not a real env",
                         "approach": "nsrt_learning",
                         "seed": 123})
    utils.update_config({"env": "blocks",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "num_train_tasks": 3,
                         "do_option_learning": True,
                         "option_learner": "not a real option learner"})
    with pytest.raises(NotImplementedError):
        create_option_learner()
    # Reset configuration.
    utils.update_config({"env": "blocks",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "do_option_learning": False})
