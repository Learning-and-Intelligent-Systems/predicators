"""Tests for option learning.
"""

import pytest
import numpy as np
from predicators.src.envs import create_env
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
                         "option_learner": "no_learning"})
    train_tasks = next(env.train_tasks_generator())
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for traj, _ in ground_atom_dataset:
        for act in traj.actions:
            assert act.has_option()
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]
    pnads = learn_strips_operators(segments)
    strips_ops = [pnad.op for pnad in pnads]
    datastores = [pnad.datastore for pnad in pnads]
    assert len(strips_ops) == len(datastores) == 4
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, datastores)
    assert len(option_specs) == len(strips_ops) == 4
    assert len(env.options) == 1
    PickPlace = next(iter(env.options))
    assert option_specs == [(PickPlace, []) for _ in range(4)]
    for datastore, spec in zip(datastores, option_specs):
        for (segment, _) in datastore:
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
                         "option_learner": "no_learning"})


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
                         "option_learner": "oracle"})
    train_tasks = next(env.train_tasks_generator())
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for traj, _ in ground_atom_dataset:
        for act in traj.actions:
            assert not act.has_option()
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]
    pnads = learn_strips_operators(segments)
    strips_ops = [pnad.op for pnad in pnads]
    datastores = [pnad.datastore for pnad in pnads]
    assert len(strips_ops) == len(datastores) == 3
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, datastores)
    assert len(option_specs) == len(strips_ops) == 3
    assert len(env.options) == 1
    PickPlace = next(iter(env.options))
    assert option_specs == [(PickPlace, []), (PickPlace, []), (PickPlace, [])]
    for datastore, spec in zip(datastores, option_specs):
        for (segment, _) in datastore:
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
                         "option_learner": "no_learning"})


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
                         "option_learner": "oracle"})
    train_tasks = next(env.train_tasks_generator())
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for traj, _ in ground_atom_dataset:
        for act in traj.actions:
            assert not act.has_option()
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]
    pnads = learn_strips_operators(segments)
    strips_ops = [pnad.op for pnad in pnads]
    datastores = [pnad.datastore for pnad in pnads]
    assert len(strips_ops) == len(datastores) == 4
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, datastores)
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
    for datastore, spec in zip(datastores, option_specs):
        for (segment, _) in datastore:
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
                         "option_learner": "no_learning"})


def test_simple_option_learner_cover_multistep_options():
    """Tests for _SimpleOptionLearner for the cover_multistep_options
    environment.
    """
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

    env = create_env("cover_multistep_options")
    approach = create_approach("nsrt_learning", env.simulate, env.predicates,
                               env.options, env.types, env.action_space)
    for train_tasks in env.train_tasks_generator():
        dataset = create_dataset(env, train_tasks)
        # approach.learn_from_offline_dataset(dataset, train_tasks)

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

    test_tasks = env.get_test_tasks()
    relevant_tasks_indices = [0, 1, 3, 4, 6, 7, 9]
    new_tasks = [t for (i, t) in enumerate(test_tasks) if i in relevant_tasks_indices]
    num_solved = 0
    approach.reset_metrics()
    for i, task in enumerate(new_tasks):
        print(task)
        print(end="", flush=True)
        # Each of these tasks involves one pick-place operation.
        # We have have learned two NSRTs.
        # We want to replace the sampler for each NSRT based on the actual ideal
        # sampling parameter.
        # To find this, we could solve the task via the oracle approach, and
        # then look at the state-action trajectory, and extract it from there.
        oracle_approach = create_approach("oracle", env.simulate,
        env.predicates, env.options, env.types, env.action_space)
        oracle_approach._num_calls += 1
        seed = oracle_approach._seed + oracle_approach._num_calls
        timeout = 10
        plan, metrics = sesame_plan(task, oracle_approach._option_model,
                                    oracle_approach._get_current_nsrts(),
                                    oracle_approach._get_current_predicates(),
                                    timeout, seed)
        for metric in ["num_skeletons_optimized",
                       "num_failures_discovered",
                       "plan_length"]:
            oracle_approach._metrics[f"total_{metric}"] += metrics[metric]

        print("PLAN: ", plan)
        # get actual policy first to debug
        policy = utils.option_plan_to_policy(plan)
        traj, _, _, = utils.run_policy_on_task(policy, task, env.simulate, env.predicates, False)
        print("TRAJECTORY: ", traj, len(traj))

        # Go from plan to policy.
        def option_plan_to_policy(plan):
            queue = list(plan)  # Don't modify plan, just in case
            assert len(queue) == 2
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
        policy = option_plan_to_policy(plan)
        state = task.init
        atoms = utils.abstract(state, env.predicates)
        states_pick = [state]
        states_place = []
        states = states_pick
        already_switched = False
        if task.goal.issubset(atoms):
            goal_reached = True
        else:
            goal_reached = False
            for _ in range(10):
                act, switched = policy(state)
                if not already_switched:
                    if switched:
                        states.append(state)
                        states = states_place
                        states.append(state)
                        already_switched = True
                else:
                    state = env.simulate(state, act)
                    atoms = utils.abstract(state, env.predicates)
                    states.append(state)
                    if task.goal.issubset(atoms):
                        goal_reached = True
                        break

        print("STATES PICK", states_pick, len(states_pick))
        print("STATES PLACE: ", states_place, len(states_place))
        # Let's replace the sampler in that with the sampler we want for this task.
        nsrts_temp = copy.deepcopy(nsrts)
        objects = list(states_pick[0])
        block = [b for b in objects if b.type == env._block_type][0]
        robot = [r for r in objects if r.type == env._robot_type][0]
        # Place sampler.
        objects_that_change = [block, robot]
        param_pick = []
        for j, o in enumerate(objects_that_change):
            object_param = states_pick[0][o] - states_pick[-1][o]
            param_pick.extend(object_param)
        def pick_sampler(state, rng, objs):
            return param_pick
        first = nsrts_temp[0]
        # first._sampler = pick_sampler
        first_nsrt = strips_ops[0].make_nsrt(first.option, first.option_vars, pick_sampler)
        # Place sampler.
        param_place = []
        for j, o in enumerate(objects_that_change):
            object_param = states_place[0][o] - states_place[-1][o]
            param_place.extend(object_param)
        def place_sampler(state, rng, objs):
            return param_place
        second = nsrts_temp[1]
        second_nsrt = strips_ops[1].make_nsrt(second.option, second.option_vars, place_sampler)
        # second._sampler = place_sampler

        # Now try to solve this task with the learned NSRTs that have their samplers
        # replaced.
        approach = create_approach("nsrt_learning", env.simulate, env.predicates,
                                   env.options, env.types, env.action_space)
        approach._nsrts = [first_nsrt, second_nsrt]
        try:
            policy = approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure) as e:
            print(f"Task {i+1} / {len(test_tasks)}: Approach failed to "
                  f"solve with error: {e}")
            continue
        try:
            _, video, solved = utils.run_policy_on_task(
                policy, task, env.simulate, env.predicates,
                CFG.max_num_steps_check_policy, CFG.make_videos, env.render)
        except EnvironmentFailure as e:
            print(f"Task {i+1} / {len(test_tasks)}: Environment failed "
                  f"with error: {e}")
            continue
        if solved:
            print(f"Task {i+1} / {len(test_tasks)}: SOLVED")
            num_solved += 1
        else:
            print(f"Task {i+1} / {len(test_tasks)}: Policy failed")
        # if CFG.make_videos:
        #     outfile = f"{utils.get_config_path_str()}__task{i}.mp4"
        #     utils.save_video(outfile, video)

    # assert len(option_specs) == len(strips_ops) == 3
    # assert len(env.options) == 1
    # PickPlace = next(iter(env.options))
    # assert option_specs == [(PickPlace, []), (PickPlace, []), (PickPlace, [])]
    # for partition, spec in zip(partitions, option_specs):
    #     for (segment, _) in partition:
    #         assert not segment.has_option()
    #         # This call should add an option to the segment.
    #         option_learner.update_segment_from_option_spec(segment, spec)
    #         assert segment.has_option()
    #         option = segment.get_option()
    #         # In cover env, param == action array.
    #         assert option.parent == PickPlace
    #         assert np.allclose(option.params, segment.actions[0].arr)
    # # Reset configuration.
    # utils.update_config({"env": "cover",
    #                      "approach": "nsrt_learning",
    #                      "seed": 123,
    #                      "do_option_learning": False})


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
                         "option_learner": "not a real option learner"})
    with pytest.raises(NotImplementedError):
        create_option_learner()
    # Reset configuration.
    utils.update_config({"env": "blocks",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "option_learner": "no_learning"})
