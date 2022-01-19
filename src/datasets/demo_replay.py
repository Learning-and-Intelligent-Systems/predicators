"""Create offline datasets by collecting demonstrations and replaying."""

from typing import List
import numpy as np
from predicators.src.approaches import create_approach, ApproachTimeout, ApproachFailure
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.envs import BaseEnv, EnvironmentFailure
from predicators.src.structs import Dataset, _GroundNSRT, Task, \
    LowLevelTrajectory
from predicators.src.datasets.demo_only import create_demo_data
from predicators.src.settings import CFG
from predicators.src import utils


def create_demo_replay_data(env: BaseEnv, train_tasks: List[Task]) -> Dataset:
    """Create offline datasets by collecting demos and replaying."""
    oracle_approach = create_approach("oracle", env.simulate, env.predicates,
                                      env.options, env.types, env.action_space)
    num_equal, total = 0, 0

    demo_dataset = create_demo_data(env, train_tasks)
    # We will sample from states uniformly at random.
    # The reason for doing it this way, rather than combining
    # all states into one list, is that we want to compute
    # all ground NSRTs once per trajectory only, rather
    # than once per state.
    weights = np.array([len(traj.states) for traj in demo_dataset])
    weights = weights / sum(weights)
    # Ground all NSRTs once per trajectory
    all_nsrts = get_gt_nsrts(env.predicates, env.options)
    ground_nsrts = []
    for traj in demo_dataset:
        objects = sorted(traj.states[0])
        # Assumes objects should be the same within a traj
        assert all(set(objects) == set(s) for s in traj.states)
        ground_nsrts_traj: List[_GroundNSRT] = []
        for nsrt in all_nsrts:
            these_ground_nsrts = utils.all_ground_nsrts(nsrt, objects)
            ground_nsrts_traj.extend(these_ground_nsrts)
        ground_nsrts.append(ground_nsrts_traj)
    assert len(ground_nsrts) == len(demo_dataset)
    # Perform replays
    rng = np.random.default_rng(CFG.seed)
    replay_dataset = []
    for _ in range(CFG.offline_data_num_replays):
        # Sample a trajectory
        traj_idx = rng.choice(len(demo_dataset), p=weights)
        traj = demo_dataset[traj_idx]
        # Sample a state
        # We don't allow sampling the final state in the trajectory here,
        # because there's no guarantee that an initiable option exists
        # from that state
        assert len(traj.states) > 1
        state_idx = rng.choice(len(traj.states) - 1)
        state = traj.states[state_idx]
        # Sample a random option that is initiable
        nsrts = ground_nsrts[traj_idx]
        assert len(nsrts) > 0
        while True:
            sampled_nsrt = nsrts[rng.choice(len(nsrts))]
            option = sampled_nsrt.sample_option(state, rng)
            if option.initiable(state):
                break
        # Execute the option
        try:
            replay_traj = utils.option_to_trajectory(
                state,
                env.simulate,
                option,
                max_num_steps=CFG.max_num_steps_option_rollout)
        except EnvironmentFailure:
            # We ignore replay data which leads to an environment failure.
            continue
        if CFG.option_learner != "no_learning":
            for act in replay_traj.actions:
                act.unset_option()

        task = Task(replay_traj.states[-1], demo_dataset[traj_idx].goal)

        try:

            if utils.abstract(task.init, env.predicates).issuperset(demo_dataset[traj_idx].goal):
                num_equal += 1
                total += 1

                continue

            else:
                policy = oracle_approach.solve(
                    task,
                    timeout=CFG.offline_data_planning_timeout)

                continued_traj, _, solved = utils.run_policy_on_task(
                    policy, task, env.simulate,
                    CFG.max_num_steps_check_policy)
                assert solved
                replay_actions = list(demo_dataset[traj_idx].actions[:state_idx]) + \
                    list(continued_traj.actions)

                demo_cost_to_go = _actions_to_cost_to_go(demo_dataset[traj_idx].actions)
                replay_cost_to_go = 1 + _actions_to_cost_to_go(replay_actions)
                assert demo_cost_to_go <= replay_cost_to_go
                num_equal += (demo_cost_to_go == replay_cost_to_go)
                total += 1

                if demo_cost_to_go == replay_cost_to_go:
                    continue

        except (ApproachFailure, ApproachTimeout):
            print("WARNING: could not finish plan.")
            total += 1

        replay_traj = LowLevelTrajectory(replay_traj.states, replay_traj.actions,
                                         False, traj.goal)

        replay_dataset.append(replay_traj)


    return demo_dataset + replay_dataset


def _actions_to_cost_to_go(actions):
    ctg = 0
    last_option = None
    for action in actions:
        if action.get_option() != last_option:  # TODO: should change this to `is` later, Option objects don't have an equals method defined
            last_option = action.get_option()
            ctg += 1
    return ctg
