"""Create offline datasets by collecting demonstrations and replaying."""

from typing import List, Set
import numpy as np
from predicators.src.approaches import BaseApproach, ApproachFailure, \
    ApproachTimeout, OracleApproach
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, _GroundNSRT, Task, \
    LowLevelTrajectory, GroundAtom
from predicators.src.datasets.demo_only import create_demo_data
from predicators.src.settings import CFG
from predicators.src import utils


def create_demo_replay_data(env: BaseEnv, train_tasks: List[Task]) -> Dataset:
    """Create offline datasets by collecting demos and replaying."""
    demo_dataset = create_demo_data(env, train_tasks)
    # We will sample from states uniformly at random.
    # The reason for doing it this way, rather than combining
    # all states into one list, is that we want to compute
    # all ground NSRTs once per trajectory only, rather
    # than once per state.
    weights = np.array(
        [len(traj.states) for traj in demo_dataset.trajectories])
    weights = weights / sum(weights)
    # Ground all NSRTs once per trajectory
    all_nsrts = get_gt_nsrts(env.predicates, env.options)
    ground_nsrts = []
    for traj in demo_dataset.trajectories:
        objects = sorted(traj.states[0])
        # Assumes objects should be the same within a traj
        assert all(set(objects) == set(s) for s in traj.states)
        ground_nsrts_traj: List[_GroundNSRT] = []
        for nsrt in all_nsrts:
            these_ground_nsrts = utils.all_ground_nsrts(nsrt, objects)
            ground_nsrts_traj.extend(these_ground_nsrts)
        ground_nsrts.append(ground_nsrts_traj)
    assert len(ground_nsrts) == len(demo_dataset.trajectories)
    # Perform replays
    rng = np.random.default_rng(CFG.seed)
    replay_trajectories: List[LowLevelTrajectory] = []
    while len(replay_trajectories) < CFG.offline_data_num_replays:
        # Sample a trajectory
        traj_idx = rng.choice(len(demo_dataset.trajectories), p=weights)
        traj = demo_dataset.trajectories[traj_idx]
        goal = train_tasks[traj.train_task_idx].goal
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
            option = sampled_nsrt.sample_option(state, goal, rng)
            if option.initiable(state):
                break
        # Execute the option
        try:
            replay_traj = utils.option_to_trajectory(
                state,
                env.simulate,
                option,
                max_num_steps=CFG.max_num_steps_option_rollout)
            # Add task goal into the trajectory.
            replay_traj = LowLevelTrajectory(
                replay_traj.states,
                replay_traj.actions,
                _is_demo=False,
                _train_task_idx=traj.train_task_idx)
        except utils.EnvironmentFailure:
            # We ignore replay data which leads to an environment failure.
            continue
        goal = train_tasks[traj.train_task_idx].goal
        if CFG.option_learner != "no_learning":
            for act in replay_traj.actions:
                act.unset_option()
        replay_trajectories.append(replay_traj)

    assert len(replay_trajectories) == CFG.offline_data_num_replays

    return Dataset(demo_dataset.trajectories + replay_trajectories)
