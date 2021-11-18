"""Create offline datasets by collecting demonstrations and replaying.
"""

from typing import List
import numpy as np
from predicators.src.approaches.oracle_approach import get_gt_nsrts
from predicators.src.envs import BaseEnv, EnvironmentFailure
from predicators.src.structs import Dataset, _GroundNSRT
from predicators.src.datasets.demo_only import create_demo_data
from predicators.src.settings import CFG
from predicators.src import utils


def create_demo_replay_data(env: BaseEnv) -> Dataset:
    """Create offline datasets by collecting demos and replaying.
    """
    demo_dataset = create_demo_data(env)
    # We will sample from states uniformly at random.
    # The reason for doing it this way, rather than combining
    # all states into one list, is that we want to compute
    # all ground NSRTs once per trajectory only, rather
    # than once per state.
    weights = np.array([len(traj) for traj in demo_dataset])
    weights = weights / sum(weights)
    # Ground all NSRTs once per trajectory
    all_nsrts = get_gt_nsrts(env.predicates, env.options)
    ground_nsrts = []
    for (ss, _) in demo_dataset:
        objects = sorted(ss[0])
        # Assumes objects should be the same within a traj
        assert all(set(objects) == set(s) for s in ss)
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
        traj_states = demo_dataset[traj_idx][0]
        # Sample a state
        # We don't allow sampling the final state in the trajectory here,
        # because there's no guarantee that an initiable option exists
        # from that state
        state = traj_states[rng.choice(len(traj_states)-1)]
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
                state, env.simulate, option,
                max_num_steps=CFG.max_num_steps_option_rollout)
        except EnvironmentFailure:
            # We ignore replay data which leads to an environment failure.
            continue
        if CFG.do_option_learning:
            for act in replay_traj[1]:
                act.unset_option()
        replay_dataset.append(replay_traj)
    return demo_dataset + replay_dataset
