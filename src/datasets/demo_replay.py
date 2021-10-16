"""Create offline datasets by collecting demonstrations and replaying.
"""

from typing import List
import numpy as np
from predicators.src.approaches.oracle_approach import get_gt_ops
from predicators.src.envs import BaseEnv, EnvironmentFailure
from predicators.src.structs import Dataset, _GroundOperator
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
    # all ground operators once per trajectory only, rather
    # than once per state.
    weights = np.array([len(traj) for traj in demo_dataset])
    weights = weights / sum(weights)
    # Ground all operators once per trajectory
    operators = get_gt_ops(env.predicates, env.options)
    ground_operators = []
    for (ss, _) in demo_dataset:
        objects = sorted(ss[0])
        # Assumes objects should be the same within a traj
        assert all(set(objects) == set(s) for s in ss)
        ground_operators_traj: List[_GroundOperator] = []
        for op in operators:
            ground_ops = utils.all_ground_operators(op, objects)
            ground_operators_traj.extend(ground_ops)
        ground_operators.append(ground_operators_traj)
    assert len(ground_operators) == len(demo_dataset)
    # Perform replays
    rng = np.random.default_rng(CFG.seed)
    replay_dataset = []
    for _ in range(CFG.offline_data_num_replays):
        # Sample a trajectory
        traj_idx = rng.choice(len(demo_dataset), p=weights)
        traj_states = demo_dataset[traj_idx][0]
        # Sample a state
        state = traj_states[rng.choice(len(traj_states))]
        atoms = utils.abstract(state, env.predicates)
        # Sample an applicable operator
        applicable_ops = list(utils.get_applicable_operators(
            ground_operators[traj_idx], atoms))
        assert len(applicable_ops) > 0
        sampled_op = applicable_ops[rng.choice(len(applicable_ops))]
        # Sample a random option
        option = sampled_op.sample_option(state, rng)
        # Execute the option
        try:
            replay_traj = utils.option_to_trajectory(
                state, env.simulate, option,
                max_num_steps=CFG.max_num_steps_option_rollout)
        except EnvironmentFailure:
            # We ignore replay data which leads to an environment failure.
            continue
        if not CFG.include_options_in_offline_data:
            for act in replay_traj[1]:
                act.unset_option()
        replay_dataset.append(replay_traj)
    return demo_dataset + replay_dataset
