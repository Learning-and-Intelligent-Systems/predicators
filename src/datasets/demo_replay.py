"""Create offline datasets by collecting demonstrations and replaying."""

from typing import List, Sequence
import numpy as np
from predicators.src.approaches import BaseApproach, ApproachFailure, \
    ApproachTimeout, create_approach
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.envs import BaseEnv, EnvironmentFailure
from predicators.src.structs import Dataset, _GroundNSRT, Task, \
    LowLevelTrajectory, Action
from predicators.src.datasets.demo_only import create_demo_data
from predicators.src.settings import CFG
from predicators.src import utils


def create_demo_replay_data(env: BaseEnv,
                            train_tasks: List[Task],
                            nonoptimal_only: bool = False) -> Dataset:
    """Create offline datasets by collecting demos and replaying.

    If nonoptimal_only is True, perform planning after each replay to
    check if the replay is optimal. If it is, filter it out.
    """
    if nonoptimal_only:
        # Oracle is used to check if replays are optimal.
        oracle_approach = create_approach("oracle", env.simulate,
                                          env.predicates, env.options,
                                          env.types, env.action_space)
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
    replay_dataset: Dataset = []
    while len(replay_dataset) < CFG.offline_data_num_replays:
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
            # Add task goal into the trajectory.
            replay_traj = LowLevelTrajectory(replay_traj.states,
                                             replay_traj.actions,
                                             _is_demo=False,
                                             _goal=traj.goal)
        except EnvironmentFailure:
            # We ignore replay data which leads to an environment failure.
            continue

        if nonoptimal_only and _replay_is_optimal(replay_traj, traj, state_idx,
                                                  oracle_approach, env):
            continue

        if CFG.option_learner != "no_learning":
            for act in replay_traj.actions:
                act.unset_option()
        replay_dataset.append(replay_traj)

    assert len(replay_dataset) == CFG.offline_data_num_replays

    return demo_dataset + replay_dataset


def _replay_is_optimal(replay_traj: LowLevelTrajectory,
                       demo_traj: LowLevelTrajectory, state_idx: int,
                       oracle_approach: BaseApproach, env: BaseEnv) -> bool:
    """Plan from the end of the replay to the goal and check whether the result
    is as good as the demo.

    The state_idx is the index into demo_traj.states where the replay
    started.
    """
    assert demo_traj.states[state_idx].allclose(replay_traj.states[0])
    # Plan starting at the end of the replay trajectory to the demo goal.
    task = Task(replay_traj.states[-1], demo_traj.goal)
    try:
        policy = oracle_approach.solve(
            task, timeout=CFG.offline_data_planning_timeout)

        continued_traj, _, solved = utils.run_policy_on_task(
            policy, task, env.simulate, CFG.max_num_steps_check_policy)
        assert solved
    except (ApproachFailure, ApproachTimeout):
        # If planning fails, assume that the replay was not optimal.
        return False

    # Get the cost-to-go from the actions. This is not just len(actions)
    # because we care about the length of the option trajectory, not the
    # length of the low-level action trajectory.
    demo_cost_to_go = _actions_to_cost_to_go(demo_traj.actions)
    replay_actions = demo_traj.actions[:state_idx] + continued_traj.actions
    # The +1 is for the replay itself, which consists of one option.
    replay_cost_to_go = 1 + _actions_to_cost_to_go(replay_actions)
    assert demo_cost_to_go <= replay_cost_to_go, "Demo was not optimal."
    return demo_cost_to_go == replay_cost_to_go


def _actions_to_cost_to_go(actions: Sequence[Action]) -> float:
    """Helper for _replay_is_optimal() that gets the option cost-to-go from a
    list of actions, assuming each option has cost 1."""
    ctg = 0.0
    last_option = None
    for action in actions:
        current_option = action.get_option()
        if not current_option is last_option:
            last_option = current_option
            ctg += 1.0
    return ctg
