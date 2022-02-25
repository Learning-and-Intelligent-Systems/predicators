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


def create_demo_replay_data(env: BaseEnv,
                            train_tasks: List[Task],
                            nonoptimal_only: bool = False) -> Dataset:
    """Create offline datasets by collecting demos and replaying.

    If nonoptimal_only is True, perform planning after each replay to
    check if the replay is optimal. If it is, filter it out.
    """
    if nonoptimal_only:
        # Oracle is used to check if replays are optimal.
        oracle_approach = OracleApproach(
            env.predicates,
            env.options,
            env.types,
            env.action_space,
            train_tasks,
            task_planning_heuristic=CFG.offline_data_task_planning_heuristic,
            max_skeletons_optimized=CFG.offline_data_max_skeletons_optimized)
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
        if nonoptimal_only and _replay_is_optimal(replay_traj, traj, state_idx,
                                                  oracle_approach, env, goal):
            continue

        if CFG.option_learner != "no_learning":
            for act in replay_traj.actions:
                act.unset_option()
        replay_trajectories.append(replay_traj)

    assert len(replay_trajectories) == CFG.offline_data_num_replays

    return Dataset(demo_dataset.trajectories + replay_trajectories)


def _replay_is_optimal(replay_traj: LowLevelTrajectory,
                       demo_traj: LowLevelTrajectory, state_idx: int,
                       oracle_approach: BaseApproach, env: BaseEnv,
                       goal: Set[GroundAtom]) -> bool:
    """Plan from the end of the replay to the goal and check whether the result
    is as good as the demo.

    The state_idx is the index into demo_traj.states where the replay
    started.
    """
    assert demo_traj.states[state_idx].allclose(replay_traj.states[0])
    # Plan starting at the end of the replay trajectory to the demo goal.
    task = Task(replay_traj.states[-1], goal)
    try:
        policy = oracle_approach.solve(
            task, timeout=CFG.offline_data_planning_timeout)

        continued_traj, _, solved = utils.run_policy_on_task(
            policy, task, env.simulate, CFG.max_num_steps_check_policy)
        assert solved
    except (ApproachFailure, ApproachTimeout):
        # If planning fails, assume that the replay was not optimal.
        return False

    # Get the costs-to-go for both the demo actions and the replay actions.
    # This is not just len(actions) because we care about the length of the
    # option trajectory, not the length of the low-level action trajectory.
    demo_cost_to_go = utils.num_options_in_action_sequence(demo_traj.actions)
    replay_actions = demo_traj.actions[:state_idx] + continued_traj.actions
    # The +1 is for the replay itself, which consists of one option.
    replay_cost_to_go = 1 + utils.num_options_in_action_sequence(
        replay_actions)
    assert demo_cost_to_go <= replay_cost_to_go, "Demo was not optimal."
    return demo_cost_to_go == replay_cost_to_go
