"""Utility functions that are used in testing."""

from typing import Callable, Optional, List, Tuple
from predicators.src.structs import State, Action, Task, Image, Video, \
    LowLevelTrajectory, _Option
from predicators.src.utils import run_policy_with_simulator
from predicators.src.settings import CFG


def option_to_trajectory(init_state: State,
                         simulator: Callable[[State, Action],
                                             State], option: _Option,
                         max_num_steps: int) -> LowLevelTrajectory:
    """A light wrapper around run_policy_with_simulator that takes in an option
    and uses achieving its terminal() condition as the termination_function."""
    assert option.initiable(init_state)
    return run_policy_with_simulator(option.policy, simulator, init_state,
                                     option.terminal, max_num_steps)


def run_policy_with_simulator_on_task(
    policy: Callable[[State], Action],
    task: Task,
    simulator: Callable[[State, Action], State],
    max_num_steps: int,
    render: Optional[Callable[[State, Task, Optional[Action]],
                              List[Image]]] = None,
) -> Tuple[LowLevelTrajectory, Video, bool]:
    """A light wrapper around run_policy_with_simulator that takes in a task
    and uses achieving the task's goal as the termination_function.

    Returns the trajectory and whether it achieves the task goal. Also
    optionally returns a video, if a render function is provided.
    """
    traj = run_policy_with_simulator(policy, simulator, task.init,
                                     task.goal_holds, max_num_steps)
    goal_reached = task.goal_holds(traj.states[-1])
    video: Video = []
    # Video rendering can be toggled on inline in tests, but by default it's
    # turned off for efficiency, hence the pragma.
    if render is not None:  # pragma: no cover
        for i, state in enumerate(traj.states):
            act = traj.actions[i] if i < len(traj.states) - 1 else None
            video.extend(render(state, task, act))
    return traj, video, goal_reached


def policy_solves_task(policy: Callable[[State], Action], task: Task,
                       simulator: Callable[[State, Action], State]) -> bool:
    """A light wrapper around run_policy_with_simulator_on_task that returns
    whether the given policy solves the given task."""
    _, _, goal_reached = run_policy_with_simulator_on_task(
        policy, task, simulator, CFG.max_num_steps_check_policy)
    return goal_reached
