"""Utility functions that should only be used in tests."""
from typing import Callable, Optional, List, Tuple
from predicators.src.structs import State, Action, Task, Image, Video, \
    LowLevelTrajectory
from predicators.src.utils import simulate_policy_until
from predicators.src.settings import CFG


def simulate_policy_on_task(
    policy: Callable[[State], Action],
    task: Task,
    simulator: Callable[[State, Action], State],
    max_num_steps: int,
    render: Optional[Callable[[State, Task, Optional[Action]],
                              List[Image]]] = None,
) -> Tuple[LowLevelTrajectory, Video, bool]:
    """A light wrapper around run_policy_until that takes in a task and uses
    achieving the task's goal as the termination_function.

    Returns the trajectory and whether it achieves the task goal. Also
    optionally returns a video, if a render function is provided.
    """

    def _goal_check(state: State) -> bool:
        return all(goal_atom.holds(state) for goal_atom in task.goal)

    traj = simulate_policy_until(policy, simulator, task.init, _goal_check,
                                 max_num_steps)
    goal_reached = _goal_check(traj.states[-1])
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
    """A light wrapper around run_policy_on_task that returns whether the given
    policy solves the given task."""

    def _goal_check(state: State) -> bool:
        return all(goal_atom.holds(state) for goal_atom in task.goal)

    traj = simulate_policy_until(policy, simulator, task.init, _goal_check,
                                 CFG.max_num_steps_check_policy)
    return _goal_check(traj.states[-1])
