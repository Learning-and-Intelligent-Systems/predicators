"""Methods for segmenting low-level trajectories into segments."""

from typing import List, Callable
from predicators.src.structs import LowLevelTrajectory, Segment, State, \
    Action, GroundAtomTrajectory
from predicators.src.settings import CFG


def segment_trajectory(trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a ground atom trajectory."""
    if CFG.segmenter == "atom_changes":
        return _segment_with_atom_changes(trajectory)
    if CFG.segmenter == "option_changes":
        return _segment_with_option_changes(trajectory)
    raise NotImplementedError(f"Unrecognized segmenter: {CFG.segmenter}.")


def _segment_with_atom_changes(
        trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a trajectory whenever the abstract state changes."""

    _, all_atoms = trajectory

    def _switch_fn(t: int) -> bool:
        return all_atoms[t] != all_atoms[t + 1]

    return _segment_with_switch_function(trajectory, _switch_fn)


def _segment_with_option_changes(
        trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a trajectory whenever the (assumed known) option changes."""

    traj, _ = trajectory

    def _switch_fn(t: int) -> bool:
        # Segment by checking whether the option changes on the next step.
        option_t = traj.actions[t].get_option()
        # As a special case, if this is the last time step, then use the
        # option's terminal function to check if it completed.
        if t == len(traj.actions) - 1:
            return option_t.terminal(traj.states[t + 1])
        return not option_t is traj.actions[t + 1].get_option()

    return _segment_with_switch_function(trajectory, _switch_fn)


def _segment_with_switch_function(
        trajectory: GroundAtomTrajectory,
        switch_fn: Callable[[int], bool]) -> List[Segment]:
    """Helper for other segmentation methods.

    The switch_fn takes a timestep and returns True if the trajectory
    should be segmented at that timestep.
    """
    segments = []
    traj, all_atoms = trajectory
    assert len(traj.states) == len(all_atoms)
    assert len(traj.states) > 0
    current_segment_states: List[State] = []
    current_segment_actions: List[Action] = []
    current_segment_init_atoms = all_atoms[0]
    for t in range(len(traj.actions)):
        current_segment_states.append(traj.states[t])
        current_segment_actions.append(traj.actions[t])
        if switch_fn(t):
            # Include the final state as the end of this segment.
            current_segment_states.append(traj.states[t + 1])
            current_segment_traj = LowLevelTrajectory(current_segment_states,
                                                      current_segment_actions)
            current_segment_final_atoms = all_atoms[t + 1]
            if traj.actions[t].has_option():
                segment = Segment(current_segment_traj,
                                  current_segment_init_atoms,
                                  current_segment_final_atoms,
                                  traj.actions[t].get_option())
            else:
                # If we're in option learning mode, include the default option
                # here; replaced later during option learning.
                segment = Segment(current_segment_traj,
                                  current_segment_init_atoms,
                                  current_segment_final_atoms)
            segments.append(segment)
            current_segment_states = []
            current_segment_actions = []
            current_segment_init_atoms = current_segment_final_atoms
    # Don't include the last current segment because it didn't result in a
    # switch. (E.g., with option_changes, the option may not have terminated.)
    return segments
