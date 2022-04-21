"""Methods for segmenting low-level trajectories into segments."""

from typing import Callable, List

from predicators.src import utils
from predicators.src.envs import get_or_create_env
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.settings import CFG
from predicators.src.structs import Action, GroundAtomTrajectory, \
    LowLevelTrajectory, Segment, State


def segment_trajectory(trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a ground atom trajectory."""
    if CFG.segmenter == "atom_changes":
        return _segment_with_atom_changes(trajectory)
    if CFG.segmenter == "option_changes":
        return _segment_with_option_changes(trajectory)
    if CFG.segmenter == "oracle":
        return _segment_with_oracle(trajectory)
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
        # As a special case, if this is the last timestep, then use the
        # option's terminal function to check if it completed.
        if t == len(traj.actions) - 1:
            return option_t.terminal(traj.states[t + 1])
        return option_t is not traj.actions[t + 1].get_option()

    return _segment_with_switch_function(trajectory, _switch_fn)


def _segment_with_oracle(trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a trajectory using oracle NSRTs.

    If options are known, just uses _segment_with_option_changes().

    Otherwise, starting at the beginning of the trajectory, keeps track of
    which oracle ground NSRTs are applicable. When any of them have their
    effects achieved, that marks the switch point between segments.
    """
    traj, all_atoms = trajectory
    if not traj.actions or traj.actions[0].has_option():
        return _segment_with_option_changes(trajectory)
    env = get_or_create_env(CFG.env)
    gt_nsrts = get_gt_nsrts(env.predicates, env.options)
    ground_nsrts = {
        ground_nsrt
        for nsrt in gt_nsrts
        for ground_nsrt in utils.all_ground_nsrts(nsrt, list(traj.states[0]))
    }
    atoms = all_atoms[0]
    current_nsrts = list(utils.get_applicable_operators(ground_nsrts, atoms))

    def _switch_fn(t: int) -> bool:
        nonlocal current_nsrts  # update at each switch point
        atoms = all_atoms[t + 1]
        # Check if any of the current NSRT effects hold.
        for ground_nsrt in current_nsrts:
            # Check effects.
            if not ground_nsrt.add_effects.issubset(atoms) or (
                    ground_nsrt.delete_effects & atoms):
                continue
            # Time to segment. Update the current NSRTs.
            current_nsrts = list(
                utils.get_applicable_operators(ground_nsrts, atoms))
            return True
        # Not yet time to segment.
        return False

    return _segment_with_switch_function(trajectory, _switch_fn)


def _segment_with_switch_function(
        trajectory: GroundAtomTrajectory,
        switch_fn: Callable[[int], bool]) -> List[Segment]:
    """Helper for other segmentation methods.

    The switch_fn takes in a timestep and returns True if the trajectory
    should be segmented at the end of that timestep.
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
    # Don't include the last segment because it didn't result in a switch.
    # E.g., with option_changes, the option may not have terminated.
    return segments
