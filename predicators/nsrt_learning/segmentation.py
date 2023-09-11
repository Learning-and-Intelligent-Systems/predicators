"""Methods for segmenting low-level trajectories into segments."""

from typing import Callable, List, Optional, Set

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Predicate, Segment, State


def segment_trajectory(
        ll_traj: LowLevelTrajectory,
        predicates: Set[Predicate],
        atom_seq: Optional[List[Set[GroundAtom]]] = None) -> List[Segment]:
    """Segment a ground atom trajectory."""
    # Start with the segmenters that don't need atom_seq. Still pass it in
    # because if it was provided, it can be used to avoid calling abstract.
    if CFG.segmenter == "option_changes":
        return _segment_with_option_changes(ll_traj, predicates, atom_seq)
    if CFG.segmenter == "every_step":
        return _segment_with_switch_function(ll_traj, predicates, atom_seq,
                                             lambda _: True)
    # All segmenters below need atom_seq. Create it if it wasn't passed in.
    if atom_seq is None:
        atom_seq = [utils.abstract(s, predicates) for s in ll_traj.states]
    if CFG.segmenter == "atom_changes":
        return _segment_with_atom_changes(ll_traj, predicates, atom_seq)
    if CFG.segmenter == "oracle":
        return _segment_with_oracle(ll_traj, predicates, atom_seq)
    if CFG.segmenter == "contacts":
        return _segment_with_contact_changes(ll_traj, predicates, atom_seq)
    raise NotImplementedError(f"Unrecognized segmenter: {CFG.segmenter}.")


def _segment_with_atom_changes(
        ll_traj: LowLevelTrajectory, predicates: Set[Predicate],
        atom_seq: List[Set[GroundAtom]]) -> List[Segment]:
    """Segment a trajectory whenever the abstract state changes."""

    def _switch_fn(t: int) -> bool:
        return atom_seq[t] != atom_seq[t + 1]

    return _segment_with_switch_function(ll_traj, predicates, atom_seq,
                                         _switch_fn)


def _segment_with_contact_changes(
        ll_traj: LowLevelTrajectory, predicates: Set[Predicate],
        atom_seq: List[Set[GroundAtom]]) -> List[Segment]:
    """Segment a trajectory based on contact changes.

    Since environments do not expose contacts, this is implemented in an
    environment-specific way. We assume that some predicates represent
    contacts and we look for changes in those contact predicates.
    """

    if CFG.env == "stick_button":
        keep_pred_names = {"Grasped", "Pressed"}
    elif CFG.env in ("cover", "cover_multistep_options", "pybullet_cover"):
        keep_pred_names = {"Covers", "HandEmpty", "Holding"}
    elif CFG.env in ("blocks", "pybullet_blocks"):
        keep_pred_names = {"Holding", "On", "OnTable"}
    elif CFG.env == "doors":
        keep_pred_names = {"TouchingDoor", "InRoom"}
    elif CFG.env == "touch_point":
        keep_pred_names = {"Touched"}
    elif CFG.env == "coffee":
        keep_pred_names = {"Holding", "HandEmpty", "MachineOn", "CupFilled"}
    elif CFG.env == "exit_garage":
        keep_pred_names = {"ObstacleCleared", "CarHasExited"}
    else:
        raise NotImplementedError("Contact-based segmentation not implemented "
                                  f"for environment {CFG.env}.")

    # If some predicates are excluded, we need to load predicates from the
    # environment in case contact-based ones are excluded. Note that this is
    # not really leaking information because the same effect could be achieved
    # by implementing environment-specific contact detection functions that use
    # the low-level states only; this is just a more concise way to do that.
    env = get_or_create_env(CFG.env)
    keep_preds = {p for p in env.predicates if p.name in keep_pred_names}
    assert len(keep_preds) == len(keep_pred_names)
    all_keep_atoms = []
    for state in ll_traj.states:
        all_keep_atoms.append(utils.abstract(state, keep_preds))

    def _switch_fn(t: int) -> bool:
        return all_keep_atoms[t] != all_keep_atoms[t + 1]

    return _segment_with_switch_function(ll_traj, predicates, atom_seq,
                                         _switch_fn)


def _segment_with_option_changes(
        ll_traj: LowLevelTrajectory, predicates: Set[Predicate],
        atom_seq: Optional[List[Set[GroundAtom]]]) -> List[Segment]:
    """Segment a trajectory whenever the (assumed known) option changes."""

    def _switch_fn(t: int) -> bool:
        # Segment by checking whether the option changes on the next step.
        option_t = ll_traj.actions[t].get_option()
        # As a special case, if this is the last timestep, then use the
        # option's terminal function to check if it completed, or see if the
        # termination was due to max_num_steps_option_rollout.
        if t == len(ll_traj.actions) - 1:
            # Calculate the number of steps since the option changed.
            backward_t = t
            while backward_t > 0:
                if ll_traj.actions[backward_t -
                                   1].get_option() is not option_t:
                    break
                backward_t -= 1
            option_duration = t - backward_t + 1
            if option_duration >= CFG.max_num_steps_option_rollout:
                return True
            return option_t.terminal(ll_traj.states[t + 1])
        return option_t is not ll_traj.actions[t + 1].get_option()

    return _segment_with_switch_function(ll_traj, predicates, atom_seq,
                                         _switch_fn)


def _segment_with_oracle(ll_traj: LowLevelTrajectory,
                         predicates: Set[Predicate],
                         atom_seq: List[Set[GroundAtom]]) -> List[Segment]:
    """Segment a trajectory using oracle NSRTs.

    If options are known, just uses _segment_with_option_changes().

    Otherwise, starting at the beginning of the trajectory, keeps track of
    which oracle ground NSRTs are applicable. When any of them have their
    effects achieved, that marks the switch point between segments.
    """
    if ll_traj.actions and ll_traj.actions[0].has_option():
        assert CFG.option_learner == "no_learning"
        return _segment_with_option_changes(ll_traj, predicates, atom_seq)
    env = get_or_create_env(CFG.env)
    env_options = get_gt_options(env.get_name())
    gt_nsrts = get_gt_nsrts(env.get_name(), env.predicates, env_options)
    objects = list(ll_traj.states[0])
    ground_nsrts = {
        ground_nsrt
        for nsrt in gt_nsrts
        for ground_nsrt in utils.all_ground_nsrts(nsrt, objects)
    }
    atoms = atom_seq[0]
    all_expected_next_atoms = [
        utils.apply_operator(n, atoms)
        for n in utils.get_applicable_operators(ground_nsrts, atoms)
    ]

    def _switch_fn(t: int) -> bool:
        nonlocal all_expected_next_atoms  # update at each switch point
        next_atoms = atom_seq[t + 1]
        # Check if any of the current NSRT effects hold.
        for expected_next_atoms in all_expected_next_atoms:
            # Check if we have reached the expected next atoms.
            if expected_next_atoms != next_atoms:
                continue
            # Time to segment. Update the expected next atoms.
            applicable_nsrts = utils.get_applicable_operators(
                ground_nsrts, next_atoms)
            all_expected_next_atoms = [
                utils.apply_operator(n, next_atoms) for n in applicable_nsrts
            ]
            return True
        # Not yet time to segment.
        return False

    return _segment_with_switch_function(ll_traj, predicates, atom_seq,
                                         _switch_fn)


def _segment_with_switch_function(
        ll_traj: LowLevelTrajectory, predicates: Set[Predicate],
        atom_seq: Optional[List[Set[GroundAtom]]],
        switch_fn: Callable[[int], bool]) -> List[Segment]:
    """Helper for other segmentation methods.

    The switch_fn takes in a timestep and returns True if the trajectory
    should be segmented at the end of that timestep.
    """
    segments = []
    assert len(ll_traj.states) > 0
    current_segment_states: List[State] = []
    current_segment_actions: List[Action] = []
    if atom_seq is not None:
        assert len(ll_traj.states) == len(atom_seq)
        current_segment_init_atoms = atom_seq[0]
    else:
        s0 = ll_traj.states[0]
        current_segment_init_atoms = utils.abstract(s0, predicates)
    for t in range(len(ll_traj.actions)):
        current_segment_states.append(ll_traj.states[t])
        current_segment_actions.append(ll_traj.actions[t])
        if switch_fn(t):
            # Include the final state as the end of this segment.
            current_segment_states.append(ll_traj.states[t + 1])
            current_segment_traj = LowLevelTrajectory(current_segment_states,
                                                      current_segment_actions)
            if atom_seq is not None:
                current_segment_final_atoms = atom_seq[t + 1]
            else:
                st1 = ll_traj.states[t + 1]
                current_segment_final_atoms = utils.abstract(st1, predicates)
            if ll_traj.actions[t].has_option():
                segment = Segment(current_segment_traj,
                                  current_segment_init_atoms,
                                  current_segment_final_atoms,
                                  ll_traj.actions[t].get_option())
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
