"""Methods for segmenting low-level trajectories into segments."""

from typing import Callable, List

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.settings import CFG
from predicators.structs import Action, GroundAtomTrajectory, \
    LowLevelTrajectory, Segment, State


def segment_trajectory(trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a ground atom trajectory."""
    if CFG.segmenter == "atom_changes":
        return _segment_with_atom_changes(trajectory)
    if CFG.segmenter == "option_changes":
        return _segment_with_option_changes(trajectory)
    if CFG.segmenter == "oracle":
        return _segment_with_oracle(trajectory)
    if CFG.segmenter == "contacts":
        return _segment_with_contact_changes(trajectory)
    if CFG.segmenter == "every_step":
        return _segment_with_switch_function(trajectory, lambda _: True)
    if CFG.segmenter == "spot":  # pragma: no cover
        return _segment_with_spot_changes(trajectory)
    raise NotImplementedError(f"Unrecognized segmenter: {CFG.segmenter}.")


def _segment_with_atom_changes(
        trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a trajectory whenever the abstract state changes."""

    _, all_atoms = trajectory

    def _switch_fn(t: int) -> bool:
        return all_atoms[t] != all_atoms[t + 1]

    return _segment_with_switch_function(trajectory, _switch_fn)


def _segment_with_contact_changes(
        trajectory: GroundAtomTrajectory) -> List[Segment]:
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

    traj, _ = trajectory
    # If some predicates are excluded, we need to load predicates from the
    # environment in case contact-based ones are excluded. Note that this is
    # not really leaking information because the same effect could be achieved
    # by implementing environment-specific contact detection functions that use
    # the low-level states only; this is just a more concise way to do that.
    env = get_or_create_env(CFG.env)
    keep_preds = {p for p in env.predicates if p.name in keep_pred_names}
    assert len(keep_preds) == len(keep_pred_names)
    all_keep_atoms = []
    for state in traj.states:
        all_keep_atoms.append(utils.abstract(state, keep_preds))

    def _switch_fn(t: int) -> bool:
        return all_keep_atoms[t] != all_keep_atoms[t + 1]

    return _segment_with_switch_function(trajectory, _switch_fn)


def _segment_with_option_changes(
        trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a trajectory whenever the (assumed known) option changes."""

    traj, _ = trajectory

    def _switch_fn(t: int) -> bool:
        # Segment by checking whether the option changes on the next step.
        option_t = traj.actions[t].get_option()
        # As a special case, if this is the last timestep, then use the
        # option's terminal function to check if it completed, or see if the
        # termination was due to max_num_steps_option_rollout.
        if t == len(traj.actions) - 1:
            if t >= CFG.max_num_steps_option_rollout - 1:
                return True
            return option_t.terminal(traj.states[t + 1])
        return option_t is not traj.actions[t + 1].get_option()

    return _segment_with_switch_function(trajectory, _switch_fn)


def _segment_with_spot_changes(
        trajectory: GroundAtomTrajectory) -> List[Segment]:  # pragma: no cover

    traj, _ = trajectory

    def _switch_fn(t: int) -> bool:
        # Actions without options are "special". We include them in the options
        # that came before them. For example, if an object gets lost during
        # placing, the special "find" action is included in the segment for
        # placing. Note that the current implementation assumes that the
        # regular options are singleton options (terminate immediately).
        act = traj.actions[t]
        if not act.has_option():
            assert t > 0
            last_act = traj.actions[t - 1]
            last_option = last_act.get_option()
            act.set_option(last_option)
        if t == len(traj.actions) - 1:
            return True
        next_action_has_option = traj.actions[t + 1].has_option()
        return next_action_has_option

    return _segment_with_switch_function(trajectory, _switch_fn)


def _segment_with_oracle(trajectory: GroundAtomTrajectory) -> List[Segment]:
    """Segment a trajectory using oracle NSRTs.

    If options are known, just uses _segment_with_option_changes().

    Otherwise, starting at the beginning of the trajectory, keeps track of
    which oracle ground NSRTs are applicable. When any of them have their
    effects achieved, that marks the switch point between segments.
    """
    traj, all_atoms = trajectory
    if traj.actions and traj.actions[0].has_option():
        assert CFG.option_learner == "no_learning"
        return _segment_with_option_changes(trajectory)
    env = get_or_create_env(CFG.env)
    env_options = get_gt_options(env.get_name())
    gt_nsrts = get_gt_nsrts(env.get_name(), env.predicates, env_options)
    objects = list(traj.states[0])
    ground_nsrts = {
        ground_nsrt
        for nsrt in gt_nsrts
        for ground_nsrt in utils.all_ground_nsrts(nsrt, objects)
    }
    atoms = all_atoms[0]
    all_expected_next_atoms = [
        utils.apply_operator(n, atoms)
        for n in utils.get_applicable_operators(ground_nsrts, atoms)
    ]

    def _switch_fn(t: int) -> bool:
        nonlocal all_expected_next_atoms  # update at each switch point
        next_atoms = all_atoms[t + 1]
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
