"""The predicators side of the real-robot interface: a factory that builds the
in-process ``RealRobot``, and the helpers that turn buffered actions into the
segments it executes.

**babyrobot is optional and must never be imported at module level.**  It ships
as the private git submodule ``submodules/BabyRobotPredicator`` and is
deliberately absent from ``install_requires``, so predicators' CI -- and any
checkout without access to that repo -- has no ``babyrobot`` on the path.

Scope: turning buffered actions into robot traffic. The caller rolls a plan out
in sim, buffers the joint-target actions, and hands them to ``execute_chunks``,
which splits each chunk into move / gripper segments and ships them. A chunk is
one unit of "execute this, then optionally look". Deciding *when* to ship, and
what to do with any observation that comes back, belongs to the caller --
``RealRobotExecutor``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from predicators.settings import CFG
from predicators.structs import Action, Array

if TYPE_CHECKING:  # pragma: no cover -- typing only; never imported at runtime
    from babyrobot.realrobot.messages import Segment
    from babyrobot.realrobot.real_robot import RealRobot

    from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot

_MISSING_BABYROBOT = (
    "real-robot execution needs the private BabyRobotPredicator package, "
    "which predicators carries as the git submodule "
    "submodules/BabyRobotPredicator. Check it out and install it:\n"
    "    git submodule update --init submodules/BabyRobotPredicator\n"
    "    pip install -e submodules/BabyRobotPredicator")

# How much wider than the GRASP COMMAND counts as a release. The splitter only
# ever sees commands, never achieved positions, so this is measured against
# what Grasp asked for -- not against where the fingers came to rest.
#
# STOPGAP, and a fitted constant rather than a derived one. After a grasp the
# carry phases command `achieved - 1mm`, and the fingers stall on the object
# well short of the grasp command, so the carry command sits ABOVE it without
# anything having been released: measured on a Pick as 0.00000 -> 0.00558,
# which cleared the old 0.005 by 0.58mm and shipped close, open, close to the
# arm. A genuine release measures 0.0122 on the same scale. 0.008 is simply a
# value between the two.
#
# It cannot be derived, because the two cases are indistinguishable from the
# command stream: both are "widen after a grasp" and only the magnitude
# differs. The real fix is to stop inferring intent from widths and carry the
# skill's own finger_status through on Action.extra_info, at which point this
# constant only guards actions that arrive without it.
_RELEASE_EPS = 0.008


class MissingBabyRobotError(ImportError):
    """babyrobot is not importable, so no real robot can be constructed."""


@dataclass(frozen=True)
class GripperJointLayout:
    """Where the finger joints sit in an action array, and what open / closed
    finger values look like.

    This is everything the splitting helpers need to read a gripper
    command off a joint-target action, so they depend on four numbers
    rather than on a ``SingleArmPyBulletRobot``.
    """
    left_finger_joint_idx: int
    right_finger_joint_idx: int
    open_fingers: float
    closed_fingers: float

    @property
    def finger_joint_idxs(self) -> Tuple[int, int]:
        """The finger entries, which arm-only waypoints drop."""
        return (self.left_finger_joint_idx, self.right_finger_joint_idx)


def gripper_joint_layout_from_robot(
        robot: "SingleArmPyBulletRobot") -> GripperJointLayout:
    """Read the layout off a pybullet robot."""
    return GripperJointLayout(
        left_finger_joint_idx=robot.left_finger_joint_idx,
        right_finger_joint_idx=robot.right_finger_joint_idx,
        open_fingers=robot.open_fingers,
        closed_fingers=robot.closed_fingers)


def make_real_robot(
        dry: Optional[bool] = None,
        perception: Any = None,
        home_joints: Optional[Sequence[float]] = None) -> "RealRobot":
    """Construct the in-process ``RealRobot``, importing babyrobot lazily.

    ``dry`` defaults to ``CFG.real_robot_dry`` (no arm is built, so arm
    calls are no-ops); ``perception`` defaults to whatever
    ``CFG.real_robot_perception`` names. Raises ``MissingBabyRobotError``
    -- naming the submodule and the install command -- when babyrobot is
    absent, which is the failure a checkout without access hits.
    """
    # pylint: disable=import-outside-toplevel,import-error
    try:
        from babyrobot.realrobot.real_robot import RealRobot as _RealRobot
    except ImportError as e:
        raise MissingBabyRobotError(_MISSING_BABYROBOT) from e
    if dry is None:
        dry = CFG.real_robot_dry
    if perception is None:
        perception = _make_perception()
    return _RealRobot(perception=perception, dry=dry, home_joints=home_joints)


def _make_perception() -> Any:
    """Build the perception source named by ``CFG.real_robot_perception``.

    ``"zed"`` (the default) is the live session-scoped ZED perception the
    closed loop needs: one instance held open for the whole run, which
    ``RealRobot`` opens on construction and closes on ``close()``.
    ``"scene_file"`` replays ``CFG.domino_real_scene`` -- a cameraless
    stand-in that always reports the captured layout, so it exercises the
    plumbing but never reports a topple. ``"none"`` (or ``None``, which
    is what a launcher config's ``"none"`` becomes) leaves the robot
    without cameras at all, which only a blind open-loop run can use.

    The table height is passed through rather than left to babyrobot's
    own default: perception and the base -> world transplant have to
    agree on where the table is, and they are configured separately.
    """
    # pylint: disable=import-outside-toplevel,import-error
    kind = CFG.real_robot_perception
    # A launcher config cannot deliver the string: utils.string_to_python
    # _object maps both "None" and "none" to None on the way in from the
    # command line, so a config asking for no cameras arrives as None.
    if kind is None or kind == "none":
        return None
    if kind == "scene_file":
        from babyrobot.realrobot.perception import FileDominoPerception
        return FileDominoPerception(CFG.domino_real_scene)
    if kind == "zed":
        from babyrobot.realrobot.perception import DominoPerception
        return DominoPerception(table_z=float(CFG.domino_real_table_z))
    raise ValueError(f"unknown real_robot_perception: {kind!r}")


def reset_arm(robot: "RealRobot", joints: Sequence[float]) -> Sequence[float]:
    """Home the arm to ``joints`` and open the gripper (blocking).

    Returns the joint positions the arm reports afterwards.
    """
    # pylint: disable=import-outside-toplevel,import-error
    from babyrobot.realrobot.messages import ResetArmRequest
    reply = robot.reset_arm(ResetArmRequest(joints=tuple(joints)))
    return reply.joints


def reset_env(robot: "RealRobot",
              joints: Optional[Sequence[float]] = None) -> Any:
    """Home the arm, wait for a human to rearrange the scene, then look.
    Blocking, and deliberately unbounded.

    Returns the observation captured *after* the human confirmed.
    """
    # pylint: disable=import-outside-toplevel,import-error
    from babyrobot.realrobot.messages import ResetEnvRequest
    request = ResetEnvRequest(
        joints=tuple(joints) if joints is not None else None)
    return robot.reset_env(request)


def execute_chunks(robot: "RealRobot",
                   chunks: Sequence[Sequence[Action]],
                   layout: GripperJointLayout,
                   observe: bool = False,
                   settle_s: float = 0.0) -> List[Any]:
    """Split each chunk of buffered actions into move / gripper segments and
    execute the chunks in order (blocking).

    One chunk is one unit of "execute this, then optionally look": with
    ``observe`` the reply carries one observation per chunk, which is
    how the executor gets a look at the scene per option. With
    ``observe=False`` nothing comes back and the caller's world state
    stays the sim's prediction.

    Chunks that split into no segments are dropped rather than shipped,
    so an empty chunk cannot silently consume one of the observations
    the caller is about to zip against its chunks.
    """
    # pylint: disable=import-outside-toplevel,import-error
    from babyrobot.realrobot.messages import StepRequest
    segmented = [_split_actions(actions, layout) for actions in chunks]
    request_chunks = tuple(
        tuple(segments) for segments in segmented if segments)
    if not request_chunks:
        return []
    reply = robot.step(
        StepRequest(chunks=request_chunks, observe=observe, settle_s=settle_s))
    return list(reply.observations)


def _split_actions(actions: Sequence[Action],
                   layout: GripperJointLayout) -> List["Segment"]:
    """Joint-target actions -> [Segment(move, waypoints) | Segment(gripper)].

    Consecutive same-gripper steps coalesce into one move of arm waypoints,
    with the finger joints removed.

    Stateless ACROSS calls: gripper tracking restarts every call, so a chunk
    that begins already holding an object re-emits its leading ``close``.
    ``RealRobot`` drops that redundant command session-wide, which is what
    makes per-chunk shipping safe.
    """
    # pylint: disable=import-outside-toplevel,import-error
    from babyrobot.realrobot.messages import Segment as _Segment
    gidx = layout.finger_joint_idxs
    closed, opened = layout.closed_fingers, layout.open_fingers

    # A width this far above `closed` is tight enough to be a grasp. Only used
    # to spot the START of a grasp; a release is judged against the grasp, not
    # against this.
    close_tol = 0.05 * abs(opened - closed)

    def arm_only(arr: Array) -> Tuple[float, ...]:
        return tuple(float(v) for i, v in enumerate(arr) if i not in gidx)

    segments: List["Segment"] = []
    cur_grip: str = ""
    # Tightest width commanded since the hand closed; inf while it is open.
    grip_ref = math.inf
    # Width at which the hand last released; inf before any release.
    open_ref = math.inf
    moves: List[Tuple[float, ...]] = []
    for action in actions:
        arr = action.arr
        v = float(arr[layout.left_finger_joint_idx])
        if cur_grip == "close":
            # Judge a release against the GRASP width, not against
            # `closed_fingers`. Otherwise the release width is still under
            # the closed threshold. That is what made the hand hold on
            # through the retreat and drop the domino from transport height.
            if v > grip_ref + _RELEASE_EPS:
                g = "open"
            else:
                g = "close"
                grip_ref = min(grip_ref, v)
        else:
            # Re-close only well below where the hand released, since a firm
            # grasp lets go at a width still under `closed_fingers`.
            g = ("close" if v <= closed + close_tol
                 and v < open_ref - _RELEASE_EPS else "open")
            if g == "close":
                grip_ref = v
        if g != cur_grip:
            if g == "open":
                open_ref = v
                grip_ref = math.inf
            if moves:
                segments.append(_Segment(type="move", waypoints=tuple(moves)))
                moves = []
            segments.append(_Segment(type="gripper", command=g))
            cur_grip = g
        moves.append(arm_only(arr))
    if moves:
        segments.append(_Segment(type="move", waypoints=tuple(moves)))
    return segments
