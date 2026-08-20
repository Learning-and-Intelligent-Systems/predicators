"""Driving a real arm from a simulated env's rollout.

predicators rolls each option out in simulation exactly as it always has. This
module ships the resulting joint trajectory to the real arm, looks at the
scene, and writes what it saw back into the simulated **twin**. The env stays
pure simulation and never learns that a robot exists: it exposes one optional
collaborator (``PyBulletEnv.attach_executor``) and calls it after each reset
and each step.

Why the twin has to be corrected, rather than merely handing the agent a
perceived state: the episode loop is ``obs = env.step(act)``, and
``PyBulletEnv`` advances *its own physics client* and reads the observation
back out of it. Writing perception into the twin is how perception
reaches the agent at all. If we perceive without syncing, the correction is
overwritten by the twin's own simulation on the very next action.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, cast

import numpy as np

from predicators import utils
from predicators.envs.base_env import BaseEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.ground_truth_models.skill_factories.wait import \
    note_external_state_change
from predicators.pybullet_helpers.real_robot_bridge import execute_chunks, \
    make_real_robot, reset_arm, reset_env
from predicators.pybullet_helpers.real_robot_recorder import episode_stamp, \
    make_episode_recorder
from predicators.pybullet_helpers.real_robot_snapshot import \
    MarkerlessSnapshotPerception
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Observation, State

# The domain conversions the env must provide, and what each is for. Checked
# when an executor is built, so a missing one names itself instead of surfacing
# as an AttributeError several hundred robot-moving actions later.
_REQUIRED_HOOKS = {
    "state_from_observation": "convert a perceived observation into a State",
    "task_from_observation": "build a task from a perceived observation",
}


def _ends_at(option: Any, obs: Observation) -> bool:
    """Whether ``option`` ends at ``obs``, without disturbing the option.

    ``terminal`` may be stateful: ``Wait`` counts consecutive settled
    steps in ``option.memory``, and the option's own policy is already
    counting that series one call per step. Asking here without putting
    the memory back would insert an extra sample per step, so ``Wait``
    would judge the scene settled in a third of the steps it actually
    takes.
    """
    saved = dict(option.memory)
    try:
        return cast(bool, option.terminal(obs))
    finally:
        option.memory.clear()
        option.memory.update(saved)


class _DomainHooks(Protocol):
    """The domain-specific conversions, which no base class can declare.

    Everything else this module calls is on ``PyBulletEnv``. These two
    are not: turning a perceived observation into a ``State`` or a task
    needs to know what the observation *means*, which is the domain
    env's knowledge alone. Python has no intersection type, so the
    requirement is stated here and checked at construction.
    """

    def state_from_observation(self, obs: Any, prev_state: State) -> State:
        """Correct ``prev_state`` with what was just perceived."""

    def task_from_observation(self, obs: Any,
                              train_or_test: str) -> EnvironmentTask:
        """Build a task from what was just perceived."""


class OptionBoundaryBuffer:
    """Collects actions and hands back one option's worth at its boundary.

    Pure: no robot, no env, no config. An action carrying no option is
    dropped rather than buffered -- there is no boundary to attribute it
    to, and shipping it alone would be motion belonging to no skill.
    """

    def __init__(self) -> None:
        self._actions: List[Action] = []

    def __len__(self) -> int:
        return len(self._actions)

    def add(self, action: Action, obs: Observation) -> Optional[List[Action]]:
        """Buffer ``action``; return a chunk when it ends its option.

        ``obs`` is the state the action led to, which is what the
        option's own ``terminal`` is defined on.
        """
        if not action.has_option():
            return None
        self._actions.append(action)
        if not _ends_at(action.get_option(), obs):
            return None
        chunk, self._actions = self._actions, []
        return chunk

    def discard(self) -> int:
        """Drop whatever is buffered; return how many actions were lost."""
        lost = len(self._actions)
        self._actions = []
        return lost


class TwinCorrector:
    """Writes perception into the simulated twin, and reports divergence."""

    def __init__(self, env: PyBulletEnv, divergence_atol: float) -> None:
        self._env = env
        self._divergence_atol = divergence_atol
        # Looks so far, used to name the dumps in the order they happened.
        self._look_count = 0
        # Largest per-object position disagreement between the twin and the
        # real scene at the last look, in metres; None before the first look.
        self.last_divergence: Optional[float] = None

    def absorb(self, observation: Any) -> Observation:
        """Write what the cameras saw into the twin; return its new reading.

        Divergence is measured against the twin's *pre-sync* state --
        the sim's prediction of where the scene would be -- because that
        is the comparison that says reality went somewhere the model did
        not. ``_set_state``'s own reconstruction check cannot answer
        this: it round-trips the state it was asked to write and
        measures whether PyBullet could realize the request. A toppled
        domino is perfectly realizable, so it never fires on the
        interesting case.
        """
        predicted = self._env.get_observation()
        assert isinstance(predicted, State)
        domain = cast(_DomainHooks, self._env)
        perceived = domain.state_from_observation(observation, predicted)
        self.last_divergence = _max_position_divergence(predicted, perceived)
        self._look_count += 1
        per_object = _per_object_divergence(predicted, perceived)
        # Log every look, not only the ones over tolerance: a run whose looks
        # all behaved should still say so, and the per-object breakdown is
        # what distinguishes one bad capture from a systematic offset.
        logging.info(
            "real robot: look %d, worst %.4f m (tolerance %.3f m); %s",
            self._look_count, self.last_divergence or float("nan"),
            self._divergence_atol, ", ".join(f"{obj.name} {dist:.4f}"
                                             for obj, dist in per_object))
        if self.last_divergence is not None and \
                self.last_divergence > self._divergence_atol:
            logging.warning(
                "real robot: the scene is %.3f m from where the twin "
                "predicted (tolerance %.3f m); the twin is being corrected, "
                "but the current plan was made against the prediction",
                self.last_divergence, self._divergence_atol)
        _dump_look(self._look_count, predicted, perceived, per_object,
                   self.last_divergence)
        self._env.sync_to_state(perceived)
        # No need to refresh the env's cached observation by hand:
        # PyBulletEnv.get_observation re-reads the state out of PyBullet, so
        # this picks the corrected world up (and re-caches it).
        return self._env.get_observation()


def _per_object_divergence(predicted: State,
                           perceived: State) -> List[Tuple[Any, float]]:
    """Per-object ``(object, distance)``, worst first.

    ``_max_position_divergence`` answers "how bad", which is what the
    tolerance is checked against; this answers "which object", which is
    what tells a knocked domino apart from a table-height offset shared
    by all of them.
    """
    out = []
    for obj in predicted.data:
        if obj not in perceived.data:
            continue
        if not {"x", "y", "z"}.issubset(obj.type.feature_names):
            continue
        delta = np.array(
            [predicted.get(obj, f) - perceived.get(obj, f) for f in "xyz"])
        out.append((obj, float(np.linalg.norm(delta))))
    return sorted(out, key=lambda pair: pair[1], reverse=True)


def _dump_look(index: int, predicted: State, perceived: State,
               per_object: List[Tuple[Any,
                                      float]], worst: Optional[float]) -> None:
    """Write one look to ``CFG.real_robot_observation_dump_dir`` as JSON.

    Records the twin's prediction beside what was perceived, so a
    session can be re-examined offline -- including the looks that
    raised no warning. Failing to write must never take the arm down
    mid-episode, so any error here is logged and swallowed.
    """
    out_dir = CFG.real_robot_observation_dump_dir
    if not out_dir:
        return

    def _poses(state: State) -> Dict[str, List[float]]:
        return {
            obj.name: [float(state.get(obj, f)) for f in "xyz"]
            for obj in state.data
            if {"x", "y", "z"}.issubset(obj.type.feature_names)
        }

    record = {
        "look": index,
        "worst_divergence": worst,
        "divergence_atol": CFG.real_robot_divergence_atol,
        "predicted": _poses(predicted),
        "perceived": _poses(perceived),
        "per_object": {obj.name: dist
                       for obj, dist in per_object},
    }
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"look_{index:04d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
    except OSError as exc:  # pragma: no cover - disk trouble only
        logging.warning("real robot: could not write %s (%s)", out_dir, exc)


class RealRobotExecutor:
    """Ships each option's trajectory to the arm and corrects the twin.

    Implements ``PyBulletEnv``'s ``ActionExecutor`` port. Settings are
    constructor arguments rather than reads of the global config, so a
    test configures one by building it.
    """

    def __init__(self,
                 env: PyBulletEnv,
                 robot: Any,
                 observe_at_boundaries: bool = True,
                 settle_s: float = 0.0,
                 divergence_atol: float = 0.02,
                 human_reset: bool = True,
                 open_loop_episode: bool = False,
                 record_from_option: str = "",
                 recorder: Any = None) -> None:
        missing = sorted(name for name in _REQUIRED_HOOKS
                         if not callable(getattr(env, name, None)))
        if missing:
            raise TypeError(
                f"{type(env).__name__} cannot be driven on real hardware: "
                "it is missing " +
                ", ".join(f"{name}() (to {_REQUIRED_HOOKS[name]})"
                          for name in missing))
        if observe_at_boundaries and not getattr(robot, "has_perception",
                                                 False):
            raise ValueError(
                "asked to look at the scene between options, but the robot "
                "has no perception configured; set real_robot_perception, or "
                "turn the option-boundary look off for a blind open-loop run")
        if human_reset and not getattr(robot, "has_perception", False):
            raise ValueError(
                "human resets rebuild each episode's task from the scene, so "
                "the robot needs perception; set real_robot_perception, or "
                "turn real_robot_human_reset off to keep the captured scene")
        if open_loop_episode and observe_at_boundaries:
            raise ValueError(
                "open-loop episodes ship the whole plan after the episode is "
                "simulated, so there is no moment between two options at "
                "which to look at the scene; turn "
                "real_robot_observe_at_option_boundary off, or turn "
                "real_robot_open_loop_episode off to keep the boundary looks")
        self._env = env
        self._robot = robot
        self._observe = observe_at_boundaries
        self._open_loop = open_loop_episode
        # Option name the take is opened in front of. Empty records the whole
        # batch, which is what every episode did before this existed.
        self._record_from_option = record_from_option
        # Chunks held back for the end of the episode, each with the name of
        # the option that produced it -- the name is what _ship_episode finds
        # the recording boundary by. Only ever non-empty under open-loop; the
        # per-boundary path ships and forgets.
        self._pending: List[Tuple[str, List[Action]]] = []
        # Records each episode to an SVO take for offline pose estimation.
        # None when the run is not recording. Opened here rather than at the
        # first episode: a learning cycle is many episodes, and per-episode
        # camera init and warmup would otherwise be paid every time.
        self._recorder = recorder
        if self._recorder is not None:
            self._recorder.open()
            if self._boxes_wanted() and not record_from_option:
                # Before the first episode, while a human is still standing
                # at the bench: one drag window for the whole run rather than
                # one per take. Only correct because the take opens at the
                # reset, so the arrangement here is the one its first frame
                # shows. With record_from_option set it does not, and the
                # draw moves to the boundary in _ship_episode.
                self._recorder.ensure_boxes()
        # Whether a take is currently open, so an episode that ends without
        # ever having started one does not try to stop it.
        self._recording_episode = False
        # Episodes begun this run, used to name takes.
        self._episode_num = 0
        # The split and task the current episode is for, kept because
        # open-loop names its take at the ship rather than at the reset.
        self._episode_split: Tuple[str, int] = ("train", 0)
        self._settle_s = settle_s
        self._human_reset = human_reset
        self._buffer = OptionBoundaryBuffer()
        self._corrector = TwinCorrector(env, divergence_atol)
        # The scene has to be arranged before the first episode, so a reset is
        # owed from the start. Set again by every reset, i.e. once per episode.
        self._reset_pending = True
        # What the last reset saw, kept so both splits rebuild from the same
        # arrangement. None until the first look.
        self._reset_observation: Optional[Observation] = None
        # The twin's home arm configuration, captured now: an executor is
        # attached right after the env is built, so the simulated arm is still
        # at home. A scene reset has to send the real arm somewhere before the
        # human reaches in, and this is that somewhere .
        self._home_arm = self._home_arm_joints(env.get_observation())
        # How many times the scene has been (re)perceived for a task. Read by
        # tests and worth logging on a hardware session.
        self.resets_done = 0

    @property
    def last_divergence(self) -> Optional[float]:
        """How far the scene was from the twin at the last look."""
        return self._corrector.last_divergence

    # -- the ActionExecutor port -------------------------------------------
    def tasks_for(self, train_or_test: str) -> Optional[List[EnvironmentTask]]:
        """Rebuild this split's task from the scene, resetting it first.

        **Why this happens here and not in ``reset``.** The task has to
        be rebuilt before anything consumes it, and the two callers
        consume it differently:

        * *Evaluation* solves at reset. ``_solve_task``
          (``main.py:774``) calls ``cogman.reset(env_task)`` with no
          override policy, so ``_reset_policy`` runs
          ``approach.solve(task)`` -- before ``env.reset`` is ever
          called. A human reset performed in ``env.reset`` would
          therefore plan against a scene that no longer exists.
        * *Exploration* does not solve at reset: the online loop sets
          an override policy first (``main.py:551``), so
          ``_reset_policy`` takes that branch. The task still matters,
          because it is what ``env.reset`` initializes the simulated
          twin from -- a stale one starts every episode from the
          captured scene rather than the one just arranged.

        For a real environment "give me the train task" honestly means
        "look at the scene", so that is what it does.

        **Both splits are served by one look.** A physical reset arranges
        one scene, and the person who arranged it meant it for whatever
        runs next -- so the perceived observation is kept and the second
        split rebuilds from it rather than being refused. Consuming the
        look on whichever split asked first is what left the other one
        holding the captured-scene task: with the online loop off,
        ``main.py`` requests the train tasks during setup, so the *test*
        task -- the one that gets solved -- silently stayed on the scene
        JSON while the arm executed in the real one.

        Returns None -- leaving the env's captured-scene task alone --
        when no scene has been looked at yet, or when human resets are
        off. The latter is what keeps a fixed-plan replay reproducible.
        """
        if not self._human_reset:
            self._refuse_stale_task(train_or_test)
            return None
        if self._reset_pending:
            # Homes the arm out of the way, blocks until the human confirms,
            # then perceives.
            self._reset_observation = reset_env(self._robot, self._home_arm)
            self._reset_pending = False
            self.resets_done += 1
            logging.info(
                "real robot: scene reset #%d; rebuilding the %s task "
                "from what the cameras see", self.resets_done, train_or_test)
        elif self._reset_observation is not None:
            logging.info(
                "real robot: rebuilding the %s task from scene reset #%d "
                "(one arrangement serves both splits)", train_or_test,
                self.resets_done)
        else:
            return None
        domain = cast(_DomainHooks, self._env)
        # One task, because a physical scene is one scene. The env keeps the
        # list length stable, so task indices already handed out stay valid.
        return [
            domain.task_from_observation(self._reset_observation,
                                         train_or_test)
        ]

    def _refuse_stale_task(self, train_or_test: str) -> None:
        """Refuse to hand back the captured scene while the cameras are live.

        Falling through to it is silent: the JSON's poses look like a
        scene, planning succeeds against them, and the twin only jumps
        to the truth at the first option boundary -- by which point the
        plan was written for a world that is not there. Raising here
        costs a run that was going to be meaningless anyway.
        """
        if CFG.real_robot_perception != "zed":
            return  # not looking at anything, so nothing to be stale about
        if CFG.real_robot_allow_captured_scene_task:
            return  # replaying a plan written against exactly those poses
        raise ValueError(
            f"the {train_or_test} task would come from "
            f"{CFG.domino_real_scene!r}, a snapshot, while the cameras are "
            "live and no one has looked at the scene. Turn "
            "real_robot_human_reset on so the task is rebuilt from a look, "
            "or set real_robot_allow_captured_scene_task if you mean to "
            "replay a plan written against that capture.")

    def after_reset(self, train_or_test: str, task_idx: int,
                    obs: Observation) -> None:
        """Home the real arm to wherever the twin just reset to.

        The twin is reset first because its home joint configuration is
        what the option trajectories are planned from.

        Both train and test splits execute. Real mode is a property of
        an executor being attached, not of the split.

        This is also where the next reset is owed from: an episode has
        just begun, so the *following* task request has to face a
        freshly arranged scene. Marking it here rather than at the end
        of an episode is what makes it exactly one prompt per episode.
        """
        self._reset_pending = True
        if self._pending:
            # finish_execution did not run for the previous episode. Shipping
            # these now would drive the arm through the last episode's plan
            # against this episode's scene, so they are dropped instead.
            logging.warning(
                "real robot: dropping %d option(s) left over from an episode "
                "that never finished executing", len(self._pending))
            self._pending = []
        lost = self._buffer.discard()
        if lost:
            # The previous episode ended mid-option (step limit, or an
            # exception). Half a skill is not worth executing on the arm, so
            # it is dropped rather than shipped late.
            logging.warning(
                "real robot: dropping %d buffered action(s) from an episode "
                "that ended mid-option; they were never shipped", lost)
        reset_arm(self._robot, self._home_arm_joints(obs))
        # Under open-loop the arm does nothing between here and the batch, so
        # recording now would capture the twin simulating -- a static scene,
        # and the larger half of the take. Measured on run_20260817_165815:
        # 258 s recorded against 153 s of motion. The take is started at the
        # ship instead. Per-boundary shipping has motion throughout the
        # episode, so it still records from the reset.
        self._episode_split = (train_or_test, task_idx)
        if not self._open_loop:
            self._start_recording(train_or_test, task_idx)

    def _start_recording(self, train_or_test: str, task_idx: int) -> None:
        """Begin this episode's take, after the arm is home.

        After the homing rather than before it: the arm's trip to home
        is not part of the episode being measured, and the take is the
        input to pose estimation rather than an archive of the session.
        A take left open by an episode that never finished is stopped
        first, so this episode does not append itself to the last one's
        recording.
        """
        if self._recorder is None:
            return
        if self._recording_episode:
            logging.warning(
                "real robot: a take was still open at the start of an "
                "episode; closing it before starting this one")
            self._recorder.stop_episode()
            self._recording_episode = False
        self._episode_num += 1
        self._recorder.start_episode(
            episode_stamp(train_or_test, task_idx, self._episode_num))
        self._recording_episode = True

    def after_step(self, action: Action, obs: Observation) -> Observation:
        """Buffer the action, and ship at an option boundary.

        Under open-loop the completed chunk is held instead of shipped,
        and ``obs`` comes back untouched. That is not a special case so
        much as the same one: shipping with ``observe`` off returns no
        observations, so the loop below never runs and this method
        already returned ``obs`` unchanged. Deferring a call whose only
        effect is on the arm cannot change what the rollout sees.
        """
        chunk = self._buffer.add(action, obs)
        if chunk is None:
            return obs
        if self._open_loop:
            self._pending.append((action.get_option().name, chunk))
            return obs
        observations = execute_chunks(self._robot, [chunk],
                                      self._env.gripper_joint_layout(),
                                      observe=self._observe,
                                      settle_s=self._settle_s)
        for observation in observations:
            obs = self._corrector.absorb(observation)
        # The correction moved objects, but the scene did not move. Options
        # that judge the scene settled have to be told, or every look would
        # read as motion and they would never see it come to rest.
        if isinstance(obs, State):
            note_external_state_change(action.get_option(), obs)
        return obs

    def after_episode(self, completed: bool) -> None:
        """Ship the episode's motion, or drop it if it never finished.

        Only open-loop has anything outstanding; per-boundary shipping
        has already happened by the time this runs.

        A partial plan is dropped rather than shipped. The buffer holds
        whole options, so what survives an abnormal end is a prefix --
        half a bridge, or a transport with no place at the end of it --
        and the arm would execute it with nobody having decided it was
        a good idea. The information that it was partial exists only
        here, so this is the last place that judgement can be made.

        The recording is stopped either way, in a ``finally``: shipping
        must not happen on an abnormal end, but a take left open runs
        until the disk fills. The two have opposite defaults, which is
        why one is conditional and the other is not.
        """
        try:
            self._ship_episode(completed)
        finally:
            self._stop_recording()

    def _ship_episode(self, completed: bool) -> None:
        """Send the episode's buffered motion to the arm, or drop it."""
        pending, self._pending = self._pending, []
        lost_partial = self._buffer.discard()
        if not pending:
            return
        if not completed:
            logging.warning(
                "real robot: dropping %d buffered option(s) unshipped -- the "
                "episode did not run to completion, so what is buffered is a "
                "partial plan", len(pending))
            return
        if lost_partial:
            # A completed episode should not also have a half-option in
            # hand; if it does, the chunks are still whole and shippable,
            # but the discrepancy is worth a line in the log.
            logging.warning(
                "real robot: episode completed with %d action(s) mid-option; "
                "shipping the %d whole option(s) and dropping those",
                lost_partial, len(pending))
        names = [name for name, _ in pending]
        chunks = [chunk for _, chunk in pending]
        start = self._recording_boundary(names)
        # Everything before the boundary runs unrecorded. The twin has already
        # simulated all of it, so splitting the shipment costs one round trip
        # to the controller and NOT a planner call -- which is why this does
        # not give back what open-loop batching bought. The arm coming to rest
        # here is a gain of its own: the free-run is anchored at the last rest
        # state before the push, and now there really is one.
        if start:
            self._ship_batch(chunks[:start], "prologue")
            if self._boxes_wanted() and self._recorder is not None:
                # HERE, not at run start. The boxes are prompts for the take's
                # first frame, and the prologue has just rearranged the row --
                # boxes drawn before it point at where two dominoes used to
                # be, and stage 2 fits masks to whatever is inside the box it
                # was given rather than reporting that the box is empty. The
                # arm is at rest and the scene is final, which is also the
                # only moment a human can draw them correctly.
                self._recorder.ensure_boxes()
        # The take brackets the motion, not the episode: everything before
        # this point is the twin simulating, with the arm parked.
        if self._open_loop:
            self._start_recording(*self._episode_split)
        self._ship_batch(chunks[start:], "batch")

    @staticmethod
    def _boxes_wanted() -> bool:
        """Whether this run draws its own stage-2 prompt boxes.

        False when a boxes file was given: that is the unattended path,
        and it names an arrangement the caller vouches for.
        """
        return bool(CFG.real_robot_pick_boxes_at_start
                    and not CFG.real_robot_snapshot_boxes_json)

    def _recording_boundary(self, names: List[str]) -> int:
        """Index of the first chunk the take should be open in front of.

        Only the cascade is scored, so recording the pick-and-place that
        arranges the row buys nothing and costs most of the take: on
        run_20260818_092302 the push landed 107 s into a 131 s track, and
        post-processing scales with frames.

        Recording everything is the fallback, deliberately. Too much video
        is slow; too little is an episode whose first topple happened off
        camera, and the first onset is what every interval is measured
        against.
        """
        wanted = self._record_from_option
        if not wanted:
            return 0
        for index, name in enumerate(names):
            if name == wanted:
                return index
        logging.warning(
            "real robot: asked to record from option %r, which this episode "
            "never ran (it ran %s); recording the whole batch instead", wanted,
            ", ".join(names) or "nothing")
        return 0

    def _ship_batch(self, chunks: List[List[Action]], label: str) -> None:
        """Send one contiguous run of chunks, logging both clocks.

        ``execute_chunks`` packs the list into a single StepRequest, and
        ``_split_actions`` restarts its gripper tracking per call, which
        ``RealRobot`` dedups session-wide -- the same property that made
        per-boundary shipping safe, and what lets the episode be split
        in two here without the arm seeing a redundant gripper command.
        """
        if not chunks:
            return
        started_monotonic_ns = time.monotonic_ns()
        started_wall_ns = time.time_ns()
        logging.info(
            "real robot: shipping %d option(s) as one %s "
            "(monotonic_ns=%d wall_ns=%d)", len(chunks), label,
            started_monotonic_ns, started_wall_ns)
        execute_chunks(self._robot,
                       chunks,
                       self._env.gripper_joint_layout(),
                       observe=self._observe,
                       settle_s=self._settle_s)
        logging.info(
            "real robot: %s done (monotonic_ns=%d wall_ns=%d, "
            "elapsed %.3fs)", label, time.monotonic_ns(), time.time_ns(),
            (time.monotonic_ns() - started_monotonic_ns) / 1e9)

    def _stop_recording(self) -> None:
        """End this episode's take, if one is open."""
        if self._recorder is None or not self._recording_episode:
            return
        self._recording_episode = False
        self._recorder.stop_episode()

    # -- helpers -----------------------------------------------------------
    def _home_arm_joints(self, obs: Observation) -> List[float]:
        """The twin's home arm joints, fingers dropped.

        ``reset_arm`` takes the 7 arm joints; the twin's joint vector
        also carries the two finger joints, and the layout is what says
        which entries those are.
        """
        assert isinstance(obs, utils.PyBulletState), \
            f"the twin must observe joint positions, got {type(obs).__name__}"
        fingers = set(self._env.gripper_joint_layout().finger_joint_idxs)
        return [
            float(v) for i, v in enumerate(obs.joint_positions)
            if i not in fingers
        ]


def _max_position_divergence(predicted: State,
                             perceived: State) -> Optional[float]:
    """Largest ``(x, y, z)`` distance between the same object in two states.

    Objects missing from either state, or without positional features,
    are skipped; ``None`` means there was nothing comparable.
    """
    worst: Optional[float] = None
    for obj in predicted.data:
        if obj not in perceived.data:
            continue
        if not {"x", "y", "z"}.issubset(obj.type.feature_names):
            continue
        delta = np.array(
            [predicted.get(obj, f) - perceived.get(obj, f) for f in "xyz"])
        distance = float(np.linalg.norm(delta))
        worst = distance if worst is None else max(worst, distance)
    return worst


def _snapshot_perception(recorder: Any) -> MarkerlessSnapshotPerception:
    """The scene look that a snapshot rebuild uses instead of a live one."""
    serials = recorder.serials
    if CFG.real_robot_snapshot_fuse_cameras:
        if len(serials) != 2:
            raise ValueError(
                "real_robot_snapshot_fuse_cameras fits the scene from two "
                f"cameras and fuses them, but the recorder holds {serials}; "
                "give it exactly two, or turn the setting off to fit from "
                "one.")
        return MarkerlessSnapshotPerception(
            recorder,
            serials=serials,
            frames=CFG.real_robot_snapshot_frames,
            boxes_json_by_camera=_snapshot_boxes_by_camera(serials))
    serial = CFG.real_robot_snapshot_camera or (serials[0] if serials else "")
    if not serial:
        raise ValueError(
            "real_robot_snapshot_rebuild needs a camera to fit the scene "
            "from, and the recorder reported no serials; set "
            "real_robot_snapshot_camera.")
    if serials and serial not in serials:
        raise ValueError(
            f"real_robot_snapshot_camera {serial!r} is not one of the "
            f"recorder's cameras {serials}; the scene is fitted from a take "
            "that session records, so it has to be one of them.")
    return MarkerlessSnapshotPerception(recorder,
                                        serial=serial,
                                        frames=CFG.real_robot_snapshot_frames)


def _snapshot_boxes_by_camera(serials: Sequence[str]) -> Dict[str, str]:
    """``real_robot_snapshot_boxes_json_by_camera``, parsed and checked.

    Checked here rather than left to the pipeline because the failure it
    prevents is expensive and late: a serial that is not one of the recorder's
    means that camera simply gets no boxes, and the first anyone hears of it is
    stage 2 opening a drag window in the middle of a learning run.
    """
    raw = CFG.real_robot_snapshot_boxes_json_by_camera
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as e:
        raise ValueError(
            "real_robot_snapshot_boxes_json_by_camera must be a JSON object "
            f'mapping ZED serial to a boxes.json path, e.g. {{"{serials[0]}": '
            f'"/path/boxes.json"}}; could not parse it: {e}') from e
    if not isinstance(parsed, dict):
        raise ValueError(
            "real_robot_snapshot_boxes_json_by_camera must be a JSON OBJECT "
            f"keyed by ZED serial, got {type(parsed).__name__}")
    unknown = sorted(set(map(str, parsed)) - set(map(str, serials)))
    if unknown:
        raise ValueError(
            f"real_robot_snapshot_boxes_json_by_camera names camera(s) "
            f"{unknown} that the recorder does not hold {list(serials)}; "
            "boxes are per camera, so a serial that is not recorded means "
            "some camera has none.")
    return {str(k): str(v) for k, v in parsed.items()}


def attach_real_robot(env: BaseEnv,
                      robot: Any = None) -> Optional[RealRobotExecutor]:
    """Attach a real-robot executor to ``env`` when the config asks for it.

    Returns ``None`` (having done nothing) when ``real_robot_execute``
    is off and the executor otherwise.
    """
    if not CFG.real_robot_execute:
        return None
    # A contradiction in the config alone, so it is reported before anything
    # about the env or the hardware is examined.
    if CFG.real_robot_record_episodes and CFG.real_robot_perception == "zed":
        raise ValueError(
            "real_robot_record_episodes and a live \"zed\" perception both "
            "want to own the same cameras, and a ZED admits one owner. "
            "Recording feeds the offline markerless pipeline, which does not "
            "need a live look: set real_robot_perception to \"scene_file\" "
            "(or \"none\"), or turn real_robot_snapshot_rebuild on to rebuild "
            "each episode's task from a short take on the recorder's own "
            "session instead.")
    if CFG.real_robot_snapshot_rebuild and not CFG.real_robot_record_episodes:
        raise ValueError(
            "real_robot_snapshot_rebuild takes its snapshot on the episode "
            "recorder's open session, so it needs "
            "real_robot_record_episodes. Opening cameras of its own is the "
            "collision this design exists to avoid.")
    if not isinstance(env, PyBulletEnv):
        raise TypeError(
            f"real_robot_execute needs a PyBullet-backed env to act as the "
            f"twin, but {CFG.env} is a {type(env).__name__}. The twin is what "
            "turns an option into the joint trajectory the arm executes.")
    # The recorder is built BEFORE the robot: under snapshot rebuild the
    # robot's perception is a look served by the recorder's session, so the
    # session has to exist to be handed over.
    recorder = (make_episode_recorder()
                if CFG.real_robot_record_episodes else None)
    if robot is None:
        robot = make_real_robot(perception=_snapshot_perception(recorder)
                                if CFG.real_robot_snapshot_rebuild else None)
    executor = RealRobotExecutor(
        env,
        robot,
        observe_at_boundaries=CFG.real_robot_observe_at_option_boundary,
        settle_s=CFG.real_robot_settle_s,
        divergence_atol=CFG.real_robot_divergence_atol,
        human_reset=CFG.real_robot_human_reset,
        open_loop_episode=CFG.real_robot_open_loop_episode,
        record_from_option=CFG.real_robot_record_from_option,
        recorder=recorder)
    env.attach_executor(executor)
    return executor
