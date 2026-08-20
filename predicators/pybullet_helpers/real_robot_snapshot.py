"""Rebuilding an episode's task from a short markerless take.

The live ``"zed"`` perception is the *marker* pipeline, and the 20 mm ArUco
markers are not resolvable at this camera distance -- 1 of ~7 on one camera and
0 on the other -- so "look at the scene between episodes" cannot be served that
way on this bench. This module serves it the way the markerless pipeline does:
record a short take, fit the known domino box to the masked depth, and read the
scene JSON that falls out.

**Why this does not fight the episode recorder for the cameras.** It does not
open any. A ZED admits one owner, and the owner is the recorder's session; a
snapshot is simply a second, short take on that same open session, taken while
no episode take is running. That is the whole reason this exists rather than a
second capture process.

**Where it plugs in.** ``RealRobot.reset_env`` homes the arm, blocks until a
human confirms the scene is arranged, and only then calls
``perception.observe()``. That last call is exactly the moment a snapshot
should be taken, so this class duck-types babyrobot's perception protocol
(``open`` / ``observe`` / ``close``) and is injected through
``make_real_robot(perception=...)``. Nothing in the reset flow changes.

babyrobot is optional and must never be imported at module level, as in
``real_robot_bridge``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from predicators.settings import CFG

_MISSING_BABYROBOT = (
    "markerless snapshot rebuild needs the private BabyRobotPredicator "
    "package, which predicators carries as the git submodule "
    "submodules/BabyRobotPredicator. Check it out and install it:\n"
    "    git submodule update --init submodules/BabyRobotPredicator\n"
    "    pip install -e submodules/BabyRobotPredicator")

# Stages 1-4 over one recording: (svo, bundle, camera_serial) -> scene JSON.
StageRunner = Callable[[str, str, str], str]
# The same over TWO recordings of one scene, fused: ({serial: svo}, bundle) ->
# scene JSON. A separate type rather than a widened StageRunner because the two
# do genuinely different work, and a caller injecting one should not silently
# satisfy the other.
MultiStageRunner = Callable[[Dict[str, str], str], str]


class MissingBabyRobotError(ImportError):
    """babyrobot is not importable, so no snapshot can be fitted."""


class MarkerlessSnapshotPerception:
    """A scene look served by a short take plus the markerless pipeline.

    Duck-types babyrobot's perception protocol so ``RealRobot`` can hold
    one of these wherever it would hold a ``DominoPerception``.

    ``open`` and ``close`` are deliberately no-ops: this object owns no
    cameras. Taking them here is what would collide with the episode
    recorder, and the collision is the thing being avoided.

    The pipeline runner and the scene loader are injectable so the whole
    lifecycle is testable without a GPU, a camera, or the submodule.
    """

    def __init__(
            self,
            recorder: Any,
            serial: Optional[str] = None,
            runner: Optional[StageRunner] = None,
            scene_loader: Optional[Callable[[str], Any]] = None,
            work_dir: Optional[str] = None,
            frames: int = 5,
            serials: Optional[Sequence[str]] = None,
            multi_runner: Optional[MultiStageRunner] = None,
            boxes_json_by_camera: Optional[Dict[str, str]] = None) -> None:
        # One camera (``serial``) or two (``serials``). With two, the take is
        # fitted once per camera and the results are fused, which is what lets a
        # domino hidden from one view still be reported -- the occlusion case
        # the second camera exists for. The take already holds both recordings:
        # the recorder opens every camera it was given, so the second SVO costs
        # nothing extra to produce and was simply being discarded.
        if serials is None:
            serials = [serial] if serial else []
        self._serials = [str(s) for s in serials]
        if not self._serials:
            raise ValueError(
                "a snapshot rebuild needs at least one camera serial to fit "
                "the scene from")
        self._recorder = recorder
        self._runner = runner
        self._multi_runner = multi_runner
        self._scene_loader = scene_loader
        self._work_dir = work_dir or os.path.join("logs", "zed_snapshots")
        self._frames = int(frames)
        self._boxes_json_by_camera = dict(boxes_json_by_camera or {})
        # Snapshots fitted this run, newest last. Read by tests, and worth
        # having in a hardware session's log.
        self.scenes: list[str] = []

    @property
    def has_perception(self) -> bool:
        """``RealRobot`` asks this before allowing a look."""
        return True

    def open(self) -> None:
        """No cameras of our own to open; the recorder owns them."""

    def close(self) -> None:
        """No cameras of our own to close."""

    def observe(self, settle_s: float = 0.0) -> Any:
        """Take a snapshot, fit it, and return the scene as an observation.

        ``settle_s`` is honoured by the take itself rather than by a
        sleep: the frames are grabbed after the human has stepped away
        and the arm is already home, so there is nothing left moving to
        settle for.
        """
        del settle_s  # nothing is in motion at a scene reset
        take_dir, svos = self._record_snapshot()
        bundle = os.path.join(self._work_dir, os.path.basename(take_dir))
        scene_json = self._run_stages(svos, bundle)
        self.scenes.append(scene_json)
        logging.info("real robot: snapshot scene fitted from %s to %s",
                     ", ".join(self._serials), scene_json)
        return self._load_scene(scene_json)

    # -- helpers -----------------------------------------------------------
    def _record_snapshot(self) -> Tuple[str, Dict[str, str]]:
        """Take a short take on the recorder's session; return (dir, svos).

        Every configured camera's recording has to be there. A take missing
        one is not silently fitted from the other: with two cameras
        configured the scene is meant to be fused, and quietly falling back
        to one would produce a scene that looks exactly like a fused one and
        is not, right where a domino the missing camera could see is the
        reason for having it.
        """
        take_dir, meta = self._recorder.snapshot(frames=self._frames)
        ext = str((meta or {}).get("svo_ext", ".svo2"))
        svos: Dict[str, str] = {}
        for serial in self._serials:
            svo = os.path.join(take_dir, f"zed_{serial}{ext}")
            if not os.path.exists(svo):
                raise FileNotFoundError(
                    f"the snapshot take {take_dir} has no recording for ZED "
                    f"{serial} (expected {os.path.basename(svo)}); every "
                    "camera the scene is fitted from must be one the recorder "
                    "was given")
            svos[serial] = svo
        return take_dir, svos

    def _run_stages(self, svos: Dict[str, str], bundle: str) -> str:
        """Fit the take, through the injected runner or the real pipeline."""
        if len(svos) > 1:
            if self._multi_runner is not None:
                return self._multi_runner(svos, bundle)
            return _default_multi_runner(svos, bundle,
                                         self._boxes_json_by_camera)
        (serial, svo), = svos.items()
        if self._runner is not None:
            return self._runner(svo, bundle, serial)
        return _default_runner(svo, bundle, serial)

    def _load_scene(self, scene_json: str) -> Any:
        """Read the fitted scene as an observation."""
        if self._scene_loader is not None:
            return self._scene_loader(scene_json)
        # pylint: disable=import-outside-toplevel,import-error
        try:
            from babyrobot.realrobot.perception import FileDominoPerception
        except ImportError as e:
            raise MissingBabyRobotError(_MISSING_BABYROBOT) from e
        # Both pipelines write the same scene JSON, which is why the loader
        # that replays a captured scene reads a freshly fitted one unchanged.
        return FileDominoPerception(scene_json).observe(0.0)


def _default_runner(svo: str, bundle: str, serial: str) -> str:
    """Run markerless stages 1-4 over ``svo``, returning the scene JSON.

    Delegates to the submodule's own staging rather than re-deriving the
    argv for four scripts: it already resolves the interpreter the
    stages need (they want pyzed and ultralytics together, which is not
    the env running predicators) and reports a stage failure by name.
    """
    # pylint: disable=import-outside-toplevel,import-error
    try:
        from babyrobot.scene.capture_markerless import MarkerlessCapture, \
            load_boxes, resolve_python, run_stages
    except ImportError as e:
        raise MissingBabyRobotError(_MISSING_BABYROBOT) from e
    # run_stages reads ``boxes``, not ``boxes_json`` -- resolving the file is
    # the caller's job, and doing it here is what makes the rebuild unattended.
    boxes_json = CFG.real_robot_snapshot_boxes_json or None
    boxes = tuple(load_boxes(boxes_json)) if boxes_json else None
    if boxes is None:
        logging.warning(
            "real robot: no real_robot_snapshot_boxes_json, so stage 2 will "
            "open the drag window and wait for a human to draw one box per "
            "domino -- every episode. Point it at an earlier run's boxes.json "
            "to make the rebuild unattended.")
    config = _capture_config(MarkerlessCapture, serial, boxes)
    return run_stages(resolve_python(), svo, bundle, config)


def _capture_config(capture_cls: Any, serial: str, boxes: Any) -> Any:
    """The capture options every snapshot fit uses, whatever camera it is
    for."""
    return capture_cls(
        camera=serial,
        boxes=boxes,
        frames=CFG.real_robot_snapshot_frames,
        resolution=CFG.real_robot_recording_resolution,
        # The twin's base->world transplant and the fit have to agree on where
        # the table is, and they are configured separately -- so the fit is
        # told, rather than left to measure its own.
        table_z=float(CFG.domino_real_table_z),
        z_mode=CFG.real_robot_snapshot_z_mode,
        viz=CFG.real_robot_snapshot_viz)


def _default_multi_runner(svos: Dict[str, str], bundle: str,
                          boxes_json_by_camera: Dict[str, str]) -> str:
    """Fit the take from every camera and fuse, returning the scene JSON.

    The fused scene is written in the same shape a single-camera fit produces,
    so the loader below -- and everything downstream of it -- reads it
    unchanged.

    Prompt boxes are per camera and cannot be shared: a box is pixel
    coordinates on one camera's frame 0, and the other camera is looking at
    the scene from somewhere else entirely, so the same numbers would prompt
    SAM-2 at whatever happens to lie at those pixels in a different view.
    A camera with no boxes configured opens the drag window and waits for a
    human -- every episode -- which is what the warning is about.
    """
    # pylint: disable=import-outside-toplevel,import-error
    try:
        from babyrobot.scene.capture_markerless import MarkerlessCapture, \
            resolve_python, run_stages_multi
    except ImportError as e:
        raise MissingBabyRobotError(_MISSING_BABYROBOT) from e
    missing = [s for s in svos if not boxes_json_by_camera.get(s)]
    if missing:
        logging.warning(
            "real robot: no prompt boxes for ZED %s, so stage 2 will open the "
            "drag window and wait for a human to draw one box per domino on "
            "that camera -- every episode. Set "
            "real_robot_snapshot_boxes_json_by_camera to an earlier run's "
            "boxes.json for each camera to make the rebuild unattended.",
            ", ".join(missing))
    config = _capture_config(MarkerlessCapture, next(iter(svos)), None)
    return run_stages_multi(resolve_python(),
                            svos,
                            bundle,
                            config,
                            boxes_json_by_camera=boxes_json_by_camera)
