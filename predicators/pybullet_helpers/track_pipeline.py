"""Turning an episode's recording into a pose track, off the critical path.

Step 2 records a take per episode; Step 3 scores against a track. This module
is the bridge: it runs the markerless pipeline over each take and writes the
``dominoes_traj.json`` the fit reads.

**Why it runs in the background and not inline.** The pipeline takes minutes --
roughly 3x the length of the take -- so waiting for it inside the episode loop
would serialise post-processing into execution and undo open-loop batching. It
is launched as a detached process when the take closes and joined at the end of
the run, which overlaps it with the next episode's human scene reset and arm
motion. It parallelises across takes at ~2 GB of a 24 GB card, so several
episodes in flight is the intended state rather than a hazard.

**Why a manifest.** A learning cycle produces many takes, and the fit has to
know which track belongs to which episode -- and which episodes had a camera
drop out and should not be trusted at all. The manifest is written as each take
closes, so it is complete and readable even if the run is killed before the
pipeline finishes; entries point at tracks that may not exist yet, and the
reader treats a missing one as "not ready" rather than as an error.

babyrobot is optional and must never be imported at module level.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

from predicators.settings import CFG

# Written by the recorder, read by the fit. One per run.
MANIFEST_NAME = "tracks.json"
# Stage 4's own output name, which is what the fit loads.
TRACK_NAME = "dominoes_traj.json"
# Each background job's stdout and stderr, beside the bundle it produces.
LOG_NAME = "pipeline.log"


class MarkerlessTrackProcessor:
    """Launches the markerless pipeline over finished takes.

    One process per take, started when the take closes and never waited
    on until the run ends. The script is the submodule's own
    ``run_markerless.sh``, which is the same staging a human runs by
    hand -- so a track produced here and one produced at the terminal
    are the same artifact.
    """

    def __init__(self,
                 script: Optional[str] = None,
                 boxes_json: Optional[str] = None,
                 z_mode: str = "contact",
                 max_frames: Optional[int] = None,
                 trim: bool = True,
                 trim_args: str = "",
                 jobs: int = 0,
                 viz: bool = False,
                 launcher: Any = None) -> None:
        self._script = script or _default_script()
        self._boxes_json = boxes_json
        self._z_mode = z_mode
        self._max_frames = max_frames
        self._trim = trim
        self._trim_args = trim_args
        self._jobs_requested = jobs
        self._viz = viz
        # Injectable so tests drive the whole lifecycle without a GPU: takes
        # (argv, env) and returns something with poll()/wait().
        self._launcher = launcher or _popen
        self._jobs: List[Any] = []

    def launch(self, svo: str, bundle: str, serial: str) -> Optional[Any]:
        """Start the pipeline over ``svo``, writing into ``bundle``.

        Returns the handle, or None when the pipeline could not be
        started. A failure to launch is logged and swallowed: the take
        is already on disk and can be processed by hand, so it must not
        take the run down with it.
        """
        if not os.path.exists(self._script):
            logging.error(
                "cannot post-process %s: no markerless driver at %s. The "
                "take is recorded and can be processed by hand.", svo,
                self._script)
            return None
        # ``given`` rather than ``manual``: stage 2 would otherwise open a
        # drag window and wait for a human, in the middle of a learning run.
        argv = [self._script, svo, bundle, "given", "--z-mode", self._z_mode]
        env = dict(os.environ)
        env["SERIAL"] = str(serial)
        if self._jobs_requested:
            # Stage 4 fans out and saturates whatever it is given; the driver's
            # own default is 16, which left half a 32-core box idle for the
            # largest step in the pipeline.
            env["JOBS"] = str(self._jobs_requested)
        if not self._viz:
            # Skip masks_overlay.mp4, which is rendered before stage 4 and so
            # delays the track by its full cost. Ignored by a driver that
            # predates the flag -- it renders the overlay as it always did,
            # rather than failing -- which is what lets this be set before the
            # submodule has it.
            env["TRACK_VIZ"] = "0"
        if self._max_frames:
            env["MAX_FRAMES"] = str(self._max_frames)
        if self._trim:
            # Drop the still lead-in at stage 1, so SAM-2 never sees it. An
            # episode take is bracketed by dead air of its own making: the
            # recording starts at the reset and the twin then simulates every
            # option with the arm parked, which on run_20260817_162250 was
            # 152 s of a static scene out of a 420 s take. Both of the scan's
            # failure modes keep frames rather than lose them, so this is
            # safe to leave on.
            #
            # Ignored by a driver that predates the flag rather than being an
            # error, which is what lets it be set before the submodule has it.
            env["TRIM"] = "1"
            if self._trim_args:
                env["TRIM_ARGS"] = self._trim_args
        boxes = _read_boxes(self._boxes_json)
        if boxes is None:
            logging.error(
                "cannot post-process %s: stage 2 needs initialization boxes "
                "and real_robot_snapshot_boxes_json is unset or unreadable. "
                "Without them the pipeline would stop for a human.", svo)
            return None
        env["BOXES"] = boxes
        os.makedirs(bundle, exist_ok=True)
        # Each job's own log. Without it a failed stage is a missing track and
        # no reason -- the job is detached, so its output has nowhere else to
        # go, and the run only notices minutes later when the fit finds
        # nothing.
        log_path = os.path.join(bundle, LOG_NAME)
        try:
            job = self._launcher(argv, env, log_path)
        except OSError as e:
            logging.error("could not start the markerless pipeline on %s: %s",
                          svo, e)
            return None
        logging.info("post-processing %s -> %s (in the background; log: %s)",
                     svo, os.path.join(bundle, TRACK_NAME), log_path)
        self._jobs.append(job)
        return job

    def pending(self) -> int:
        """How many pipeline jobs are still running."""
        return sum(1 for j in self._jobs if j.poll() is None)

    def wait_all(self, timeout: Optional[float] = None) -> None:
        """Block until every launched job has finished.

        Called at run teardown. A job that fails is logged rather than
        raised: by this point the run is over, and the takes survive
        for a re-run by hand.
        """
        for job in self._jobs:
            try:
                code = job.wait(timeout=timeout)
            except Exception as e:  # pylint: disable=broad-except
                logging.error("waiting on a post-processing job failed: %s", e)
                continue
            finally:
                handle = getattr(job, "predicators_log_handle", None)
                if handle is not None:
                    handle.close()
            if code not in (0, None):
                logging.error(
                    "a markerless post-processing job exited %s; its track "
                    "will be missing and that episode will be skipped. What "
                    "went wrong is in %s", code,
                    getattr(job, "predicators_log_path", "its bundle's log"))

    def set_boxes(self, boxes_json: str) -> None:
        """Use these prompt boxes for every take from now on."""
        self._boxes_json = boxes_json


def pick_boxes(svo: str, bundle: str, serial: str) -> Optional[str]:
    """Draw the prompt boxes once, interactively, and return boxes.json.

    Stages 1 and 2 over a snapshot of the scene as it stands at the
    start of the run. Blocking and deliberately so: it opens a window
    and waits for a human to drag one box per domino.

    Once per RUN rather than once per episode, which is what makes an
    otherwise-interactive pipeline usable in a learning loop. The
    assumption is that the scene the boxes were drawn on is the scene
    every episode starts from -- true for a fixed-plan replay, which
    trains and tests on one arrangement. It is also self-checking: if
    the layout moves far enough that a box no longer sits on its
    domino, stage 3's frame-0 identity check aborts that take rather
    than tracking the wrong thing.
    """
    python = _resolve_python()
    stage1 = [
        python,
        os.path.join(_markerless_dir(), "svo_to_bundle.py"), "--svo", svo,
        "--out", bundle, "--serial",
        str(serial), "--max-frames", "5"
    ]
    stage2 = [
        python,
        os.path.join(_markerless_dir(), "init_boxes.py"), "--bundle", bundle,
        "--source", "manual", "--viz"
    ]
    for argv, label in ((stage1, "stage 1 (snapshot -> bundle)"),
                        (stage2, "stage 2 (DRAG ONE BOX PER DOMINO)")):
        logging.info("boxes: %s", label)
        try:
            completed = subprocess.run(argv, check=False)
        except OSError as e:
            logging.error("could not run %s: %s", label, e)
            return None
        if completed.returncode != 0:
            logging.error("%s exited %d; no boxes were produced", label,
                          completed.returncode)
            return None
    boxes = os.path.join(bundle, "boxes.json")
    if not os.path.exists(boxes):
        logging.error("stage 2 finished but wrote no %s", boxes)
        return None
    logging.info("boxes: drawn once for this run -> %s", boxes)
    return boxes


def _markerless_dir() -> str:
    """Where the markerless stage scripts live."""
    return os.path.dirname(_default_script())


def _resolve_python() -> str:
    """The interpreter the stages need (pyzed and ultralytics together)."""
    try:
        # pylint: disable-next=import-outside-toplevel,import-error
        from babyrobot.scene.capture_markerless import resolve_python
        return str(resolve_python())
    except ImportError:
        return os.environ.get(
            "MARKERLESS_PY",
            os.environ.get(
                "ROBOT_ML_PY",
                os.path.expanduser("~/miniforge3/envs/robot-ml/bin/python")))


def _popen(argv: List[str], env: Dict[str, str], log_path: str) -> Any:
    """Start a detached pipeline process, logging to ``log_path``.

    The handle is attached to the returned process rather than closed
    here: the child writes to it for minutes after this returns, and
    letting it be garbage-collected would close the descriptor out from
    under a running stage.
    """
    # pylint: disable-next=consider-using-with
    handle = open(log_path, "w", encoding="utf-8")
    # pylint: disable-next=consider-using-with
    job = subprocess.Popen(argv,
                           env=env,
                           stdout=handle,
                           stderr=subprocess.STDOUT)
    job.predicators_log_handle = handle  # type: ignore[attr-defined]
    job.predicators_log_path = log_path  # type: ignore[attr-defined]
    return job


def _read_boxes(boxes_json: Optional[str]) -> Optional[str]:
    """The prompt boxes as the JSON string the driver's BOXES expects.

    ``init_boxes.py`` writes records -- ``{"id", "box", "label"}`` under
    a ``boxes`` key -- while the ``BOXES`` env it reads back expects a
    bare ``[[x0, y0, x1, y1], ...]``. Handing the records over unchanged
    makes stage 2 iterate a dict and die on ``int('id')``, minutes into
    a run, after the arm has already executed the episode. So the
    coordinates are unwrapped here, in list order, which is the id order
    the writer enumerates.
    """
    if not boxes_json or not os.path.exists(boxes_json):
        return None
    try:
        with open(boxes_json, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    records = raw.get("boxes") if isinstance(raw, dict) else raw
    if not records:
        return None
    boxes: List[List[int]] = []
    for record in records:
        coords = record.get("box") if isinstance(record, dict) else record
        if not coords or len(list(coords)) != 4:
            logging.error("%s has a box that is not [x0, y0, x1, y1]: %r",
                          boxes_json, record)
            return None
        boxes.append([int(v) for v in coords])
    return json.dumps(boxes)


def _default_script() -> str:
    """Where the markerless driver lives inside the submodule."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(root, "submodules", "BabyRobotPredicator",
                        "pose_estimation", "markerless", "run_markerless.sh")


def make_track_processor() -> MarkerlessTrackProcessor:
    """Build the processor the config asks for."""
    return MarkerlessTrackProcessor(
        boxes_json=CFG.real_robot_snapshot_boxes_json or None,
        z_mode=CFG.real_robot_track_z_mode,
        max_frames=CFG.real_robot_recording_max_frames or None,
        trim=CFG.real_robot_trim_still_frames,
        trim_args=CFG.real_robot_trim_args,
        jobs=CFG.real_robot_track_jobs,
        viz=CFG.real_robot_track_viz)


def write_manifest(path: str, episodes: List[Dict[str, Any]]) -> None:
    """Write the run manifest, replacing whatever was there.

    Rewritten in full after every episode rather than appended to, so a
    run killed mid-way leaves a valid JSON document rather than a
    truncated one.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"episodes": episodes}, f, indent=2)
        f.write("\n")


def read_manifest(path: str) -> List[Dict[str, Any]]:
    """The manifest's episode entries, newest last."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    episodes = raw.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{path} has no episodes list")
    return episodes
