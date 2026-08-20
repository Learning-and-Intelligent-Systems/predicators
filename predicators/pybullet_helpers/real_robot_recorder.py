"""Recording an episode's execution as SVO takes, for offline pose estimation.

The counterpart to ``real_robot_bridge`` for cameras rather than the arm: the
executor says "an episode started" / "an episode ended", and this module turns
that into one SVO take per episode. Nothing here estimates a pose. The take is
post-processed later -- the markerless pipeline runs at roughly 3x real time,
so no perception result can come back inside the episode that produced it, and
execution must not wait for one.

**babyrobot is optional and must never be imported at module level**, exactly
as in ``real_robot_bridge``: it ships as the private git submodule
``submodules/BabyRobotPredicator`` and is absent from ``install_requires``, so
predicators' CI has no ``babyrobot`` on the path.

Why recording cannot share a run with live ZED perception: both open the same
physical cameras, and a ZED admits one owner. ``attach_real_robot`` refuses the
combination rather than letting the second ``open()`` fail partway into a
hardware session.
"""
from __future__ import annotations

import atexit
import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from predicators.settings import CFG

_MISSING_BABYROBOT = (
    "episode recording needs the private BabyRobotPredicator package, which "
    "predicators carries as the git submodule "
    "submodules/BabyRobotPredicator. Check it out and install it:\n"
    "    git submodule update --init submodules/BabyRobotPredicator\n"
    "    pip install -e submodules/BabyRobotPredicator")


class MissingBabyRobotError(ImportError):
    """babyrobot is not importable, so no recorder can be constructed."""


class EpisodeRecorder:
    """One SVO take per episode, from a session that opens the ZEDs once.

    Wraps babyrobot's ``ZedRecorderSession`` so the executor depends on
    four calls rather than on the SDK: cameras are opened once for the
    whole run (a learning cycle is many episodes, and per-episode camera
    init and warmup would otherwise be paid every time) and a take is
    started and stopped around each episode.

    Takes the session as an argument rather than building one, so tests
    drive the whole lifecycle with a stub and no hardware.
    """

    def __init__(self,
                 session: Any,
                 max_frames: Optional[int] = None,
                 export_mp4: bool = False,
                 export_depth: bool = False,
                 processor: Any = None,
                 track_dir: Optional[str] = None,
                 camera: Optional[str] = None) -> None:
        self._session = session
        self._max_frames = max_frames
        # Runs the markerless pipeline over each finished take, in the
        # background. None means takes are recorded and left for a human to
        # process.
        self._processor = processor
        self._track_dir = track_dir or os.path.join("logs", "zed_tracks")
        # Which camera the poses are fitted from. Markerless is
        # single-camera -- the second's cloud is not fused -- and the two are
        # not interchangeable: on measured ground truth one is 6x better on
        # orientation (1.03 deg median against 6.29) while the other tracks
        # 99.9% of frames against 82%. None takes the session's first serial,
        # which is an arbitrary default rather than a considered one.
        self._camera = str(camera) if camera else None
        # One entry per episode, in order, written to the manifest as each
        # take closes.
        self._episodes: List[Dict[str, Any]] = []
        # Whether this run's prompt boxes have been drawn. The call site
        # depends on WHEN the take opens -- at the reset, or in front of a
        # later option -- so ensure_boxes is called from more than one place
        # and has to be idempotent.
        self._boxes_drawn = False
        # Exports are deliberately off during a run. ``stop_take`` can write
        # mp4s and depth inline, but that is the expensive offline work --
        # doing it here would serialise post-processing into the episode loop
        # and undo open-loop execution. ``svo_to_bundle`` does depth later,
        # replaying the .svo.
        self._export_mp4 = export_mp4
        self._export_depth = export_depth
        # Every take this run has produced, newest last: (take_dir, usable).
        # The fit needs to know which episodes have a trustworthy track.
        self.takes: List[Tuple[str, bool]] = []
        # Scene snapshots taken between episodes, newest last. Separate from
        # ``takes``: a snapshot is an input to a task, not a record of an
        # execution, and the fit must not mistake one for the other.
        self.snapshots: List[str] = []
        self._closed = False

    @property
    def serials(self) -> List[str]:
        """The ZED serials this session holds open."""
        return [str(s) for s in getattr(self._session, "serials", [])]

    @property
    def fit_camera(self) -> str:
        """The serial the poses are fitted from.

        Raises when the configured camera is not one the session
        records: a take has no recording for it, so every episode would
        fail at post-processing with a missing file rather than here,
        before the run has cost anything.
        """
        serials = self.serials
        if self._camera is None:
            return serials[0] if serials else ""
        if serials and self._camera not in serials:
            raise ValueError(
                f"real_robot_track_camera {self._camera!r} is not one of the "
                f"cameras this session records {serials}; the poses are "
                "fitted from a take that session wrote.")
        return self._camera

    @property
    def last_take_dir(self) -> Optional[str]:
        """Where the most recent episode was recorded, or None."""
        return self.takes[-1][0] if self.takes else None

    def open(self) -> None:
        """Open the cameras, once for the whole run."""
        self._session.open()

    def start_episode(self, stamp: Optional[str] = None) -> Optional[str]:
        """Begin this episode's take; return its directory.

        Failures raise. Recording is the point of a run configured this
        way, so an episode that cannot record is an episode whose
        hardware time and human scene reset would be spent for nothing
        -- better to say so before the arm moves than to discover it
        when the track is missing.
        """
        take_dir = self._session.start_take(stamp=stamp,
                                            max_frames=self._max_frames)
        logging.info("real robot: recording episode to %s", take_dir)
        return take_dir

    def stop_episode(self) -> Optional[Dict[str, Any]]:
        """End this episode's take; return its ``meta.json`` contents.

        Never raises. By the time this runs the arm has already moved,
        so a recording problem must not also destroy the run around it;
        it is logged and the take is marked unusable instead. A camera
        that dropped out mid-episode yields a short track that looks
        perfectly well formed, which is the failure worth being loud
        about.
        """
        if self._closed:
            return None
        try:
            meta = self._session.stop_take(export_mp4=self._export_mp4,
                                           export_depth=self._export_depth)
        except Exception as e:  # pylint: disable=broad-except
            # stop_take re-raises whatever a grab thread died of.
            logging.error(
                "real robot: recording failed for this episode (%s); its "
                "track is unusable", e)
            self.takes.append(("<failed>", False))
            return None
        if meta is None:  # nothing was recording
            return None
        take_dir = str(meta.get("take_dir", "<unknown>"))
        errors = list(meta.get("errors") or [])
        usable = not errors
        if not usable:
            logging.error(
                "real robot: take %s reported %d camera error(s): %s; its "
                "track is unusable", take_dir, len(errors),
                "; ".join(str(e) for e in errors))
        else:
            logging.info("real robot: take %s complete (clock=%s, sdk=%s)",
                         take_dir, meta.get("timestamp_clock"),
                         meta.get("sdk_version"))
        self.takes.append((take_dir, usable))
        self._post_process(take_dir, meta, usable)
        return meta

    def _post_process(self, take_dir: str, meta: Dict[str, Any],
                      usable: bool) -> None:
        """Record this episode in the manifest and start its pipeline.

        An unusable take is still recorded, marked so, and NOT
        processed: a track fitted to a recording that lost a camera
        mid-episode would be a well-formed track of the wrong thing,
        which is worse than none.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.pybullet_helpers.track_pipeline import \
            MANIFEST_NAME, TRACK_NAME, write_manifest
        serial = self.fit_camera
        ext = str(meta.get("svo_ext", ".svo2"))
        svo = os.path.join(take_dir, f"zed_{serial}{ext}")
        bundle = os.path.join(self._track_dir, os.path.basename(take_dir))
        entry: Dict[str, Any] = {
            "episode": len(self._episodes) + 1,
            "take_dir": take_dir,
            "svo": svo,
            "bundle": bundle,
            "track": os.path.join(bundle, TRACK_NAME),
            "usable": usable,
        }
        self._episodes.append(entry)
        if usable and self._processor is not None:
            started = self._processor.launch(svo, bundle, serial)
            entry["processing"] = started is not None
        try:
            write_manifest(os.path.join(self._track_dir, MANIFEST_NAME),
                           self._episodes)
        except OSError as e:
            logging.error("could not write the track manifest: %s", e)

    def ensure_boxes(self, picker: Any = None) -> Optional[str]:
        """Draw this run's prompt boxes now, once, if none were given.

        **Called at the moment the take will open, not at run start.**
        The boxes are SAM-2 prompts applied to the take's first frame,
        so they have to describe the arrangement that frame shows. While
        the take began at the reset those were the same thing and the
        call sat at run start; recording from a later option makes them
        different, and run-start boxes then point at where two dominoes
        USED to be. That is not a failure the pipeline can detect --
        it fits masks to whatever is inside the box it was given.

        Drawn once per run, not once per take: a fixed-plan replay
        arranges the same row every episode, so the boxes drawn at the
        first take's boundary are right for every later one.

        Returns the boxes file, or None if there was nothing to do or
        it failed. A failure is not fatal here: it is reported, and the
        takes are still recorded for processing by hand.
        """
        if self._processor is None or self._boxes_drawn:
            return None
        self._boxes_drawn = True
        # pylint: disable-next=import-outside-toplevel
        from predicators.pybullet_helpers.track_pipeline import pick_boxes
        picker = picker or pick_boxes
        take_dir, meta = self.snapshot(frames=5)
        serial = self.fit_camera
        ext = str((meta or {}).get("svo_ext", ".svo2"))
        svo = os.path.join(take_dir, f"zed_{serial}{ext}")
        bundle = os.path.join(self._track_dir, "boxes")
        boxes = picker(svo, bundle, serial)
        if not boxes:
            logging.error(
                "no prompt boxes for this run, so takes will be recorded but "
                "not post-processed; draw them by hand and set "
                "real_robot_snapshot_boxes_json")
            return None
        self._processor.set_boxes(boxes)
        return str(boxes)

    def snapshot(self,
                 frames: int = 5) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Record a few frames of the scene as it stands; return (dir, meta).

        A second, short take on the *already-open* session, which is the
        whole point: a ZED admits one owner, so a scene look that opened
        its own cameras would collide with the episode recording. Taken
        between episodes, while no episode take is running.

        Kept out of ``takes``: those are the episode tracks the fit
        consumes, and a snapshot is an input to a task rather than a
        record of an execution.
        """
        # The counter, not just the clock: start_take makes the directory with
        # exist_ok, so two snapshots in the same second would silently write
        # into one directory and the second would inherit the first's frames.
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stamp = f"{now}_snap{len(self.snapshots) + 1:03d}"
        take_dir = self._session.start_take(stamp=stamp, max_frames=frames)
        meta = self._session.stop_take(export_mp4=False, export_depth=False)
        self.snapshots.append(str(take_dir))
        logging.info("real robot: scene snapshot recorded to %s", take_dir)
        return str(take_dir), meta

    def close(self) -> None:
        """Release the cameras, stopping an in-flight take first.

        Idempotent, and safe to call from ``atexit``: a session left
        recording would keep writing until the disk filled.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._session.close()
        except Exception as e:  # pylint: disable=broad-except
            logging.error("real robot: closing the recorder failed: %s", e)
        if self._processor is not None:
            pending = self._processor.pending()
            if pending:
                logging.info(
                    "waiting for %d markerless post-processing job(s); each "
                    "runs about 3x the length of its take", pending)
            self._processor.wait_all()


def make_episode_recorder(
        serials: Optional[Sequence[str]] = None) -> EpisodeRecorder:
    """Build the recorder named by the config, importing babyrobot lazily.

    Registered with ``atexit`` because predicators has no run-teardown
    hook to hang it on: ``attach_real_robot`` is called once from
    ``main`` and nothing calls back when the run ends.
    """
    # pylint: disable=import-outside-toplevel,import-error
    try:
        from pose_estimation.record_zed_video import DEFAULT_SERIALS, \
            ZedRecorderSession
    except ImportError as e:
        raise MissingBabyRobotError(_MISSING_BABYROBOT) from e
    out_dir = CFG.real_robot_recording_dir or os.path.join("logs", "zed_takes")
    session = ZedRecorderSession(
        serials=list(serials if serials is not None else DEFAULT_SERIALS),
        resolution=CFG.real_robot_recording_resolution,
        fps=CFG.real_robot_recording_fps,
        out_dir=out_dir)
    max_frames = CFG.real_robot_recording_max_frames or None
    processor = None
    if CFG.real_robot_process_takes:
        # pylint: disable-next=import-outside-toplevel
        from predicators.pybullet_helpers.track_pipeline import \
            make_track_processor
        processor = make_track_processor()
    recorder = EpisodeRecorder(session,
                               max_frames=max_frames,
                               processor=processor,
                               track_dir=CFG.real_robot_track_dir or None,
                               camera=CFG.real_robot_track_camera or None)
    atexit.register(recorder.close)
    return recorder


def episode_stamp(train_or_test: str, task_idx: int, episode_num: int) -> str:
    """Name a take so the episodes it came from are recoverable.

    Sorts chronologically (the timestamp leads) and still says which
    task and which episode of the run produced it, because a learning
    cycle revisits the same task index many times.
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{now}_{train_or_test}{task_idx}_ep{episode_num:03d}"
