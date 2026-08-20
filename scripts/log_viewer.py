#!/usr/bin/env python3
"""Local web viewer for agent experiment logs under logs/.

Stdlib-only TensorBoard-style browser for run directories of the form
logs/<family>/<env>-<approach>/seed<N>/run_<timestamp>/. Features:

  * runs overview with per-episode pass/fail chips, costs, and run comparison
  * episode markdown transcripts with collapsible turns, inline images, and
    the run's saved episode video from videos/<same run subdir>/
  * unified diffs between simulator_versions / predicates_versions files
  * image galleries, ANSI-colored info.log / debug.log, and a full file tree
    so every logged file is inspectable
  * per-run kill buttons for live runs and per-run deletion of the log dir
    (and its mirrored videos dir), targeted at that run's PIDs only - or,
    on a Slurm cluster, at its own job id
  * cluster-aware liveness: served from a login node, where a run's process
    is out of reach of ps/lsof, squeue says which runs are still going, each
    job pinned to its exact run dir via the "Logging to ..." line in its
    stdout file - so parallel launches of one config all show live

Format contracts this viewer relies on:
  * episode filenames {query:03d}_{kind}[_task{K}]_{YYYYMMDD_HHMMSS}.md from
    session_log_filename() in predicators/agent_sdk/tools.py
  * markdown layout (## sections, ### Turn N, trailing "**Result:** ..." line)
    from format_conversation_markdown() in agent_sdk/log_formatter.py
  * "Captured as the current answer"/"NOT CAPTURED" capture verdicts and
    "Goal achieved: True|False" rollout lines inside tool-result blocks,
    from evaluate_option_plan in agent_sdk/tools/testing.py
  * "Test results: defaultdict(..., {...})" lines in info.log

Format contracts, continued:
  * videos/<approach>/<experiment_id>/seed<N>/run_<timestamp>/ mirrors the log
    dir, via the run subdir utils.configure_logging shares with save_video;
    for a log dir later moved or renamed by hand, the "Wrote out to" lines
    in its info.log recover the actual video dir (see _video_base)
  * video names ...__task<K+1>[_failure]__cycle<C>.mp4 from main.py _save_video
  * interaction video names ...__ep<i>__cycle<C>.mp4 (no __task part) from
    main.py _generate_interaction_results under CFG.make_interaction_videos;
    one per interaction episode, <i> its 0-based index within the cycle
    (older runs wrote one ...__cycle<C>.mp4 concatenating the whole cycle)

Usage:
    python scripts/log_viewer.py [--logs logs] [--videos videos] [--port 8765]
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import getpass
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

LOGS_ROOT = ""  # absolute, set in main()
VIDEOS_ROOT = ""  # absolute, set in main(); may not exist

EPISODE_RE = re.compile(r"^(\d{3})_([a-z]+)(?:_task(\d+))?_(\d{8}_\d{6})\.md$")
# Tail of a test video written by main.py _save_video, whose task number is
# 1-based (task_idx+1) unlike the 0-based task in an episode filename.
VIDEO_RE = re.compile(r"__task(\d+)(_failure)?__cycle([^.]*)\.mp4$")
# Tail of an interaction video from _generate_interaction_results (under
# CFG.make_interaction_videos): one per interaction episode, no __task
# part. Test videos also end in __cycle<C>.mp4, so check VIDEO_RE first.
INTERACTION_VIDEO_RE = re.compile(r"__ep(\d+)__cycle([^.]*)\.mp4$")
# Legacy tail from before per-episode saving: one video per learning
# cycle concatenating all of its explore episodes.
INTERACTION_CYCLE_VIDEO_RE = re.compile(r"__cycle([^.]*)\.mp4$")
RESULT_RE = re.compile(
    r"\*\*Result:\*\* (\d+) turns, \$([\d.]+) this solve, \$([\d.]+) total")
GOAL_RE = re.compile(r"Goal achieved: (True|False)")
# Session-level capture verdicts from evaluate_option_plan (agent_sdk/
# tools/testing.py). A "Goal achieved" line only says one sim rollout
# reached the goal atoms; the capture verdict is what decides whether the
# session actually produced an answer. A best-effort capture (budget
# exhausted) is not an agent-side solve.
CAPTURED_RE = re.compile(r"Captured as the current answer( \(best-effort\b)?")
NOT_CAPTURED_RE = re.compile(r"NOT CAPTURED|\(plan NOT captured\)")
NUM_SOLVED_RE = re.compile(r"'num_solved': ([\d.]+)")
NUM_TOTAL_RE = re.compile(r"'num_total': ([\d.]+)")
AVG_TEST_REWARD_RE = re.compile(r"'avg_test_reward': (-?[\d.]+)")
# One task's entry in main.py's "Per-task rewards: task0=0.95, ..." line,
# logged right after the "Tasks solved" line that closes a test round.
PER_TASK_REWARD_RE = re.compile(r"task(\d+)=(-?[\d.]+)")
# Env-side verdict lines from predicators/main.py _run_testing, e.g.
# "INFO: Task 2 / 5: Policy failed to reach goal" or
# "INFO: [main.py] Task 1 / 5: approach failed with error: ..."
TASK_VERDICT_RE = re.compile(
    r"^INFO: (?:\[main\.py\] )?Task (\d+) / (\d+): (.+)$")
TASKS_SOLVED_RE = re.compile(r"Tasks solved: (\d+) / (\d+)")
# Env evaluator verdict for one explore (train) interaction episode, from
# main.py _generate_interaction_results, plus its optional follow-up lines
# naming a rejection reason or the below-early-stop-bar caveat.
INTERACTION_RE = re.compile(
    r"^INFO: Interaction episode on train task \d+: reward=(-?[\d.]+), "
    r"terminated=(True|False), accepted=(True|False)")
INTERACTION_REJECT_RE = re.compile(
    r"^INFO: Interaction episode on train task \d+ REJECTED by the "
    r"env: (.+)$")
INTERACTION_BAR_RE = re.compile(
    r"^INFO: Interaction episode on train task \d+ solved but (.+): "
    r"does NOT count as solved for early stopping\.$")
# The transcript save line each agent session leaves in info.log; it pairs
# explore sessions with the interaction episodes that execute their
# requests (see _parse_info_log).
SAVED_EP_RE = re.compile(r"Saved local sandbox query/response to .*[/\\]"
                         r"(\d{3})_([a-z]+)(?:_task\d+)?_\d{8}_\d{6}\.md")
# utils.save_video's announcement of a written video file. Its
# <approach>/<experiment_id>/seed<N>/run_<timestamp> tail names the video
# dir the run actually wrote, which the mirrored-layout assumption gets
# wrong whenever a run's log dir was later moved or renamed by hand.
VIDEO_SAVED_RE = re.compile(r"Wrote out to "
                            r"\S*?([^\s/]+/[^\s/]+/seed\d+/run_\d{8}_\d{6})/")
# main.py logs this only after the pipeline returns, so it is the one
# trustworthy "this run completed" marker; a crash or a kill leaves none.
DONE_RE = re.compile(r"^Main script terminated in")
# A live run is a main.py process whose flags name the run's log dir,
# which utils.configure_logging builds as approach/experiment_id/seed<N>.
PS_ARG_RE = re.compile(r"--(approach|experiment_id|seed)[= ]+(\S+)")
ANSI_RE = re.compile(r"\x1b\[([\d;]*)m")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi"}
TEXT_EXTS = {
    ".txt", ".log", ".json", ".yaml", ".yml", ".csv", ".pddl", ".cfg", ".ini",
    ".sh", ".tex"
}
CODE_EXTS = {".py"}
# Episode-grid geometry, in px. A task always gets TASK_W of space no
# matter how many times it was retried, so its chips land at the same x in
# every run and experiment; retries stack inside that space. MISC_W holds
# the round's explore/learn chips, which are unbounded and so wrap too.
TASK_W = 108
ROUND_W = 34
MISC_W = 3 * TASK_W
# Fixed widths for the remaining columns of the index runs table, in the
# order they are declared there. None is the episodes column, whose width
# depends on the task count and is filled in at render time.
RUN_COL_W = (30, 200, 62, 96, None, 178, 68, 132, 132, 72)
# Episodes of one run, in file order; see _parse_episode for the fields.
EpList = List[Dict[str, Any]]
# A run's videos as {(task, cycle tag): [(filename, is_failure)]}.
VidMap = Dict[Tuple[int, str], List[Tuple[str, bool]]]
# Bare image paths inside code blocks / inline code, e.g. "Saved images:".
PATH_IN_TEXT_RE = re.compile(r"(?:/|\./)?[\w][\w./\-]*\.(?:png|jpe?g|gif)")

# abs path -> ((mtime, size), parsed value)
_parse_cache: Dict[str, Tuple[Tuple[float, int], Any]] = {}


def esc(text: str) -> str:
    """HTML-escape &, <, >, and double quotes in text."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;").replace('"', "&quot;"))


def q(text: str) -> str:
    """URL-encode text for a query string, escaping every character."""
    return urllib.parse.quote(text, safe="")


def _join_under(root: str, rel: str) -> Optional[str]:
    """Resolve rel against root; return abs path or None if escaping."""
    if not root or not rel:
        return None
    if rel.startswith(("/", "\\")) or ".." in rel.split("/"):
        return None
    path = os.path.realpath(os.path.join(root, rel))
    root = os.path.realpath(root)
    if path != root and not path.startswith(root + os.sep):
        return None
    return path


def safe_join(rel: str) -> Optional[str]:
    """Resolve rel against LOGS_ROOT; return abs path or None if escaping."""
    return _join_under(LOGS_ROOT, rel)


def safe_join_video(rel: str) -> Optional[str]:
    """Resolve rel against VIDEOS_ROOT; return abs path or None if escaping."""
    return _join_under(VIDEOS_ROOT, rel)


def _cached(path: str, fn: Callable[[str], Any]) -> Any:
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime, st.st_size)
    hit = _parse_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    value = fn(path)
    _parse_cache[path] = (key, value)
    return value


def read_text(path: str, max_bytes: Optional[int] = None) -> Tuple[str, bool]:
    """Read a UTF-8 file, keeping only its last max_bytes if given.

    Returns (text, truncated).
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if max_bytes and size > max_bytes:
                f.seek(size - max_bytes)
                data = f.read()
                return data.decode("utf-8", "replace"), True
            return f.read().decode("utf-8", "replace"), False
    except OSError as e:
        return f"(error reading file: {e})", False


# ---------------------------------------------------------------- scanning

_RUN_NAME_TS_RE = re.compile(r"^run_(\d{8})_(\d{6})$")


def _run_start_ts(name: str, mtime: float) -> float:
    """Start time from the run dir name; dir mtime as a fallback."""
    m = _RUN_NAME_TS_RE.match(name)
    if not m:
        return mtime
    try:
        return datetime.datetime.strptime(
            m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return mtime


def _fmt_duration(seconds: float) -> str:
    """Compact duration like "2h 05m", "43m", or "<1m"."""
    minutes = int(seconds // 60)
    if minutes < 1:
        return "<1m"
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02}m"
    return f"{minutes}m"


def _run_activity_ts(run_abs: str, dir_mtime: float) -> float:
    """Last-activity time: newest mtime of the run dir's direct children.

    A directory's mtime only moves when a direct entry is created,
    removed, or renamed - appends never move it, at any depth. Every run
    appends info.log and debug.log at the run root for as long as it is
    alive, so a shallow scan tracks activity without walking the tree.
    Subdirectories are skipped: tooling that pokes into a dead run (e.g.
    its sandbox) moves their mtimes long after the run ended.
    """
    latest = dir_mtime
    try:
        with os.scandir(run_abs) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        continue
                    latest = max(latest, entry.stat().st_mtime)
                except OSError:
                    continue
    except OSError:
        pass
    return latest


def find_runs() -> List[Dict[str, Any]]:
    """Return run dicts sorted by experiment, then newest first."""
    runs: List[Dict[str, Any]] = []
    for dirpath, dirnames, _ in os.walk(LOGS_ROOT):
        rel = os.path.relpath(dirpath, LOGS_ROOT)
        if rel.count(os.sep) > 4:
            dirnames[:] = []
            continue
        for d in list(dirnames):
            if d.startswith("run_"):
                run_rel = os.path.normpath(os.path.join(rel, d)).replace(
                    os.sep, "/")
                parts = run_rel.split("/")
                seed = next((p for p in parts if p.startswith("seed")), "")
                exp = "/".join(p for p in parts[:-1]
                               if not p.startswith("seed")) or "(root)"
                run_abs = os.path.join(dirpath, d)
                dir_mtime = os.path.getmtime(run_abs)
                runs.append({
                    "rel": run_rel,
                    "exp": exp,
                    "seed": seed,
                    "name": d,
                    "mtime": dir_mtime,
                    "activity": _run_activity_ts(run_abs, dir_mtime),
                })
                dirnames.remove(d)  # do not descend into run dirs
    runs.sort(key=lambda r:
              (r["exp"], r["seed"], -_run_start_ts(r["name"], r["mtime"])))
    return runs


# (exp, seed, run name) triples pinned to a live process, plus (exp, seed)
# keys of live processes that could not be pinned to a specific run.
LiveProcs = Tuple[Set[Tuple[str, str, str]], Set[Tuple[str, str]]]

_LSOF_RUN_RE = re.compile(r"(?:/(seed\d+))?/(run_\d{8}_\d{6})(?:/|$)")


def _open_run_names(pids: List[str]) -> Dict[str, Set[Tuple[str, str]]]:
    """(seed, run name) pairs in each pid's open file paths, per lsof.

    The seed is "" when the open path has no seed<N> component right
    above the run dir.
    """
    try:
        out = subprocess.run(["lsof", "-n", "-P", "-Fn", "-p", ",".join(pids)],
                             capture_output=True,
                             text=True,
                             check=False,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    names: Dict[str, Set[Tuple[str, str]]] = {}
    pid = ""
    for line in out.splitlines():
        if line.startswith("p"):
            pid = line[1:]
        elif line.startswith("n") and pid:
            m = _LSOF_RUN_RE.search(line[1:])
            if m:
                names.setdefault(pid, set()).add((m.group(1)
                                                  or "", m.group(2)))
    return names


def _live_proc_matches() -> List[Tuple[str, str, str, Set[str]]]:
    """One (pid, exp, seed, pinned run names) per live main.py process.

    The run names are those lsof pins the process to; the set is empty
    when lsof could not resolve any, in which case the process belongs
    to the newest run of its experiment and seed (see live_runs).
    """
    try:
        out = subprocess.run(["ps", "-Axo", "pid=,args="],
                             capture_output=True,
                             text=True,
                             check=False,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []  # no ps: fall back to marker-only statuses
    procs: Dict[str, Tuple[str, str]] = {}
    for line in out.splitlines():
        if "main.py" not in line:
            continue
        pid, _, args = line.strip().partition(" ")
        flags = dict(PS_ARG_RE.findall(args))
        if pid.isdigit() and "approach" in flags and "experiment_id" in flags:
            procs[pid] = (f"{flags['approach']}/{flags['experiment_id']}",
                          f"seed{flags.get('seed', '0')}")
    open_names = _open_run_names(list(procs)) if procs else {}
    matches: List[Tuple[str, str, str, Set[str]]] = []
    for pid, (exp, seed) in procs.items():
        # Parallel launches stamp sibling seeds with the same run name,
        # so a pin only counts when the open path agrees on the seed.
        matched = {
            name
            for path_seed, name in open_names.get(pid, set())
            if path_seed in ("", seed)
        }
        matches.append((pid, exp, seed, matched))
    return matches


# Slurm compact states in which a job's main.py is still alive on its
# compute node; every other state (PD, CD, F, TO, ...) has either not
# started a process yet or has none left.
_SLURM_LIVE_STATES = "R,S,ST,CG"
# squeue -O fields and column widths, each wide enough that Slurm never
# truncates a value: the id scancel takes, the array task id (the run's
# seed), the per-task numeric id (what %j expands to in a stdout path),
# the job name (its experiment_id), and the stdout path.
_SQUEUE_FIELDS = (("JobArrayID", 32), ("ArrayTaskID", 24), ("JobID", 24),
                  ("Name", 512), ("stdout", 1024))


def _squeue_rows(
        fields: Tuple[Tuple[str, int], ...]) -> Optional[List[List[str]]]:
    """This user's live Slurm jobs, one row of stripped fields per job.

    None means squeue is missing or rejected the format - which is also
    how a machine with no Slurm at all looks - as opposed to [], which
    means Slurm is there and the user has nothing running.
    """
    fmt = ",".join(f"{name}:{width}" for name, width in fields)
    try:
        proc = subprocess.run([
            "squeue", "-u",
            getpass.getuser(), "-h", "-t", _SLURM_LIVE_STATES, "-O", fmt
        ],
                              capture_output=True,
                              text=True,
                              check=False,
                              timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    rows: List[List[str]] = []
    for line in proc.stdout.splitlines():
        cells: List[str] = []
        pos = 0
        for _, width in fields:
            cells.append(line[pos:pos + width].strip())
            pos += width
        rows.append(cells)
    return rows


def _slurm_live_jobs() -> List[Tuple[str, str, str, str, str]]:
    """(job id, approach, experiment_id, seed, stdout) per live Slurm job.

    The submit scripts under scripts/<cluster>/ name each job after its
    experiment_id and give it one array task per seed, and Slurm expands
    the job's stdout path from the same config - <env>__<approach>__...,
    whose second field is the approach (see cluster_utils
    config_to_logfile and utils.get_config_path_str). The approach comes
    back "" when that path is absent or shaped otherwise. The stdout
    path has its %-specifiers expanded (see _expand_stdout_path); it is
    "" when unknown.
    """
    if shutil.which("squeue") is None:
        return []  # not a cluster: spare every page render two spawns
    rows = _squeue_rows(_SQUEUE_FIELDS)
    if rows is None:  # older Slurm may not know the stdout field
        older = _squeue_rows(_SQUEUE_FIELDS[:-1])
        rows = [r + [""] for r in older] if older is not None else []
    jobs: List[Tuple[str, str, str, str, str]] = []
    for job_id, task_id, plain_id, name, stdout in rows:
        stdout = _expand_stdout_path(stdout, job_id, task_id, plain_id, name)
        parts = os.path.basename(stdout).split("__")
        approach = parts[1] if len(parts) > 2 else ""
        # An array job carries the seed as its task id; a single-seed job
        # has it spelled out in the stdout path instead, right after the
        # approach (get_config_path_str) or after the experiment_id.
        seed = next((f for f in [task_id] + parts[2:4] if f.isdigit()), "")
        if name and seed:
            jobs.append((job_id, approach, name, f"seed{seed}", stdout))
    return jobs


def _expand_stdout_path(stdout: str, job_id: str, task_id: str, plain_id: str,
                        name: str) -> str:
    """Resolve a squeue stdout path to an absolute expanded filename.

    Depending on the Slurm version, the stdout field comes back either
    expanded or still holding the -o pattern's %-specifiers, so the
    common ones are substituted here; a path with any other specifier
    left is dropped rather than half-resolved. A relative path is
    anchored at the submission-time working directory, which for the
    submit scripts of this repo is the repo root: the parent of
    LOGS_ROOT.
    """
    if "%" in stdout:
        array_master = job_id.partition("_")[0]
        for spec, value in (("%%", "%"), ("%A", array_master), ("%a", task_id),
                            ("%j", plain_id or job_id), ("%x", name),
                            ("%u", getpass.getuser())):
            stdout = stdout.replace(spec, value)
        if "%" in stdout:
            return ""
    if stdout and not os.path.isabs(stdout):
        stdout = os.path.join(os.path.dirname(LOGS_ROOT), stdout)
    return stdout


# The run dir a main.py process announces in its first log lines (see
# utils.log_initial_info); the job's stdout file captures it, ANSI color
# codes and all, which pins the job to one run_<timestamp> exactly.
_LOGGING_TO_RE = re.compile(
    r"Logging to \S*?([^\s/]+/[^\s/]+/seed\d+/run_\d{8}_\d{6})/")
# Positive pins per stdout path. The announcement is in the first lines
# of the file and never changes, so a hit is cacheable forever; a miss
# is retried every render, since the file may simply not be there yet.
_stdout_pin_cache: Dict[str, str] = {}


def _job_run_rel(stdout: str) -> str:
    """The run (rel to LOGS_ROOT) a Slurm job's stdout file announces.

    Returns "" when the file is unreadable (e.g. the viewer host does
    not share the cluster filesystem) or does not name a run, in which
    case the job stays attributed to the newest run of its experiment
    and seed.
    """
    if not stdout:
        return ""
    hit = _stdout_pin_cache.get(stdout)
    if hit is not None:
        return hit
    try:
        with open(stdout, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(65536)
    except OSError:
        return ""
    m = _LOGGING_TO_RE.search(head)
    if not m:
        return ""
    _stdout_pin_cache[stdout] = m.group(1)
    return m.group(1)


def _slurm_owned(
    runs: List[Dict[str, Any]]
) -> Tuple[Dict[Tuple[str, str, str], Set[str]], Dict[Tuple[str, str],
                                                      Set[str]]]:
    """Runs owned by live Slurm jobs: exact pins, then loose fallbacks.

    Each live job lands in exactly one of the two mappings. A job whose
    stdout file announces a run dir that exists on disk is pinned to it:
    (exp, seed, run name) -> job ids. The rest fall back to
    (exp, seed) -> job ids, attributed to the newest run of that key: a
    job names its run by experiment_id and seed, and the approach - the
    rest of the exp key - is only as good as the job's stdout path, so a
    job whose approach matches no run on disk widens to every experiment
    of that name and seed, which is nearly always just one.
    """
    by_exp_id: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    on_disk: Set[Tuple[str, str, str]] = set()
    for r in runs:
        key = (r["exp"], r["seed"])
        by_exp_id.setdefault((r["exp"].rpartition("/")[2], r["seed"]),
                             set()).add(key)
        on_disk.add((r["exp"], r["seed"], r["name"]))
    pinned: Dict[Tuple[str, str, str], Set[str]] = {}
    loose: Dict[Tuple[str, str], Set[str]] = {}
    for job_id, approach, exp_id, seed, stdout in _slurm_live_jobs():
        run_rel = _job_run_rel(stdout)
        if run_rel:
            run_key = _run_key(run_rel)
            if run_key in on_disk:
                pinned.setdefault(run_key, set()).add(job_id)
                continue
        candidates = by_exp_id.get((exp_id, seed), set())
        exact = (f"{approach}/{exp_id}", seed)
        for key in ({exact} if exact in candidates else candidates):
            loose.setdefault(key, set()).add(job_id)
    return pinned, loose


def live_runs(runs: List[Dict[str, Any]]) -> LiveProcs:
    """Live main.py processes, pinned to the runs they log into.

    A process names its log dir through --approach/--experiment_id/--seed
    but not the run_<timestamp> leaf, so ps alone identifies the
    experiment and seed only - ambiguous when parallel launches of the
    same config overlap. Each process keeps its run's log files open,
    though, so lsof recovers the leaf. Processes lsof cannot resolve are
    returned as bare keys, which run_status attributes to the key's
    newest run: the one such a process made when it started.

    Slurm jobs are the second source. On a cluster the viewer serves a
    login node while the runs execute on compute nodes, where ps and
    lsof see nothing at all, so squeue answers instead. A job's stdout
    file names the exact run dir its process logs into, so most jobs
    arrive pinned; one whose stdout cannot be read stays a bare key
    (see _slurm_owned).
    """
    pinned: Set[Tuple[str, str, str]] = set()
    unpinned: Set[Tuple[str, str]] = set()
    for _, exp, seed, names in _live_proc_matches():
        if names:
            pinned.update((exp, seed, name) for name in names)
        else:
            unpinned.add((exp, seed))
    slurm_pinned, slurm_loose = _slurm_owned(runs)
    pinned.update(slurm_pinned)
    unpinned.update(slurm_loose)
    return pinned, unpinned


# -------------------------------------------------------- kill and delete


def _run_key(run_rel: str) -> Tuple[str, str, str]:
    """(exp, seed, run name) of a run path relative to LOGS_ROOT."""
    parts = run_rel.split("/")
    seed = next((p for p in parts if p.startswith("seed")), "")
    exp = "/".join(p
                   for p in parts[:-1] if not p.startswith("seed")) or "(root)"
    return exp, seed, parts[-1]


def _is_newest_run(exp: str, seed: str, name: str,
                   runs: List[Dict[str, Any]]) -> bool:
    """Whether name is the newest run of its experiment and seed."""
    # find_runs sorts newest first within each (exp, seed).
    return name == next(
        (r["name"] for r in runs if (r["exp"], r["seed"]) == (exp, seed)), "")


def pids_for_run(run_rel: str) -> List[str]:
    """PIDs of the live main.py processes attributed to this run.

    Attribution mirrors run_status on the index page: a pinned process
    must name the run among its open files, and a process lsof could
    not pin owns the newest run of its experiment and seed. Only these
    PIDs are ever signalled - never a name-based sweep, since parallel
    sessions share the machine.
    """
    exp, seed, name = _run_key(run_rel)
    matches = [m for m in _live_proc_matches() if (m[1], m[2]) == (exp, seed)]
    if not matches:
        return []
    newest = _is_newest_run(exp, seed, name, find_runs())
    return [
        pid for pid, _, _, names in matches
        if name in names or (not names and newest)
    ]


def slurm_jobs_for_run(run_rel: str) -> List[str]:
    """Ids of the live Slurm jobs attributed to this run.

    Attribution mirrors run_status: a job whose stdout file announces
    this exact run dir owns it outright; a job that could not be pinned
    that way owns the newest run of its experiment and seed. Only these
    ids are ever cancelled, so a pinned job of an older parallel launch
    is reachable and a kill on the newest run spares its siblings.
    """
    exp, seed, name = _run_key(run_rel)
    runs = find_runs()
    pinned, loose = _slurm_owned(runs)
    jobs = set(pinned.get((exp, seed, name), set()))
    if _is_newest_run(exp, seed, name, runs):
        jobs.update(loose.get((exp, seed), set()))
    return sorted(jobs)


def run_is_live(run_rel: str) -> bool:
    """Whether a local process or a Slurm job is still running this run."""
    return bool(pids_for_run(run_rel) or slurm_jobs_for_run(run_rel))


def _descendant_pids(pids: List[str]) -> List[str]:
    """Transitive child PIDs of pids, from a single ps snapshot."""
    try:
        out = subprocess.run(["ps", "-Axo", "pid=,ppid="],
                             capture_output=True,
                             text=True,
                             check=False,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children: Dict[str, List[str]] = {}
    for line in out.splitlines():
        pid, _, ppid = line.strip().partition(" ")
        children.setdefault(ppid.strip(), []).append(pid)
    seen = set(pids)
    stack = list(pids)
    found: List[str] = []
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                found.append(child)
                stack.append(child)
    return found


def kill_run(run_rel: str, sig: int = signal.SIGTERM) -> Tuple[bool, str]:
    """Signal every process attributed to this run, children included.

    Workers and helper subprocesses would survive a signal to the
    main.py process alone, so its whole descendant tree is signalled. A
    run owned by a Slurm job has no process on this machine, so its job
    is cancelled instead; scancel handles the escalation to KILL on the
    compute node itself, so sig does not apply to that path.
    """
    pids = pids_for_run(run_rel)
    if pids:
        targets = pids + _descendant_pids(pids)
        for pid in targets:
            try:
                os.kill(int(pid), sig)
            except (OSError, ValueError):
                pass  # already exited, or a stale ps line
        return True, (f"sent signal {int(sig)} to {len(targets)} "
                      f"process(es): {', '.join(targets)}")
    jobs = slurm_jobs_for_run(run_rel)
    if not jobs:
        return False, "no live process or Slurm job found for this run"
    try:
        proc = subprocess.run(["scancel"] + jobs,
                              capture_output=True,
                              text=True,
                              check=False,
                              timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"scancel failed: {e}"
    if proc.returncode != 0:
        return False, f"scancel failed: {proc.stderr.strip()}"
    return True, f"cancelled Slurm job(s): {', '.join(jobs)}"


def _prune_empty_parents(path: str, root: str) -> None:
    """Remove now-empty ancestor dirs of path, stopping at root."""
    root = os.path.realpath(root)
    cur = os.path.dirname(os.path.realpath(path))
    while cur.startswith(root + os.sep):
        try:
            os.rmdir(cur)
        except OSError:
            return  # not empty (or gone): nothing further can be empty
        cur = os.path.dirname(cur)


def delete_run(run_rel: str, kill: bool) -> Tuple[bool, str]:
    """Delete a run's log dir and its mirrored videos dir.

    A run with a live process is refused unless kill is set, in which
    case the process tree gets SIGTERM, a grace period to shut down,
    then SIGKILL for stragglers before the files go. A run held by a
    Slurm job is cancelled the same way, and waited on until the job
    leaves the queue.
    """
    run_abs = safe_join(run_rel)
    if (not run_abs or not os.path.isdir(run_abs)
            or not os.path.basename(run_abs).startswith("run_")):
        return False, "not a run directory"
    if run_is_live(run_rel):
        if not kill:
            return False, "run has a live process; kill it first"
        kill_run(run_rel)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and run_is_live(run_rel):
            time.sleep(0.5)
        if pids_for_run(run_rel):
            kill_run(run_rel, signal.SIGKILL)
            time.sleep(0.5)
        if slurm_jobs_for_run(run_rel):
            # Nothing here can force a compute node to let go, and its
            # logging would recreate files under a half-deleted dir.
            return False, "Slurm job is still cancelling; retry shortly"
    # Resolve the video dir before the log dir goes: a moved run's dir
    # is found through its info.log (see _video_base).
    video_abs = _video_base(run_rel)
    try:
        shutil.rmtree(run_abs)
    except OSError as e:
        return False, f"delete failed: {e}"
    _prune_empty_parents(run_abs, LOGS_ROOT)
    if video_abs and os.path.isdir(video_abs):
        shutil.rmtree(video_abs, ignore_errors=True)
        _prune_empty_parents(video_abs, VIDEOS_ROOT)
    return True, "deleted"


def _parse_episode(path: str) -> Dict[str, Any]:
    text, _ = read_text(path)
    info: Dict[str, Any] = {}
    captures = CAPTURED_RE.findall(text)
    if captures or NOT_CAPTURED_RE.search(text):
        # The session used evaluate_option_plan: its capture verdicts are
        # the agent-side outcome. "Goal achieved" lines are per-rollout
        # goal-atom checks that stay True even when the evaluator rejects
        # the plan (solved=False), so they must not decide the session.
        # A later clean capture is never displaced by a rejected one, so
        # any non-best-effort capture means the session holds an answer.
        info["goal"] = any(note == "" for note in captures)
    else:
        goals = GOAL_RE.findall(text)
        info["goal"] = (goals[-1] == "True") if goals else None
    m: Optional[re.Match[str]] = None
    for m in RESULT_RE.finditer(text):
        pass
    if m:
        info["turns"] = int(m.group(1))
        info["solve_cost"] = float(m.group(2))
        info["total_cost"] = float(m.group(3))
    return info


def list_episodes(run_abs: str) -> List[Dict[str, Any]]:
    """Episode metadata for md files at the run dir root."""
    episodes: List[Dict[str, Any]] = []
    try:
        names = sorted(os.listdir(run_abs))
    except OSError:
        return episodes
    for name in names:
        m = EPISODE_RE.match(name)
        if not m:
            continue
        ep: Dict[str, Any] = {
            "file": name,
            "num": int(m.group(1)),
            "kind": m.group(2),
            "task": int(m.group(3)) if m.group(3) else None,
            "ts": m.group(4),
        }
        ep.update(_cached(os.path.join(run_abs, name), _parse_episode) or {})
        episodes.append(ep)
    return episodes


def _parse_info_log(path: str) -> Dict[str, Any]:
    """Extract authoritative test/explore outcomes from main.py's info.log.

    Returns {"totals": [(num_solved, num_total, avg_test_reward), ...] one
    per cycle (avg_test_reward is None in logs that predate the metric),
    "rounds": [{task_idx0: {"solved": bool, "msg": str,
    "reward": float?}, ...}, ...],
    "explore": {episode_num: {"reward": float, "terminated": bool,
    "accepted": bool, "msg": str}, ...},
    "test_round": {episode_num: round_idx, ...}} where each round is one
    test phase, closed by a "Tasks solved: X / Y" line.

    Both explore and test verdicts are paired by stream order. Explore:
    within a cycle every explore session logs its transcript save line
    before any interaction episode executes, and interaction episodes run
    in session order, so the oldest unmatched explore session owns the
    next verdict; a learn save line closes the cycle and drops sessions
    that never produced an interaction episode. Test: every test session
    of a round logs its save line before the "Tasks solved" line that
    closes the round, so all test episodes saved since the previous round
    belong to the round that line closes. (Counting learn episodes
    instead is wrong: only some configs run an initial pre-learning test
    round, so the offset between learn count and round index varies.)
    """
    text, _ = read_text(path, max_bytes=8 * 1024 * 1024)
    text = ANSI_RE.sub("", text)
    m_vid = VIDEO_SAVED_RE.search(text)
    totals: List[Tuple[int, int, Optional[float]]] = []
    rounds: List[Dict[int, Dict[str, Any]]] = []
    current: Dict[int, Dict[str, Any]] = {}
    explore: Dict[int, Dict[str, Any]] = {}
    test_round: Dict[int, int] = {}
    pending: List[int] = []
    pending_test: List[int] = []
    last_explore: Optional[int] = None
    done = False
    for line in text.splitlines():
        if DONE_RE.match(line):
            done = True
            continue
        if "Test results:" in line:
            ms, mt = NUM_SOLVED_RE.search(line), NUM_TOTAL_RE.search(line)
            mr = AVG_TEST_REWARD_RE.search(line)
            if ms and mt:
                totals.append(
                    (int(float(ms.group(1))), int(float(mt.group(1))),
                     float(mr.group(1)) if mr else None))
            continue
        m = TASK_VERDICT_RE.match(line)
        if m:
            msg = m.group(3).strip()
            # A later verdict for the same task within a round (e.g. the
            # impossible-goal SOLVED after an approach failure) overwrites.
            current[int(m.group(1)) - 1] = {
                "solved": msg.startswith("SOLVED"),
                "msg": msg,
            }
            continue
        if TASKS_SOLVED_RE.search(line):
            rounds.append(current)
            current = {}
            for num in pending_test:
                test_round[num] = len(rounds) - 1
            pending_test.clear()
            continue
        # Logged after the "Tasks solved" line, so this decorates the
        # round that line just closed.
        if "Per-task rewards:" in line and rounds:
            for tm in PER_TASK_REWARD_RE.finditer(line):
                verdict = rounds[-1].get(int(tm.group(1)))
                if verdict is not None:
                    verdict["reward"] = float(tm.group(2))
            continue
        m = SAVED_EP_RE.search(line)
        if m:
            if m.group(2) == "explore":
                pending.append(int(m.group(1)))
            elif m.group(2) == "learn":
                pending.clear()
            elif m.group(2) == "test":
                pending_test.append(int(m.group(1)))
            continue
        m = INTERACTION_RE.match(line)
        if m:
            last_explore = pending.pop(0) if pending else None
            if last_explore is not None:
                explore[last_explore] = {
                    "reward": float(m.group(1)),
                    "terminated": m.group(2) == "True",
                    "accepted": m.group(3) == "True",
                    "msg": "",
                }
            continue
        m = INTERACTION_REJECT_RE.match(line)
        if m and last_explore is not None:
            explore[last_explore]["msg"] = "REJECTED: " + m.group(1).strip()
            continue
        m = INTERACTION_BAR_RE.match(line)
        if m and last_explore is not None:
            explore[last_explore]["msg"] = "solved but " + m.group(1).strip()
    if current:  # run still in progress or crashed mid-round
        rounds.append(current)
    return {
        "totals": totals,
        "rounds": rounds,
        "explore": explore,
        "test_round": test_round,
        "done": done,
        "video_rel": m_vid.group(1) if m_vid else "",
    }


def _explore_results(episodes: EpList) -> List[Tuple[int, int, float]]:
    """Per-round explore outcomes as (accepted, total, mean env reward).

    Rounds with no executed interaction episode yet (hence no env
    verdict) are omitted rather than shown as 0/N.
    """
    by_round: Dict[int, EpList] = {}
    for ep in episodes:
        if ep["kind"] == "explore":
            by_round.setdefault(ep.get("round", 0), []).append(ep)
    out: List[Tuple[int, int, float]] = []
    for rnd in sorted(by_round):
        eps = by_round[rnd]
        rewards = [ep["env_reward"] for ep in eps if "env_reward" in ep]
        if not rewards:
            continue
        solved = sum(1 for ep in eps if ep.get("env_accepted"))
        out.append((solved, len(eps), sum(rewards) / len(rewards)))
    return out


def explore_results_str(summary: Dict[str, Any]) -> str:
    """The run's explore rounds as e.g. "1/2 (r 0.75) → 2/2 (r 0.90)"."""
    return " → ".join(f"{s}/{t} (r {mr:.2f})"
                      for s, t, mr in summary.get("explore_results", []))


def test_results_str(summary: Dict[str, Any]) -> str:
    """The run's test phases as e.g. "0/1 (r 0.00) → 1/1 (r 0.90)".

    The reward tag is omitted for logs that predate avg_test_reward.
    """
    parts = []
    for solved, total, reward in summary.get("test_results", []):
        s = f"{solved}/{total}"
        if reward is not None:
            s += f" (r {reward:.2f})"
        parts.append(s)
    return " → ".join(parts)


def run_summary(run_rel: str) -> Optional[Dict[str, Any]]:
    """Episodes, env-side test/explore outcomes, rounds, and total cost."""
    run_abs = safe_join(run_rel)
    if not run_abs or not os.path.isdir(run_abs):
        return None
    episodes = list_episodes(run_abs)
    parsed = _cached(os.path.join(run_abs, "info.log"), _parse_info_log) or {}
    rounds = parsed.get("rounds", [])
    explore_verdicts = parsed.get("explore", {})
    test_round = parsed.get("test_round", {})
    # ep["round"] (learn episodes seen so far) drives only the grid-row
    # layout; env verdicts pair by the stream-order test_round map, since
    # the learn count offset from the round index varies per config.
    learn_seen = 0
    interactions_seen = 0
    for ep in episodes:
        ep["round"] = learn_seen
        if ep["kind"] == "learn":
            learn_seen += 1
            interactions_seen = 0
        if ep["kind"] == "explore":
            verdict = explore_verdicts.get(ep["num"])
            if verdict is not None:
                ep["env_reward"] = verdict["reward"]
                ep["env_terminated"] = verdict["terminated"]
                ep["env_accepted"] = verdict["accepted"]
                ep["env_msg"] = verdict["msg"]
                # Interaction episodes execute in session order, so this
                # session's episode -- and its __ep<i> video -- is the
                # i-th among the cycle's verdict-earning explore sessions.
                ep["interaction_idx"] = interactions_seen
                interactions_seen += 1
        round_i = test_round.get(ep["num"])
        if (ep["kind"] == "test" and ep["task"] is not None
                and round_i is not None and round_i < len(rounds)):
            verdict = rounds[round_i].get(ep["task"])
            if verdict is not None:
                ep["env_solved"] = verdict["solved"]
                ep["env_msg"] = verdict["msg"]
                if verdict.get("reward") is not None:
                    ep["env_reward"] = verdict["reward"]
    total_cost = max((e.get("total_cost", 0.0) for e in episodes), default=0.0)
    return {
        "episodes": episodes,
        "test_results": parsed.get("totals", []),
        "explore_results": _explore_results(episodes),
        "rounds": rounds,
        "total_cost": total_cost,
        "done": parsed.get("done", False),
    }


def run_stamp(run_rel: str) -> str:
    """Cheap change fingerprint of a run dir (file count, mtime, bytes)."""
    run_abs = safe_join(run_rel)
    if not run_abs or not os.path.isdir(run_abs):
        return "gone"
    count, max_mtime, total = 0, 0.0, 0
    for dirpath, _, files in os.walk(run_abs):
        for f in files:
            try:
                st = os.stat(os.path.join(dirpath, f))
            except OSError:
                continue
            count += 1
            total += st.st_size
            max_mtime = max(max_mtime, st.st_mtime)
    return f"{int(count)}-{int(max_mtime)}-{int(total)}"


def index_stamp() -> str:
    """Fingerprint of the runs overview: run set + run-root activity."""
    parts = []
    for r in find_runs():
        run_abs = safe_join(r["rel"])
        if not run_abs:
            continue
        try:
            info = os.path.join(run_abs, "info.log")
            info_size = os.path.getsize(info) if os.path.exists(info) else 0
        except OSError:
            continue
        # Activity at minute granularity: the modified column displays
        # minutes, so finer resolution would only cause no-op reloads.
        parts.append(f"{r['rel']}:{int(r['mtime'])}:{int(info_size)}:"
                     f"{int(r['activity'] // 60)}")
    return f"{len(parts)}:{int(hash(tuple(parts)) & 4294967295)}"


# ------------------------------------------------------------ asset lookup


def resolve_asset(ref: str, run_rel: str) -> Optional[str]:
    """Map an image/file reference inside a transcript to a /raw URL."""
    ref = ref.strip().strip("`'\"")
    run_abs = safe_join(run_rel)
    if not run_abs:
        return None
    candidates = []
    if ref.startswith("/"):
        candidates.append(ref)
        # Absolute path from another machine: re-root at the run dir name.
        marker = "/" + os.path.basename(run_abs) + "/"
        if marker in ref:
            candidates.append(os.path.join(run_abs, ref.split(marker, 1)[1]))
    else:
        rel = ref[2:] if ref.startswith("./") else ref
        candidates += [
            os.path.join(run_abs, rel),
            os.path.join(run_abs, "sandbox", rel),
            os.path.join(run_abs, "test_images", os.path.basename(rel)),
            os.path.join(run_abs, "sandbox", "test_images",
                         os.path.basename(rel)),
        ]
    root = os.path.realpath(LOGS_ROOT)
    for cand in candidates:
        real = os.path.realpath(cand)
        if real.startswith(root + os.sep) and os.path.isfile(real):
            return "/raw?p=" + q(
                os.path.relpath(real, root).replace(os.sep, "/"))
    return None


def _image_size_uncached(path: str) -> Optional[Tuple[int, int]]:
    """Pixel (width, height) parsed from the file header, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                return (int.from_bytes(head[16:20], "big"),
                        int.from_bytes(head[20:24], "big"))
            if head[:6] in (b"GIF87a", b"GIF89a"):
                return (int.from_bytes(head[6:8], "little"),
                        int.from_bytes(head[8:10], "little"))
            if head[:2] == b"BM":
                return (int.from_bytes(head[18:22], "little"),
                        abs(int.from_bytes(head[22:26], "little",
                                           signed=True)))
            if head[:2] == b"\xff\xd8":  # JPEG: scan segments for a SOF.
                f.seek(2)
                while True:
                    byte = f.read(1)
                    if not byte:
                        return None
                    if byte != b"\xff":
                        continue
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if not marker:
                        return None
                    code = marker[0]
                    # Standalone markers carry no length field.
                    if code in (0x01, 0xD8) or 0xD0 <= code <= 0xD7:
                        continue
                    if code in (0xD9, 0xDA):  # EOI / start of scan data.
                        return None
                    len_bytes = f.read(2)
                    if len(len_bytes) < 2:
                        return None
                    length = int.from_bytes(len_bytes, "big")
                    if length < 2:  # corrupt; would seek backwards
                        return None
                    if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                        body = f.read(5)
                        if len(body) < 5:
                            return None
                        return (int.from_bytes(body[3:5], "big"),
                                int.from_bytes(body[1:3], "big"))
                    f.seek(length - 2, 1)
    except OSError:
        pass
    return None


def size_attrs(path: str) -> str:
    """width/height attributes for an <img> tag, or "" if unknown.

    Lazy-loaded images otherwise occupy no space until fetched, so a
    reloaded page is far shorter than it was and restoring the saved
    scrollTop clamps to the bottom. Intrinsic dimensions let the browser
    reserve the exact layout before any image loads.
    """
    dims = _cached(path, _image_size_uncached)
    return f" width='{dims[0]}' height='{dims[1]}'" if dims else ""


def size_attrs_for_url(url: Optional[str]) -> str:
    """size_attrs for a /raw image URL as built by resolve_asset."""
    if not url or not url.startswith("/raw?p="):
        return ""
    path = safe_join(urllib.parse.unquote(url[len("/raw?p="):]))
    return size_attrs(path) if path else ""


def thumbs_for_paths(text: str, run_rel: str, cap: int = 24) -> str:
    """Row of thumbnail links for up to cap image paths found in text."""
    seen, out = set(), []
    for m in PATH_IN_TEXT_RE.finditer(text):
        url = resolve_asset(m.group(0), run_rel)
        if url and url not in seen:
            seen.add(url)
            out.append(f'<a class="thumb" href="{url}" target="_blank">'
                       f'<img loading="lazy"{size_attrs_for_url(url)} '
                       f'src="{url}"></a>')
        if len(out) >= cap:
            break
    if not out:
        return ""
    return f"<div class=\"thumbs\">{''.join(out)}</div>"


# --------------------------------------------------------------- markdown

INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
LINK_MD_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def render_inline(line: str, run_rel: str) -> str:
    """Inline markdown on an already-raw line; escapes HTML itself."""
    tokens: List[str] = []

    def stash(html_frag: str) -> str:
        tokens.append(html_frag)
        return f"\x00{int(len(tokens) - 1)}\x00"

    def sub_code(m: re.Match[str]) -> str:
        content = m.group(1)
        frag = f"<code>{esc(content)}</code>"
        url = resolve_asset(content, run_rel)
        if url and os.path.splitext(content)[1].lower() in IMG_EXTS:
            frag += (f' <a class="thumb inline" href="{url}" target="_blank">'
                     f'<img loading="lazy"{size_attrs_for_url(url)} '
                     f'src="{url}"></a>')
        return stash(frag)

    def sub_img(m: re.Match[str]) -> str:
        alt, src = m.group(1), m.group(2)
        url = resolve_asset(src, run_rel) or esc(src)
        return stash(f'<img class="mdimg" loading="lazy"'
                     f'{size_attrs_for_url(url)} '
                     f'alt="{esc(alt)}" src="{url}">')

    def sub_link(m: re.Match[str]) -> str:
        label, href = m.group(1), m.group(2)
        url = resolve_asset(href, run_rel) if not href.startswith(
            ("http://", "https://")) else esc(href)
        return stash(f'<a href="{url or esc(href)}" '
                     f'target="_blank">{esc(label)}</a>')

    line = INLINE_CODE_RE.sub(sub_code, line)
    line = IMG_MD_RE.sub(sub_img, line)
    line = LINK_MD_RE.sub(sub_link, line)
    line = esc(line)
    line = BOLD_RE.sub(r"<strong>\1</strong>", line)
    for i, frag in enumerate(tokens):
        line = line.replace(f"\x00{int(i)}\x00", frag)
    return line


def render_md_chunk(lines: List[str], run_rel: str) -> str:
    """Render markdown lines (no section splitting) to HTML."""
    out = []
    i, n = 0, len(lines)
    para: List[str] = []
    list_tag: Optional[str] = None

    def flush_para() -> None:
        if para:
            joined = "<br>".join(render_inline(l, run_rel) for l in para)
            out.append(f"<p>{joined}</p>")
            del para[:]

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_para()
            close_list()
            lang = stripped[3:].strip()
            block = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code = "\n".join(block)
            out.append(f'<pre class="code" data-lang="{esc(lang)}"><code>'
                       f"{esc(code)}</code></pre>")
            out.append(thumbs_for_paths(code, run_rel))
            continue
        if not stripped:
            flush_para()
            close_list()
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para()
            close_list()
            level = min(len(m.group(1)) + 1, 6)  # demote: page owns h1
            heading = render_inline(m.group(2), run_rel)
            out.append(f"<h{level}>{heading}</h{level}>")
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,})$", stripped):
            flush_para()
            close_list()
            out.append("<hr>")
            i += 1
            continue
        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            flush_para()
            if list_tag != "ul":
                close_list()
                out.append("<ul>")
                list_tag = "ul"
            out.append(f"<li>{render_inline(m.group(1), run_rel)}</li>")
            i += 1
            continue
        m = re.match(r"^\d+\.\s+(.*)$", stripped)
        if m:
            flush_para()
            if list_tag != "ol":
                close_list()
                out.append("<ol>")
                list_tag = "ol"
            out.append(f"<li>{render_inline(m.group(1), run_rel)}</li>")
            i += 1
            continue
        if stripped.startswith(">"):
            flush_para()
            close_list()
            quoted = render_inline(stripped.lstrip("> "), run_rel)
            out.append(f"<blockquote>{quoted}</blockquote>")
            i += 1
            continue
        # Indented pseudo-code blocks (state dumps) keep their spacing.
        if line.startswith(("    ", "\t")) and not para:
            close_list()
            block = []
            while i < n and (lines[i].startswith(
                ("    ", "\t")) or not lines[i].strip()):
                if not lines[i].strip() and (i + 1 >= n
                                             or not lines[i + 1].startswith(
                                                 ("    ", "\t"))):
                    break
                block.append(lines[i])
                i += 1
            code = "\n".join(block)
            out.append(f'<pre class="code"><code>{esc(code)}</code></pre>')
            out.append(thumbs_for_paths(code, run_rel))
            continue
        para.append(line)
        i += 1
    flush_para()
    close_list()
    return "\n".join(out)


def split_fence_aware(
    lines: List[str], pattern: re.Pattern[str]
) -> Tuple[List[str], List[Tuple[re.Match[str], List[str]]]]:
    """Split lines at non-fenced lines matching pattern.

    Returns (head_lines, [(match, body_lines), ...]).
    """
    head: List[str] = []
    sections: List[Tuple[re.Match[str], List[str]]] = []
    current_body: Optional[List[str]] = None
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else pattern.match(line)
        if m:
            current_body = []
            sections.append((m, current_body))
        elif current_body is not None:
            current_body.append(line)
        else:
            head.append(line)
    return head, sections


H2_RE = re.compile(r"^##\s+(.*)$")
TURN_RE = re.compile(r"^###\s+(Turn\s+\d+.*)$")


def turn_hint(body_lines: List[str]) -> str:
    """First non-decorative line of a turn body, as a short escaped hint."""
    for line in body_lines:
        s = line.strip()
        if s.startswith("```"):
            continue
        s = re.sub(r"[*`#\[\]]", "", s).strip()
        if not s or re.fullmatch(r"[-=_.|:]+", s):
            continue
        if len(s) > 90:
            s = s[:90] + "…"
        return esc(s)
    return ""


def render_md_document(text: str, run_rel: str) -> str:
    """Full markdown doc: collapsible ## sections, nested Turn details."""
    lines = text.splitlines()
    head, sections = split_fence_aware(lines, H2_RE)
    out = [render_md_chunk(head, run_rel)]
    for m, body in sections:
        title = m.group(1)
        sub_head, turns = split_fence_aware(body, TURN_RE)
        inner = [render_md_chunk(sub_head, run_rel)]
        for j, (tm, tbody) in enumerate(turns):
            is_last = j == len(turns) - 1
            open_attr = " open" if is_last else ""
            inner.append(f'<details class="turn"{open_attr}><summary>'
                         f'{esc(tm.group(1))} <span class="hint">'
                         f'{turn_hint(tbody)}</span></summary>'
                         f'{render_md_chunk(tbody, run_rel)}</details>')
        default_open = (turns or title.lower().startswith(
            ("goal", "conversation", "prompt")))
        out.append(
            '<details class="section"%s><summary>%s</summary>%s</details>' %
            (" open" if default_open else "", esc(title), "\n".join(inner)))
    return "\n".join(out)


# ------------------------------------------------------------- ANSI logs

ANSI_COLORS = {
    "30": "#6e7681",
    "31": "#e5534b",
    "32": "#57ab5a",
    "33": "#c69026",
    "34": "#539bf5",
    "35": "#b083f0",
    "36": "#39c5cf",
    "37": "#adbac7",
    "90": "#6e7681",
    "91": "#ff938a",
    "92": "#6bc46d",
    "93": "#daaa3f",
    "94": "#6cb6ff",
    "95": "#dcbdfb",
    "96": "#56d4dd",
    "97": "#cdd9e5",
}


def ansi_to_html(text: str) -> str:
    """Convert ANSI color/bold escape codes in text to HTML spans."""
    out, pos, open_span = [], 0, False
    for m in ANSI_RE.finditer(text):
        out.append(esc(text[pos:m.start()]))
        pos = m.end()
        codes = (m.group(1) or "0").split(";")
        if open_span:
            out.append("</span>")
            open_span = False
        styles = []
        for c in codes:
            if c in ANSI_COLORS:
                styles.append(f"color:{ANSI_COLORS[c]}")
            elif c == "1":
                styles.append("font-weight:bold")
        if styles:
            out.append(f"<span style=\"{';'.join(styles)}\">")
            open_span = True
    out.append(esc(text[pos:]))
    if open_span:
        out.append("</span>")
    return "".join(out)


# ------------------------------------------------------------ HTML shell

CSS = """
:root {
  --bg: #ffffff; --fg: #24292f; --muted: #57606a; --border: #d0d7de;
  --panel: #f6f8fa; --accent: #0969da; --ok: #1a7f37; --bad: #cf222e;
  --code-bg: #f6f8fa; --add: #dafbe1; --del: #ffebe9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #c9d1d9; --muted: #8b949e; --border: #30363d;
    --panel: #161b22; --accent: #58a6ff; --ok: #3fb950; --bad: #f85149;
    --code-bg: #161b22; --add: #12261e; --del: #35181a;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
  Arial, sans-serif; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre { font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
.topbar { position: sticky; top: 0; z-index: 10; display: flex; gap: 12px;
  align-items: center; padding: 8px 16px; background: var(--panel);
  border-bottom: 1px solid var(--border); }
.topbar h1 { font-size: 15px; margin: 0; white-space: nowrap; }
.topbar input { flex: 1; max-width: 420px; padding: 4px 10px;
  border: 1px solid var(--border); border-radius: 6px;
  background: var(--bg); color: var(--fg); }
.topbar .crumb { color: var(--muted); font-size: 13px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
button { padding: 4px 10px; border: 1px solid var(--border);
  border-radius: 6px; background: var(--bg); color: var(--fg);
  cursor: pointer; font-size: 12px; }
button:hover { border-color: var(--accent); }
.layout { display: flex; height: calc(100vh - 45px); }
.sidebar { width: 340px; min-width: 240px; overflow-y: auto; padding: 10px;
  border-right: 1px solid var(--border); background: var(--panel);
  resize: horizontal; }
.content { flex: 1; overflow-y: auto; padding: 16px 24px; }
.content { max-width: 100%; }
.content pre.code { background: var(--code-bg); padding: 10px 12px;
  border: 1px solid var(--border); border-radius: 6px; overflow-x: auto;
  white-space: pre; }
.content h2, .content h3 { border-bottom: 1px solid var(--border);
  padding-bottom: 4px; }
details.section, details.turn { border: 1px solid var(--border);
  border-radius: 6px; margin: 8px 0; background: var(--bg); }
details.section > summary, details.turn > summary { cursor: pointer;
  padding: 6px 12px; font-weight: 600; background: var(--panel);
  border-radius: 6px; }
details[open].section > summary, details[open].turn > summary {
  border-bottom: 1px solid var(--border);
  border-radius: 6px 6px 0 0; }
details.section > *:not(summary), details.turn > *:not(summary) {
  margin-left: 12px; margin-right: 12px; }
details.turn { margin-left: 8px; }
summary .hint { color: var(--muted); font-weight: 400; font-size: 12px; }
.chip { display: inline-block; padding: 0 7px; border-radius: 10px;
  font-size: 11px; line-height: 18px; border: 1px solid var(--border);
  color: var(--muted); margin-right: 3px; white-space: nowrap; }
.chip.ok { color: var(--ok); border-color: var(--ok); }
.chip.bad { color: var(--bad); border-color: var(--bad); }
.chip.kind-explore { color: #b083f0; border-color: #b083f0; }
.chip.kind-learn { color: #daaa3f; border-color: #daaa3f; }
/* Lifecycle, not verdict: green and red stay reserved for env evals. */
.chip.live { color: var(--accent); border-color: var(--accent);
  animation: pulse 1.8s ease-in-out infinite; }
.chip.stopped { color: #daaa3f; border-color: #daaa3f; }
@keyframes pulse { 50% { opacity: .45; } }
@media (prefers-reduced-motion: reduce) {
  .chip.live { animation: none; }
}
.banner { padding: 10px 14px; border-radius: 6px; margin-bottom: 12px;
  border: 1px solid var(--border); background: var(--panel);
  display: flex; gap: 18px; flex-wrap: wrap; }
.banner.warn { border-color: var(--bad); color: var(--bad);
  font-weight: 600; }
.ok { color: var(--ok); font-weight: 700; }
.bad { color: var(--bad); font-weight: 700; }
table.grid { border-collapse: collapse; margin: 10px 0; }
table.grid th, table.grid td { border: 1px solid var(--border);
  padding: 4px 10px; text-align: left; font-size: 13px; }
table.grid th { background: var(--panel); }
table.grid.runs { table-layout: fixed; }
table.epgrid { border-collapse: collapse; margin: 0;
  table-layout: fixed; }
table.epgrid td { border: none; padding: 1px 0; line-height: 20px;
  vertical-align: top; }
.sidebar .nav a { display: block; padding: 3px 6px; border-radius: 4px;
  color: var(--fg); font-size: 13px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.sidebar .nav a:hover { background: var(--bg); text-decoration: none; }
.sidebar .nav a.active { background: var(--accent); color: #fff; }
.sidebar .nav a.active .chip, .sidebar .nav a.active .muted {
  color: #fff; border-color: #fff; }
.sidebar h3 { font-size: 12px; text-transform: uppercase;
  letter-spacing: .5px; color: var(--muted); margin: 14px 0 4px; }
.sidebar details { margin-left: 8px; }
.sidebar details summary { cursor: pointer; font-size: 13px;
  color: var(--muted); }
.thumbs { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0; }
/* width/height attrs on <img> reserve layout space before lazy images
   load (needed for exact scroll restore); auto on the free axis keeps
   the aspect ratio instead of taking the attr value literally. */
.thumb img { height: 110px; width: auto; border: 1px solid var(--border);
  border-radius: 4px; }
.thumb.inline img { height: 60px; width: auto; vertical-align: middle; }
img.mdimg { max-width: 480px; max-height: 400px; height: auto;
  border: 1px solid var(--border); border-radius: 6px; display: block;
  margin: 6px 0; }
/* A test and a failure video can both exist for one task, so the players
   sit in a wrapping row rather than assuming a single video. */
.vids { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 12px; }
figure.vid { margin: 0; }
figure.vid video { max-width: 480px; width: 100%;
  border: 1px solid var(--border); border-radius: 6px; display: block;
  background: #000; }
figure.vid figcaption { font-size: 11px; margin-top: 4px; }
.gallery { display: grid; gap: 8px;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
.gallery a { text-align: center; font-size: 11px; color: var(--muted);
  overflow-wrap: anywhere; }
.gallery img { width: 100%; height: auto; border: 1px solid var(--border);
  border-radius: 4px; }
pre.logview { white-space: pre-wrap; overflow-wrap: anywhere; }
.diff .add { background: var(--add); display: block; }
.diff .del { background: var(--del); display: block; }
.diff .hunk { color: var(--accent); display: block; }
#lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.85);
  display: none; align-items: center; justify-content: center;
  z-index: 100; cursor: zoom-out; }
#lightbox img { max-width: 96vw; max-height: 96vh; }
.runrow.hidden { display: none; }
#refreshpill { position: fixed; right: 18px; bottom: 18px; z-index: 50;
  background: var(--accent); color: #fff; border: none;
  padding: 8px 14px; border-radius: 18px; font-weight: 600;
  box-shadow: 0 2px 8px rgba(0,0,0,.35); }
details.grp { border: 1px solid var(--border); border-radius: 6px;
  margin: 8px 0; }
details.grp > summary { cursor: pointer; padding: 6px 12px;
  font-weight: 600; background: var(--panel); border-radius: 6px; }
details.grp[open] > summary { border-bottom: 1px solid var(--border);
  border-radius: 6px 6px 0 0; }
details.grp.family > summary { font-size: 15px; }
details.grp > *:not(summary) { margin: 8px 12px; }
details.grp.hidden { display: none; }
.muted { color: var(--muted); }
button.copybtn { padding: 0 3px; margin-left: 5px; border: none;
  background: none; color: var(--muted); font-size: 12px;
  visibility: hidden; }
tr.runrow:hover button.copybtn { visibility: visible; }
button.copybtn:hover { color: var(--accent); }
button.copybtn.copied { color: var(--ok); visibility: visible; }
button.rowbtn { padding: 0 5px; margin-left: 5px;
  border: 1px solid transparent; border-radius: 4px; background: none;
  color: var(--muted); font-size: 11px; visibility: hidden; }
tr.runrow:hover button.rowbtn { visibility: visible; }
button.rowbtn:hover { color: var(--bad); border-color: var(--bad); }
""" + f"""
table.epgrid td.task {{ width: {TASK_W}px; }}
table.epgrid td.misc {{ width: {MISC_W}px; }}
table.epgrid td.rnd {{ width: {ROUND_W}px; }}
"""

JS = """
function $(s, r) { return (r || document).querySelector(s); }
function $all(s, r) { return Array.from((r || document).querySelectorAll(s)); }

// Lightbox for images.
document.addEventListener('click', function(e) {
  var a = e.target.closest('a.thumb, a.shot');
  var img = e.target.closest('img.mdimg, .gallery img');
  var src = null;
  if (a) { src = a.href; }
  else if (img) { src = img.src; }
  if (src && /raw\\?p=/.test(src)) {
    e.preventDefault();
    var lb = $('#lightbox');
    lb.firstElementChild.src = src;
    lb.style.display = 'flex';
  }
});
document.addEventListener('DOMContentLoaded', function() {
  var lb = document.createElement('div');
  lb.id = 'lightbox';
  lb.innerHTML = '<img>';
  lb.onclick = function() { lb.style.display = 'none'; };
  document.body.appendChild(lb);
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') lb.style.display = 'none';
  });
});

// Copy-path buttons next to run names.
document.addEventListener('click', function(e) {
  var b = e.target.closest('button.copybtn');
  if (!b) return;
  var done = function() {
    b.textContent = '\\u2713';
    b.classList.add('copied');
    setTimeout(function() {
      b.textContent = '\\u29c9';
      b.classList.remove('copied');
    }, 1200);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(b.dataset.copy).then(done);
  } else {  // non-localhost http has no clipboard API
    var ta = document.createElement('textarea');
    ta.value = b.dataset.copy;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
    done();
  }
});

function setAllDetails(open) {
  $all('.content details').forEach(function(d) { d.open = open; });
}

// Index page: filter + compare + collapsible groups.
function groupKey(d) { return 'lv-grp:' + d.dataset.key; }
function restoreGroups() {
  $all('details.grp').forEach(function(d) {
    var v = localStorage.getItem(groupKey(d));
    if (v !== null) d.open = v === '1';
  });
}
document.addEventListener('DOMContentLoaded', restoreGroups);
document.addEventListener('toggle', function(e) {
  var d = e.target;
  if (d.classList && d.classList.contains('grp') && !window._filtering)
    localStorage.setItem(groupKey(d), d.open ? '1' : '0');
}, true);
function setAllGroups(open) {
  window._filtering = true;
  $all('details.grp').forEach(function(d) {
    d.open = open;
    localStorage.setItem(groupKey(d), open ? '1' : '0');
  });
  window._filtering = false;
}
function filterRuns(text) {
  text = text.toLowerCase();
  sessionStorage.setItem('lv-filter', text);
  window._filtering = true;
  $all('.runrow').forEach(function(row) {
    row.classList.toggle('hidden',
      !!text && row.dataset.key.indexOf(text) === -1);
  });
  $all('details.grp.exp').forEach(function(d) {
    var any = $all('.runrow', d).some(function(r) {
      return !r.classList.contains('hidden');
    });
    d.classList.toggle('hidden', !any);
    if (text) d.open = any;
  });
  $all('details.grp.family').forEach(function(d) {
    var any = $all('details.grp.exp', d).some(function(x) {
      return !x.classList.contains('hidden');
    });
    d.classList.toggle('hidden', !any);
    if (text) d.open = any;
  });
  if (!text) restoreGroups();
  window._filtering = false;
}
// Index page: sort runs by seed (the server order) or by start time.
// Time mode interleaves each experiment's seeds newest-first; the
// experiment and family groups themselves stay in place.
function sortMode() { return localStorage.getItem('lv-sort') || 'seed'; }
function toggleSort() {
  localStorage.setItem('lv-sort', sortMode() === 'seed' ? 'time' : 'seed');
  paintSortBtn();
  applySort();
}
function paintSortBtn() {
  var b = $('#sortbtn');
  if (b) b.textContent = 'sort: ' + sortMode();
}
function applySort() {
  if (!$('#sortbtn')) return;
  var byTime = sortMode() === 'time';
  $all('table.runs').forEach(function(t) {
    var rows = $all('tr.runrow', t);
    rows.sort(function(a, b) {
      if (!byTime && a.dataset.seed !== b.dataset.seed)
        return a.dataset.seed < b.dataset.seed ? -1 : 1;
      return b.dataset.start - a.dataset.start;
    });
    var tbody = rows.length ? rows[0].parentNode : null;
    rows.forEach(function(r) { tbody.appendChild(r); });
  });
}
function compareSelected() {
  var sel = $all('input.cmp:checked').map(function(c) { return c.value; });
  if (sel.length < 2) { alert('Select at least two runs to compare.'); return; }
  location.href = '/compare?runs=' + encodeURIComponent(sel.join(';'));
}

// Kill / delete buttons on index run rows. POST only, so the auto-
// refresh GETs can never trip these; reload shortly after success so
// the status chip reflects the process actually exiting.
function postRun(url, msg) {
  if (!confirm(msg)) return;
  fetch(url, {method: 'POST'}).then(function(r) {
    r.text().then(function(t) {
      if (!r.ok) { alert(t); return; }
      setTimeout(function() { location.reload(); }, 600);
    });
  }).catch(function(e) { alert('request failed: ' + e); });
}
function killRun(rel) {
  postRun('/kill?d=' + encodeURIComponent(rel),
          'Kill the live process of\\n' + rel + ' ?');
}
function deleteRun(rel, live) {
  var msg = live
    ? 'This run appears LIVE:\\n' + rel +
      '\\nKill its process AND delete its log dir (and videos)?'
    : 'Delete the log dir (and videos) of\\n' + rel + ' ?';
  postRun('/delete?d=' + encodeURIComponent(rel) + (live ? '&kill=1' : ''),
          msg);
}

// Run page: hash routing into the content pane.
function loadHash() {
  var content = $('#content');
  if (!content) return;
  var h = decodeURIComponent(location.hash.slice(1));
  var run = content.dataset.run;
  var url = null;
  if (h.indexOf('f=') === 0) {
    url = '/view?d=' + encodeURIComponent(run) +
          '&f=' + encodeURIComponent(h.slice(2));
  } else if (h.indexOf('gallery=') === 0) {
    url = '/gallery?d=' + encodeURIComponent(run) +
          '&p=' + encodeURIComponent(h.slice(8));
  } else if (h.indexOf('diff=') === 0) {
    url = '/diff?d=' + encodeURIComponent(run) +
          '&f=' + encodeURIComponent(h.slice(5));
  } else if (content.dataset.default) {
    location.hash = '#f=' + content.dataset.default;
    return;
  }
  if (!url) return;
  content.innerHTML = '<p class="muted">Loading…</p>';
  fetch(url).then(function(r) { return r.text(); }).then(function(html) {
    content.innerHTML = html;
    var r = window._restore;
    window._restore = null;
    if (r && r.open) {
      var ds = $all('details', content);
      r.open.forEach(function(i) { if (ds[i]) ds[i].open = true; });
    }
    var want = (r && typeof r.content === 'number') ? r.content : 0;
    content.scrollTop = want;
    if (want && content.scrollTop < want) {
      // The set clamped: some content (images without intrinsic sizes,
      // videos) has not laid out yet; re-apply once. When the first set
      // sticks, no timer runs, so a user scrolling right away is not
      // yanked back.
      setTimeout(function() {
        if (content.scrollTop < want) content.scrollTop = want;
      }, 400);
    }
    $all('.sidebar .nav a').forEach(function(a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + h);
    });
  });
}
window.addEventListener('hashchange', loadHash);
document.addEventListener('DOMContentLoaded', loadHash);

// Preserve scroll positions (and open sections) across reloads. The
// scrollable panes are inner divs, so native scroll restoration does not
// cover them.
function scrollKey() {
  return 'lv-scroll:' + location.pathname + location.search + location.hash;
}
function saveScroll() {
  var state = {};
  var c = $('#content');
  var p = $('.content');
  if (c) {
    state.content = c.scrollTop;
    state.open = $all('details', c)
      .map(function(d, i) { return d.open ? i : -1; })
      .filter(function(i) { return i >= 0; });
  } else if (p) {
    state.page = p.scrollTop;
  }
  var sb = $('.sidebar');
  if (sb) state.sidebar = sb.scrollTop;
  try { sessionStorage.setItem(scrollKey(), JSON.stringify(state)); }
  catch (e) {}
}
window.addEventListener('beforeunload', saveScroll);
window.addEventListener('pagehide', saveScroll);
window._restore = (function() {
  var raw = null;
  try { raw = sessionStorage.getItem(scrollKey()); } catch (e) {}
  if (!raw) return null;
  sessionStorage.removeItem(scrollKey());
  try { return JSON.parse(raw); } catch (e) { return null; }
})();

// Auto-refresh: poll /stamp every 10s; reload only when logs changed.
var REFRESH_MS = 10000;
function autoOn() { return localStorage.getItem('lv-auto') !== '0'; }
function toggleAuto() {
  localStorage.setItem('lv-auto', autoOn() ? '0' : '1');
  paintAutoBtn();
}
function paintAutoBtn() {
  var b = $('#arbtn');
  if (b) b.textContent = 'auto-refresh: ' + (autoOn() ? 'on' : 'off');
}
function showRefreshPill() {
  if ($('#refreshpill')) return;
  var p = document.createElement('button');
  p.id = 'refreshpill';
  p.textContent = 'Logs updated — refresh';
  p.onclick = function() { location.reload(); };
  document.body.appendChild(p);
}
function pollStamp() {
  if (document.visibilityState !== 'visible') return;
  var content = $('#content');
  var url = '/stamp' + (content && content.dataset.run
    ? '?d=' + encodeURIComponent(content.dataset.run) : '');
  fetch(url).then(function(r) { return r.text(); }).then(function(s) {
    if (window._stamp === undefined) { window._stamp = s; return; }
    if (s !== window._stamp) {
      window._stamp = s;
      if (autoOn()) { location.reload(); } else { showRefreshPill(); }
    }
  }).catch(function() {});
}
document.addEventListener('DOMContentLoaded', function() {
  paintAutoBtn();
  paintSortBtn();
  applySort();
  pollStamp();
  setInterval(pollStamp, REFRESH_MS);
  var f = $('#runfilter');
  if (f) {
    f.value = sessionStorage.getItem('lv-filter') || '';
    if (f.value) filterRuns(f.value);
  }
  var r = window._restore;
  if (r) {
    var sb = $('.sidebar');
    if (sb && typeof r.sidebar === 'number') sb.scrollTop = r.sidebar;
    var p = $('.content');
    if (!$('#content') && p && typeof r.page === 'number') {
      p.scrollTop = r.page;
      window._restore = null;
    }
  }
});

function filterLog() {
  var needle = $('#logfilter').value.toLowerCase();
  $all('#logview > span.logline').forEach(function(l) {
    l.style.display =
      !needle || l.textContent.toLowerCase().indexOf(needle) !== -1
      ? '' : 'none';
  });
}
"""


def page(title: str, topbar_extra: str, body: str) -> str:
    """Wrap body in the full HTML page shell (topbar, CSS, and JS)."""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{esc(title)}</title><style>{CSS}</style>"
        f"<script>{JS}</script></head>"
        "<body><div class='topbar'><h1><a href='/'>log viewer</a></h1>"
        f"{topbar_extra}<button id='arbtn' onclick='toggleAuto()'></button>"
        f"</div>{body}</body></html>")


def chip(label: Any, cls: str = "", title: str = "") -> str:
    """Small labeled span with an optional CSS class and tooltip."""
    return (f'<span class="chip {cls}" title="{esc(title)}">'
            f'{esc(str(label))}</span>')


def test_mark(ep: Dict[str, Any]) -> Tuple[str, str, str]:
    """(mark, css class, tooltip) for a test episode.

    Only the env-side verdict from info.log is authoritative, so only it
    earns a colour. The transcript's own "Goal achieved" claim is shown
    parenthesized and grey: it is a fallback that can still flip once
    the episode is evaluated, and colouring it reads as a settled
    outcome.
    """
    if "env_solved" in ep:
        mark = "✓" if ep["env_solved"] else "✗"
        if "env_reward" in ep:
            mark += f" {ep['env_reward']:.2f}"
        if ep["env_solved"]:
            return mark, "ok", "env eval: SOLVED"
        return mark, "bad", "env eval: " + ep.get("env_msg", "failed")
    if ep.get("goal") is True:
        return "(✓)", "", "agent-reported only; no env verdict in info.log"
    if ep.get("goal") is False:
        return "(✗)", "", "agent-reported only; no env verdict in info.log"
    return "", "", ""


def explore_mark(ep: Dict[str, Any]) -> Tuple[str, str, str]:
    """(mark, css class, tooltip) for an explore episode's env verdict.

    The mark carries the episode's env reward alongside the ✓/✗ because
    reward is the explore-side score of interest (the early-stopping bar
    reads it, not just the boolean).
    """
    mark = ("✓" if ep["env_accepted"] else "✗") + f" {ep['env_reward']:.2f}"
    cls = "ok" if ep["env_accepted"] else "bad"
    title = (f"env eval: reward={ep['env_reward']:.2f}, "
             f"terminated={ep['env_terminated']}, "
             f"accepted={ep['env_accepted']}")
    if ep.get("env_msg"):
        title += "; " + ep["env_msg"]
    return mark, cls, title


def _split_episodes(
    episodes: EpList
) -> Tuple[Dict[int, Dict[int, EpList]], Dict[int, EpList]]:
    """Episodes bucketed by round, as (test-by-task, non-test)."""
    tests: Dict[int, Dict[int, EpList]] = {}
    misc: Dict[int, EpList] = {}
    for ep in episodes:
        rnd = ep.get("round", 0)
        if ep["kind"] == "test" and ep["task"] is not None:
            tests.setdefault(rnd, {}).setdefault(ep["task"], []).append(ep)
        else:
            misc.setdefault(rnd, []).append(ep)
    return tests, misc


def grid_layout(runs: List[EpList]) -> Dict[str, Any]:
    """Column layout shared by every run's episode grid.

    Holding the task set and the round column fixed across every run on
    the page is what lets task t1's chips land at the same x whether the
    run above it retried t0 twice or not, and whether it belongs to the
    same experiment or not.
    """
    tasks: Set[int] = set()
    rounds = False
    misc = False
    for episodes in runs:
        by_round, misc_by_round = _split_episodes(episodes)
        rounds = rounds or max(list(by_round) + list(misc_by_round),
                               default=0) > 0
        misc = misc or bool(misc_by_round)
        for by_task in by_round.values():
            tasks.update(by_task)
    return {"tasks": sorted(tasks), "rounds": rounds, "misc": misc}


def grid_width(layout: Dict[str, Any]) -> int:
    """Pixel width of an episode grid drawn with this layout."""
    return (len(layout["tasks"]) * TASK_W +
            (ROUND_W if layout["rounds"] else 0) +
            (MISC_W if layout["misc"] else 0))


def episode_grid(episodes: EpList,
                 layout: Optional[Dict[str, Any]] = None) -> str:
    """Chips laid out one row per test round, one column per task.

    Vertical alignment makes it easy to compare a task's outcome against
    earlier rounds and against other runs; retries within a round stack
    inside their task's column rather than widening it.
    """
    if layout is None:
        layout = grid_layout([episodes])
    tests, misc = _split_episodes(episodes)
    if not tests and not misc:
        return ""
    n_rounds = max(list(tests) + list(misc)) + 1
    rows = []
    for rnd in range(n_rounds):
        cells = []
        if layout["rounds"]:
            label = f"r{int(rnd + 1)}" if n_rounds > 1 else ""
            cells.append(f"<td class='muted rnd'>{label}</td>")
        for task in layout["tasks"]:
            # One retry per line: two short chips would otherwise share a
            # line while longer ones stack, making the column read ragged.
            chips = "".join(f"<div>{_test_chip(ep)}</div>"
                            for ep in tests.get(rnd, {}).get(task, []))
            cells.append(f"<td class='task'>{chips}</td>")
        if layout["misc"]:
            chips = "".join(_misc_chip(ep) for ep in misc.get(rnd, []))
            cells.append(f"<td class='misc'>{chips}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (f"<table class='epgrid' style='width:{grid_width(layout)}px'>"
            f"{''.join(rows)}</table>")


def run_status(r: Dict[str, Any], summary: Dict[str, Any], live: LiveProcs,
               is_newest: bool) -> Tuple[str, str]:
    """(label, css class) for a run's lifecycle state.

    Completion is read off info.log, since main.py logs its terminating
    line only after the pipeline returns. A run that was killed or that
    crashed leaves no such line - but neither does a run that is still
    going, so the two are told apart by whether a process is still
    alive, locally or as a Slurm job on a compute node (see live_runs).
    """
    if summary.get("done"):
        return "done", "done"
    pinned, unpinned = live
    if (r["exp"], r["seed"], r["name"]) in pinned:
        return "running", "live"
    if is_newest and (r["exp"], r["seed"]) in unpinned:
        return "running", "live"
    return "interrupted", "stopped"


def status_chip(r: Dict[str, Any], summary: Dict[str, Any], live: LiveProcs,
                is_newest: bool) -> str:
    """Chip announcing whether a run is live, finished, or stopped."""
    label, cls = run_status(r, summary, live, is_newest)
    title = {
        "done":
        "info.log ends with main.py's completion line",
        "live":
        "a live main.py process holds this run's logs open, or a live "
        "Slurm job's stdout names this run dir, or this is the newest "
        "run of a live process or job that could not be pinned",
        "stopped":
        "no completion line in info.log, and no live process or Slurm "
        "job: killed or crashed",
    }[cls]
    return chip(label, cls, title)


def _test_chip(ep: Dict[str, Any]) -> str:
    """Chip for one test episode, e.g. "003 t1 ✓"."""
    mark, cls, title = test_mark(ep)
    label = f"{int(ep['num']):03} t{ep['task']}"
    if mark:
        label += " " + mark
    return chip(label, cls, title)


def _misc_chip(ep: Dict[str, Any]) -> str:
    """Chip for one non-test episode, e.g. "002 explore ✓ 0.70"."""
    label = f"{int(ep['num']):03} {ep['kind']}"
    title = ""
    if "env_accepted" in ep:
        mark, _, title = explore_mark(ep)
        label += " " + mark
    return chip(label, "kind-" + ep["kind"], title)


# ----------------------------------------------------------------- pages


def run_row(r: Dict[str, Any], summary: Dict[str, Any], layout: Dict[str, Any],
            live: LiveProcs, is_newest: bool) -> str:
    """Table row summarizing one run for the index page."""
    eps = summary.get("episodes", [])
    tr_str = test_results_str(summary) or "-"
    cost = summary.get("total_cost", 0.0)
    fmt = "%Y-%m-%d %H:%M"
    start_ts = _run_start_ts(r["name"], r["mtime"])
    sstr = datetime.datetime.fromtimestamp(start_ts).strftime(fmt)
    mstr = datetime.datetime.fromtimestamp(r["activity"]).strftime(fmt)
    dur_str = _fmt_duration(max(0.0, r["activity"] - start_ts))
    status, _ = run_status(r, summary, live, is_newest)
    # The status joins the filter key, so "running" narrows to live runs.
    key = f"{r['exp']} {r['seed']} {r['name']} {status}".lower()
    cost_str = f"${cost:.2f}" if cost else "-"
    # Path to paste into a terminal at the server's working directory.
    copy_path = os.path.relpath(os.path.join(LOGS_ROOT, r["rel"]))
    is_live = status == "running"
    esc_rel = esc(r["rel"])
    kill_btn = ""
    if is_live:
        kill_btn = ("<button class='rowbtn' title='Kill the live process "
                    "of this run, or scancel its Slurm job' "
                    f"onclick='killRun(\"{esc_rel}\")'>"
                    "kill</button>")
    del_title = ("Kill the live process, then delete this run"
                 if is_live else "Delete this run's log dir and videos")
    live_flag = "true" if is_live else "false"
    del_btn = (f"<button class='rowbtn' title='{del_title}' "
               f"onclick='deleteRun(\"{esc_rel}\", {live_flag})'>"
               "✕</button>")
    return ("<tr class='runrow' "
            f"data-key='{esc(key)}' data-seed='{esc(r['seed'])}' "
            f"data-start='{start_ts:.0f}'>"
            f"<td><input type='checkbox' class='cmp' value='{esc(r['rel'])}'>"
            "</td>"
            f"<td><a href='/run?d={q(r['rel'])}'>{esc(r['name'])}</a>"
            f"<button class='copybtn' data-copy='{esc(copy_path)}' "
            f"title='Copy run path'>⧉</button>{del_btn}</td>"
            f"<td>{esc(r['seed'])}</td>"
            f"<td>{status_chip(r, summary, live, is_newest)}{kill_btn}</td>"
            f"<td>{episode_grid(eps, layout)}</td>"
            f"<td>{esc(tr_str)}</td>"
            f"<td>{cost_str}</td>"
            f"<td class='muted'>{sstr}</td>"
            f"<td class='muted'>{mstr}</td>"
            f"<td class='muted'>{dur_str}</td></tr>")


def index_page() -> str:
    """Runs overview page grouped by family and experiment."""
    runs = find_runs()
    families: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in runs:
        fam, _, rest = r["exp"].partition("/")
        families.setdefault(fam, {}).setdefault(rest or fam, []).append(r)
    summaries = {r["rel"]: run_summary(r["rel"]) or {} for r in runs}
    # The task and round columns are laid out once for the whole page, so
    # a task's chips line up across runs, experiments, and families. The
    # explore/learn column sits to the right of every task column, so it
    # can stay per-family without costing any of that alignment - which
    # spares families that never explore its reserved width.
    page_layout = grid_layout(
        [s.get("episodes", []) for s in summaries.values()])
    live = live_runs(runs)
    # A process lsof could not pin owns the newest run of its experiment
    # and seed: the one it made at startup. Older runs of that key are
    # dead ancestors - unless lsof pinned a live process to them.
    newest: Dict[Tuple[str, str], Tuple[float, str]] = {}
    for r in runs:
        key = (r["exp"], r["seed"])
        ts = _run_start_ts(r["name"], r["mtime"])
        if ts >= newest.get(key, (0.0, ""))[0]:
            newest[key] = (ts, r["rel"])
    body = [
        "<div class='content' style='height:calc(100vh - 45px);"
        "overflow:auto'>"
    ]
    if not runs:
        body.append(
            f"<p>No run_* directories found under {esc(LOGS_ROOT)}.</p>")
    table_head = ("<tr><th></th><th>run</th><th>seed</th><th>status</th>"
                  "<th>episodes</th><th>test results (info.log)</th>"
                  "<th>cost</th><th>started</th><th>modified</th>"
                  "<th>time</th></tr>")
    for fam in sorted(families):
        exps = families[fam]
        fam_runs = [r for rs in exps.values() for r in rs]
        fam_misc = grid_layout(
            [summaries[r["rel"]].get("episodes", []) for r in fam_runs])
        layout = dict(page_layout, misc=fam_misc["misc"])
        widths = [w or grid_width(layout) for w in RUN_COL_W]
        cols = "<colgroup>" + "".join(f"<col style='width:{w}px'>"
                                      for w in widths) + "</colgroup>"
        body.append(f"<details class='grp family' data-key='fam:{esc(fam)}' "
                    f"open><summary>{esc(fam)} <span class='muted'>"
                    f"({len(exps)} experiments, {len(fam_runs)} runs)"
                    "</span></summary>")
        for expname in sorted(exps):
            rows = "".join(
                run_row(r, summaries[r["rel"]], layout, live, newest[(
                    r["exp"], r["seed"])][1] == r["rel"])
                for r in exps[expname])
            body.append(f"<details class='grp exp' data-key='exp:{esc(fam)}/"
                        f"{esc(expname)}'>"
                        f"<summary>{esc(expname)} <span class='muted'>"
                        f"({len(exps[expname])} runs)</span></summary>"
                        f"<table class='grid runs' "
                        f"style='width:{sum(widths)}px'>"
                        f"{cols}{table_head}{rows}</table></details>")
        body.append("</details>")
    body.append("</div>")
    topbar = ("<input id='runfilter' placeholder='filter runs…' "
              "oninput='filterRuns(this.value)'>"
              "<button id='sortbtn' onclick='toggleSort()' title='seed: "
              "group rows by seed, newest first within each seed; time: "
              "newest runs first within each experiment'></button>"
              "<button onclick='setAllGroups(true)'>expand all</button>"
              "<button onclick='setAllGroups(false)'>collapse all</button>"
              "<button onclick='compareSelected()'>Compare selected"
              f"</button><span class='crumb'>{esc(LOGS_ROOT)}</span>")
    return page("runs - log viewer", topbar, "".join(body))


def file_tree_html(run_abs: str, run_rel: str, rel: str = "") -> str:
    """Nested details tree; big image dirs collapse to a gallery link."""
    abs_dir = os.path.join(run_abs, rel) if rel else run_abs
    try:
        entries = sorted(os.listdir(abs_dir))
    except OSError:
        return ""
    dirs = [e for e in entries if os.path.isdir(os.path.join(abs_dir, e))]
    files = [e for e in entries if not os.path.isdir(os.path.join(abs_dir, e))]
    out = []
    for d in dirs:
        child_rel = (rel + "/" + d) if rel else d
        child_abs = os.path.join(abs_dir, d)
        try:
            child_entries = os.listdir(child_abs)
        except OSError:
            child_entries = []
        n_imgs = sum(1 for c in child_entries
                     if os.path.splitext(c)[1].lower() in IMG_EXTS)
        if n_imgs > 20 and n_imgs > len(child_entries) * 0.8:
            out.append(f"<div class='nav'><a href='#gallery={q(child_rel)}'>"
                       f"{esc(d)}/ <span class='muted'>"
                       f"({n_imgs} images, gallery)</span>"
                       "</a></div>")
            continue
        # Expand the top-level sandbox/ dir by default; it holds the
        # test_images and session logs users reach for most often.
        open_attr = " open" if not rel and d == "sandbox" else ""
        out.append(f"<details{open_attr}><summary>{esc(d)}/ "
                   f"<span class='muted'>"
                   f"({len(child_entries)})</span>"
                   "</summary>"
                   f"{file_tree_html(run_abs, run_rel, child_rel)}</details>")
    shown = files[:200]
    out.append("<div class='nav'>")
    for f in shown:
        child_rel = (rel + "/" + f) if rel else f
        out.append(f"<a href='#f={q(child_rel)}'>{esc(f)}</a>")
    if len(files) > 200:
        out.append(f"<span class='muted'>… {int(len(files) - 200)} "
                   "more files</span>")
    out.append("</div>")
    return "".join(out)


def run_page(run_rel: str) -> Optional[str]:
    """Single-run page: episode sidebar, file tree, and content pane."""
    run_abs = safe_join(run_rel)
    if not run_abs or not os.path.isdir(run_abs):
        return None
    summary = run_summary(run_rel)
    assert summary is not None
    eps = summary["episodes"]
    side = ["<div class='sidebar'>"]
    if eps:
        side.append("<h3>Episodes</h3><div class='nav'>")
        for ep in eps:
            mark, cls, title = "", "", ""
            if ep["kind"] == "test":
                mark, cls, title = test_mark(ep)
            elif "env_accepted" in ep:
                mark, cls, title = explore_mark(ep)
            cost = (f"  ${ep['solve_cost']:.2f}" if "solve_cost" in ep else "")
            task_str = (f" task{ep['task']}" if ep["task"] is not None else "")
            mark_str = " " + mark if mark else ""
            label = f"{ep['num']:03d} {ep['kind'] + task_str}{mark_str}"
            side.append(f"<a href='#f={q(ep['file'])}'>"
                        f"<span class='chip {cls}' title='{esc(title)}'>"
                        f"{esc(label)}</span>"
                        f"<span class='muted'>{esc(cost)}</span></a>")
        side.append("</div>")
    other_md = sorted(f for f in os.listdir(run_abs)
                      if f.endswith(".md") and not EPISODE_RE.match(f))
    logs = sorted(f for f in os.listdir(run_abs) if f.endswith(".log"))
    if other_md:
        side.append("<h3>Prompts</h3><div class='nav'>")
        side += [f"<a href='#f={q(f)}'>{esc(f)}</a>" for f in other_md]
        side.append("</div>")
    if logs:
        side.append("<h3>Logs</h3><div class='nav'>")
        side += [f"<a href='#f={q(f)}'>{esc(f)}</a>" for f in logs]
        side.append("</div>")
    side.append("<h3>All files</h3>")
    side.append(file_tree_html(run_abs, run_rel))
    side.append("</div>")
    default = q(eps[0]["file"]) if eps else ""
    tr_str = test_results_str(summary)
    results_span = ("test results: " + esc(tr_str)) if tr_str else ""
    ex_str = explore_results_str(summary)
    explore_span = ""
    if ex_str:
        explore_span = ("<span title='per-cycle explore (train) episodes: "
                        "accepted/total (mean env reward)'>explore: "
                        f"{esc(ex_str)}</span>")
    cost_span = (f"total cost: ${summary['total_cost']:.2f}"
                 if summary["total_cost"] else "")
    banner = ("<span title='per-cycle test phases: solved/total "
              f"(avg env reward)'>{results_span}</span>{explore_span}"
              f"<span>{cost_span}</span>")
    side_html = "".join(side)
    body = (f"<div class='layout'>{side_html}"
            "<div style='flex:1;display:flex;flex-direction:column;"
            "overflow:hidden'>"
            f"<div class='banner' style='margin:8px 16px 0'>{banner}"
            "<span style='margin-left:auto'>"
            "<button onclick='setAllDetails(true)'>expand all</button> "
            "<button onclick='setAllDetails(false)'>collapse all</button>"
            "</span></div>"
            f"<div class='content' id='content' data-run='{esc(run_rel)}' "
            f"data-default='{default}'><p class='muted'>Select a file.</p>"
            "</div></div></div>")
    topbar = f"<span class='crumb'>{esc(run_rel)}</span>"
    return page(run_rel + " - log viewer", topbar, body)


# ------------------------------------------------------------- fragments


def _cycle_tag(rnd: int) -> str:
    """The cycle tag main.py stamps into the video names of a test round.

    Round 0 is the pre-learning test phase, which main.py runs with
    online_learning_cycle=None; round r>0 is the test after learning
    cycle r-1, since main.py passes that loop's 0-based index.
    """
    return "cycleNone" if rnd == 0 else f"cycle{rnd - 1}"


def _video_base(run_rel: str) -> Optional[str]:
    """The on-disk video dir of a run, tolerating moved log dirs.

    videos/ mirrors logs/ by construction, but a log dir renamed or
    moved after the fact (e.g. experiment families reorganized by hand)
    leaves the mirror stale; the "Wrote out to" lines in the run's own
    info.log then name where its videos actually live.
    """
    base = safe_join_video(run_rel)
    if base and os.path.isdir(base):
        return base
    run_abs = safe_join(run_rel)
    if not run_abs:
        return None
    parsed = _cached(os.path.join(run_abs, "info.log"), _parse_info_log) or {}
    rel = parsed.get("video_rel", "")
    if rel:
        alt = safe_join_video(rel)
        if alt and os.path.isdir(alt):
            return alt
    return None


def video_url_rel(run_rel: str, name: str) -> str:
    """The /rawvideo path of one of a run's video files."""
    base = _video_base(run_rel)
    if base:
        rel = os.path.relpath(base, VIDEOS_ROOT).replace(os.sep, "/")
        if not rel.startswith(".."):
            return rel + "/" + name
    return run_rel + "/" + name


def videos_for_run(run_rel: str) -> VidMap:
    """Test videos of a run, from videos/<run_rel>/.

    Keyed by the 0-based task index used everywhere else in this viewer,
    so the 1-based number in the filename is converted here and nowhere
    else.
    """
    base = _video_base(run_rel)
    out: VidMap = {}
    if not base:
        return out
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for name in names:
        m = VIDEO_RE.search(name)
        if not m:
            # E.g. cogman videos, or the taskless interaction videos that
            # interaction_videos_for_run keys by cycle instead.
            continue
        key = (int(m.group(1)) - 1, "cycle" + m.group(3))
        out.setdefault(key, []).append((name, bool(m.group(2))))
    return out


def interaction_videos_for_run(
        run_rel: str) -> Dict[str, List[Tuple[Optional[int], str]]]:
    """Interaction videos of a run, keyed by cycle tag (e.g. "cycle0").

    These are the CFG.make_interaction_videos recordings of the cycle's
    explore episodes; they carry no __task part, unlike test videos.
    Each entry is (episode index within the cycle, filename), or (None,
    filename) for a legacy whole-cycle concatenation.
    """
    base = _video_base(run_rel)
    out: Dict[str, List[Tuple[Optional[int], str]]] = {}
    if not base:
        return out
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for name in names:
        if VIDEO_RE.search(name):
            continue  # a test video, keyed by task in videos_for_run
        m = INTERACTION_VIDEO_RE.search(name)
        if m:
            out.setdefault("cycle" + m.group(2), []).append(
                (int(m.group(1)), name))
            continue
        m = INTERACTION_CYCLE_VIDEO_RE.search(name)
        if m:
            out.setdefault("cycle" + m.group(1), []).append((None, name))
    return out


def _video_figure(url: str, label: str) -> str:
    """One video player with its caption and raw link."""
    return (f"<figure class='vid'><video controls preload='metadata' "
            f"src='{url}'></video><figcaption class='muted'>{label} · "
            f"<a class='muted' href='{url}' target='_blank'>raw</a>"
            f"</figcaption></figure>")


def episode_videos(ep: Dict[str, Any], run_rel: str) -> str:
    """Players for the env episode this transcript belongs to.

    main.py names a test video by task and learning cycle rather than by
    query, so the two transcripts of a replanned task (001/002
    ..._task0) are two views of one env episode and correctly share its
    video. An explore transcript gets the video of its own interaction
    episode, matched by index within the cycle; on legacy runs whose one
    interaction video concatenates the whole cycle, each of that cycle's
    explore transcripts shows that shared video.
    """
    out = []
    if ep["kind"] == "test" and ep["task"] is not None:
        key = (int(ep["task"]), _cycle_tag(int(ep.get("round", 0))))
        for name, is_failure in videos_for_run(run_rel).get(key, []):
            url = "/rawvideo?p=" + q(video_url_rel(run_rel, name))
            label = "failure video" if is_failure else "test video"
            out.append(_video_figure(url, label))
    elif ep["kind"] == "explore":
        # Interaction videos are stamped with the 0-based learning cycle
        # directly (no cycleNone offset like test rounds), and an explore
        # episode's round -- learn episodes seen before it -- IS its cycle.
        tag = f"cycle{int(ep.get('round', 0))}"
        for ep_idx, name in interaction_videos_for_run(run_rel).get(tag, []):
            if ep_idx is None:
                label = ("interaction video (all explore episodes "
                         "of this cycle)")
            elif ep_idx == ep.get("interaction_idx"):
                label = "interaction video"
            else:
                continue  # another explore session's episode
            url = "/rawvideo?p=" + q(video_url_rel(run_rel, name))
            out.append(_video_figure(url, label))
    return f"<div class='vids'>{''.join(out)}</div>" if out else ""


def episode_banner(ep: Dict[str, Any], n_rounds: int = 0) -> str:
    """Header banner summarizing an episode's verdict, cost, and turns."""
    parts = []
    if ep["kind"] == "test":
        if "env_solved" in ep:
            rtag = (f", reward {ep['env_reward']:.2f}"
                    if "env_reward" in ep else "")
            if ep["env_solved"]:
                parts.append(f"<span class='ok'>SOLVED ✓ (env eval{rtag})"
                             "</span>")
            else:
                parts.append("<span class='bad'>FAILED ✗ (env eval: "
                             f"{esc(ep.get('env_msg', ''))}{rtag})</span>")
        else:
            parts.append("<span class='muted'>no env verdict in info.log"
                         "</span>")
    elif ep["kind"] == "explore" and "env_accepted" in ep:
        reward = ep["env_reward"]
        if ep["env_accepted"]:
            parts.append("<span class='ok'>ACCEPTED ✓ (env eval: reward "
                         f"{reward:.2f})</span>")
        else:
            # Terminated-but-not-accepted means the env evaluator rejected
            # the episode for a rule violation; otherwise the goal was
            # simply never reached.
            verdict = ("REJECTED"
                       if ep.get("env_terminated") else "NOT SOLVED")
            parts.append(f"<span class='bad'>{verdict} ✗ (env eval: reward "
                         f"{reward:.2f})</span>")
        if ep.get("env_msg"):
            parts.append(f"<span class='muted'>{esc(ep['env_msg'])}</span>")
    if ep.get("goal") is True:
        parts.append("<span class='muted'>agent-reported: goal achieved"
                     "</span>")
    elif ep.get("goal") is False:
        parts.append("<span class='muted'>agent-reported: goal NOT achieved"
                     "</span>")
    if ep.get("task") is not None:
        parts.append(f"task {int(ep['task'])}")
    if n_rounds > 1 and "round" in ep:
        parts.append(f"round {int(ep['round'])}")
    parts.append(f"kind: {ep['kind']}")
    if "turns" in ep:
        parts.append(f"{int(ep['turns'])} turns")
    if "solve_cost" in ep:
        parts.append(f"${ep['solve_cost']:.2f} this solve / "
                     f"${ep['total_cost']:.2f} total")
    spans = "".join(f"<span>{p}</span>" for p in parts)
    banner = f"<div class='banner'>{spans}</div>"
    if ep.get("env_solved") is False and ep.get("goal") is True:
        banner += ("<div class='banner warn'>⚠ The transcript claims the "
                   "goal was achieved, but the env-side evaluation says "
                   "this task FAILED. Trust the env verdict.</div>")
    elif ep.get("env_solved") is True and ep.get("goal") is False:
        banner += ("<div class='banner warn'>⚠ The env-side evaluation "
                   "says SOLVED although the transcript reports the goal "
                   "as not achieved.</div>")
    return banner


def versions_nav(run_abs: str, file_rel: str) -> str:
    """Prev/diff links when the file sits in a *_versions directory."""
    d = os.path.dirname(file_rel)
    if not d.endswith("_versions"):
        return ""
    dir_abs = os.path.join(run_abs, d)
    try:
        siblings = sorted(os.listdir(dir_abs))
    except OSError:
        return ""
    name = os.path.basename(file_rel)
    if name not in siblings:
        return ""
    idx = siblings.index(name)
    links = []
    if idx > 0:
        links.append(f"<a href='#diff={q(file_rel)}'>"
                     f"diff vs {esc(siblings[idx - 1])}</a>")
        links.append(f"<a href='#f={q(d + '/' + siblings[idx - 1])}'>"
                     f"← {esc(siblings[idx - 1])}</a>")
    if idx < len(siblings) - 1:
        links.append(f"<a href='#f={q(d + '/' + siblings[idx + 1])}'>"
                     f"{esc(siblings[idx + 1])} →</a>")
    links_str = " &nbsp;|&nbsp; ".join(links)
    return (f"<div class='banner'>version {idx + 1}/{len(siblings)} &nbsp; "
            f"{links_str}</div>")


def view_fragment(run_rel: str, file_rel: str) -> str:
    """HTML fragment rendering one file (markdown, log, image, code, ...)."""
    run_abs = safe_join(run_rel)
    file_abs = safe_join(run_rel + "/" + file_rel)
    if not run_abs or not file_abs:
        return "<p>Not found.</p>"
    if os.path.isdir(file_abs):
        tree = file_tree_html(run_abs, run_rel, file_rel)
        return f"<h2>{esc(file_rel)}/</h2>{tree}"
    if not os.path.isfile(file_abs):
        return "<p>Not found.</p>"
    name = os.path.basename(file_rel)
    ext = os.path.splitext(name)[1].lower()
    raw_rel = os.path.relpath(file_abs, os.path.realpath(LOGS_ROOT))
    raw_url = "/raw?p=" + q(raw_rel.replace(os.sep, "/"))
    header = (f"<h2>{esc(file_rel)} <a class='muted' style='font-size:12px' "
              f"href='{raw_url}' target='_blank'>raw</a></h2>")
    if ext in IMG_EXTS:
        return header + (f"<img class='mdimg' style='max-width:96%'"
                         f"{size_attrs(file_abs)} src='{raw_url}'>")
    if ext in VIDEO_EXTS:
        return header + (f"<video controls style='max-width:96%' "
                         f"src='{raw_url}'></video>")
    if ext == ".md":
        banner = ""
        if EPISODE_RE.match(name):
            summary = run_summary(run_rel) or {}
            matches = [
                e for e in summary.get("episodes", []) if e["file"] == name
            ]
            if matches:
                banner = (episode_banner(matches[0],
                                         len(summary.get("rounds", []))) +
                          episode_videos(matches[0], run_rel))
        text, truncated = read_text(file_abs, max_bytes=4 * 1024 * 1024)
        note = ("<p class='muted'>(truncated to last 4MB)</p>"
                if truncated else "")
        return header + banner + note + render_md_document(text, run_rel)
    if ext == ".log" or name in ("info.log", "debug.log"):
        text, truncated = read_text(file_abs, max_bytes=4 * 1024 * 1024)
        note = ("<p class='muted'>(showing last 4MB)</p>" if truncated else "")
        lines = "".join(f"<span class='logline'>{ansi_to_html(l)}\n</span>"
                        for l in text.splitlines())
        return (header + note +
                "<input id='logfilter' placeholder='filter lines…' "
                "oninput='filterLog()' style='width:300px;padding:4px 10px;"
                "border:1px solid var(--border);border-radius:6px;"
                "background:var(--bg);color:var(--fg)'>" +
                f"<pre class='code logview' id='logview'>{lines}</pre>")
    if ext in CODE_EXTS or ext in TEXT_EXTS or ext == "":
        text, truncated = read_text(file_abs, max_bytes=4 * 1024 * 1024)
        note = ("<p class='muted'>(truncated to last 4MB)</p>"
                if truncated else "")
        extra = versions_nav(run_abs, file_rel)
        body = f"<pre class='code'><code>{esc(text)}</code></pre>"
        if ext == ".txt":
            body += thumbs_for_paths(text, run_rel)
        return header + extra + note + body
    size = os.path.getsize(file_abs)
    return header + (f"<p><a href='{raw_url}'>download</a> "
                     f"({size / 1024.0:.1f} KB)</p>")


def diff_fragment(run_rel: str, file_rel: str) -> str:
    """Unified diff of a versioned file against its previous sibling."""
    run_abs = safe_join(run_rel)
    file_abs = safe_join(run_rel + "/" + file_rel)
    if not run_abs or not file_abs or not os.path.isfile(file_abs):
        return "<p>Not found.</p>"
    d = os.path.dirname(file_rel)
    dir_abs = os.path.join(run_abs, d)
    siblings = sorted(os.listdir(dir_abs))
    name = os.path.basename(file_rel)
    idx = siblings.index(name) if name in siblings else 0
    if idx == 0:
        return "<p>No previous version to diff against.</p>"
    prev = siblings[idx - 1]
    old, _ = read_text(os.path.join(dir_abs, prev))
    new, _ = read_text(file_abs)
    diff = difflib.unified_diff(old.splitlines(),
                                new.splitlines(),
                                fromfile=prev,
                                tofile=name,
                                lineterm="")
    out = []
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            out.append(f"<span class='add'>{esc(line)}</span>")
        elif line.startswith("-") and not line.startswith("---"):
            out.append(f"<span class='del'>{esc(line)}</span>")
        elif line.startswith("@@"):
            out.append(f"<span class='hunk'>{esc(line)}</span>")
        else:
            out.append(esc(line) + "\n")
    header = (f"<h2>diff: {esc(prev)} → {esc(name)}</h2>"
              f"<p><a href='#f={q(d + '/' + prev)}'>view {esc(prev)}</a> "
              "&nbsp; "
              f"<a href='#f={q(file_rel)}'>view {esc(name)}</a></p>")
    if len(out) == 0:
        return header + "<p class='muted'>Files are identical.</p>"
    return header + f"<pre class='code diff'>{''.join(out)}</pre>"


def gallery_fragment(run_rel: str, dir_rel: str) -> str:
    """Grouped image gallery for a directory of screenshots."""
    dir_abs = safe_join(run_rel + "/" + dir_rel)
    if not dir_abs or not os.path.isdir(dir_abs):
        return "<p>Not found.</p>"
    root = os.path.realpath(LOGS_ROOT)
    images = sorted(f for f in os.listdir(dir_abs)
                    if os.path.splitext(f)[1].lower() in IMG_EXTS)
    groups: Dict[str, List[str]] = {}
    for f in images:
        m = re.match(r"^(iter\d+(?:_task\d+)?_test\d+|task\d+)", f)
        groups.setdefault(m.group(1) if m else "(other)", []).append(f)
    out = [
        f"<h2>{esc(dir_rel)}/ <span class='muted'>"
        f"({len(images)} images)</span></h2>"
    ]
    for gname in sorted(groups):
        imgs = groups[gname]
        cells = []
        for f in imgs:
            rel = os.path.relpath(os.path.join(dir_abs, f), root)
            url = "/raw?p=" + q(rel.replace(os.sep, "/"))
            cells.append(f"<a class='shot' href='{url}' target='_blank'>"
                         f"<img loading='lazy'"
                         f"{size_attrs(os.path.join(dir_abs, f))} "
                         f"src='{url}'><br>{esc(f)}</a>")
        open_attr = " open" if len(groups) == 1 else ""
        cells_html = "".join(cells)
        out.append(f"<details{open_attr}><summary>{esc(gname)} "
                   f"<span class='muted'>({len(imgs)})"
                   f"</span></summary><div class='gallery'>{cells_html}</div>"
                   "</details>")
    return "".join(out)


def compare_page(run_rels: List[str]) -> str:
    """Side-by-side comparison table across several runs."""
    summaries: List[Tuple[str, Dict[str, Any]]] = []
    for rel in run_rels:
        s = run_summary(rel)
        if s:
            summaries.append((rel, s))
    if len(summaries) < 2:
        return page(
            "compare", "", "<div class='content'>"
            "<p>Need at least two valid runs.</p></div>")
    all_tasks = sorted({
        ep["task"]
        for _, s in summaries for ep in s["episodes"]
        if ep["kind"] == "test" and ep["task"] is not None
    })
    head = "".join(
        f"<th><a href='/run?d={q(rel)}'>{esc(rel.split('/')[-1])}</a><br>"
        f"<span class='muted'>{esc('/'.join(rel.split('/')[:-1]))}</span></th>"
        for rel, _ in summaries)

    def _verdict_mark(v: Dict[str, Any]) -> str:
        mark = "✓" if v["solved"] else "✗"
        if v.get("reward") is not None:
            mark += f" {v['reward']:.2f}"
        cls = "ok" if v["solved"] else "bad"
        return f"<span class='{cls}' title='{esc(v['msg'])}'>{mark}</span>"

    rows = []
    for task in all_tasks:
        cells = []
        for _, s in summaries:
            verdicts = [r[task] for r in s.get("rounds", []) if task in r]
            if verdicts:
                marks = " → ".join(_verdict_mark(v) for v in verdicts)
            else:
                attempts = [
                    ep for ep in s["episodes"]
                    if ep["kind"] == "test" and ep["task"] == task
                ]
                mark_parts = []
                for ep in attempts:
                    if ep.get("goal"):
                        cls, mk = "ok", "(✓)"
                    elif ep.get("goal") is False:
                        cls, mk = "bad", "(✗)"
                    else:
                        cls, mk = "muted", "?"
                    mark_parts.append(f"<span class='{cls}'>{mk}</span>")
                marks = "".join(mark_parts)
                if marks:
                    marks += " <span class='muted'>agent-reported</span>"
            cells.append(f"<td>{marks or '-'}</td>")
        rows.append(f"<tr><th>task {int(task)}</th>{''.join(cells)}</tr>")

    def stat_row(label: str, fn: Callable[[Dict[str, Any]], str]) -> str:
        cols = "".join(f"<td>{fn(s)}</td>" for _, s in summaries)
        return f"<tr><th>{label}</th>{cols}</tr>"

    rows.append(
        stat_row("test results", lambda s: esc(test_results_str(s) or "-")))
    rows.append(
        stat_row("explore results",
                 lambda s: esc(explore_results_str(s) or "-")))
    rows.append(stat_row("episodes", lambda s: str(len(s["episodes"]))))
    rows.append(
        stat_row(
            "explore / learn", lambda s:
            f"{sum(1 for e in s['episodes'] if e['kind'] == 'explore')} / "
            f"{sum(1 for e in s['episodes'] if e['kind'] == 'learn')}"))
    rows.append(
        stat_row(
            "total cost", lambda s: (f"${s['total_cost']:.2f}")
            if s["total_cost"] else "-"))
    rows_html = "".join(rows)
    body = ("<div class='content' style='height:calc(100vh - 45px);"
            "overflow-y:auto'><h2>Run comparison</h2>"
            "<table class='grid'><tr><th></th>"
            f"{head}</tr>{rows_html}</table></div>")
    return page("compare - log viewer", "", body)


# ------------------------------------------------------------ HTTP layer


class Handler(BaseHTTPRequestHandler):
    """HTTP handler serving the log-viewer pages, fragments, and raw files."""

    def log_message(
            self,
            format: str,  # pylint: disable=redefined-builtin
            *args: Any) -> None:
        pass

    def send_html(self, html_text: str, status: int = 200) -> None:
        """Send an HTML response with the given status code."""
        data = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text: str, status: int = 200) -> None:
        """Send a plain-text response with the given status code."""
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_raw(self, path: str) -> None:
        """Stream a file to the client, honoring HTTP Range requests."""
        ctype = mimetypes.guess_type(path)[0]
        if not ctype or os.path.splitext(path)[1].lower() in (
            {".md", ".log", ".py"} | TEXT_EXTS):
            ctype = "text/plain; charset=utf-8"
        size = os.path.getsize(path)
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)$", range_header or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            else:
                start = max(size - int(m.group(2)), 0)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{int(size)}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range",
                             f"bytes {int(start)}-{int(end)}/{int(size)}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self) -> None:
        """Dispatch a POST (kill / delete) request with plain-text reply."""
        try:
            # A web page anywhere can fire a cross-origin POST at
            # localhost; destructive endpoints only honor same-origin.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host', '')}":
                self.send_text("cross-origin POST rejected", status=403)
                return
            parsed = urllib.parse.urlparse(self.path)
            params = {
                k: v[0]
                for k, v in urllib.parse.parse_qs(parsed.query).items()
            }
            if parsed.path == "/kill":
                ok, msg = kill_run(params.get("d", ""))
            elif parsed.path == "/delete":
                ok, msg = delete_run(params.get("d", ""),
                                     kill=params.get("kill") == "1")
            else:
                self.send_text("not found", status=404)
                return
            self.send_text(msg, status=200 if ok else 409)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # pylint: disable=broad-except
            traceback.print_exc()
            try:
                self.send_text(str(e), status=500)
            except OSError:
                pass

    def do_GET(self) -> None:
        """Dispatch a GET request, rendering a 500 page on errors."""
        try:
            self.route()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # pylint: disable=broad-except
            traceback.print_exc()
            try:
                self.send_html(f"<pre>{esc(str(e))}</pre>", status=500)
            except OSError:
                pass

    def route(self) -> None:
        """Map the request path to the matching page or fragment handler."""
        parsed = urllib.parse.urlparse(self.path)
        params = {
            k: v[0]
            for k, v in urllib.parse.parse_qs(parsed.query).items()
        }
        route = parsed.path
        if route == "/":
            self.send_html(index_page())
        elif route == "/run":
            html_text = run_page(params.get("d", ""))
            if html_text is None:
                self.send_html("<p>Run not found.</p>", status=404)
            else:
                self.send_html(html_text)
        elif route == "/view":
            self.send_html(
                view_fragment(params.get("d", ""), params.get("f", "")))
        elif route == "/diff":
            self.send_html(
                diff_fragment(params.get("d", ""), params.get("f", "")))
        elif route == "/gallery":
            self.send_html(
                gallery_fragment(params.get("d", ""), params.get("p", "")))
        elif route == "/compare":
            rels = [r for r in params.get("runs", "").split(";") if r]
            self.send_html(compare_page(rels))
        elif route == "/stamp":
            d = params.get("d", "")
            self.send_text(run_stamp(d) if d else index_stamp())
        elif route == "/raw":
            path = safe_join(params.get("p", ""))
            if not path or not os.path.isfile(path):
                self.send_html("<p>Not found.</p>", status=404)
            else:
                self.send_raw(path)
        elif route == "/rawvideo":
            path = safe_join_video(params.get("p", ""))
            if not path or not os.path.isfile(path):
                self.send_html("<p>Not found.</p>", status=404)
            else:
                self.send_raw(path)
        else:
            self.send_html("<p>Not found.</p>", status=404)


def main() -> None:
    """Parse args, set LOGS_ROOT, and serve the viewer until interrupted."""
    global LOGS_ROOT, VIDEOS_ROOT  # pylint: disable=global-statement
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--logs",
                        default="logs",
                        help="logs root directory (default: logs)")
    parser.add_argument("--videos",
                        default="videos",
                        help="videos root directory, laid out like --logs "
                        "(default: videos)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    LOGS_ROOT = os.path.abspath(args.logs)
    # Videos are optional: a run with none simply shows no player.
    VIDEOS_ROOT = os.path.abspath(args.videos)
    if not os.path.isdir(LOGS_ROOT):
        sys.exit(f"logs directory not found: {LOGS_ROOT}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {LOGS_ROOT} at http://{args.host}:{int(args.port)}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
