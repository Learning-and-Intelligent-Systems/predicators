"""Tests for versioned-snapshot helpers in ``predicators.agent_sdk.tools``.

Covers two pieces of plumbing introduced for the file-driven simulator /
predicates synthesis pipeline:

* ``finalize_versioned_snapshot`` — the "take one more snapshot if the
  live file changed" helper run after the agent session closes.
* ``make_write_snapshot_hook`` — the PostToolUse hook that snapshots
  ``simulator.py`` / ``predicates.py`` after every Write/Edit/MultiEdit.

Both are pure-Python and side-effect on the filesystem only; no agent
SDK calls are made.
"""
# pylint: disable=protected-access,unused-import
import asyncio
from types import SimpleNamespace

# Bootstrap circular imports before pulling from predicators.agent_sdk.
import predicators.utils  # noqa: F401 — required for import side effects
from predicators.agent_sdk.tools.snapshots import _SnapshotTarget, \
    finalize_versioned_snapshot, make_write_snapshot_hook

# ── finalize_versioned_snapshot ──────────────────────────────────────


def test_finalize_versioned_snapshot_missing_live_file(tmp_path):
    """Returns ``None`` and writes nothing when the live file is absent."""
    versions = tmp_path / "simulator_versions"
    versions.mkdir()
    tag = finalize_versioned_snapshot(
        str(tmp_path / "simulator.py"),
        str(versions),
        cycle_idx=1,
        artifact_name="simulator",
    )
    assert tag is None
    assert not list(versions.iterdir())


def test_finalize_versioned_snapshot_creates_first_snapshot(tmp_path):
    """First call writes ``cycle_001_vers_001`` and returns its tag."""
    live = tmp_path / "simulator.py"
    versions = tmp_path / "simulator_versions"
    live.write_text("# v1\n")
    tag = finalize_versioned_snapshot(str(live),
                                      str(versions),
                                      cycle_idx=1,
                                      artifact_name="simulator")
    assert tag == "cycle_001_vers_001"
    snapshots = sorted(p.name for p in versions.iterdir())
    assert snapshots == ["cycle_001_vers_001_simulator.py"]
    assert (versions /
            "cycle_001_vers_001_simulator.py").read_text() == "# v1\n"


def test_finalize_versioned_snapshot_dedup_on_unchanged_file(tmp_path):
    """A no-op finalize on unchanged content reuses the prior tag."""
    live = tmp_path / "predicates.py"
    versions = tmp_path / "predicates_versions"
    live.write_text("LEARNED_PREDICATES = []\n")
    first = finalize_versioned_snapshot(str(live),
                                        str(versions),
                                        cycle_idx=2,
                                        artifact_name="predicates")
    second = finalize_versioned_snapshot(str(live),
                                         str(versions),
                                         cycle_idx=2,
                                         artifact_name="predicates")
    assert first == second == "cycle_002_vers_001"
    assert len(list(versions.iterdir())) == 1


def test_finalize_versioned_snapshot_bumps_on_change(tmp_path):
    """Changed content increments ``vers_YYY`` within the same cycle."""
    live = tmp_path / "simulator.py"
    versions = tmp_path / "simulator_versions"
    live.write_text("# v1\n")
    finalize_versioned_snapshot(str(live),
                                str(versions),
                                cycle_idx=1,
                                artifact_name="simulator")
    live.write_text("# v2\n")
    tag = finalize_versioned_snapshot(str(live),
                                      str(versions),
                                      cycle_idx=1,
                                      artifact_name="simulator")
    assert tag == "cycle_001_vers_002"
    names = sorted(p.name for p in versions.iterdir())
    assert names == [
        "cycle_001_vers_001_simulator.py",
        "cycle_001_vers_002_simulator.py",
    ]


def test_finalize_versioned_snapshot_new_cycle_restarts_vers_yyy(tmp_path):
    """A new cycle starts at ``vers_001`` even when other cycles populated the
    same directory."""
    live = tmp_path / "simulator.py"
    versions = tmp_path / "simulator_versions"
    live.write_text("# v1\n")
    finalize_versioned_snapshot(str(live),
                                str(versions),
                                cycle_idx=1,
                                artifact_name="simulator")
    # Mutate and finalize as cycle 2; same content as cycle 1 still gets
    # a fresh cycle_002 entry because cycle 2 has no prior snapshots.
    live.write_text("# v2\n")
    tag = finalize_versioned_snapshot(str(live),
                                      str(versions),
                                      cycle_idx=2,
                                      artifact_name="simulator")
    assert tag == "cycle_002_vers_001"
    names = sorted(p.name for p in versions.iterdir())
    assert names == [
        "cycle_001_vers_001_simulator.py",
        "cycle_002_vers_001_simulator.py",
    ]


def test_finalize_versioned_snapshot_other_artifact_ignored(tmp_path):
    """Existing files for a *different* ``artifact_name`` don't influence the
    version count."""
    live = tmp_path / "predicates.py"
    versions = tmp_path / "shared_versions"
    versions.mkdir()
    # Sibling simulator snapshot for the same cycle — must not affect
    # the predicates counter.
    (versions / "cycle_001_vers_007_simulator.py").write_text("sim")
    live.write_text("preds")
    tag = finalize_versioned_snapshot(str(live),
                                      str(versions),
                                      cycle_idx=1,
                                      artifact_name="predicates")
    assert tag == "cycle_001_vers_001"
    assert (versions / "cycle_001_vers_001_predicates.py").exists()


# ── make_write_snapshot_hook ────────────────────────────────────────


def _run_hook(hook, tool_name, file_path):
    """Synchronously invoke the async hook with a mocked hook_input."""
    hook_input = SimpleNamespace(tool_name=tool_name,
                                 tool_input={"file_path": file_path})
    return asyncio.run(hook(hook_input, None, None))


def _make_hook(tmp_path, cycle_idx=1):
    sandbox = tmp_path
    sim = sandbox / "simulator.py"
    preds = sandbox / "predicates.py"
    sim_vd = sandbox / "simulator_versions"
    preds_vd = sandbox / "predicates_versions"
    targets = [
        _SnapshotTarget(str(sim), str(sim_vd), "simulator", lambda: cycle_idx),
        _SnapshotTarget(str(preds), str(preds_vd), "predicates",
                        lambda: cycle_idx),
    ]
    return make_write_snapshot_hook(targets, sandbox_dir=str(sandbox)), {
        "sim": sim,
        "preds": preds,
        "sim_vd": sim_vd,
        "preds_vd": preds_vd,
    }


def test_write_hook_snapshots_simulator_on_write(tmp_path):
    """Write tool with the simulator path produces a new snapshot."""
    hook, paths = _make_hook(tmp_path)
    paths["sim"].write_text("# rules\n")
    _run_hook(hook, "Write", "./simulator.py")
    snapshots = sorted(p.name for p in paths["sim_vd"].iterdir())
    assert snapshots == ["cycle_001_vers_001_simulator.py"]


def test_write_hook_ignores_unrelated_tools(tmp_path):
    """Read / Bash / Grep firing on the simulator path don't snapshot."""
    hook, paths = _make_hook(tmp_path)
    paths["sim"].write_text("# rules\n")
    for tool in ("Read", "Bash", "Grep", "Glob", "NotebookEdit"):
        _run_hook(hook, tool, "./simulator.py")
    assert not paths["sim_vd"].exists() or not list(paths["sim_vd"].iterdir())


def test_write_hook_dedup_on_no_op_edit(tmp_path):
    """Edit producing identical content does not append a new snapshot."""
    hook, paths = _make_hook(tmp_path)
    paths["sim"].write_text("body\n")
    _run_hook(hook, "Write", "./simulator.py")
    _run_hook(hook, "Edit", "./simulator.py")
    _run_hook(hook, "MultiEdit", "./simulator.py")
    snapshots = list(paths["sim_vd"].iterdir())
    assert len(snapshots) == 1


def test_write_hook_resolves_absolute_and_relative_paths(tmp_path):
    """A relative ``./predicates.py`` and an absolute path resolve to the same
    target — both trigger snapshots, but dedup means only one file."""
    hook, paths = _make_hook(tmp_path)
    paths["preds"].write_text("LEARNED_PREDICATES = []\n")
    _run_hook(hook, "Write", "./predicates.py")
    _run_hook(hook, "Edit", str(paths["preds"]))  # same content, absolute
    snapshots = list(paths["preds_vd"].iterdir())
    assert len(snapshots) == 1
    assert snapshots[0].name == "cycle_001_vers_001_predicates.py"


def test_write_hook_ignores_files_outside_target_list(tmp_path):
    """A write to some random file in the sandbox does not snapshot."""
    hook, paths = _make_hook(tmp_path)
    other = tmp_path / "scratch.py"
    other.write_text("print('hi')\n")
    _run_hook(hook, "Write", "./scratch.py")
    assert not paths["sim_vd"].exists() or not list(paths["sim_vd"].iterdir())
    assert (not paths["preds_vd"].exists()
            or not list(paths["preds_vd"].iterdir()))


def test_write_hook_swallows_exceptions(tmp_path):
    """A snapshot failure must not propagate — hooks failing should never break
    the agent's edit loop."""
    hook, _paths = _make_hook(tmp_path)
    # Missing file_path is one quiet failure path; a non-string is another.
    hook_input = SimpleNamespace(tool_name="Write", tool_input={})
    asyncio.run(hook(hook_input, None, None))
    hook_input = SimpleNamespace(tool_name="Edit", tool_input=None)
    asyncio.run(hook(hook_input, None, None))
    # Inputs that look valid but the snapshot helper trips on (unwritable
    # versions dir) should also not raise — point a target at a path that
    # cannot be created, fire the hook, expect no exception.
    bad_target = _SnapshotTarget(
        live_file=str(tmp_path / "simulator.py"),
        versions_dir="/dev/null/cannot/create",
        artifact_name="simulator",
        cycle_index_provider=lambda: 1,
    )
    bad_hook = make_write_snapshot_hook([bad_target],
                                        sandbox_dir=str(tmp_path))
    (tmp_path / "simulator.py").write_text("body")
    asyncio.run(
        bad_hook(
            SimpleNamespace(tool_name="Write",
                            tool_input={"file_path": "./simulator.py"}),
            None,
            None,
        ))


def test_write_hook_uses_cycle_provider_at_call_time(tmp_path):
    """The cycle index is read each time the hook fires, not captured up front,
    so consecutive cycles land in different filenames."""
    sandbox = tmp_path
    sim = sandbox / "simulator.py"
    sim_vd = sandbox / "simulator_versions"
    cycle = [1]
    target = _SnapshotTarget(str(sim), str(sim_vd), "simulator",
                             lambda: cycle[0])
    hook = make_write_snapshot_hook([target], sandbox_dir=str(sandbox))

    sim.write_text("# c1\n")
    _run_hook(hook, "Write", "./simulator.py")
    cycle[0] = 2
    sim.write_text("# c2\n")
    _run_hook(hook, "Edit", "./simulator.py")

    snapshots = sorted(p.name for p in sim_vd.iterdir())
    assert snapshots == [
        "cycle_001_vers_001_simulator.py",
        "cycle_002_vers_001_simulator.py",
    ]
