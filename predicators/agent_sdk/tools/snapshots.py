"""Write-time versioned snapshots of agent-edited sandbox files."""
import hashlib
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Sim-learning tools ───────────────────────────────────────────


class _SnapshotTarget:  # pylint: disable=too-few-public-methods
    """One file to watch for write-time snapshots."""

    def __init__(
        self,
        live_file: str,
        versions_dir: str,
        artifact_name: str,
        cycle_index_provider: Callable[[], int],
    ) -> None:
        self.live_file = os.path.realpath(live_file)
        self.versions_dir = versions_dir
        self.artifact_name = artifact_name
        self.cycle_index_provider = cycle_index_provider


def make_write_snapshot_hook(
    targets: List[_SnapshotTarget],
    sandbox_dir: str,
) -> Callable[..., Any]:
    """Build a PostToolUse hook that snapshots target files on Write/Edit.

    The returned async callable matches the Claude Agent SDK's hook
    signature ``(hook_input, tool_use_id, hook_context) -> dict``. It
    fires after a successful Write / Edit / MultiEdit / NotebookEdit
    and, if the tool's ``file_path`` (resolved against ``sandbox_dir``)
    matches any target's ``live_file``, writes a new versioned snapshot
    (via :func:`finalize_versioned_snapshot`).

    Dedup-by-hash means a no-op Edit that produces identical content
    leaves no new file. Failures are swallowed — a snapshot hook
    failing should never break the agent's edit loop.
    """
    abs_sandbox = os.path.abspath(sandbox_dir)

    def _resolve(path: str) -> str:
        if os.path.isabs(path):
            return os.path.realpath(path)
        return os.path.realpath(os.path.join(abs_sandbox, path))

    target_by_path: Dict[str,
                         _SnapshotTarget] = {t.live_file: t
                                             for t in targets}

    async def _hook(hook_input: Any, _tool_use_id: Any,
                    _context: Any) -> Dict[str, Any]:
        try:
            tool_name = getattr(hook_input, "tool_name", None)
            if tool_name not in {"Write", "Edit", "MultiEdit"}:
                return {}
            tool_input = getattr(hook_input, "tool_input", None) or {}
            raw_path = tool_input.get("file_path")
            if not raw_path:
                return {}
            resolved = _resolve(raw_path)
            target = target_by_path.get(resolved)
            if target is None:
                return {}
            finalize_versioned_snapshot(
                target.live_file,
                target.versions_dir,
                cycle_idx=int(target.cycle_index_provider()),
                artifact_name=target.artifact_name,
            )
        except Exception:  # pylint: disable=broad-except
            # Never let a snapshot failure break the agent's edit loop.
            pass
        return {}

    return _hook


def finalize_versioned_snapshot(
    live_file: str,
    versions_dir: str,
    cycle_idx: int,
    artifact_name: str,
) -> Optional[str]:
    """Take a final ``cycle_XXX_vers_(YYY+1)`` snapshot if needed.

    Called from the approach after the agent session ends so that any
    post-evaluation edits to ``live_file`` (which would otherwise be
    lost — the synthesis tools only snapshot on eval calls) are
    captured. If the live file's hash matches the highest existing
    ``cycle_XXX_vers_YYY_<artifact_name>.py`` in ``versions_dir`` (this
    cycle), the existing tag is returned and no new file is written.

    Args:
        live_file: Host path to the file (e.g. simulator.py).
        versions_dir: Directory containing the per-call snapshots.
        cycle_idx: Current cycle (1-indexed) — used to find the highest
            existing ``vers_YYY`` for this cycle and to name the new
            snapshot.
        artifact_name: Stem used in the filename, e.g. ``"simulator"``
            or ``"predicates"``.

    Returns the final version tag (``cycle_XXX_vers_YYY``) or ``None``
    if ``live_file`` does not exist.
    """
    if not os.path.isfile(live_file):
        return None
    with open(live_file, "rb") as f:
        live_raw = f.read()
    live_digest = hashlib.sha256(live_raw).hexdigest()

    prefix = f"cycle_{cycle_idx:03d}_vers_"
    suffix = f"_{artifact_name}.py"
    highest_vers = 0
    highest_path: Optional[str] = None
    if os.path.isdir(versions_dir):
        for name in os.listdir(versions_dir):
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            vers_str = name[len(prefix):-len(suffix)]
            try:
                vers = int(vers_str)
            except ValueError:
                continue
            if vers > highest_vers:
                highest_vers = vers
                highest_path = os.path.join(versions_dir, name)

    if highest_path is not None:
        with open(highest_path, "rb") as f:
            existing_digest = hashlib.sha256(f.read()).hexdigest()
        if existing_digest == live_digest:
            return f"cycle_{cycle_idx:03d}_vers_{highest_vers:03d}"

    os.makedirs(versions_dir, exist_ok=True)
    new_vers = highest_vers + 1
    snap_path = os.path.join(
        versions_dir,
        f"cycle_{cycle_idx:03d}_vers_{new_vers:03d}_{artifact_name}.py")
    with open(snap_path, "wb") as f:
        f.write(live_raw)
    return f"cycle_{cycle_idx:03d}_vers_{new_vers:03d}"


class _ArtifactSnapshotter:
    """Per-call versioned snapshotting for one artifact file.

    Used by the synthesis-tools factories to dedup snapshots by SHA256
    and tag each load with ``cycle_XXX_vers_YYY``. ``YYY`` is per
    instance and starts at 0 — it resets each time a new snapshotter is
    created (typically once per factory call). ``XXX`` is read from
    ``cycle_index_provider`` at each call so live cycle bumps are
    reflected in subsequent tags.
    """

    def __init__(
        self,
        live_file: str,
        versions_dir: str,
        artifact_name: str,
        cycle_index_provider: Optional[Callable[[], int]],
        missing_file_hint: str = "",
    ) -> None:
        self._live_file = live_file
        self._versions_dir = versions_dir
        self._artifact_name = artifact_name
        self._cycle_index_provider = cycle_index_provider
        self._missing_file_hint = missing_file_hint
        self._version_count = 0
        self._last_digest: Optional[str] = None

    def current_cycle(self) -> int:
        """Return the active learning-cycle index, or 0 if unknown."""
        if self._cycle_index_provider is None:
            return 0
        try:
            return int(self._cycle_index_provider())
        except Exception:  # pylint: disable=broad-except
            return 0

    def snapshot(
        self,
        path: Optional[str] = None,
    ) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        """Read the live file and write a versioned snapshot on change.

        Returns ``(raw_bytes, version_tag, error_msg)``. On a missing
        file, ``raw_bytes`` and ``version_tag`` are ``None`` and
        ``error_msg`` carries a user-facing message (suffixed with
        ``missing_file_hint`` when configured).

        ``path`` may override the configured ``live_file`` per call —
        the snapshotter still writes into the configured
        ``versions_dir`` under ``artifact_name``, sharing the version
        counter and digest cache so dedup spans both files.
        """
        target = path or self._live_file
        if not os.path.isfile(target):
            msg = (f"{self._artifact_name.capitalize()} file not found: "
                   f"{target}.")
            if self._missing_file_hint:
                msg = f"{msg} {self._missing_file_hint}"
            return None, None, msg
        with open(target, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()
        cycle_idx = self.current_cycle()
        if digest != self._last_digest:
            self._version_count += 1
            os.makedirs(self._versions_dir, exist_ok=True)
            snap_path = os.path.join(
                self._versions_dir, f"cycle_{cycle_idx:03d}_vers_"
                f"{self._version_count:03d}_{self._artifact_name}.py")
            with open(snap_path, "wb") as f:
                f.write(raw)
            self._last_digest = digest
        return raw, (f"cycle_{cycle_idx:03d}_vers_"
                     f"{self._version_count:03d}"), None
