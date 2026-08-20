"""Sandbox directory scaffolding shared by the session managers.

Creates and populates the per-run sandbox (reference files, CLAUDE.md,
PreToolUse hook, git repo for Glob visibility) and owns the embedded
hook script. Prompt *text* lives in :mod:`sandbox_prompts`; this module
is filesystem and git.
"""
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from predicators.agent_sdk.tools.sandbox_guard import \
    SANDBOX_HIDDEN_MODULES_PATTERN, SANDBOX_INTROSPECTION, \
    SANDBOX_SYSTEM_ROOTS

logger = logging.getLogger(__name__)

# Timeout for the git calls that make sandbox files Glob-visible.
GIT_TIMEOUT_S = 5

# ---------------------------------------------------------------------------
# Hook script that blocks Read/Write/Edit/Glob/Grep outside the sandbox.
# Python imports (via the PYTHONPATH mount) are NOT affected by this hook
# since they go through the Python interpreter, not Claude's built-in tools.
# ---------------------------------------------------------------------------

# NOTE: raw string so the embedded regex backslashes (\\s, \\.) survive
# verbatim. The Bash screening mirrors ``tools._screen_text_for_sandbox
# _escape`` (which guards ``run_python``); the shared constants are
# injected below so the two guards cannot drift on WHAT they block, only
# on how they anchor it (Python source vs. shell command strings).
_VALIDATE_SANDBOX_TEMPLATE = r'''#!/usr/bin/env python3
"""Sandbox PreToolUse guard.

Blocks built-in file tools (Read/Write/Edit/Glob/Grep) whose target path
resolves outside the sandbox, and heuristically screens Bash commands for
out-of-sandbox reads / predicators-source introspection. Best effort: a
determined script can still escape (env vars, subprocess, computed paths);
OS-level isolation is the hard boundary. Kept dependency-free so it stays
cheap to run on every tool call.
"""
import json
import os
import re
import sys

SYSTEM_ROOTS = __SYSTEM_ROOTS__
INTROSPECTION = __INTROSPECTION__
PATH_RE = re.compile(r"""(?:^|(?<=[\s'"`(=]))((?:/|\.\.)[^\s'"`)<>|;:,]*)""")
# Hidden-implementation imports (e.g. inside `python -c`); the public
# authoring surface (predicators.structs / predicators.utils) stays
# importable. The module alternation is injected from tools.py.
HIDDEN_IMPORT_RE = re.compile(
    r"(?:^|[\s;])(?:from|import)\s+" + __HIDDEN_MODULES_PATTERN__)

data = json.load(sys.stdin)
tool_name = data.get("tool_name", "")
tool_input = data.get("tool_input", {})
sandbox = os.path.realpath(os.getcwd())


def within(path):
    resolved = os.path.realpath(
        path if os.path.isabs(path) else os.path.join(sandbox, path))
    return resolved == sandbox or resolved.startswith(sandbox + os.sep)


def deny(reason):
    json.dump({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }, sys.stdout)
    sys.exit(0)


# File-path tools: validate the single target path (empty -> cwd, allow).
if tool_name in ("Read", "Write", "Edit", "Glob", "Grep"):
    key = "file_path" if tool_name in ("Read", "Write", "Edit") else "path"
    file_path = tool_input.get(key, "")
    if file_path and not within(file_path):
        deny("Blocked: " + file_path +
             " resolves outside the sandbox directory")
    sys.exit(0)

# Bash: heuristically screen the command string for escapes.
if tool_name == "Bash":
    command = tool_input.get("command", "")
    for needle in INTROSPECTION:
        if needle in command:
            deny("Blocked: '" + needle + "' may read predicators source "
                 "outside the sandbox; use ./reference/ and the MCP tools")
    if HIDDEN_IMPORT_RE.search(command):
        deny("Blocked: importing predicators env/ground-truth modules "
             "would expose implementation the sandbox hides; use "
             "./reference/ and the MCP tools")
    for match in PATH_RE.finditer(command):
        token = match.group(1)
        if within(token):
            continue
        if token.startswith("/") and not any(
                token == r or token.startswith(r + "/")
                for r in SYSTEM_ROOTS):
            # Absolute but not a real filesystem path (printed data) — skip.
            continue
        deny("Blocked: path '" + token +
             "' in the command resolves outside the sandbox directory")
    sys.exit(0)

# Anything else: allow.
sys.exit(0)
'''

VALIDATE_SANDBOX_SCRIPT = (_VALIDATE_SANDBOX_TEMPLATE.replace(
    "__SYSTEM_ROOTS__", repr(SANDBOX_SYSTEM_ROOTS)).replace(
        "__INTROSPECTION__", repr(SANDBOX_INTROSPECTION)).replace(
            "__HIDDEN_MODULES_PATTERN__",
            repr(SANDBOX_HIDDEN_MODULES_PATTERN)))

SANDBOX_SETTINGS: Dict[str, Any] = {
    "hooks": {
        "PreToolUse": [{
            "matcher":
            "Read|Write|Edit|Glob|Grep|Bash",
            "hooks": [{
                "type": "command",
                "command": "python3 .claude/validate_sandbox.py",
            }],
        }]
    }
}


def find_repo_root() -> Path:
    """Return the repository root by locating ``setup.py`` upward."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "setup.py").exists():
            return parent
    raise RuntimeError(
        "Could not find predicators repo root: no setup.py found in any "
        f"parent of {__file__}")


def git_commit_all(sandbox_dir: str,
                   message: str,
                   paths: Optional[list] = None,
                   check: bool = False) -> None:
    """Stage ``paths`` (default: everything) and commit as the sandbox user.

    Claude Code indexes git-tracked files at session startup, so files
    must be committed before the session starts to be Glob-visible.
    """
    add_target = paths if paths is not None else ["-A"]
    subprocess.run(["git", "add", *add_target],
                   cwd=sandbox_dir,
                   capture_output=True,
                   timeout=GIT_TIMEOUT_S,
                   check=check)
    subprocess.run(
        [
            "git", "commit", "-q", "-m", message, "--author",
            "sandbox <sandbox@local>"
        ],
        cwd=sandbox_dir,
        capture_output=True,
        timeout=GIT_TIMEOUT_S,
        check=check,
        env={
            **os.environ, "GIT_COMMITTER_NAME": "sandbox",
            "GIT_COMMITTER_EMAIL": "sandbox@local"
        },
    )


def setup_sandbox_directory(
    sandbox_dir: str,
    repo_root: str,
    extra_reference_files: Dict[str, str],
    claude_md_content: str,
    system_prompt: str,
    log_dir: str,
    seed_scratchpad: bool = True,
    phase: Optional[str] = None,
) -> None:
    """Create and populate a sandbox directory for the agent.

    Sets up:
    - ``reference/`` with curated files copied from the host repo
    - ``CLAUDE.md`` with agent instructions
    - ``.claude/settings.json`` with PreToolUse hooks
    - ``.claude/validate_sandbox.py`` hook script
    - ``.git/`` marker so Claude CLI treats the sandbox as project root
    - ``session_logs/``, ``test_images/``, ``proposed_code/`` subdirectories
    - ``full_system_prompt[_{phase}].md`` in *log_dir* for easy inspection

    Args:
        sandbox_dir: Absolute path to the sandbox directory.
        repo_root: Absolute path to the predicators repository root.
        extra_reference_files: Mapping of destination paths (relative to
            ``sandbox/reference/``) to source paths (relative to repo root).
        claude_md_content: Content for the ``CLAUDE.md`` file.
        system_prompt: Full system prompt to log for inspection.
        log_dir: Directory for host-visible logs.
        phase: Optional phase tag (e.g. ``"solve"``, ``"synthesis"``). When
            provided, the logged prompt is suffixed so solve and synthesis
            prompts don't overwrite each other across phase switches.
    """
    os.makedirs(sandbox_dir, exist_ok=True)
    sandbox = Path(sandbox_dir)
    logger.info("Setting up sandbox directory: %s", sandbox_dir)

    # 1. Copy reference files from host repo
    registry = dict(extra_reference_files)
    ref_dir = sandbox / "reference"
    for dest_rel, src_rel in registry.items():
        src = Path(repo_root) / src_rel
        dest = ref_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(str(src), str(dest))
        else:
            logger.warning("Reference file not found: %s", src)

    # 2. Real git repo so Claude CLI treats sandbox as project root
    #    (A plain .git file isn't recognised; Glob/Read resolve from the
    #    project root, so we need an actual repo here.)
    git_dir = sandbox / ".git"
    need_initial_commit = False
    if not git_dir.is_dir():
        if git_dir.exists():
            git_dir.unlink()  # remove old marker file
        subprocess.run(["git", "init", "-q"], cwd=str(sandbox), check=True)
        need_initial_commit = True

    # 3. .claude/settings.json with PreToolUse hooks
    #    Every write below is explicitly utf-8: these strings carry em dashes,
    #    and bare write_text() encodes with the locale's preferred encoding,
    #    so on a C/POSIX locale the whole sandbox setup dies with
    #    "'ascii' codec can't encode character '—'" -- and because setup
    #    runs once per query, every solve attempt fails identically.
    claude_dir = sandbox / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "settings.json").write_text(
        json.dumps(SANDBOX_SETTINGS, indent=2) + "\n", encoding="utf-8")

    # 4. validate_sandbox.py hook script
    (claude_dir / "validate_sandbox.py").write_text(VALIDATE_SANDBOX_SCRIPT,
                                                    encoding="utf-8")

    # 5. CLAUDE.md
    (sandbox / "CLAUDE.md").write_text(claude_md_content, encoding="utf-8")

    # 6. Create subdirectories and seed files
    for subdir in ("session_logs", "test_images", "proposed_code"):
        (sandbox / subdir).mkdir(exist_ok=True)
    # Seed empty scratchpad if enabled
    if seed_scratchpad:
        notes_path = sandbox / "notes.md"
        if not notes_path.exists():
            notes_path.write_text("", encoding="utf-8")

    # 7. Log full system prompt to main log dir for easy inspection.
    #    Suffix with the phase tag when provided so solve and synthesis
    #    prompts don't overwrite each other across phase switches.
    os.makedirs(log_dir, exist_ok=True)
    prompt_filename = ("full_system_prompt.md"
                       if not phase else f"full_system_prompt_{phase}.md")
    with open(os.path.join(log_dir, prompt_filename), "w",
              encoding="utf-8") as f:
        f.write(system_prompt)

    # 8. Initial commit so files are git-tracked (Glob-visible).
    if need_initial_commit:
        git_commit_all(str(sandbox), "sandbox init", check=True)

    logger.info(
        "Sandbox directory ready: %d reference files copied",
        sum(1 for d, s in registry.items() if (Path(repo_root) / s).exists()),
    )
