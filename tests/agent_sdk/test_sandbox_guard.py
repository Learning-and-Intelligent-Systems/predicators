"""Tests for the sandbox escape guard and oversize-output spilling.

Covers ``_screen_text_for_sandbox_escape`` (used in-process by
``run_python``), ``_make_spilling_text_result`` (oversize tool output
spilled into the sandbox instead of dumped to ``~/.claude/...``), and
the self-contained ``VALIDATE_SANDBOX_SCRIPT`` Bash/file-path hook.
"""
# pylint: disable=protected-access
from __future__ import annotations

import json
import subprocess
import sys

from predicators.agent_sdk.sandbox_setup import VALIDATE_SANDBOX_SCRIPT
from predicators.agent_sdk.tools.results import _make_spilling_text_result
from predicators.agent_sdk.tools.sandbox_guard import \
    _screen_text_for_sandbox_escape

# (text, should_block) — exercised against both the in-process screen and
# the live hook script (as a Bash command).
_ALLOW = [
    "from predicators.structs import State",
    "python3 my_experiment.py",
    "exec(open('./simulator.py').read())",
    'print("loading...")',
    "x = 10 / 2",
    'grep -rn "pattern" ./session_logs',
    'print("/done")',  # absolute-looking but not a real system root
    'print("see https://example.com/a/b")',  # URL, not a path
]
_BLOCK = [
    "import inspect; inspect.getsource(State)",
    "inspect.getfile(State)",
    'open("/etc/passwd").read()',
    "cat /Users/me/.claude/projects/x/tool-results/y.txt",
    "open('../../secret.txt')",
    "grep x /usr/lib/python3/site-packages/foo.py",
    # Hidden-implementation imports (run_20260717_182040 seed1 turn 22
    # pulled real domino constants this way); the public authoring
    # surface (predicators.structs, in _ALLOW) stays importable.
    "from predicators.envs.pybullet_domino import PyBulletDominoEnv",
    "import predicators.ground_truth_models.domino as gt",
]


def test_screen_allows_in_sandbox_paths(tmp_path) -> None:
    """Legitimate relative/in-sandbox code is not flagged."""
    sandbox = str(tmp_path)
    inside = tmp_path / "reference" / "structs.py"
    assert _screen_text_for_sandbox_escape(f"open('{inside}')",
                                           sandbox) is None
    for text in _ALLOW:
        assert _screen_text_for_sandbox_escape(text, sandbox) is None, text


def test_screen_blocks_escapes(tmp_path) -> None:
    """Absolute/``..`` escapes and source introspection are flagged."""
    sandbox = str(tmp_path)
    for text in _BLOCK:
        assert _screen_text_for_sandbox_escape(text, sandbox) is not None, text


def _run_hook(tmp_path, tool_name, tool_input) -> bool:
    """Run the generated hook script; return True if it denied the call."""
    script = tmp_path / "validate_sandbox.py"
    script.write_text(VALIDATE_SANDBOX_SCRIPT)
    payload = json.dumps({
        "tool_name": tool_name,
        "tool_input": tool_input,
    })
    out = subprocess.run([sys.executable, "validate_sandbox.py"],
                         cwd=str(tmp_path),
                         input=payload,
                         capture_output=True,
                         text=True,
                         check=True)
    return '"deny"' in out.stdout


def test_hook_screens_bash(tmp_path) -> None:
    """The hook script blocks escaping Bash commands and allows safe ones."""
    for text in _ALLOW:
        assert not _run_hook(tmp_path, "Bash", {"command": text}), text
    for text in _BLOCK:
        assert _run_hook(tmp_path, "Bash", {"command": text}), text


def test_hook_validates_file_paths(tmp_path) -> None:
    """The hook blocks file tools targeting paths outside the sandbox."""
    inside = str(tmp_path / "x.txt")
    assert not _run_hook(tmp_path, "Read", {"file_path": inside})
    assert not _run_hook(tmp_path, "Read", {"file_path": "./y.txt"})
    assert _run_hook(tmp_path, "Read", {"file_path": "/etc/passwd"})
    assert _run_hook(tmp_path, "Grep", {"path": "/usr/lib"})
    # Tools the hook does not screen are allowed through.
    assert not _run_hook(tmp_path, "WebFetch", {"url": "http://x"})


def test_spilling_inline_small_and_spills_large(tmp_path) -> None:
    """Small output stays inline; large output spills into the sandbox."""
    text = _make_spilling_text_result(str(tmp_path))
    small = text("hello")
    assert small == {"content": [{"type": "text", "text": "hello"}]}
    assert not (tmp_path / "tool_outputs").exists()

    big = "\n".join(f"line {i} " + "x" * 100 for i in range(2000))
    res = text(big)
    preview = res["content"][0]["text"]
    assert "output too large to inline" in preview
    assert "./tool_outputs/result_0001.txt" in preview
    spilled = tmp_path / "tool_outputs" / "result_0001.txt"
    assert spilled.exists() and spilled.read_text() == big
    # Counter advances on the next oversize result.
    assert "result_0002.txt" in text(big)["content"][0]["text"]


def test_spilling_noop_without_sandbox() -> None:
    """With no sandbox dir, output is always returned inline."""
    text = _make_spilling_text_result(None)
    big = "z" * 50000
    assert text(big) == {"content": [{"type": "text", "text": big}]}
