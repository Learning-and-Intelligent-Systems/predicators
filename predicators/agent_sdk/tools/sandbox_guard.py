"""Sandbox-escape screening for agent-supplied text."""
import os
import re
from typing import Optional

# Filesystem roots that, when they prefix an absolute path outside the
# sandbox, mark it as a real escape (vs. data like "/done" printed by code).
SANDBOX_SYSTEM_ROOTS = (
    "/Users",
    "/home",
    "/root",
    "/etc",
    "/usr",
    "/opt",
    "/var",
    "/private",
    "/tmp",
    "/bin",
    "/sbin",
    "/lib",
    "/sys",
    "/proc",
    "/dev",
    "/mnt",
    "/srv",
)
# Predicators-source introspection that reaches outside the sandbox. The
# bare ``getsource`` substring also covers ``getsourcefile`` /
# ``getsourcelines``; ``inspect.getfile`` is matched explicitly so the
# generic ``getfilesystemencoding`` etc. don't false-positive.
SANDBOX_INTROSPECTION = ("getsource", "inspect.getfile", "site-packages")
# Hidden-implementation predicators imports inside agent-executed
# Python: the exec runs in-process, so ``from predicators.envs... import
# ...`` would hand the agent the real env classes and ground-truth
# constants the sandbox deliberately hides (observed in
# run_20260717_182040 seed1 turn 22). Public authoring surfaces
# (``predicators.structs``, ``predicators.utils``) stay importable -
# agent-written simulator.py code depends on them.
# The module alternation is shared with the sandbox hook script (see
# sandbox_setup.VALIDATE_SANDBOX_SCRIPT), which anchors it differently
# for shell-command strings; sharing the alternation keeps the two
# guards covering the same modules by construction.
SANDBOX_HIDDEN_MODULES_PATTERN = r"predicators\.(?:envs|ground_truth_models)\b"
_SANDBOX_PREDICATORS_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+" + SANDBOX_HIDDEN_MODULES_PATTERN, re.MULTILINE)
# Host filesystem prefixes scrubbed from tracebacks surfaced to agents:
# absolute source paths both leak the layout and invite out-of-sandbox
# probing (an audited run responded to one with ``find /``).
_HOST_REPO_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", ".."))


def _scrub_host_paths(text: str) -> str:
    """Strip host-absolute path prefixes from traceback text."""
    text = text.replace(_HOST_REPO_ROOT + os.sep, "")
    home = os.path.expanduser("~")
    if home and home != "/":
        text = text.replace(home + os.sep, "~" + os.sep)
    return text


# Path-like tokens: absolute (``/foo``) or parent-traversal (``..``/``../foo``),
# anchored at a boundary (start, whitespace, quote, ``(`` or ``=``) so we skip
# ``/`` inside URLs (preceded by ``:``), division, and ``./relative`` paths
# (which stay inside the sandbox).
_SANDBOX_PATH_RE = re.compile(
    r"""(?:^|(?<=[\s'"`(=]))((?:/|\.\.)[^\s'"`)<>|;:,]*)""")


def _screen_text_for_sandbox_escape(text: str,
                                    sandbox_dir: str) -> Optional[str]:
    """Best-effort screen of a Bash command / ``run_python`` code string.

    Returns a short deny reason if ``text`` looks like it reads outside
    ``sandbox_dir`` — an absolute or ``..`` path resolving out of the
    sandbox, or predicators-source introspection — else ``None``.

    This is a heuristic: a determined script can still escape (env vars,
    ``subprocess``, computed paths), so OS-level isolation (the docker
    sandbox) remains the only hard boundary. The equivalent self-contained
    logic for Bash lives in ``sandbox_setup.VALIDATE_SANDBOX_SCRIPT``;
    keep the two in sync.
    """
    for needle in SANDBOX_INTROSPECTION:
        if needle in text:
            return (f"'{needle}' may read predicators source outside the "
                    "sandbox; use the MCP tools and ./reference/ files")
    if _SANDBOX_PREDICATORS_IMPORT_RE.search(text):
        return ("importing predicators env/ground-truth modules would "
                "expose implementation the sandbox hides; use the "
                "provided namespace and the MCP tools / ./reference/ "
                "files")
    sandbox = os.path.realpath(sandbox_dir)
    for match in _SANDBOX_PATH_RE.finditer(text):
        token = match.group(1)
        resolved = os.path.realpath(
            token if os.path.isabs(token) else os.path.join(sandbox, token))
        if resolved == sandbox or resolved.startswith(sandbox + os.sep):
            continue
        if token.startswith("/") and not any(
                token == root or token.startswith(root + "/")
                for root in SANDBOX_SYSTEM_ROOTS):
            # Absolute but not a real filesystem path (e.g. printed data
            # like "/done") — don't flag.
            continue
        return f"path '{token}' resolves outside the sandbox directory"
    return None
