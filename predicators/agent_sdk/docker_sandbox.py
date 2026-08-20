"""Docker-sandboxed agent session manager.

Runs ``ClaudeSDKClient`` inside a Docker container so that the agent's
built-in tools (Bash, Read, Write, Edit, Glob, Grep, Task*) all execute
in an isolated environment.  Custom predicator MCP tools are created in-process
inside the container via the same ``create_mcp_tools()`` code used on
the host.

The host predicators source tree is mounted read-only at
``/opt/predicators`` for Python imports (``PYTHONPATH``).  PreToolUse
hooks block the agent's built-in tools (Read, Write, Edit, Glob, Grep)
from accessing anything outside ``/sandbox/``, so the agent cannot
browse environment source code or ground truth models directly.  Curated
reference files are copied into ``/sandbox/reference/`` for the agent to
read.  The agent can write and run Python scripts in ``/sandbox/``, and
``from predicators.structs import State`` works via the mount.

Shared data (pickled context and results) passes through ``/data``.

Behavioral notes relative to the shared base
(:mod:`predicators.agent_sdk.session_base`):

- ``query()`` is a subprocess orchestrator: each call runs one fresh
  container (no persistent client), so ``start_session``, ``close``,
  and ``_recover_session`` are no-ops.
- The incremental markdown log is written in-container; the host only
  prepends a metadata header afterwards.
- Cost accounting reuses the base delta scheme with the baseline reset
  to zero per query, since every container session starts from zero.

Usage
-----
When the ``agent_sdk_use_docker_sandbox`` flag is ``True``, the
``AgentSessionMixin`` creates a ``DockerSessionManager`` in place of the
normal ``AgentSessionManager``.  The interface is identical::

    manager = DockerSessionManager(...)
    responses = await manager.query("Solve this task...")
    await manager.close()

Build the image first::

    bash docker/build.sh
"""
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import dill as pkl

from predicators.agent_sdk.config import SessionConfig
from predicators.agent_sdk.sandbox_prompts import build_sandbox_system_prompt
from predicators.agent_sdk.session_base import SandboxSessionManagerBase
from predicators.agent_sdk.tools import ToolContext, session_log_filename
from predicators.settings import CFG

logger = logging.getLogger(__name__)

# Grace period past the per-query agent timeout before the container is
# force-killed (covers container startup + result pickling).
_CONTAINER_TIMEOUT_SLACK_S = 120

# Tail sizes for error reporting when a container run fails.
_STDIO_TAIL_CHARS = 2000
_STDERR_TAIL_LINES = 20

# Build Docker-specific prompts from shared templates.
# CLAUDE.md is built per-instance with the phase tag so the agent reads
# phase-appropriate strategy guidance every turn (see build_claude_md).
_SANDBOX_SYSTEM_PROMPT = build_sandbox_system_prompt(
    env_description="an isolated Docker sandbox",
    workspace_description="/sandbox/",
    ref_path="/sandbox/reference/",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_claude_oauth_token() -> Optional[str]:
    """Extract the Claude Code OAuth access token from the macOS Keychain.

    Returns ``None`` on non-macOS platforms or when the token cannot be
    found.  On macOS, ``claude login`` stores credentials under the
    service name ``"Claude Code-credentials"``.
    """
    if sys.platform != "darwin":
        return None
    try:  # type: ignore[unreachable]
        result = subprocess.run(
            [
                "security", "find-generic-password", "-s",
                "Claude Code-credentials", "-w"
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        creds = json.loads(result.stdout.strip())
        return creds.get("claudeAiOauth", {}).get("accessToken")
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError):
        return None


# _flush_log stays unimplemented on purpose: logs flush inside the
# container, and a host-side call should fail loudly.
# pylint: disable-next=abstract-method
class DockerSessionManager(SandboxSessionManagerBase):
    """Runs ClaudeSDKClient inside Docker with built-in + custom MCP tools.

    Matches the ``AgentSessionManager`` interface so that all agent-based
    approaches work unchanged.  Each ``query()`` call:

    1. Serializes ``ToolContext`` + message to pickle in a temp directory.
    2. Runs ``docker run ...`` with the predicators source mounted at
       ``/opt/predicators:ro`` (for Python imports) and a curated sandbox
       at ``/sandbox`` (for agent file operations).
    3. Inside Docker, the runner script creates ``ClaudeSDKClient`` with
       both built-in tools AND custom MCP tools, queries the agent, and
       pickles back responses + mutated proposals.
    4. Host reads back the pickled results.

    PreToolUse hooks restrict the agent's built-in tools (Read, Write,
    Edit, Glob, Grep) to ``/sandbox/`` only.  Python imports via
    ``PYTHONPATH`` are unaffected.
    """

    _log_label = "Docker"

    def __init__(
        self,
        system_prompt: str,
        log_dir: str,
        model_name: str,
        tool_context: ToolContext,
        tool_names: Optional[List[str]] = None,
        image: str = "predicators-sandbox",
        extra_reference_files: Optional[Dict[str, str]] = None,
        phase: Optional[str] = None,
        config: Optional[SessionConfig] = None,
    ) -> None:
        # Append sandbox instructions to the system prompt
        super().__init__(system_prompt=system_prompt + _SANDBOX_SYSTEM_PROMPT,
                         log_dir=log_dir,
                         model_name=model_name,
                         tool_context=tool_context,
                         tool_names=tool_names,
                         extra_reference_files=extra_reference_files,
                         phase=phase,
                         config=config)
        self._image = image
        self._last_kind: str = "query"

    # -- Session lifecycle --

    async def start_session(self) -> None:
        """No-op: each query() is a fresh docker run."""

    async def close(self) -> None:
        """No-op: the sandbox directory is kept on disk for inspection."""

    async def _recover_session(self) -> None:
        """No-op: each query is independent."""

    async def query(self,
                    message: str,
                    kind: str = "query") -> List[Dict[str, Any]]:
        """Run the agent in Docker and return collected response messages.

        Returns the same ``List[Dict[str, Any]]`` format as
        ``AgentSessionManager.query()``.
        """
        self._query_count += 1
        self._tool_context.turn_id = self._query_count
        self._last_kind = kind

        # Ensure sandbox is set up (lazy init, persists across queries)
        self._ensure_sandbox_dir()

        # 1. Create temp directory for data exchange
        tmp_dir = tempfile.mkdtemp(prefix="pred-docker-")
        input_path = os.path.join(tmp_dir, "query_input.pkl")
        output_path = os.path.join(tmp_dir, "query_output.pkl")

        # Compute final log filename upfront so the container can write
        # directly to the log directory (incremental updates visible on host).
        # Counter-first layout: alphabetical sort matches chronological
        # order across mixed ``learn``/``test``/``explore`` phases.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = session_log_filename(
            self._query_count, kind, timestamp,
            getattr(self._tool_context, "test_task_idx", None))
        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)
            incremental_log_path = os.path.join(self._log_dir, log_filename)
        else:
            incremental_log_path = os.path.join(tmp_dir, "query_log.md")

        try:
            # 2. Pickle QueryInput
            # Tell the container where to write the incremental log.
            # If _log_dir is set, it's mounted at /log inside the container.
            container_log_path = (f"/log/{log_filename}"
                                  if self._log_dir else "/data/query_log.md")
            query_input = {
                "tool_context": self._tool_context,
                "message": message,
                "system_prompt": self._system_prompt,
                "model_name": self._model_name,
                "max_turns": self._config.max_turns,
                "max_buffer_size": self._config.max_buffer_size,
                "reasoning_effort": self._config.reasoning_effort,
                "tool_names": self._tool_names,
                "cfg_snapshot": dict(CFG.__dict__),
                "log_path": container_log_path,
            }
            with open(input_path, "wb") as f:
                pkl.dump(query_input, f)

            logger.info(
                "Docker query %d: message length=%d, model=%s",
                self._query_count,
                len(message),
                self._model_name,
            )

            # 3. Build docker run command.  Resolve authentication once
            # per query (the Keychain OAuth lookup is a subprocess call
            # shared by the command and env builders).
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            oauth_token = None if api_key else _get_claude_oauth_token()
            container_name = f"pred-sandbox-{uuid.uuid4().hex[:8]}"
            docker_cmd = self._build_docker_command(container_name, tmp_dir,
                                                    api_key, oauth_token)

            # 4. Run Docker container
            logger.info(
                "Starting Docker sandbox: container=%s image=%s",
                container_name,
                self._image,
            )
            env = self._build_env(api_key, oauth_token)

            proc = subprocess.Popen(
                docker_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Stream stderr in real-time so tool calls / agent messages
            # appear on the host terminal as they happen.
            stderr_lines: List[str] = []
            try:
                timeout_sec = (self._config.agent_timeout +
                               _CONTAINER_TIMEOUT_SLACK_S)
                import threading  # pylint: disable=import-outside-toplevel

                def _stream_stderr() -> None:
                    assert proc.stderr is not None
                    for line in proc.stderr:
                        line = line.rstrip("\n")
                        stderr_lines.append(line)
                        logger.info("%s", line)

                stderr_thread = threading.Thread(target=_stream_stderr,
                                                 daemon=True)
                stderr_thread.start()

                # Wait for stdout (captured for error reporting)
                stdout_data = proc.stdout.read() if proc.stdout else ""
                proc.wait(timeout=timeout_sec)
                stderr_thread.join(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                logger.error("Docker container timed out after %ds",
                             timeout_sec)
                stdout_data = ""

            if proc.returncode != 0:
                logger.error(
                    "Docker container exited with code %d.\nstdout: %s\n"
                    "stderr (last 2000 chars): %s",
                    proc.returncode,
                    stdout_data[-_STDIO_TAIL_CHARS:]
                    if stdout_data else "(empty)",
                    "\n".join(stderr_lines)[-_STDIO_TAIL_CHARS:]
                    if stderr_lines else "(empty)",
                )
            else:
                logger.info("Docker container exited successfully.")

            # 5. Load query output
            if os.path.exists(output_path):
                with open(output_path, "rb") as f_in:
                    query_output = pkl.load(f_in)

                responses = query_output.get("responses", [])
                proposals = query_output.get("iteration_proposals")

                # 6. Merge proposals back into host ToolContext
                if proposals is not None:
                    logger.info(
                        "Docker proposals: proposed_options=%s, "
                        "retract=%s",
                        [o.name for o in proposals.proposed_options],
                        sorted(proposals.retract_option_names),
                    )
                    self._tool_context.iteration_proposals = proposals
                    # Sync proposed/retracted options into ctx.options so
                    # the host-side parser can find them.
                    self._tool_context.options |= proposals.proposed_options
                    if proposals.retract_option_names:
                        self._tool_context.options = {
                            o
                            for o in self._tool_context.options
                            if o.name not in proposals.retract_option_names
                        }
                    logger.info(
                        "After Docker sync: tool_context.options=%s",
                        sorted(o.name for o in self._tool_context.options),
                    )
                else:
                    logger.warning(
                        "Docker output has iteration_proposals=None; "
                        "no proposals synced.")

                # Track costs/turns via the base delta accounting.  Each
                # docker query is a fresh in-container session whose
                # cumulative cost restarts from zero, so reset the delta
                # baseline first: every result then charges its full
                # cumulative cost.
                self._last_cost_usd = 0.0
                for resp in responses:
                    if resp.get("type") == "result":
                        self._account_result(resp)
            else:
                logger.error(
                    "No output pickle found at %s. Container may have "
                    "crashed.", output_path)
                responses = [{
                    "type":
                    "error",
                    "error":
                    (f"Docker container failed (exit code "
                     f"{proc.returncode}). "
                     f"stderr: {''.join(stderr_lines[-_STDERR_TAIL_LINES:])}"),
                }]

            # 7. Finalize query log - the incremental log was written
            # directly to _log_dir as markdown (updated per-message).
            # Prepend host metadata header now that the container is done.
            if os.path.exists(incremental_log_path) and self._log_dir:
                try:
                    with open(incremental_log_path, encoding="utf-8") as lf:
                        existing = lf.read()
                    header_lines = [
                        f"- **Query:** {self._query_count}",
                        f"- **Timestamp:** {timestamp}",
                        f"- **Session:** {self._session_id}",
                        f"- **Image:** {self._image}",
                        "",
                        "",
                    ]
                    with open(incremental_log_path, "w",
                              encoding="utf-8") as lf:
                        lf.write("\n".join(header_lines) + existing)
                    logger.info("Finalized docker query/response at %s",
                                incremental_log_path)
                except Exception:  # pylint: disable=broad-except
                    logger.warning("Failed to enrich log at %s",
                                   incremental_log_path,
                                   exc_info=True)
            else:
                self._save_query_response_log(message, responses)

            # Track in-memory for conversation replay
            self._conversation_log.append({
                "query": message,
                "response": responses,
            })

            self._track_fatal_response(responses)
            return responses

        finally:
            # Cleanup temp data directory (sandbox persists across queries)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _session_info_extras(self) -> Dict[str, Any]:
        """Extra session-info keys: manager type + container image."""
        return {
            "session_type": "docker",
            "docker_image": self._image,
        }

    # -- Internal helpers --

    def _build_docker_command(self, container_name: str, tmp_dir: str,
                              api_key: Optional[str],
                              oauth_token: Optional[str]) -> List[str]:
        """Build the ``docker run`` command."""
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--cap-add=NET_ADMIN",
            "--cap-add=NET_RAW",
        ]

        # Authentication: prefer ANTHROPIC_API_KEY, fall back to OAuth
        if api_key:
            cmd += ["-e", "ANTHROPIC_API_KEY"]
        elif oauth_token:
            # The token value itself is added to env in _build_env()
            cmd += ["-e", "CLAUDE_CODE_OAUTH_TOKEN"]
        else:
            # Fall back to bind-mounting ~/.claude
            claude_cfg = Path(
                os.environ.get("CLAUDE_CONFIG_DIR",
                               str(Path.home() / ".claude")))
            cmd += ["-v", f"{claude_cfg}:/home/node/.claude"]

        # Mount predicators source for Python imports (hidden from agent
        # tools by the PreToolUse hook - only Python's import system can
        # read these files).
        cmd += ["-v", f"{self._repo_root}:/opt/predicators:ro"]
        cmd += ["-e", "PYTHONPATH=/opt/predicators"]

        # Mount curated sandbox directory
        cmd += ["-v", f"{self._sandbox_dir}:/sandbox"]

        # Mount data exchange directory
        cmd += ["-v", f"{tmp_dir}:/data"]

        # Mount log directory for incremental log updates visible on host
        if self._log_dir:
            log_dir_abs = os.path.abspath(self._log_dir)
            cmd += ["-v", f"{log_dir_abs}:/log"]

        # Working directory
        cmd += ["-w", "/sandbox"]

        # Image
        cmd.append(self._image)

        # Command: run the agent runner script from the mounted source
        cmd += [
            "python3",
            "-u",
            "/opt/predicators/predicators/agent_sdk/docker_agent_runner.py",
            "/data/query_input.pkl",
            "/data/query_output.pkl",
        ]

        return cmd

    def _build_env(self, api_key: Optional[str],
                   oauth_token: Optional[str]) -> Dict[str, str]:
        """Build environment dict for the docker subprocess."""
        # Pass through host env, stripping CLAUDECODE* vars
        env = {
            k: v
            for k, v in os.environ.items() if not k.startswith("CLAUDECODE")
        }

        # Ensure ANTHROPIC_API_KEY is passed through if set
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        elif oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token

        return env

    def _save_query_response_log(self, query: str,
                                 response: List[Dict[str, Any]]) -> None:
        """Save query and response to a timestamped markdown file."""
        if not self._log_dir:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        kind = self._last_kind
        filename = session_log_filename(
            self._query_count, kind, timestamp,
            getattr(self._tool_context, "test_task_idx", None))
        filepath = os.path.join(self._log_dir, filename)

        lines = [
            f"- **Query:** {self._query_count}",
            f"- **Timestamp:** {timestamp}",
            f"- **Session:** {self._session_id}",
            f"- **Image:** {self._image}",
            "",
            "# Docker Query",
            "",
            "## Prompt",
            "",
            query,
            "",
            "## Response",
            "",
        ]
        for entry in response:
            lines.append(
                f"```json\n{json.dumps(entry, indent=2, default=str)}\n```")
            lines.append("")

        os.makedirs(self._log_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Saved docker query/response to %s", filepath)
