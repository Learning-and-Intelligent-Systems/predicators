"""Shared base classes and helpers for the agent session managers.

Three session managers share one interface (see
``SessionManagerProtocol`` in :mod:`session_manager`):

- ``AgentSessionManager``: in-process client, no sandbox, JSON logs.
- ``LocalSandboxSessionManager``: in-process client confined to a local
  sandbox cwd via hooks, markdown logs.
- ``DockerSessionManager``: stateless container per query, markdown
  logs written in-container.

This module owns everything they have in common: the base manager state
(cost/turn accounting, conversation log, session-info persistence,
close/recovery), the streamed receive-loop, sandbox directory
scaffolding for the two sandboxed managers, and the assembly of
``ClaudeAgentOptions``.

``build_agent_options``, ``build_sandbox_mcp``, ``block_preview``, and
``stream_agent_response`` are module-level functions (not methods) so
the in-container Docker runner can use them too.  They are pure
functions of their arguments - no ``CFG`` reads - because the container
receives all settings explicitly via the pickled ``query_input``.
"""
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from predicators.agent_sdk.config import SessionConfig
from predicators.agent_sdk.log_formatter import truncate
from predicators.agent_sdk.response_parser import parse_message
from predicators.agent_sdk.sandbox_prompts import build_claude_md
from predicators.agent_sdk.sandbox_setup import find_repo_root, \
    setup_sandbox_directory
from predicators.agent_sdk.thinking import resolve_thinking_config
from predicators.settings import CFG

logger = logging.getLogger(__name__)

# Character cap for per-block debug previews of agent output.
_TEXT_PREVIEW_CHARS = 200

# Reasoning-effort levels accepted by the Claude Agent SDK.
_VALID_EFFORTS = frozenset({"low", "medium", "high", "max"})

# Error banners the SDK CLI surfaces as ordinary assistant text (one
# turn, $0.00 - not an exception) when the backend is unusable; matched
# case-insensitively against responses that made no tool call.
_FATAL_RESPONSE_PATTERNS = (
    "disabled claude subscription access",
    "use an anthropic api key",
    "invalid api key",
    "credit balance is too low",
    "oauth token has expired",
    "please run /login",
    "authentication_error",
)


class AgentSessionFatalError(Exception):
    """The agent session backend is failing in a way no retry can fix.

    Raised after ``CFG.agent_sdk_max_consecutive_fatal_queries``
    consecutive queries died without the agent doing any work (see
    :func:`query_fatal_error`).  Deliberately NOT an ``ApproachFailure``
    subclass: per-task and per-attempt handlers catch and absorb those,
    while this error must propagate and terminate the whole run instead
    of burning the attempt/replan/cycle budgets on instant failures
    (run_20260721_161159 spent 10 cycles on ~300 one-second auth-error
    queries without the agent ever running).  Every broad ``except``
    between a session query and ``main`` must re-raise it.
    """


def query_fatal_error(response: List[Dict[str, Any]]) -> Optional[str]:
    """Why this query's response looks fatally broken, or ``None``.

    A response only qualifies when the agent made NO tool call: real
    sessions that end badly mid-work (turn cap, deadline interrupt,
    transport drop) all have tool calls behind them, while auth /
    billing / config failures die on the first assistant turn.  Within
    that gate three signals count: a known fatal banner in the assistant
    or result text, an error result (excluding ``error_max_turns``, the
    ordinary turn-cap budget end), or a stream error entry.
    """
    texts: List[str] = []
    stream_error: Optional[str] = None
    result_error: Optional[str] = None
    for entry in response:
        etype = entry.get("type")
        if etype == "assistant":
            for block in entry.get("content", []):
                if block.get("type") == "tool_use":
                    return None
                if block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
        elif etype == "error":
            stream_error = str(entry.get("error") or "unknown stream error")
        elif etype == "result":
            if entry.get("is_error") and \
                    entry.get("subtype") != "error_max_turns":
                result_error = str(
                    entry.get("result") or entry.get("subtype")
                    or "unknown error result")
    for text in texts + ([result_error] if result_error else []):
        low = text.lower()
        for pattern in _FATAL_RESPONSE_PATTERNS:
            if pattern in low:
                return text.strip()
    if result_error is not None:
        return f"error result: {result_error}"
    if stream_error is not None:
        return f"stream error: {stream_error}"
    return None


def validate_reasoning_effort(reasoning_effort: str) -> Optional[str]:
    """Normalize a reasoning-effort setting, raising on garbage.

    Returns one of the SDK's accepted levels, or ``None`` when the
    setting is ``""``/``"default"`` (leave the SDK default).  Any other
    value raises ``ValueError`` - on every path, including inside the
    Docker runner, so a bad setting fails loudly instead of being
    silently dropped.
    """
    effort = (reasoning_effort or "").strip().lower()
    if not effort or effort == "default":
        return None
    if effort not in _VALID_EFFORTS:
        raise ValueError(f"agent_sdk_reasoning_effort must be one of "
                         f"{sorted(_VALID_EFFORTS)} or ''/'default'; got "
                         f"{reasoning_effort!r}")
    return effort


def build_sandbox_mcp(
        tool_context: Any,
        tool_names: Optional[List[str]]) -> Tuple[Any, List[str]]:
    """Create the in-process predicator MCP server and full tool list.

    Shared by the local sandbox manager (host side) and the Docker agent
    runner (in-container): the MCP tools are closures over
    ``tool_context``, and the allowed tools are Claude's built-ins plus
    the custom MCP tools.
    """
    # pylint: disable=import-outside-toplevel
    from claude_agent_sdk import create_sdk_mcp_server

    from predicators.agent_sdk.tools import BUILTIN_TOOLS, MCP_SERVER_NAME, \
        create_mcp_tools, get_allowed_tool_list

    # pylint: enable=import-outside-toplevel
    tools = create_mcp_tools(tool_context, tool_names=tool_names)
    mcp_server = create_sdk_mcp_server(
        name=MCP_SERVER_NAME,
        version="1.0.0",
        tools=tools,
    )
    allowed_tools = BUILTIN_TOOLS + get_allowed_tool_list(tool_names)
    return mcp_server, allowed_tools


def build_agent_options(*,
                        system_prompt: str,
                        model_name: str,
                        allowed_tools: List[str],
                        mcp_server: Any,
                        max_turns: int,
                        max_buffer_size: int,
                        reasoning_effort: str = "",
                        cwd: Optional[str] = None,
                        setting_sources: Optional[List[str]] = None,
                        hooks: Optional[Dict[str, Any]] = None) -> Any:
    """Assemble the ``ClaudeAgentOptions`` shared by all session managers.

    Pure function of its arguments (no ``CFG`` reads) so the Docker
    runner can call it in-container with values shipped in
    ``query_input``.  ``cwd`` and ``setting_sources`` are only passed
    for the local sandbox; ``hooks`` only when non-empty.  Raises
    ``ValueError`` when ``reasoning_effort`` is invalid (see
    ``validate_reasoning_effort``).
    """
    # pylint: disable=import-outside-toplevel
    from claude_agent_sdk import ClaudeAgentOptions

    from predicators.agent_sdk.tools import MCP_SERVER_NAME

    # pylint: enable=import-outside-toplevel
    # Model-dependent thinking config: adaptive on sonnet-5+ (where
    # budget_tokens is rejected with a 400 and depth is controlled via
    # ``effort``), manual extended thinking with a fixed budget on older
    # models like claude-sonnet-4-6.
    thinking = resolve_thinking_config(model_name)
    effort = validate_reasoning_effort(reasoning_effort)
    extra: Dict[str, Any] = {}
    if cwd is not None:
        extra["cwd"] = cwd
    if setting_sources is not None:
        extra["setting_sources"] = setting_sources
    return ClaudeAgentOptions(
        allowed_tools=allowed_tools,
        # Disallowing ToolSearch turns off tool-search deferral, so
        # every predicator MCP tool schema is loaded up front. With
        # deferral on, every audited run burned 5-9 turns on the
        # ToolSearch ritual (select: misses on unprefixed names, bare
        # `mcp__predicator_tools` miscalls) and re-paid it after each
        # compaction - the handful of core tools are always needed,
        # so deferring their schemas saves nothing.
        # AskUserQuestion: sessions run unattended, so there is never a
        # user to answer; without the ban, agents at a dead end call it
        # repeatedly and burn turns reasoning about the opaque error it
        # returns headless.
        disallowed_tools=["ToolSearch", "AskUserQuestion"],
        mcp_servers={MCP_SERVER_NAME: mcp_server},
        permission_mode="bypassPermissions",
        system_prompt=system_prompt,
        model=model_name,
        max_turns=max_turns,
        max_buffer_size=max_buffer_size,
        thinking=thinking,  # type: ignore[arg-type]
        effort=effort,  # type: ignore[arg-type]
        hooks=(hooks if hooks else None),  # type: ignore[arg-type]
        **extra,
    )


def block_preview(block: Dict[str, Any]) -> Optional[str]:
    """One-line progress preview for an assistant content block.

    Covers text, tool_use, and thinking blocks (``parse_message``
    surfaces ``ThinkingBlock`` as a dict carrying a ``thinking`` key);
    returns ``None`` for anything else.
    """
    btype = block.get("type", "")
    if btype == "text":
        return f"Agent: {block['text'][:_TEXT_PREVIEW_CHARS]}..."
    if btype == "tool_use":
        params = block.get("input") or {}
        param_summary = ", ".join(f"{k}={truncate(v)}"
                                  for k, v in params.items())
        return f"Agent tool call: {block['name']}({param_summary})"
    thinking = block.get("thinking")
    if thinking:
        return f"Agent [thinking]: {str(thinking)[:_TEXT_PREVIEW_CHARS]}..."
    return None


def _default_report_block(dt: float, preview: str) -> None:
    """Default per-block reporter: debug-log with step timing."""
    logger.debug("[+%.2fs] %s", dt, preview)


async def stream_agent_response(
    client: Any,
    message: str,
    *,
    log_label: str = "Agent",
    on_entry: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    report_block: Optional[Callable[[float, str], None]] = None,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    flush: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    on_error: Optional[Callable[[], Awaitable[None]]] = None,
) -> List[Dict[str, Any]]:
    """Send ``message`` and drain the streamed response into a list.

    Owns the receive-loop shape shared by all three session managers and
    the in-container runner: parse each SDK message, append it, report
    per-block progress (text, tool calls, thinking) via ``report_block``
    with the wall-clock delta since the previous message (model thinking
    before a tool call, tool execution before the next message, etc.),
    account result entries via ``on_result``, and ``flush`` after every
    message so the incremental log stays current mid-turn.  ``on_entry``
    is awaited per entry (e.g. the local sandbox's deadline interrupt).

    On a stream error an ``{"type": "error"}`` entry is appended and
    flushed, then ``on_error`` (e.g. session recovery) is awaited; the
    failed message is NOT resent - the caller sees the error entry and
    decides whether to retry.
    """
    if report_block is None:
        report_block = _default_report_block
    collected: List[Dict[str, Any]] = []
    prev_t = time.perf_counter()
    try:
        await client.query(message)
        async for msg in client.receive_response():
            entry = parse_message(msg)
            if entry is None:
                continue
            collected.append(entry)
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            if on_entry is not None:
                await on_entry(entry)

            # Log side-effects
            if entry["type"] == "assistant":
                for block in entry.get("content", []):
                    preview = block_preview(block)
                    if preview is not None:
                        report_block(dt, preview)
            elif entry["type"] == "result" and on_result is not None:
                on_result(entry)

            # Flush log after each message
            if flush is not None:
                flush(collected)

    except Exception as e:  # pylint: disable=broad-except
        logger.error("%s session error: %s", log_label, e)
        collected.append({"type": "error", "error": str(e)})
        if flush is not None:
            flush(collected)
        if on_error is not None:
            await on_error()

    return collected


class BaseAgentSessionManager:
    """Common state and behavior for the three session managers.

    Owns the session/cost/turn bookkeeping, the property trio of the
    ``SessionManagerProtocol`` surface, session-info persistence, the
    default close/recovery for client-holding managers (the Docker
    manager overrides both with no-ops), and the streamed query runner
    built on :func:`stream_agent_response`.
    """

    # Human-readable label used in shared log messages ("Agent
    # iteration complete...", "Error closing agent session...").
    _log_label = "Agent"

    # Consecutive fatal-looking queries across ALL managers in the
    # process (a class attribute on purpose): fresh-context restarts
    # close and recreate the manager on every attempt, so an instance
    # counter would reset exactly when the loop it must stop is
    # spinning.  Reset to 0 by any healthy query.
    _consecutive_fatal_queries = 0

    def __init__(self,
                 system_prompt: str,
                 log_dir: str,
                 model_name: str,
                 tool_context: Any = None,
                 config: Optional[SessionConfig] = None) -> None:
        self._system_prompt = system_prompt
        self._log_dir = log_dir
        self._model_name = model_name
        self._config = config if config is not None else \
            SessionConfig.from_cfg()
        self._tool_context = tool_context
        self._client: Any = None
        self._started = False
        self._session_id: Optional[str] = None
        self._total_cost_usd: float = 0.0
        # total_cost_usd from the SDK is the cumulative session cost; track
        # the last value to derive each query's per-solve (marginal) cost.
        self._last_cost_usd: float = 0.0
        self._total_turns: int = 0
        self._query_count: int = 0
        self._conversation_log: List[Dict[str, Any]] = []
        self._current_log_meta: Dict[str, Any] = {}

    # -- Properties matching the SessionManagerProtocol surface --

    @property
    def session_id(self) -> Optional[str]:
        """Return the current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    @property
    def conversation_log(self) -> List[Dict[str, Any]]:
        """Return the in-memory log of all query/response pairs."""
        return self._conversation_log

    @property
    def tool_names(self) -> List[str]:
        """Return short tool names (without MCP prefix)."""
        from predicators.agent_sdk.tools import \
            MCP_SERVER_NAME  # pylint: disable=import-outside-toplevel
        prefix = f"mcp__{MCP_SERVER_NAME}__"
        return [
            t[len(prefix):] if t.startswith(prefix) else t
            for t in self._qualified_tool_names()
        ]

    def _qualified_tool_names(self) -> List[str]:
        """Fully-qualified allowed tool names for this manager."""
        raise NotImplementedError

    # -- Session lifecycle --

    async def start_session(self) -> None:
        """Start (or lazily prepare) the underlying agent session."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the agent session."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Error closing %s session: %s",
                               self._log_label.lower(), e)
            finally:
                self._client = None
                self._started = False

    async def _recover_session(self) -> None:
        """Attempt to recover from a session error.

        Reconnects only; the failed message is NOT resent - the caller
        sees the error entry and decides whether to retry.
        """
        logger.warning("Attempting %s session recovery...",
                       self._log_label.lower())
        try:
            if self._client is not None:
                try:
                    await self._client.disconnect()
                except Exception:  # pylint: disable=broad-except
                    pass
            self._started = False
            await self.start_session()
            logger.info("%s session recovered.", self._log_label)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("%s session recovery failed: %s", self._log_label, e)

    # -- Query streaming --

    def _flush_log(self, filepath: str, response: List[Dict[str,
                                                            Any]]) -> None:
        """Rewrite the incremental query log with the current response."""
        raise NotImplementedError

    def _account_result(self, entry: Dict[str, Any]) -> None:
        """Fold one result entry into the cost/turn totals.

        The cost is the session's cumulative total; the per-solve cost
        is the delta since the last result.  A drop below the last value
        means the session was reset (e.g. recovery, or Docker's fresh
        session per query), so the new cumulative is itself the delta.
        """
        cost = entry.get("total_cost_usd")
        turns = entry.get("num_turns")
        solve_cost: Optional[float] = None
        if cost is not None:
            if cost >= self._last_cost_usd:
                solve_cost = float(cost - self._last_cost_usd)
            else:
                solve_cost = float(cost)
            self._last_cost_usd = cost
            self._total_cost_usd += solve_cost
            self._current_log_meta["solve_cost_usd"] = solve_cost
            self._current_log_meta["total_cost_usd"] = self._total_cost_usd
        if turns is not None:
            self._total_turns += turns
        logger.info(
            "%s iteration complete. Turns: %s, "
            "Cost this solve: $%s, Total cost so far: $%s", self._log_label,
            turns or '?',
            f"{solve_cost:.4f}" if solve_cost is not None else '?',
            f"{self._total_cost_usd:.4f}")

    async def _run_streamed_query(
        self,
        message: str,
        *,
        log_path: Optional[str],
        kind: str,
        on_entry: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the shared receive-loop for one query and finalize logs.

        Streams via :func:`stream_agent_response` (cost accounting, per-
        message flushing, error recovery), then writes the final flush
        and appends to the in-memory conversation log.
        """
        flush: Optional[Callable[[List[Dict[str, Any]]], None]] = None
        if log_path is not None:

            def _flush(collected: List[Dict[str, Any]]) -> None:
                assert log_path is not None
                self._flush_log(log_path, collected)

            flush = _flush

        start = time.perf_counter()
        collected = await stream_agent_response(
            self._client,
            message,
            log_label=self._log_label,
            on_entry=on_entry,
            on_result=self._account_result,
            flush=flush,
            on_error=self._recover_session,
        )
        elapsed = time.perf_counter() - start
        logger.info("[agent-interaction] kind=%s took %.2fs (%d messages)",
                    kind, elapsed, len(collected))

        # Final flush to ensure everything is saved
        if log_path:
            self._flush_log(log_path, collected)
            logger.info("Saved %s query/response to %s",
                        self._log_label.lower(), log_path)

        # Track in-memory for conversation replay
        self._conversation_log.append({
            "query": message,
            "response": collected,
        })
        self._track_fatal_response(collected)
        return collected

    def _track_fatal_response(self, response: List[Dict[str, Any]]) -> None:
        """Terminate the run after consecutive fatally-broken queries.

        Each fatal query (see :func:`query_fatal_error`) returns in ~1 s
        at $0.00 and is otherwise indistinguishable from a no-capture
        attempt, so without this check the solve restart / replan /
        online-cycle budgets grind through hundreds of instant failures.
        The counter is process-wide (class attribute) and any healthy
        query resets it; at ``agent_sdk_max_consecutive_fatal_queries``
        the raised :class:`AgentSessionFatalError` propagates past the
        per-task handlers and ends the run.
        """
        limit = CFG.agent_sdk_max_consecutive_fatal_queries
        if limit <= 0:
            return
        reason = query_fatal_error(response)
        if reason is None:
            BaseAgentSessionManager._consecutive_fatal_queries = 0
            return
        count = BaseAgentSessionManager._consecutive_fatal_queries + 1
        BaseAgentSessionManager._consecutive_fatal_queries = count
        logger.warning(
            "%s query died without the agent doing any work "
            "(%d/%d consecutive): %s", self._log_label, count, limit, reason)
        if count >= limit:
            raise AgentSessionFatalError(
                f"{count} consecutive agent queries died without the agent "
                f"doing any work; latest: {reason}. The session backend is "
                "unusable (auth/billing/config failure), so the run is "
                "terminated instead of burning the remaining attempt/replan/"
                "cycle budgets on instant failures.")

    # -- Session info persistence --

    def _session_info_extras(self) -> Dict[str, Any]:
        """Manager-specific extra keys for ``session_info.json``."""
        return {}

    def save_session_info(self) -> None:
        """Save session metadata to log directory."""
        os.makedirs(self._log_dir, exist_ok=True)
        info: Dict[str, Any] = {
            "session_id": self._session_id,
            "total_cost_usd": self._total_cost_usd,
            "total_turns": self._total_turns,
            "model": self._model_name,
        }
        info.update(self._session_info_extras())
        path = os.path.join(self._log_dir, "session_info.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        logger.info("Saved session info to %s", path)


# An intermediate abstract base: transport methods stay unimplemented
# on purpose (loud NotImplementedError beats a silent no-op).
# pylint: disable-next=abstract-method
class SandboxSessionManagerBase(BaseAgentSessionManager):
    """Shared state and sandbox scaffolding for the sandboxed managers.

    Both sandbox managers confine the agent to a per-run sandbox
    directory seeded with curated reference files; this base owns the
    sandbox path bookkeeping and lazy population.  The session transport
    (in-process client vs. one docker run per query) stays in the
    subclasses.
    """

    def __init__(
        self,
        system_prompt: str,
        log_dir: str,
        model_name: str,
        tool_context: Any,
        tool_names: Optional[List[str]] = None,
        extra_reference_files: Optional[Dict[str, str]] = None,
        phase: Optional[str] = None,
        config: Optional[SessionConfig] = None,
    ) -> None:
        super().__init__(system_prompt=system_prompt,
                         log_dir=log_dir,
                         model_name=model_name,
                         tool_context=tool_context,
                         config=config)
        self._tool_names = tool_names
        self._extra_reference_files = extra_reference_files or {}
        self._repo_root = str(find_repo_root())
        self._phase = phase
        # Sandbox path is deterministic from log_dir; expose it on the
        # tool context eagerly so callers that build sandbox-relative
        # paths before the first query() see the right value. Directory
        # creation + file copying still happen lazily in
        # ``_ensure_sandbox_dir`` on first use.
        self._sandbox_dir: str = os.path.abspath(
            os.path.join(self._log_dir, "sandbox"))
        self._tool_context.sandbox_dir = self._sandbox_dir
        self._tool_context.image_save_dir = str(
            os.path.join(self._sandbox_dir, "test_images"))
        self._sandbox_populated = False

    def _qualified_tool_names(self) -> List[str]:
        from predicators.agent_sdk.tools import \
            BUILTIN_TOOLS  # pylint: disable=import-outside-toplevel
        names = list(BUILTIN_TOOLS)
        if self._tool_names:
            names += self._tool_names
        return names

    def _ensure_sandbox_dir(self) -> None:
        """Create and populate the sandbox directory if it doesn't exist.

        The path itself is set in ``__init__`` (so callers can use it
        before the first query); this method handles dir creation and
        seeding, which is idempotent across calls but only needs to run
        once per session.
        """
        if self._sandbox_populated:
            return
        setup_sandbox_directory(
            sandbox_dir=self._sandbox_dir,
            repo_root=self._repo_root,
            extra_reference_files=self._extra_reference_files,
            claude_md_content=build_claude_md(phase=self._phase),
            system_prompt=self._system_prompt,
            log_dir=self._log_dir,
            seed_scratchpad=self._config.use_scratchpad,
            phase=self._phase,
        )
        self._sandbox_populated = True
