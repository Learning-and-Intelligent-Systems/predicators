"""Agent session lifecycle management for Claude SDK."""
import asyncio
import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from predicators.agent_sdk.config import SessionConfig
from predicators.agent_sdk.session_base import BaseAgentSessionManager, \
    build_agent_options
from predicators.settings import CFG

logger = logging.getLogger(__name__)


@runtime_checkable
class SessionManagerProtocol(Protocol):
    """Structural interface shared by the three session managers.

    Implemented by :class:`AgentSessionManager` (in-process, no
    sandbox), ``LocalSandboxSessionManager`` (in-process, sandbox cwd +
    hooks), and ``DockerSessionManager`` (stateless container per
    query). Consumers - ``AgentSessionMixin`` and the agent explorers -
    must depend on this, not a concrete manager.
    """

    session_id: Optional[str]

    @property
    def tool_names(self) -> List[str]:
        """Fully-qualified allowed tool names for this session."""
        ...  # pylint: disable=unnecessary-ellipsis

    @property
    def conversation_log(self) -> List[Dict[str, Any]]:
        """In-memory log of all query/response pairs."""
        ...  # pylint: disable=unnecessary-ellipsis

    async def start_session(self) -> None:
        """Start (or lazily prepare) the underlying agent session."""
        ...  # pylint: disable=unnecessary-ellipsis

    async def query(self,
                    message: str,
                    kind: str = "query") -> List[Dict[str, Any]]:
        """Send one message and return the collected response entries."""
        ...  # pylint: disable=unnecessary-ellipsis

    async def close(self) -> None:
        """Tear down the underlying agent session, if any."""
        ...  # pylint: disable=unnecessary-ellipsis

    def save_session_info(self) -> None:
        """Persist session metadata to the log directory."""
        ...  # pylint: disable=unnecessary-ellipsis


class AgentSessionManager(BaseAgentSessionManager):
    """Wraps ClaudeSDKClient for persistent sessions with custom MCP tools.

    Unlike the sandboxed managers, this one takes a pre-built MCP server
    and fully-qualified tool list, runs without a sandbox cwd, and
    writes its incremental query logs as JSON (not markdown).
    """

    def __init__(self,
                 system_prompt: str,
                 mcp_server: Any,
                 log_dir: str,
                 model_name: str,
                 allowed_tools: Optional[List[str]] = None,
                 tool_context: Any = None,
                 config: Optional[SessionConfig] = None) -> None:
        # tool_context is an optional ToolContext reference - read at
        # session start so the caller can inject ``extra_session_hooks``
        # between sessions without rebuilding the manager.
        super().__init__(system_prompt=system_prompt,
                         log_dir=log_dir,
                         model_name=model_name,
                         tool_context=tool_context,
                         config=config)
        self._mcp_server = mcp_server
        self._allowed_tools = allowed_tools

    def _qualified_tool_names(self) -> List[str]:
        return list(self._allowed_tools or [])

    async def start_session(self) -> None:
        """Start a new Claude SDK client session."""
        from claude_agent_sdk import \
            ClaudeSDKClient  # pylint: disable=import-outside-toplevel

        extra_hooks: Dict[str, Any] = {}
        if self._tool_context is not None:
            extra_hooks = dict(
                getattr(self._tool_context, "extra_session_hooks", {}) or {})
        options = build_agent_options(
            system_prompt=self._system_prompt,
            model_name=self._model_name,
            allowed_tools=self._allowed_tools or [],
            mcp_server=self._mcp_server,
            max_turns=self._config.max_turns,
            max_buffer_size=self._config.max_buffer_size,
            reasoning_effort=self._config.reasoning_effort,
            hooks=extra_hooks,
        )

        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self._started = True
        logger.info("Agent SDK session started.")

    def _init_incremental_log(self,
                              query: str,
                              kind: str = "query") -> Optional[str]:
        """Initialize log file for incremental writing.

        Returns filepath.
        """
        if not CFG.log_file:
            return None

        self._query_count += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Counter-first layout: alphabetical sort matches chronological
        # order across mixed ``learn``/``test``/``explore`` phases.
        filename = f"{self._query_count:03d}_{kind}_{timestamp}.json"
        filepath = os.path.join(self._log_dir, filename)
        os.makedirs(self._log_dir, exist_ok=True)

        self._current_log_meta = {
            "query_number": self._query_count,
            "kind": kind,
            "timestamp": timestamp,
            "query": query,
            "session_id": self._session_id,
        }
        # Write initial state (empty response)
        self._flush_log(filepath, [])
        return filepath

    def _flush_log(self, filepath: str, response: List[Dict[str,
                                                            Any]]) -> None:
        """Rewrite log file with current accumulated response."""
        log_data = {**self._current_log_meta, "response": response}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, default=str)

    async def query(self,
                    message: str,
                    kind: str = "query") -> List[Dict[str, Any]]:
        """Send a message to the agent and collect all response messages.

        Returns a list of dicts with message content for logging.
        """
        if not self._started:
            await self.start_session()
        log_path = self._init_incremental_log(message, kind=kind)
        return await self._run_streamed_query(message,
                                              log_path=log_path,
                                              kind=kind)


def run_async_sync(coro: Any) -> Any:
    """Run ``coro`` to completion from synchronous code.

    Reuses a running event loop via nest_asyncio when one is active,
    otherwise falls back to ``asyncio.run``.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-outside-toplevel
            nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def run_query_sync(session: Any, message: str,
                   **query_kwargs: Any) -> List[Dict[str, Any]]:
    """Synchronously run ``session.query(message, **query_kwargs)``.

    Extra kwargs (e.g. ``kind="learn"`` for log-file tagging) are
    forwarded to ``query``.
    """
    return run_async_sync(session.query(message, **query_kwargs))
