"""Mixin providing shared agent session infrastructure.

Extracts common code for ToolContext initialization, lazy
AgentSessionManager creation, async-to-sync bridging, and agent explorer
creation shared by AgentModelFreeApproach and its subclasses.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Set

from gym.spaces import Box

from predicators.agent_sdk.config import SessionConfig
from predicators.agent_sdk.session_manager import AgentSessionManager, \
    SessionManagerProtocol, run_async_sync, run_query_sync
from predicators.agent_sdk.tools import ALL_TOOL_NAMES, ToolContext, \
    create_mcp_tools, get_allowed_tool_list
from predicators.explorers import create_explorer
from predicators.explorers.base_explorer import BaseExplorer
from predicators.settings import CFG
from predicators.structs import ParameterizedOption, Predicate, Task, Type

logger = logging.getLogger(__name__)


class AgentSessionMixin:
    """Mixin that provides shared agent session infrastructure.

    Subclasses must override:
      - _get_agent_system_prompt()

    And may optionally override:
      - _get_solve_tool_names()         -- complete tool surface for
        solve / explore sessions. May mix static MCP tool names with
        names of dynamic ``SdkMcpTool`` instances. ``None`` = all
        static MCP tools, ``[]`` = none.
      - _get_synthesis_tool_names()     -- complete tool surface for
        synthesis sessions (``_learning_mode=True``). Same shape /
        semantics as the solve hook, independent value.

    Dynamic ``SdkMcpTool`` instances are supplied by the approach
    directly: it assigns them to ``ctx.extra_mcp_tools`` before
    opening a synthesis session and clears the field afterwards. The
    mixin asserts the instance names line up with the names declared
    in :meth:`_get_synthesis_tool_names`.
    """

    _log_subdir: str = "agent"  # fallback; _get_log_dir prefers get_name()

    # --- Host-class contract ------------------------------------------ #
    # The mixin is mixed into BaseApproach subclasses; these declare the
    # host attributes it reads, so a typo fails type-checking instead of
    # silently hitting a getattr default. ``_learning_mode`` is flipped
    # by the sim-learning approach around synthesis sessions; the class
    # default keeps plain solve-only hosts working without declaring it.
    _learning_mode: bool = False
    _types: Set[Type]
    _action_space: Box
    _train_tasks: List[Any]

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def _init_agent_session_state(
        self,
        types: Set[Type],
        predicates: Set[Predicate],
        options: Set[ParameterizedOption],
        train_tasks: List[Task],
    ) -> None:
        """Initialize ToolContext and lazy agent session placeholders."""
        self._tool_context = ToolContext(
            types=types,
            predicates=predicates,
            options=options,
            train_tasks=train_tasks,
        )
        self._agent_session: Optional[SessionManagerProtocol] = None
        self._agent_session_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Customization hooks (override in subclasses)
    # ------------------------------------------------------------------ #

    def _get_agent_system_prompt(self) -> str:
        """Return the system prompt for the agent session."""
        raise NotImplementedError

    def _get_solve_tool_names(self) -> Optional[List[str]]:
        """Return the complete tool surface for solve / explore sessions.

        May mix static MCP tool names with names of dynamic
        ``SdkMcpTool`` instances. ``None`` means "all static MCP tools";
        override to subset.
        """
        return None

    def _get_synthesis_tool_names(self) -> Optional[List[str]]:
        """Return the complete tool surface for the synthesis session.

        Selected when ``_learning_mode`` is True. Independent of the
        solve list — the two phases may share names or be disjoint. Each
        name must back either a static MCP tool (member of
        ``ALL_TOOL_NAMES``) or a dynamic ``SdkMcpTool`` instance the
        approach attaches via ``ctx.extra_mcp_tools``. Default ``[]``
        means no tools (approaches with no synthesis phase need not
        override).
        """
        return []

    def _get_sandbox_reference_files(self) -> Dict[str, str]:
        """Return extra reference files for the docker sandbox.

        Maps destination paths (relative to ``/sandbox/reference/``) to
        source paths (relative to the repo root).  Override in
        subclasses to provide approach-specific reference material.
        """
        return {}

    # ------------------------------------------------------------------ #
    # Shared implementations
    # ------------------------------------------------------------------ #

    def _ensure_agent_session(self) -> None:
        """Create the agent session manager if needed.

        When ``SessionConfig.use_docker_sandbox`` is ``True``, creates a
        ``DockerSessionManager`` that runs ``ClaudeSDKClient`` inside a
        Docker container with full built-in tools (Bash, Read, Write,
        …). Otherwise creates the normal in-process
        ``AgentSessionManager``.
        """
        if self._agent_session is not None:
            return

        # Session config is read once here (at session-construction
        # time, never import time) and handed to whichever manager is
        # built below.
        config = SessionConfig.from_cfg()

        # Pick the declared tool surface by phase. ``_learning_mode`` is
        # the same signal the system-prompt branch reads, so tools and
        # prompt stay in sync. Each approach declares its solve and
        # synthesis tool sets independently — they may be disjoint.
        # ``tool_names`` is the *complete* declared list (may mix static
        # MCP names with names of dynamic SdkMcpTool instances).
        if self._learning_mode:
            tool_names = self._get_synthesis_tool_names()  # pylint: disable=assignment-from-none
        else:
            tool_names = self._get_solve_tool_names()  # pylint: disable=assignment-from-none

        # Sanity: every dynamic name in the declared list must have a
        # backing tool attached to ``ctx.extra_mcp_tools``. Static MCP
        # names (``ALL_TOOL_NAMES``) are excluded — they're materialized
        # downstream by ``create_mcp_tools``. Catches typos and missing
        # builder hooks before the agent silently fails to invoke a
        # declared-but-missing tool.
        declared = set(tool_names or ())
        dynamic_declared = declared - set(ALL_TOOL_NAMES)
        if dynamic_declared:
            attached = list(self._tool_context.extra_mcp_tools or ())
            built = {getattr(t, "name", "") for t in attached}
            missing = dynamic_declared - built
            phase_for_msg = "synthesis" if self._learning_mode else "solve"
            assert not missing, (
                f"Dynamic tool name(s) {sorted(missing)} declared in "
                f"_get_{phase_for_msg}_tool_names but no matching tool "
                f"attached to ctx.extra_mcp_tools — add them to the "
                f"builder or drop the names.")

        phase = "synthesis" if self._learning_mode else "solve"
        approach_name = getattr(type(self), "get_name",
                                lambda: type(self).__name__)()
        if tool_names is None:
            logger.info(
                "[%s] %s session tool surface: ALL static MCP tools "
                "(no subset declared).", approach_name, phase)
        else:
            static = sorted(n for n in tool_names if n in set(ALL_TOOL_NAMES))
            dynamic = sorted(n for n in tool_names
                             if n not in set(ALL_TOOL_NAMES))
            lines = [
                f"[{approach_name}] {phase} session tool surface "
                f"({len(tool_names)} total):"
            ]
            for n in static:
                lines.append(f"  static   {n}")
            for n in dynamic:
                lines.append(f"  dynamic  {n}")
            logger.info("\n".join(lines))

        session: SessionManagerProtocol
        if config.use_docker_sandbox:
            from predicators.agent_sdk.docker_sandbox import \
                DockerSessionManager  # pylint: disable=import-outside-toplevel
            session = DockerSessionManager(
                system_prompt=self._get_agent_system_prompt(),
                log_dir=self._get_log_dir(),
                model_name=config.model_name,
                tool_context=self._tool_context,
                tool_names=tool_names,
                image=config.docker_image,
                extra_reference_files=self._get_sandbox_reference_files(),
                phase=phase,
                config=config,
            )
        elif config.use_local_sandbox:
            from predicators.agent_sdk.local_sandbox import \
                LocalSandboxSessionManager  # pylint: disable=import-outside-toplevel
            session = LocalSandboxSessionManager(
                system_prompt=self._get_agent_system_prompt(),
                log_dir=self._get_log_dir(),
                model_name=config.model_name,
                tool_context=self._tool_context,
                tool_names=tool_names,
                extra_reference_files=self._get_sandbox_reference_files(),
                phase=phase,
                config=config,
            )
        else:
            from claude_agent_sdk import \
                create_sdk_mcp_server  # pylint: disable=import-outside-toplevel

            tools = create_mcp_tools(self._tool_context, tool_names=tool_names)
            mcp_server = create_sdk_mcp_server(
                name="predicator_tools",
                version="1.0.0",
                tools=tools,
            )

            session = AgentSessionManager(
                system_prompt=self._get_agent_system_prompt(),
                mcp_server=mcp_server,
                log_dir=self._get_log_dir(),
                model_name=config.model_name,
                allowed_tools=get_allowed_tool_list(tool_names),
                tool_context=self._tool_context,
                config=config,
            )

        self._agent_session = session
        if self._agent_session_id is not None:
            session.session_id = self._agent_session_id

        # Save system prompt to log directory. Suffix with the phase tag
        # so solve and synthesis prompts don't overwrite each other across
        # phase switches.
        log_dir = self._get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        prompt_path = os.path.join(log_dir, f"system_prompt_{phase}.md")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(self._get_agent_system_prompt())

    def _get_log_dir(self) -> str:
        """Return the log directory, using the approach name."""
        if hasattr(CFG, 'log_file') and CFG.log_file:
            return CFG.log_file
        name = (
            self.get_name()  # type: ignore[attr-defined]
            if hasattr(self, 'get_name') else self._log_subdir)
        return os.path.join("logs", name)

    def _close_agent_session(self) -> None:
        """Close and discard the current agent session, if one exists."""
        if self._agent_session is None:
            return
        session = self._agent_session
        self._agent_session = None
        try:
            run_async_sync(session.close())
        except Exception:  # pylint: disable=broad-except
            pass

    def _query_agent_sync(self, message: str,
                          **query_kwargs: Any) -> List[Dict[str, Any]]:
        """Synchronous wrapper for async agent query.

        Extra kwargs (e.g. ``kind="learn"``) are forwarded to the
        session's ``query`` method for log-file tagging.
        """
        self._ensure_agent_session()
        assert self._agent_session is not None
        return run_query_sync(self._agent_session, message, **query_kwargs)

    def _create_agent_explorer(
        self,
        predicates: Set[Predicate],
        options: Set[ParameterizedOption],
        name: str = "agent_plan",
    ) -> BaseExplorer:
        """Create an agent explorer with tool_context and agent_session."""
        self._ensure_agent_session()
        return create_explorer(
            name,
            predicates,
            options,
            self._types,
            self._action_space,
            self._train_tasks,
            tool_context=self._tool_context,
            agent_session=self._agent_session,
        )
