"""Custom MCP tool definitions for the agent SDK approach.

This package replaces the former single-module ``tools.py``. Layout:

- ``registry``: tool-name rosters and the session tool-list surface.
- ``context``: ``ToolContext`` / ``PlanCapture`` shared session state.
- ``results``: tool-result formatting and sandbox-file helpers.
- ``sandbox_guard``: sandbox-escape screening for agent-supplied text.
- ``budget``: solve-attempt budget footer and watchdog.
- ``scene``: scene rendering and state-manipulation helpers.
- ``verdicts``: task-evaluator verdicts and ground-sampler loading.
- ``inspection`` / ``proposals`` / ``testing`` / ``planning`` /
  ``exploration`` / ``journal_tools``: the static MCP tool builders,
  assembled by ``assembly.create_mcp_tools``.
- ``snapshots``: versioned write-time snapshots of agent-edited files.
- ``python_exec``: shared python-exec core (run_python /
  explore_python).
- ``synthesis`` / ``params_view`` / ``predicate_synthesis`` /
  ``sampler_synthesis``: synthesis-session tool factories.

This facade re-exports the package's public surface (plus a few
underscore names kept for pre-split imports); new code should import
private helpers from their home submodules.
"""
# pylint: disable=unused-import
from predicators.agent_sdk.tools.assembly import create_mcp_tools
from predicators.agent_sdk.tools.context import PlanCapture, ToolContext
from predicators.agent_sdk.tools.params_view import _ParamsView
from predicators.agent_sdk.tools.predicate_synthesis import \
    create_predicate_synthesis_tools
from predicators.agent_sdk.tools.registry import ALL_TOOL_NAMES, \
    BUILTIN_TOOLS, EXPLORATION_TOOL_NAMES, INSPECTION_TOOL_NAMES, \
    JOURNAL_TOOL_NAMES, MCP_SERVER_NAME, PLANNING_TOOL_NAMES, \
    PREDICATE_SYNTHESIS_TOOL_NAMES, PROPOSAL_TOOL_NAMES, \
    RETRACTION_TOOL_NAMES, SAMPLER_SYNTHESIS_TOOL_NAMES, \
    SYNTHESIS_TOOL_NAMES, TESTING_TOOL_NAMES, explore_python_replaces_tools, \
    get_allowed_tool_list, list_session_tool_names
from predicators.agent_sdk.tools.results import _make_coercing_tool, \
    _make_spilling_text_result, _save_option_to_sandbox, \
    session_log_filename
from predicators.agent_sdk.tools.sampler_synthesis import \
    create_sampler_synthesis_tools
from predicators.agent_sdk.tools.sandbox_guard import \
    SANDBOX_HIDDEN_MODULES_PATTERN, SANDBOX_INTROSPECTION, \
    SANDBOX_SYSTEM_ROOTS, _screen_text_for_sandbox_escape
from predicators.agent_sdk.tools.scene import agent_render_resolution, \
    apply_state_modifications, draw_pybullet_annotation, format_object_poses, \
    render_pybullet_image, render_scene_image
from predicators.agent_sdk.tools.snapshots import _SnapshotTarget, \
    finalize_versioned_snapshot, make_write_snapshot_hook
from predicators.agent_sdk.tools.synthesis import create_synthesis_tools
from predicators.agent_sdk.tools.verdicts import _resolve_task_evaluator, \
    evaluate_states_with, load_ground_sampler_fns, make_solved_check

__all__ = [
    "ALL_TOOL_NAMES",
    "BUILTIN_TOOLS",
    "EXPLORATION_TOOL_NAMES",
    "INSPECTION_TOOL_NAMES",
    "JOURNAL_TOOL_NAMES",
    "MCP_SERVER_NAME",
    "PLANNING_TOOL_NAMES",
    "PREDICATE_SYNTHESIS_TOOL_NAMES",
    "PROPOSAL_TOOL_NAMES",
    "RETRACTION_TOOL_NAMES",
    "SAMPLER_SYNTHESIS_TOOL_NAMES",
    "SANDBOX_HIDDEN_MODULES_PATTERN",
    "SANDBOX_INTROSPECTION",
    "SANDBOX_SYSTEM_ROOTS",
    "SYNTHESIS_TOOL_NAMES",
    "TESTING_TOOL_NAMES",
    "PlanCapture",
    "ToolContext",
    "agent_render_resolution",
    "apply_state_modifications",
    "create_mcp_tools",
    "create_predicate_synthesis_tools",
    "create_sampler_synthesis_tools",
    "create_synthesis_tools",
    "draw_pybullet_annotation",
    "evaluate_states_with",
    "explore_python_replaces_tools",
    "finalize_versioned_snapshot",
    "format_object_poses",
    "get_allowed_tool_list",
    "list_session_tool_names",
    "load_ground_sampler_fns",
    "make_solved_check",
    "make_write_snapshot_hook",
    "render_pybullet_image",
    "render_scene_image",
    "session_log_filename",
]
