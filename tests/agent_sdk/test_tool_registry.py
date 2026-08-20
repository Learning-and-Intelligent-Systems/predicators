"""Smoke tests for the agent-SDK tool registry.

Guards against drift between the ``@tool("name", ...)`` decorators
inside the factory functions and the name tuples exported from
``predicators.agent_sdk.tools``.  If a new tool is added (or renamed)
without updating the constants, these tests fail.
"""
# pylint: disable=protected-access,import-outside-toplevel
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, List, Optional, Set, cast

from predicators.agent_sdk.tools import ALL_TOOL_NAMES, BUILTIN_TOOLS, \
    MCP_SERVER_NAME, PREDICATE_SYNTHESIS_TOOL_NAMES, SYNTHESIS_TOOL_NAMES, \
    ToolContext, create_mcp_tools, create_predicate_synthesis_tools, \
    create_synthesis_tools, get_allowed_tool_list, list_session_tool_names
from predicators.approaches.agent_session_mixin import AgentSessionMixin


def _required_names(names: Optional[List[str]]) -> List[str]:
    """Narrow an Optional tool-name list (None means "all static MCP tools",
    which these tests never exercise)."""
    assert names is not None
    return names


def _names(tools: Iterable[Any]) -> Set[str]:
    return {getattr(t, "name", "") for t in tools}


def test_create_mcp_tools_matches_all_tool_names() -> None:
    """``create_mcp_tools`` exposes exactly the ``ALL_TOOL_NAMES`` names.

    That holds when the config opts into every conditionally-built tool;
    without the opt-ins the surface must NOT carry them (baseline arms
    would otherwise gain the code-execution tool silently).
    Conditionally-built tools MUST still appear in ``ALL_TOOL_NAMES``:
    the session-open sanity check classifies any declared name outside
    it as a dynamic tool and asserts when no builder attached it
    (run_20260718_124622 failed every solve query because
    ``record_journal`` was missing from the roster).
    """
    from predicators import utils
    utils.reset_config({
        "agent_planner_use_explore_python": True,
        "agent_solve_use_journal": True,
    })
    tools = create_mcp_tools(ToolContext())
    assert _names(tools) == set(ALL_TOOL_NAMES)
    utils.reset_config({
        "agent_planner_use_explore_python": False,
        "agent_solve_use_journal": False,
    })
    tools = create_mcp_tools(ToolContext())
    disabled = {"explore_python", "record_journal"}
    assert _names(tools) == set(ALL_TOOL_NAMES) - disabled


def test_create_synthesis_tools_matches_constant(tmp_path) -> None:
    """``create_synthesis_tools`` builds exactly the synthesis name tuple, plus
    the ``sim.fit`` backend as a non-tool callable."""
    toolkit = create_synthesis_tools(
        exec_ns={},
        base_pred_triples=[],
        inferred_residual_features={},
        simulator_file=str(tmp_path / "simulator.py"),
        versions_dir=str(tmp_path / "simulator_versions"),
        approach=None,
    )
    assert _names(toolkit.tools) == set(SYNTHESIS_TOOL_NAMES)
    assert callable(toolkit.fit_runner)
    assert callable(toolkit.residuals_runner)
    # No simulator file yet: the runners report it instead of raising.
    report = toolkit.fit_runner()
    assert "simulator.py" in report and "Write" in report
    report = toolkit.residuals_runner()
    assert "simulator.py" in report and "Write" in report


def test_sysid_fit_gate_traj_idxs_vs_fixed(tmp_path) -> None:
    """On the PHYSICAL_PARAMS path, ``fixed`` is rejected (pinning goes through
    the param's bounds in the declaration) while ``traj_idxs`` passes.

    the gate as an exploratory subset fit - with no approach bound it then
    stops at the no-approach error rather than the gate.
    """
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text("PHYSICAL_PARAMS = [\n"
                        "    ParamSpec('lateral_friction', 0.2, lo=0.01, "
                        "hi=1.0)\n"
                        "]\n")
    toolkit = create_synthesis_tools(
        exec_ns={},
        base_pred_triples=[],
        inferred_residual_features={},
        simulator_file=str(sim_file),
        versions_dir=str(tmp_path / "simulator_versions"),
        approach=None,
    )
    out = toolkit.fit_runner(fixed={"lateral_friction": 0.3})
    assert "fixed is not supported" in out
    assert "narrowing" in out
    out = toolkit.fit_runner(traj_idxs=[0])
    assert "fixed is not supported" not in out
    assert "requires a bound approach" in out
    # With an approach bound, an empty subset is refused before any fit
    # machinery runs (the guard precedes all approach attribute access,
    # so a bare stub suffices).
    toolkit = create_synthesis_tools(
        exec_ns={},
        base_pred_triples=[],
        inferred_residual_features={},
        simulator_file=str(sim_file),
        versions_dir=str(tmp_path / "simulator_versions"),
        approach=cast(Any, SimpleNamespace()),
    )
    out = toolkit.fit_runner(traj_idxs=[])
    assert "traj_idxs is empty" in out


def test_create_predicate_synthesis_tools_matches_constant(tmp_path) -> None:
    """Predicate-synthesis builder matches the predicate-synthesis name
    tuple."""
    approach_stub = SimpleNamespace(_fitted_params={})
    tools = create_predicate_synthesis_tools(
        predicates_file=str(tmp_path / "predicates.py"),
        predicates_versions_dir=str(tmp_path / "predicates_versions"),
        approach=approach_stub,
        trajectories=[],
    )
    assert _names(tools) == set(PREDICATE_SYNTHESIS_TOOL_NAMES)


def test_list_session_tool_names_defaults() -> None:
    """Default ``list_session_tool_names`` returns all MCP + builtin tools."""
    grouped = list_session_tool_names()
    assert grouped["mcp"] == list(ALL_TOOL_NAMES)
    assert grouped["extra"] == []
    assert grouped["builtin"] == list(BUILTIN_TOOLS)


def test_list_session_tool_names_filters_and_combines() -> None:
    """Filtered MCP names drop unknowns; ``extra_mcp_tools`` pass through."""
    fake = SimpleNamespace(name="run_python")
    grouped = list_session_tool_names(
        mcp_filter=["inspect_options", "not_a_tool", "inspect_trajectories"],
        extra_mcp_tools=[fake],
        include_builtin=False,
    )
    assert grouped == {
        "mcp": ["inspect_options", "inspect_trajectories"],
        "extra": ["run_python"],
    }


def test_synthesis_tool_names_default_is_empty() -> None:
    """No synthesis MCP filter by default — approaches with no synthesis phase
    get an empty allowlist for free."""
    obj = AgentSessionMixin()
    assert not obj._get_synthesis_tool_names()


def test_solve_and_synthesis_tool_names_are_independent() -> None:
    """Subclasses can declare disjoint solve / synthesis tool sets."""

    # pylint: disable=abstract-method
    class _Approach(AgentSessionMixin):

        def _get_solve_tool_names(self) -> Optional[List[str]]:
            return ["inspect_options", "evaluate_option_plan"]

        def _get_synthesis_tool_names(self) -> Optional[List[str]]:
            return ["inspect_trajectories", "run_python"]

    obj = _Approach()
    assert obj._get_solve_tool_names() == [
        "inspect_options", "evaluate_option_plan"
    ]
    assert obj._get_synthesis_tool_names() == [
        "inspect_trajectories", "run_python"
    ]


def test_get_allowed_tool_list_passes_dynamic_names_through() -> None:
    """The allowlist must include dynamic tool names verbatim — the declared
    list is the single source of truth, with no silent filtering against
    ``ALL_TOOL_NAMES``."""
    allowed = get_allowed_tool_list([
        "inspect_options",  # static
        "run_python",  # dynamic synthesis tool
        "evaluate_predicate_quality",  # dynamic predicate-synthesis
    ])
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    assert allowed == [
        f"{prefix}inspect_options",
        f"{prefix}run_python",
        f"{prefix}evaluate_predicate_quality",
    ]


def test_coercing_tool_accepts_numeric_strings() -> None:
    """Numeric strings are coerced for integer/number schema properties.

    Coercion happens before the handler runs - harness-side validation
    used to hard-reject '0' for an integer arg, costing agents whole
    tools (inspect_trajectories went 0-for-6 in one audited run).
    """
    import asyncio

    from predicators.agent_sdk.tools.results import _make_coercing_tool

    captured: dict = {}

    def fake_tool(name: str, description: str, schema: Any) -> Any:
        del description

        def deco(fn: Any) -> Any:
            captured["schema"] = schema
            return SimpleNamespace(name=name, handler=fn)

        return deco

    ctool = _make_coercing_tool(fake_tool)

    @ctool(
        "t", "d", {
            "type": "object",
            "properties": {
                "idx": {
                    "type": "integer"
                },
                "frac": {
                    "type": "number"
                },
                "label": {
                    "type": "string"
                },
            },
        })
    async def handler(args: dict) -> dict:
        return {"args": args}

    schema = captured["schema"]
    assert schema["properties"]["idx"]["type"] == ["integer", "string"]
    assert schema["properties"]["frac"]["type"] == ["number", "string"]
    assert schema["properties"]["label"]["type"] == "string"
    out = asyncio.new_event_loop().run_until_complete(
        handler.handler({
            "idx": "3",
            "frac": "0.5",
            "label": "x"
        }))
    assert out["args"] == {"idx": 3, "frac": 0.5, "label": "x"}
    bad = asyncio.new_event_loop().run_until_complete(
        handler.handler({"idx": "abc"}))
    assert bad.get("is_error")
    # A schema with no numeric props passes through untouched.
    @ctool("u", "d", {
        "type": "object",
        "properties": {
            "s": {
                "type": "string"
            }
        }
    })
    async def handler2(args: dict) -> dict:
        return args

    assert captured["schema"]["properties"]["s"]["type"] == "string"


def test_resolve_task_evaluator_reads_task_field() -> None:
    """``_resolve_task_evaluator`` reads ``Task.evaluator`` off the context's
    train tasks / current task (the evaluator rides on the Task itself)."""
    import numpy as np

    from predicators.agent_sdk.tools.verdicts import _resolve_task_evaluator
    from predicators.structs import Object, State, Task, TaskEvaluator, Type

    cup_type = Type("cup_type", ["f"])
    init = State({Object("cup", cup_type): np.array([0.0])})
    evaluator = TaskEvaluator(set())
    with_eval = Task(init, set(), evaluator=evaluator)
    without_eval = Task(init, set())

    ctx = ToolContext()
    ctx.train_tasks = [with_eval, without_eval]
    ctx.current_task = with_eval
    assert _resolve_task_evaluator(ctx, 0) is evaluator
    assert _resolve_task_evaluator(ctx, 1) is None
    assert _resolve_task_evaluator(ctx, 2) is None  # out of range
    assert _resolve_task_evaluator(ctx, "current") is evaluator
    assert _resolve_task_evaluator(ctx, None) is evaluator
    ctx.current_task = None
    assert _resolve_task_evaluator(ctx, None) is None


def test_agent_render_resolution() -> None:
    """Agent-facing renders are capped at agent_sdk_image_max_px on the long
    side by scoping down the camera resolution."""
    from predicators import utils
    from predicators.agent_sdk.tools import agent_render_resolution
    from predicators.settings import CFG

    utils.reset_config({
        "agent_sdk_image_max_px": 512,
        "pybullet_camera_width": 900,
        "pybullet_camera_height": 450,
    })
    with agent_render_resolution():
        # Longest side capped, aspect ratio preserved.
        assert CFG.pybullet_camera_width == 512
        assert CFG.pybullet_camera_height == 256
    # Restored on exit.
    assert CFG.pybullet_camera_width == 900
    assert CFG.pybullet_camera_height == 450
    # Restored even when the body raises.
    try:
        with agent_render_resolution():
            raise RuntimeError
    except RuntimeError:
        pass
    assert CFG.pybullet_camera_width == 900
    # Cameras already within the cap are untouched (never upscales).
    utils.reset_config({
        "agent_sdk_image_max_px": 512,
        "pybullet_camera_width": 300,
        "pybullet_camera_height": 300,
    })
    with agent_render_resolution():
        assert CFG.pybullet_camera_width == 300
    # 0 disables the cap.
    utils.reset_config({
        "agent_sdk_image_max_px": 0,
        "pybullet_camera_width": 900,
        "pybullet_camera_height": 900,
    })
    with agent_render_resolution():
        assert CFG.pybullet_camera_width == 900


def test_explore_python_replaces_refine() -> None:
    """When explore_python is on, the tools it subsumes are dropped unless
    agent_planner_explore_python_keep_replaced_tools asks for both."""
    from predicators import utils
    from predicators.approaches.agent_model_based_approach import \
        AgentModelBasedApproach
    base = {
        "env": "cover",
        "approach": "agent_bilevel",
        "agent_planner_use_simulator": True,
    }
    obj = object.__new__(AgentModelBasedApproach)

    utils.reset_config(base)
    names = _required_names(obj._get_solve_tool_names())
    assert "refine_plan_sketch" in names
    assert "explore_python" not in names

    utils.reset_config({**base, "agent_planner_use_explore_python": True})
    names = _required_names(obj._get_solve_tool_names())
    assert "explore_python" in names
    assert "refine_plan_sketch" not in names  # subsumed: sim.refine

    utils.reset_config({
        **base, "agent_planner_use_explore_python": True,
        "agent_planner_explore_python_keep_replaced_tools": True
    })
    names = _required_names(obj._get_solve_tool_names())
    assert "explore_python" in names
    assert "refine_plan_sketch" in names


def test_synthesis_tool_names_explore_python() -> None:
    """Synthesis sessions never surface explore_python: the probe rides inside
    run_python's namespace (one exec namespace per session) and the inspect
    digests are prompt-injected.

    Fitting, residual reports, plan validation, and scene work are probe
    methods (``sim.fit`` / ``sim.residuals`` / ``sim.refine`` /
    ``sim.run`` / ``sim.reset`` + ``sim.render``), not tools, so the
    roster carries only ``run_python`` (+ per-arm evaluators).
    """
    from predicators import utils
    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    from predicators.approaches.agent_sim_predicate_invention_approach import \
        AgentSimPredicateInventionApproach

    sim_learn = object.__new__(AgentSimLearningApproach)
    sim_learn._do_synthesize_samplers = False
    invention = object.__new__(AgentSimPredicateInventionApproach)
    invention._do_synthesize_samplers = False

    for use_probe_tool in (False, True):
        utils.reset_config(
            {"agent_planner_use_explore_python": use_probe_tool})
        names = _required_names(sim_learn._get_synthesis_tool_names())
        assert "explore_python" not in names  # probe rides inside run_python
        # Digests replaced the inspect tools on every synthesis surface.
        assert not any(n.startswith("inspect_") for n in names)
        assert "run_python" in names
        names = _required_names(invention._get_synthesis_tool_names())
        assert "explore_python" not in names
        assert "evaluate_plan_refinement" not in names  # sim.refine + sim.run
        assert "evaluate_step_fit" not in names  # sim.fit
        assert "evaluate_predicate_quality" in names

    # On the solve side the probe subsumes the remaining inspect tools
    # (trajectories in its namespace, sim.task for the task digest).
    utils.reset_config({
        "env": "cover",
        "approach": "agent_sim_predicate_invention",
        "agent_planner_use_simulator": True,
        "agent_planner_use_explore_python": True,
    })
    names = _required_names(invention._get_solve_tool_names())
    assert "explore_python" in names
    assert not any(n.startswith("inspect_") for n in names)

    # Without the probe, the solve roster keeps the trajectory/task
    # inspect tools (there is no namespace to subsume them into) but
    # never inspect_options/inspect_types (digests are in the prompt).
    utils.reset_config({
        "env": "cover",
        "approach": "agent_sim_predicate_invention",
        "agent_planner_use_simulator": True,
        "agent_planner_use_explore_python": False,
    })
    names = _required_names(invention._get_solve_tool_names())
    assert "inspect_trajectories" in names
    assert "inspect_train_tasks" in names
    assert "inspect_options" not in names
    assert "inspect_types" not in names
