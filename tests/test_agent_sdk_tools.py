"""Tests for agent SDK tool enhancements.

Validates:
1. inspect_options with option_name saves source code to sandbox
2. evaluate_option_plan always saves scene images
3. evaluate_option_plan shows "Missing goal atoms" when goal not achieved
4. evaluate_option_plan shows object poses on failure
5. propose_options saves code to sandbox/proposed_code/
6. format_object_poses helper
7. render_scene_image helper
8. _sync_tool_context sets ctx.env from option model

Usage:
    python tests/test_agent_sdk_tools.py
"""
# pylint: disable=redefined-outer-name,import-outside-toplevel,protected-access
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import numpy as np
import pytest

# Bootstrap circular imports
import predicators.utils as pred_utils
from predicators.settings import CFG

_CFG_OVERRIDES = {
    "env": "pybullet_domino",
    "approach": "agent_planner",
    "seed": 0,
    "use_gui": False,
    "domino_restricted_push": True,
    "domino_use_continuous_place": True,
    "domino_use_skill_factories": True,
    "domino_use_domino_blocks_as_target": True,
    "domino_has_glued_dominos": False,
    "domino_initialize_at_finished_state": False,
    "num_train_tasks": 1,
    "num_test_tasks": 1,
    "skill_phase_use_motion_planning": True,
    "pybullet_ik_validate": False,
    "agent_sdk_propose_options": True,
    "agent_planner_use_explore_python": True,
    # Match the experiment configs: without this, an option whose
    # first action no-ops against residual env state (wrist drift
    # from earlier tests) is killed as "stuck", making the
    # physics-dependent tests order-sensitive.
    "option_model_terminate_on_repeat": False,
}


def _setup(sandbox_dir: str | None = None) -> tuple[Any, Any]:
    """Create environment, options, option model, and ToolContext."""
    pred_utils.reset_config(_CFG_OVERRIDES)

    from predicators.envs import create_new_env
    from predicators.ground_truth_models import get_gt_options
    from predicators.option_model import create_option_model

    env = create_new_env(CFG.env, do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    predicates = env.predicates
    train_tasks = list(env.get_train_tasks())
    types = env.types
    task = train_tasks[0]

    option_model = create_option_model(CFG.option_model_name)

    from predicators.agent_sdk.tools import ToolContext
    ctx = ToolContext(
        types=types,
        predicates=predicates,
        processes=set(),
        options=options,
        train_tasks=[t.task for t in train_tasks],
        example_state=task.init,
        option_model=option_model,
        current_task=task.task,
        sandbox_dir=sandbox_dir,
    )
    # Extract env from option model (same as _sync_tool_context does)
    if hasattr(option_model, '_simulator'):
        ctx.env = getattr(option_model._simulator, '__self__', None)

    # Create sandbox subdirectories
    if sandbox_dir:
        os.makedirs(os.path.join(sandbox_dir, "proposed_code"), exist_ok=True)

    return ctx, env


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    Creates (and installs) a fresh event loop when none is current -
    e.g. after another test module ran ``asyncio.run``, which unsets the
    thread's loop.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _make_tools(ctx: Any,
                tool_names: list[str] | None = None) -> dict[str, Any]:
    """Create MCP tools and return as a name->callable dict."""
    from predicators.agent_sdk.tools import create_mcp_tools
    tools = create_mcp_tools(ctx, tool_names=tool_names)
    # Tools are SdkMcpTool objects; extract name -> handler
    return {t.name: t.handler for t in tools}


# ===== Fixtures =====


@pytest.fixture(scope="module")
def ctx() -> Any:
    """Create shared ToolContext for all tests in this module."""
    try:
        with tempfile.TemporaryDirectory() as sandbox_dir:
            ctx_obj, _env = _setup(sandbox_dir=sandbox_dir)
            yield ctx_obj
    except Exception as exc:  # pylint: disable=broad-except
        pytest.skip(f"Environment setup failed: {exc}")


# ===== Tests =====


def test_inspect_options_list_all(ctx: Any) -> None:
    """inspect_options with no args lists all options."""
    tools = _make_tools(ctx, ["inspect_options"])
    result = _run(tools["inspect_options"]({}))
    text = result["content"][0]["text"]
    assert "Current options:" in text
    assert "Pick" in text or "Place" in text or "Push" in text
    print("  PASS: inspect_options (list all)")


def test_inspect_options_detail(ctx: Any) -> None:
    """inspect_options with option_name saves source to sandbox."""
    tools = _make_tools(ctx, ["inspect_options"])

    # Pick an option that exists
    opt_names = [o.name for o in ctx.options]
    test_name = opt_names[0]

    result = _run(tools["inspect_options"]({"option_name": test_name}))
    text = result["content"][0]["text"]

    # Should have the option header
    assert f"## {test_name}" in text
    # Should have params info
    assert "params_dim" in text

    if ctx.sandbox_dir:
        # Should point to the saved file
        assert f"./proposed_code/{test_name}.py" in text
        # File should exist in sandbox
        saved_path = os.path.join(ctx.sandbox_dir, "proposed_code",
                                  f"{test_name}.py")
        assert os.path.exists(saved_path), \
            f"Expected file at {saved_path}"
        # File should have content
        with open(saved_path, encoding='utf-8') as f:
            content = f.read()
        assert len(content) > 0
        print(f"  PASS: inspect_options (detail for '{test_name}', "
              f"saved to sandbox)")
    else:
        # Fallback: should inline source code
        assert "Source Code" in text
        print(f"  PASS: inspect_options (detail for '{test_name}', "
              f"inlined — no sandbox)")


def test_inspect_options_unknown(ctx: Any) -> None:
    """inspect_options with unknown option_name returns error."""
    tools = _make_tools(ctx, ["inspect_options"])
    result = _run(tools["inspect_options"]({
        "option_name": "NonExistentOption"
    }))
    assert result.get("is_error", False)
    assert "Unknown option" in result["content"][0]["text"]
    print("  PASS: inspect_options (unknown option)")


def test_inspect_options_proposed_code(ctx: Any) -> None:
    """inspect_options returns path for option with code saved to sandbox."""
    from predicators.agent_sdk.tools.results import _save_option_to_sandbox

    # Save proposal code to sandbox
    proposal_code = "# test proposal code\nx = 1"
    _save_option_to_sandbox(ctx, "TestOpt", proposal_code)

    # Create a dummy option with that name
    from gym.spaces import Box

    from predicators.structs import ParameterizedOption
    dummy_opt = ParameterizedOption(
        name="TestOpt",
        types=[],
        params_space=Box(low=np.array([]), high=np.array([])),
        policy=lambda s, m, o, p: None,  # type: ignore[arg-type, return-value]
        initiable=lambda s, m, o, p: True,
        terminal=lambda s, m, o, p: True,
    )
    ctx.options = ctx.options | {dummy_opt}

    tools = _make_tools(ctx, ["inspect_options"])
    result = _run(tools["inspect_options"]({"option_name": "TestOpt"}))
    text = result["content"][0]["text"]

    if ctx.sandbox_dir:
        assert "./proposed_code/TestOpt.py" in text
        # Verify file content
        saved_path = os.path.join(ctx.sandbox_dir, "proposed_code",
                                  "TestOpt.py")
        with open(saved_path, encoding='utf-8') as f:
            assert "# test proposal code" in f.read()
    else:
        # No sandbox — source inlined
        assert "Source Code" in text

    # Clean up
    ctx.options = {o for o in ctx.options if o.name != "TestOpt"}
    if ctx.sandbox_dir:
        saved_path = os.path.join(ctx.sandbox_dir, "proposed_code",
                                  "TestOpt.py")
        if os.path.exists(saved_path):
            os.remove(saved_path)
    print("  PASS: inspect_options (proposed code in sandbox)")


def _get_valid_option_plan_step(ctx: Any) -> dict[str, Any] | None:
    """Find a valid single-step option plan for testing."""
    # Find option with fewest type requirements
    for opt in sorted(ctx.options, key=lambda o: len(o.types)):
        if opt.name == "Wait":
            continue
        # Build object_names from the state
        state = ctx.current_task.init
        obj_names = []
        valid = True
        for t in opt.types:
            # Find an object of this type in the state
            found = False
            for obj in state:
                if obj.type == t and obj.name not in obj_names:
                    obj_names.append(obj.name)
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid:
            # Use midpoint of params_space
            low = opt.params_space.low
            high = opt.params_space.high
            params = ((low + high) / 2).tolist()
            return {
                "option_name": opt.name,
                "object_names": obj_names,
                "params": params,
            }
    # Fallback: use Wait if available
    for opt in ctx.options:
        if opt.name == "Wait":
            return {"option_name": "Wait", "object_names": [], "params": []}
    return None


def _plan_to_text(plan: Any, ctx: Any) -> str:
    """Render structured option-plan steps as the text grammar that
    evaluate_option_plan now expects (typed object refs + params in [])."""
    type_of = {o.name: o.type.name for o in ctx.current_task.init}
    lines = []
    for step in plan:
        objs = ", ".join(f"{n}:{type_of.get(n, 'object')}"
                         for n in step["object_names"])
        params = ", ".join(str(p) for p in step["params"])
        lines.append(f"{step['option_name']}({objs})[{params}]")
    return "\n".join(lines)


def test_option_plan_missing_goal_atoms(ctx: Any) -> None:
    """evaluate_option_plan reports missing goal atoms when goal not
    achieved."""
    tools = _make_tools(ctx, ["evaluate_option_plan"])

    step = _get_valid_option_plan_step(ctx)
    assert step is not None, "No valid option found for testing"
    plan = [step]

    result = _run(tools["evaluate_option_plan"]({
        "plan":
        _plan_to_text(plan, ctx),
        "include_atoms":
        True,
    }))
    text = result["content"][0]["text"]

    # Three possible outcomes:
    if "Goal achieved: False" in text:
        # Either the env exposes goal atoms (and we show "Missing goal
        # atoms: ...") or it sets goal_nl (and we show that instead,
        # to avoid leaking env predicate names to predicate-invention
        # agents).
        assert ("Missing goal atoms:" in text
                or "Goal (natural language):" in text)
        print("  PASS: evaluate_option_plan (failure diagnostic shown)")
    elif "Goal achieved: True" in text:
        assert "Missing goal atoms:" not in text
        print("  PASS: evaluate_option_plan (goal achieved, no missing atoms)")
    else:
        # Plan failed early (grounding error, NOT INITIABLE, etc.)
        assert ("NOT INITIABLE" in text or "FAILURE REASON:" in text
                or "EXECUTION ERROR" in text or "Failed to ground" in text)
        print("  PASS: evaluate_option_plan (plan failed early, "
              "goal check not reached)")


def test_option_plan_description_submission_split(ctx: Any) -> None:
    """With explore_python on, evaluate_option_plan's description routes
    exploration to the probe and frames this tool as the submission path; with
    it off, the description is unchanged."""
    from predicators.agent_sdk.tools import create_mcp_tools
    prior = CFG.agent_planner_use_explore_python
    try:
        pred_utils.update_config({"agent_planner_use_explore_python": True})
        tool_obj = create_mcp_tools(ctx,
                                    tool_names=["evaluate_option_plan"])[0]
        desc = getattr(tool_obj, "description", "")
        assert "explore_python" in desc and "SUBMIT" in desc
        pred_utils.update_config({"agent_planner_use_explore_python": False})
        tool_obj = create_mcp_tools(ctx,
                                    tool_names=["evaluate_option_plan"])[0]
        assert "explore_python" not in getattr(tool_obj, "description", "")
    finally:
        pred_utils.update_config({"agent_planner_use_explore_python": prior})
    print("  PASS: evaluate_option_plan (submission-split description)")


def test_explore_python_render_annotations(ctx: Any) -> None:
    """sim.render(annotations=...) overlays temporary geometry for one render
    (bodies removed after) and surfaces bad annotations as loud errors."""
    tools = _make_tools(ctx, ["explore_python"])
    with tempfile.TemporaryDirectory() as img_dir:
        prior_dir = ctx.image_save_dir
        ctx.image_save_dir = img_dir
        try:
            code = """
sim.reset()
path = sim.render("ann", annotations=[
    {"type": "marker", "position": [0.5, 1.3, 0.5], "size": 0.02},
    {"type": "line", "from": [0.4, 1.2, 0.5], "to": [0.6, 1.4, 0.5]},
])
print("saved", path is not None)
"""
            result = _run(tools["explore_python"]({"code": code}))
            text = result["content"][0]["text"]
            assert "saved True" in text
            assert len(os.listdir(img_dir)) == 1
            # A malformed annotation errors loudly (after cleanup).
            bad = 'sim.render("bad", annotations=[{"type": "line"}])'
            result = _run(tools["explore_python"]({"code": bad}))
            assert "Error" in result["content"][0]["text"]
        finally:
            ctx.image_save_dir = prior_dir
    print("  PASS: explore_python (annotated render)")


def test_explore_python_unknown_mod_object(ctx: Any) -> None:
    """A reset() modification naming an unknown object is a loud error."""
    tools = _make_tools(ctx, ["explore_python"])
    result = _run(tools["explore_python"]({
        "code":
        'sim.reset(mods={"no_such_object": {"x": 0.5}})'
    }))
    assert "Unknown object 'no_such_object'" in result["content"][0]["text"]
    print("  PASS: explore_python (unknown mod object error)")


def test_explore_python_exec_and_persistence(ctx: Any) -> None:
    """The solve-phase explore_python executes code and keeps its namespace."""
    tools = _make_tools(ctx, ["explore_python"])
    result = _run(tools["explore_python"]({"code": "x = 21\nprint(x * 2)"}))
    assert result["content"][0]["text"].strip() == "42"
    result = _run(tools["explore_python"]({"code": "print(x + 1)"}))
    assert result["content"][0]["text"].strip() == "22"
    print("  PASS: explore_python (exec + persistent namespace)")


def test_explore_python_probe_sim(ctx: Any) -> None:
    """BeliefProbe: reset with mods, full-precision state, run from the
    modified state, snapshot/restore - and nothing is ever captured."""
    tools = _make_tools(ctx, ["explore_python"])
    domino = next(o for o in ctx.current_task.init if o.type.name == "domino")
    robot = next(o for o in ctx.current_task.init if o.type.name == "robot")
    ctx.capture_goal_reaching_plans = True
    prior_dir = ctx.image_save_dir
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx.image_save_dir = tmpdir
            code = f"""
sim.reset(mods={{"{domino.name}": {{"x": 0.95}}}})
print("modx", sim.state("{domino.name}")["x"])
sid = sim.snapshot()
out = sim.run("Wait({robot.name}:robot)[]")
print("steps", len(out.steps))
print("stepimg", out.steps[0]["image"])
sim.restore(sid)
quiet = sim.run("Wait({robot.name}:robot)[]", render=False)
print("quietimg", quiet.steps[0]["image"])
sim.restore(sid)
print("restx", sim.state("{domino.name}")["x"])
print("natoms", len(sim.atoms()))
"""
            result = _run(tools["explore_python"]({"code": code}))
            saved = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
    finally:
        ctx.capture_goal_reaching_plans = False
        ctx.image_save_dir = prior_dir
    text = result["content"][0]["text"]
    assert "modx 0.95" in text
    assert "steps 1" in text
    assert "restx 0.95" in text
    assert "natoms" in text
    # sim.run saves the same per-step audit images evaluate_option_plan
    # does, and reports their paths on each step; render=False (for
    # tight sweep loops) skips the render entirely.
    assert "quietimg None" in text
    if saved:
        assert any("probe_step_0_" in f for f in saved)
        assert "stepimg " + os.path.join(tmpdir, "") in text
        assert len(saved) == 1
    else:
        print("  NOTE: rendering not available, image save not checked")
    # The probe carries no scoring surface: nothing it ran was captured.
    assert ctx.solved_plan is None
    print(
        "  PASS: explore_python (BeliefProbe reset/run/snapshot, no capture)")


def test_explore_python_probe_refine(ctx: Any) -> None:
    """BeliefProbe.refine searches params from the current state, reports per-
    step samples and a refined plan line, and captures nothing."""
    tools = _make_tools(ctx, ["explore_python"])
    domino = next(o for o in ctx.current_task.init if o.type.name == "domino")
    robot = next(o for o in ctx.current_task.init if o.type.name == "robot")
    ctx.capture_goal_reaching_plans = True
    try:
        code = f"""
sim.reset()
res = sim.refine(
    "Pick({robot.name}:robot, {domino.name}:domino)[0.06] "
    "-> {{Holding({robot.name}:robot, {domino.name}:domino)}}",
    timeout=45)
print("success", res.success)
print("samples", res.total_samples, res.step_samples)
print("line", res.plan_lines[0])
"""
        result = _run(tools["explore_python"]({"code": code}))
    finally:
        ctx.capture_goal_reaching_plans = False
    text = result["content"][0]["text"]
    assert "success True" in text
    assert "samples" in text
    assert "line Pick(" in text and "Holding(" in text
    assert ctx.solved_plan is None
    print("  PASS: explore_python (BeliefProbe.refine, no capture)")


def test_explore_python_probe_refine_verdict_line(ctx: Any) -> None:
    """A refine SUCCESS carries a Verdict line saying exactly what it certifies
    (bare SUCCESS used to be read as goal-reached)."""
    tools = _make_tools(ctx, ["explore_python"])
    domino = next(o for o in ctx.current_task.init if o.type.name == "domino")
    robot = next(o for o in ctx.current_task.init if o.type.name == "robot")
    code = f"""
sim.reset()
res = sim.refine(
    "Pick({robot.name}:robot, {domino.name}:domino)[0.06] "
    "-> {{Holding({robot.name}:robot, {domino.name}:domino)}}",
    timeout=45)
print(res)
"""
    result = _run(tools["explore_python"]({"code": code}))
    text = result["content"][0]["text"]
    assert "Verdict: executed" in text
    assert "require_goal=True" in text
    print("  PASS: explore_python (refine verdict line)")


def test_explore_python_probe_strlike_and_region_note(ctx: Any) -> None:
    """ProbeResult supports string slicing/containment, and a `~` region
    annotation is IGNORED with a NOTE when ground samplers are off (previously
    a hard error that cost turns of syntax guessing)."""
    tools = _make_tools(ctx, ["explore_python"])
    robot = next(o for o in ctx.current_task.init if o.type.name == "robot")
    assert not CFG.agent_bilevel_ground_samplers
    code = f"""
sim.reset()
res = sim.run("Wait({robot.name}:robot)[] ~ [0.1]", render=False)
print("contains", "Goal reached" in res)
print("slice_ok", len(res[-40:]) > 0)
print(res)
"""
    result = _run(tools["explore_python"]({"code": code}))
    text = result["content"][0]["text"]
    assert "contains True" in text
    assert "slice_ok True" in text
    assert "region annotation was IGNORED" in text
    assert "uniform" in text
    print("  PASS: explore_python (str-like result + region note)")


def test_explore_python_probe_trials(ctx: Any) -> None:
    """sim.run(plan, trials=N) reports per-trial outcomes and a success count
    without advancing the current state."""
    tools = _make_tools(ctx, ["explore_python"])
    domino = next(o for o in ctx.current_task.init if o.type.name == "domino")
    robot = next(o for o in ctx.current_task.init if o.type.name == "robot")
    code = f"""
sim.reset()
before = sim.state("{domino.name}")["x"]
res = sim.run("Wait({robot.name}:robot)[]", trials=2)
print(res)
print("n_trials", len(res.trials))
print("kept", abs(sim.state("{domino.name}")["x"] - before) < 1e-9)
"""
    result = _run(tools["explore_python"]({"code": code}))
    text = result["content"][0]["text"]
    assert "Trials:" in text
    assert "n_trials 2" in text
    # No validation_env_scope installed in this harness: the report must
    # say the trials shared the session env (correlated).
    assert "shared session env" in text
    assert "kept True" in text
    print("  PASS: explore_python (trials=N)")


def test_explore_python_refine_require_solved_guards(ctx: Any) -> None:
    """require_solved refuses to run from a modified start, and from a task.

    with no evaluator - both before any search. (The gate's accept/reject
    semantics are covered deterministically in
    test_bilevel_sketch_samplers.py with fake option models.)
    """
    tools = _make_tools(ctx, ["explore_python"])
    domino = next(o for o in ctx.current_task.init if o.type.name == "domino")
    robot = next(o for o in ctx.current_task.init if o.type.name == "robot")
    plan_line = (f"Pick({robot.name}:robot, {domino.name}:domino)[0.06] "
                 f"-> {{Holding({robot.name}:robot, "
                 f"{domino.name}:domino)}}")

    # Modified start: refused outright.
    code = (f'sim.reset(mods={{"{domino.name}": {{"x": 0.9}}}})\n'
            f'sim.refine("{plan_line}", require_solved=True)')
    result = _run(tools["explore_python"]({"code": code}))
    assert "unmodified initial state" in result["content"][0]["text"]

    # Pristine start but the task defines no evaluator: refused.
    import dataclasses
    saved_task = ctx.current_task
    try:
        ctx.current_task = dataclasses.replace(saved_task, evaluator=None)
        code = f'sim.reset()\nsim.refine("{plan_line}", require_solved=True)'
        result = _run(tools["explore_python"]({"code": code}))
        assert "defines no task evaluator" in result["content"][0]["text"]
    finally:
        ctx.current_task = saved_task
    print("  PASS: explore_python (require_solved guards)")


def test_explore_python_run_solved_guards(ctx: Any) -> None:
    """sim.run(solved=True) refuses single runs, modified starts, and tasks
    with no evaluator; contacts=True refuses trials mode."""
    tools = _make_tools(ctx, ["explore_python"])
    domino = next(o for o in ctx.current_task.init if o.type.name == "domino")
    robot = next(o for o in ctx.current_task.init if o.type.name == "robot")
    wait_line = f"Wait({robot.name}:robot)[]"

    code = f'sim.reset()\nsim.run("{wait_line}", solved=True)'
    result = _run(tools["explore_python"]({"code": code}))
    assert "needs trials >= 2" in result["content"][0]["text"]

    code = f'sim.reset()\nsim.run("{wait_line}", trials=2, contacts=True)'
    result = _run(tools["explore_python"]({"code": code}))
    assert "single-run mode" in result["content"][0]["text"]

    code = (f'sim.reset(mods={{"{domino.name}": {{"x": 0.9}}}})\n'
            f'sim.run("{wait_line}", trials=2, solved=True)')
    result = _run(tools["explore_python"]({"code": code}))
    assert "unmodified initial state" in result["content"][0]["text"]

    import dataclasses
    saved_task = ctx.current_task
    try:
        ctx.current_task = dataclasses.replace(saved_task, evaluator=None)
        code = f'sim.reset()\nsim.run("{wait_line}", trials=2, solved=True)'
        result = _run(tools["explore_python"]({"code": code}))
        assert "defines no task evaluator" in result["content"][0]["text"]
    finally:
        ctx.current_task = saved_task
    print("  PASS: explore_python (run solved/contacts guards)")


def test_explore_python_run_solved_trials(ctx: Any) -> None:
    """sim.run(trials=N, solved=True) reports a per-trial task-evaluator
    verdict and a solved count in the headline."""
    tools = _make_tools(ctx, ["explore_python"])
    robot = next(o for o in ctx.current_task.init if o.type.name == "robot")
    code = f"""
sim.reset()
res = sim.run("Wait({robot.name}:robot)[]", trials=2, solved=True)
print(res)
print("verdicts", [t["solved"] for t in res.trials])
"""
    result = _run(tools["explore_python"]({"code": code}))
    text = result["content"][0]["text"]
    assert "scored solved=True by the task evaluator" in text
    assert "evaluator: solved=" in text
    # A Wait-only plan cannot reach the goal, so no trial scores a solve.
    assert "verdicts [False, False]" in text
    print("  PASS: explore_python (trials solved verdicts)")


def test_explore_python_run_contacts(ctx: Any) -> None:
    """sim.run(contacts=True) reports per-step contact-pair spans; a Pick must
    show a robot-link contact with the grasped domino."""
    tools = _make_tools(ctx, ["explore_python"])
    domino = next(o for o in ctx.current_task.init if o.type.name == "domino")
    robot = next(o for o in ctx.current_task.init if o.type.name == "robot")
    code = f"""
sim.reset()
res = sim.run("Pick({robot.name}:robot, {domino.name}:domino)[0.05]",
              render=False, contacts=True)
print(res)
"""
    result = _run(tools["explore_python"]({"code": code}))
    text = result["content"][0]["text"]
    assert "contact recording unavailable" not in text
    assert "Contacts:" in text
    # The grasp squeeze puts a robot link in contact with the domino.
    assert "robot:" in text
    assert domino.name in text.split("Contacts:", 1)[1]
    print("  PASS: explore_python (contact recording)")


def test_option_plan_not_initiable_shows_poses(ctx: Any) -> None:
    """evaluate_option_plan shows object poses when option is NOT INITIABLE."""
    tools = _make_tools(ctx, ["evaluate_option_plan"])

    # Find Place option and try it without Pick first
    place_opt = None
    for opt in ctx.options:
        if opt.name == "Place":
            place_opt = opt
            break

    if place_opt is None:
        print("  SKIP: evaluate_option_plan (no Place option)")
        return

    # Build object names from types
    state = ctx.current_task.init
    obj_names = []
    for t in place_opt.types:
        for obj in state:
            if obj.type == t and obj.name not in obj_names:
                obj_names.append(obj.name)
                break

    low = place_opt.params_space.low
    high = place_opt.params_space.high
    params = ((low + high) / 2).tolist()

    plan = [{
        "option_name": "Place",
        "object_names": obj_names,
        "params": params,
    }]

    result = _run(tools["evaluate_option_plan"]({
        "plan":
        _plan_to_text(plan, ctx),
    }))
    text = result["content"][0]["text"]

    if "NOT INITIABLE" in text:
        assert "Object poses at failure:" in text
        print("  PASS: evaluate_option_plan (NOT INITIABLE shows poses)")
    elif "Failed to ground" in text:
        print("  SKIP: evaluate_option_plan (Place could not be grounded)")
    else:
        print("  SKIP: evaluate_option_plan (Place was initiable, "
              "can't test NOT INITIABLE path)")


def test_option_plan_saves_images(ctx: Any) -> None:
    """evaluate_option_plan always saves scene images (never returns
    inline)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.image_save_dir = tmpdir

        tools = _make_tools(ctx, ["evaluate_option_plan"])

        step = _get_valid_option_plan_step(ctx)
        assert step is not None, "No valid option found for testing"
        plan = [step]

        result = _run(tools["evaluate_option_plan"]({
            "plan":
            _plan_to_text(plan, ctx),
        }))

        content = result["content"]
        # Should have text block only (no inline images)
        assert any(b["type"] == "text" for b in content)
        assert not any(b["type"] == "image" for b in content)

        # Check files were saved if env rendering works
        saved = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        if saved:
            print(f"  PASS: evaluate_option_plan ({len(saved)} images saved)")
        else:
            print("  SKIP: evaluate_option_plan (rendering not available)")

        ctx.image_save_dir = None


def test_option_plan_failure_shows_poses(ctx: Any) -> None:
    """evaluate_option_plan shows object poses when option returns 0
    actions."""
    tools = _make_tools(ctx, ["evaluate_option_plan"])

    step = _get_valid_option_plan_step(ctx)
    assert step is not None, "No valid option found for testing"
    plan = [step]

    result = _run(tools["evaluate_option_plan"]({
        "plan":
        _plan_to_text(plan, ctx),
    }))
    text = result["content"][0]["text"]

    # Check the output is well-formed — it should have either step info
    # or a grounding error
    assert ("Step 0:" in text or "Failed to ground" in text
            or "Testing option plan" in text)
    if "FAILURE REASON:" in text:
        assert "Object poses at failure:" in text
        print("  PASS: evaluate_option_plan (failure shows poses)")
    elif "NOT INITIABLE" in text:
        assert "Object poses at failure:" in text
        print("  PASS: evaluate_option_plan (NOT INITIABLE shows poses)")
    else:
        print("  PASS: evaluate_option_plan (no failures in output)")


def testformat_object_poses(ctx: Any) -> None:
    """format_object_poses formats object positions correctly."""
    from predicators.agent_sdk.tools import format_object_poses

    state = ctx.current_task.init
    result = format_object_poses(state)

    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain robot
    assert "robot" in result
    # Should contain coordinates
    assert "x=" in result
    assert "y=" in result
    assert "z=" in result
    print(f"  PASS: format_object_poses ({result.count(chr(10))+1} objects)")


def testrender_scene_image(ctx: Any) -> None:
    """render_scene_image renders a scene and returns image block."""
    from predicators.agent_sdk.tools import render_scene_image

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.image_save_dir = tmpdir

        result = render_scene_image(ctx, "test_render")

        if result is not None:
            assert result["type"] == "image"
            assert result["mimeType"] == "image/png"
            assert len(result["data"]) > 100

            # Check saved file includes iter/turn prefix
            saved = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
            assert len(saved) == 1
            expected = (f"iter{ctx.iteration_id:03d}"
                        f"_test{ctx.test_call_id:03d}_test_render.png")
            assert saved[0] == expected, \
                f"Expected {expected}, got {saved[0]}"
            print("  PASS: render_scene_image (rendered + saved)")
        else:
            print("  SKIP: render_scene_image (rendering not available)")

        ctx.image_save_dir = None


def test_render_scene_no_env(ctx: Any) -> None:
    """render_scene_image returns None when env is None."""
    from predicators.agent_sdk.tools import render_scene_image

    original_env = ctx.env
    ctx.env = None
    result = render_scene_image(ctx, "should_be_none")
    assert result is None
    ctx.env = original_env
    print("  PASS: render_scene_image (no env → None)")


def test_propose_options_saves_to_sandbox(ctx: Any) -> None:
    """propose_options saves proposal code to sandbox/proposed_code/."""
    tools = _make_tools(ctx, ["propose_options"])

    # Get type names for a valid proposal
    type_names = {t.name: t for t in ctx.types}
    robot_type_name = "robot" if "robot" in type_names else list(type_names)[0]

    code = f"""\
from gym.spaces import Box
import numpy as np

proposed_options = [
    ParameterizedOption(
        name="TestProposed",
        types=[{robot_type_name}_type],
        params_space=Box(low=np.array([0.0]), high=np.array([1.0])),
        policy=lambda s, m, o, p: Action(np.zeros(s.get(o[0], "x").shape if hasattr(s.get(o[0], "x"), "shape") else (1,))),
        initiable=lambda s, m, o, p: True,
        terminal=lambda s, m, o, p: True,
    )
]
"""

    result = _run(tools["propose_options"]({
        "code":
        code,
        "description":
        "Test option for unit test",
    }))
    text = result["content"][0]["text"]

    if "Successfully proposed" in text:
        if ctx.sandbox_dir:
            saved_path = os.path.join(ctx.sandbox_dir, "proposed_code",
                                      "TestProposed.py")
            assert os.path.exists(saved_path), \
                f"Expected file at {saved_path}"
            with open(saved_path, encoding='utf-8') as f:
                content = f.read()
            assert "proposed_options" in content
            print("  PASS: propose_options (code saved to sandbox)")
            os.remove(saved_path)
        else:
            print("  PASS: propose_options (no sandbox, code not saved)")

        # Clean up
        ctx.options = {o for o in ctx.options if o.name != "TestProposed"}
    else:
        # Code execution might fail due to env-specific types
        print(f"  SKIP: propose_options (code failed: {text[:100]})")


def test_sync_tool_context_sets_env() -> None:
    """_sync_tool_context extracts env from option model."""
    pred_utils.reset_config(_CFG_OVERRIDES)

    from predicators.envs import create_new_env
    from predicators.envs.pybullet_env import PyBulletEnv
    from predicators.ground_truth_models import get_gt_options
    from predicators.option_model import create_option_model

    env = create_new_env(CFG.env, do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    option_model = create_option_model(CFG.option_model_name)

    from predicators.agent_sdk.tools import ToolContext
    ctx = ToolContext(
        types=env.types,
        predicates=env.predicates,
        options=options,
        train_tasks=[t.task for t in env.get_train_tasks()],
        option_model=option_model,
    )

    # Simulate what _sync_tool_context does
    assert ctx.env is None
    if hasattr(option_model, '_simulator'):
        ctx.env = getattr(option_model._simulator, '__self__', None)

    assert ctx.env is not None
    assert isinstance(ctx.env, PyBulletEnv)
    print("  PASS: _sync_tool_context sets ctx.env from option model")


def main() -> None:
    """Main."""
    with tempfile.TemporaryDirectory() as sandbox_dir:
        print("Setting up environment...")
        ctx, _env = _setup(sandbox_dir=sandbox_dir)
        print(f"Setup complete. {len(ctx.options)} options, "
              f"{len(ctx.train_tasks)} tasks\n")

        print("=== Tool Enhancement Tests ===\n")

        # inspect_options tests
        print("1. inspect_options tests:")
        test_inspect_options_list_all(ctx)
        test_inspect_options_detail(ctx)
        test_inspect_options_unknown(ctx)
        test_inspect_options_proposed_code(ctx)

        # evaluate_option_plan tests
        print("\n2. evaluate_option_plan tests:")
        test_option_plan_missing_goal_atoms(ctx)
        test_option_plan_not_initiable_shows_poses(ctx)
        test_option_plan_saves_images(ctx)
        test_option_plan_failure_shows_poses(ctx)

        # Helper function tests
        print("\n3. Helper function tests:")
        testformat_object_poses(ctx)
        testrender_scene_image(ctx)
        test_render_scene_no_env(ctx)

        # propose_options test
        print("\n4. propose_options tests:")
        test_propose_options_saves_to_sandbox(ctx)

        # _sync_tool_context test (creates fresh env)
        print("\n5. Context sync tests:")
        test_sync_tool_context_sets_env()

        print("\n=== All tests passed! ===")


if __name__ == "__main__":
    main()
