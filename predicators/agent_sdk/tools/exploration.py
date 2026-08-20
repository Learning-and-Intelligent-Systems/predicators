"""The explore_python solve-phase exploration tool."""
from typing import Any, Callable, Dict

from predicators.agent_sdk.config import ToolSurfaceConfig
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool
from predicators.agent_sdk.tools.results import _region_syntax_blurb


def belief_probe_blurb(synthesis_probe: bool) -> str:
    """The BeliefProbe surface description, shared by every prompt/tool surface
    that offers the probe.

    Solve sessions offer it through the standalone ``explore_python``
    tool; synthesis sessions bind the same facade as ``sim`` inside
    ``run_python``'s namespace. One renderer so the two descriptions
    cannot drift. ``synthesis_probe`` selects the candidate-simulator
    wording (task_idx-required resets, ``sim.fit``, and the
    fit/refine/forward-run validation protocol).
    """
    if synthesis_probe:
        sim_desc = (
            "`sim` (a BeliefProbe over the CANDIDATE simulator: your current "
            "simulator.py with freshly MCMC-fitted params, rebuilt "
            "automatically when the file changes - so probes always "
            "exercise what you just wrote; errors until a loadable "
            "simulator.py exists)")
        reset_desc = (
            "`sim.reset(task_idx, mods=None)` sets the current state "
            "to a train task's init (task_idx is required in this "
            "session), optionally with "
            "feature overrides (`mods={'obj': {'x': 1.05}}`); ")
        task_desc = ("`sim.task(task_idx)` describes a train task (goal, "
                     "objects, initial atoms and state) without touching "
                     "the current state; "
                     "`sim.fit(traj_idxs=None, fixed=None)` MCMC-fits "
                     "PARAM_SPECS (loaded fresh from simulator.py) against "
                     "the recorded data and returns the report (SSE "
                     "init->fit, fitted values, identifiability when "
                     "PHYSICAL_PARAMS is declared). No arguments = the "
                     "CANONICAL fit the probe deploys (system-ID values "
                     "applied to the planning env); traj_idxs (subset of "
                     "trajectories; on the system-ID path a "
                     "cross-trajectory consistency check) or fixed "
                     "({name: value} pins; rule params only) = "
                     "EXPLORATORY diagnostic, nothing published. Expensive "
                     "- call after meaningful rule edits, not in loops; "
                     "`sim.residuals(max_transitions=100, abs_tol=1e-4, "
                     "rel_tol=1e-3, num_worst_examples=3, "
                     "fit_params=False)` per-feature residual report for "
                     "the current simulator.py rules (mismatch counts, "
                     "mean/max abs error, vs-no-rule-baseline improvement, "
                     "worst-N example transitions) - the fast inner loop "
                     "for finding WHICH rule to fix. It is teacher-forced "
                     "(each step predicted from the RECORDED state), so it "
                     "CANNOT rule out a mis-set physical parameter: "
                     "compounding errors reset every step. "
                     "`sim.residuals(rollout=True, sweep_params=None, "
                     "phys_params=None, sweep_num_points=6)` is the "
                     "OPEN-LOOP counterpart: replays each recorded "
                     "trajectory's actions free-running and reports the "
                     "divergence at the current baselines. "
                     "sweep_params=[names] (or 'all') additionally sweeps "
                     "each named env-registry physical parameter across "
                     "its plausible range ('this data is explained Nx "
                     "better at a different friction'); "
                     "phys_params={name: value} instead scores ONE "
                     "hypothesized point and reports the SSE ratio vs the "
                     "baseline (the cheap primitive for your own targeted "
                     "sweeps). A sweep is slow (one fresh-env rollout per "
                     "candidate per segment, minutes for the full "
                     "registry) but it is the ONLY residual view that can "
                     "see physical-parameter error - run one (e.g. "
                     "sweep_params='all') BEFORE deciding whether to "
                     "declare PHYSICAL_PARAMS, in either direction; ")
    else:
        sim_desc = "`sim` (a BeliefProbe over the belief simulator)"
        reset_desc = (
            "`sim.reset(task_idx=None, mods=None)` sets the current state "
            "to a task's init (current task by default), optionally with "
            "feature overrides (`mods={'obj': {'x': 1.05}}`); ")
        task_desc = ("`sim.task(task_idx=None)` describes a task - goal, "
                     "objects, initial atoms and state (current task by "
                     "default) - without touching the current state; ")
    return (f"{sim_desc}, `BeliefProbe()` "
            "(extra independent instances). BeliefProbe API: "
            f"{reset_desc}{task_desc}"
            "`sim.run(plan_text, render=True, trials=1, solved=False, "
            "contacts=False)` executes an option "
            "plan FROM THE CURRENT "
            "STATE (same grammar as evaluate_option_plan; print the result "
            "for per-step outcomes incl. saved per-step scene-image paths - "
            "view them with the Read tool; pass render=False inside tight "
            "sweep loops) and advances the state; `-> {subgoals}` "
            "annotations are CHECKED - each step's report lists annotated "
            "atoms that did not hold in its post-state, so one continuous "
            "run of a refined plan is the forward-validation pass (a "
            "refine-pass that diverges here means a rule is more "
            "permissive than the env); trials=N repeats the plan "
            "N times (fresh physics per trial when available) and returns "
            "the per-trial outcomes + success count WITHOUT advancing the "
            "state - use it for reliability estimates instead of "
            "hand-rolled repeat loops (restore/rerun repeats share solver "
            "state and read optimistic); solved=True (trials>=2, from an "
            "unmodified reset() state) also scores each trial with the "
            "TASK EVALUATOR (per-trial solved/reward) - reaching the goal "
            "atoms is NOT the same as being scored a solve, so check this "
            "BEFORE submitting; contacts=True (single run) reports, per "
            "step, which robot links touched which objects and which "
            "object pairs touched, with action spans - use it to verify "
            "WHAT caused motion (e.g. an intended push vs. the arm "
            "brushing the scene); "
            "`sim.state()` / "
            "`sim.state('obj')` full-precision features; `sim.atoms()`; "
            "`sim.render(label, annotations=None)` saves an image "
            "(returns its path; Read it to view), "
            "optionally overlaying marker/line/rectangle dicts "
            "(`{'type': 'marker', 'position': [x, y, z], 'color': "
            "[r, g, b], 'size': s}`; lines use `from`/`to`, rectangles "
            "`min_corner`/`max_corner`) to check offsets and reference "
            "points visually; `sim.snapshot()` / "
            "`sim.restore(id)` bank and rewind states (use to re-try "
            "different actions from one setup, or resume after a fixed "
            "plan prefix without re-running it); "
            "`sim.refine(sketch_text, timeout=60, require_goal=False, "
            "require_solved=False)` runs "
            "backtracking parameter search FROM THE CURRENT STATE (same "
            "grammar/search as refine_plan_sketch"
            f"{_region_syntax_blurb()}; "
            "success = each step establishes its `-> {subgoals}` "
            "annotation, and the result's Verdict line states what it "
            "certifies) - refine a plan SUFFIX from a snapshot so the "
            "budget goes to the step that matters; the result reports "
            "best-found params even on timeout, per-step sample counts, and "
            "the deepest near-miss. require_solved=True (only from an "
            "unmodified reset() state) additionally requires the task "
            "evaluator to score the final rollout solved=True, rejecting "
            "goal-reaching-but-unscored candidates during the search.")


def _build_exploration_tools(ctx: ToolContext, _text_result: Callable,
                             tool: Callable) -> Dict[str, Any]:
    """Solve-phase ``explore_python`` over the BeliefProbe exploration facade.

    The namespace is the probe facade, numpy, and the collected real
    trajectories as read-only evidence (see ``build_probe_namespace`` -
    nothing evaluator-shaped beyond the probe's gated paths): the probe
    reuses the exact machinery behind ``evaluate_option_plan`` (same
    plan grammar, same option-model executor, same renderer) but
    carries no scoring surface - nothing run here can be captured as
    the answer, so it is safe to hand the agent as a freely composable
    physics probe. Built only when the session's config opts in: the
    ``tool_names=None`` legacy surface would otherwise grant every
    default-configured session an in-process exec tool. Synthesis
    sessions do not surface this tool at all - there the same facade is
    merged into ``run_python``'s namespace (one exec namespace per
    session; see ``_get_synthesis_tool_names``).
    """
    surface_cfg = ToolSurfaceConfig.from_cfg()
    if not surface_cfg.use_explore_python:
        return {}
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.belief_probe import build_probe_namespace

    submit_desc = (
        "EXPLORATORY "
        "ONLY: nothing run here is captured as your answer - preview "
        "the evaluator's verdict with sim.run(solved=True), then "
        "validate and submit the final plan via evaluate_option_plan "
        "from the true initial state.")
    explore_python = _make_python_exec_tool(
        tool,
        name="explore_python",
        description=(
            "Execute Python code for cheap physics/geometry exploration in "
            "a persistent namespace (variables survive across calls - "
            "define helpers and sweep loops once, reuse them). Available: " +
            belief_probe_blurb(synthesis_probe=False) +
            " Also bound: `np`; `trajectories` (the recorded REAL "
            "offline+online trajectories, read-only evidence - use them to "
            "check the belief model against what actually happened; each "
            "has `is_demo`, `train_task_idx`, `states`, `actions`) and "
            "`describe_trajectory(traj_idx, include_states=True, "
            "include_atoms=False, max_timesteps=10)` for a per-timestep "
            "digest of one of them. "
            "print() output is "
            "returned; oversize output is spilled to "
            "`tool_outputs/explore_python/` (Read/Grep it back). " +
            (f"Each call has a "
             f"{surface_cfg.explore_python_call_timeout:.0f}s "
             "wall-clock limit (checked between sim calls, plus a hard stop "
             "for sim-free code; printed output up to the stop is "
             "returned): budget sweeps accordingly - "
             "prefer coarse-to-fine over exhaustive grids, and print "
             "intermediate bests so partial results survive a stop. "
             if surface_cfg.explore_python_call_timeout > 0 else "") +
            f"{submit_desc}"),
        exec_ns=build_probe_namespace(ctx),
        sandbox_dir=ctx.sandbox_dir,
        text_result=_text_result,
        budget_ctx=ctx,
    )
    return {"explore_python": explore_python}
