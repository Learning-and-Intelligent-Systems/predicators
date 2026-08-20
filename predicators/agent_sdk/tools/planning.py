"""Planning tools: generate_bilevel_plan, generate_abstract_plan, and
refine_plan_sketch."""
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from predicators import utils
from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.config import RefinementConfig
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.results import _error_result
from predicators.agent_sdk.tools.sandbox_guard import _scrub_host_paths
from predicators.agent_sdk.tools.tasks import _resolve_task
from predicators.agent_sdk.tools.verdicts import _belief_rollout_verdict, \
    _format_evaluator_verdict, _resolve_task_evaluator, \
    load_ground_sampler_fns, make_solved_check
from predicators.planning_with_processes import \
    run_task_plan_with_processes_once
from predicators.settings import CFG


def _build_planning_tools(ctx: ToolContext, _text_result: Callable,
                          tool: Callable) -> Dict[str, Any]:
    """Planning tools (generate bilevel / abstract plans)."""

    @tool(
        "generate_bilevel_plan",
        "Generate a concrete option plan using the bilevel planner. Returns "
        "grounded options with sampled continuous parameters, simulated "
        "step-by-step via the option model.",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index. Omit to use the current "
                    "solve-time task (if available)."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Planning timeout in seconds",
                    "default": 30
                },
            },
        },
    )
    async def generate_bilevel_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        task_idx = args.get("task_idx")
        timeout = args.get("timeout", 30)

        # Resolve task
        resolved, task_err = _resolve_task(ctx, task_idx)
        if task_err is not None:
            return task_err
        assert resolved is not None
        task = resolved.task
        task_label = resolved.description

        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates
        all_procs = ctx.processes | ctx.iteration_proposals.proposed_processes
        all_types = ctx.types | ctx.iteration_proposals.proposed_types

        # Get abstract plan
        try:
            plan, _atoms_seq, metrics = run_task_plan_with_processes_once(
                task,
                all_procs,
                all_preds,
                all_types,
                timeout,
                seed=CFG.seed,
                _task_planning_heuristic=CFG.process_task_planning_heuristic,
                max_horizon=float(CFG.horizon))
        except Exception as e:  # pylint: disable=broad-except
            return _text_result(f"Planning failed for {task_label}.\n"
                                f"Reason: {type(e).__name__}: {e}")

        if not plan:
            return _text_result(
                f"Planner returned empty plan for {task_label}.")

        # Sample options and simulate
        rng = np.random.default_rng(CFG.seed)
        state = task.init
        lines = [
            f"Bilevel plan for {task_label} "
            f"({len(plan)} steps, "
            f"{metrics.get('num_nodes_expanded', '?')} nodes expanded):"
        ]

        option_plan_lines = []
        for step_idx, ground_proc in enumerate(plan):
            try:
                option = ground_proc.sample_option(state, task.goal, rng)
            except Exception as e:  # pylint: disable=broad-except
                lines.append(
                    f"Step {step_idx}: {ground_proc.name}"
                    f"({', '.join(str(o) for o in ground_proc.objects)}) "
                    f"- SAMPLE FAILED: {e}")
                break

            # Format option
            obj_strs = ", ".join(f"{o.name}:{o.type.name}"
                                 for o in option.objects)
            params_str = ", ".join(f"{p:.4f}" for p in option.params)
            option_line = f"{option.name}({obj_strs})[{params_str}]"
            option_plan_lines.append(option_line)

            # Simulate
            if ctx.option_model is not None:
                try:
                    next_state, num_actions = \
                        ctx.option_model.get_next_state_and_num_actions(
                            state, option)
                    atoms_before = utils.abstract(state, all_preds)
                    atoms_after = utils.abstract(next_state, all_preds)
                    added = atoms_after - atoms_before
                    deleted = atoms_before - atoms_after
                    lines.append(
                        f"Step {step_idx}: {option_line} "
                        f"({num_actions} actions)"
                        f"\n  Added:   "
                        f"{{{', '.join(str(a) for a in sorted(added))}}}"
                        f"\n  Deleted: "
                        f"{{{', '.join(str(a) for a in sorted(deleted))}}}")
                    state = next_state
                except Exception as e:  # pylint: disable=broad-except
                    lines.append(f"Step {step_idx}: {option_line} "
                                 f"- SIMULATION ERROR: {e}")
                    break
            else:
                lines.append(f"Step {step_idx}: {option_line}")

        # Check goal via env-side classifiers so the result is robust
        # to invented predicates that don't reuse env names.
        if ctx.option_model is not None:
            goal_achieved = task.goal_holds(state)
            lines.append(f"\nGoal achieved: {goal_achieved}")

        lines.append("\n## Option Plan (copy-paste format):")
        lines.extend(option_plan_lines)

        return _text_result("\n".join(lines))

    @tool(
        "generate_abstract_plan",
        "Generate an abstract plan skeleton without continuous parameters. "
        "Returns option names and objects with parameter space info so you "
        "can fill in continuous parameters yourself.",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index. Omit to use the current "
                    "solve-time task (if available)."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Planning timeout in seconds",
                    "default": 30
                },
            },
        },
    )
    async def generate_abstract_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        task_idx = args.get("task_idx")
        timeout = args.get("timeout", 30)

        # Resolve task
        resolved, task_err = _resolve_task(ctx, task_idx)
        if task_err is not None:
            return task_err
        assert resolved is not None
        task = resolved.task
        task_label = resolved.description

        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates
        all_procs = ctx.processes | ctx.iteration_proposals.proposed_processes
        all_types = ctx.types | ctx.iteration_proposals.proposed_types

        try:
            plan, _atoms_seq, metrics = run_task_plan_with_processes_once(
                task,
                all_procs,
                all_preds,
                all_types,
                timeout,
                seed=CFG.seed,
                _task_planning_heuristic=CFG.process_task_planning_heuristic,
                max_horizon=float(CFG.horizon))
        except Exception as e:  # pylint: disable=broad-except
            return _text_result(f"Planning failed for {task_label}.\n"
                                f"Reason: {type(e).__name__}: {e}")

        if not plan:
            return _text_result(
                f"Planner returned empty plan for {task_label}.")

        lines = [
            f"Abstract plan for {task_label} "
            f"({len(plan)} steps, "
            f"{metrics.get('num_nodes_expanded', '?')} nodes expanded):",
            "",
        ]

        for step_idx, ground_proc in enumerate(plan):
            obj_strs = ", ".join(f"{o.name}:{o.type.name}"
                                 for o in ground_proc.option_objs)
            option = ground_proc.option
            params_dim = option.params_space.shape[0]
            if params_dim > 0:
                low = option.params_space.low.tolist()
                high = option.params_space.high.tolist()
                param_info = (f"  params_dim={params_dim}, "
                              f"low={low}, high={high}")
            else:
                param_info = "  (no continuous params)"
            lines.append(
                f"Step {step_idx}: {option.name}({obj_strs})\n{param_info}")

        # Include conditions for context
        lines.append("\n## Process conditions:")
        for step_idx, ground_proc in enumerate(plan):
            conds = ", ".join(
                str(a) for a in sorted(ground_proc.condition_at_start))
            adds = ", ".join(str(a) for a in sorted(ground_proc.add_effects))
            dels = ", ".join(
                str(a) for a in sorted(ground_proc.delete_effects))
            lines.append(f"Step {step_idx} ({ground_proc.name}):"
                         f"\n  Conditions: {{{conds}}}"
                         f"\n  Add effects: {{{adds}}}"
                         f"\n  Delete effects: {{{dels}}}")

        return _text_result("\n".join(lines))

    _gs_refine_doc = (
        "Confine a step's sampling with a GROUND SAMPLER after its "
        "`[params]`: either a region `~ [w1, w2]` (per-parameter "
        "half-widths; the exact center is tried first, then ALL further "
        "samples for the step are drawn uniformly from "
        "`[center - w, center + w]` clipped to the option's range - a zero "
        "width pins every draw to the center), or `~ my_sampler` naming an "
        "entry of `GROUND_SAMPLERS` in the sandbox file "
        "`ground_samplers.py`, which you Write/Edit and which is reloaded "
        "fresh on every call (each entry is "
        "`fn(state, subgoal_atoms, rng, objects) -> params`, so it can "
        "shape any state-dependent distribution). A ground sampler "
        "overrides any learned per-skill sampler for that step. "
        if RefinementConfig.from_cfg().ground_samplers else "")
    _gs_refine_plan_doc = (
        ", optionally followed by a ground sampler: `~ [w1, w2]` "
        "half-widths around those params, or `~ my_sampler` naming a "
        "GROUND_SAMPLERS entry in ground_samplers.py"
        if RefinementConfig.from_cfg().ground_samplers else "")

    @tool(
        "refine_plan_sketch",
        "FIND continuous parameters for a plan SKETCH: run a backtracking "
        "search over the option model, then — on success — forward-validate "
        "the refined plan. Unlike evaluate_option_plan (which runs your EXACT "
        "params with no search), this takes a sketch and lets the search find "
        "params. You may seed it by appending `[p1, p2]` per step (use `[]` "
        "for none); the search tries them first, then samples. " +
        _gs_refine_doc + "`plan` is one "
        "option call per line with typed object references (`obj:type`) and "
        "every argument supplied; add `-> {Atom(obj:type, ...)}` subgoal "
        "annotations (effectively required after open-ended skills like Place, "
        "and for Wait to say when it should end — prefix an atom with NOT to "
        "require it become false). When the task has an evaluator, success is "
        "also gated on its scoring: a parameterization that reaches the goal "
        "atoms but scores as a non-solve (no success credit in its reward) is "
        "discarded and the search resamples. On SUCCESS it reports the exact "
        "PARAMETERS it found per step — submit those via evaluate_option_plan, "
        "which is the delivery path; refine_plan_sketch itself does NOT "
        "submit. Also reports the verdict (SUCCESS / TIMEOUT / "
        "SAMPLE_EXHAUSTED with the stuck step / FORWARD_VALIDATION_FAILED / "
        "SCORED_NON_SOLVE) and time used. Requires a simulator (option "
        "model). Slower than evaluate_option_plan — use it to find params "
        "for hard steps, not to submit.",
        {
            "type": "object",
            "properties": {
                "plan": {
                    "type":
                    "string",
                    "description":
                    "Option-skeleton plan text, one option call per "
                    "line, typed `obj:type` references, every argument "
                    "supplied; optional `-> {Atom(...)}` subgoal per step, "
                    "and `[p1, p2]` proposed continuous params per step "
                    "(`[]` for none) when param-proposing is enabled" +
                    _gs_refine_plan_doc + ".",
                },
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index. Omit to use the current "
                    "solve-time task (if available).",
                },
                "timeout": {
                    "type":
                    "number",
                    "description":
                    "Refinement timeout in seconds. Omit for an auto "
                    "value that scales with sketch length; the value "
                    "used is reported back.",
                },
            },
            "required": ["plan"],
        },
    )
    async def refine_plan_sketch(args: Dict[str, Any]) -> Dict[str, Any]:
        refine_cfg = RefinementConfig.from_cfg()
        if ctx.option_model is None:
            return _error_result(
                "refine_plan_sketch requires a simulator (no option model "
                "in ToolContext).")

        # Resolve the task (mirrors evaluate_option_plan).
        resolved, task_err = _resolve_task(ctx, args.get("task_idx"))
        if task_err is not None:
            return task_err
        assert resolved is not None
        task = resolved.task
        task_idx = resolved.label

        all_options = ctx.options | ctx.iteration_proposals.proposed_options
        all_predicates = (ctx.predicates
                          | ctx.iteration_proposals.proposed_predicates)
        # Keep the option model's name map in sync with proposed options so
        # refinement can ground them (matches evaluate_option_plan).
        model = ctx.option_model
        model._name_to_parameterized_option = (  # type: ignore[attr-defined]  # pylint: disable=protected-access
            {o.name: o
             for o in all_options})
        # Union declared types with those reachable from options/predicates/
        # objects so typed `obj:type` references in the sketch resolve.
        types = set(ctx.types)
        for opt in all_options:
            types.update(opt.types)
        for pred in all_predicates:
            types.update(pred.types)
        types.update(o.type for o in task.init)

        plan_text = (args.get("plan") or "").strip()
        if not plan_text:
            return _error_result("`plan` is required (option-skeleton text).")
        try:
            # strict: the `plan` argument is pure sketch text (see
            # evaluate_option_plan) - unparseable lines must error, not be
            # silently dropped.
            # Named `~ my_sampler` references resolve against the agent's
            # ground_samplers.py, reloaded fresh so edits between calls
            # take effect; a broken file is surfaced instead of silently
            # falling back to uniform draws.
            gs_fns, gs_err = load_ground_sampler_fns(ctx)
            if gs_err is not None:
                return _error_result(gs_err)
            parse_notices: List[str] = []
            sketch = bilevel_sketch.parse_sketch_from_text(
                plan_text,
                task,
                predicates=all_predicates,
                options=all_options,
                types=types,
                parse_continuous_params=refine_cfg.use_llm_initial_params,
                strict=True,
                parse_ground_samplers=refine_cfg.ground_samplers,
                ground_sampler_fns=gs_fns or None,
                notices=parse_notices,
            )
        except Exception as e:  # pylint: disable=broad-except
            return _error_result(f"Could not parse plan sketch: {e}")
        if not sketch:
            return _error_result(
                "Parsed empty plan sketch. Check that every line names a "
                "known option with typed `obj:type` arguments matching the "
                "Options digest in your prompt.")

        timeout, timeout_source = bilevel_sketch.resolve_refine_timeout(
            args.get("timeout"),
            len(sketch),
            per_step=refine_cfg.refinement_timeout_per_step,
            minimum=refine_cfg.refinement_timeout_min)

        # Refinement accepts a parameterization only if the task evaluator
        # also scores its rollout as a solve: a candidate that reaches the
        # goal atoms yet earns no success credit (e.g. the poker, not the
        # cascade, toppled the target) is discarded and refinement is
        # resampled with a fresh rng, all attempts sharing the one timeout
        # budget. Without this gate the search happily converges onto
        # parameterizations the env would score as non-solves and reports
        # SUCCESS on them (run_20260713_172854 seed0 task1 test034). The
        # gate reads ONLY the public (terminated, reward, solved) triple -
        # the standard RL end-of-episode observables - so it grants the
        # search nothing the agent could not compute itself, and it never
        # depends on a reward sign convention.
        attempts = max(1, refine_cfg.refine_evaluator_attempts)
        discarded_rewards: List[float] = []
        verdict_line: Optional[str] = None
        non_solve = False
        start = time.perf_counter()
        success, report = False, ""
        plan: List[Any] = []

        # In-search version of the same gate: reject a goal-atom-reaching
        # candidate DURING backtracking when the evaluator scores it as a
        # non-solve, so the search keeps moving from the same node (its
        # upstream samples intact) instead of converging onto uncertifiable
        # parameters and needing a cold restart below. The restart loop
        # stays as the safety net for verdict flakiness: the post-hoc
        # check re-rolls the accepted plan, and a re-roll that scores
        # differently (the sim is nondeterministic across runs) still
        # triggers a resample.
        _gate_evaluator = _resolve_task_evaluator(ctx, task_idx)
        solved_check = None
        if _gate_evaluator is not None:
            solved_check = make_solved_check(
                _gate_evaluator,
                getattr(ctx.option_model, "sim_env", None),
                on_reject=discarded_rewards.append)

        for attempt in range(attempts):
            remaining = timeout - (time.perf_counter() - start)
            if attempt and remaining < 5.0:
                break
            try:
                success, report, plan = \
                    bilevel_sketch.refine_and_validate_report(
                        task,
                        sketch,
                        ctx.option_model,
                        predicates=all_predicates,
                        timeout=remaining if attempt else timeout,
                        rng=np.random.default_rng(CFG.seed + attempt),
                        max_samples_per_step=refine_cfg.max_samples_per_step,
                        check_subgoals=refine_cfg.check_subgoals,
                        log_state=refine_cfg.log_state,
                        parameterized_samplers=ctx.parameterized_samplers
                        or None,
                        run_id="planner_refine",
                        timeout_source=timeout_source,
                        solved_check=solved_check,
                    )
            except Exception:  # pylint: disable=broad-except
                tb = _scrub_host_paths(traceback.format_exc())
                return _error_result(f"Refinement raised:\n{tb}")
            non_solve = False
            if not (success and plan):
                break
            scored = _belief_rollout_verdict(ctx, task, task_idx, plan,
                                             all_predicates)
            if scored is None:
                break
            verdict, coarse = scored
            if coarse or not verdict["terminated"] or verdict["solved"]:
                verdict_line = _format_evaluator_verdict(verdict,
                                                         coarse=coarse)
                break
            discarded_rewards.append(verdict["reward"])
            success = False
            non_solve = True

        # A failed search whose DEEPEST blocker was the in-search gate is
        # the verdict the restart loop expresses as SCORED_NON_SOLVE: the
        # sketch reaches the goal atoms but never certifiably, so say that
        # (with the change-the-sketch advice). A search that rejected a
        # candidate along the way but then failed on something else (an
        # upstream IK wall, a timeout mid-descent) keeps its own headline
        # - the near-miss line and the discard NOTE still surface the
        # rejections.
        if (not success and discarded_rewards
                and "scored non-solve" in report):
            non_solve = True
        rewards_str = ", ".join(f"{r:.2f}" for r in discarded_rewards)
        if non_solve:
            report = (
                "FAILURE: SCORED_NON_SOLVE\n"
                f"  Refinement found goal-atom-reaching parameters "
                f"{len(discarded_rewards)} time(s), but the task evaluator "
                f"scored every such rollout as a non-solve (rewards: "
                f"{rewards_str}; no success credit). The real env applies "
                "the same scoring, so these parameters can never count as "
                "a solve - change the sketch (e.g. different placements or "
                "orientations), not just the parameters.\n"
                "Last attempt detail:\n" + report)
        elif discarded_rewards:
            passed_tail = (
                "; the result above is from parameters that passed the "
                "evaluator's scoring." if success else ".")
            report += (
                f"\n  NOTE: {len(discarded_rewards)} earlier "
                f"parameterization(s) reached the goal atoms but scored as "
                f"non-solves (rewards: {rewards_str}) and were discarded "
                f"during the search{passed_tail}")

        # refine_plan_sketch is a parameter FINDER, not a submission path: on
        # success, append the parameters the search found per step so the
        # agent can submit these exact values via evaluate_option_plan (the
        # only delivery path). It deliberately does NOT capture a solved plan.
        if success and plan:
            param_lines = []
            for i, gopt in enumerate(plan):
                objs = ", ".join(o.name for o in gopt.objects)
                par = ", ".join(f"{p:.4f}" for p in gopt.params)
                param_lines.append(f"  {i}: {gopt.name}({objs})[{par}]")
            report += ("\n\nParameters found (submit these exact values via "
                       "evaluate_option_plan):\n" + "\n".join(param_lines))
            if verdict_line is not None:
                report += "\n" + verdict_line

        if parse_notices:
            report = "\n".join(f"NOTE: {n}" for n in parse_notices) + \
                "\n" + report

        return _text_result(f"Task {task_idx}:\n{report}")

    # ------------------------------------------------------------------ #
    # Scene annotation
    # ------------------------------------------------------------------ #

    return {
        "generate_bilevel_plan": generate_bilevel_plan,
        "generate_abstract_plan": generate_abstract_plan,
        "refine_plan_sketch": refine_plan_sketch,
    }
