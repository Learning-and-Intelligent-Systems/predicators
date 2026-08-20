"""Testing tools, including the evaluate_option_plan capture surface."""
import contextlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.config import RefinementConfig, ToolSurfaceConfig, \
    ValidationConfig
from predicators.agent_sdk.tools.budget import _budget_footer
from predicators.agent_sdk.tools.capture import BestEffortReason, \
    CaptureDecision, _decide_capture
from predicators.agent_sdk.tools.context import ToolContext, \
    _capture_task_key, decorrelated_rollout_seed
from predicators.agent_sdk.tools.results import _error_result
from predicators.agent_sdk.tools.scene import format_object_poses, \
    render_scene_image
from predicators.agent_sdk.tools.tasks import _resolve_task
from predicators.agent_sdk.tools.verdicts import _EvalStateCollector, \
    _format_evaluator_verdict, _resolve_task_evaluator, evaluate_states_with, \
    load_ground_sampler_fns
from predicators.settings import CFG


def _build_testing_tools(ctx: ToolContext, _text_result: Callable,
                         tool: Callable) -> Dict[str, Any]:
    """Evaluation tools (run predicates / option plans against tasks)."""

    @tool(
        "evaluate_predicate_on_trajectory",
        "Evaluate a predicate's truth value across timesteps in a trajectory",
        {
            "type": "object",
            "properties": {
                "predicate_name": {
                    "type": "string",
                    "description": "Name of the predicate to test"
                },
                "traj_idx": {
                    "type": "integer",
                    "description": "Trajectory index"
                },
                "object_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Object names to ground the predicate on"
                },
            },
            "required": ["predicate_name", "traj_idx", "object_names"],
        },
    )
    async def evaluate_predicate_on_trajectory(
            args: Dict[str, Any]) -> Dict[str, Any]:
        pred_name = args["predicate_name"]
        traj_idx = args["traj_idx"]
        object_names = args["object_names"]

        # Find the predicate
        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates
        pred = None
        for p in all_preds:
            if p.name == pred_name:
                pred = p
                break
        if pred is None:
            return _error_result(f"Predicate '{pred_name}' not found.")

        all_trajs = ctx.offline_trajectories + ctx.online_trajectories
        if not all_trajs:
            return _error_result("No trajectories available yet.")
        if traj_idx < 0 or traj_idx >= len(all_trajs):
            return _error_result(f"Invalid traj_idx {traj_idx}. "
                                 f"Available: 0-{len(all_trajs)-1}")

        traj = all_trajs[traj_idx]

        # Find objects by name
        objects = []
        for name in object_names:
            found = None
            for obj in traj.states[0]:
                if obj.name == name:
                    found = obj
                    break
            if found is None:
                avail = [o.name for o in sorted(traj.states[0], key=str)]
                return _error_result(f"Object '{name}' not found in "
                                     f"trajectory {traj_idx}. "
                                     f"Available: {avail}")
            objects.append(found)

        results = []
        for t_step, state in enumerate(traj.states):
            try:
                val = pred.holds(state, objects)
                results.append(f"t={t_step}: {val}")
            except Exception as e:  # pylint: disable=broad-except
                results.append(f"t={t_step}: ERROR ({e})")

        return _text_result(
            f"Predicate {pred_name}({', '.join(object_names)}) "
            f"over trajectory {traj_idx}:\n" + "\n".join(results))

    # Tool descriptions bake config values at BUILD time (session open);
    # the handlers below re-read config at CALL time.
    _gs_eval_doc = (
        "Runs your exact params with NO sampling (a `~` ground-sampler "
        "annotation - `~ [w1, w2]` region or `~ my_sampler` - is accepted "
        "but IGNORED here; only refine_plan_sketch uses it). "
        if RefinementConfig.from_cfg().ground_samplers else
        "Runs your exact params with NO sampling. ")

    # When the session carries explore_python, the two surfaces divide
    # cleanly: exploration (modified states, partial plans, sweeps)
    # belongs in the probe, and this tool is the SUBMISSION path.
    _probe_split_doc = (
        " This tool always runs from the task's TRUE initial state and is "
        "the ONLY path that captures an answer: do your exploration "
        "(modified states, partial plans, parameter sweeps) in "
        "explore_python, then validate and SUBMIT the final plan here."
        if ToolSurfaceConfig.from_cfg().use_explore_python else "")

    @tool(
        "evaluate_option_plan",
        "Execute a fully-specified plan on a task via the option model and "
        "report the result at each step. `plan` is text — one option per "
        "line, same grammar as refine_plan_sketch: "
        "`Option(obj1:type1, obj2:type2)[param1, param2] -> {Atom(obj:type), "
        "...}` (typed object refs; EXACT continuous params in `[]`, `[]` for "
        "none; optional `-> {atoms}` subgoals, prefix NOT to require false). "
        + _gs_eval_doc + "Use include_states/"
        "include_atoms to control output. If the plan reaches the goal on the "
        "CURRENT task (omit task_idx), it is captured as your answer, and the "
        "per-step subgoals make it execute closed-loop (monitored, with "
        "replan-on-divergence). Capture is gated: a goal-reaching plan is "
        "re-run several times (simulation varies across runs) and a FLAKY "
        "plan is reported instead of captured - add margin and resubmit. "
        "When identified physical parameters are active, it is also re-run "
        "at a grid of perturbations spanning +-1 sigma of those parameters "
        "(the physics fit's own uncertainty); a PARAM-SENSITIVE plan is "
        "reported instead of captured - add design margin so it succeeds "
        "across the whole range. "
        "When the task has an evaluator, a goal-reaching plan the evaluator "
        "still scores as a non-solve (no success credit in its reward) is "
        "NOT captured (the real env applies the same scoring, so it could "
        "never count as a solve)." + _probe_split_doc,
        {
            "type": "object",
            "properties": {
                "plan": {
                    "type":
                    "string",
                    "description":
                    "Plan text, one option per line: "
                    "`Option(obj1:type1, obj2:type2)[p1, p2] -> "
                    "{Atom(obj:type), ...}` (exact params in `[]`; `[]` for "
                    "none; optional `-> {atoms}` subgoals, NOT-prefix to "
                    "require false).",
                },
                "include_states": {
                    "type":
                    "boolean",
                    "description":
                    "Include the full low-level state feature dict after each "
                    "step",
                    "default":
                    True
                },
                "include_atoms": {
                    "type": "boolean",
                    "description":
                    "Include atoms added/deleted after each step",
                    "default": True
                },
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index to test on. Omit to use "
                    "the current solve-time task."
                },
            },
            "required": ["plan"],
        },
    )
    async def evaluate_option_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        refine_cfg = RefinementConfig.from_cfg()
        validation_cfg = ValidationConfig.from_cfg()
        ctx.test_call_id += 1
        # Snapshot for the [budget] footer's per-call delta; this handler
        # increments the counter itself (initial rollout + validation
        # repeats), and without the snapshot the footer reports the
        # attempt's cumulative total as "+N this call".
        rollouts_before = ctx.attempt_rollout_count

        if ctx.option_model is None:
            return _error_result("No option model available in ToolContext.")

        # Sync the option model's option map with all current options
        # (GT + proposed) so it stays in sync after propose/retract.
        all_options = ctx.options | ctx.iteration_proposals.proposed_options
        opt_map = {o.name: o for o in all_options}
        model = ctx.option_model
        model._name_to_parameterized_option = (  # type: ignore[attr-defined]  # pylint: disable=protected-access
            opt_map)

        task_idx = args.get("task_idx")
        plan_text = (args.get("plan") or "").strip()
        include_states = args.get("include_states", False)
        include_atoms = args.get("include_atoms", True)

        resolved, task_err = _resolve_task(ctx, task_idx)
        if task_err is not None:
            return task_err
        assert resolved is not None
        task = resolved.task
        task_label = resolved.label
        is_current = resolved.is_current

        lines = [f"Testing option plan on task {task_label}:"]
        saved_image_paths: List[str] = []

        all_predicates = (ctx.predicates
                          | ctx.iteration_proposals.proposed_predicates)

        if not plan_text:
            return _error_result("`plan` is required (option plan text).")
        # Parse the text plan into a sketch (options + objects + exact params +
        # subgoals) using the SAME grammar/parser as refine_plan_sketch.
        types = set(ctx.types)
        for opt in all_options:
            types.update(opt.types)
        for pred in all_predicates:
            types.update(pred.types)
        types.update(o.type for o in task.init)
        try:
            # strict: the `plan` argument is pure plan text, so a line that
            # fails to parse is an error the agent must see - silently
            # dropping it (the freeform default) executes a different plan
            # than the agent asked for.
            gs_fns, gs_err = load_ground_sampler_fns(ctx)
            if gs_err is not None:
                return _error_result(gs_err)
            parse_notices: List[str] = []
            sketch_steps = bilevel_sketch.parse_sketch_from_text(
                plan_text,
                task,
                predicates=all_predicates,
                options=all_options,
                types=types,
                parse_continuous_params=True,
                strict=True,
                parse_ground_samplers=refine_cfg.ground_samplers,
                ground_sampler_fns=gs_fns or None,
                notices=parse_notices)
        except Exception as e:  # pylint: disable=broad-except
            return _error_result(f"Could not parse plan: {e}")
        lines.extend(f"NOTE: {n}" for n in parse_notices)
        if not sketch_steps:
            return _error_result(
                "Parsed empty plan. Each line must be "
                "`Option(obj:type, ...)[params] -> {subgoals}` with a known "
                "option, typed object refs, and exact params in `[]`.")
        # Ground each step with its parsed exact params.
        grounded_plan: List[Any] = []
        for step_idx, st in enumerate(sketch_steps):
            params = (st.initial_params if st.initial_params is not None else
                      np.array([], dtype=np.float32))
            try:
                grounded_plan.append(
                    st.option.ground(list(st.objects),
                                     np.asarray(params, dtype=np.float32)))
            except Exception as e:  # pylint: disable=broad-except
                return _error_result(f"Failed to ground step {step_idx} "
                                     f"({st.option.name}): {e}")

        # Per-low-level-step states + option labels for the task-evaluator
        # verdict below (see _EvalStateCollector for why per-step states).
        eval_collector = _EvalStateCollector(model, task.init)

        # Per-step report callback, driven by the shared forward executor.
        def _report_step(i: int, outcome: Any) -> None:
            eval_collector.collect(outcome)
            opt = outcome.option
            sig = f"{opt.name}({[o.name for o in opt.objects]})"
            if not outcome.initiable:
                atoms = utils.abstract(outcome.pre_state, ctx.predicates)
                atoms_str = ", ".join(str(a) for a in sorted(atoms))
                lines.append(f"Step {i}: {sig} - NOT INITIABLE\n"
                             f"  Current atoms: {{{atoms_str}}}\n"
                             f"  Object poses at failure:\n"
                             f"{format_object_poses(outcome.pre_state)}")
                return
            step_line = f"Step {i}: {sig} ({outcome.num_actions} actions)"
            if outcome.failure_reason is not None:
                step_line += (f"\n  FAILURE REASON: {outcome.failure_reason}"
                              "\n  Object poses at failure:\n"
                              f"{format_object_poses(outcome.pre_state)}")
            post = outcome.post_state
            if post is not None and include_atoms:
                before = utils.abstract(outcome.pre_state, ctx.predicates)
                after = utils.abstract(post, ctx.predicates)
                added_s = ", ".join(str(a) for a in sorted(after - before))
                del_s = ", ".join(str(a) for a in sorted(before - after))
                step_line += (f"\n  Added:   {{{added_s}}}"
                              f"\n  Deleted: {{{del_s}}}")
            if post is not None and include_states:
                step_line += ("\n  State:\n" +
                              post.dict_str(indent=4, num_decimal_points=4))
            lines.append(step_line)
            img_block = render_scene_image(ctx, f"step_{i}_{opt.name}")
            if img_block and img_block.get("saved_path"):
                saved_image_paths.append(img_block["saved_path"])

        # Execute exactly like the real closed-loop executor: abort at the
        # first failing option (0-action collision / not-initiable / env
        # failure) instead of pressing on. Otherwise forward simulation can
        # continue past a collision and report a goal that the real rollout —
        # which ends the episode at that failed option — never reaches.
        ctx.attempt_rollout_count += 1
        result = bilevel_sketch.execute_plan_forward(task,
                                                     grounded_plan,
                                                     ctx.option_model,
                                                     predicates=all_predicates,
                                                     sketch=sketch_steps,
                                                     on_step=_report_step,
                                                     stop_on_failure=True)

        final_atoms = utils.abstract(result.final_state, ctx.predicates)
        # Use the env's goal-check (its own classifiers); robust to invented
        # predicates that don't reuse env names.
        goal_reached = result.goal_reached
        # One more real-executor constraint the option model doesn't enforce:
        # the episode is capped at `CFG.horizon` low-level steps. A plan whose
        # goal is reached only after more steps than the horizon allows will
        # time out in real rollout, so don't count it as achieved/captured.
        horizon = CFG.horizon
        within_horizon = (result.actions_to_goal is not None
                          and result.actions_to_goal <= horizon)
        goal_achieved = (goal_reached and result.clean_to_goal
                         and within_horizon)
        # Task-evaluator verdict on this belief-sim rollout, computed BEFORE
        # capture: the real evaluator applies the same certificate, so a
        # goal-reaching but illegitimate plan can never count as a solve and
        # must not be captured as the answer (run_20260712_173955 tasks 1-2:
        # flagged-illegitimate captures stood all session and were executed
        # only to be rejected). Failure-tolerant: verdict stays None when the
        # task has no evaluator or nothing executed. A coarse verdict
        # (option-boundary states only) can falsely reject a legitimate
        # cascade, so it never blocks capture.
        evaluator = _resolve_task_evaluator(ctx, task_label)
        verdict: Optional[Dict[str, Any]] = None
        if evaluator is not None and len(eval_collector.states) > 1:
            try:
                verdict = evaluate_states_with(evaluator,
                                               eval_collector.states,
                                               eval_collector.labels,
                                               sim_env=getattr(
                                                   ctx.option_model, "sim_env",
                                                   None))
            except Exception as e:  # pylint: disable=broad-except
                logging.debug("Task-evaluator verdict failed: %s", e)
        evaluator_rejected = (verdict is not None and not verdict["legitimate"]
                              and not eval_collector.coarse)
        # An evaluator rejection only disqualifies a capture when the goal
        # atoms actually hold via an illegitimate route - a reward hack (e.g.
        # the agent knocked the target over directly). An honest shortfall,
        # where the rollout simply fails to reach the goal, is ALSO
        # legitimate=False (there is no genuine cascade to certify), but that
        # is exactly what a best-effort submission is meant to capture, so it
        # must not be conflated with a reward hack.
        reward_hack = (evaluator_rejected and verdict is not None
                       and verdict["terminated"])

        # Multi-rollout validation of a capture candidate. The shared sim
        # env is nondeterministic across repeats (motion-planner sampling,
        # physics-solver state), which is the same variability the real
        # rollout will sample - a plan that only sometimes succeeds here is
        # a margin-free plan that will likely fail on the real env
        # (run_20260712_192457 task 1: a sim-validated 2-hop relay died on a
        # ~9mm placement drift). So a goal-reaching plan is captured only
        # after every one of validation_cfg.rollouts total
        # rollouts succeeds; a flaky repeat is reported to the agent, who
        # still has the session to add margin and resubmit.
        def _validation_rollout() -> Tuple[bool, str]:
            """One extra rollout of the exact plan; (ok, failure detail)."""
            v_collector = _EvalStateCollector(model, task.init)
            r = bilevel_sketch.execute_plan_forward(
                task,
                grounded_plan,
                model,
                predicates=all_predicates,
                sketch=sketch_steps,
                on_step=v_collector.on_step,
                stop_on_failure=True)
            if r.first_failure_idx is not None:
                fr = r.steps[r.first_failure_idx].failure_reason
                opt = r.steps[r.first_failure_idx].option
                return False, (f"step {r.first_failure_idx} "
                               f"({opt.name}) failed: {fr}")
            if not r.goal_reached:
                missing = task.goal - utils.abstract(r.final_state,
                                                     ctx.predicates)
                missing_str = ", ".join(str(a) for a in sorted(missing))
                detail = f" (missing: {{{missing_str}}})" if missing else ""
                return False, f"goal not reached{detail}"
            if not (r.actions_to_goal is not None
                    and r.actions_to_goal <= horizon):
                return False, (f"goal reached only after "
                               f"{r.actions_to_goal} low-level steps, past "
                               f"the episode horizon ({horizon})")
            # Same legitimacy rule as the first rollout: a non-coarse
            # illegitimate verdict fails the validation.
            if (evaluator is not None and len(v_collector.states) > 1
                    and not v_collector.coarse):
                try:
                    v = evaluate_states_with(evaluator,
                                             v_collector.states,
                                             v_collector.labels,
                                             sim_env=getattr(
                                                 ctx.option_model, "sim_env",
                                                 None))
                    if not v["legitimate"]:
                        return False, (
                            "this rollout reached the goal atoms but the "
                            "task evaluator scored it as a non-solve "
                            f"(solved=False, reward={v['reward']:.2f})")
                except Exception as e:  # pylint: disable=broad-except
                    logging.debug("Validation-rollout verdict failed: %s", e)
            return True, ""

        flaky_detail: Optional[str] = None
        validation_note = ""
        n_rollouts = max(1, validation_cfg.rollouts)
        # Escalated gate once this task has produced a FLAKY rejection: the
        # agent is provably tuning in a marginal region, where a lucky
        # streak passes the base gate and dies on the single real episode
        # (run_20260717_182321: a 20/20-swept relay placement validated 3/3,
        # then missed the target for real).
        capture_task_key = _capture_task_key(ctx)
        if capture_task_key in ctx.flaky_capture_task_keys:
            n_rollouts = max(n_rollouts, validation_cfg.rollouts_after_flaky)
        # Fresh env per validation rollout when the approach provides one:
        # repeats on the shared env are correlated (its reset cannot
        # reconstruct state exactly), so only fresh envs sample the same
        # distribution the real episode will.
        fresh_scope = (ctx.validation_env_scope
                       if validation_cfg.fresh_env else None)
        rollout_outcomes: List[str] = []
        if (ctx.capture_goal_reaching_plans and is_current and goal_achieved
                and not evaluator_rejected and grounded_plan
                and n_rollouts > 1):
            # Run ALL validation rollouts even after a failure: the
            # per-rollout outcome list distinguishes failure modes (a
            # physics-tail fizzle vs. an IK stall vs. a certificate
            # rejection) and yields a reliability estimate - a bare
            # "rollout k FAILED" left agents guessing which
            # (run_20260717_182040 seed0 turn 214).
            for repeat_idx in range(2, n_rollouts + 1):
                ctx.attempt_rollout_count += 1
                # decorrelated_rollout_seed: a fresh env alone gives
                # bit-identical repeats (motion planning reads the
                # constant CFG.seed at call time), so without it the
                # validation repeats re-run the capture rollout verbatim
                # and detect nothing. The capture rollout itself keeps
                # the base seed; repeats sample execution variability.
                with (fresh_scope() if fresh_scope is not None else
                      contextlib.nullcontext()), \
                        decorrelated_rollout_seed(repeat_idx - 1):
                    ok, why = _validation_rollout()
                if ok:
                    rollout_outcomes.append(
                        f"rollout {repeat_idx}: goal reached")
                else:
                    rollout_outcomes.append(
                        f"rollout {repeat_idx}: FAILED - {why}")
                    if flaky_detail is None:
                        flaky_detail = (f"rollout {repeat_idx}/{n_rollouts} "
                                        f"FAILED: {why}")
            if flaky_detail is None:
                fresh_note = (", each on a freshly constructed simulator "
                              "instance" if fresh_scope is not None else "")
                validation_note = (
                    f" Validated {n_rollouts}/{n_rollouts} rollouts (the "
                    "simulator's motion planning and physics stepping vary "
                    "across runs; repeats sample that execution "
                    f"variability{fresh_note}).")

        # Physics-margin gate: the execution repeats above all run AT the
        # fitted physical params, so they cannot see a plan whose success
        # band excludes the fit's parameter error (run_20260723_091108: a
        # capture validated 8/8 at fitted lateral_friction 0.5319 failed
        # deterministically at true 0.5, just outside the design's band).
        # Re-run the plan at each +-1-posterior-sigma perturbation of the
        # identified params, on a fresh env (perturbing the shared env
        # would leak into later tool calls) at the BASE planner seed, so a
        # failure is attributable to the physics perturbation alone.
        param_sensitive_detail: Optional[str] = None
        margin_outcomes: List[str] = []
        if (validation_cfg.physics_margin and fresh_scope is not None
                and ctx.physics_margin_provider is not None
                and ctx.capture_goal_reaching_plans and is_current
                and goal_achieved and not evaluator_rejected and grounded_plan
                and flaky_detail is None):
            for point in ctx.physics_margin_provider() or []:
                ctx.attempt_rollout_count += 1
                with fresh_scope(physical_overrides=point):
                    ok, why = _validation_rollout()
                desc = ", ".join(f"{k}={v:.4g}"
                                 for k, v in sorted(point.items()))
                if ok:
                    margin_outcomes.append(
                        f"physics point ({desc}): goal reached")
                else:
                    margin_outcomes.append(
                        f"physics point ({desc}): FAILED - {why}")
                    if param_sensitive_detail is None:
                        param_sensitive_detail = f"at {desc}: {why}"
            if margin_outcomes and param_sensitive_detail is None:
                validation_note += (
                    " Physics-margin check passed: the plan also reached "
                    f"the goal at all {len(margin_outcomes)} grid points "
                    "spanning +-1 sigma of the identified physical "
                    "parameters.")

        def _stash_uncaptured_submission() -> None:
            """Remember the best refused submission of this attempt.

            The journal auto-entry records it at attempt end, so the
            plan (and its honest evaluator reward) survives the fresh-
            context restart and the final best-effort nudge can resubmit
            it instead of the attempt's work vanishing with its context.
            """
            reward = float(verdict["reward"]) if verdict is not None else None
            prev = ctx.best_uncaptured_reward
            if ctx.best_uncaptured_plan_lines is not None and (
                    reward is None or (prev is not None and reward <= prev)):
                return
            ctx.best_uncaptured_reward = reward
            ctx.best_uncaptured_plan_lines = list(
                bilevel_sketch.format_plan_lines(grounded_plan))

        # The capture decision itself is pure (see _decide_capture, which
        # also documents the best-effort-mode semantics); the branches
        # below apply its ctx mutations and format its messages.
        capture_outcome = _decide_capture(
            capture_enabled=ctx.capture_goal_reaching_plans,
            is_current_task=is_current,
            have_plan=bool(grounded_plan),
            goal_achieved=goal_achieved,
            evaluator_rejected=evaluator_rejected,
            reward_hack=reward_hack,
            flaky=flaky_detail is not None,
            best_effort_mode=ctx.capture_best_effort_plan,
            have_validated_capture=bool(ctx.solved_plan_reached_goal),
            param_sensitive=param_sensitive_detail is not None)
        decision = capture_outcome.decision
        captured = capture_outcome.captured
        if captured:
            # Capture the plan with a sketch that keeps only the subgoals
            # that actually held (so the closed-loop monitor won't flag a
            # spurious divergence on a wrong annotation).
            validated_solve = decision is CaptureDecision.VALIDATED_CAPTURE
            captured_sketch = []
            for i, st in enumerate(sketch_steps):
                post = (result.steps[i].post_state
                        if i < len(result.steps) else None)
                if post is not None:
                    after = utils.abstract(post, all_predicates)
                    pos_held = {
                        a
                        for a in (st.subgoal_atoms or set()) if a in after
                    }
                    neg_held = {
                        a
                        for a in (st.subgoal_neg_atoms or set())
                        if a not in after
                    }
                else:
                    pos_held, neg_held = set(), set()
                captured_sketch.append(
                    bilevel_sketch.SketchStep(option=st.option,
                                              objects=st.objects,
                                              subgoal_atoms=pos_held or None,
                                              subgoal_neg_atoms=neg_held
                                              or None))
            ctx.solved_plan = grounded_plan
            ctx.solved_sketch = captured_sketch
            ctx.solved_plan_reached_goal = validated_solve
            ctx.solved_plan_eval_reward = (float(verdict["reward"])
                                           if verdict is not None else None)
            n_annot = sum(1 for s in captured_sketch
                          if s.subgoal_atoms or s.subgoal_neg_atoms)
            reason = capture_outcome.best_effort_reason
            if reason is None:
                best_effort_note = ""
            elif reason is BestEffortReason.GOAL_NOT_REACHED:
                best_effort_note = (" (best-effort: goal NOT reached, "
                                    "accepted because the attempt budget is "
                                    "exhausted; it executes for its honest "
                                    "reward but will not count as a solve)")
            elif reason is BestEffortReason.REWARD_HACK:
                best_effort_note = (" (best-effort: the rollout reaches the "
                                    "goal atoms but the task evaluator "
                                    "scores it as a non-solve, and the real "
                                    "env applies the same scoring; accepted "
                                    "because the attempt budget is exhausted "
                                    "- it executes for its honest reward but "
                                    "will not count as a solve)")
            elif reason is BestEffortReason.FLAKY:
                best_effort_note = (f" (best-effort: {flaky_detail}; "
                                    "accepted because the attempt budget is "
                                    "exhausted - it executes for its honest "
                                    "reward but may not reproduce its "
                                    "solve)")
            else:
                assert reason is BestEffortReason.PARAM_SENSITIVE
                best_effort_note = (" (best-effort: failed "
                                    f"{param_sensitive_detail}; accepted "
                                    "because the attempt budget is exhausted "
                                    "- it executes for its honest reward but "
                                    "may fail under the true physics)")
            lines.append(f"Captured as the current answer{best_effort_note}: "
                         f"{len(grounded_plan)} steps, "
                         f"{n_annot} with subgoal annotations for closed-loop "
                         f"monitoring.{validation_note}")
        elif decision is CaptureDecision.FLAKY_NO_CAPTURE:
            # Record the task so later submissions face the escalated
            # gate - flakiness here is evidence the whole parameter
            # region is marginal, not just this point.
            ctx.flaky_capture_task_keys.add(capture_task_key)
            _stash_uncaptured_submission()
            escalated_n = max(max(1, validation_cfg.rollouts),
                              validation_cfg.rollouts_after_flaky)
            n_ok = 1 + sum(1 for o in rollout_outcomes if "FAILED" not in o)
            per_rollout = "\n".join(f"  {o}" for o in rollout_outcomes)
            lines.append(
                f"FLAKY (plan NOT captured): the plan reached the goal on "
                f"rollout 1 but {flaky_detail}. Per-rollout outcomes "
                f"(estimated reliability {n_ok}/{n_rollouts}):\n"
                f"  rollout 1: goal reached\n{per_rollout}\n"
                "The simulator's motion "
                "planning and physics stepping vary across runs, and the "
                "real environment samples the same variability - a plan "
                "that only sometimes succeeds in simulation will likely "
                "fail for real. Add margin (e.g. tighter spacing, aim "
                "impacts closer to the middle of the fall path) and "
                "resubmit. Because this task has now produced a flaky "
                f"submission, captures require {escalated_n}/{escalated_n} "
                "successful rollouts: fix the margin rather than "
                "resubmitting near-identical parameters.")
        elif decision is CaptureDecision.PARAM_SENSITIVE_NO_CAPTURE:
            _stash_uncaptured_submission()
            per_point = "\n".join(f"  {o}" for o in margin_outcomes)
            lines.append(
                "PARAM-SENSITIVE (plan NOT captured): the plan passed "
                "execution validation at the fitted physical parameters "
                f"but FAILED {param_sensitive_detail}.\n"
                "Physics-margin rollouts (a grid spanning +-1 sigma of "
                "the identified physical parameters, the sysID fit's "
                f"own uncertainty):\n{per_point}\n"
                "The fitted values are uncertain at this scale and the "
                "real environment may sit anywhere in that range - "
                "including BETWEEN passing points: success can be "
                "non-monotonic in a physical parameter, so a design must "
                "hold across the whole range, not just at the values you "
                "tuned at. Add margin to the DESIGN (not the execution) - "
                "e.g. tighter spacing or impacts nearer the middle of the "
                "fall path - then resubmit.")
        elif decision is CaptureDecision.REWARD_HACK_NO_CAPTURE:
            assert verdict is not None
            _stash_uncaptured_submission()
            lines.append(
                "NOT CAPTURED: the rollout reaches the goal atoms but the "
                "task evaluator scores it as a non-solve (solved=False, "
                f"reward={verdict['reward']:.2f}). The real env applies the "
                "same scoring, so executing this plan cannot count as a "
                "solve. Find a plan whose rollout the evaluator scores "
                "solved=True.")
        elif decision is CaptureDecision.WRONG_TASK_NOTE:
            lines.append(
                f"NOTE: this ran on train task {task_label}, NOT the current "
                "task, so it is NOT captured as your answer. To submit, "
                "re-run the plan on the current task (omit task_idx).")
        if result.first_failure_idx is not None:
            fr = result.steps[result.first_failure_idx].failure_reason
            lines.append(
                f"\nPlan FAILED at step {result.first_failure_idx}: {fr}")
        final_atoms_str = ", ".join(str(a) for a in sorted(final_atoms))
        lines.append(f"\nFinal atoms: {{{final_atoms_str}}}")
        if task.goal_nl:
            lines.append(f"Goal (natural language): {task.goal_nl}")
        else:
            goal_str = ", ".join(str(g) for g in sorted(task.goal))
            lines.append(f"Goal: {{{goal_str}}}")
        lines.append(f"Goal achieved: {goal_achieved}")
        # Task-evaluator verdict line (verdict computed above, before the
        # capture decision it gates). On a FLAKY rejection this verdict is
        # rollout 1's only - printing it unlabeled next to a failing
        # rollout's non-solve read as two contradictory verdicts in one
        # message (run_20260717_182040 seed1 turn 96).
        if verdict is not None:
            vline = _format_evaluator_verdict(verdict,
                                              coarse=eval_collector.coarse)
            if flaky_detail is not None and not captured:
                vline += (" [rollout 1 only - NOT the operative outcome; "
                          "this submission was rejected as FLAKY above]")
            lines.append(vline)
        # Goal atoms hold but the plan needs more low-level steps than the
        # episode horizon allows: say so and that it was NOT captured, so the
        # agent shortens the plan instead of stopping on a false positive.
        # (A best-effort capture still happens above; then only warn.)
        if goal_reached and not within_horizon and not captured:
            lines.append(
                f"NOT EXECUTABLE (plan was NOT captured): reaching the goal "
                f"takes {result.actions_to_goal} low-level steps but the "
                f"episode horizon is {horizon}. The real executor will run "
                f"out of steps — shorten the plan (fewer or quicker steps) "
                f"before resubmitting.")
        elif goal_reached and not within_horizon:
            lines.append(
                f"WARNING: reaching the goal takes {result.actions_to_goal} "
                f"low-level steps but the episode horizon is {horizon}, so "
                f"the real executor will run out of steps before the goal.")
        # Print the missing goal atoms even when the goal is stated in
        # natural language: "Goal achieved: False" with no per-atom
        # diagnosis left agents unable to tell a near-miss from a
        # non-starter, and validation-rollout failures already name the
        # missing atoms - this just makes rollout 1 report the same way.
        if not goal_reached:
            missing = task.goal - final_atoms
            missing_str = ", ".join(str(a) for a in sorted(missing))
            lines.append(f"Missing goal atoms: {{{missing_str}}}")

        # Append image save paths to text output
        if saved_image_paths:
            lines.append("\nSaved images:")
            for p in saved_image_paths:
                lines.append(f"  {p}")

        # Build result with text only (images are saved to disk)
        return _text_result("\n".join(lines) +
                            _budget_footer(ctx, rollouts_before))

    return {
        "evaluate_predicate_on_trajectory": evaluate_predicate_on_trajectory,
        "evaluate_option_plan": evaluate_option_plan,
    }
