"""Agent model-based approach: the agent delivers a simulator-validated plan.

The agent plans a sequence of parameterized skills with object bindings,
subgoal atoms after each step, and continuous parameters, and must
DELIVER it as an ``evaluate_option_plan`` capture on the current task -
nothing it did not validate in the simulator (the model) is ever
executed. A backtracking parameter search remains available to the agent
as a tool (``refine_plan_sketch`` / ``sim.refine``) and to mid-episode
suffix replans, but there is no approach-side refinement of unvalidated
sketches.

Registered under the CLI approach name ``agent_bilevel`` (kept stable so
existing configs and logs remain valid).

Example command::

    python predicators/main.py --env pybullet_domino \
        --approach agent_bilevel --seed 0 \
        --num_train_tasks 1 --num_test_tasks 1 \
        --num_online_learning_cycles 1 --explorer agent_plan
"""
import dataclasses
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.session_base import AgentSessionFatalError
from predicators.agent_sdk.sketch_types import SketchStep as _SketchStep
from predicators.agent_sdk.tools import BUILTIN_TOOLS, \
    explore_python_replaces_tools, load_ground_sampler_fns
from predicators.approaches import ApproachFailure
from predicators.approaches.agent_model_free_approach import \
    AgentModelFreeApproach
from predicators.execution_monitoring.subgoal_annotations_monitor import \
    SubgoalExecutionStatus
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, _Option

# Fraction of agent_solve_attempt_wall_clock below which the attempt's
# budget counts as spent, not merely close to it: agents that watch the
# [budget] footer wrap up shortly BEFORE the deadline, and a query whose
# remaining tools would only refuse has nothing left to give. Used to
# label why an attempt ended (see _attempt_end_reason).
_SPENT_WALL_FRACTION = 0.2

# Cap on the natural-language goal text in a task's journal entry: long
# enough for any real goal_nl, short enough that a runaway goal string
# cannot crowd the journal's entry budget.
_JOURNAL_GOAL_MAX_CHARS = 400

# Final-submission nudge (see _nudge_final_submission). It runs once per
# task, on the LAST attempt, so it accepts a plan that falls short of a
# validated solve: the restarts are gone and a partial plan beats
# forfeiting the task.
_FINAL_SUBMIT_NUDGE = (
    "You are out of exploration budget for this attempt. Do NOT explore "
    "further. In as few tool calls as possible, submit your single best "
    "plan NOW via evaluate_option_plan on the current task (omit "
    "task_idx), using the best parameters you have already validated. "
    "It is captured as your answer even if it does not fully reach the "
    "goal or does not score as a solve; then finish.")


@dataclasses.dataclass
class _CaptureInfo:
    """Metadata of the most recently consumed captured plan.

    Recorded by :meth:`AgentModelBasedApproach._consume_validated_plan` so
    the restart loop can distinguish a validated solve (return
    immediately) from a best-effort capture (bank it, rank across
    attempts by evaluator reward) and journal the plan.
    """
    validated: bool
    reward: Optional[float]
    plan_lines: List[str]


class AgentModelBasedApproach(AgentModelFreeApproach):
    """Model-based planning: the agent plans a skeleton with subgoals and
    parameters and submits it as a simulator-validated capture.

    Extends AgentModelFreeApproach - reuses agent session, tools,
    trajectory management, exploration, save/load.  Overrides solving
    with the capture-only query loop plus the restart/journal machinery.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if CFG.agent_bilevel_max_execution_replans > 0 and \
                CFG.execution_monitor != "subgoal_annotations":
            raise ValueError(
                "agent_bilevel_max_execution_replans > 0 requires "
                "--execution_monitor subgoal_annotations (got "
                f"{CFG.execution_monitor!r}): divergence detection lives "
                "in the execution monitor, so without it test execution "
                "is silently open-loop.")
        # Live status of the currently executing annotated plan, exported
        # to the subgoal_annotations execution monitor. None whenever no
        # monitored plan is active (exploration, replanning disabled).
        self._exec_status: Optional[SubgoalExecutionStatus] = None
        # Per-episode replan budget, refreshed by reset_for_new_episode.
        self._exec_replans_left = 0
        # Whether the most recent sketch query ended because the agent hit
        # agent_sdk_max_agent_turns_per_iteration. Set by
        # _query_agent_for_plan_sketch; every query ending is terminal for
        # its attempt, so this only labels WHY the attempt ended (see
        # _attempt_end_reason) in the logs and the journal.
        self._last_sketch_query_hit_turn_cap = False
        # Why the last capture-less attempt ended (_attempt_end_reason),
        # sampled inside the attempt while its budget signals are still
        # armed - _solve clears them before writing the journal entry.
        self._last_attempt_end_reason = ""
        # Metadata of the last capture consumed by
        # _consume_validated_plan; read by the restart loop in _solve.
        self._last_capture_info: Optional[_CaptureInfo] = None
        # Tasks whose goal + init-state journal entry is already written
        # (one context entry per task, at the top of its section).
        self._journal_task_context_recorded: Set[Any] = set()
        # Snapshot of that set at begin_test_phase: test-task keys are
        # rolled back with the journal itself, so a later evaluation
        # (whose entries were removed) re-writes its context entries.
        self._pre_test_journal_context_keys: Optional[Set[Any]] = None

    @classmethod
    def get_name(cls) -> str:
        return "agent_bilevel"

    # ------------------------------------------------------------------ #
    # Execution monitoring (closed-loop test execution)
    # ------------------------------------------------------------------ #

    def reset_for_new_episode(self) -> None:
        super().reset_for_new_episode()
        self._exec_status = None
        self._exec_replans_left = CFG.agent_bilevel_max_execution_replans
        # Optionally give each test solve a fresh agent conversation. reset()
        # fires once per test task (not on mid-episode replans, which go
        # through step()); the next query lazily rebuilds the session with the
        # same sandbox + artifacts but empty chat context. Test-phase only, so
        # exploration episodes keep their shared session.
        if CFG.agent_fresh_session_per_test_task and self._in_test_phase:
            self._close_agent_session()

    def get_execution_monitoring_info(self) -> List[Any]:
        if self._exec_status is None:
            return []
        return [self._exec_status]

    def begin_test_phase(self) -> None:
        super().begin_test_phase()
        self._pre_test_journal_context_keys = set(
            self._journal_task_context_recorded)

    def end_test_phase(self) -> None:
        super().end_test_phase()
        # The journal rollback removed this evaluation's entries, so its
        # task-context dedup keys must go too - the same test tasks are
        # re-solved next evaluation and need fresh goal + init entries.
        if self._pre_test_journal_context_keys is not None:
            self._journal_task_context_recorded = \
                self._pre_test_journal_context_keys
            self._pre_test_journal_context_keys = None

    # ------------------------------------------------------------------ #
    # Agent session hooks
    # ------------------------------------------------------------------ #

    def _get_synthesis_tool_names(self) -> Optional[List[str]]:
        """No synthesis phase in this approach - declare an empty set."""
        return []

    def _get_solve_tool_names(self) -> Optional[List[str]]:
        # Bilevel solving hands continuous refinement to a search, so the
        # agent also gets refine_plan_sketch (backtracking refinement +
        # forward validation on a param-free sketch). Needs a simulator.
        # explore_python's sim.refine subsumes it (same search core, from
        # any state); when explore_python is on, the standalone tool is
        # offered only if the keep-replaced-tools flag asks for both.
        tools = list(super()._get_solve_tool_names() or [])
        if CFG.agent_planner_use_simulator and \
                not explore_python_replaces_tools():
            tools.append("refine_plan_sketch")
        return tools

    # ------------------------------------------------------------------ #
    # System prompt (simplified - no parameter tuning workflow)
    # ------------------------------------------------------------------ #

    def _get_agent_system_prompt(self) -> str:
        propose = CFG.agent_bilevel_use_llm_initial_params
        # When explore_python replaces the standalone refine tool, every
        # guidance mention must point at the probe equivalent instead of
        # a tool the session lacks.
        probe_replaces = explore_python_replaces_tools()
        refine_ref = ("sim.refine (in explore_python)"
                      if probe_replaces else "refine_plan_sketch")
        # What a sketch step consists of (shared between modes).
        if propose:
            sketch_desc = (
                "a sequence of skills (parameterized options) with object "
                "arguments, subgoal atoms after each step, and the continuous "
                "parameters for each step")
        else:
            sketch_desc = (
                "a sequence of skills (parameterized options) with object "
                "arguments and subgoal atoms after each step, plus continuous "
                f"parameters you find with {refine_ref}")

        # One session serves two query kinds with different delivery
        # contracts: SOLVE (hard capture gate) and EXPLORE (experiment
        # sketch - the belief model may lack goal-critical dynamics, so
        # a goal-reaching capture can be impossible by construction).
        # State both contracts; each query's Instructions pick one.
        job = ("Your job is to produce a plan - " + sketch_desc +
               " - that reaches the goal. Queries arrive in two kinds; "
               "each query's Instructions section states which contract "
               "applies.\n"
               "- SOLVE queries: you DELIVER by running "
               "evaluate_option_plan with per-step subgoals on the current "
               "task until it reaches the goal - that captured plan is "
               "your ONLY accepted output, so do not finish until "
               "evaluate_option_plan reaches the goal.\n"
               "- EXPLORE queries: the deliverable is your final "
               "plan-sketch TEXT, an experiment to run in the real "
               "environment. A simulator-validated sketch is preferred "
               "when the current model supports one; when it does not, "
               "submit the sketch most likely to work in reality instead "
               "of grinding for a capture the model cannot produce.\n"
               "evaluate_option_plan runs your EXACT parameters with no "
               "sampling, so every parameter must be right. To find working "
               f"values you MAY use {refine_ref} while reasoning (it "
               "searches for parameters but is slower); read the parameters "
               "it reports and submit them via evaluate_option_plan. Use "
               "whatever tools help.")
        if propose:
            job += (" Where many values work, any reasonable parameter is "
                    "fine; where good values are hard to hit (tight "
                    f"tolerances), use {refine_ref} to search for one.")
            if CFG.agent_bilevel_ground_samplers:
                job += (" Confine its search near your estimate by appending "
                        "a region `~ [w1, w2]` of per-parameter half-widths "
                        "after a step's `[params]`, or `~ my_sampler` naming "
                        "a GROUND_SAMPLERS entry you wrote in "
                        "ground_samplers.py for state-dependent regions.")

        # Keep responses short: the model's deliberation is the main driver of
        # the output-token overflow, and testing is often faster than deriving.
        brevity = (
            " Keep your reasoning concise: prefer making a concrete attempt "
            f"and testing it with {refine_ref} / evaluate_option_plan to "
            "let the simulator tell you what's wrong.")
        params_clause = job + brevity + "\n\n"
        # Keep the subgoal-annotation template's option format consistent with
        # the solve prompt: show the [params] slot iff the agent proposes them.
        param_slot = "[param1, param2]" if propose else ""
        wait_slot = "[]" if propose else ""
        return (
            "You are a planning agent. You observe task environments through "
            "inspection tools and generate plan sketches to achieve goals. "
            "You have access to read-only tools to inspect predicates, "
            "options, trajectories, and training tasks.\n\n"
            f"{params_clause}"
            "Some effects may not be immediate - if an action triggers a "
            "delayed process (e.g. gradual accumulation, propagation "
            "through contacting objects, a sensor catching up to an "
            "actuator), insert a Wait after it so the effect has time "
            "to occur before the next action.\n\n"
            "## Subgoal Annotations\n"
            "After each step, annotate which predicate atoms should hold "
            "after that step succeeds. Use the format:\n"
            f"  OptionName(obj1:type1, obj2:type2){param_slot} -> "
            "{Pred(obj1:type1), Pred2(obj1:type1, obj2:type2)}\n"
            "Always use typed references (obj:type) in subgoal atoms.\n"
            "Annotate EVERY step whose effect the predicates can express. "
            "Annotations are not just search hints: refinement validates "
            "each annotated step, and at execution time they are checked "
            "against the real state, so a step that diverges can be "
            "detected and replanned instead of silently dooming the rest "
            "of the plan. Prefer atoms that NEWLY hold (or stop holding) "
            "because of the step - atoms that were already true beforehand "
            "cannot reveal divergence. A step you cannot annotate is a "
            "blind spot for both search and recovery.\n"
            "For Wait steps, the annotation also specifies exactly when the "
            "Wait should terminate. Use `NOT Pred(...)` for atoms that should "
            f"become false (e.g. `Wait(robot:robot){wait_slot} -> "
            "{Ready(widget:widget)}`).")

    # ------------------------------------------------------------------ #
    # Solve prompt (no continuous params, subgoal format)
    # ------------------------------------------------------------------ #

    def _build_solve_prompt(self, task: Task) -> str:
        """Build prompt asking for a plan sketch without continuous params."""
        journal_text = ""
        if CFG.agent_solve_use_journal:
            # pylint: disable-next=import-outside-toplevel
            from predicators.agent_sdk import journal as journal_mod
            journal_text = journal_mod.read_journal(
                self._tool_context.sandbox_dir)
        return bilevel_sketch.build_solve_prompt(
            task,
            all_predicates=self._get_all_predicates(),
            all_options=self._get_all_options(),
            trajectory_summary=self._build_trajectory_summary(),
            tool_names=self._solve_prompt_tool_names(),
            initial_image_section=self._initial_image_section(),
            propose_params=CFG.agent_bilevel_use_llm_initial_params,
            require_tool_validation=True,
            ground_samplers=CFG.agent_bilevel_ground_samplers,
            journal=journal_text,
            physics_margin=CFG.agent_plan_validation_physics_margin,
        )

    def _solve_prompt_tool_names(self) -> Optional[List[str]]:
        """Tool list advertised in the solve prompt's "Available Tools".

        Mirrors what the explore prompt lists (the explorer renders
        ``agent_session.tool_names``): the same MCP subset *plus* the
        sandbox's built-in tools (Bash/Read/Write/...). The built-ins are
        only actually granted under the local or docker sandbox -- which
        is exactly when ``LocalSandboxSessionManager.tool_names`` prepends
        them -- so they are advertised only then. Without a sandbox the
        list is the bare MCP subset, unchanged.
        """
        names = self._get_solve_tool_names()
        if names is None:
            return None
        if CFG.agent_sdk_use_local_sandbox or CFG.agent_sdk_use_docker_sandbox:
            return list(BUILTIN_TOOLS) + names
        return names

    # ------------------------------------------------------------------ #
    # Solving
    # ------------------------------------------------------------------ #

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        replan_policy = self._maybe_replan_from_divergence(task, timeout)
        if replan_policy is not None:
            return replan_policy
        ctx = self._tool_context
        self._record_task_context_in_journal(task)
        max_attempts = max(1, CFG.agent_solve_max_attempts)
        wall_clock = CFG.agent_solve_attempt_wall_clock
        # Best best-effort capture across attempts, ranked by evaluator
        # reward. A validated (evaluator-solved) capture returns
        # immediately; only when no attempt produces one does the best
        # banked policy execute for its honest reward.
        best_policy: Optional[Callable[[State], Action]] = None
        best_reward = -float("inf")
        last_failure: Optional[ApproachFailure] = None
        for attempt in range(1, max_attempts + 1):
            if CFG.agent_solve_fresh_context:
                # Fresh conversation per attempt (and per test task): a
                # failed attempt's context carries its confidently wrong
                # world model (run_20260717_230436 seed1's "hard collision
                # boundary" that its identical sibling placed through), so
                # a restart is the cheapest de-anchoring mechanism. Curated
                # knowledge travels through the solve journal instead.
                self._close_agent_session()
            ctx.begin_attempt(attempt, wall_clock)
            self._last_capture_info = None
            policy: Optional[Callable[[State], Action]] = None
            unexpected: Optional[Exception] = None
            try:
                policy = self._solve_attempt(task)
            except ApproachFailure as e:
                last_failure = e
            except AgentSessionFatalError:
                # The session backend is unusable (auth/billing/config);
                # neither a banked capture nor further restarts can help.
                # Re-raise so the run terminates (the finally still runs
                # for bookkeeping).
                raise
            except Exception as e:  # pylint: disable=broad-except
                # ApproachTimeout is a SIBLING of ApproachFailure (both
                # subclass ExceptionWithInfo), and env/SDK errors can
                # also escape - none of them may skip the cleanup below,
                # and a banked capture from an earlier attempt should
                # still execute rather than be forfeited (handled after
                # the finally).
                unexpected = e
            finally:
                # Attempt bookkeeping must not outlive the attempt on ANY
                # exit path (including KeyboardInterrupt): stale fields
                # would append bogus [budget] footers and mislabel journal
                # entries in later sessions sharing this ToolContext.
                ctx.attempt_deadline = None
                info = self._take_capture_info()
                self._record_attempt_in_journal(attempt, max_attempts, policy,
                                                info)
                ctx.attempt_start = None
                ctx.attempt_index = 0
            if unexpected is not None:
                if best_policy is not None:
                    logging.warning(
                        "[%s] Solve attempt %d/%d raised %s; executing the "
                        "banked best-effort capture instead of forfeiting.",
                        self._run_id, attempt, max_attempts, unexpected)
                    return best_policy
                raise unexpected
            if policy is not None and (info is None or info.validated):
                # Defensive: every capture path records metadata, so a
                # missing record is treated as a validated solve rather
                # than banked at unknown reward.
                return policy
            if policy is not None:
                assert info is not None
                reward = (info.reward
                          if info.reward is not None else -float("inf"))
                if best_policy is None or reward > best_reward:
                    best_policy = policy
                    best_reward = reward
            if attempt < max_attempts:
                logging.info(
                    "[%s] Solve attempt %d/%d ended without a validated "
                    "solve%s; restarting with %s context.", self._run_id,
                    attempt, max_attempts, " (best-effort capture banked)"
                    if policy is not None else "",
                    "fresh" if CFG.agent_solve_fresh_context else "the same")
        if best_policy is not None:
            logging.info(
                "[%s] No validated solve in %d attempt(s); executing the "
                "best best-effort capture (evaluator reward %s).",
                self._run_id, max_attempts,
                f"{best_reward:.2f}" if best_reward > -float("inf") else "n/a")
            return best_policy
        if last_failure is not None:
            raise last_failure
        raise ApproachFailure(
            f"Bilevel solve produced no captured plan in {max_attempts} "
            "attempt(s).")

    def _attempt_wall_spent(self) -> bool:
        """Whether the attempt's wall clock is spent (or nearly so).

        True once less than :data:`_SPENT_WALL_FRACTION` of the wall
        clock remains, not merely once the deadline passes: agents that
        watch the [budget] footer wrap up shortly BEFORE the deadline
        (run_20260718_125643 ended an attempt with 2 minutes left), and
        an attempt whose remaining tools would only refuse is spent in
        every sense that matters. Tool-side refusals keep using the
        exact deadline, so the closing minutes still allow submissions.
        """
        deadline = self._tool_context.attempt_deadline
        if deadline is None:
            return False
        floor = _SPENT_WALL_FRACTION * CFG.agent_solve_attempt_wall_clock
        return time.monotonic() > deadline - floor

    def _attempt_end_reason(self) -> str:
        """Why the attempt's single query ended, for logs and journal."""
        if self._last_sketch_query_hit_turn_cap:
            return "turn cap"
        if self._attempt_wall_spent():
            return "wall clock spent"
        return "no submission"

    def _take_capture_info(self) -> Optional[_CaptureInfo]:
        """Pop the metadata _consume_validated_plan recorded (or None).

        An accessor rather than a bare attribute read: the attribute is
        set as a side effect of _solve_attempt, which mypy cannot see,
        so reading it directly right after ``= None`` is flagged
        unreachable.
        """
        info = self._last_capture_info
        self._last_capture_info = None
        return info

    def _append_journal_auto_entry(self, header: str,
                                   body_lines: List[str]) -> bool:
        """Best-effort append of a harness-written solve-journal entry.

        Callers guard with :meth:`_journal_active`. Returns True on a
        successful write; a failed write is logged, never raised - the
        journal must not be able to fail a solve.
        """
        sandbox_dir = self._tool_context.sandbox_dir
        assert self._journal_active() and sandbox_dir
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk import journal as journal_mod
        try:
            journal_mod.append_entry(
                sandbox_dir,
                header,
                "\n".join(body_lines),
                max_chars=journal_mod.MAX_AUTO_ENTRY_CHARS)
        except OSError as e:
            logging.warning("Journal entry %r failed: %s", header, e)
            return False
        return True

    def _journal_task_label(self) -> str:
        """The journal's label for the task currently being solved."""
        idx = self._tool_context.test_task_idx
        return f"task {idx}" if idx is not None else "task ?"

    def _record_task_context_in_journal(self, task: Task) -> None:
        """Append the task's goal + init-state entry, once per task.

        Written at the START of the task's first attempt so it tops the
        task's journal section - above even the agent's own in-attempt
        notes - keeping every later entry interpretable: a recorded
        plan's geometry is only meaningful relative to its layout. The
        init dict uses the exact representation the solve prompt shows
        (including its excluded-objects filtering).
        """
        if not self._journal_active():
            return
        key = (self._tool_context.test_task_idx, id(task))
        if key in self._journal_task_context_recorded:
            return
        if task.goal_nl:
            goal_txt = " ".join(task.goal_nl.split())
            if len(goal_txt) > _JOURNAL_GOAL_MAX_CHARS:
                goal_txt = goal_txt[:_JOURNAL_GOAL_MAX_CHARS].rstrip() + "..."
        else:
            goal_txt = ", ".join(str(a) for a in sorted(task.goal, key=str))
        body = [f"- goal: {goal_txt}", "- initial state features:"]
        body.extend(f"  {line}"
                    for line in task.init.dict_str(indent=2).splitlines())
        header = f"{self._journal_task_label()} goal + initial state (auto)"
        if self._append_journal_auto_entry(header, body):
            self._journal_task_context_recorded.add(key)

    def _record_attempt_in_journal(self, attempt: int, max_attempts: int,
                                   policy: Optional[Any],
                                   info: Optional[_CaptureInfo]) -> None:
        """Auto-append this attempt's factual record to the solve journal.

        The harness-written record (outcome, budget spent, captured or
        best refused plan) guarantees the journal's essentials even when
        the agent records nothing; agent-authored lessons arrive
        separately via the record_journal tool.
        """
        if not self._journal_active():
            return
        ctx = self._tool_context
        body = [f"- outcome: {self._attempt_outcome_text(policy, info)}"]
        if ctx.attempt_start is not None:
            elapsed_min = (time.monotonic() - ctx.attempt_start) / 60.0
            body.append(f"- budget spent: {elapsed_min:.1f} min, "
                        f"{ctx.attempt_rollout_count} sim rollouts")
        if info is not None and info.plan_lines:
            body.append("- captured plan:")
            body.extend(f"  {line}" for line in info.plan_lines)
        elif ctx.best_uncaptured_plan_lines:
            # Nothing captured, but the attempt's best refused submission
            # (evaluator non-solve or flaky) is worth carrying: a later
            # attempt - or the final best-effort nudge - can resubmit it
            # instead of the work vanishing with the attempt's context.
            reward_txt = (f"evaluator reward {ctx.best_uncaptured_reward:.2f}"
                          if ctx.best_uncaptured_reward is not None else
                          "no evaluator verdict")
            body.append(f"- best refused submission ({reward_txt}, "
                        "not captured):")
            body.extend(f"  {line}" for line in ctx.best_uncaptured_plan_lines)
        header = (f"{self._journal_task_label()} attempt "
                  f"{attempt}/{max_attempts} (auto)")
        self._append_journal_auto_entry(header, body)

    def _attempt_outcome_text(self, policy: Optional[Any],
                              info: Optional[_CaptureInfo]) -> str:
        """One-line outcome for an attempt's journal record."""
        if policy is None:
            # The reason is the one fact a fresh-context restart cannot
            # rediscover: it tells the next attempt whether the last one
            # ran out of budget or talked itself out of submitting.
            if self._last_attempt_end_reason:
                return f"no capture ({self._last_attempt_end_reason})"
            return "no capture"
        if info is None or info.validated:
            return "SOLVED (validated capture)"
        if info.reward is not None:
            return f"best-effort capture (evaluator reward {info.reward:.2f})"
        return "best-effort capture (no evaluator verdict)"

    def _solve_attempt(self, task: Task) -> Callable[[State], Action]:
        """One full solve attempt: a single agent query on one session.

        The attempt's budgets are the wall clock
        (``agent_solve_attempt_wall_clock``) and the query's turn cap;
        the only deliverable is an ``evaluate_option_plan`` capture
        (consumed via :meth:`_consume_validated_plan`).

        However that query ends - a spent budget, an unparseable sketch,
        or a session that simply never submitted - the attempt is over.
        A second query on the same conversation would re-explore from a
        context that already contains whatever went wrong
        (run_20260808_113951 queries 004-009: three full-price queries
        restating the same "no plan can reach the goal" argument), so the
        fresh-context restart is the only retry. See :meth:`_end_attempt`
        for the one exception, on the final attempt.
        """
        self._sync_tool_context()
        self._tool_context.current_task = task
        # Let evaluate_option_plan record a goal-reaching
        # plan on this task into solved_plan/solved_sketch (consumed below).
        self._tool_context.capture_goal_reaching_plans = True
        # Render the initial state so the agent can see the scene layout.
        self._render_initial_state_image(task)
        # Whether later fresh-context restarts exist after this attempt;
        # decides whether this attempt pays for the final-submission nudge
        # (see _end_attempt). attempt_index == 0 (no restart loop in
        # flight) behaves like a final attempt.
        restarts_remain = (0 < self._tool_context.attempt_index < max(
            1, CFG.agent_solve_max_attempts))
        # Clear any prior capture so we only act on this query's result.
        self._tool_context.clear_plan_capture()
        self._last_sketch_query_hit_turn_cap = False
        self._last_attempt_end_reason = ""
        try:
            self._query_agent_for_plan_sketch(task)
        except AgentSessionFatalError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            # The agent may have validated a working plan via
            # refine_plan_sketch even if its final text didn't parse.
            policy = self._consume_validated_plan()
            if policy is not None:
                return policy
            logging.warning("[%s] Solve query failed: %s", self._run_id, e)
        else:
            # Fast path: the agent already refined + forward-validated
            # a plan on this task via refine_plan_sketch - return it
            # directly instead of re-refining the (possibly different)
            # final-text sketch.
            policy = self._consume_validated_plan()
            if policy is not None:
                return policy
            # The agent must itself reach a confirmed
            # evaluate_option_plan capture (consumed above) so we
            # never execute a plan it didn't verify.
            logging.info("[%s] Query ended without a validated plan.",
                         self._run_id)
        # Sample the end reason before _end_attempt: the nudge suspends
        # the attempt deadline, and _solve clears it outright before the
        # journal entry is written.
        self._last_attempt_end_reason = self._attempt_end_reason()
        policy = self._end_attempt(restarts_remain)
        if policy is not None:
            return policy
        raise ApproachFailure("Bilevel solve failed: the attempt's agent "
                              "query produced no captured plan "
                              f"({self._last_attempt_end_reason}).")

    def _end_attempt(
            self,
            restarts_remain: bool) -> Optional[Callable[[State], Action]]:
        """End an attempt that produced no capture.

        With later fresh-context restarts remaining, end with NO nudge
        (return None): the restart is the retry, and the journal
        auto-entry already records the attempt's best refused
        submission. On the final attempt the best-effort submission
        nudge is the ultimate fallback - return whatever policy it
        captures (None when even that yields nothing, giving up on the
        task).
        """
        if restarts_remain:
            return None
        return self._nudge_final_submission()

    # ------------------------------------------------------------------ #
    # Plan sketch extraction
    # ------------------------------------------------------------------ #

    def _query_agent_for_plan_sketch(self, task: Task) -> List[_SketchStep]:
        """Query agent for a plan sketch and parse it."""
        sketch_file = CFG.agent_bilevel_plan_sketch_file
        if sketch_file:
            # An absolute path is used as-is; a bare filename is resolved
            # against the configured plan-sketch directory under scripts/.
            if os.path.isabs(sketch_file):
                filepath = sketch_file
            else:
                filepath = (
                    f"{utils.get_path_to_predicators_root()}/scripts/"
                    f"{CFG.agent_bilevel_plan_sketch_dir}/{sketch_file}")
            with open(filepath, "r", encoding="utf-8") as f:
                plan_text = f.read().strip()
            logging.info("Loaded plan sketch from file: %s", sketch_file)
        else:
            prompt = self._build_solve_prompt(task)
            responses = self._query_agent_sync(prompt, kind="test")
            # Record cap-exhaustion before parsing: a capped session usually
            # has no final text, so the "empty plan text" failure below is
            # still attributable to the turn cap by _attempt_end_reason.
            self._last_sketch_query_hit_turn_cap = \
                self._responses_hit_turn_cap(responses)
            plan_text = self._extract_option_plan_text(responses)

        if not plan_text:
            raise ApproachFailure("Agent returned empty plan text.")

        # Tolerant parse of the agent's final text; named `~ my_sampler`
        # references resolve against the sandbox's ground_samplers.py (a
        # broken file just drops the annotations here - this is the
        # best-effort fallback path, not the strict tool path).
        gs_fns, gs_err = load_ground_sampler_fns(self._tool_context)
        if gs_err is not None:
            logging.warning("[%s] %s", self._run_id, gs_err)
        sketch = bilevel_sketch.parse_sketch_from_text(
            plan_text,
            task,
            predicates=self._get_all_predicates(),
            options=self._get_all_options(),
            types=self._types,
            parse_continuous_params=CFG.agent_bilevel_use_llm_initial_params,
            parse_ground_samplers=CFG.agent_bilevel_ground_samplers,
            ground_sampler_fns=gs_fns or None,
        )

        if not sketch:
            option_names = sorted(o.name for o in self._get_all_options())
            raise ApproachFailure(f"Parsed empty plan sketch from agent.\n"
                                  f"  Plan text:\n{plan_text}\n"
                                  f"  Available option names: {option_names}")

        logging.info(
            "[%s] Agent produced sketch with %d steps, %d with "
            "subgoals.", self._run_id, len(sketch),
            sum(1 for s in sketch if s.subgoal_atoms))
        return sketch

    @staticmethod
    def _responses_hit_turn_cap(responses: List[Dict[str, Any]]) -> bool:
        """Whether a query's response stream ended on the SDK turn cap.

        The SDK reports the cap as result subtype ``error_max_turns``;
        the num_turns comparison is a fallback for backends whose result
        entries lack the subtype field.
        """
        max_turns = CFG.agent_sdk_max_agent_turns_per_iteration
        for entry in responses:
            if entry.get("type") != "result":
                continue
            if entry.get("subtype") == "error_max_turns":
                return True
            num_turns = entry.get("num_turns")
            if num_turns is not None and num_turns >= max_turns:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Backtracking refinement (used by mid-episode suffix replans)
    # ------------------------------------------------------------------ #

    def _refine_sketch(
        self,
        task: Task,
        sketch: List[_SketchStep],
        timeout: float,
        attempt: int = 0,
        on_step_fail: Optional[Callable[[int, List[Optional[_Option]], str],
                                        None]] = None,
    ) -> Tuple[List[_Option], bool]:
        """Backtracking search over continuous parameters for a plan sketch.

        Returns ``(plan, success)``.  On success, ``plan`` is a list of
        grounded options that achieves the task goal.  On failure,
        ``plan`` is the longest partial refinement found.

        This is the approach-flavored entry to
        ``bilevel_sketch.refine_sketch`` (which stays settings-free):
        it gathers the approach-owned inputs (option model, predicates,
        samplers, run id), reads the search knobs from ``CFG``, and
        first passes the task through :meth:`_attach_initial_latent` so
        partially-observable approaches can seed ``task.init.latent``
        with the initial latent block. Used by mid-episode suffix
        replans and by the offline replay scripts under
        ``scripts/domino_debug/``.

        ``attempt`` perturbs the RNG so retries explore different
        samples - without it, refinement is deterministic in
        ``CFG.seed`` and a forward-validation failure would loop on
        the identical plan. ``on_step_fail`` is forwarded to the search
        (called with the step index, the partial plan, and the failure
        reason whenever a step fails to refine).
        """
        task = self._attach_initial_latent(task)
        assert self._option_model is not None, \
            "agent_bilevel requires a simulator " \
            "(agent_planner_use_simulator=True)."
        outcome = bilevel_sketch.refine_sketch(
            task,
            sketch,
            self._option_model,
            predicates=self._get_all_predicates(),
            timeout=timeout,
            rng=np.random.default_rng(CFG.seed + attempt),
            max_samples_per_step=CFG.agent_bilevel_max_samples_per_step,
            check_subgoals=CFG.agent_bilevel_check_subgoals,
            log_state=CFG.agent_bilevel_log_state,
            run_id=self._run_id,
            parameterized_samplers=self._get_all_samplers(),
            on_step_fail=on_step_fail,
        )
        return outcome.plan, outcome.success

    def _attach_initial_latent(self, task: Task) -> Task:
        """Hook for partial-observability approaches to seed the latent.

        Subclasses that thread a ``latent`` state block through the
        simulator (e.g. ``AgentPOSimPredicateInventionApproach``)
        override this to attach an initial latent to
        ``task.init.latent`` before refinement begins. The default
        returns ``task`` unchanged - fully-observable approaches need do
        nothing.
        """
        return task

    def _sample_params(self, option: ParameterizedOption, _state: State,
                       rng: np.random.Generator) -> np.ndarray:
        """Sample continuous parameters for an option."""
        return bilevel_sketch.sample_params(option, rng)

    def _parse_subgoal_annotations(
        self,
        text: str,
        predicates: Set[Predicate],
        objects: Sequence[Object],
    ) -> List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]]:
        """Shim over ``bilevel_sketch.parse_subgoal_annotations``."""
        option_names = {o.name for o in self._get_all_options()}
        return bilevel_sketch.parse_subgoal_annotations(
            text, predicates, objects, option_names)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _maybe_replan_from_divergence(
            self, task: Task,
            timeout: int) -> Optional[Callable[[State], Action]]:
        """Handle a mid-episode re-solve triggered by the subgoal_annotations
        execution monitor.

        CogMan calls solve() identically at episode start and on a
        monitor-triggered replan; ``_exec_status`` distinguishes them
        (non-None only while a monitored plan executes;
        reset_for_new_episode clears it at episode start). On a replan
        ``task.init`` is the real state where the just-finished step's
        annotation failed. Divergence is usually a continuous-execution
        problem (a sampled parameter whose real outcome differed from
        the option-model rollout), not a wrong skeleton, so we first try
        to resume a suffix of the executed sketch (cheap, no agent
        query; see :meth:`_replan_suffix`). When that fails, the default
        is to fail the episode: a fresh agent sketch query would re-open
        a turn budget the attempt already spent (set
        ``CFG.agent_bilevel_replan_agent_fallback`` to return None and
        fall through to one instead). Also raises ApproachFailure when
        the episode's replan budget is exhausted so the episode fails
        fast instead of running the horizon open-loop.
        """
        status = self._exec_status
        if status is None or status.steps_initiated == 0:
            return None
        self._exec_status = None
        failed_idx = status.steps_initiated - 1
        steps = list(status.sketch)
        failed_name = steps[failed_idx].option.name
        if self._exec_replans_left <= 0:
            raise ApproachFailure(
                f"Subgoal divergence after step {failed_idx} "
                f"({failed_name}). No execution replans left.")
        self._exec_replans_left -= 1
        logging.info(
            "Subgoal divergence after step %d (%s). Replanning from the "
            "current state (%d execution replans left).", failed_idx,
            failed_name, self._exec_replans_left)
        policy = self._replan_suffix(task.init, task, steps, failed_idx,
                                     timeout)
        if policy is None:
            if not CFG.agent_bilevel_replan_agent_fallback:
                raise ApproachFailure(
                    f"Subgoal divergence after step {failed_idx} "
                    f"({failed_name}): no suffix of the executed sketch "
                    "refines from here, and the fresh-agent-sketch "
                    "fallback is disabled "
                    "(agent_bilevel_replan_agent_fallback).")
            # No suffix of the executed skeleton refines from here; fall
            # through to pay for a fresh agent sketch.
            logging.info("Suffix replan failed; querying the agent for a "
                         "fresh sketch.")
        return policy

    def _nudge_final_submission(self) -> Optional[Callable[[State], Action]]:
        """One short follow-up query on the LAST attempt, after its query ended
        with no captured plan: tell the agent to submit its best plan now.

        A session that hits the turn cap mid-iteration contributes
        nothing, even when it has a near-working plan in context; this
        converts that dead end into a submission attempt at the cost of a
        few turns.

        The submitted plan is captured and executed even if its belief
        rollout does not reach the goal, is scored a non-solve by the
        task evaluator, or is flaky: there are no restarts left, and a
        partial plan beats forfeiting the task.
        """
        nudge = _FINAL_SUBMIT_NUDGE
        if CFG.agent_solve_use_journal:
            nudge += (
                " If an earlier attempt's entry in the Solve Journal "
                "records a better plan (captured or refused) than "
                "anything from this attempt, resubmit that plan instead."
                " After the submission, call record_journal ONCE with a "
                "short factual entry for later fresh-context attempts and "
                "tasks: what you tried (exact parameters), the key "
                "measurements, and what to try differently - facts and "
                "measurements only, no verdicts like 'impossible'.")
        # SUSPEND (not clear) the attempt deadline for the nudge query:
        # its cooperative refusals and the sandbox interrupt backstop
        # must not block the submission (or the journal entry) itself.
        # Restore it afterwards rather than leaking the None: _solve's
        # per-attempt bookkeeping owns clearing the deadline, and a
        # helper that silently disarms the wall clock is a trap for any
        # future caller that runs mid-attempt.
        saved_deadline = self._tool_context.attempt_deadline
        self._tool_context.attempt_deadline = None
        self._tool_context.capture_best_effort_plan = True
        try:
            self._query_agent_sync(nudge, kind="test")
        except AgentSessionFatalError:
            raise
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Final-submission nudge failed: %s", e)
        finally:
            self._tool_context.capture_best_effort_plan = False
            self._tool_context.attempt_deadline = saved_deadline
        policy = self._consume_validated_plan()
        if policy is not None:
            logging.info(
                "[%s] Final-submission nudge produced a validated plan.",
                self._run_id)
        return policy

    def _consume_validated_plan(self) -> Optional[Callable[[State], Action]]:
        """Return a policy from an agent-validated plan, or None.

        ``evaluate_option_plan`` records a captured (goal-reaching,
        validated) plan on the current solve task into the tool context.
        Returning that exact simulator-verified plan guarantees the
        agent's tool-validated answer is what executes, and avoids a
        fresh refinement that with a different seed might not reproduce
        it.
        """
        capture = self._tool_context.take_plan_capture()
        if not capture.plan:
            return None
        # A capture with reached_goal False was accepted under the
        # best-effort nudge; anything else is a validated solve.
        validated = capture.reached_goal is not False
        lines = list(
            bilevel_sketch.format_plan_lines(capture.plan,
                                             sketch=capture.sketch))
        self._last_capture_info = _CaptureInfo(validated=validated,
                                               reward=capture.eval_reward,
                                               plan_lines=lines)
        verdict = ("simulator-verified" if validated else
                   "best-effort: not a validated solve in the belief rollout")
        # Log the full plan (options + continuous params + subgoal
        # annotations) so the run log shows exactly what will execute.
        logging.info(
            "[%s] Using agent-validated plan from capture "
            "(%d steps, %s):\n%s", self._run_id, len(capture.plan), verdict,
            "\n".join(lines))
        return self._plan_to_policy(capture.plan, sketch=capture.sketch)

    def _plan_to_policy(
        self,
        plan: List[_Option],
        sketch: Optional[List[_SketchStep]] = None,
    ) -> Callable[[State], Action]:
        """Wrap a grounded option plan into a step-by-step policy.

        With ``CFG.agent_bilevel_max_execution_replans > 0`` and a full
        per-step sketch, the policy also publishes a live
        ``SubgoalExecutionStatus`` (via
        ``get_execution_monitoring_info``) that the subgoal_annotations
        execution monitor reads to check, at each option boundary, that
        the just-finished step's annotation holds in the REAL state. On
        divergence the monitor makes CogMan re-invoke solve(), which
        lands in :meth:`_maybe_replan_from_divergence`.
        """
        predicates = self._get_all_predicates()

        def _abstract(s: State) -> Set[GroundAtom]:
            return utils.abstract(s, predicates)

        monitored = (CFG.agent_bilevel_max_execution_replans > 0
                     and sketch is not None and len(sketch) == len(plan))

        queue = list(plan)
        total = len(queue)
        status: Optional[SubgoalExecutionStatus] = None
        if monitored:
            assert sketch is not None
            status = SubgoalExecutionStatus(sketch=list(sketch))
            self._exec_status = status

        def _option_policy(state: State) -> _Option:
            del state  # unused
            if not queue:
                logging.info("Option plan exhausted after %d options.", total)
                # See the twin of this raise in utils.option_plan_to_policy:
                # a finished plan and a failed option arrive as the same
                # exception type, so the normal terminus is flagged.
                raise utils.OptionExecutionFailure(
                    "Option plan exhausted!", info={"plan_exhausted": True})
            option = queue.pop(0)
            num_done = total - len(queue)
            if status is not None:
                status.steps_initiated = num_done
                status.current_option = option
            next_option = None if not queue else queue[0].simple_str()
            logging.info("Executing option %d/%d: %s (remaining=%d, next=%s)",
                         num_done, total, option.simple_str(), len(queue),
                         next_option)
            return option

        inner = utils.option_policy_to_policy(
            _option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            abstract_function=_abstract)
        return self._wrap_option_failures(inner)

    def _replan_suffix(
        self,
        state: State,
        task: Task,
        sketch: List[_SketchStep],
        failed_idx: int,
        timeout: int,
    ) -> Optional[Callable[[State], Action]]:
        """Cheap-first recovery: re-refine a suffix of the current sketch.

        Divergence is usually a continuous-execution problem (a sampled
        parameter whose real outcome differed from the option-model
        rollout), not a wrong skeleton, so before paying for a fresh
        agent sketch we retry the one we have. Candidate resume points
        run from the failed step backward to just after the latest
        earlier annotated step whose subgoals still hold in the current
        state. The holds-check only bounds the walk-back - annotations
        are optional and can hold coincidentally (e.g. a final
        SwitchOff's {Off} atom holds before the switch was ever touched)
        - so every candidate suffix must still refine AND forward-
        validate from the current state before we trust it. Returns None
        when no suffix candidate validates.
        """
        assert self._option_model is not None
        sub_task = Task(state, task.goal)
        resume_floor = 0
        for j in range(failed_idx - 1, -1, -1):
            step = sketch[j]
            if step.subgoal_atoms is None and step.subgoal_neg_atoms is None:
                continue
            pos_ok = all(a.holds(state) for a in (step.subgoal_atoms or set()))
            neg_ok = not any(
                a.holds(state) for a in (step.subgoal_neg_atoms or set()))
            if pos_ok and neg_ok:
                resume_floor = j + 1
                break
        start = time.perf_counter()
        for j in range(failed_idx, resume_floor - 1, -1):
            remaining = timeout - (time.perf_counter() - start)
            if remaining <= 0:
                break
            suffix = list(sketch[j:])
            plan, success = self._refine_sketch(sub_task,
                                                suffix,
                                                remaining,
                                                attempt=j)
            if not success:
                logging.info(
                    "Suffix replan: refinement failed resuming at "
                    "step %d.", j)
                continue
            ok, reason = bilevel_sketch.validate_plan_forward(
                sub_task,
                plan,
                self._option_model,
                predicates=self._get_all_predicates(),
                sketch=suffix,
                run_id=self._run_id,
            )
            if ok:
                logging.info(
                    "Suffix replan: resuming executed sketch at step %d "
                    "(%d steps).", j, len(plan))
                return self._plan_to_policy(plan, sketch=suffix)
            logging.info(
                "Suffix replan: forward validation failed resuming at "
                "step %d: %s", j, reason)
        return None
