"""Shared mutable state between the approach and the MCP tools."""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from predicators.agent_sdk.proposal_exec import ProposalBundle
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import CausalProcess, LowLevelTrajectory, \
    ParameterizedOption, ParameterizedSampler, Predicate, State, Task, Type


@dataclass(frozen=True)
class PlanCapture:
    """A captured plan popped off a :class:`ToolContext` in one piece.

    Returned by :meth:`ToolContext.take_plan_capture` so consumers see
    the four ``solved_plan*`` fields as the single value they are:
    ``plan`` is falsy when nothing was captured.
    """
    plan: Optional[Any]
    sketch: Optional[Any]
    reached_goal: Optional[bool]
    eval_reward: Optional[float]


@dataclass
class ToolContext:
    """Shared mutable state between the approach and MCP tools."""
    types: Set[Type] = field(default_factory=set)
    predicates: Set[Predicate] = field(default_factory=set)
    processes: Set[CausalProcess] = field(default_factory=set)
    options: Set[ParameterizedOption] = field(default_factory=set)
    train_tasks: List[Task] = field(default_factory=list)
    offline_trajectories: List[LowLevelTrajectory] = field(
        default_factory=list)
    online_trajectories: List[LowLevelTrajectory] = field(default_factory=list)
    example_state: Optional[State] = None
    option_model: Optional[_OptionModelBase] = None
    # Synthesis-session override for the explore_python probe: a lazy
    # builder over the CANDIDATE simulator.py (fresh MCMC fit, cached
    # until the file changes). When set, BeliefProbe executes against it
    # instead of ``option_model`` - which during synthesis is the stale
    # pre-synthesis model (real physics on cycle 1: a live-env leak).
    # Installed by the sim-learning approach around its synthesis
    # session only; None everywhere else (solve sessions, and
    # oracle-sim-program sampler sessions where ``option_model`` IS the
    # deployed belief model).
    probe_option_model_provider: Optional[Callable[[],
                                                   _OptionModelBase]] = None
    # The ``sim.fit`` backend for synthesis sessions: fits the candidate
    # simulator's PARAM_SPECS against the recorded data and returns the
    # report text (see ``SynthesisToolkit.fit_runner``). None in solve
    # sessions - the deployed belief model is fixed there, so the probe
    # rejects ``fit`` calls.
    probe_fit_provider: Optional[Callable[..., str]] = None
    # Synthesis-session ``sim.residuals`` backend: computes the
    # per-feature residual report for the current simulator.py rules
    # (see ``SynthesisToolkit.residuals_runner``). None in solve
    # sessions - residuals are a learning diagnostic.
    probe_residuals_provider: Optional[Callable[..., str]] = None
    # Active-experiment info-gain scorer, synced from the learning
    # approach when info-seeking exploration is on:
    # ``(state, atoms) -> disagreement``. The agent_bilevel explorer
    # passes it into refinement so continuous-parameter search prefers
    # candidates that straddle the learned model's decision boundaries.
    # None ⇒ plain feasibility search (default).
    atom_disagreement_fn: Optional[Callable[[State, Any], float]] = None
    # Synthesized per-skill samplers (option name -> sampler), synced from
    # the learning approach when agent_sim_learn_parameterized_samplers is on.
    # The agent_bilevel explorer and synthesis tools pass these into
    # refinement so continuous-parameter search aims at each step's subgoal
    # instead of drawing uniformly. Empty ⇒ uniform sampling (default).
    parameterized_samplers: Dict[str, ParameterizedSampler] = field(
        default_factory=dict)
    current_task: Optional[Task] = None
    iteration_proposals: ProposalBundle = field(default_factory=ProposalBundle)
    planning_results: Dict[str, Any] = field(default_factory=dict)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    skill_factory_context: Dict[str, Any] = field(default_factory=dict)
    proposals_disabled: bool = False  # set True during test-time solving
    log_dir: Optional[str] = None
    env: Optional[Any] = None  # simulator env reference (for rendering)
    image_save_dir: Optional[str] = None  # sandbox path for rendered images
    sandbox_dir: Optional[str] = None  # sandbox root directory
    gt_options_ref_path: Optional[str] = None  # sandbox-relative ref file
    show_option_source: bool = True  # set False when using GT options
    iteration_id: int = 0  # current learning iteration (outer loop)
    turn_id: int = 0  # current query/turn within the session
    # Index of the test task currently being solved (0-based), mirroring
    # main.py's ``test_task_idx``. None outside the test phase. Threaded into
    # the saved session-log filename so test queries are attributable to a task.
    test_task_idx: Optional[int] = None
    test_call_id: int = 0  # incremented per evaluate_option_plan call
    # Managed by AgentSessionMixin: populated from
    # `_build_synthesis_mcp_tools` at session-open, reset to [] for
    # solve sessions. Approaches should not write to this directly —
    # override the builder hook instead.
    extra_mcp_tools: list = field(default_factory=list)
    # Extra Claude Agent SDK ``HookMatcher`` instances applied to the
    # next session that's started. Read once at session start, then
    # frozen for the session's lifetime. Subclasses set this before
    # opening a fresh session and clear it on close.
    extra_session_hooks: Dict[str, list] = field(default_factory=dict)
    # Populated by AgentBilevelExplorer so learning approaches can diff
    # mental-model subgoals against real trajectories.
    # TODO(sim-learning): consume these in learn_from_interaction_results.
    last_sketch_subgoals: Optional[Any] = None
    last_sketch_options: Optional[Any] = None
    # Set by AgentBilevelExplorer per request: did the mental model reach
    # the task goal during refinement? Read by get_interaction_requests to
    # stamp InteractionRequest.mental_model_solved (None ⇒ no verdict).
    last_mental_model_solved: Optional[bool] = None
    # Sketch-line descriptions of the exploration plans already generated
    # this online-learning cycle (a cycle's requests are all generated
    # before any executes). Cleared by get_interaction_requests per cycle,
    # appended by AgentBilevelExplorer per request, and shown in the next
    # explore prompt so the agent proposes a complementary plan instead of
    # repeating the identical one for every request.
    cycle_scheduled_plans: List[str] = field(default_factory=list)
    # Digest of the latest rollout system-ID fit's weak spots
    # (unexplainable segments, unidentified/insensitive params,
    # cross-cycle conflicts), synced from the sim-learning approach.
    # The agent_bilevel explorer appends it to its experiment guidance
    # so the next exploration targets the gaps. None ⇒ no fit ran yet
    # (or it had no weak spots).
    sysid_diagnostics: Optional[str] = None
    # Set by refine_plan_sketch / evaluate_option_plan when a plan is verified
    # to reach the goal on the CURRENT solve task: the simulator-verified plan
    # (grounded options with found params) and the parallel subgoal sketch.
    # The bilevel approach returns this directly instead of re-refining, so
    # the agent's tool-validated answer is exactly what gets executed. None ⇒
    # nothing captured this query.
    solved_plan: Optional[Any] = None
    solved_sketch: Optional[Any] = None
    # Whether the captured solved_plan counts as a validated solve in its
    # belief-sim rollout(s): goal reached, evaluator-certified, and every
    # validation rollout passed. False ⇒ it was a best-effort capture (see
    # below). Cleared together with solved_plan.
    solved_plan_reached_goal: Optional[bool] = None
    # Gate for the above: only approaches that consume captured plans
    # (AgentModelBasedApproach) set this True. Keeps the open-loop
    # planner, which also uses evaluate_option_plan, from recording
    # spurious captures.
    capture_goal_reaching_plans: bool = False
    # Set (with capture_goal_reaching_plans) only for the final-submission
    # nudge after an attempt exhausted its turn budget: evaluate_option_plan
    # then captures the agent's submitted plan on the current task even if it
    # does not reach the goal, is scored a non-solve by the task evaluator,
    # or is flaky, so the approach executes the best-effort plan (for its
    # honest reward) instead of paying for another full-budget attempt. A
    # best-effort capture never displaces a validated-solve capture.
    capture_best_effort_plan: bool = False
    # Fresh-physics scope for capture-validation rollouts: a callable
    # returning a context manager. While entered, ``ctx.option_model``
    # simulates on a freshly constructed env instance instead of the shared
    # session env, whose reset cannot reconstruct state exactly (solver
    # warm-start state, velocity residuals), making repeated rollouts
    # correlated with each other and systematically offset from the fresh
    # real env. Accepts an optional ``physical_overrides`` keyword (a
    # param-name -> value dict applied to the fresh env on top of the
    # identified params) for the physics-margin rollouts. Installed by
    # AgentSimLearningApproach (see ``_fresh_validation_env_scope``);
    # None ⇒ validation rollouts share the session env. Gated by
    # agent_plan_validation_fresh_env.
    validation_env_scope: Optional[Callable[..., Any]] = None
    # Physics-margin points for the capture gate: a zero-arg callable
    # returning the current grid of perturbations spanning +-1 posterior
    # sigma of the identified physical params (full override dicts,
    # ascending; empty when no fit with nonzero posterior width is
    # deployed). A callable rather than a stored list so the points
    # always track the LATEST applied fit. Installed by
    # AgentSimLearningApproach; consumed by evaluate_option_plan under
    # agent_plan_validation_physics_margin and by the sim.run physics
    # sweep.
    physics_margin_provider: Optional[Callable[[], List[Dict[str,
                                                             float]]]] = None
    # Capture-task keys (see ``_capture_task_key``) that have produced a
    # FLAKY rejection in evaluate_option_plan. A flaky submission is direct
    # evidence the agent is tuning in a marginal region where a lucky
    # streak can pass the base rollout gate (run_20260717_182321: a
    # 20/20-swept placement validated 3/3, then failed the real episode),
    # so subsequent captures on these tasks must clear the escalated
    # agent_plan_validation_rollouts_after_flaky gate instead.
    flaky_capture_task_keys: Set[Any] = field(default_factory=set)
    # Task-evaluator reward of the rollout that produced the current
    # solved_plan capture (None when no evaluator verdict was computed).
    # The restart loop ranks best-effort captures across attempts by it.
    # Cleared together with solved_plan.
    solved_plan_eval_reward: Optional[float] = None
    # Restart-loop attempt bookkeeping, set by AgentModelBasedApproach._solve
    # around each attempt. ``attempt_start``/``attempt_deadline`` are
    # time.monotonic() values; the deadline is enforced cooperatively by
    # the probe (every sim call) and explore_python, and surfaced in tool
    # results as a budget footer. None ⇒ no attempt in flight / no wall
    # clock. The deadline is cleared before the final-submission nudge so
    # nothing blocks the submission itself.
    attempt_index: int = 0
    attempt_start: Optional[float] = None
    attempt_deadline: Optional[float] = None
    # Count of full-plan belief-sim rollouts this attempt (probe runs,
    # trials, capture-validation repeats). Reset per attempt; shown in
    # the budget footer so sweeps carry a visible price.
    attempt_rollout_count: int = 0
    # Best submission on the current task this attempt that
    # evaluate_option_plan evaluated but refused to capture (evaluator
    # scored it a non-solve, or it was flaky), ranked by evaluator
    # reward. Reset per attempt; the journal auto-entry records it so a
    # later attempt (or the final best-effort nudge) can resubmit it
    # instead of the attempt's work vanishing with its context.
    best_uncaptured_plan_lines: Optional[List[str]] = None
    best_uncaptured_reward: Optional[float] = None
    # Per-call deadline for the explore_python call currently executing
    # (agent_sdk_explore_python_call_timeout); enforced at the same
    # probe checkpoints as attempt_deadline. None ⇒ no call in flight.
    explore_call_deadline: Optional[float] = None

    def begin_attempt(self, index: int, wall_clock: float) -> None:
        """Start restart-loop bookkeeping for solve attempt ``index``.

        Resets everything scoped to a single attempt (rollout count,
        best refused submission) and arms the wall-clock deadline
        (``wall_clock <= 0`` ⇒ no deadline). The matching teardown stays
        in ``AgentModelBasedApproach._solve``'s finally block,
        interleaved with its journal write.
        """
        self.attempt_index = index
        self.attempt_rollout_count = 0
        self.best_uncaptured_plan_lines = None
        self.best_uncaptured_reward = None
        self.attempt_start = time.monotonic()
        self.attempt_deadline = (self.attempt_start +
                                 wall_clock if wall_clock > 0 else None)

    def clear_plan_capture(self) -> None:
        """Clear the four ``solved_plan*`` fields together.

        They form one value (see :class:`PlanCapture`); clearing any of
        them individually would leave a stale mix.
        """
        self.solved_plan = None
        self.solved_sketch = None
        self.solved_plan_reached_goal = None
        self.solved_plan_eval_reward = None

    def take_plan_capture(self) -> PlanCapture:
        """Pop the captured plan, clearing it so it cannot be reused.

        The returned capture's ``plan`` is falsy when nothing was
        captured since the last clear.
        """
        capture = PlanCapture(plan=self.solved_plan,
                              sketch=self.solved_sketch,
                              reached_goal=self.solved_plan_reached_goal,
                              eval_reward=self.solved_plan_eval_reward)
        self.clear_plan_capture()
        return capture


def _capture_task_key(ctx: ToolContext) -> Any:
    """Stable identity of the task behind a ``task_idx="current"`` capture.

    Keys ``ctx.flaky_capture_task_keys`` so a FLAKY rejection escalates
    the validation gate for later submissions on the SAME task only.
    Test-time solves are keyed by the test task index (stable across the
    sessions and replans of one task); exploration/synthesis captures
    fall back to the learning iteration, which at worst escalates
    conservatively across that cycle's tasks.
    """
    if ctx.test_task_idx is not None:
        return ("test", ctx.test_task_idx)
    return ("iter", ctx.iteration_id)


@contextmanager
def decorrelated_rollout_seed(rollout_idx: int) -> Iterator[None]:
    """Give one repeat rollout its own motion-planner seed.

    A fresh env per rollout is NOT sufficient for independent samples:
    every stochastic step of a rollout (BiRRT sampling, IK restarts)
    reads the constant ``CFG.seed`` at call time, so N fresh-env repeats
    of the same plan are bit-identical replays and "N/N reliable" is one
    effective sample. run_20260722_204632 (domino_high_friction_turn
    seed 2) captured a plan validated 13/13 that was a coin flip on the
    real episode; every trials-mode result in that run's sessions was
    0/N or N/N, never mixed. Offsetting the seed per rollout varies the
    planned motion paths within the tolerance the planner already
    accepts - the same execution variability that separates the
    validation context from the real episode - so repeats become a real
    sample of execution noise.

    ``rollout_idx`` 0 keeps the base seed: the canonical first rollout
    stays reproducible and single-run (``trials=1``) behavior is
    unchanged. Enter this scope AFTER any fresh env is created so env
    construction (and its task-cache key) still sees the base seed.
    """
    if rollout_idx == 0:
        yield
        return
    base_seed = CFG.seed
    CFG.seed = base_seed + rollout_idx
    try:
        yield
    finally:
        CFG.seed = base_seed
