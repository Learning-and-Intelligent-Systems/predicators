"""Agent model-free approach: fixed-vocabulary open-loop planning.

The agent plans directly from its own world knowledge - no simulator
(model) is required to validate a plan before execution. Combines online
trajectory collection (via AgentPlanExplorer) with open-loop option plan
generation (via Claude Agent SDK). No predicate/process/type invention -
just stores trajectories and generates plans.

Registered under the CLI approach name ``agent_planner`` (kept stable so
existing configs and logs remain valid).

Example command:
    python predicators/main.py --env pybullet_domino \
        --approach agent_planner --seed 0 \
        --num_train_tasks 1 --num_test_tasks 1 \
        --num_online_learning_cycles 1 --explorer agent_plan
"""
import copy
import datetime
import inspect as _inspect
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, \
    cast

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.rendering import save_task_state_image
from predicators.agent_sdk.session_base import AgentSessionFatalError
from predicators.agent_sdk.tools import agent_render_resolution, \
    explore_python_replaces_tools
from predicators.agent_sdk.tools.inspection import render_options_digest, \
    render_types_digest
from predicators.approaches import ApproachFailure
from predicators.approaches.agent_session_mixin import AgentSessionMixin
from predicators.approaches.base_approach import BaseApproach
from predicators.explorers import create_explorer
from predicators.explorers.base_explorer import BaseExplorer
from predicators.ground_truth_models import \
    augment_state_with_helper_objects, augment_task_with_helper_objects, \
    merge_gt_helper_predicates, merge_gt_helper_types
from predicators.option_model import _OptionModelBase, create_option_model
from predicators.settings import CFG
from predicators.structs import Action, Dataset, GroundAtom, \
    InteractionRequest, InteractionResult, LowLevelTrajectory, Object, \
    ParameterizedOption, ParameterizedSampler, Predicate, State, Task, Type


class AgentModelFreeApproach(AgentSessionMixin, BaseApproach):
    """Fixed-vocabulary open-loop planning via Claude Agent SDK.

    - Collects trajectories online using AgentPlanExplorer
    - At solve time, queries the agent for an option plan
    - No predicate/process/type invention
    """

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 *args: Any,
                 option_model: Optional[_OptionModelBase] = None,
                 **kwargs: Any) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, *args, **kwargs)
        # Optionally hand the agent the ground-truth helper scaffolding (e.g.
        # the domino/fan grid loc/side types and grid predicates) so an
        # "agent-with-grid" ablation plans over the oracle's vocabulary. Opt-in
        # via CFG.use_gt_helpers; no-op for envs without a helper factory or
        # when off. Merge here (before the agent session inits below) so the
        # session, solve-time abstraction, and _get_all_predicates see them.
        if self._use_gt_helpers():
            self._types = merge_gt_helper_types(self._types, CFG.env)
            self._initial_predicates = merge_gt_helper_predicates(
                self._initial_predicates, CFG.env)
        self._offline_dataset = Dataset([])
        self._online_trajectories: List[LowLevelTrajectory] = []
        self._option_model: Optional[_OptionModelBase] = (
            option_model if option_model is not None else
            self._create_planner_option_model())
        # Terminate Wait on atom change using the approach's predicates (which
        # may include invented ones), looked up lazily so the lambda picks up
        # predicates invented after __init__. When the grid ablation is on,
        # re-derive helper objects first so grid predicates (e.g. BallAtLoc)
        # stay evaluable on the otherwise helper-free execution states.
        if self._option_model is not None and \
                CFG.wait_option_terminate_on_atom_change:
            cast(  # pylint: disable=protected-access
                Any, self._option_model)._abstract_function = (
                    lambda s: utils.abstract(self._maybe_augment_state(s),
                                             self._get_all_predicates()))
        self._online_learning_cycle = 0
        # Synthesized per-skill samplers (option name -> sampler). Empty for
        # the base planner; learning subclasses populate it. Threaded into
        # bilevel refinement via _get_all_samplers() so continuous-parameter
        # search aims at each step's subgoal instead of drawing uniformly.
        self._synthesized_samplers: Dict[str, ParameterizedSampler] = {}
        self._requests_train_task_idxs: Optional[List[int]] = None
        self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._pre_test_conversation_log: Optional[List[Dict[str, Any]]] = None
        # True only between begin_test_phase / end_test_phase, so per-episode
        # hooks can act on test solves without touching exploration episodes.
        self._in_test_phase = False
        # 0-based index of the test task being solved, mirroring main.py's
        # ``test_task_idx``. Incremented per test solve; threaded into the
        # session-log filename via the ToolContext.
        self._test_task_idx = -1
        # Solve-journal snapshot taken at begin_test_phase (None = no
        # journal file existed) and whether it was captured successfully.
        # end_test_phase archives the full test-phase journal outside the
        # sandbox and rolls the file back to this snapshot, so learning
        # entries persist across cycles while one evaluation's test-task
        # entries never leak into the next evaluation.
        self._pre_test_journal: Optional[str] = None
        self._pre_test_journal_valid = False
        # Scene renders attempted this episode. The first is the true initial
        # state; later ones come from mid-episode replans and get distinct
        # filenames so they don't overwrite the init snapshot. Reset in
        # reset_for_new_episode.
        self._episode_scene_renders = 0
        # Filename of the most recently saved scene render, consumed by
        # _initial_image_section so the prompt references the image matching
        # the state the agent is actually planning from.
        self._last_scene_image_name: Optional[str] = None

        # Initializes _tool_context and _agent_session_id (see mixin). Use the
        # (possibly helper-augmented) vocabulary so the agent session exposes
        # the grid types/predicates when CFG.use_gt_helpers is on.
        self._init_agent_session_state(self._types, self._initial_predicates,
                                       initial_options, train_tasks)

        # Capture the underlying env once at construction. The initial option
        # model wraps ``env.simulate`` (a bound method), so ``__self__`` is the
        # env. Later cycles may rebuild ``_option_model`` with a plain learned
        # simulator that has no ``__self__``; pinning the env reference here
        # keeps scene rendering (the probe's sim.render) working in every
        # synthesis/solve cycle.
        env_self = getattr(getattr(self._option_model, '_simulator', None),
                           '__self__', None)
        if env_self is not None:
            self._tool_context.env = env_self

    @classmethod
    def get_name(cls) -> str:
        return "agent_planner"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_log_dir(self) -> str:
        """Return per-run log directory (created by configure_logging)."""
        log_dir = super()._get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        logging.info("Logging agent queries/responses to: %s", log_dir)
        return log_dir

    # ------------------------------------------------------------------ #
    # Overridable helpers (for subclass customisation)
    # ------------------------------------------------------------------ #

    def _use_gt_helpers(self) -> bool:
        """Whether to hand the agent the ground-truth helper scaffolding.

        Opt-in via ``CFG.use_gt_helpers`` (the process-planning
        approaches read it too). When on, the grid helper
        types/predicates are merged into the agent's vocabulary and the
        solved task is augmented with the grid objects + oracle goal
        (see ``__init__`` / ``_solve``).
        """
        return CFG.use_gt_helpers

    def _maybe_augment_state(self, state: State) -> State:
        """Re-derive GT helper objects on a state when the ablation is on.

        Executed states are helper-free (the grid is injected only into
        the planning task), so this keeps helper predicates evaluable
        during execution and Wait-on-atom-change termination. No-op when
        helpers are disabled or the env has no helper factory.
        """
        if self._use_gt_helpers():
            return augment_state_with_helper_objects(state, CFG.env)
        return state

    def _get_all_options(self) -> Set[ParameterizedOption]:
        """Return the full set of options available for planning."""
        return self._initial_options

    def _get_all_predicates(self) -> Set[Predicate]:
        """Return the full set of predicates for abstraction."""
        return self._initial_predicates

    def _get_all_samplers(self) -> Dict[str, ParameterizedSampler]:
        """Return synthesized per-skill samplers (option name -> sampler).

        Empty by default; learning subclasses populate the backing
        field. Threaded into bilevel refinement so parameter search aims
        at each step's subgoal.
        """
        return self._synthesized_samplers

    def _get_all_trajectories(self) -> List[LowLevelTrajectory]:
        """Return all trajectories (offline + online)."""
        return self._offline_dataset.trajectories + self._online_trajectories

    def _create_planner_option_model(self) -> Optional[_OptionModelBase]:
        """Build the option model the planner tests plans against.

        Honors two CFG knobs:

        * ``agent_planner_use_simulator`` -- when False, returns ``None``
          so the agent gets no ``evaluate_option_plan`` rollouts and must
          plan open-loop from data + LLM reasoning (the model-free
          baseline).
        * ``agent_planner_use_base_simulator`` -- when True (and a
          simulator is used), wraps the *base* env
          (``skip_residual_dynamics=True``), denying the planner the delayed
          ``_domain_specific_step`` dynamics; otherwise wraps the real env.
        """
        if not CFG.agent_planner_use_simulator:
            return None
        return create_option_model(
            CFG.option_model_name,
            skip_residual_dynamics=CFG.agent_planner_use_base_simulator)

    # ------------------------------------------------------------------ #
    # AgentSessionMixin hooks
    # ------------------------------------------------------------------ #

    # -- Prompt building blocks ----------------------------------------- #

    _SYSTEM_PROMPT_BASE = (
        "You are a planning agent. You observe task environments through "
        "inspection tools and generate option plans to achieve goals. "
        "You have access to read-only tools to inspect predicates, "
        "options, trajectories, and training tasks. Use these to "
        "understand the environment and generate effective plans.\n\n"
        "Some effects may not be immediate - if an action triggers a "
        "delayed process (e.g. water filling, dominoes cascading, "
        "heating), insert a Wait after it so the effect has time "
        "to occur before the next action. The Wait action holds the "
        "robot's current pose. You can annotate Wait with target atoms "
        "using `-> {atoms}` to specify exactly when it should terminate "
        "(e.g. `Wait(robot:Robot)[] -> {Boiled(water:water_type)}`). "
        "Use `NOT Pred(...)` for atoms that should become false. "
        "If no annotation is provided, the Wait terminates on any atom "
        "change. Without a Wait, the robot will proceed to the next "
        "action before the delayed effect has occurred, which might "
        "cause the plan to fail.")

    _SCRATCHPAD_SECTION = """
## Scratchpad - CRITICAL
You MUST maintain `./notes.md` as your working memory. \
**Read it at the very start of the session** and **read it \
again before every evaluate_option_plan call** to remind yourself \
what you already tried. **Update it immediately after every \
evaluate_option_plan call** - no exceptions.

Use this exact format for each option you are tuning:

```
## <OptionName> - Parameter Search
| # | params | outcome | notes |
|---|--------|---------|-------|
| 1 | [x, y, ...] | IK fail | ... |
| 2 | [x, y, ...] | success, JugNotAt... | ... |
```

After every test, append a row and update these summary fields:
- **Confirmed working params**: (list any that achieve the desired atoms)
- **Explored ranges**: e.g. "x: 0.9–1.05, y: 1.4–1.55" - look for GAPS
- **Unreachable region**: e.g. "y > 1.47 always IK-fails"
- **Next hypothesis**: what to try and why

The cycle is: Read notes → plan next experiment → run test → \
update notes → repeat. Without this loop you WILL forget what \
you tried and repeat the same failed parameters. Treat notes.md \
as your lab notebook - write after every single experiment.

**If you notice you have NOT updated notes after a test, STOP \
and update before doing anything else.**"""

    # -- System prompt --------------------------------------------------- #

    def _get_agent_system_prompt(self) -> str:
        use_scratchpad = CFG.agent_planner_use_scratchpad

        sections = [self._SYSTEM_PROMPT_BASE]

        # Scratchpad
        if use_scratchpad:
            sections.append(self._SCRATCHPAD_SECTION)

        # Tuning workflow (numbered steps, dynamic)
        steps = []
        if use_scratchpad:
            steps.append(
                "**Read `./notes.md` before every test**, then **update it "
                "immediately after every evaluate_option_plan call**. Record "
                "what you tried, what happened, and what you learned. "
                "This is your memory - without it you will repeat failures.")
        steps += [
            "**Review past session logs** in `./session_logs/` if available. "
            "Previous queries and tool results from earlier sessions are "
            "saved there. Read them to build on prior knowledge.",
            "**Inspect rendered images** from `./test_images/` when "
            "something goes wrong to understand the actual outcome.",
            "**Expect geometric offsets.** The target position for "
            "options is often offset from the reference object's reported "
            "position due to object geometry. Explore a wide range around "
            "the object's coordinates, not just values close to the "
            "reported position.",
            "**Search coarse-to-fine.** For each continuous parameter, "
            "start with a WIDE grid spanning most of the valid range "
            "(e.g. test 4–5 spread-out values across [low, high]). "
            "Identify which coarse region works, THEN refine within it. "
            "Never spend more than 3 attempts tweaking values in a small "
            "neighborhood - if none work, jump to a different region. "
            "Check your notes for gaps in the explored range.",
            "**Vary ALL params, not just position.** Orientation and "
            "other parameters change offsets and feasibility. If an "
            "action fails at a position, try different values for the "
            "other parameters before giving up on that region. Test at "
            "least 2-3 values for each non-position parameter.",
        ]
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        sections.append(
            f"\n## Continuous Parameter Tuning\nFollow this workflow:\n"
            f"{numbered}")

        return "\n".join(sections)

    def _get_sandbox_reference_files(self) -> Dict[str, str]:
        files: Dict[str, str] = {
            "skill_factories/base.py":
            "predicators/ground_truth_models/skill_factories/base.py",
            "skill_factories/__init__.py":
            "predicators/ground_truth_models/skill_factories/__init__.py",
            "skill_factories/pick.py":
            "predicators/ground_truth_models/skill_factories/pick.py",
            "skill_factories/move_to.py":
            "predicators/ground_truth_models/skill_factories/move_to.py",
            "skill_factories/place.py":
            "predicators/ground_truth_models/skill_factories/place.py",
            "skill_factories/push.py":
            "predicators/ground_truth_models/skill_factories/push.py",
            "skill_factories/pour.py":
            "predicators/ground_truth_models/skill_factories/pour.py",
            "skill_factories/wait.py":
            "predicators/ground_truth_models/skill_factories/wait.py",
        }
        options_path = _get_gt_options_module_path(CFG.env)
        if options_path:
            files["options.py"] = options_path
        return files

    def _get_solve_tool_names(self) -> Optional[List[str]]:
        # inspect_types / inspect_options are never offered: their
        # digests are static per session, so the solve prompt injects
        # them directly (same renderers - see _build_solve_prompt);
        # a zero-turn prompt section beats a one-turn tool call that
        # every fresh-context attempt would re-pay.
        tools = []
        # When the probe is present it subsumes the remaining inspect
        # tools too: `trajectories` / `describe_trajectory` in
        # explore_python's namespace and `sim.task()`. The extra
        # use_simulator guard keeps them for (hypothetical) sim-free
        # configs where explore_python itself is never offered below.
        probe_subsumes = (CFG.agent_planner_use_simulator
                          and explore_python_replaces_tools())
        if not probe_subsumes:
            tools += ["inspect_trajectories", "inspect_train_tasks"]
        # The remaining tools require a simulator: evaluate_option_plan
        # rolls fully-specified plans out through the option model.
        # None are offered when the planner has no simulator.
        # (refine_plan_sketch, which backtracking-refines a param-free sketch,
        # is exposed only by AgentModelBasedApproach.)
        if CFG.agent_planner_use_simulator:
            tools.append("evaluate_option_plan")
            if CFG.agent_planner_use_explore_python:
                tools.append("explore_python")
        if CFG.agent_solve_use_journal:
            tools.append("record_journal")
        return tools

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        self._offline_dataset = dataset
        self._tool_context.offline_trajectories = dataset.trajectories
        if dataset.trajectories:
            self._tool_context.example_state = \
                dataset.trajectories[0].states[0]

    def get_interaction_requests(self) -> List[InteractionRequest]:
        explorer = self._create_explorer()
        requests: List[InteractionRequest] = []
        self._requests_train_task_idxs = []
        # A cycle's requests are all generated before any executes, so the
        # explorer shows each query the plans already scheduled this cycle and
        # asks for a complementary one. Fresh list per cycle.
        self._tool_context.cycle_scheduled_plans = []
        for _ in range(CFG.online_nsrt_learning_requests_per_cycle):
            task_idx = self._rng.choice(len(self._train_tasks))
            # Clear so a planning explorer's verdict is read fresh per request;
            # non-planning explorers leave it None (no verdict).
            self._tool_context.last_mental_model_solved = None
            policy, termination_function = explorer.get_exploration_strategy(
                task_idx, CFG.timeout)
            req = InteractionRequest(train_task_idx=task_idx,
                                     act_policy=policy,
                                     query_policy=lambda s: None,
                                     termination_function=termination_function,
                                     mental_model_solved=self._tool_context.
                                     last_mental_model_solved)
            requests.append(req)
            self._requests_train_task_idxs.append(task_idx)
        return requests

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        assert self._requests_train_task_idxs is not None
        # Subclasses (e.g. AgentSimLearningApproach) may track the snapshot
        # tags of the simulator/predicates files in effect when the explorer
        # generated these plans. Tag each new trajectory so the next
        # learn-phase prompt can surface provenance. ``None`` for approaches
        # that don't track versions.
        sim_version: Optional[str] = getattr(self,
                                             "_current_simulator_version",
                                             None)
        preds_version: Optional[str] = getattr(self,
                                               "_current_predicates_version",
                                               None)
        samplers_version: Optional[str] = getattr(self,
                                                  "_current_samplers_version",
                                                  None)
        for i, result in enumerate(results):
            task_idx = self._requests_train_task_idxs[i]
            traj = LowLevelTrajectory(
                result.states,
                result.actions,
                _train_task_idx=task_idx,
                _source_simulator_version=sim_version,
                _source_predicates_version=preds_version,
                _source_samplers_version=samplers_version,
                _env_reward=result.episode_reward,
                _env_terminated=result.episode_terminated,
            )
            self._online_trajectories.append(traj)

        # Update tool context
        self._sync_tool_context()

        logging.info(
            "[Run %s] Cycle %s: collected %d trajectories, %d total online.",
            self._run_id, self._online_learning_cycle, len(results),
            len(self._online_trajectories))

        self.save(self._online_learning_cycle)
        self._online_learning_cycle += 1

    # ------------------------------------------------------------------ #
    # Solving
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wrap_option_failures(
            policy: Callable[[State], Action]) -> Callable[[State], Action]:
        """Wrap a policy so OptionExecutionFailure surfaces as ApproachFailure.

        Bilevel planning and the base open-loop planner both build a
        low-level policy from a grounded option plan; this adapter is
        their single place to translate the harness's option-execution
        exception into the ApproachFailure CogMan expects.
        """

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._sync_tool_context()
        # When enabled, plan over the oracle's grid-augmented task: inject the
        # grid loc/side objects and rewrite the goal to the grid BallAtLoc so
        # the agent sees the oracle's scaffolding. Augmentation preserves
        # goal_nl. No-op otherwise.
        if self._use_gt_helpers():
            task = augment_task_with_helper_objects(task, CFG.env)
        self._tool_context.current_task = task
        # Render the initial state so the agent can see the scene layout.
        self._render_initial_state_image(task)
        try:
            option_plan = self._query_agent_for_option_plan(task)
        except AgentSessionFatalError:
            # An ApproachFailure would be absorbed per-task; the broken
            # session backend must terminate the run instead.
            raise
        except Exception as e:
            raise ApproachFailure(f"Agent failed to produce option plan: {e}")

        preds = self._get_all_predicates()
        policy = utils.option_plan_to_policy(
            option_plan,
            max_option_steps=CFG.max_num_steps_option_rollout,
            abstract_function=lambda s: utils.abstract(
                self._maybe_augment_state(s), preds))

        return self._wrap_option_failures(policy)

    def _render_initial_state_image(self, task: Task) -> Optional[str]:
        """Render the state this solve starts from and save to the sandbox.

        The first render of an episode is the true initial state
        (``task{N:03d}_initial_state.png``); later renders come from
        mid-episode replans and are saved as
        ``task{N:03d}_replan{K}_state.png`` so they don't overwrite the
        init snapshot (the replan "task" is rooted at the current,
        partially-executed state).

        Returns the saved image path, or None if rendering is unavailable.
        """
        self._last_scene_image_name = None
        env = self._tool_context.env
        if env is None:
            return None
        try:
            # The session/sandbox (and thus ``image_save_dir`` on the
            # ToolContext) is created lazily on the first agent query. This
            # render runs *before* that query in ``_solve``, so on the very
            # first test task the dir would still be None and task0's image
            # would be silently skipped; ensure the session (and dir) exist
            # first. Inside the try so a session-creation hiccup leaves
            # rendering best-effort rather than crashing the solve.
            self._ensure_agent_session()
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Failed to render initial state image: %s", e)
            return None
        save_dir = self._tool_context.image_save_dir
        if save_dir is None:
            return None
        task_id = self._tool_context.test_task_idx
        replan_idx = self._episode_scene_renders
        # Count attempts, not successes: if the init render fails, a later
        # replan render still must not masquerade as the init image.
        self._episode_scene_renders += 1
        if task_id is not None:
            stem = f"task{task_id:03d}"
        else:
            stem = ""
        if replan_idx == 0:
            filename = f"{stem}_initial_state.png" if stem \
                else "initial_state.png"
        else:
            filename = f"{stem}_replan{replan_idx}_state.png" if stem \
                else f"replan{replan_idx}_state.png"
        with agent_render_resolution():
            saved_path = save_task_state_image(env, task, save_dir, filename)
        if saved_path is not None:
            self._last_scene_image_name = filename
        return saved_path

    def _initial_image_section(self) -> str:
        """Return a prompt section pointing at the current solve's rendered
        scene image, or an empty string if none was rendered.

        ``_render_initial_state_image`` must have been called first;
        this references whichever file that call saved (init or replan
        snapshot), so replan queries point at the current scene rather
        than the stale episode-init image.
        """
        save_dir = self._tool_context.image_save_dir
        img_name = self._last_scene_image_name
        if not save_dir or img_name is None:
            return ""
        if not os.path.exists(os.path.join(save_dir, img_name)):
            return ""
        # cwd of the agent is the sandbox root, so reference test_images/.
        return ("\n## Initial State Image\n"
                "A rendering of the scene this plan starts from has been "
                f"saved to `./test_images/{img_name}`. **Read this image "
                "first** to understand the spatial layout before "
                "planning.\n")

    # ------------------------------------------------------------------ #
    # Test phase lifecycle
    # ------------------------------------------------------------------ #

    def begin_test_phase(self) -> None:
        """Snapshot the learning conversation log and solve journal."""
        self._in_test_phase = True
        self._test_task_idx = -1
        if self._agent_session is not None:
            self._pre_test_conversation_log = copy.deepcopy(
                self._agent_session.conversation_log)
        else:
            self._pre_test_conversation_log = None
        self._snapshot_journal_for_test_phase()

    def end_test_phase(self) -> None:
        """Restore the conversation log and journal to pre-test state."""
        self._in_test_phase = False
        self._tool_context.test_task_idx = None
        if self._agent_session is not None \
                and self._pre_test_conversation_log is not None:
            # In-place restore through the public property (it returns
            # the live list), so any other holder of the reference sees
            # the rollback too.
            log = self._agent_session.conversation_log
            log[:] = self._pre_test_conversation_log
        self._pre_test_conversation_log = None
        self._archive_and_rollback_test_journal()

    def _journal_active(self) -> bool:
        """Whether solve-journal entries can be written at all."""
        return bool(CFG.agent_solve_use_journal
                    and self._tool_context.sandbox_dir)

    def _snapshot_journal_for_test_phase(self) -> None:
        """Capture the learning-only journal content at test-phase entry.

        The snapshot is what ``end_test_phase`` rolls the journal back
        to. A failed capture leaves ``_pre_test_journal_valid`` False so
        the rollback is skipped rather than destroying learning entries.
        """
        self._pre_test_journal = None
        self._pre_test_journal_valid = False
        if not self._journal_active():
            return
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk import journal as journal_mod
        try:
            self._pre_test_journal = journal_mod.read_raw(
                self._tool_context.sandbox_dir)
            self._pre_test_journal_valid = True
        except OSError as e:
            logging.warning(
                "[%s] Failed to snapshot the solve journal at test-phase "
                "start; test-phase entries will NOT be rolled back: %s",
                self._run_id, e)

    def _archive_and_rollback_test_journal(self) -> None:
        """Archive the test-phase journal, then roll it back.

        Each evaluation must be independent of previous evaluations:
        entries recorded while solving test tasks (harness auto-entries
        and agent notes) would otherwise leak this evaluation's test
        tasks into the next one. Learning entries - the pre-test
        snapshot - persist across cycles. Before the rollback, the full
        journal (learning + this evaluation's additions) is copied to
        the run's log dir, which lives outside the sandbox so the agent
        cannot read it, for later inspection.
        """
        if not self._pre_test_journal_valid:
            return
        snapshot = self._pre_test_journal
        self._pre_test_journal = None
        self._pre_test_journal_valid = False
        sandbox_dir = self._tool_context.sandbox_dir
        if not self._journal_active():
            return
        assert sandbox_dir is not None
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk import journal as journal_mod
        try:
            content = journal_mod.read_raw(sandbox_dir)
            if content is not None:
                # One archive per online-learning cycle; the cycle counter
                # advances with each learning phase, so evaluations map to
                # distinct files (a same-cycle re-eval overwrites its own).
                archive_path = os.path.join(
                    self._get_log_dir(),
                    f"journal_eval_cycle{self._online_learning_cycle}.md")
                with open(archive_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logging.info(
                    "[%s] Archived the test-phase solve journal to %s",
                    self._run_id, archive_path)
            journal_mod.restore(sandbox_dir, snapshot)
        except OSError as e:
            logging.warning(
                "[%s] Failed to archive/roll back the test-phase solve "
                "journal: %s", self._run_id, e)

    def reset_for_new_episode(self) -> None:
        """Advance the test-task counter at each test episode start.

        CogMan calls this exactly once per test task (via
        ``cogman.reset`` in main.py's ``_solve_task``) and never on a
        replan inside an episode, so the counter stays in lockstep with
        main.py's ``test_task_idx``. The index reaches the sandbox via
        the ToolContext and lands in the session-log filename. No-op
        outside the test phase.
        """
        super().reset_for_new_episode()
        # New episode -> the next scene render is a true init snapshot.
        self._episode_scene_renders = 0
        if self._in_test_phase:
            self._test_task_idx += 1
            self._tool_context.test_task_idx = self._test_task_idx

    def _query_agent_for_option_plan(self, task: Task) -> list:
        """Query the agent for an option plan and parse it."""
        prompt = self._build_solve_prompt(task)
        responses = self._query_agent_sync(prompt, kind="test")
        plan_text = self._extract_option_plan_text(responses)

        if not plan_text:
            # Log the raw responses for debugging.
            n_responses = len(responses)
            types = [r.get("type") for r in responses]
            raise ApproachFailure(
                f"Agent returned empty plan text. "
                f"Got {n_responses} responses with types: {types}")

        return self._parse_and_ground_plan(plan_text, task)

    def _solve_prompt_visualize_line(self) -> str:
        """The stuck-step visualization bullet: the probe's staging + render is
        the only visualization surface, so the bullet appears only when
        explore_python is offered."""
        if CFG.agent_planner_use_simulator and \
                CFG.agent_planner_use_explore_python:
            return (
                "- **Use explore_python when stuck** - after 3+ failures on "
                "the same step, STOP testing and use explore_python "
                "(`sim.reset(mods={...})`, then `sim.render(...)`) to move "
                "the object to several candidate positions and "
                "orientations. It's free (no physics). Find the right "
                "region visually, then test.\n")
        return ""

    def _solve_prompt_scratchpad_line(self) -> str:
        """Return the notes.md bullet for the solve prompt, or empty."""
        if CFG.agent_planner_use_scratchpad:
            return (
                "- **Read `./notes.md` before every "
                "evaluate_option_plan call** "
                "and **update it immediately after each call** - append a "
                "row to the parameter table and update the explored-ranges "
                "summary. If you realize you forgot to update, STOP and "
                "update before doing anything else.\n")
        return ""

    def _build_solve_prompt(self, task: Task) -> str:
        """Build the prompt for generating an option plan."""
        init_state = task.init
        objects = list(init_state)

        # Objects
        obj_strs = []
        for obj in sorted(objects, key=lambda o: o.name):
            obj_strs.append(f"  {obj.name}: {obj.type.name}")

        # Goal. Only expose goal atoms whose predicate is in the agent's
        # current predicate set (same filter as the bilevel sketch
        # prompt): approaches that strip env predicates rely on goal_nl
        # to communicate the goal.
        visible_preds = self._get_all_predicates()
        goal_strs = [
            str(a) for a in sorted(task.goal, key=str)
            if a.predicate in visible_preds
        ]

        # Types and options: the same digests the inspect_types /
        # inspect_options tools would serve, injected here so those
        # tools need not be offered (see _get_solve_tool_names).
        types_digest = render_types_digest(self._tool_context.types)
        options_digest = render_options_digest(
            self._get_all_options(),
            gt_options_ref_path=self._tool_context.gt_options_ref_path)

        # Current atoms
        atoms = utils.abstract(init_state, self._get_all_predicates())
        atom_strs = [str(a) for a in sorted(atoms, key=str)]

        # Trajectory summary
        traj_summary = self._build_trajectory_summary()

        # State features (compact)
        state_str = init_state.dict_str(indent=2)

        # Available tools
        tool_names = self._get_solve_tool_names()
        tools_str = ""
        if tool_names:
            tool_list = "\n".join(f"  - {t}" for t in tool_names)
            tools_str = f"\n## Available Tools\n{tool_list}\n"

        # Natural language goal description (if available)
        goal_nl_section = ""
        if task.goal_nl:
            goal_nl_section = f"""
## Goal Description
{task.goal_nl}
"""

        # Initial state image reference
        initial_image_section = self._initial_image_section()

        if CFG.agent_planner_use_simulator:
            instructions_intro = (
                "Use your available tools to inspect the environment and "
                "test your plan before committing to it.")
        else:
            instructions_intro = (
                "You do NOT have a simulator to test plans against. Inspect "
                "the trajectory data and reason carefully about the dynamics, "
                "then commit to your best open-loop plan.")

        prompt = f"""You are solving a task. \
Generate an option plan to achieve the goal.
{goal_nl_section}
## Goal Atoms
{chr(10).join(goal_strs)}

## Initial State Atoms
{chr(10).join(atom_strs)}

## Initial State Features
{state_str}
{initial_image_section}
## Objects
{chr(10).join(obj_strs)}

## Object Types
{types_digest}

## Available Options
{options_digest}
{traj_summary}{tools_str}
## Instructions
{instructions_intro}

Based on the task information and any past trajectory data, output an option plan to achieve the goal.

After any action whose desired subgoal depends on a delayed process (e.g. water \
filling, dominoes cascading, heating), insert a Wait action to let the process \
complete before proceeding. You can annotate Wait with target atoms using \
`-> {{atoms}}` to specify exactly when it should terminate. Use `NOT Pred(...)` for \
atoms that should become false. If no annotation is provided, the Wait terminates on \
any atom change. Only use Wait when there is a genuine delayed effect; do not insert \
it between actions with immediate effects (e.g. Pick, Place).

For Wait with target atoms: `Wait(robot:Robot)[] -> {{Boiled(water:water_type)}}`
For negated targets: `Wait(robot:Robot)[] -> {{NOT Touching(a:block, b:block)}}`

**Important - parameter tuning workflow:**
- When a step fails or produces unexpected results, inspect the rendered images \
in `./test_images/` to see what actually happened in the scene.
{self._solve_prompt_scratchpad_line()}\
- Review past session logs in `./session_logs/` if available - they contain prior queries and results.
- When a step fails (e.g. IK error), use the image + object poses to reason about \
WHY and adjust params directionally. Don't just try random nearby values.
{self._solve_prompt_visualize_line()}\
- **Vary all parameters, not just position** - orientation and other params affect \
both the outcome and whether the action succeeds. Try 2-3 values for each \
non-position parameter per target region.
- **Search coarse-to-fine**: spread initial attempts across the full parameter range. \
If 3 nearby values all fail the same way, jump to a very different region instead of \
continuing to tweak. Check your notes for gaps in explored ranges.

Output the plan with one option per line in this exact format:
 OptionName(obj1:type1, obj2:type2)[param1, param2]

If an option has no continuous parameters, use empty brackets: OptionName(obj1:type1)[]

Output ONLY the option plan lines at the end, after any analysis."""

        return prompt

    def _build_trajectory_summary(self) -> str:
        """Summarize trajectory data for context."""
        all_trajs = self._get_all_trajectories()
        if not all_trajs:
            return ""

        max_trajs = CFG.agent_sdk_max_trajectories_in_context
        recent = all_trajs[-max_trajs:]
        all_preds = self._get_all_predicates()
        lines = [
            f"\n## Trajectory Summary ({len(all_trajs)} total, "
            f"showing last {len(recent)})"
        ]

        for i, traj in enumerate(recent):
            n_steps = len(traj.actions)
            init_atoms = utils.abstract(traj.states[0], all_preds)
            final_atoms = utils.abstract(traj.states[-1], all_preds)
            new_atoms = final_atoms - init_atoms
            lost_atoms = init_atoms - final_atoms
            lines.append(f"\nTrajectory {i}: {n_steps} steps")
            if new_atoms:
                lines.append(
                    f"  Gained: "
                    f"{', '.join(str(a) for a in sorted(new_atoms, key=str))}")
            if lost_atoms:
                lines.append(
                    f"  Lost: "
                    f"{', '.join(str(a) for a in sorted(lost_atoms, key=str))}"
                )

        return "\n".join(lines)

    def _extract_option_plan_text(self, responses: List[Dict[str,
                                                             Any]]) -> str:
        """Extract plan text from the last assistant text response.

        Only uses the final assistant message to avoid including
        intermediate reasoning/tool-call text that precedes the actual
        option plan.
        """
        last_text_parts: List[str] = []
        for resp in responses:
            if resp.get("type") == "assistant":
                parts = [
                    block.get("text", "") for block in resp.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if parts:
                    last_text_parts = parts
        return "\n".join(last_text_parts)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip markdown code fences wrapping the plan text."""
        lines = text.split('\n')
        # Remove leading/trailing ``` lines (with optional language tag).
        while lines and lines[0].strip().startswith('```'):
            lines.pop(0)
        while lines and lines[-1].strip().startswith('```'):
            lines.pop()
        return '\n'.join(lines)

    def _parse_wait_annotations(
        self,
        text: str,
        predicates: Set[Predicate],
        objects: Sequence[Object],
    ) -> List[Tuple[Set[GroundAtom], Set[GroundAtom]]]:
        """Parse ``-> {atoms}`` annotations from plan lines.

        Returns a list parallel to the option lines in the text. Each
        entry is ``(positive_atoms, negative_atoms)`` for Wait lines
        with annotations, or ``(set(), set())`` otherwise.
        """
        results: List[Tuple[Set[GroundAtom], Set[GroundAtom]]] = []
        option_names = {o.name for o in self._get_all_options()}
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            first_token = stripped.split('(')[0]
            if first_token not in option_names:
                if results:
                    break
                continue
            if first_token == "Wait" and '->' in stripped:
                pos, neg = utils.parse_wait_target_annotations(
                    stripped, predicates, objects)
                results.append((pos, neg))
            else:
                results.append((set(), set()))
        return results

    def _parse_and_ground_plan(self, plan_text: str, task: Task) -> list:
        """Parse option plan text and ground into executable options."""
        objects = list(task.init)
        all_options = self._get_all_options()
        option_names = sorted(o.name for o in all_options)

        # Strip markdown code fences that agents often wrap plans in.
        cleaned_text = self._strip_code_fences(plan_text)

        # Extract Wait target annotations before stripping them.
        wait_annotations = self._parse_wait_annotations(
            cleaned_text, self._get_all_predicates(), objects)

        # Strip annotations so the option plan parser doesn't choke.
        parseable_text = utils.strip_wait_annotations(cleaned_text)

        parsed = utils.parse_model_output_into_option_plan(
            parseable_text,
            objects,
            self._types,
            all_options,
            parse_continuous_params=True)
        if not parsed:
            raise ApproachFailure(f"Parsed empty option plan from agent.\n"
                                  f"  Plan text:\n{plan_text}\n"
                                  f"  Available option names: {option_names}")

        grounded = []
        for i, (option, objs, params) in enumerate(parsed):
            try:
                params_arr = np.array(params, dtype=np.float32)
                ground_opt = option.ground(objs, params_arr)
                # Inject Wait target atoms from annotations.
                if (ground_opt.name == "Wait" and i < len(wait_annotations)):
                    pos, neg = wait_annotations[i]
                    if pos:
                        ground_opt.memory["wait_target_atoms"] = pos
                    if neg:
                        ground_opt.memory["wait_target_neg_atoms"] = neg
                grounded.append(ground_opt)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("[Run %s] Failed to ground option "
                                "%s: %s", self._run_id, option.name, e)
                break

        if not grounded:
            raise ApproachFailure("No options successfully grounded.")
        logging.info("[Run %s] Agent produced plan with %d options.",
                     self._run_id, len(grounded))
        return grounded

    # ------------------------------------------------------------------ #
    # Explorer
    # ------------------------------------------------------------------ #

    def _create_explorer(self) -> BaseExplorer:
        """Create explorer for interaction requests."""
        if CFG.explorer in ("agent_plan", "agent_bilevel"):
            self._sync_tool_context()
            return self._create_agent_explorer(
                self._get_all_predicates(),
                self._get_all_options(),
                name=CFG.explorer,
            )
        return create_explorer(
            CFG.explorer,
            self._get_all_predicates(),
            self._get_all_options(),
            self._types,
            self._action_space,
            self._train_tasks,
        )

    def _sync_tool_context(self) -> None:
        """Push current approach state into the shared ToolContext.

        The MCP tools (inspect_options, evaluate_option_plan, etc.) read
        from the ToolContext dataclass, not the approach directly. This
        keeps them in sync after mutations (e.g. new trajectories
        collected, options added). Called before each solve and learning
        interaction. Subclasses should call super() and then set
        additional fields (e.g. skill_factory_context).
        """
        self._tool_context.types = self._types
        # The agent's predicate vocabulary, not the raw env set: tools
        # abstract states, list predicates, and parse plan annotations
        # from this, so stripped predicates (agent_sim_learning
        # allowlist) must not leak in and invented ones must appear.
        self._tool_context.predicates = self._get_all_predicates()
        self._tool_context.options = self._initial_options
        self._tool_context.show_option_source = True
        ref_root = "/sandbox" if CFG.agent_sdk_use_docker_sandbox else "."
        self._tool_context.gt_options_ref_path = \
            f"{ref_root}/reference/options.py"
        self._tool_context.train_tasks = self._train_tasks
        self._tool_context.offline_trajectories = \
            self._offline_dataset.trajectories
        self._tool_context.online_trajectories = self._online_trajectories
        self._tool_context.log_dir = self._get_log_dir()
        self._tool_context.option_model = self._option_model
        # Synthesized samplers, so the explorer and synthesis tools thread the
        # same per-skill samplers into refinement that the approach uses.
        self._tool_context.parameterized_samplers = self._get_all_samplers()
        # Wire the active-experiment info-gain scorer when a learning subclass
        # exposes one and info-seeking exploration is on. Syncing the bound
        # method (not a snapshot) keeps it pointed at the latest fit/ensemble.
        # getattr guard: non-learning approaches lack it.
        if CFG.agent_explorer_info_seeking:
            self._tool_context.atom_disagreement_fn = getattr(
                self, "score_atom_disagreement", None)
        else:
            self._tool_context.atom_disagreement_fn = None
        all_trajs = (self._offline_dataset.trajectories +
                     self._online_trajectories)
        if all_trajs:
            self._tool_context.example_state = all_trajs[0].states[0]

        # Refresh env from the option model only if extraction succeeds. After
        # sim learning, ``_simulator`` may be a plain lambda with no
        # ``__self__``; don't clobber the env reference seeded in ``__init__``
        # in that case.
        if self._option_model is not None and \
                hasattr(self._option_model, '_simulator'):
            env_self = getattr(
                self._option_model._simulator,  # pylint: disable=protected-access
                '__self__',
                None)
            if env_self is not None:
                self._tool_context.env = env_self

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    # Filename suffix for the pickled approach state. Subclasses that
    # persist extra fields override this so their saves don't collide
    # with the base planner's.
    _save_suffix: str = "AgentPlanner"

    def _extra_save_state(self) -> Dict[str, Any]:
        """Subclass hook: extra (key -> value) pairs to persist.

        Merged into the base save dict; restored by the matching
        :meth:`_load_extra_save_state`.
        """
        return {}

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        """Subclass hook: restore fields written by _extra_save_state.

        Called after the base fields are restored and ``_run_id`` has
        been refreshed, but before the tool context is re-synced.
        """

    def save(self, online_learning_cycle: Optional[int] = None) -> None:
        """Save approach state to disk."""
        save_path = utils.get_approach_save_path_str()
        path = f"{save_path}_{online_learning_cycle}.{self._save_suffix}"
        save_dict = {
            "offline_dataset":
            self._offline_dataset,
            "online_trajectories":
            self._online_trajectories,
            "online_learning_cycle":
            self._online_learning_cycle,
            "run_id":
            self._run_id,
            "agent_session_id":
            (self._agent_session.session_id if self._agent_session else None),
            **self._extra_save_state(),
        }
        with open(path, "wb") as f:
            pkl.dump(save_dict, f)
        logging.info("[Run %s] Saved approach to %s", self._run_id, path)

    def load(self, online_learning_cycle: Optional[int] = None) -> None:
        save_path = utils.get_approach_load_path_str()
        path = f"{save_path}_{online_learning_cycle}.{self._save_suffix}"
        with open(path, "rb") as f:
            save_dict = pkl.load(f)

        self._offline_dataset = save_dict["offline_dataset"]
        self._online_trajectories = save_dict["online_trajectories"]
        self._online_learning_cycle = save_dict["online_learning_cycle"] + 1
        # pylint: disable=attribute-defined-outside-init
        # (_agent_session_id is initialized via the agent-session mixin.)
        self._agent_session_id = save_dict.get("agent_session_id")

        # New run_id for continued execution (each run gets its own dir), but
        # log the original run_id for reference.
        original_run_id = save_dict.get("run_id", "unknown")
        self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self._load_extra_save_state(save_dict)

        # Re-sync tool context (subclass fields are restored first).
        self._sync_tool_context()

        logging.info(
            "[Run %s] Loaded from previous run %s: %d offline, %d online "
            "trajectories", self._run_id, original_run_id,
            len(self._offline_dataset.trajectories),
            len(self._online_trajectories))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _get_gt_options_module_path(env_name: str) -> Optional[str]:
    """Return repo-relative path to the options.py for the given env.

    Looks up the GroundTruthOptionFactory subclass that handles
    *env_name* and returns the path to its module file, relative to
    the repository root (e.g.
    ``predicators/ground_truth_models/boil/options.py``).
    """
    # Importing ground_truth_models triggers import_submodules, which
    # registers all factory subclasses.
    from predicators.ground_truth_models import \
        GroundTruthOptionFactory  # pylint: disable=import-outside-toplevel
    for cls in utils.get_all_subclasses(GroundTruthOptionFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            module = _inspect.getmodule(cls)
            if module and module.__name__:
                return module.__name__.replace(".", os.sep) + ".py"
    return None
