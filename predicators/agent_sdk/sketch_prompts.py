"""Prompt construction for bilevel plan-sketch solving/exploration.

Split out of ``bilevel_sketch`` (see that module's docstring for the
full layout); holds ``build_solve_prompt``, the solve/explore prompt
asking the agent for a plan sketch in the grammar that
``sketch_parsing`` reads back.
"""
from typing import Optional, Sequence, Set

from predicators import utils
from predicators.structs import ParameterizedOption, Predicate, Task


def build_solve_prompt(
    task: Task,
    *,
    all_predicates: Set[Predicate],
    all_options: Set[ParameterizedOption],
    trajectory_summary: str = "",
    tool_names: Optional[Sequence[str]] = None,
    experiment_guidance: str = "",
    scheduled_plans: Optional[Sequence[str]] = None,
    initial_image_section: str = "",
    propose_params: bool = False,
    require_tool_validation: bool = False,
    explore_mode: bool = False,
    ground_samplers: bool = False,
    journal: str = "",
    physics_margin: bool = False,
) -> str:
    """Build the bilevel solve/explore prompt asking for a plan sketch.

    ``ground_samplers`` is the caller-threaded value of
    ``RefinementConfig.ground_samplers``; when False the prompt never
    mentions the ``~`` annotation channel.

    Mirrors ``AgentModelBasedApproach._build_solve_prompt`` but takes
    dependencies explicitly so explorers can reuse it.

    ``scheduled_plans`` lists sketch-line descriptions of exploration
    plans already generated this online-learning cycle (all of a cycle's
    requests are generated before any executes). When given, the prompt
    asks for a plan that still achieves the goal but differs meaningfully,
    so the cycle's interaction data is complementary instead of the same
    plan repeated per request.

    ``propose_params`` switches the prompt from "param-free sketch, search
    finds all continuous params" to "propose your best continuous params in
    ``[...]`` per step; the search refines them and samples on failure".

    ``require_tool_validation`` tells the agent it MUST submit a
    goal-reaching ``evaluate_option_plan`` run on the current task (the
    captured, validated plan is the only output) - used when the approach
    has no refinement fallback. When False, validation is merely
    encouraged.

    ``explore_mode`` marks the query as an exploration request whose
    sketch will run in the REAL environment as an experiment. It adds a
    belief-model disclosure (the simulator is base physics plus dynamics
    learned so far, so an unlearned mechanism shows zero effect) and an
    explicit delivery contract: a simulator-failing sketch is a valid
    deliverable when the goal depends on a mechanism the belief model
    lacks. Without it, agents have burned entire sessions exhaustively
    proving such a mechanism's absence instead of submitting the
    experiment that would let it be learned.

    ``journal`` is the run's solve-journal content (see
    ``predicators/agent_sdk/journal.py``): the curated record of earlier
    attempts' outcomes and lessons, injected so fresh-context sessions
    inherit what worked (and what was already swept) without inheriting
    failed attempts' conclusions.

    ``physics_margin`` is the caller-threaded value of
    ``CFG.agent_plan_validation_physics_margin``: when True (and
    ``require_tool_validation``), the submit guidance tells the agent
    the capture gate also sweeps the identified physical parameters'
    uncertainty range, and to pre-check designs with
    ``sim.run(..., physics_sweep=True)`` instead of discovering
    PARAM-SENSITIVE rejections one submission at a time.
    """
    assert not (explore_mode and require_tool_validation), (
        "explore_mode accepts an uncaptured experiment sketch, which "
        "contradicts the hard capture gate of require_tool_validation")

    init_state = task.init
    objects = list(init_state)

    obj_strs = []
    for obj in sorted(objects, key=lambda o: o.name):
        obj_strs.append(f"  {obj.name}: {obj.type.name}")

    # Only expose goal atoms whose predicate is in the agent's current
    # predicate set. Approaches that strip env predicates (e.g.
    # agent_sim_predicate_invention) rely on goal_nl to communicate the
    # goal; leaking unfiltered task.goal atoms would expose predicates the
    # agent is supposed to invent for itself.
    goal_strs = [
        str(a) for a in sorted(task.goal, key=str)
        if a.predicate in all_predicates
    ]

    option_strs = []
    for opt in sorted(all_options, key=lambda o: o.name):
        type_sig = ", ".join(t.name for t in opt.types)
        params_dim = opt.params_space.shape[0]
        if params_dim > 0:
            low = opt.params_space.low.tolist()
            high = opt.params_space.high.tolist()
            label = "params" if propose_params else "auto-searched params"
            if opt.params_description:
                desc = ", ".join(opt.params_description)
                param_info = (f"  [{label}: {desc}, "
                              f"range {low} to {high}]")
            else:
                kind = f"{params_dim}d"
                param_info = (f"  [{label}: {kind}, "
                              f"range {low} to {high}]")
        else:
            param_info = ""
        option_strs.append(f"  {opt.name}({type_sig}){param_info}")

    atoms = utils.abstract(init_state, all_predicates)
    atom_strs = [str(a) for a in sorted(atoms, key=str)]

    state_str = init_state.dict_str(indent=2)

    tools_str = ""
    if tool_names:
        tool_list = "\n".join(f"  - {t}" for t in tool_names)
        tools_str = f"\n## Available Tools\n{tool_list}\n"

    experiment_section = ""
    if experiment_guidance:
        experiment_section = (f"\n## Experiment Guidance\n"
                              f"{experiment_guidance}\n")

    belief_model_section = ""
    if explore_mode:
        belief_model_section = (
            "\n## Belief-Model Simulator\n"
            "The simulator behind your tools is the current BELIEF MODEL: "
            "known base physics plus whatever additional dynamics have been "
            "learned from real interaction data so far. A mechanism that "
            "has not been learned yet is simply ABSENT from it - the "
            "simulator shows zero effect for that mechanism no matter how "
            "you arrange the probe, and early in learning this can include "
            "the very mechanism the goal depends on. Treat a null effect "
            "after a few well-aimed probes as \"not in the belief model "
            "yet\", NOT as evidence about the real environment, and do not "
            "spend the session exhaustively confirming the absence.\n")

    scheduled_plans_section = ""
    if scheduled_plans:
        plan_blocks = "\n".join(f"Plan {i + 1}:\n{p}"
                                for i, p in enumerate(scheduled_plans))
        scheduled_plans_section = (
            "\n## Plans Already Scheduled This Cycle\n"
            "The plan(s) below are already queued to run on this same task "
            "before any learning happens, so their interaction data will be "
            "collected regardless of what you propose now.\n"
            f"{plan_blocks}\n"
            "\nPropose a plan that still achieves the goal but differs "
            "meaningfully from the plan(s) above, so this cycle's data is "
            "complementary rather than redundant. Only if no meaningfully "
            "different goal-reaching plan exists, repeat the best plan.\n")

    journal_section = ""
    if journal:
        journal_section = (
            "\n## Solve Journal (record of earlier attempts)\n"
            "You start with fresh context. The journal below is this run's "
            "persistent record from earlier solve attempts and tasks: "
            "auto-recorded outcomes (captured plans, rewards, budgets) "
            "plus agent-recorded lessons. Use it - reproduce what worked, "
            "do not repeat parameter sweeps it already covers - but treat "
            "any recorded conclusion skeptically: re-verify cheap claims "
            "rather than inheriting them, especially from failed "
            "attempts.\n"
            "Journal protocol for this attempt:\n"
            "- FIRST list the journal's untried leads, then execute or "
            "explicitly retire (with a measurement) each promising lead "
            "BEFORE re-opening a family an earlier attempt already marked "
            "exhausted or opening a brand-new one. Attempts have been "
            "wasted re-litigating condemned designs while a recorded, "
            "concrete, untried lead sat unexecuted.\n"
            "- A negative claim is only as broad as the family actually "
            "swept: before trusting 'X never works', check what was "
            "tested - a claim derived from one orientation, formula, or "
            "region says nothing about the rest.\n"
            "- If two entries conflict (one rules a mechanism out, another "
            "recommends it), BOTH demote to open questions: design the "
            "cheap experiment that decides between them instead of "
            "silently trusting either.\n"
            "Add your own lessons for future attempts with the "
            "record_journal tool (facts and measurements only).\n\n"
            f"{journal}\n")

    goal_nl_section = ""
    if task.goal_nl:
        goal_nl_section = f"\n## Goal Description\n{task.goal_nl}\n"

    # The env's public reward form (success condition + costs), when the
    # task ships an evaluator that states one. Without it, agents burn
    # turns reverse-engineering scores and induce false rules from them
    # (run_20260729_001752 hypothesized a "-0.10 binary route-rejected
    # flag" and a "-0.30 excess-blue penalty" from raw numbers the
    # formula decodes instantly). Public by design: reward FORM only,
    # never oracle quantities.
    scoring_section = ""
    evaluator = getattr(task, "evaluator", None)
    if evaluator is not None:
        objective = evaluator.objective_description()
        if objective:
            scoring_section = (
                "\n## Scoring (env ground-truth reward)\n"
                f"{objective}\n"
                "Decode every reward you observe with this scoring rule "
                "before hypothesizing any other mechanism - there are no "
                "hidden reward terms.\n")

    goal_atoms_section = ""
    if goal_strs:
        goal_atoms_section = (f"\n## Goal Atoms\n{chr(10).join(goal_strs)}\n")

    pred_strs = []
    for pred in sorted(all_predicates, key=lambda p: p.name):
        type_sig = ", ".join(t.name for t in pred.types)
        line = f"  {pred.name}({type_sig})"
        if pred.natural_language_assertion is not None:
            names = [t.name for t in pred.types]
            line += f" — {pred.natural_language_assertion(names)}"
        pred_strs.append(line)

    # Tool-availability-aware references: when explore_python replaces
    # the standalone refine tool (see
    # agent_planner_explore_python_keep_replaced_tools), guidance must
    # point at the probe equivalents instead of tools the session lacks.
    # ``tool_names=None`` keeps the legacy all-tools wording.
    tool_set = set(tool_names) if tool_names is not None else None

    def _has_tool(name: str) -> bool:
        return tool_set is None or name in tool_set

    probe_refine = (not _has_tool("refine_plan_sketch")
                    and _has_tool("explore_python"))
    refine_ref = ("`sim.refine` (in `explore_python`)"
                  if probe_refine else "`refine_plan_sketch`")
    if _has_tool("explore_python"):
        visualize_advice = (
            "- Use `explore_python` (`sim.reset(mods={...})`, then "
            "`sim.render(...)`) to move objects to candidate positions and "
            "orientations for free (no physics) and find the right region "
            "visually before testing.\n")
    else:
        # No visualization surface offered: no bullet, rather than
        # advice naming a capability the session lacks.
        visualize_advice = ""

    # Advice for a step the search reports stuck (SAMPLE_EXHAUSTED). When the
    # agent proposes params it tunes that step's values; otherwise it can only
    # change the skeleton.
    deep_tune_advice = (
        "deep-tune just that step (it needs precise values from you), then "
        "re-test it. When deep-tuning a step with `evaluate_option_plan`:\n"
        "- Inspect the rendered images in `./test_images/` to see what "
        "actually happened.\n"
        "- For a failure like an IK error or collision, use the image and "
        "object poses to reason about WHY and adjust params directionally — "
        "don't try random nearby values.\n" + visualize_advice +
        "- Vary ALL parameters, not just position — orientation and others "
        "affect both the outcome and whether the action succeeds.\n"
        "- Search coarse-to-fine: spread attempts across the full range; if "
        "several nearby values fail the same way, jump to a different region "
        "instead of continuing to tweak.\n"
        "- Before steering a search with a DERIVED formula or geometric "
        "prediction (e.g. which way an object moves, falls, or deflects), "
        "validate the formula on one clean controlled experiment first. A "
        "wrong formula makes correct designs look refuted, and that false "
        "negative then silently excludes the right design family from the "
        "rest of the search.")
    revise_sketch_advice = (
        "revise the sketch — try different objects, a different ordering, an "
        "added intermediate step, or a corrected subgoal annotation — then "
        "re-test.")

    if propose_params:
        ground_sampler_guidance = ""
        ground_sampler_format = ""
        if ground_samplers:
            ground_sampler_guidance = (
                " Confine its search near your estimate by appending a "
                "region `~ [w1, w2]` (per-parameter half-widths) after a "
                "step's `[params]`: the exact center is tried first, then "
                "every sample for that step stays inside "
                "`[center - w, center + w]` instead of the full range. For "
                "regions a fixed window cannot express (state-dependent or "
                "curved), write a function in `ground_samplers.py` "
                "(`GROUND_SAMPLERS = {\"my_sampler\": fn}`, "
                "`fn(state, subgoal_atoms, rng, objects) -> params`) and "
                "reference it as `~ my_sampler` instead; the file is "
                f"reloaded on every {refine_ref} call.")
            ground_sampler_format = (
                f"\n(For {refine_ref} only, a step may add a search "
                "region after its params: "
                "`OptionName(obj1:type1)[p1, p2] ~ [w1, w2] -> {...}`, or "
                "`... ~ my_sampler` naming a GROUND_SAMPLERS entry.)")
        sketch_kind_guidance = (
            "Generate a plan — the sequence of options with object arguments "
            "and continuous parameters in `[...]` per step (see each option's "
            "params and range above; use `[]` for options with no "
            "parameters).\n\n"
            "Explore designs before tuning parameters: when several "
            "qualitatively different designs could work (different objects, "
            "orientations, sides, orderings, or mechanisms), run a cheap "
            "test of each and compare failure modes BEFORE fine-tuning any "
            "one of them. Parameter tuning cannot rescue the wrong design - "
            "if a design keeps failing the same way as you tune it, switch "
            "designs rather than tightening values. And before building on "
            "a physics rule or constraint you inferred from a single "
            "observation, re-test it once with a clean experiment; a wrong "
            "rule adopted early can quietly rule out the correct designs.\n\n"
            "Spend effort on parameters in proportion to difficulty:\n"
            "- Where a WIDE range of values works, any reasonable value is "
            "fine — don't over-tune these.\n"
            "- Where good values are hard to hit — tight tolerances or exact "
            "relative placements (e.g. positioning one object at a precise "
            f"offset from another) - use {refine_ref} to search for a "
            "working value (it's slower) and read the value it found." +
            ground_sampler_guidance)
        format_block = (
            "Output the plan with one option per line in this format:\n"
            "  OptionName(obj1:type1, obj2:type2)[param1, param2] -> "
            "{Pred(obj1:type1), Pred2(obj1:type1, obj2:type2)}\n"
            "  Wait(robot:robot)[] -> "
            "{Pred3(obj1:type1), NOT Pred4(obj1:type1, obj2:type2)}" +
            ground_sampler_format)
    else:
        sketch_kind_guidance = (
            "Generate a plan sketch — the sequence of options with object "
            "arguments, WITHOUT continuous parameters; a backtracking search "
            "finds them for you.")
        format_block = (
            "Output the plan sketch with one option per line in this "
            "format:\n"
            "  OptionName(obj1:type1, obj2:type2) -> "
            "{Pred(obj1:type1), Pred2(obj1:type1, obj2:type2)}\n"
            "  Wait(robot:robot) -> "
            "{Pred3(obj1:type1), NOT Pred4(obj1:type1, obj2:type2)}")

    if require_tool_validation:
        stuck_advice = (deep_tune_advice
                        if propose_params else revise_sketch_advice)
        margin_guidance = ""
        if physics_margin:
            margin_guidance = (
                "Capture also requires the plan to succeed at a grid of "
                "perturbations spanning +-1 sigma of the identified "
                "physical parameters (the physics fit's own uncertainty); "
                "a plan that fails any point is reported PARAM-SENSITIVE "
                "instead of captured. Success can be NON-MONOTONIC in a "
                "physical parameter - a design can pass just above and "
                "just below a value and fail exactly at it - so tune "
                "designs that pass the WHOLE range: pre-check with "
                "`sim.run(plan_text, physics_sweep=True)` in explore_python "
                "(same points as the gate, one deterministic rollout each) "
                "instead of discovering rejections one submission at a "
                "time. ")
        submit_guidance = (
            "SUBMIT via `evaluate_option_plan`: pass your full plan as text "
            "(one option per line, `Option(obj:type)[params] -> {subgoals}`, "
            "with EXACT params) and run it on the CURRENT task (omit "
            "task_idx). When it reaches the goal, that plan is captured as "
            "your answer, so do NOT finish until evaluate_option_plan "
            "CONFIRMS the capture. A goal-reaching plan is re-run several "
            "times before capture (simulation varies across runs); if it is "
            "reported FLAKY, add margin to the fragile step and resubmit. " +
            margin_guidance +
            "CAPTURE FIRST, OPTIMIZE SECOND: when the reward charges for "
            "resources used (read the scoring section), a captured "
            "modest-reward solve outscores an uncaptured optimal attempt "
            "by the entire success bonus - so bank a ROBUST goal-reaching "
            "design early, even an over-built one (extra margin, extra "
            "resources), and only then spend remaining budget improving "
            "it. This is safe: a newly VALIDATED capture replaces the "
            "banked one, while a rejected submission (flaky, "
            "param-sensitive, evaluator-rejected, or short of the goal) "
            "never displaces it - but do not resubmit designs that are "
            "not strictly better, since a validated worse plan would "
            "replace the banked answer. Robust-but-wasteful designs live "
            "AWAY from the feasibility boundary that minimal designs sit "
            "on, so they are usually far easier to find and validate. "
            "It runs your EXACT parameters with no sampling. To find "
            f"working parameters you MAY use {refine_ref} (it searches "
            "but is slower); read the parameters it reports and submit them "
            "via evaluate_option_plan. If a step does not reach its subgoal, "
            + stuck_advice)
    elif propose_params:
        submit_guidance = (
            f"You may validate with {refine_ref} (it tries your "
            "parameters first, then samples) and deep-tune any step it "
            "reports stuck before finishing.")
    else:
        submit_guidance = (
            f"You may vet a sketch with {refine_ref} before finishing; "
            "the backtracking search will find continuous parameters.")
    if explore_mode:
        submit_guidance += (
            " Your plan will be executed in the REAL environment as an "
            "experiment, and the data it produces is what the next "
            "learning cycle uses to correct the belief model. A plan "
            "that reaches the goal in the simulator is ideal when one "
            "exists; when the goal depends on a mechanism the belief "
            "model lacks (see the Belief-Model Simulator section), "
            "still submit the plan most likely to achieve the goal in "
            "reality - reason from the goal description, the scene "
            "geometry, and physical common sense - and annotate the "
            "subgoals that SHOULD hold if the mechanism works. The "
            "disagreement between the model's prediction and reality "
            "is exactly the signal exploration exists to collect, so a "
            "simulator-failing sketch is a valid, useful deliverable "
            "there. Do NOT keep searching for a simulator-validated "
            "plan the belief model cannot produce.")

    if require_tool_validation:
        # Plain text is NOT a submission in this mode; saying "output the
        # plan lines" as the closing instruction has led agents (especially
        # right after an SDK context compaction, whose text-only summary
        # instruction bleeds into the task) to answer with an unvalidated
        # text sketch and finish, wasting the whole attempt.
        closing_block = (
            "Your answer is ONLY accepted from a goal-reaching "
            "`evaluate_option_plan` run on the CURRENT task; final text "
            "alone is discarded, so never finish without that validated "
            "run. Tool calls are permitted on every turn of this "
            "conversation. If an earlier context summary says a turn was "
            "text-only, that applied to writing the summary itself, not to "
            "this task; resume calling tools. After the goal-reaching run, "
            "repeat its plan lines as your final text.")
    elif explore_mode:
        closing_block = (
            "This is an EXPLORE query: your final plan-sketch text IS the "
            "deliverable (a simulator-validated capture is welcome but NOT "
            "required). Output ONLY the plan sketch lines at the end, "
            "after any analysis.")
    else:
        closing_block = (
            "Output ONLY the plan sketch lines at the end, after any "
            "analysis.")

    prompt = f"""You are solving a task. \
Generate a plan sketch to achieve the goal.
{goal_nl_section}{scoring_section}{goal_atoms_section}{belief_model_section}\
{experiment_section}
## Initial State Atoms
{chr(10).join(atom_strs)}

## Initial State Features
{state_str}
{initial_image_section}
## Objects
{chr(10).join(obj_strs)}

## Available Options
{chr(10).join(option_strs)}

## Available Predicates (for subgoal annotations)
{chr(10).join(pred_strs)}
{trajectory_summary}{tools_str}{journal_section}\
{scheduled_plans_section}
## Instructions
Use your available tools to inspect the environment before producing the plan.

{sketch_kind_guidance}

{submit_guidance}

Annotate subgoal atoms after EVERY step whose effect your predicates can \
express, using `-> {{atoms}}`. Prefer atoms that NEWLY hold (or stop \
holding) because of the step — atoms that were already true beforehand \
reveal nothing. Annotations are load-bearing: the search validates each \
annotated step, and during execution they are checked against the real \
state so a diverged step triggers replanning instead of silently dooming \
the rest of the plan.

After any action whose desired subgoal depends on a delayed process (e.g. \
water filling, dominoes cascading, heating), insert a Wait action. For Wait \
steps, annotate with the atoms the process should produce — this tells the \
system exactly when the Wait should end rather than terminating on any \
incidental atom change. Use `NOT Pred(...)` for atoms that should become false.

{format_block}

Always use typed references (obj:type) in both option arguments AND subgoal \
atoms. If you omit `-> {{atoms}}` on a step, the search only checks that the \
option executed (non-zero actions) and execution monitoring is blind there — \
omit it only when no available predicate can express the step's effect.

{closing_block}"""

    return prompt
