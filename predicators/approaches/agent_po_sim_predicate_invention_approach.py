"""Partial-observability (PO) sim-learning + predicate-invention approach.

Extends ``AgentSimPredicateInventionApproach`` to handle envs where some
causally-important features are hidden from the agent-visible observation
(motivating example: ``jug.heat_level`` in ``pybullet_boil`` when
``CFG.partially_observable`` is True). The synthesizing Claude agent now:

* writes rules with a 5-arg signature
  ``rule(state, latent, history, updates, params)`` so they can carry a
  ``latent`` state dict across steps and/or read the prior observation
  history;
* optionally declares ``LATENT_INIT`` (a dict, or zero-arg callable
  returning one) giving the initial latent block;
* invents predicates that may be observation-only OR latent-aware
  (``classifier(state, objs, latent=None)``).

Two natural patterns the prompt presents (agent picks per latent):

* **Counter + threshold** - carry a step counter; flip an observable
  when the counter crosses a learnable threshold.
* **Physical latent + readout** - carry an estimate of the unobserved
  physical quantity; map it through a monotone readout to an observable.

A ``State`` flowing through the simulator is one *sample* of the augmented
state - observable features in ``State.data`` plus the inferred latent
dimensions in ``State.latent`` (e.g. ``{"heat": ...}`` or
``{"streak": ...}``); a belief is the (here point-mass) distribution over
such samples.

All latent *mechanics* now live in ``AgentSimLearningApproach`` and
activate automatically when the loaded rules use the 5-arg signature:
recurrent MCMC fitting (``compute_sse_recurrent`` /
``fit_params_recurrent``), the latent-threaded combined simulator (the
latent rides the opaque ``State.latent`` field, so the
``(State, Action) -> State`` option-model interface is unchanged and
backtracking restores the latent at each search node), ``LATENT_INIT``
loading, and initial-latent seeding. This subclass therefore only adds the
synthesis *prompt* teaching the agent to write such rules (on top of
predicate invention). Latent-aware predicates work uniformly:
``Predicate.holds`` auto-reads ``state.latent`` when no explicit kwarg is
passed, so the same classifier is correct during
``evaluate_predicate_quality`` *and* inside
``bilevel_sketch.refine_sketch``.

Example command::

    python predicators/main.py --env pybullet_boil \
        --approach agent_po_sim_predicate_invention --seed 0 \
        --num_train_tasks 10 --num_test_tasks 5 \
        --partially_observable True \
        --num_online_learning_cycles 2 --explorer agent_plan
"""

from typing import Dict

from predicators.approaches.agent_sim_predicate_invention_approach import \
    AgentSimPredicateInventionApproach


class AgentPOSimPredicateInventionApproach(AgentSimPredicateInventionApproach):
    """Partial-observability variant: rules carry a `latent` block across
    steps.

    The latent mechanics (recurrent fitting, latent-threaded combined
    simulator, ``LATENT_INIT`` loading, initial-latent seeding) live in
    ``AgentSimLearningApproach`` and activate automatically when the loaded
    rules use the recurrent 5-arg signature. This subclass only adds the
    synthesis prompt teaching the agent to write such rules - predicate
    invention plus a partial-observability prompt.
    """

    @classmethod
    def get_name(cls) -> str:
        return "agent_po_sim_predicate_invention"

    # ── Prompt overrides ─────────────────────────────────────────

    def _rule_signature_section(self) -> str:
        # Present only the recurrent 5-arg signature as canonical, so the
        # PO prompt never advertises the 3-arg form the recurrent engine
        # rejects. Full latent guidance is in the appended "## Recurrent
        # rules (partial observability)" section.
        return _PO_RULE_SIGNATURE_SECTION

    def _residual_rule_signature(self) -> str:
        # Keep the geometric-gate worked example on the same 5-arg shape.
        return "def residual_rule(state, latent, history, updates, params):"

    def _extra_synthesis_system_prompt(self) -> str:
        base = super()._extra_synthesis_system_prompt()
        return base + "\n\n" + _RECURRENT_PROMPT_SECTION

    def _extra_synthesis_message(self, extra_paths: Dict[str, str]) -> str:
        base = super()._extra_synthesis_message(extra_paths)
        return base + "\n\n" + _RECURRENT_MESSAGE_SECTION


_PO_RULE_SIGNATURE_SECTION = '''\
### Rule signature

This is a **partial-observability** task. Write every rule with the
recurrent 5-arg signature below — the 2nd parameter MUST be named
`latent` (the engine inspects each rule's signature and threads the
latent block / read-only history only into rules that declare it):

```python
def rule(state, latent, history, updates, params):
    # state:   the current env State (observable features only)
    # latent:  Dict[str, Any], mutated in place — the hidden dims you
    #          infer, threaded across steps (see "Recurrent rules" below)
    # history: List[Tuple[State, Optional[Action]]], read-only; newest last
    # updates: Dict[Object, Dict[str, float]] accumulated from prior rules
    # params:  Dict[str, float], one entry per ParamSpec
    #
    # Accumulate, don't replace:
    #     updates.setdefault(obj, {})[feat] = new_value
    # Return the same dict.
    ...
```

A rule that needs no hidden state can ignore its `latent`/`history`
args, but keep the 5-arg shape so the tools and the fitting engine call
every rule the same way. See "## Recurrent rules (partial observability)"
below for `LATENT_INIT` and the two latent-modelling patterns.'''

_RECURRENT_PROMPT_SECTION = """\
## Recurrent rules (partial observability)

This approach handles partial observability: the observation may omit
causally-important quantities — there may be several, one, or none.
Anything omitted is *absent entirely* from the state (it appears under
no name, not even as a NaN placeholder), so you cannot read it and
must *infer* its existence and dynamics from how the observable
features evolve. Inspect the trajectories first to judge how many
latents (if any) you need: a feature that drifts or ramps with no
visible observed driver is likely downstream of an accumulating
latent; if every observable is already explained by other observed
quantities, you need no latent at all — keep the 5-arg signature and
simply leave `latent` untouched. One common case: a hidden continuous
quantity surfaced only through a derived observable that ramps once the
latent crosses a threshold.

Model the hidden state explicitly: each ``State`` you predict is one
sample of an *augmented* state — observable features in ``state.data``
plus the latent dimensions you infer in ``state.latent`` (a free-form
dict like ``{"level": 0.73}`` or ``{"count": 22}``). Write rules with
the recurrent 5-arg signature so they can read and advance that latent:

```python
def my_rule(state, latent, history, updates, params):
    # state    : current observation State (no hidden features)
    # latent   : Dict[str, Any], mutated in place — the latent state
    #            dims you track, threaded across steps
    # history  : List[Tuple[State, Optional[Action]]], read-only;
    #            most recent last; first action is None
    # updates  : ResidualUpdate dict, also mutated in place
    # params   : Dict[str, float] (fitted scalars)
    ...
    return updates
```

The 2nd parameter MUST be named ``latent`` — the engine inspects each
rule's signature and only threads the latent block into rules that
declare it. Declare the initial latent block:

```python
LATENT_INIT = {"level": 0.0, "count": 0}
# OR a zero-arg callable returning such a dict.
# Use ParamSpec("name", ...) values to make an init value learnable.
```

Every rule uses this 5-arg signature, so the tools and the fitting
engine call them all the same way. A rule that needs no hidden state
simply ignores its `latent`/`history` arguments.

### Structure the latent like the state (per-object)

The augmented state is the observable features in ``state.data`` *plus*
the latent dims you infer: a jug's hidden ``heat`` is just another
feature of that jug that happens to be unobserved. So **shape the latent
like ``data`` — object first, then feature**: ``latent[jug.name]["heat"]``
should read in parallel with ``state.get(jug, "water_volume")``. The
hidden quantities almost always belong to *individual* objects (each jug
its own heat, each faucet its own spill buffer), and with several
same-type objects a flat ``{"heat": 0.0}`` collapses them into one shared
accumulator, which is wrong — exactly as your rules must loop over every
object rather than indexing ``[0]``.

```python
LATENT_INIT = {}          # {jug_name: {"heat": value}}, filled lazily

def heat_rule(state, latent, history, updates, params):
    jugs = [o for o in state.data if o.type.name == "jug"]
    for jug in jugs:
        jl = latent.setdefault(jug.name, {})    # this jug's hidden dims
        h = jl.get("heat", 0.0)
        if on_active_burner(state, jug, params):
            h += 1.0
        jl["heat"] = h
        updates.setdefault(jug, {})["bubbling_level"] = readout(h, params)
    return updates
```

Two deliberate differences from ``data``, though — the latent is **not**
a typed feature array, and must not be made into one: (1) key by the
stable string ``obj.name``, not the live ``Object`` (``data`` keys by
``Object``, but the latent is deep-copied / reconstructed at every search
node, so a live key risks identity mismatch); (2) keep it a free-form
JSON-like nest of dicts / numbers with no registered schema — the agent
invents these dims, and the engine threads and deep-copies whatever
structure you put here. A genuinely global hidden quantity (a world
clock, ambient temperature) stays a top-level scalar rather than being
forced under an object. (Top-level scalar latent entries may be
``ParamSpec``s to make their initial value learnable; seed each
per-object slot lazily from such a shared init.)

The type, feature, latent, and parameter names in the examples below
(`widget`, `fixture`, `progress`, `level`, ...) are illustrative — use
whatever your prompt digests and the trajectory data actually report
for your task.

### Two synthesis patterns (agent picks per latent)

**Pattern A — Counter + threshold.** Carry a step counter; flip the
observable when it crosses a learnable threshold. Same statistical
shape as a delayed discrete event:

```python
PARAM_SPECS = [ParamSpec("delay", init_value=33, lo=1, hi=200)]
LATENT_INIT = {"count": 0}

def count_rule(state, latent, history, updates, params):
    active = is_widget_at_fixture(state)  # observable check
    fixture_on = state.get(fixture, "is_on") > 0.5
    if active and fixture_on:
        latent["count"] += 1
    else:
        latent["count"] = 0
    fired = latent["count"] >= params["delay"]
    updates[widget]["progress"] = 1.0 if fired else 0.0
    return updates
```

**Pattern B — Physical latent + readout.** Carry an estimate of the
unobserved quantity; map it through a (typically monotone) function to
predict the observable. Higher resolution: the observable co-varies
smoothly with the latent before the symbolic "done" point.

```python
PARAM_SPECS = [ParamSpec("rate", init_value=0.03, lo=0.0, hi=0.1)]
LATENT_INIT = {"level": 0.0}

def level_rule(state, latent, history, updates, params):
    active = is_widget_at_fixture(state)
    fixture_on = state.get(fixture, "is_on") > 0.5
    if active and fixture_on:
        latent["level"] += params["rate"]
    lvl = latent["level"]
    # monotone readout: ramps from 0 once `lvl` passes an onset (~0.85)
    updates[widget]["progress"] = max(0.0, min(1.0, (lvl - 0.85) / 0.15))
    return updates
```

**How to choose.** Look at the derived observable in the inspect
tools:
- Smooth ramp across many steps ⇒ Pattern B (partial-progress
  signal; rate identifiable from a single trajectory by slope-fit).
- Clean discrete flip at a variable tick ⇒ Pattern A may suffice
  (one learnable threshold, calibrated from the empirical
  flip-time distribution across trajectories).
- Mixing is fine: different rules / different latents can use
  different patterns within the same simulator.

### Keep carried state in `latent`, not in your emitted observables

Anything your rule must remember across steps — a counter, an accumulated
level, an irreversible "done" flag — belongs in `latent`. Treat the
observables you write to `updates` as **outputs only**: recompute them
from `latent` (and base-owned inputs) each step; never read one of your
own emitted features back in as state. The planner resets and replays
states during refinement, and only `latent` is guaranteed to be threaded
across those jumps — an emitted observable may not survive a reset, so a
rule that latches on its own output can pass a step-by-step rollout yet
break at refinement time. Patterns A and B above already follow this: the
observable is a fresh readout of `latent`. (Reading features the base sim
owns — positions, `is_on`, `is_held` — is fine; those are restored
faithfully.)

### Predicate signature

Classifiers may stay observation-only or take an optional ``latent``
kwarg. The latent block is available at refinement time too — the
planner threads it through ``state.latent`` across search nodes, and
``Predicate.holds`` auto-routes it into classifiers that opted in. Be
defensive: at the very first step ``state.latent`` may still be ``{}``
if the agent's ``LATENT_INIT`` is empty, and during predicate-quality
scoring on *raw env* trajectories ``latent`` will be the block
materialised by the agent's rules (so still meaningful, but only as
accurate as the rules themselves).

```python
# Observation-only (robust to bad rule chains; preferred when the
# observable carries enough signal):
Predicate("ProcessDone", [widget_type],
          lambda s, objs, latent=None:
              s.get(objs[0], "progress") > 0.5)

# Latent-aware (inherits simulator correctness; defend against
# missing keys at step 0):
Predicate("ProcessDone", [widget_type],
          lambda s, objs, latent=None:
              (latent or {}).get("level", 0.0) >= params["done_thresh"])
```

The kwarg MUST be named exactly ``latent`` for the auto-routing to
fire. Trade-off: latent-aware predicates inherit the simulator's
correctness; observation-only predicates are robust to bad rules
but only work when the observable carries enough signal.

### Diagnostics

`evaluate_predicate_quality` rolls each trajectory through your
simulator to materialise the latent before scoring classifiers, so
latent-aware predicates get a real block there. Use the eval
report to localise failures (bad rule chain vs. bad threshold).
"""

_RECURRENT_MESSAGE_SECTION = """\
## Partial observability

Some causally-important quantities may be absent from the agent-visible
observation entirely (under no name, not even as NaN) — possibly
several, possibly none. Inspect the trajectories first to judge whether
any hidden process is at work and which observable features are your
window into it; then, if any latents are needed, choose Pattern A or
Pattern B (or mix) to model the underlying dynamics in `latent`.
"""
