"""Tests for per-step region annotations (``[center] ~ [widths]``) in sketch
parsing (``sketch_parsing``) and refinement (``sketch_refinement``).

A region annotation gives a step's LLM-proposed params per-dimension
half-widths: the exact center is tried once, then every later draw for
the step is uniform inside ``clip([center - w, center + w], box)``
instead of the full option box, taking precedence over any per-skill
sampler (region > sampler > uniform).
"""

import asyncio
from typing import Any

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.sketch_parsing import format_step_line, \
    parse_sketch_from_text, strip_region_annotations
from predicators.agent_sdk.sketch_refinement import refine_sketch
from predicators.agent_sdk.sketch_types import GroundSampler, SketchStep
from predicators.agent_sdk.tools import ToolContext, create_mcp_tools
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: False,
)

# A zero-dim option (no continuous params) for region-arity tests.
_Wait0 = ParameterizedOption(
    "Wait0",
    types=[_block_type],
    params_space=Box(low=np.zeros(0, dtype=np.float32),
                     high=np.zeros(0, dtype=np.float32)),
    policy=_noop_policy,
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: False,
)

_ReachedHi = Predicate("ReachedHi", [_block_type],
                       lambda s, o: s.get(o[0], "x") >= 0.9)


class _FakeOptionModel:
    """Deterministic model: Move sets block.x to its parameter value."""

    last_execution_failure = None

    def __init__(self):
        self.num_calls = 0
        self.last_trajectory = None

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step, counting the call."""
        self.num_calls += 1
        nxt = state.copy()
        if len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        return nxt, 1


def _task_hi():
    init = State({_block: np.array([0.0], dtype=np.float32)})
    return Task(init, {GroundAtom(_ReachedHi, [_block])})


def _region_step(center, width, subgoal=True):
    return SketchStep(
        option=_Move,
        objects=[_block],
        subgoal_atoms={GroundAtom(_ReachedHi, [_block])} if subgoal else None,
        initial_params=np.array([center], dtype=np.float32),
        ground_sampler=GroundSampler(center=np.array([center],
                                                     dtype=np.float32),
                                     width=np.array([width],
                                                    dtype=np.float32)))


def _refine(step, **kwargs):
    defaults = dict(predicates={_ReachedHi},
                    timeout=10.0,
                    rng=np.random.default_rng(0),
                    max_samples_per_step=50,
                    check_subgoals=True,
                    check_final_goal=False)
    defaults.update(kwargs)
    return refine_sketch(_task_hi(), [step], _FakeOptionModel(), **defaults)


def _parse(text, strict=True, parse_continuous_params=True):
    return parse_sketch_from_text(
        text,
        _task_hi(),
        predicates={_ReachedHi},
        options={_Move, _Wait0},
        types={_block_type},
        parse_continuous_params=parse_continuous_params,
        strict=strict)


# --------------------------------------------------------------------------- #
# Parsing.
# --------------------------------------------------------------------------- #


def test_parse_region_happy_path():
    """Center, width, and subgoal all populate from one annotated line."""
    sketch = _parse(
        "Move(block0:block)[0.7] ~ [0.1] -> {ReachedHi(block0:block)}")
    assert len(sketch) == 1
    assert np.allclose(sketch[0].initial_params, [0.7])
    assert np.allclose(sketch[0].ground_sampler.width, [0.1])
    assert GroundAtom(_ReachedHi, [_block]) in sketch[0].subgoal_atoms


def test_parse_region_strict_width_without_center_errors():
    """A region on a step with the explicit `[]` no-seed is an error."""
    with pytest.raises(ValueError, match="requires proposed center params"):
        _parse("Move(block0:block)[] ~ [0.1]")


def test_parse_region_strict_arity_mismatch_errors():
    """Half-width count must match the option's parameter count."""
    with pytest.raises(ValueError, match="expects 1"):
        _parse("Move(block0:block)[0.7] ~ [0.1, 0.2]")


def test_parse_region_strict_negative_width_errors():
    """Negative half-widths are rejected."""
    with pytest.raises(ValueError, match="must be >= 0"):
        _parse("Move(block0:block)[0.7] ~ [-0.1]")


def test_parse_region_strict_empty_width_block_errors():
    """An empty `~ []` block is an error, not an implicit no-region."""
    with pytest.raises(ValueError, match="empty"):
        _parse("Move(block0:block)[0.7] ~ []")


def test_parse_region_strict_non_numeric_errors():
    """A non-numeric half-width names the bad block."""
    with pytest.raises(ValueError, match="non-numeric"):
        _parse("Move(block0:block)[0.7] ~ [abc]")


def test_parse_region_strict_multiple_blocks_errors():
    """Two `~ [...]` blocks on one line are ambiguous."""
    with pytest.raises(ValueError, match="multiple"):
        _parse("Move(block0:block)[0.7] ~ [0.1] ~ [0.2]")


def test_parse_region_strict_zero_dim_option_errors():
    """A region on a zero-parameter option cannot have a center."""
    with pytest.raises(ValueError, match="requires proposed center params"):
        _parse("Wait0(block0:block)[] ~ [0.1]")


def test_parse_region_nonstrict_drops_bad_width_keeps_step():
    """Tolerant mode drops a malformed width but keeps the step + center."""
    sketch = _parse("Move(block0:block)[0.7] ~ [0.1, 0.2]", strict=False)
    assert len(sketch) == 1
    assert np.allclose(sketch[0].initial_params, [0.7])
    assert sketch[0].ground_sampler is None


def test_parse_region_ignored_when_params_disabled():
    """With param parsing off the annotation is inert (no width, no crash)."""
    sketch = _parse(
        "Move(block0:block)[0.7] ~ [0.1] -> {ReachedHi(block0:block)}",
        strict=False,
        parse_continuous_params=False)
    assert len(sketch) == 1
    assert sketch[0].initial_params is None
    assert sketch[0].ground_sampler is None
    assert GroundAtom(_ReachedHi, [_block]) in sketch[0].subgoal_atoms


def test_strip_region_annotations():
    """`~ [widths]` is removed so the params parser sees only `[params]`."""
    out = strip_region_annotations(
        "Move(block0:block)[0.7] ~ [0.1] -> {ReachedHi(block0:block)}")
    assert out == "Move(block0:block)[0.7] -> {ReachedHi(block0:block)}"
    # A line without an annotation is untouched.
    assert strip_region_annotations(
        "Move(block0:block)[0.7]") == "Move(block0:block)[0.7]"


def test_format_step_line_shows_width_and_annotation_round_trips():
    """The rendered region annotation parses back to the same values.

    format_step_line uses bare object names, so the test retypes the
    refs before parsing the line again.
    """
    line = format_step_line(0,
                            "Move", [_block],
                            params=[0.85],
                            subgoal_atoms={GroundAtom(_ReachedHi, [_block])},
                            params_width=[0.1])
    assert "[0.8500] ~ [0.1000]" in line
    sketch = _parse("Move(block0:block)" +
                    line.split("Move(block0)", maxsplit=1)[1])
    assert len(sketch) == 1
    assert np.allclose(sketch[0].initial_params, [0.85])
    assert np.allclose(sketch[0].ground_sampler.width, [0.1])
    assert GroundAtom(_ReachedHi, [_block]) in sketch[0].subgoal_atoms


def test_parse_region_disabled_falls_back_to_uniform():
    """With parse_ground_samplers=False a `~` annotation is accepted but
    ignored: the params still seed the search, no ground sampler is
    installed, and a notice explains the fallback (an error here cost
    every audited run turns of syntax guessing)."""
    notices = []
    sketch = parse_sketch_from_text("Move(block0:block)[0.7] ~ [0.1]",
                                    _task_hi(),
                                    predicates={_ReachedHi},
                                    options={_Move},
                                    types={_block_type},
                                    parse_continuous_params=True,
                                    strict=True,
                                    parse_ground_samplers=False,
                                    notices=notices)
    assert len(sketch) == 1
    assert np.allclose(sketch[0].initial_params, [0.7])
    assert sketch[0].ground_sampler is None
    assert len(notices) == 1
    assert "IGNORED" in notices[0]
    assert "uniform" in notices[0]
    # Without a notices list the annotation is still silently ignored.
    sketch = parse_sketch_from_text("Move(block0:block)[0.7] ~ [0.1]",
                                    _task_hi(),
                                    predicates={_ReachedHi},
                                    options={_Move},
                                    types={_block_type},
                                    parse_continuous_params=True,
                                    strict=True,
                                    parse_ground_samplers=False)
    assert sketch[0].ground_sampler is None


def _hi_band_fn(state, subgoal_atoms, rng, objects):
    del state, subgoal_atoms, objects
    return np.array([0.9 + 0.05 * rng.random()], dtype=np.float32)


def test_parse_named_ground_sampler():
    """`~ name` resolves against ground_sampler_fns into a code sampler."""
    sketch = parse_sketch_from_text(
        "Move(block0:block)[0.1] ~ hi_band -> {ReachedHi(block0:block)}",
        _task_hi(),
        predicates={_ReachedHi},
        options={_Move},
        types={_block_type},
        parse_continuous_params=True,
        strict=True,
        ground_sampler_fns={"hi_band": _hi_band_fn})
    assert len(sketch) == 1
    gs = sketch[0].ground_sampler
    assert gs is not None
    assert gs.fn is _hi_band_fn
    assert gs.name == "hi_band"
    # The center seed is independent of the named sampler.
    assert np.allclose(sketch[0].initial_params, [0.1])


def test_parse_named_ground_sampler_no_center_ok():
    """Unlike a window, a named sampler needs no center params."""
    sketch = parse_sketch_from_text(
        "Move(block0:block)[] ~ hi_band",
        _task_hi(),
        predicates={_ReachedHi},
        options={_Move},
        types={_block_type},
        parse_continuous_params=True,
        strict=True,
        ground_sampler_fns={"hi_band": _hi_band_fn})
    assert len(sketch) == 1
    assert sketch[0].initial_params is None
    assert sketch[0].ground_sampler.fn is _hi_band_fn


def test_parse_named_ground_sampler_unknown_errors():
    """An unknown `~ name` is a strict error listing what is available."""
    with pytest.raises(ValueError, match="unknown ground sampler 'nope'"):
        parse_sketch_from_text("Move(block0:block)[0.1] ~ nope",
                               _task_hi(),
                               predicates={_ReachedHi},
                               options={_Move},
                               types={_block_type},
                               parse_continuous_params=True,
                               strict=True,
                               ground_sampler_fns={"hi_band": _hi_band_fn})


def test_format_step_line_shows_sampler_name():
    """A named ground sampler renders as `~ name` after the params."""
    line = format_step_line(0,
                            "Move", [_block],
                            params=[0.1],
                            sampler_name="hi_band")
    assert "[0.1000] ~ hi_band" in line


# --------------------------------------------------------------------------- #
# Refinement semantics.
# --------------------------------------------------------------------------- #


def test_region_draws_confined_to_window():
    """After the failing center try, draws stay inside [c - w, c + w]."""
    # Center 0.85 misses x >= 0.9; the window [0.75, 0.95] still contains
    # passing values, unlike a pin, so regional sampling recovers.
    plan, success, total = _refine(_region_step(0.85, 0.1),
                                   max_samples_per_step=200)
    assert success
    assert total > 1  # the exact center failed first
    assert 0.9 <= float(plan[0].params[0]) <= 0.95


def test_region_takes_precedence_over_sampler():
    """A registered per-skill sampler is never consulted for a region step."""

    def sampler(*_args):
        raise AssertionError("sampler called despite region annotation")

    plan, success, _ = _refine(_region_step(0.5, 0.5),
                               max_samples_per_step=200,
                               parameterized_samplers={"Move": sampler})
    assert success
    assert float(plan[0].params[0]) >= 0.9


def test_region_window_clipped_to_box():
    """An oversized width clips to the option box (draws stay in-box)."""
    plan, success, _ = _refine(_region_step(0.95, 10.0),
                               max_samples_per_step=200)
    assert success
    assert 0.0 <= float(plan[0].params[0]) <= 1.0


def test_region_zero_width_pins_draws_to_center():
    """Zero width pins every draw to the center and caps the step at one
    attempt (re-drawing an identical value cannot help)."""
    # Passing center: succeeds immediately with the exact value.
    plan, success, total = _refine(_region_step(0.95, 0.0))
    assert success
    assert np.isclose(float(plan[0].params[0]), 0.95)
    assert total == 1
    # Failing center: exhausts after the single pinned attempt instead of
    # burning the full per-step budget on identical draws.
    _, success, total = _refine(_region_step(0.5, 0.0))
    assert not success
    assert total == 1


def test_region_applies_on_info_seeking_path():
    """Info-seeking pooling draws its candidates from the region window."""
    plan, success, _ = _refine(
        _region_step(0.95, 0.05),
        info_scorer=lambda s, _a: float(s.get(_block, "x")),
        info_n_feasible_target=4)
    assert success
    # Window [0.9, 1.0]: every pooled candidate clears the subgoal.
    assert float(plan[0].params[0]) >= 0.9


def test_region_step_not_capped_by_deterministic_sampler():
    """A deterministic-flagged sampler must not collapse a region step to a
    single attempt: the region bypasses the sampler entirely."""

    def sampler(*_args):
        return np.array([0.95], dtype=np.float32)

    sampler.deterministic = True
    plan, success, total = _refine(_region_step(0.5, 0.5),
                                   max_samples_per_step=200,
                                   parameterized_samplers={"Move": sampler})
    assert success
    # The failing center consumed the first attempt; regional draws (not a
    # single deterministic try) then found a passing value.
    assert total > 1
    assert float(plan[0].params[0]) >= 0.9


def _named_step(fn, name="named"):
    return SketchStep(option=_Move,
                      objects=[_block],
                      subgoal_atoms={GroundAtom(_ReachedHi, [_block])},
                      ground_sampler=GroundSampler(fn=fn, name=name))


def test_named_ground_sampler_draws_from_fn():
    """A code ground sampler owns the step's draws (state + subgoal seen)."""
    seen = []

    def fn(state, subgoal_atoms, rng, objects):
        seen.append(
            (state.get(_block,
                       "x"), set(subgoal_atoms), [o.name for o in objects]))
        return np.array([0.9 + 0.05 * rng.random()], dtype=np.float32)

    plan, success, _ = _refine(_named_step(fn))
    assert success
    assert float(plan[0].params[0]) >= 0.9
    x0, atoms, objs = seen[0]
    assert x0 == 0.0
    assert GroundAtom(_ReachedHi, [_block]) in atoms
    assert objs == ["block0"]


def test_named_ground_sampler_bad_shape_falls_back_to_uniform():
    """A misbehaving code sampler falls back to uniform draws per draw."""

    def bad(*_args):
        return np.array([0.5, 0.5], dtype=np.float32)  # shape (2,) != (1,)

    plan, success, _ = _refine(_named_step(bad), max_samples_per_step=200)
    assert success  # uniform fallback still lands x >= 0.9
    assert float(plan[0].params[0]) >= 0.9


def test_named_ground_sampler_deterministic_flag_caps_step():
    """A code ground sampler may flag itself deterministic (one attempt)."""

    def fn(*_args):
        return np.array([0.5], dtype=np.float32)  # never passes x >= 0.9

    fn.deterministic = True
    _, success, total = _refine(_named_step(fn))
    assert not success
    assert total == 1


# --------------------------------------------------------------------------- #
# Tool surfaces.
# --------------------------------------------------------------------------- #


def _run_tool(tool_name, args, ground_samplers=True, sandbox_dir=None):
    utils.reset_config({
        "agent_bilevel_use_llm_initial_params": True,
        "agent_bilevel_max_samples_per_step": 200,
        "agent_bilevel_ground_samplers": ground_samplers,
    })
    task = _task_hi()
    ctx = ToolContext(
        types={_block_type},
        predicates={_ReachedHi},
        processes=set(),
        options={_Move},
        train_tasks=[task],
        example_state=task.init,
        option_model=_FakeOptionModel(),
        current_task=task,
        sandbox_dir=sandbox_dir,
    )
    tools = {
        t.name: t.handler
        for t in create_mcp_tools(ctx, tool_names=[tool_name])
    }
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result: Any = loop.run_until_complete(tools[tool_name](args))
    return result["content"][0]["text"]


def test_refine_plan_sketch_tool_accepts_region_grammar():
    """The MCP handler parses the region and searches inside its window."""
    text = _run_tool(
        "refine_plan_sketch", {
            "plan": ("Move(block0:block)[0.85] ~ [0.1] -> "
                     "{ReachedHi(block0:block)}"),
            "timeout":
            10,
        })
    assert "SUCCESS" in text
    assert "Parameters found" in text
    # The reported parameter came from the window's passing band.
    param = float(text.split("Move(block0)[")[1].split("]")[0])
    assert 0.9 <= param <= 0.95


def test_refine_plan_sketch_tool_rejects_bad_region():
    """Strict tool parsing surfaces a malformed region as a clear error."""
    text = _run_tool("refine_plan_sketch", {
        "plan": "Move(block0:block)[0.85] ~ [0.1, 0.2]",
        "timeout": 10,
    })
    assert "Could not parse plan sketch" in text
    assert "expects 1" in text


def test_evaluate_option_plan_ignores_region():
    """evaluate_option_plan runs the exact center; the region is inert."""
    text = _run_tool(
        "evaluate_option_plan", {
            "plan": ("Move(block0:block)[0.95] ~ [0.05] -> "
                     "{ReachedHi(block0:block)}"),
            "include_states":
            False,
            "include_atoms":
            False,
        })
    # Goal achieved proves the exact center 0.95 ran (only x >= 0.9 passes);
    # a searched/perturbed value could not be distinguished, so also check
    # the report is a plain execution (no refinement verdict lines).
    assert "Goal achieved: True" in text


def test_refine_plan_sketch_tool_ignores_region_when_disabled():
    """With agent_bilevel_ground_samplers off, the annotation is a no-op.

    The params still seed the search but sampling stays uniform, and
    the report says so - baseline arms still cannot use the channel,
    but agents no longer burn turns on an error.
    """
    text = _run_tool("refine_plan_sketch", {
        "plan": ("Move(block0:block)[0.85] ~ [0.1] -> "
                 "{ReachedHi(block0:block)}"),
        "timeout":
        10,
    },
                     ground_samplers=False)
    assert "Could not parse plan sketch" not in text
    assert "IGNORED" in text
    assert "uniform" in text


def test_refine_plan_sketch_tool_named_ground_sampler(tmp_path):
    """A `~ name` reference loads GROUND_SAMPLERS from the sandbox and confines
    the step's draws to the function's distribution."""
    (tmp_path / "ground_samplers.py").write_text("""\
def _hi_band(state, subgoal_atoms, rng, objects):
    del state, subgoal_atoms, objects
    return np.array([0.9 + 0.05 * rng.random()], dtype=np.float32)

GROUND_SAMPLERS = {"hi_band": _hi_band}
""",
                                                 encoding="utf-8")
    text = _run_tool("refine_plan_sketch", {
        "plan": ("Move(block0:block)[0.1] ~ hi_band -> "
                 "{ReachedHi(block0:block)}"),
        "timeout":
        10,
    },
                     sandbox_dir=str(tmp_path))
    assert "SUCCESS" in text
    param = float(text.split("Move(block0)[")[1].split("]")[0])
    # The failing center 0.1 was tried once; the named sampler landed a
    # value inside its own band.
    assert 0.9 <= param <= 0.95


def test_refine_plan_sketch_tool_unknown_named_sampler(tmp_path):
    """An unresolvable `~ name` is a clear strict error, listing what is
    loaded."""
    text = _run_tool("refine_plan_sketch", {
        "plan": "Move(block0:block)[0.1] ~ nope",
        "timeout": 10,
    },
                     sandbox_dir=str(tmp_path))
    assert "Could not parse plan sketch" in text
    assert "unknown ground sampler 'nope'" in text


def test_refine_plan_sketch_tool_broken_ground_samplers_file(tmp_path):
    """A ground_samplers.py that fails to exec is surfaced as an error the
    agent can fix, not silently ignored."""
    (tmp_path / "ground_samplers.py").write_text("raise RuntimeError('bad')\n",
                                                 encoding="utf-8")
    text = _run_tool("refine_plan_sketch", {
        "plan": "Move(block0:block)[0.95]",
        "timeout": 10,
    },
                     sandbox_dir=str(tmp_path))
    assert "Error loading" in text
    assert "bad" in text
    assert "Parameters found" not in text
