"""Tests for the ``evaluate_sampler`` sampler-synthesis MCP tool.

Drives the real tool handler against a stub approach: loading
``LEARNED_SAMPLERS`` from ``samplers.py``, installing the validated dict
onto the approach, skip warnings for bad entries, error paths, snapshot
versioning, and the sanity check's empty-subgoal-set contract.
"""

import asyncio
from typing import Any, Dict, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.proposal_exec import build_exec_context, \
    load_ground_samplers
from predicators.agent_sdk.tools import create_sampler_synthesis_tools
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)

_Reached = Predicate("Reached", [_block_type], lambda s, o: True)

_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=lambda _s, _m, _o, _p: Action(np.zeros(1, dtype=np.float32)),
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: False,
)


class _StubApproach:
    """The minimal approach surface create_sampler_synthesis_tools uses."""

    def __init__(self):
        init = State({_block: np.array([0.0], dtype=np.float32)})
        self._types = {_block_type}
        self._train_tasks = [Task(init, {GroundAtom(_Reached, [_block])})]
        self._fitted_params: Dict[str, float] = {}
        self._synthesized_samplers: Dict[str, Any] = {}

    def _get_all_predicates(self) -> Set[Predicate]:
        return {_Reached}

    def _get_all_options(self) -> Set[ParameterizedOption]:
        return {_Move}


def _run_evaluate_sampler(tmp_path, code=None):
    utils.reset_config({"seed": 0})
    samplers_file = str(tmp_path / "samplers.py")
    if code is not None:
        with open(samplers_file, "w", encoding="utf-8") as f:
            f.write(code)
    approach = _StubApproach()
    tools = create_sampler_synthesis_tools(
        samplers_file=samplers_file,
        samplers_versions_dir=str(tmp_path / "samplers_versions"),
        approach=approach,
        cycle_index_provider=lambda: 1,
    )
    handlers = {t.name: t.handler for t in tools}
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result: Any = loop.run_until_complete(handlers["evaluate_sampler"]({}))
    return result["content"][0]["text"], approach


_GOOD = """\
def _move_sampler(state, subgoal_atoms, rng, objects):
    del state, subgoal_atoms, objects
    return np.array([0.25 + 0.01 * rng.random()], dtype=np.float32)

LEARNED_SAMPLERS = {"Move": _move_sampler}
"""


def test_evaluate_sampler_installs_valid_samplers(tmp_path):
    """A valid samplers.py is installed onto the approach and passes the sanity
    check."""
    text, approach = _run_evaluate_sampler(tmp_path, _GOOD)
    assert "1 per-skill sampler(s) installed" in text
    assert "Move: OK" in text
    assert "3/3 within the params box" in text
    assert set(approach._synthesized_samplers) == {"Move"}  # pylint: disable=protected-access
    assert callable(approach._synthesized_samplers["Move"])  # pylint: disable=protected-access


def test_evaluate_sampler_warns_unknown_option(tmp_path):
    """An entry keyed by a non-option name is skipped with a warning."""
    code = _GOOD + "\nLEARNED_SAMPLERS['Teleport'] = _move_sampler\n"
    text, approach = _run_evaluate_sampler(tmp_path, code)
    assert "Skipped 'Teleport' (not a known option name" in text
    assert set(approach._synthesized_samplers) == {"Move"}  # pylint: disable=protected-access


def test_evaluate_sampler_warns_non_callable(tmp_path):
    """A non-callable value is skipped with a warning."""
    code = _GOOD + "\nLEARNED_SAMPLERS['Move'] = 3.0\n"
    text, approach = _run_evaluate_sampler(tmp_path, code)
    assert "Skipped 'Move' (value is not callable" in text
    assert not approach._synthesized_samplers  # pylint: disable=protected-access


def test_evaluate_sampler_reports_exec_error(tmp_path):
    """A samplers.py that raises at import time reports the traceback."""
    text, approach = _run_evaluate_sampler(tmp_path,
                                           "raise RuntimeError('boom')")
    assert "Error executing" in text
    assert "boom" in text
    assert not approach._synthesized_samplers  # pylint: disable=protected-access


def test_evaluate_sampler_reports_missing_symbol(tmp_path):
    """A file without LEARNED_SAMPLERS names the missing symbol."""
    text, _ = _run_evaluate_sampler(tmp_path, "x = 1\n")
    assert "LEARNED_SAMPLERS" in text


def test_evaluate_sampler_missing_file_hint(tmp_path):
    """A missing samplers.py returns the Write hint, not a crash."""
    text, _ = _run_evaluate_sampler(tmp_path, code=None)
    assert "Use Write to create it" in text


def test_evaluate_sampler_empty_dict_message(tmp_path):
    """An empty LEARNED_SAMPLERS asks for entries instead of sanity lines."""
    text, _ = _run_evaluate_sampler(tmp_path, "LEARNED_SAMPLERS = {}\n")
    assert "LEARNED_SAMPLERS is empty" in text
    assert "Sanity check" not in text


def test_evaluate_sampler_version_tag_bumps_on_edit(tmp_path):
    """Within one tool instance (one synthesis session), an edited samplers.py
    gets a fresh version tag; an identical reload keeps it."""
    utils.reset_config({"seed": 0})
    samplers_file = tmp_path / "samplers.py"
    samplers_file.write_text(_GOOD, encoding="utf-8")
    tools = create_sampler_synthesis_tools(
        samplers_file=str(samplers_file),
        samplers_versions_dir=str(tmp_path / "samplers_versions"),
        approach=_StubApproach(),
        cycle_index_provider=lambda: 1,
    )
    handler = {t.name: t.handler for t in tools}["evaluate_sampler"]
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def _tag():
        result: Any = loop.run_until_complete(handler({}))
        return result["content"][0]["text"].split("]")[0].lstrip("[")

    tag1 = _tag()
    tag2 = _tag()  # unchanged file: same tag (snapshot deduped)
    samplers_file.write_text(_GOOD.replace("0.25", "0.75"), encoding="utf-8")
    tag3 = _tag()
    assert tag1 == "cycle_001_vers_001"
    assert tag2 == tag1
    assert tag3 == "cycle_001_vers_002"


def test_sanity_check_raising_sampler_mentions_empty_subgoal_contract(
        tmp_path):
    """A sampler that assumes a non-empty subgoal set gets the contract spelled
    out: refinement calls samplers with subgoal_atoms=set() at steps with no
    annotation, and the sanity check does the same."""
    code = """\
def _needs_subgoal(state, subgoal_atoms, rng, objects):
    atom = next(iter(subgoal_atoms))
    del state, rng, objects, atom
    return np.array([0.5], dtype=np.float32)

LEARNED_SAMPLERS = {"Move": _needs_subgoal}
"""
    text, _ = _run_evaluate_sampler(tmp_path, code)
    assert "ERROR" in text
    assert "subgoal_atoms=set()" in text
    assert "must not crash on an empty set" in text


def test_load_ground_samplers_happy_and_bad_entries():
    """GROUND_SAMPLERS loads callables; bad keys/values warn and drop."""
    ctx = build_exec_context(types={_block_type},
                             predicates={_Reached},
                             options={_Move})
    code = """\
def _fn(state, subgoal_atoms, rng, objects):
    del state, subgoal_atoms, objects
    return np.array([0.5], dtype=np.float32)

GROUND_SAMPLERS = {"hi_band": _fn, "not-an-identifier": _fn, "seven": 7}
"""
    fns, warnings, err = load_ground_samplers(code, ctx)
    assert err is None
    assert set(fns) == {"hi_band"}
    assert len(warnings) == 2
    assert any("identifiers" in w for w in warnings)
    assert any("not callable" in w for w in warnings)


def test_load_ground_samplers_errors():
    """Exec failures and non-dict bindings load nothing, with an error."""
    ctx = build_exec_context(types={_block_type},
                             predicates={_Reached},
                             options={_Move})
    fns, _, err = load_ground_samplers("raise RuntimeError('boom')", ctx)
    assert not fns
    assert err is not None and "boom" in err
    ctx = build_exec_context(types={_block_type},
                             predicates={_Reached},
                             options={_Move})
    fns, _, err = load_ground_samplers("GROUND_SAMPLERS = [1]", ctx)
    assert not fns
    assert err is not None and "must be a dict" in err


def test_sanity_check_wrong_shape_reports_error(tmp_path):
    """A wrong-shaped return is reported with got/expected shapes."""
    code = """\
def _bad_shape(state, subgoal_atoms, rng, objects):
    del state, subgoal_atoms, rng, objects
    return np.array([0.5, 0.5], dtype=np.float32)

LEARNED_SAMPLERS = {"Move": _bad_shape}
"""
    text, _ = _run_evaluate_sampler(tmp_path, code)
    assert "ERROR" in text
    assert "returned shape (2,), expected (1,)" in text
