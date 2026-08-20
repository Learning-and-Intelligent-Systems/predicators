"""Tests for SamplerLearningMixin's loader and oracle-install logic.

Covers ``_load_samplers_from_module_file`` (missing file, exec error,
non-dict, bad entries, happy path) and
``_maybe_install_oracle_samplers`` (GT install, fallback to synthesis,
disabled no-op) on a minimal host.
"""

from typing import Any, Dict, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches import sampler_learning_mixin
from predicators.approaches.sampler_learning_mixin import SamplerLearningMixin
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


class _Host(SamplerLearningMixin):  # pylint: disable=abstract-method
    """Minimal host supplying the mixin's contract surface."""

    def __init__(self):
        init = State({_block: np.array([0.0], dtype=np.float32)})
        self._types = {_block_type}
        self._train_tasks = [Task(init, {GroundAtom(_Reached, [_block])})]
        self._fitted_params: Dict[str, float] = {}
        self._synthesized_samplers: Dict[str, Any] = {}
        self._init_sampler_learning_state()

    def _get_all_predicates(self) -> Set[Predicate]:
        return {_Reached}

    def _get_all_options(self) -> Set[ParameterizedOption]:
        return {_Move}

    def _learning_cycle_index(self) -> int:
        return 1


def _host(**config):
    utils.reset_config({"seed": 0, **config})
    return _Host()


# --------------------------------------------------------------------------- #
# _load_samplers_from_module_file.
# --------------------------------------------------------------------------- #


def test_load_samplers_missing_file_returns_empty(tmp_path):
    """A missing samplers.py loads as the empty dict (samplers optional)."""
    host = _host()
    assert not host._load_samplers_from_module_file(  # pylint: disable=protected-access
        str(tmp_path / "samplers.py"))


def test_load_samplers_exec_error_returns_empty(tmp_path):
    """A file that raises at exec time loads as the empty dict."""
    path = tmp_path / "samplers.py"
    path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    host = _host()
    assert not host._load_samplers_from_module_file(  # pylint: disable=protected-access
        str(path))


def test_load_samplers_non_dict_returns_empty(tmp_path):
    """LEARNED_SAMPLERS bound to a non-dict loads as the empty dict."""
    path = tmp_path / "samplers.py"
    path.write_text("LEARNED_SAMPLERS = [1, 2]\n", encoding="utf-8")
    host = _host()
    assert not host._load_samplers_from_module_file(  # pylint: disable=protected-access
        str(path))


def test_load_samplers_skips_unknown_and_non_callable_entries(tmp_path):
    """Unknown option names and non-callables are dropped, the rest kept."""
    path = tmp_path / "samplers.py"
    path.write_text("""\
def _fn(state, subgoal_atoms, rng, objects):
    del state, subgoal_atoms, objects
    return np.array([0.5], dtype=np.float32)

LEARNED_SAMPLERS = {"Move": _fn, "Teleport": _fn, "Reached": 7}
""",
                    encoding="utf-8")
    host = _host()
    loaded = host._load_samplers_from_module_file(str(path))  # pylint: disable=protected-access
    assert set(loaded) == {"Move"}


def test_load_samplers_happy_path(tmp_path):
    """A valid file loads a callable that draws correctly shaped params."""
    path = tmp_path / "samplers.py"
    path.write_text("""\
def _fn(state, subgoal_atoms, rng, objects):
    del state, subgoal_atoms, objects
    return np.array([0.25 + 0.01 * rng.random()], dtype=np.float32)

LEARNED_SAMPLERS = {"Move": _fn}
""",
                    encoding="utf-8")
    host = _host()
    loaded = host._load_samplers_from_module_file(str(path))  # pylint: disable=protected-access
    assert set(loaded) == {"Move"}
    draw = loaded["Move"](
        host._train_tasks[0].init,  # pylint: disable=protected-access
        set(),
        np.random.default_rng(0),
        [_block])
    assert np.asarray(draw).shape == (1, )


# --------------------------------------------------------------------------- #
# _maybe_install_oracle_samplers.
# --------------------------------------------------------------------------- #


def _gt_sampler(state, subgoal_atoms, rng, objects):
    del state, subgoal_atoms, rng, objects
    return np.array([0.5], dtype=np.float32)


def test_oracle_samplers_installed_when_available(monkeypatch):
    """With oracle_samplers on and GT available: install, skip synthesis."""
    monkeypatch.setattr(sampler_learning_mixin, "get_gt_samplers",
                        lambda _env: {"Move": _gt_sampler})
    host = _host(agent_sim_learn_parameterized_samplers=True,
                 agent_sim_learn_oracle_samplers=True)
    host._maybe_install_oracle_samplers()  # pylint: disable=protected-access
    assert host._synthesized_samplers == {"Move": _gt_sampler}  # pylint: disable=protected-access
    assert host._current_samplers_version == "oracle"  # pylint: disable=protected-access
    assert not host._do_synthesize_samplers  # pylint: disable=protected-access


def test_oracle_samplers_fall_back_to_synthesis_when_none(monkeypatch):
    """With oracle_samplers on but no GT for the env: synthesize instead."""
    monkeypatch.setattr(sampler_learning_mixin, "get_gt_samplers",
                        lambda _env: {})
    host = _host(agent_sim_learn_parameterized_samplers=True,
                 agent_sim_learn_oracle_samplers=True)
    host._maybe_install_oracle_samplers()  # pylint: disable=protected-access
    assert not host._synthesized_samplers  # pylint: disable=protected-access
    assert host._do_synthesize_samplers  # pylint: disable=protected-access


def test_samplers_disabled_no_synthesis_no_install(monkeypatch):
    """With the master gate off nothing is installed or synthesized."""

    def _boom(_env):
        raise AssertionError("get_gt_samplers called with samplers disabled")

    monkeypatch.setattr(sampler_learning_mixin, "get_gt_samplers", _boom)
    host = _host(agent_sim_learn_parameterized_samplers=False,
                 agent_sim_learn_oracle_samplers=False)
    host._maybe_install_oracle_samplers()  # pylint: disable=protected-access
    assert not host._synthesized_samplers  # pylint: disable=protected-access
    assert not host._do_synthesize_samplers  # pylint: disable=protected-access
