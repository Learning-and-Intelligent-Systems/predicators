"""Tests for ``AgentSimPredicateInventionApproach`` pure-Python helpers.

Covers the two pieces that don't need a real agent SDK to exercise:

* ``_compute_kept_initial_predicates`` — applies the allowlist and
  closure-strips derived predicates whose dependencies were removed.
* ``_load_predicates_from_module_file`` — sandbox loader for
  ``predicates.py`` that the agent writes during synthesis. Rejects
  non-Predicate entries, name collisions with the kept-env predicates,
  duplicates, and bad files; returns the valid set.
"""
# pylint: disable=protected-access,import-outside-toplevel,unused-import
from __future__ import annotations

import textwrap
from typing import Any, Set

import numpy as np
import pytest

# Bootstrap circular imports before pulling from predicators.approaches.
from predicators import utils
from predicators.structs import DerivedPredicate, Object, Predicate, State, \
    Type

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(name="cup_type")
def _cup_type():
    # Name without the ``_type`` suffix so the exec-context binding is
    # ``cup_type`` (not ``cup_type_type``), matching what the agent sees.
    return Type("cup", ["x", "y"])


def _classifier(_state, _objs):
    return True


# ── _compute_kept_initial_predicates ─────────────────────────────────


def _make_fake_self(initial_predicates: Set[Predicate],
                    kept_names: Set[str]) -> Any:
    """Build a stand-in approach instance whose only state is what
    ``_compute_kept_initial_predicates`` actually touches."""
    from predicators.approaches.agent_sim_predicate_invention_approach import \
        AgentSimPredicateInventionApproach
    fake_cls = type(
        "_FakeApproach", (AgentSimPredicateInventionApproach, ), {
            "__init__": lambda self: None,
            "_resolve_kept_names":
            lambda self, _kept=kept_names: frozenset(_kept),
        })
    fake = fake_cls()
    fake._initial_predicates = initial_predicates
    return fake


def test_kept_initial_predicates_allowlist_filter(cup_type):
    """A predicate whose name is in the allowlist is kept; others are dropped —
    this is the baseline allowlist behaviour added in commit 904f7c062 ("Drop
    env-goal mimicry")."""
    keep = Predicate("Holding", [cup_type], _classifier)
    drop = Predicate("JugAtFaucet", [cup_type], _classifier)
    fake = _make_fake_self({keep, drop}, kept_names={"Holding"})
    out = fake._compute_kept_initial_predicates()
    assert keep in out
    assert drop not in out


def test_kept_initial_predicates_strips_derived_with_missing_aux(cup_type):
    """A ``DerivedPredicate`` whose ``auxiliary_predicates`` reference a.

    *stripped* base is itself stripped — the agent must invent both,
    not see a half-broken classifier.
    """
    base_kept = Predicate("Holding", [cup_type], _classifier)
    base_dropped = Predicate("FaucetOn", [cup_type], _classifier)

    def _derived_classifier(_atoms, _objs):  # noqa: ARG001
        return True

    derived = DerivedPredicate(
        "GoalDone",
        [cup_type],
        _derived_classifier,
        auxiliary_predicates={base_dropped},
    )
    # Allowlist names include both the surviving base and the derived
    # predicate, but ``FaucetOn`` is stripped → derived must follow.
    fake = _make_fake_self({base_kept, base_dropped, derived},
                           kept_names={"Holding", "GoalDone"})
    out = fake._compute_kept_initial_predicates()
    assert base_kept in out
    assert derived not in out  # closure-stripped
    assert base_dropped not in out


def _make_fake_sim_learning_self(initial_predicates: Set[Predicate]) -> Any:
    """Stand-in for the sim-learning parent, using its REAL
    ``_resolve_kept_names`` so the CFG-override path is exercised."""
    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    fake_cls = type("_FakeSimLearning", (AgentSimLearningApproach, ), {
        "__init__": lambda self: None,
    })
    fake = fake_cls()
    fake._initial_predicates = initial_predicates
    return fake


def test_sim_learning_keeps_all_predicates_by_default(cup_type):
    """With no allowlist configured, every env predicate stays.

    The sim-learning class default is None and the CFG flag is unset, so
    even goal predicates remain available to the agent.
    """
    utils.reset_config({"agent_sim_learn_kept_predicates_names": []})
    holding = Predicate("Holding", [cup_type], _classifier)
    toppled = Predicate("Toppled", [cup_type], _classifier)
    fake = _make_fake_sim_learning_self({holding, toppled})
    assert fake._compute_kept_initial_predicates() == {holding, toppled}


def test_sim_learning_cfg_allowlist_strips_goal_predicates(cup_type):
    """The CFG allowlist strips predicates from the agent's set.

    Here the stripped predicate is a goal predicate.
    """
    utils.reset_config({"agent_sim_learn_kept_predicates_names": ["Holding"]})
    try:
        holding = Predicate("Holding", [cup_type], _classifier)
        toppled = Predicate("Toppled", [cup_type], _classifier)
        fake = _make_fake_sim_learning_self({holding, toppled})
        assert fake._compute_kept_initial_predicates() == {holding}
    finally:
        utils.reset_config({"agent_sim_learn_kept_predicates_names": []})


# ── _load_predicates_from_module_file ────────────────────────────────


def _make_loader_self(cup_type: Type,
                      kept: Set[Predicate],
                      include_train_task: bool = True) -> Any:
    """Build a stand-in approach with just the attrs the loader reads."""
    from predicators.approaches.agent_sim_predicate_invention_approach import \
        AgentSimPredicateInventionApproach

    # Provide a non-empty State so validate_predicate has something to
    # try the classifier on; the actual classifier always returns True
    # so validation passes for well-formed predicates.
    obj = Object("cup0", cup_type)
    init = State({obj: np.array([0.0, 0.0])})

    fake_task = type("_T", (), {"init": init})()

    fake_cls = type("_FakeLoaderApproach",
                    (AgentSimPredicateInventionApproach, ), {
                        "__init__": lambda self: None,
                        "_get_all_options": lambda self: set(),
                    })
    fake = fake_cls()
    fake._types = {cup_type}
    fake._kept_initial_predicates = kept
    fake._fitted_params = {}
    fake._train_tasks = [fake_task] if include_train_task else []
    return fake


def test_load_predicates_missing_file_returns_empty(cup_type, tmp_path):
    """Missing file → empty set, no exception."""
    fake = _make_loader_self(cup_type, kept=set())
    out = fake._load_predicates_from_module_file(
        str(tmp_path / "does_not_exist.py"))
    assert out == set()


def test_load_predicates_happy_path(cup_type, tmp_path):
    """A valid ``LEARNED_PREDICATES = [Predicate(...)]`` round-trips
    through exec_code_safely + validate_predicate."""
    fake = _make_loader_self(cup_type, kept=set())
    path = tmp_path / "predicates.py"
    path.write_text(
        textwrap.dedent("""
            LEARNED_PREDICATES = [
                Predicate("InventedFlag", [cup_type],
                          lambda s, objs: True),
            ]
            """))
    out = fake._load_predicates_from_module_file(str(path))
    names = {p.name for p in out}
    assert names == {"InventedFlag"}


def test_load_predicates_rejects_name_collision_with_kept(cup_type, tmp_path):
    """Invented predicate whose name collides with a kept env predicate is
    silently skipped (so the kept classifier stays authoritative)."""
    holding = Predicate("Holding", [cup_type], _classifier)
    fake = _make_loader_self(cup_type, kept={holding})
    path = tmp_path / "predicates.py"
    path.write_text(
        textwrap.dedent("""
            LEARNED_PREDICATES = [
                Predicate("Holding", [cup_type], lambda s, objs: True),
                Predicate("Good", [cup_type], lambda s, objs: True),
            ]
            """))
    out = fake._load_predicates_from_module_file(str(path))
    names = {p.name for p in out}
    assert names == {"Good"}  # "Holding" was dropped


def test_load_predicates_rejects_non_predicate_entries(cup_type, tmp_path):
    """Garbage entries (strings, ints) are skipped — the rest still load."""
    fake = _make_loader_self(cup_type, kept=set())
    path = tmp_path / "predicates.py"
    path.write_text(
        textwrap.dedent("""
            LEARNED_PREDICATES = [
                "not a predicate",
                42,
                Predicate("GoodOne", [cup_type], lambda s, objs: True),
            ]
            """))
    out = fake._load_predicates_from_module_file(str(path))
    assert {p.name for p in out} == {"GoodOne"}


def test_load_predicates_wrong_top_level_type(cup_type, tmp_path):
    """``LEARNED_PREDICATES`` must be a list — a dict returns an empty set
    rather than raising."""
    fake = _make_loader_self(cup_type, kept=set())
    path = tmp_path / "predicates.py"
    path.write_text("LEARNED_PREDICATES = {'Holding': 1}\n")
    out = fake._load_predicates_from_module_file(str(path))
    assert out == set()


def test_load_predicates_swallows_exec_errors(cup_type, tmp_path):
    """A predicates.py with a syntax error returns empty rather than bubbling
    the exception up to the synthesis loop."""
    fake = _make_loader_self(cup_type, kept=set())
    path = tmp_path / "predicates.py"
    path.write_text("def this is not valid python(\n")
    out = fake._load_predicates_from_module_file(str(path))
    assert out == set()
