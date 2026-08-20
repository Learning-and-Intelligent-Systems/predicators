"""Tests for AgentModelBasedApproach -- parsing and refinement logic."""
# pylint: disable=protected-access,import-outside-toplevel
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.approaches.agent_model_based_approach import \
    AgentModelBasedApproach, _SketchStep
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_block_type = Type("block", ["x", "y", "held"])
_robot_type = Type("robot", ["x", "y"])

_block0 = Object("block0", _block_type)
_block1 = Object("block1", _block_type)
_robot = Object("robot0", _robot_type)

_Holding = Predicate("Holding", [_block_type],
                     lambda s, o: s.get(o[0], "held") > 0.5)
_On = Predicate("On", [_block_type, _block_type],
                lambda s, o: abs(s.get(o[0], "x") - s.get(o[1], "x")) < 0.1)
_HandEmpty = Predicate("HandEmpty", [_robot_type], lambda s, o: True)

_ALL_PREDICATES = {_Holding, _On, _HandEmpty}
_ALL_OBJECTS = [_block0, _block1, _robot]


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


def _always_true(_s, _m, _o, _p):
    return True


def _always_false(_s, _m, _o, _p):
    return False


_Pick = ParameterizedOption(
    "Pick",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_always_true,
    terminal=_always_false,
)

_Place = ParameterizedOption(
    "Place",
    types=[_block_type, _block_type],
    params_space=Box(low=np.array([0.0, 0.0], dtype=np.float32),
                     high=np.array([1.0, 1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_always_true,
    terminal=_always_false,
)

_Wait = ParameterizedOption(
    "Wait",
    types=[_robot_type],
    params_space=Box(low=np.array([], dtype=np.float32),
                     high=np.array([], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_always_true,
    terminal=_always_false,
)

_ALL_OPTIONS = {_Pick, _Place, _Wait}


def _make_state(overrides=None):
    """Create a simple state with default feature values."""
    data = {
        _block0: np.array([0.1, 0.2, 0.0], dtype=np.float32),
        _block1: np.array([0.5, 0.6, 0.0], dtype=np.float32),
        _robot: np.array([0.0, 0.0], dtype=np.float32),
    }
    if overrides:
        for obj, vals in overrides.items():
            data[obj] = np.array(vals, dtype=np.float32)
    return State(data)


def _make_approach():
    """Create an AgentModelBasedApproach with mock config and option model."""
    state = _make_state()
    goal = {GroundAtom(_On, [_block0, _block1])}
    task = Task(state, goal)

    utils.reset_config({
        "env": "cover",
        "approach": "agent_bilevel",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "option_model_name": "oracle",
        "seed": 42,
        "agent_bilevel_max_samples_per_step": 10,
        "agent_bilevel_check_subgoals": True,
    })

    mock_option_model = MagicMock()
    approach = AgentModelBasedApproach(
        initial_predicates=_ALL_PREDICATES,
        initial_options=_ALL_OPTIONS,
        types={_block_type, _robot_type},
        action_space=Box(low=-1, high=1, shape=(1, )),
        train_tasks=[task],
        option_model=mock_option_model,
    )
    return approach, mock_option_model, task


# ---------------------------------------------------------------------------
# Tests: _parse_subgoal_annotations
# ---------------------------------------------------------------------------


class TestParseSubgoalAnnotations:
    """Tests for plan text subgoal parsing."""

    def test_basic_subgoals(self):
        """Test basic subgoals."""
        approach, _, _ = _make_approach()
        text = ("Pick(block0:block) -> {Holding(block0:block)}\n"
                "Place(block0:block, block1:block) -> "
                "{On(block0:block, block1:block)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 2
        # First step: Holding(block0)
        assert result[0] is not None
        pos, neg = result[0]
        assert GroundAtom(_Holding, [_block0]) in pos
        assert len(neg) == 0
        # Second step: On(block0, block1)
        assert result[1] is not None
        pos2, neg2 = result[1]
        assert GroundAtom(_On, [_block0, _block1]) in pos2
        assert len(neg2) == 0

    def test_no_subgoals(self):
        """Test no subgoals."""
        approach, _, _ = _make_approach()
        text = ("Pick(block0:block)\n"
                "Place(block0:block, block1:block)\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 2
        assert result[0] is None
        assert result[1] is None

    def test_mixed_subgoals(self):
        """Some lines have subgoals, some don't."""
        approach, _, _ = _make_approach()
        text = ("Pick(block0:block) -> {Holding(block0:block)}\n"
                "Wait(robot0:robot)\n"
                "Place(block0:block, block1:block) -> "
                "{On(block0:block, block1:block)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 3
        assert result[0] is not None
        assert result[1] is None  # Wait has no subgoal
        assert result[2] is not None

    def test_multiple_atoms_in_subgoal(self):
        """Test multiple atoms in subgoal."""
        approach, _, _ = _make_approach()
        text = (
            "Place(block0:block, block1:block) "
            "-> {On(block0:block, block1:block), HandEmpty(robot0:robot)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is not None
        pos, neg = result[0]
        assert len(pos) == 2
        assert len(neg) == 0
        assert GroundAtom(_On, [_block0, _block1]) in pos
        assert GroundAtom(_HandEmpty, [_robot]) in pos

    def test_unknown_predicate_skipped(self):
        """Test unknown predicate skipped."""
        approach, _, _ = _make_approach()
        text = "Pick(block0:block) -> {FakePred(block0:block)}\n"
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is None  # FakePred unrecognized, no valid atoms

    def test_unknown_object_skipped(self):
        """Test unknown object skipped."""
        approach, _, _ = _make_approach()
        text = "Pick(block0:block) -> {Holding(block99:block)}\n"
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is None  # block99 doesn't exist

    def test_arity_mismatch_skipped(self):
        """Test arity mismatch skipped."""
        approach, _, _ = _make_approach()
        # Holding expects 1 arg, giving 2
        text = "Pick(block0:block) -> {Holding(block0:block, block1:block)}\n"
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is None

    def test_typed_object_refs_in_subgoals(self):
        """Agent outputs obj:type in subgoal atoms — should still parse."""
        approach, _, _ = _make_approach()
        text = ("Pick(block0:block) -> {Holding(block0:block)}\n"
                "Place(block0:block, block1:block) "
                "-> {On(block0:block, block1:block)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 2
        assert result[0] is not None
        pos, _ = result[0]
        assert GroundAtom(_Holding, [_block0]) in pos
        assert result[1] is not None
        pos2, _ = result[1]
        assert GroundAtom(_On, [_block0, _block1]) in pos2

    def test_numbered_prefix_subgoals(self):
        """Agent numbers the lines (0:, 1:) — annotations must still align.

        Mirrors a real failure: the agent mirrored the numbered sketch
        format shown in logs, embedding it between prose, and the
        numbered prefix made every line parse as a non-option line so
        the annotation list came back empty/misaligned.
        """
        approach, _, _ = _make_approach()
        text = ("Some analysis the agent wrote first.\n"
                "  0: Pick(block0:block) -> {Holding(block0:block)}\n"
                "  1: Place(block0:block, block1:block) "
                "-> {On(block0:block, block1:block)}\n"
                "Rationale: ...\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 2
        assert result[0] is not None
        pos, _ = result[0]
        assert GroundAtom(_Holding, [_block0]) in pos
        assert result[1] is not None
        pos2, _ = result[1]
        assert GroundAtom(_On, [_block0, _block1]) in pos2

    def test_preamble_ignored(self):
        """Non-option lines should be ignored."""
        approach, _, _ = _make_approach()
        text = ("Here is my analysis:\n"
                "I think we should pick block0 first.\n"
                "\n"
                "Pick(block0:block) -> {Holding(block0:block)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is not None

    def test_whitespace_in_atoms(self):
        """Spaces around commas in atom arguments."""
        approach, _, _ = _make_approach()
        text = ("Place(block0:block, block1:block) -> "
                "{ On( block0:block , block1:block ) }\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is not None
        pos, _ = result[0]
        assert GroundAtom(_On, [_block0, _block1]) in pos

    def test_not_atoms_in_subgoals(self):
        """Test NOT prefix for negative target atoms."""
        approach, _, _ = _make_approach()
        text = (
            "Wait(robot0:robot) -> "
            "{Holding(block0:block), NOT On(block0:block, block1:block)}\n")
        result = approach._parse_subgoal_annotations(text, _ALL_PREDICATES,
                                                     _ALL_OBJECTS)

        assert len(result) == 1
        assert result[0] is not None
        pos, neg = result[0]
        assert GroundAtom(_Holding, [_block0]) in pos
        assert GroundAtom(_On, [_block0, _block1]) in neg


# ---------------------------------------------------------------------------
# Tests: check_wait_target_atoms
# ---------------------------------------------------------------------------


class TestCheckWaitTargetAtoms:
    """Tests that Wait terminates on target atoms, not noisy changes."""

    def test_no_targets_returns_none(self):
        """No targets in memory -> returns None (fall back to any-change)."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        # No targets in memory
        state = _make_state({_block0: [0.0, 0.0, 0.0]})
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        result = utils.check_wait_target_atoms(opt, state, abstract_fn)
        assert result is None

    def test_positive_target_met(self):
        """Wait terminates when positive target atom holds."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        target_atom = GroundAtom(_Holding, [_block0])
        opt.memory["wait_target_atoms"] = {target_atom}

        # State where Holding(block0) is true (held > 0.5)
        state_held = _make_state({_block0: [0.0, 0.0, 1.0]})
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        assert utils.check_wait_target_atoms(opt, state_held, abstract_fn) \
            is True

    def test_positive_target_not_met(self):
        """Wait does NOT terminate when target atom doesn't hold yet."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        target_atom = GroundAtom(_Holding, [_block0])
        opt.memory["wait_target_atoms"] = {target_atom}

        # State where Holding(block0) is false (held <= 0.5)
        state_not_held = _make_state({_block0: [0.0, 0.0, 0.0]})
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        assert utils.check_wait_target_atoms(opt, state_not_held,
                                             abstract_fn) is False

    def test_noisy_atom_change_ignored_with_targets(self):
        """Wait ignores noisy atom changes when specific targets are set.

        This is the key test: if the Wait is parameterized with a target
        atom (e.g. Holding(block0)), it should NOT terminate when a
        different atom changes (e.g. On(block0, block1)).
        """
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        # Only waiting for Holding(block0)
        target_atom = GroundAtom(_Holding, [_block0])
        opt.memory["wait_target_atoms"] = {target_atom}

        # State where On(block0, block1) is true (noisy change) but
        # Holding(block0) is still false
        state_noisy = _make_state({
            _block0: [0.5, 0.0, 0.0],
            _block1: [0.5, 0.0, 0.0]
        })
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        atoms = abstract_fn(state_noisy)
        # On is true (positions are close), but Holding is false
        assert GroundAtom(_On, [_block0, _block1]) in atoms
        assert GroundAtom(_Holding, [_block0]) not in atoms

        # Wait should NOT terminate (target not met, despite On changing)
        assert utils.check_wait_target_atoms(opt, state_noisy,
                                             abstract_fn) is False

    def test_negative_target_met(self):
        """Wait terminates when negative target atom is false."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        neg_atom = GroundAtom(_On, [_block0, _block1])
        opt.memory["wait_target_neg_atoms"] = {neg_atom}

        # State where On(block0, block1) is false (positions far apart)
        state = _make_state({
            _block0: [0.0, 0.0, 0.0],
            _block1: [5.0, 0.0, 0.0]
        })
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        assert utils.check_wait_target_atoms(opt, state, abstract_fn) is True

    def test_negative_target_not_met(self):
        """Wait does NOT terminate when negative target atom is still true."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        neg_atom = GroundAtom(_On, [_block0, _block1])
        opt.memory["wait_target_neg_atoms"] = {neg_atom}

        # State where On(block0, block1) is true (positions close)
        state = _make_state({
            _block0: [0.5, 0.0, 0.0],
            _block1: [0.5, 0.0, 0.0]
        })
        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)
        assert utils.check_wait_target_atoms(opt, state, abstract_fn) is False

    def test_mixed_positive_and_negative_targets(self):
        """Both positive and negative targets must be satisfied."""
        opt = _Wait.ground([_robot], np.array([], dtype=np.float32))
        opt.memory["wait_target_atoms"] = {GroundAtom(_Holding, [_block0])}
        opt.memory["wait_target_neg_atoms"] = {
            GroundAtom(_On, [_block0, _block1])
        }

        abstract_fn = lambda s: utils.abstract(s, _ALL_PREDICATES)

        # Only positive met (Holding true, On still true)
        state1 = _make_state({
            _block0: [0.5, 0.0, 1.0],
            _block1: [0.5, 0.0, 0.0]
        })
        assert utils.check_wait_target_atoms(opt, state1, abstract_fn) is False

        # Only negative met (On false, Holding false)
        state2 = _make_state({
            _block0: [0.0, 0.0, 0.0],
            _block1: [5.0, 0.0, 0.0]
        })
        assert utils.check_wait_target_atoms(opt, state2, abstract_fn) is False

        # Both met (Holding true, On false)
        state3 = _make_state({
            _block0: [0.0, 0.0, 1.0],
            _block1: [5.0, 0.0, 0.0]
        })
        assert utils.check_wait_target_atoms(opt, state3, abstract_fn) is True


# ---------------------------------------------------------------------------
# Tests: parse_wait_target_annotations and strip_wait_annotations
# ---------------------------------------------------------------------------


class TestWaitTargetParsing:
    """Tests for parse_wait_target_annotations and strip_wait_annotations."""

    def test_parse_positive_target(self):
        """Parse a positive target atom."""
        line = "Wait(robot0:robot) -> {Holding(block0:block)}"
        pos, neg = utils.parse_wait_target_annotations(line, _ALL_PREDICATES,
                                                       _ALL_OBJECTS)
        assert GroundAtom(_Holding, [_block0]) in pos
        assert len(neg) == 0

    def test_parse_negative_target(self):
        """Parse a NOT-prefixed target atom."""
        line = "Wait(robot0:robot) -> {NOT On(block0:block, block1:block)}"
        pos, neg = utils.parse_wait_target_annotations(line, _ALL_PREDICATES,
                                                       _ALL_OBJECTS)
        assert len(pos) == 0
        assert GroundAtom(_On, [_block0, _block1]) in neg

    def test_parse_mixed_targets(self):
        """Parse both positive and negative target atoms."""
        line = ("Wait(robot0:robot) -> "
                "{Holding(block0:block), NOT On(block0:block, block1:block)}")
        pos, neg = utils.parse_wait_target_annotations(line, _ALL_PREDICATES,
                                                       _ALL_OBJECTS)
        assert GroundAtom(_Holding, [_block0]) in pos
        assert GroundAtom(_On, [_block0, _block1]) in neg

    def test_parse_no_annotation(self):
        """Line without -> returns empty sets."""
        line = "Wait(robot0:robot)[]"
        pos, neg = utils.parse_wait_target_annotations(line, _ALL_PREDICATES,
                                                       _ALL_OBJECTS)
        assert len(pos) == 0
        assert len(neg) == 0

    def test_strip_annotations(self):
        """strip_wait_annotations removes -> {...} suffixes."""
        text = ("Pick(block0:block)[0.5]\n"
                "Wait(robot0:robot)[] -> {Holding(block0:block)}\n"
                "Place(block0:block, block1:block)[0.1, 0.2]\n")
        stripped = utils.strip_wait_annotations(text)
        assert "-> {" not in stripped
        assert "Pick(block0:block)[0.5]" in stripped
        assert "Wait(robot0:robot)[]" in stripped
        assert "Place(block0:block, block1:block)[0.1, 0.2]" in stripped


# ---------------------------------------------------------------------------
# Tests: _refine_sketch
# ---------------------------------------------------------------------------


class TestRefineSketch:
    """Tests for backtracking refinement search."""

    def test_empty_sketch(self):
        """Test empty sketch."""
        approach, _, task = _make_approach()
        plan, success = approach._refine_sketch(task, [], timeout=5.0)
        assert plan == []
        assert success is False

    def test_single_step_no_params(self):
        """Option with empty params_space — should succeed in 1 try."""
        approach, mock_om, task = _make_approach()

        # Option model: Wait always succeeds, goal holds after
        goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (goal_state, 5)

        sketch = [
            _SketchStep(option=_Wait, objects=[_robot], subgoal_atoms=None)
        ]
        plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        assert success is True
        assert len(plan) == 1
        assert plan[0].name == "Wait"

    def test_single_step_with_params_success(self):
        """Option with params — should find working params via sampling."""
        approach, mock_om, task = _make_approach()

        goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (goal_state, 3)

        sketch = [
            _SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None)
        ]
        plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        assert success is True
        assert len(plan) == 1

    def test_subgoal_check_pass(self):
        """Subgoal atoms hold after execution."""
        approach, mock_om, task = _make_approach()

        # After Pick, Holding(block0) should hold — set held=1
        held_state = _make_state({_block0: [0.1, 0.2, 1.0]})
        # After Place, On(block0, block1) — set x close
        goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})

        mock_om.get_next_state_and_num_actions.side_effect = [
            (held_state, 3),
            (goal_state, 3),
        ]

        sketch = [
            _SketchStep(option=_Pick,
                        objects=[_block0],
                        subgoal_atoms={GroundAtom(_Holding, [_block0])}),
            _SketchStep(option=_Place,
                        objects=[_block0, _block1],
                        subgoal_atoms={GroundAtom(_On, [_block0, _block1])}),
        ]
        plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        assert success is True
        assert len(plan) == 2

    def test_subgoal_check_fail_triggers_resample(self):
        """Subgoal atoms don't hold — should resample params."""
        approach, mock_om, task = _make_approach()

        # Holding never holds (held=0) — subgoal always fails
        bad_state = _make_state({_block0: [0.1, 0.2, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (bad_state, 3)

        sketch = [
            _SketchStep(option=_Pick,
                        objects=[_block0],
                        subgoal_atoms={GroundAtom(_Holding, [_block0])}),
        ]
        _plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        # Should exhaust all samples and fail
        assert success is False
        # Option model called max_samples times (10)
        assert mock_om.get_next_state_and_num_actions.call_count == 10

    def test_backtracking_across_steps(self):
        """Step 2 fails, causing step 1 to be re-sampled."""
        approach, mock_om, task = _make_approach()
        utils.reset_config({
            "env": "cover",
            "approach": "agent_bilevel",
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "seed": 42,
            "agent_bilevel_max_samples_per_step": 3,
            "agent_bilevel_check_subgoals": False,
        })

        call_count = 0
        goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})
        noop_state = _make_state()

        def side_effect(_state, option):
            nonlocal call_count
            call_count += 1
            if option.name == "Pick":
                return (noop_state, 3)  # Pick always succeeds
            # Place: succeed only on the last attempt
            if call_count >= 8:
                return (goal_state, 3)
            return (noop_state, 0)  # fail (noop)

        mock_om.get_next_state_and_num_actions.side_effect = side_effect

        sketch = [
            _SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None),
            _SketchStep(option=_Place,
                        objects=[_block0, _block1],
                        subgoal_atoms=None),
        ]
        plan, success = approach._refine_sketch(task, sketch, timeout=10.0)

        # Should have backtracked and eventually succeeded
        assert success is True
        assert len(plan) == 2
        assert call_count >= 4  # at least one backtrack cycle

    def test_not_initiable_triggers_resample(self):
        """Option not initiable in current state — resample."""
        approach, mock_om, task = _make_approach()
        utils.reset_config({
            "env": "cover",
            "approach": "agent_bilevel",
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "seed": 42,
            "agent_bilevel_max_samples_per_step": 3,
        })

        # Create an option that is never initiable
        not_initiable = ParameterizedOption(
            "Pick",
            types=[_block_type],
            params_space=Box(low=np.array([0.0], dtype=np.float32),
                             high=np.array([1.0], dtype=np.float32)),
            policy=_noop_policy,
            initiable=_always_false,
            terminal=_always_false,
        )

        sketch = [
            _SketchStep(option=not_initiable,
                        objects=[_block0],
                        subgoal_atoms=None)
        ]
        _plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        assert success is False
        # Option model never called since initiable is always False
        mock_om.get_next_state_and_num_actions.assert_not_called()

    def test_goal_check_on_final_step(self):
        """Final step must satisfy the task goal even without subgoals."""
        approach, mock_om, task = _make_approach()
        utils.reset_config({
            "env": "cover",
            "approach": "agent_bilevel",
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "seed": 42,
            "agent_bilevel_max_samples_per_step": 5,
            "agent_bilevel_check_subgoals": False,
        })

        # State that doesn't satisfy goal On(block0, block1)
        bad_state = _make_state({_block0: [0.9, 0.2, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (bad_state, 3)

        sketch = [
            _SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None)
        ]
        _plan, success = approach._refine_sketch(task, sketch, timeout=5.0)

        # Goal never holds → exhausts samples
        assert success is False


# ---------------------------------------------------------------------------
# Tests: _query_agent_for_plan_sketch (with mocked agent)
# ---------------------------------------------------------------------------


class TestQueryAgentForPlanSketch:
    """Tests for end-to-end sketch extraction from mock agent responses."""

    def _mock_responses(self, plan_text):
        """Build mock agent response list containing plan_text."""
        return [
            {
                "type": "assistant",
                "content": [{
                    "type": "text",
                    "text": plan_text
                }],
            },
        ]

    def test_basic_sketch_extraction(self):
        """Test basic sketch extraction."""
        approach, _, task = _make_approach()

        plan_text = ("Pick(block0:block) -> {Holding(block0:block)}\n"
                     "Place(block0:block, block1:block) -> "
                     "{On(block0:block, block1:block)}\n")

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 2
        assert sketch[0].option.name == "Pick"
        assert list(sketch[0].objects) == [_block0]
        assert sketch[0].subgoal_atoms is not None
        assert GroundAtom(_Holding, [_block0]) in sketch[0].subgoal_atoms

        assert sketch[1].option.name == "Place"
        assert list(sketch[1].objects) == [_block0, _block1]
        assert sketch[1].subgoal_atoms is not None

    def test_sketch_without_subgoals(self):
        """Test sketch without subgoals."""
        approach, _, task = _make_approach()

        plan_text = ("Pick(block0:block)\n"
                     "Place(block0:block, block1:block)\n")

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 2
        assert sketch[0].subgoal_atoms is None
        assert sketch[1].subgoal_atoms is None

    def test_sketch_with_code_fences(self):
        """Test sketch with code fences."""
        approach, _, task = _make_approach()

        plan_text = ("Here is the plan:\n"
                     "```\n"
                     "Pick(block0:block) -> {Holding(block0:block)}\n"
                     "Place(block0:block, block1:block)\n"
                     "```\n")

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 2

    def test_sketch_with_preamble(self):
        """Agent includes analysis text before the plan."""
        approach, _, task = _make_approach()

        plan_text = (
            "After inspecting the environment, I found block0 and block1.\n"
            "The goal is to place block0 on block1.\n"
            "\n"
            "Pick(block0:block)\n"
            "Place(block0:block, block1:block)\n")

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 2

    def test_sketch_with_wait(self):
        """Test sketch with wait."""
        approach, _, task = _make_approach()

        plan_text = ("Pick(block0:block) -> {Holding(block0:block)}\n"
                     "Wait(robot0:robot)\n"
                     "Place(block0:block, block1:block) -> "
                     "{On(block0:block, block1:block)}\n")

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 3
        assert sketch[0].option.name == "Pick"
        assert sketch[1].option.name == "Wait"
        assert sketch[1].subgoal_atoms is None
        assert sketch[2].option.name == "Place"

    def test_empty_response_raises(self):
        """Agent returns no text → ApproachFailure."""
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=[{
                              "type": "result",
                              "content": []
                          }]):
            with pytest.raises(ApproachFailure, match="empty plan text"):
                approach._query_agent_for_plan_sketch(task)

    def test_no_valid_options_raises(self):
        """Agent returns text with no valid option names → ApproachFailure."""
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()

        plan_text = "I don't know what to do.\nSorry!\n"

        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=self._mock_responses(plan_text)):
            with pytest.raises(ApproachFailure, match="Parsed empty"):
                approach._query_agent_for_plan_sketch(task)

    def test_sketch_from_file(self):
        """Load sketch from a saved text file via CFG option."""
        approach, _, task = _make_approach()
        sketch_path = os.path.join(_TEST_DATA_DIR, "simple_plan_sketch.txt")

        utils.reset_config({
            "env": "cover",
            "approach": "agent_bilevel",
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "seed": 42,
            "agent_bilevel_plan_sketch_file": sketch_path,
        })

        sketch = approach._query_agent_for_plan_sketch(task)

        assert len(sketch) == 2
        assert sketch[0].option.name == "Pick"
        assert list(sketch[0].objects) == [_block0]
        assert sketch[0].subgoal_atoms is not None
        assert GroundAtom(_Holding, [_block0]) in sketch[0].subgoal_atoms
        assert sketch[1].option.name == "Place"
        assert list(sketch[1].objects) == [_block0, _block1]
        assert sketch[1].subgoal_atoms is not None
        assert GroundAtom(_On, [_block0, _block1]) in sketch[1].subgoal_atoms


# ---------------------------------------------------------------------------
# Tests: _sample_params
# ---------------------------------------------------------------------------


class TestValidatePlanForward:
    """Tests for ``plan_execution.validate_plan_forward``.

    Covers the test-time forward validator that's the entire reason the
    synthesis tool can catch refinement-passes/validation-fails
    regressions.
    """

    def _grounded(self, option, objects, params=None):
        if params is None:
            params = np.zeros(option.params_space.shape[0], dtype=np.float32)
        return option.ground(list(objects), np.asarray(params,
                                                       dtype=np.float32))

    def test_goal_reached_returns_success(self):
        """Plan that reaches the goal — validator passes, no diagnosis."""
        from predicators.agent_sdk import plan_execution
        _, mock_om, task = _make_approach()
        # Final post-state satisfies the goal (On(block0, block1)).
        goal_state = _make_state({_block0: [0.55, 0.6, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (goal_state, 3)

        plan = [self._grounded(_Pick, [_block0], [0.5])]
        ok, reason = plan_execution.validate_plan_forward(
            task, plan, mock_om, predicates=_ALL_PREDICATES)
        assert ok is True
        assert reason == ""

    def test_goal_not_reached_diagnosis_names_missing_atoms(self):
        """Plan terminates but goal isn't satisfied — diagnosis names the
        missing atom set, not a generic 'validation failed'."""
        from predicators.agent_sdk import plan_execution
        _, mock_om, task = _make_approach()
        # Post-state doesn't satisfy On(block0, block1).
        bad_state = _make_state({_block0: [0.1, 0.2, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (bad_state, 3)

        plan = [self._grounded(_Pick, [_block0], [0.5])]
        ok, reason = plan_execution.validate_plan_forward(
            task, plan, mock_om, predicates=_ALL_PREDICATES)
        assert ok is False
        assert "goal not reached" in reason
        assert "On(block0:block, block1:block)" in reason

    def test_subgoal_divergence_logged_when_sketch_provided(self, caplog):
        """When the sketch is passed in, per-step subgoal divergence is logged
        with the missing atom — this is the diagnostic the synthesis agent
        needs to see *which* step's predicate is spurious."""
        import logging as _logging

        from predicators.agent_sdk import plan_execution
        _, mock_om, task = _make_approach()
        # Post-state never establishes Holding(block0). Goal is also
        # missing — but the subgoal log should fire first.
        bad_state = _make_state({_block0: [0.1, 0.2, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (bad_state, 3)

        plan = [self._grounded(_Pick, [_block0], [0.5])]
        sketch = [
            _SketchStep(option=_Pick,
                        objects=[_block0],
                        subgoal_atoms={GroundAtom(_Holding, [_block0])})
        ]
        with caplog.at_level(_logging.INFO):
            ok, _ = plan_execution.validate_plan_forward(
                task,
                plan,
                mock_om,
                predicates=_ALL_PREDICATES,
                sketch=sketch,
                run_id="test_run",
            )
        assert ok is False
        # Subgoal divergence log mentions the missing atom and the step.
        assert any("subgoal divergence at step 0" in r.message
                   and "Holding(block0:block)" in r.message
                   for r in caplog.records)

    def test_option_failure_diagnosis_names_step(self):
        """When the option model returns 0 actions (option execution failed),
        the diagnosis identifies the failing step and surfaces the option
        model's last_execution_failure."""
        from predicators.agent_sdk import plan_execution
        _, mock_om, task = _make_approach()
        # Simulate option failure: 0 actions, with a diagnostic message
        # recorded on the option model.
        mock_om.get_next_state_and_num_actions.return_value = (_make_state(),
                                                               0)
        mock_om.last_execution_failure = "IK timed out at waypoint 3"

        plan = [self._grounded(_Pick, [_block0], [0.5])]
        ok, reason = plan_execution.validate_plan_forward(
            task, plan, mock_om, predicates=_ALL_PREDICATES)
        assert ok is False
        assert "option execution failed at step 0" in reason
        assert "Pick(block0)" in reason
        assert "IK timed out at waypoint 3" in reason

    def test_empty_plan_with_goal_already_satisfied(self):
        """Empty plan + init satisfies goal → success."""
        from predicators.agent_sdk import plan_execution

        # Goal trivially holds when block0 is already on block1.
        init = _make_state({_block0: [0.55, 0.6, 0.0]})
        task = Task(init, {GroundAtom(_On, [_block0, _block1])})
        mock_om = MagicMock()
        ok, reason = plan_execution.validate_plan_forward(
            task, [], mock_om, predicates=_ALL_PREDICATES)
        assert ok is True
        assert reason == ""

    def test_empty_plan_with_unmet_goal(self):
        """Empty plan + init does NOT satisfy goal → failure with explanatory
        diagnosis."""
        from predicators.agent_sdk import plan_execution
        _, _, task = _make_approach()  # init does not satisfy goal
        mock_om = MagicMock()
        ok, reason = plan_execution.validate_plan_forward(
            task, [], mock_om, predicates=_ALL_PREDICATES)
        assert ok is False
        assert "init state does not satisfy goal" in reason

    def test_sketch_length_mismatch_ignored_gracefully(self):
        """Mismatched sketch length — validator should warn and fall back to
        goal-only checking rather than crash."""
        from predicators.agent_sdk import plan_execution
        _, mock_om, task = _make_approach()
        goal_state = _make_state({_block0: [0.55, 0.6, 0.0]})
        mock_om.get_next_state_and_num_actions.return_value = (goal_state, 3)

        plan = [self._grounded(_Pick, [_block0], [0.5])]
        # Sketch length 2, plan length 1.
        sketch = [
            _SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None),
            _SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None),
        ]
        ok, _ = plan_execution.validate_plan_forward(
            task,
            plan,
            mock_om,
            predicates=_ALL_PREDICATES,
            sketch=sketch,
        )
        # Validation still runs to completion against the goal.
        assert ok is True


class TestSampleParams:
    """TestSampleParams class."""

    def test_empty_params_space(self):
        """Test empty params space."""
        approach, _, _ = _make_approach()
        rng = np.random.default_rng(0)
        params = approach._sample_params(_Wait, _make_state(), rng)
        assert params.shape == (0, )
        assert params.dtype == np.float32

    def test_params_within_bounds(self):
        """Test params within bounds."""
        approach, _, _ = _make_approach()
        rng = np.random.default_rng(0)
        for _ in range(100):
            params = approach._sample_params(_Place, _make_state(), rng)
            assert params.shape == (2, )
            assert np.all(params >= 0.0)
            assert np.all(params <= 1.0)
            assert params.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: class metadata
# ---------------------------------------------------------------------------


def test_get_name():
    """Test get name."""
    assert AgentModelBasedApproach.get_name() == "agent_bilevel"


# ---------------------------------------------------------------------------
# Tests: closed-loop execution replanning (subgoal_annotations monitor +
# _maybe_replan_from_divergence / _replan_suffix)
# ---------------------------------------------------------------------------

_PickDone = ParameterizedOption(
    "Pick",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_always_true,
    terminal=_always_true,
)

_PlaceDone = ParameterizedOption(
    "Place",
    types=[_block_type, _block_type],
    params_space=Box(low=np.array([0.0, 0.0], dtype=np.float32),
                     high=np.array([1.0, 1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_always_true,
    terminal=_always_true,
)


def _make_two_step_plan(first_subgoals):
    """Plan [Pick, Place] whose first step is annotated with first_subgoals."""
    plan = [
        _PickDone.ground([_block0], np.array([0.5], dtype=np.float32)),
        _PlaceDone.ground([_block0, _block1],
                          np.array([0.5, 0.5], dtype=np.float32)),
    ]
    sketch = [
        _SketchStep(_PickDone, [_block0], first_subgoals),
        _SketchStep(_PlaceDone, [_block0, _block1], None),
    ]
    return plan, sketch


def _enable_replanning(approach, budget):
    """Turn on closed-loop execution and start a fresh episode."""
    utils.update_config({
        "agent_bilevel_max_execution_replans": budget,
        "execution_monitor": "subgoal_annotations",
    })
    approach.reset_for_new_episode()


def _make_monitor(approach):
    """Create the monitor and sync it with the approach, CogMan-style."""
    from predicators.execution_monitoring import create_execution_monitor
    monitor = create_execution_monitor("subgoal_annotations")
    monitor.update_approach_info(approach.get_execution_monitoring_info())
    return monitor


def _sync(monitor, approach):
    """Mimic CogMan pushing fresh approach info to the monitor."""
    monitor.update_approach_info(approach.get_execution_monitoring_info())


class TestExecutionReplanning:
    """Tests for closed-loop execution through the cogman monitor flow."""

    def test_open_loop_when_disabled(self):
        """With the flag at 0 (default), no monitoring info is exported and
        divergence is never flagged."""
        approach, _, _ = _make_approach()
        holding = {GroundAtom(_Holding, [_block0])}
        plan, sketch = _make_two_step_plan(holding)
        policy = approach._plan_to_policy(plan, sketch=sketch)
        assert not approach.get_execution_monitoring_info()
        state = _make_state()  # block0 not held: subgoal would fail
        monitor = _make_monitor(approach)
        assert not monitor.step(state)
        policy(state)  # starts Pick
        policy(state)  # Pick terminal -> starts Place without any check

    def test_monitor_silent_when_subgoals_hold(self):
        """Subgoals satisfied at the boundary: no replan is suggested."""
        approach, _, _ = _make_approach()
        _enable_replanning(approach, 2)
        holding = {GroundAtom(_Holding, [_block0])}
        plan, sketch = _make_two_step_plan(holding)
        policy = approach._plan_to_policy(plan, sketch=sketch)
        state = _make_state({_block0: [0.1, 0.2, 1.0]})  # held: subgoal ok
        monitor = _make_monitor(approach)
        # Before any option is initiated (e.g. right after a replan,
        # cogman asserts the monitor does not immediately re-fire).
        assert not monitor.step(state)
        policy(state)  # starts Pick
        _sync(monitor, approach)
        assert not monitor.step(state)  # boundary, but annotation holds
        policy(state)  # advances to Place

    def test_monitor_silent_mid_option(self):
        """A failing annotation is only checked at the option boundary."""
        approach, _, _ = _make_approach()
        _enable_replanning(approach, 2)
        holding = {GroundAtom(_Holding, [_block0])}
        # _Pick never terminates, so execution stays mid-option.
        plan = [_Pick.ground([_block0], np.array([0.5], dtype=np.float32))]
        sketch = [_SketchStep(_Pick, [_block0], holding)]
        policy = approach._plan_to_policy(plan, sketch=sketch)
        state = _make_state()  # block0 not held: subgoal fails
        policy(state)
        monitor = _make_monitor(approach)
        assert not monitor.step(state)

    def test_monitor_detects_divergence_at_boundary(self):
        """An unsatisfied annotation at the boundary suggests a replan."""
        approach, _, _ = _make_approach()
        _enable_replanning(approach, 2)
        holding = {GroundAtom(_Holding, [_block0])}
        plan, sketch = _make_two_step_plan(holding)
        policy = approach._plan_to_policy(plan, sketch=sketch)
        state = _make_state()  # block0 not held: subgoal diverges
        policy(state)  # starts Pick (terminal at every state)
        monitor = _make_monitor(approach)
        assert monitor.step(state)

    def test_suffix_replan_preferred_on_divergence(self):
        """The monitor-triggered re-solve resumes via the suffix path; no agent
        re-query."""
        approach, _, task = _make_approach()
        _enable_replanning(approach, 2)
        holding = {GroundAtom(_Holding, [_block0])}
        plan, sketch = _make_two_step_plan(holding)
        policy = approach._plan_to_policy(plan, sketch=sketch)
        state = _make_state()
        policy(state)
        monitor = _make_monitor(approach)
        assert monitor.step(state)

        # CogMan now re-invokes solve() on the current state.
        def sentinel_policy(s):
            del s  # unused
            return Action(np.full(1, 0.25, dtype=np.float32))

        approach._replan_suffix = MagicMock(return_value=sentinel_policy)
        approach._query_agent_for_plan_sketch = MagicMock()
        new_policy = approach._solve(Task(state, task.goal), timeout=10)
        assert new_policy is sentinel_policy
        approach._query_agent_for_plan_sketch.assert_not_called()
        approach._replan_suffix.assert_called_once()
        args = approach._replan_suffix.call_args.args
        assert args[0] is state  # replans from the real current state
        assert args[3] == 0  # the failed step is the annotated first step

    def test_episode_fails_when_no_suffix_validates(self):
        """Suffix path exhausted: by default the episode fails.

        A fresh sketch query would re-open the agent turn budget the
        attempt already spent, so it is opt-in
        (agent_bilevel_replan_agent_fallback).
        """
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()
        _enable_replanning(approach, 2)
        holding = {GroundAtom(_Holding, [_block0])}
        plan, sketch = _make_two_step_plan(holding)
        policy = approach._plan_to_policy(plan, sketch=sketch)
        state = _make_state()
        policy(state)
        approach._replan_suffix = MagicMock(return_value=None)
        with pytest.raises(ApproachFailure,
                           match="agent_bilevel_replan_agent_fallback"):
            approach._solve(Task(state, task.goal), timeout=10)
        approach._replan_suffix.assert_called_once()

    def test_full_resolve_when_no_suffix_validates_with_fallback(self):
        """With agent_bilevel_replan_agent_fallback, a failed suffix replan
        falls through to a fresh agent sketch."""
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()
        _enable_replanning(approach, 2)
        utils.update_config({"agent_bilevel_replan_agent_fallback": True})
        holding = {GroundAtom(_Holding, [_block0])}
        plan, sketch = _make_two_step_plan(holding)
        policy = approach._plan_to_policy(plan, sketch=sketch)
        state = _make_state()
        policy(state)
        approach._replan_suffix = MagicMock(return_value=None)
        # Reaching the fresh-sketch body raises its distinctive failure -
        # proof we fell through to a fresh agent query.
        sketch_query = MagicMock(side_effect=ApproachFailure("no sketch"))
        approach._query_agent_for_plan_sketch = sketch_query
        with patch.object(approach, '_nudge_final_submission',
                          MagicMock(return_value=None)):
            with pytest.raises(ApproachFailure, match="Bilevel solve failed"):
                approach._solve(Task(state, task.goal), timeout=10)
        sketch_query.assert_called_once()
        approach._replan_suffix.assert_called_once()

    def test_budget_shared_across_chained_replans(self):
        """Chained replans share one per-episode budget and fail fast once it
        is exhausted."""
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()
        _enable_replanning(approach, 1)
        holding = {GroundAtom(_Holding, [_block0])}
        plan, sketch = _make_two_step_plan(holding)

        def _suffix_replan(s, tsk, steps, k, t):
            del s, tsk, steps, k, t  # unused
            new_plan, new_sketch = _make_two_step_plan(holding)
            return approach._plan_to_policy(new_plan, sketch=new_sketch)

        approach._replan_suffix = MagicMock(side_effect=_suffix_replan)
        approach._query_agent_for_plan_sketch = MagicMock()
        policy = approach._plan_to_policy(plan, sketch=sketch)
        state = _make_state()
        policy(state)
        monitor = _make_monitor(approach)
        assert monitor.step(state)
        # First divergence: budget 1 -> 0, replanned policy starts.
        new_policy = approach._solve(Task(state, task.goal), timeout=10)
        new_policy(state)
        _sync(monitor, approach)
        assert monitor.step(state)
        # Second divergence: no budget left.
        with pytest.raises(ApproachFailure, match="No execution replans"):
            approach._solve(Task(state, task.goal), timeout=10)
        approach._query_agent_for_plan_sketch.assert_not_called()

    def test_reset_for_new_episode_clears_state(self):
        """A new episode refreshes the budget and clears the live status."""
        approach, _, _ = _make_approach()
        _enable_replanning(approach, 2)
        assert approach._exec_replans_left == 2
        holding = {GroundAtom(_Holding, [_block0])}
        plan, sketch = _make_two_step_plan(holding)
        approach._plan_to_policy(plan, sketch=sketch)
        assert approach.get_execution_monitoring_info()
        approach._exec_replans_left = 0
        approach.reset_for_new_episode()
        assert not approach.get_execution_monitoring_info()
        assert approach._exec_replans_left == 2

    def test_init_requires_subgoal_annotations_monitor(self):
        """Enabling the budget without the monitor is a config error."""
        _, _, task = _make_approach()
        utils.update_config({"agent_bilevel_max_execution_replans": 2})
        kwargs = dict(
            initial_predicates=_ALL_PREDICATES,
            initial_options=_ALL_OPTIONS,
            types={_block_type, _robot_type},
            action_space=Box(low=-1, high=1, shape=(1, )),
            train_tasks=[task],
            option_model=MagicMock(),
        )
        with pytest.raises(ValueError, match="subgoal_annotations"):
            AgentModelBasedApproach(**kwargs)
        utils.update_config({"execution_monitor": "subgoal_annotations"})
        AgentModelBasedApproach(**kwargs)

    def test_replan_suffix_walkback_and_validation(self):
        """_replan_suffix tries the failed step first, walks back only to the
        latest holding annotation, and forward-validates."""
        from predicators.agent_sdk import bilevel_sketch as bs
        approach, _, task = _make_approach()
        on_atom = {GroundAtom(_On, [_block0, _block1])}
        holding = {GroundAtom(_Holding, [_block0])}
        sketch = [
            _SketchStep(_PickDone, [_block0], on_atom),  # holds (x close)
            _SketchStep(_PickDone, [_block0], holding),  # does not hold
            _SketchStep(_PlaceDone, [_block0, _block1], holding),  # failed
        ]
        # block0.x=0.5 == block1.x=0.5 so On holds; held=0 so Holding fails.
        state = _make_state({_block0: [0.5, 0.2, 0.0]})
        tried = []

        def _fake_refine(tsk, suffix, remaining, attempt=0):
            del tsk, remaining, attempt  # unused
            tried.append(len(suffix))
            # Succeed only for the 2-step suffix (resume at step 1).
            if len(suffix) == 2:
                new_plan, _ = _make_two_step_plan(holding)
                return new_plan, True
            return [], False

        approach._refine_sketch = MagicMock(side_effect=_fake_refine)
        with patch.object(bs, "validate_plan_forward",
                          return_value=(True, "")):
            policy = approach._replan_suffix(state, task, sketch, 2, 10)
        assert policy is not None
        # Tried failed step (suffix len 1) first, then one step back
        # (len 2); never walked past the holding annotation at step 0.
        assert tried == [1, 2]


# ---------------------------------------------------------------------------
# Tests: scheduled-plans section in the solve/explore prompt
# ---------------------------------------------------------------------------


class TestScheduledPlansPromptSection:
    """The explore prompt shows plans already generated this cycle so the next
    request proposes a complementary plan instead of repeating the identical
    one (run_20260707_112310 emitted the same 1-step plan for both of a cycle's
    requests)."""

    @staticmethod
    def _prompt(scheduled_plans):
        from predicators.agent_sdk import sketch_prompts
        utils.reset_config({
            "env": "cover",
            "approach": "agent_bilevel",
            "seed": 42,
        })
        state = _make_state()
        task = Task(state, {GroundAtom(_On, [_block0, _block1])})
        return sketch_prompts.build_solve_prompt(
            task,
            all_predicates=_ALL_PREDICATES,
            all_options=_ALL_OPTIONS,
            scheduled_plans=scheduled_plans,
            propose_params=True,
        )

    def test_section_absent_without_scheduled_plans(self):
        """No scheduled-plans section is emitted when none were scheduled."""
        for empty in (None, []):
            prompt = self._prompt(empty)
            assert "Plans Already Scheduled This Cycle" not in prompt

    def test_section_lists_plans_and_asks_for_different_one(self):
        """Scheduled plans are listed so the agent proposes a different one."""
        plans = [
            "  0: Pick(block0)[0.5000]",
            "  0: Place(block0, block1)[0.1000, 0.2000]",
        ]
        prompt = self._prompt(plans)
        assert "## Plans Already Scheduled This Cycle" in prompt
        assert "Plan 1:\n  0: Pick(block0)[0.5000]" in prompt
        assert "Plan 2:\n  0: Place(block0, block1)[0.1000, 0.2000]" in prompt
        assert "still achieves the goal but differs meaningfully" in prompt
        # The instruction must keep the request goal-directed (this is what
        # preserves the train-solve early-stopping semantics).
        assert "repeat the best plan" in " ".join(prompt.split())


# ---------------------------------------------------------------------------
# Tests: turn-cap exhaustion handling in _solve
# ---------------------------------------------------------------------------


class TestTurnCapHandling:
    """Hitting agent_sdk_max_agent_turns_per_iteration ends the attempt with a
    best-effort submission instead of burning the sketch retries."""

    @staticmethod
    def _cap_result(subtype=None, num_turns=None):
        return {
            "type": "result",
            "subtype": subtype,
            "num_turns": num_turns,
            "total_cost_usd": 1.0,
        }

    def test_responses_hit_turn_cap(self):
        """Cap detection: subtype is authoritative, num_turns is fallback."""
        approach, _, _ = _make_approach()
        cap = approach._responses_hit_turn_cap
        assert cap([self._cap_result(subtype="error_max_turns")])
        max_turns = 50
        utils.update_config(
            {"agent_sdk_max_agent_turns_per_iteration": max_turns})
        assert cap([self._cap_result(subtype="success", num_turns=max_turns)])
        assert not cap(
            [self._cap_result(subtype="success", num_turns=max_turns - 1)])
        assert not cap([{"type": "assistant", "content": []}])
        assert not cap([])

    def test_sketch_query_records_turn_cap(self):
        """A capped session with no final text still marks the cap before the
        empty-plan-text failure propagates."""
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()
        responses = [self._cap_result(subtype="error_max_turns")]
        with patch.object(approach,
                          '_query_agent_sync',
                          return_value=responses):
            with pytest.raises(ApproachFailure, match="empty plan text"):
                approach._query_agent_for_plan_sketch(task)
        assert approach._last_sketch_query_hit_turn_cap

    def test_solve_one_query_per_attempt_on_turn_cap(self):
        """A capped attempt takes the final nudge and stops."""
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()
        query = MagicMock(
            return_value=[self._cap_result(subtype="error_max_turns")])
        nudge = MagicMock(return_value=None)
        with patch.object(approach, '_query_agent_sync', query), \
                patch.object(approach, '_nudge_final_submission', nudge):
            with pytest.raises(ApproachFailure, match="Bilevel solve failed"):
                approach._solve(task, timeout=10)
        assert query.call_count == 1  # no re-query on the same context
        nudge.assert_called_once_with()

    def test_solve_restarts_with_no_nudge_until_final_attempt(self):
        """Non-final attempts restart directly with no nudge.

        The best-effort submission nudge fires only on the FINAL
        attempt, as the ultimate fallback.
        """
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()
        utils.update_config({"agent_solve_max_attempts": 3})
        query = MagicMock(
            return_value=[self._cap_result(subtype="error_max_turns")])
        nudge = MagicMock(return_value=None)
        with patch.object(approach, '_query_agent_sync', query), \
                patch.object(approach, '_nudge_final_submission', nudge):
            with pytest.raises(ApproachFailure, match="Bilevel solve failed"):
                approach._solve(task, timeout=10)
        assert query.call_count == 3  # one full query per attempt
        nudge.assert_called_once_with()

    def test_solve_no_requery_on_non_cap_failure(self):
        """A non-cap failure (e.g. unparseable output) ends the attempt too.

        The fresh-context restart is the ONLY retry: a query that merely
        failed to submit does not buy a second full-price query on a
        context that already contains whatever went wrong.
        """
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()
        utils.update_config({"agent_solve_max_attempts": 3})
        # Well under the cap, but no plan text: a real error, not budget end.
        query = MagicMock(
            return_value=[self._cap_result(subtype="success", num_turns=5)])
        nudge = MagicMock(return_value=None)
        with patch.object(approach, '_query_agent_sync', query), \
                patch.object(approach, '_nudge_final_submission', nudge):
            with pytest.raises(ApproachFailure, match="Bilevel solve failed"):
                approach._solve(task, timeout=10)
        assert query.call_count == 3  # one per attempt, not one per query
        nudge.assert_called_once_with()

    def test_attempt_end_reason_labels_journal_outcome(self):
        """A capture-less attempt records WHY it ended.

        That reason is the one fact the next fresh-context attempt
        cannot rediscover from the transcript it no longer has.
        """
        from predicators.approaches import ApproachFailure
        approach, _, task = _make_approach()
        nudge = MagicMock(return_value=None)

        for subtype, num_turns, reason in [
            ("error_max_turns", None, "turn cap"),
            ("success", 5, "no submission"),
        ]:
            query = MagicMock(
                return_value=[self._cap_result(subtype, num_turns)])
            with patch.object(approach, '_query_agent_sync', query), \
                    patch.object(approach, '_nudge_final_submission', nudge):
                with pytest.raises(ApproachFailure, match=reason):
                    approach._solve(task, timeout=10)
            assert approach._last_attempt_end_reason == reason
            assert approach._attempt_outcome_text(
                None, None) == f"no capture ({reason})"

    def test_nudge_best_effort_flag_set_and_cleared(self):
        """The nudge exposes best-effort capture to the tools only for the
        duration of its own query."""
        approach, _, _ = _make_approach()
        seen = {}

        def _fake_query(message, **kwargs):
            del kwargs  # unused
            seen["flag"] = approach._tool_context.capture_best_effort_plan
            seen["message"] = message
            return []

        with patch.object(approach, '_query_agent_sync', _fake_query):
            policy = approach._nudge_final_submission()
        assert policy is None
        assert seen["flag"] is True
        assert "even if it does not fully reach the goal" in seen["message"]
        assert not approach._tool_context.capture_best_effort_plan

    def test_nudge_returns_captured_best_effort_plan(self):
        """The nudge consumes a captured plan into a policy even when the
        rollout did not reach the goal."""
        approach, _, _ = _make_approach()
        plan = [_Pick.ground([_block0], np.array([0.5], dtype=np.float32))]
        sketch = [_SketchStep(_Pick, [_block0], None)]

        def _fake_query(message, **kwargs):
            del message, kwargs  # unused
            # Simulate evaluate_option_plan's best-effort capture.
            assert approach._tool_context.capture_best_effort_plan
            approach._tool_context.solved_plan = plan
            approach._tool_context.solved_sketch = sketch
            approach._tool_context.solved_plan_reached_goal = False
            return []

        with patch.object(approach, '_query_agent_sync', _fake_query):
            policy = approach._nudge_final_submission()
        assert policy is not None
        assert approach._tool_context.solved_plan is None
        assert approach._tool_context.solved_plan_reached_goal is None
