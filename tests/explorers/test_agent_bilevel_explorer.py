"""Tests for AgentBilevelExplorer."""
# pylint: disable=protected-access

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.sketch_types import SketchStep
from predicators.agent_sdk.tools import ToolContext
from predicators.explorers import create_explorer
from predicators.explorers.agent_bilevel_explorer import AgentBilevelExplorer
from predicators.explorers.base_explorer import BaseExplorer
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

# ---------------------------------------------------------------------------
# Fixtures (parallel the bilevel approach tests)
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
_ALL_TYPES = {_block_type, _robot_type}


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
    data = {
        _block0: np.array([0.1, 0.2, 0.0], dtype=np.float32),
        _block1: np.array([0.5, 0.6, 0.0], dtype=np.float32),
        _robot: np.array([0.0, 0.0], dtype=np.float32),
    }
    if overrides:
        for obj, vals in overrides.items():
            data[obj] = np.array(vals, dtype=np.float32)
    return State(data)


def _make_task():
    state = _make_state()
    goal = {GroundAtom(_On, [_block0, _block1])}
    return Task(state, goal)


def _assistant_response(text: str):
    return [{
        "type": "assistant",
        "content": [{
            "type": "text",
            "text": text
        }],
    }]


def _make_explorer(option_model, query_impl):
    """Build an AgentBilevelExplorer with stubbed session + tool_context."""
    tool_context = ToolContext(
        types=_ALL_TYPES,
        predicates=_ALL_PREDICATES,
        options=_ALL_OPTIONS,
        train_tasks=[_make_task()],
        option_model=option_model,
    )
    agent_session = MagicMock()
    agent_session.query = query_impl
    agent_session.tool_names = None
    explorer = AgentBilevelExplorer(
        predicates=_ALL_PREDICATES,
        options=_ALL_OPTIONS,
        types=_ALL_TYPES,
        action_space=Box(low=-1, high=1, shape=(1, )),
        train_tasks=[_make_task()],
        max_steps_before_termination=50,
        tool_context=tool_context,
        agent_session=agent_session,
    )
    return explorer, tool_context


def _reset_config(**overrides):
    base = {
        "env": "cover",
        "approach": "agent_bilevel",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "seed": 42,
        "agent_bilevel_max_samples_per_step": 5,
        "agent_bilevel_explorer_max_samples_per_step": 5,
        "agent_bilevel_check_subgoals": True,
        "agent_bilevel_log_state": False,
        "agent_explorer_fallback_to_random": True,
        "agent_sdk_max_trajectories_in_context": 5,
    }
    base.update(overrides)
    utils.reset_config(base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_factory_registration():
    """AgentBilevelExplorer is reachable through create_explorer."""
    _reset_config()
    tool_context = ToolContext(
        types=_ALL_TYPES,
        predicates=_ALL_PREDICATES,
        options=_ALL_OPTIONS,
        train_tasks=[_make_task()],
        option_model=MagicMock(),
    )
    agent_session = MagicMock()
    explorer = create_explorer(
        "agent_bilevel",
        _ALL_PREDICATES,
        _ALL_OPTIONS,
        _ALL_TYPES,
        Box(low=-1, high=1, shape=(1, )),
        [_make_task()],
        tool_context=tool_context,
        agent_session=agent_session,
    )
    assert isinstance(explorer, BaseExplorer)
    assert isinstance(explorer, AgentBilevelExplorer)


def test_happy_path_returns_policy_and_stashes_subgoals():
    """Canned sketch → refined plan → policy and stashed subgoals."""
    _reset_config()

    goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})
    option_model = MagicMock()
    option_model.get_next_state_and_num_actions.return_value = (goal_state, 3)

    plan_text = ("Pick(block0:block)\n"
                 "Place(block0:block, block1:block) -> "
                 "{On(block0:block, block1:block)}\n")
    query = AsyncMock(return_value=_assistant_response(plan_text))

    explorer, tool_context = _make_explorer(option_model, query)
    policy, term_fn = explorer._get_exploration_strategy(0, timeout=5)

    assert callable(policy)
    assert term_fn(_make_state()) is False
    assert tool_context.last_sketch_subgoals is not None
    assert len(tool_context.last_sketch_subgoals) == 2
    # Second step's positive subgoal should be {On(block0, block1)}.
    pos2, _neg2 = tool_context.last_sketch_subgoals[1]
    assert pos2 == {GroundAtom(_On, [_block0, _block1])}
    assert tool_context.last_sketch_options == [
        ("Pick", ["block0"]),
        ("Place", ["block0", "block1"]),
    ]
    assert query.await_count == 1


def test_wait_memory_injection_on_refine():
    """Wait step with subgoal should have wait_target_atoms injected."""
    _reset_config()

    captured: list = []

    def side_effect(_state, option):
        captured.append(option)
        return (_make_state({_block0: [0.5, 0.6, 0.0]}), 3)

    option_model = MagicMock()
    option_model.get_next_state_and_num_actions.side_effect = side_effect

    plan_text = ("Wait(robot0:robot) -> {On(block0:block, block1:block)}\n")
    query = AsyncMock(return_value=_assistant_response(plan_text))
    explorer, _ = _make_explorer(option_model, query)

    explorer._get_exploration_strategy(0, timeout=5)
    assert captured, "option_model was not invoked"
    wait_opt = captured[0]
    assert wait_opt.name == "Wait"
    assert "wait_target_atoms" in wait_opt.memory
    assert wait_opt.memory["wait_target_atoms"] == {
        GroundAtom(_On, [_block0, _block1])
    }


def test_plan_truncates_at_deepest_subgoal_failure_after_backtracking():
    """Regression: explorer returns the prefix up to (and including) the
    deepest step whose subgoal backtracking couldn't satisfy.

    Reproduces the boil-task bug: the agent sketches ``Pick → Wait(Holding)
    → Place`` and the mental model's Wait does NOT produce ``Holding``.
    Backtracking runs normally — it retries Pick with different params
    and re-runs Wait each time — but since the mental model simply can't
    produce Holding under any params, Wait's subgoal keeps failing.
    After exhaustion, the explorer returns ``[Pick, Wait]`` with the last
    grounded attempts. Place is NEVER executed because refinement never
    gets past Wait.
    """
    _reset_config()

    # Mental model post-state: Holding(block0) NEVER holds (held=0).
    no_holding_state = _make_state({_block0: [0.1, 0.2, 0.0]})
    option_model = MagicMock()
    option_model.get_next_state_and_num_actions.return_value = (
        no_holding_state, 3)

    plan_text = ("Pick(block0:block)\n"
                 "Wait(robot0:robot) -> {Holding(block0:block)}\n"
                 "Place(block0:block, block1:block) -> "
                 "{On(block0:block, block1:block)}\n")
    query = AsyncMock(return_value=_assistant_response(plan_text))
    explorer, tool_context = _make_explorer(option_model, query)

    policy, _ = explorer._get_exploration_strategy(0, timeout=5)
    assert callable(policy)

    # All three sketch steps recorded in metadata — the SKETCH is the full
    # agent output; the TRUNCATION only applies to the refined plan.
    assert tool_context.last_sketch_options == [
        ("Pick", ["block0"]),
        ("Wait", ["robot0"]),
        ("Place", ["block0", "block1"]),
    ]

    executed_names = [
        call.args[1].name
        for call in option_model.get_next_state_and_num_actions.call_args_list
    ]
    # Pick and Wait were each executed at least once (backtracking likely
    # retried Pick multiple times).
    assert "Pick" in executed_names
    assert "Wait" in executed_names
    # Place must NEVER be executed in the mental model: backtracking never
    # got past the Wait subgoal failure, so Place never reached sample_fn.
    assert "Place" not in executed_names, (
        "Place must not be executed in the mental model — refinement "
        f"should have stalled at Wait's unsatisfiable subgoal, got "
        f"{executed_names}")
    # Pick has params (5 max_samples_per_step in test config), Wait has none.
    # Each backtracking cycle runs Pick + Wait once, so we expect roughly
    # 2 * max_samples_per_step mental-model calls — confirm backtracking
    # actually exercised the upstream retries (at least 2 Picks).
    assert executed_names.count("Pick") >= 2, (
        "Backtracking should have retried Pick at least twice before "
        f"giving up, got {executed_names}")


def _make_captured(pick_params, place_params):
    """Build the (solved_plan, solved_sketch) a tool capture would stash."""
    grounded_plan = [
        _Pick.ground([_block0], np.array(pick_params, dtype=np.float32)),
        _Place.ground([_block0, _block1],
                      np.array(place_params, dtype=np.float32)),
    ]
    captured_sketch = [
        SketchStep(option=_Pick, objects=[_block0], subgoal_atoms=None),
        SketchStep(option=_Place,
                   objects=[_block0, _block1],
                   subgoal_atoms={GroundAtom(_On, [_block0, _block1])}),
    ]
    return grounded_plan, captured_sketch


def test_recovers_captured_plan_when_final_text_unparseable():
    """Agent validates a plan via evaluate_option_plan but ends in prose:

    explorer recovers the captured plan instead of falling back to
    random, and seeds the captured continuous params into refinement.
    """
    _reset_config()

    goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})
    option_model = MagicMock()
    option_model.get_next_state_and_num_actions.return_value = (goal_state, 3)

    pick_params, place_params = [0.42], [0.11, 0.22]
    grounded_plan, captured_sketch = _make_captured(pick_params, place_params)

    explorer, tool_context = _make_explorer(option_model, None)

    async def query_impl(_msg, **_kw):
        # Simulate the agent capturing a validated plan via the tool during
        # the query (set AFTER the explorer's entry-time capture clear), then
        # ending with prose that does NOT parse into a sketch.
        tool_context.solved_plan = grounded_plan
        tool_context.solved_sketch = captured_sketch
        return _assistant_response("Solved it. Plan: 1. pick 2. place. Done.")

    explorer._agent_session.query = query_impl

    policy, term_fn = explorer._get_exploration_strategy(0, timeout=5)

    # Recovered (not random fallback): subgoals/options come from the capture.
    assert callable(policy)
    assert term_fn(_make_state()) is False
    assert tool_context.last_sketch_options == [
        ("Pick", ["block0"]),
        ("Place", ["block0", "block1"]),
    ]
    # The capture was consumed (cleared) so it can't leak into a later solve.
    assert tool_context.solved_plan is None
    assert tool_context.solved_sketch is None
    # Captured params were seeded as initial_params: the option model is
    # invoked with them (Pick tries them first; Place's are pooled).
    called = [
        c.args[1]
        for c in option_model.get_next_state_and_num_actions.call_args_list
    ]
    pick_calls = [o for o in called if o.name == "Pick"]
    place_calls = [o for o in called if o.name == "Place"]
    assert pick_calls and place_calls
    np.testing.assert_allclose(pick_calls[0].params, pick_params)
    assert any(
        np.allclose(o.params, place_params) for o in place_calls), \
        "captured Place params were not seeded into refinement"


def test_captured_params_seed_info_gain_search():
    """With info-seeking ON, the recovered capture's continuous params are
    seeded as candidates in the info-gain pool (not replayed verbatim)."""
    _reset_config(agent_explorer_info_seeking=True,
                  agent_explorer_info_n_feasible_target=2,
                  agent_bilevel_explorer_max_samples_per_step=4)

    goal_state = _make_state({_block0: [0.5, 0.6, 0.0]})
    option_model = MagicMock()
    option_model.get_next_state_and_num_actions.return_value = (goal_state, 3)

    place_params = [0.33, 0.44]
    grounded_plan, captured_sketch = _make_captured([0.42], place_params)

    explorer, tool_context = _make_explorer(option_model, None)
    # Wire a trivial ensemble scorer so info-seeking engages on annotated
    # steps; constant score means the seeded candidate is chosen.
    tool_context.atom_disagreement_fn = lambda _s, _atoms: 0.0

    async def query_impl(_msg, **_kw):
        tool_context.solved_plan = grounded_plan
        tool_context.solved_sketch = captured_sketch
        return _assistant_response("Done — summary only, no sketch block.")

    explorer._agent_session.query = query_impl

    policy, _ = explorer._get_exploration_strategy(0, timeout=5)
    assert callable(policy)
    # Place is the subgoal-annotated step that info-seeking pools; its captured
    # params must appear among the candidates the pool evaluated.
    place_calls = [
        c.args[1]
        for c in option_model.get_next_state_and_num_actions.call_args_list
        if c.args[1].name == "Place"
    ]
    assert any(np.allclose(o.params, place_params) for o in place_calls), \
        "captured Place params were not seeded into the info-gain pool"


def test_fallback_when_query_fails_and_flag_on():
    """Agent raises → random options fallback when flag enabled."""
    _reset_config(agent_explorer_fallback_to_random=True)

    option_model = MagicMock()

    async def failing_query(_msg):
        raise RuntimeError("boom")

    explorer, _ = _make_explorer(option_model, failing_query)
    policy, term_fn = explorer._get_exploration_strategy(0, timeout=5)
    assert callable(policy)
    assert term_fn(_make_state()) is False


def test_fallback_disabled_raises():
    """Agent raises → RequestActPolicyFailure when fallback flag off."""
    _reset_config(agent_explorer_fallback_to_random=False)

    option_model = MagicMock()

    async def failing_query(_msg):
        raise RuntimeError("boom")

    explorer, _ = _make_explorer(option_model, failing_query)
    with pytest.raises(utils.RequestActPolicyFailure):
        explorer._get_exploration_strategy(0, timeout=5)


def test_experiment_guidance_gated_by_info_seeking():
    """Experiment guidance appears iff info-seeking is on."""
    _reset_config(agent_explorer_info_seeking=True)
    explorer, _ = _make_explorer(MagicMock(), MagicMock())
    guidance = explorer._build_experiment_guidance()  # pylint: disable=protected-access
    assert "straddle the learned model's decision boundaries" in guidance
    # Off => section absent entirely.
    _reset_config(agent_explorer_info_seeking=False)
    assert explorer._build_experiment_guidance() == ""  # pylint: disable=protected-access
