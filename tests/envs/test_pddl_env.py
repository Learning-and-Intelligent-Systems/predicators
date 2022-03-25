"""Tests for PDDLEnv."""

import numpy as np
import pytest

from predicators.src import utils
from predicators.src.envs.pddl_env import _PDDLEnv


def test_pddlenv():
    """Tests for PDDLEnv()."""
    utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})

    domain_str = """; This is a comment
    (define (domain dummy)
        (:requirements :strips :typing)
        (:types fish banana)
        (:predicates
            (ate ?fish - fish ?ban - banana)
            (isMonkFish ?fish - fish)
            (isRipe ?ban - banana)
        )
        (:action eat
            :parameters (?f - fish ?b - banana)
            :precondition (and (isMonkFish ?f) (isRipe ?b))
            :effect (and (ate ?f ?b) (not (isRipe ?b)))
        )
    )"""

    problem_str1 = """; This is a comment
    (define (problem dummy-problem1)
        (:domain dummy)
        (:objects
            fish1 - fish
            ban1 - banana
        )
        (:init
            (isMonkFish fish1)
            (isRipe ban1)
        )
        (:goal (and (ate fish1 ban1)))
    )"""

    problem_str2 = """; This is a comment
    (define (problem dummy-problem2)
        (:domain dummy)
        (:objects
            fish1 fish2 - fish
            ban1 ban2 - banana
        )
        (:init
            (isMonkFish fish1)
            (isRipe ban2)
        )
        (:goal (and (ate fish1 ban2)))
    )"""

    class _DummyPDDLEnv(_PDDLEnv):

        @classmethod
        def get_name(cls):
            return "dummy"

        @property
        def _domain_str(self):
            return domain_str

        @property
        def _pddl_train_problem_generator(self):
            return lambda num, rng: [problem_str1]

        @property
        def _pddl_test_problem_generator(self):
            return lambda num, rng: [problem_str2]

    env = _DummyPDDLEnv()

    # Domain creation checks.
    assert np.allclose(env.action_space.low,
                       np.array([0, 0, 0], dtype=np.float32))
    assert np.allclose(env.action_space.high,
                       np.array([0, np.inf, np.inf], dtype=np.float32))
    # All types inherit from 'object' by default (via pyperplan).
    assert {t.name for t in env.types} == {"object", "banana", "fish"}
    type_name_to_type = {t.name: t for t in env.types}
    banana_type = type_name_to_type["banana"]
    fish_type = type_name_to_type["fish"]
    # Pyperplan parsing converts everything to lowercase.
    assert {p.name for p in env.predicates} == {"isripe", "ismonkfish", "ate"}
    pred_name_to_pred = {p.name: p for p in env.predicates}
    isRipe = pred_name_to_pred["isripe"]
    assert isRipe.types == [banana_type]
    isMonkfish = pred_name_to_pred["ismonkfish"]
    assert isMonkfish.types == [fish_type]
    ate = pred_name_to_pred["ate"]
    assert ate.types == [fish_type, banana_type]
    assert {o.name for o in env.options} == {"eat"}
    assert env.goal_predicates == {ate}
    option_name_to_option = {o.name: o for o in env.options}
    eat_option = option_name_to_option["eat"]
    assert eat_option.types == [fish_type, banana_type]
    assert eat_option.params_space.shape[0] == 0
    assert {o.name for o in env.strips_operators} == {"eat"}
    operator_name_to_operator = {o.name: o for o in env.strips_operators}
    eat_operator = operator_name_to_operator["eat"]
    eat_parameters = eat_operator.parameters
    assert [p.type for p in eat_parameters] == [fish_type, banana_type]
    fish_var, ban_var = eat_parameters
    assert eat_operator.preconditions == {
        isMonkfish([fish_var]), isRipe([ban_var])
    }
    assert eat_operator.add_effects == {ate([fish_var, ban_var])}
    assert eat_operator.delete_effects == {isRipe([ban_var])}

    # Problem creation checks.
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 1
    train_task = train_tasks[0]
    init = train_task.init
    assert {o.name for o in init} == {"fish1", "ban1"}
    obj_name_to_obj = {o.name: o for o in init}
    fish1 = obj_name_to_obj["fish1"]
    ban1 = obj_name_to_obj["ban1"]
    assert fish1.type == fish_type
    assert ban1.type == banana_type
    assert len(init[fish1]) == 0
    assert len(init[ban1]) == 0
    assert init.simulator_state == {isMonkfish([fish1]), isRipe([ban1])}
    assert train_task.goal == {ate([fish1, ban1])}
    test_tasks = env.get_test_tasks()
    assert len(test_tasks) == 1
    test_task = test_tasks[0]
    init = test_task.init
    assert {o.name for o in init} == {"fish1", "ban1", "fish2", "ban2"}
    obj_name_to_obj = {o.name: o for o in init}
    fish1 = obj_name_to_obj["fish1"]
    fish2 = obj_name_to_obj["fish2"]
    ban1 = obj_name_to_obj["ban1"]
    ban2 = obj_name_to_obj["ban2"]
    assert fish1.type == fish_type
    assert fish2.type == fish_type
    assert ban1.type == banana_type
    assert ban2.type == banana_type
    assert len(init[fish1]) == 0
    assert len(init[fish2]) == 0
    assert len(init[ban1]) == 0
    assert len(init[ban2]) == 0
    assert init.simulator_state == {isMonkfish([fish1]), isRipe([ban2])}
    assert test_task.goal == {ate([fish1, ban2])}

    # Tests for simulation.
    state = test_task.init.copy()
    with pytest.raises(NotImplementedError):
        env.render_state(state, test_task)
    inapplicable_option = eat_option.ground([fish1, ban1], [])
    assert not inapplicable_option.initiable(state)
    # This is generally not defined, but in this case, it will just give us
    # an invalid action that we can use to test simulate.
    inapplicable_action = inapplicable_option.policy(state)
    next_state = env.simulate(state, inapplicable_action)
    assert state.simulator_state == next_state.simulator_state
    assert state.allclose(next_state)
    option = eat_option.ground([fish1, ban2], [])
    assert option.initiable(state)
    action = option.policy(state)
    next_state = env.simulate(state, action)
    assert state.simulator_state != next_state.simulator_state
    assert not state.allclose(next_state)
    assert next_state.simulator_state == {
        isMonkfish([fish1]), ate([fish1, ban2])
    }
