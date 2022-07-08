"""Tests for PDDLEnv."""

import os
import shutil

import numpy as np
import pytest

from predicators.src import utils
from predicators.src.envs.pddl_env import FixedTasksBlocksPDDLEnv, \
    ProceduralTasksBlocksPDDLEnv, ProceduralTasksDeliveryPDDLEnv, \
    ProceduralTasksForestPDDLEnv, _FixedTasksPDDLEnv, _PDDLEnv
from predicators.src.structs import Action


@pytest.fixture(scope="module", name="domain_str")
def _create_domain_str():
    return """; This is a comment
    (define (domain dummy)
        (:requirements :strips :typing)
        (:types
            fish banana - object
            salmon - fish
        )
        (:predicates
            (ate ?obj - object)
            (isRipe ?ban - banana)
            (isPink ?salmon - salmon)
            (isCooked ?fish - fish)
        )
        (:action eatFish
            :parameters (?f - fish)
            :precondition (and (isCooked ?f))
            :effect (and (ate ?f))
        )
        (:action eatBanana
            :parameters (?b - banana)
            :precondition (and (isRipe ?b))
            :effect (and (ate ?b))
        )
        (:action cook
            :parameters (?s - salmon)
            :precondition (and (isPink ?s))
            :effect (and (not (isPink ?s)) (isCooked ?s))
        )
    )"""


@pytest.fixture(scope="module", name="problem_strs")
def _create_problem_strs():
    problem_str1 = """; This is a comment
    (define (problem dummy-problem1)
        (:domain dummy)
        (:objects
            fish1 - fish
            ban1 - banana
            salmon1 - salmon
        )
        (:init
            (isCooked fish1)
            (isRipe ban1)
            (isPink salmon1)
        )
        (:goal (and (ate fish1) (ate ban1) (ate salmon1)))
    )"""

    problem_str2 = """; This is a comment
    (define (problem dummy-problem2)
        (:domain dummy)
        (:objects
            fish1 fish2 - fish
            ban1 ban2 - banana
        )
        (:init
            (isCooked fish1)
            (isRipe ban2)
        )
        (:goal (and (ate fish1) (ate ban2)))
    )"""

    return [problem_str1, problem_str2]


def test_pddlenv(domain_str, problem_strs):
    """Tests for PDDLEnv()."""
    utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})
    problem_str1, problem_str2 = problem_strs

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
    assert env.get_name() == "dummy"

    # Domain creation checks.
    assert np.allclose(env.action_space.low, np.array([0, 0],
                                                      dtype=np.float32))
    assert np.allclose(env.action_space.high,
                       np.array([2, np.inf], dtype=np.float32))
    # All types inherit from 'object' by default (via pyperplan).
    type_names = {t.name for t in env.types}
    assert type_names == {"object", "banana", "fish", "salmon"}
    type_name_to_type = {t.name: t for t in env.types}
    object_type = type_name_to_type["object"]
    banana_type = type_name_to_type["banana"]
    fish_type = type_name_to_type["fish"]
    salmon_type = type_name_to_type["salmon"]
    # Pyperplan parsing converts everything to lowercase.
    assert {p.name
            for p in env.predicates
            } == {"isripe", "iscooked", "ispink", "ate"}
    pred_name_to_pred = {p.name: p for p in env.predicates}
    isRipe = pred_name_to_pred["isripe"]
    assert isRipe.types == [banana_type]
    isCooked = pred_name_to_pred["iscooked"]
    assert isCooked.types == [fish_type]
    ate = pred_name_to_pred["ate"]
    assert ate.types == [object_type]
    IsPink = pred_name_to_pred["ispink"]
    assert IsPink.types == [salmon_type]
    assert {o.name for o in env.options} == {"eatfish", "eatbanana", "cook"}
    assert env.goal_predicates == {ate}
    option_name_to_option = {o.name: o for o in env.options}
    eat_fish_option = option_name_to_option["eatfish"]
    assert eat_fish_option.types == [fish_type]
    assert eat_fish_option.params_space.shape[0] == 0
    assert {o.name
            for o in env.strips_operators} == {"eatfish", "eatbanana", "cook"}
    operator_name_to_operator = {o.name: o for o in env.strips_operators}
    eat_fish_operator = operator_name_to_operator["eatfish"]
    eat_fish_parameters = eat_fish_operator.parameters
    assert [p.type for p in eat_fish_parameters] == [fish_type]
    fish_var, = eat_fish_parameters
    assert eat_fish_operator.preconditions == {isCooked([fish_var])}
    assert eat_fish_operator.add_effects == {ate([fish_var])}
    assert eat_fish_operator.delete_effects == set()

    # Problem creation checks.
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 1
    train_task = train_tasks[0]
    init = train_task.init
    assert {o.name for o in init} == {"fish1", "ban1", "salmon1"}
    obj_name_to_obj = {o.name: o for o in init}
    fish1 = obj_name_to_obj["fish1"]
    ban1 = obj_name_to_obj["ban1"]
    salmon1 = obj_name_to_obj["salmon1"]
    assert fish1.type == fish_type
    assert ban1.type == banana_type
    assert salmon1.type == salmon_type
    assert salmon1.is_instance(fish_type)
    assert salmon1.is_instance(object_type)
    assert len(init[fish1]) == 0
    assert len(init[ban1]) == 0
    assert len(init[salmon1]) == 0
    assert init.simulator_state == {
        isCooked([fish1]),
        isRipe([ban1]), IsPink([salmon1])
    }
    assert train_task.goal == {ate([fish1]), ate([ban1]), ate([salmon1])}
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
    assert init.simulator_state == {isCooked([fish1]), isRipe([ban2])}
    assert test_task.goal == {ate([fish1]), ate([ban2])}

    # Tests for simulation.
    state = train_task.init.copy()
    with pytest.raises(NotImplementedError):
        env.render_state(state, test_task)
    with pytest.raises(NotImplementedError) as e:
        env.render_state_plt(state, test_task)
    assert "This env does not use Matplotlib" in str(e)
    inapplicable_option = eat_fish_option.ground([salmon1], [])
    assert not inapplicable_option.initiable(state)
    # This is generally not defined, but in this case, it will just give us
    # an invalid action that we can use to test simulate.
    inapplicable_action = inapplicable_option.policy(state)
    next_state = env.simulate(state, inapplicable_action)
    assert state.simulator_state == next_state.simulator_state
    assert state.allclose(next_state)
    state = test_task.init.copy()
    option = eat_fish_option.ground([fish1], [])
    assert option.initiable(state)
    action = option.policy(state)
    next_state = env.simulate(state, action)
    assert state.simulator_state != next_state.simulator_state
    assert not state.allclose(next_state)
    assert next_state.simulator_state == {
        isCooked([fish1]), ate([fish1]),
        isRipe([ban2])
    }
    # Test that when the object types don't match the operator
    # parameters, a noop occurs.
    action = Action(
        np.zeros(env.action_space.shape, dtype=env.action_space.dtype))
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)


def test_fixed_tasks_pddlenv(domain_str, problem_strs):
    """Tests for _FixedTasksPDDLEnv()."""
    utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})
    problem_str1, problem_str2 = problem_strs

    # Set up fake assets.
    asset_path = utils.get_env_asset_path("pddl/dummy_fixed_tasks",
                                          assert_exists=False)
    os.makedirs(asset_path, exist_ok=True)
    task0_path = os.path.join(asset_path, "task0.pddl")
    task42_path = os.path.join(asset_path, "task42.pddl")
    with open(task0_path, "w", encoding="utf-8") as f:
        f.write(problem_str1)
    with open(task42_path, "w", encoding="utf-8") as f:
        f.write(problem_str2)

    class _DummyFixedTasksPDDLEnv(_FixedTasksPDDLEnv):

        @classmethod
        def get_name(cls):
            return "dummy_fixed_tasks"

        @property
        def _domain_str(self):
            return domain_str

        @property
        def _pddl_problem_asset_dir(self):
            return "dummy_fixed_tasks"

        @property
        def _train_problem_indices(self):
            return [0]

        @property
        def _test_problem_indices(self):
            return [42]

    env = _DummyFixedTasksPDDLEnv()
    assert env.get_name() == "dummy_fixed_tasks"

    # Just check that the correspondence is correct. Detailed testing is
    # covered by test_pddlenv.
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 1
    train_task = train_tasks[0]
    assert len(set(train_task.init)) == 3
    test_tasks = env.get_test_tasks()
    assert len(test_tasks) == 1
    test_task = test_tasks[0]
    assert len(set(test_task.init)) == 4

    # Remove fake assets.
    shutil.rmtree(asset_path)


def test_fixed_tasks_blocks_pddl_env():
    """Tests for FixedTasksBlocksPDDLEnv class."""
    utils.reset_config({
        "env": "pddl_blocks_fixed_tasks",
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })
    env = FixedTasksBlocksPDDLEnv()
    assert {t.name for t in env.types} == {"object", "block"}
    assert {p.name
            for p in env.predicates
            } == {"on", "ontable", "clear", "handempty", "holding"}
    assert {p.name for p in env.goal_predicates} == {"on"}
    assert {o.name
            for o in env.options
            } == {"pick-up", "put-down", "stack", "unstack"}
    assert {o.name
            for o in env.strips_operators
            } == {"pick-up", "put-down", "stack", "unstack"}
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 2
    test_tasks = env.get_test_tasks()
    assert len(test_tasks) == 2
    task = train_tasks[0]
    assert {a.predicate.name for a in task.goal} == {"on"}


def test_procedural_tasks_blocks_pddl_env():
    """Tests for ProceduralTasksBlocksPDDLEnv class."""
    # Note that the procedural generation itself is tested in
    # test_pddl_procedural_generation.
    utils.reset_config({
        "env": "pddl_blocks_procedural_tasks",
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })
    env = ProceduralTasksBlocksPDDLEnv()
    assert {t.name for t in env.types} == {"object", "block"}
    assert {p.name
            for p in env.predicates
            } == {"on", "ontable", "clear", "handempty", "holding"}
    assert {p.name for p in env.goal_predicates} == {"on", "ontable"}
    assert {o.name
            for o in env.options
            } == {"pick-up", "put-down", "stack", "unstack"}
    assert {o.name
            for o in env.strips_operators
            } == {"pick-up", "put-down", "stack", "unstack"}
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 2
    test_tasks = env.get_test_tasks()
    assert len(test_tasks) == 2
    task = train_tasks[0]
    assert {a.predicate.name for a in task.goal}.issubset({"on", "ontable"})


def test_procedural_tasks_delivery_pddl_env():
    """Tests for ProceduralTasksDeliveryPDDLEnv class."""
    # Note that the procedural generation itself is tested in
    # test_pddl_procedural_generation.
    utils.reset_config({
        "env": "pddl_delivery_procedural_tasks",
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })
    env = ProceduralTasksDeliveryPDDLEnv()
    assert {t.name for t in env.types} == {"object", "loc", "paper"}
    assert {p.name
            for p in env.predicates} == {
                "ishomebase", "wantspaper", "safe", "unpacked", "satisfied",
                "carrying", "at"
            }
    assert {p.name for p in env.goal_predicates} == {"satisfied"}
    assert {o.name for o in env.options} == {"pick-up", "move", "deliver"}
    assert {o.name
            for o in env.strips_operators} == {"pick-up", "move", "deliver"}
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 2
    test_tasks = env.get_test_tasks()
    assert len(test_tasks) == 2
    task = train_tasks[0]
    assert {a.predicate.name for a in task.goal}.issubset({"satisfied"})


def test_procedural_tasks_forest_pddl_env():
    """Tests for ProceduralTasksForestPDDLEnv class."""
    # Note that the procedural generation itself is tested in
    # test_pddl_procedural_generation.
    utils.reset_config({
        "env": "pddl_forest_procedural_tasks",
        "num_train_tasks": 2,
        "num_test_tasks": 2,
    })
    env = ProceduralTasksForestPDDLEnv()
    assert {t.name for t in env.types} == {"object", "loc"}
    assert {p.name
            for p in env.predicates} == {
                "isnotwater", "ishill", "isnothill", "at", "ontrail",
                "adjacent"
            }
    assert {p.name for p in env.goal_predicates} == {"at"}
    assert {o.name for o in env.options} == {"walk", "climb"}
    assert {o.name for o in env.strips_operators} == {"walk", "climb"}
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 2
    test_tasks = env.get_test_tasks()
    assert len(test_tasks) == 2
    task = train_tasks[0]
    assert {a.predicate.name for a in task.goal}.issubset({"at"})
