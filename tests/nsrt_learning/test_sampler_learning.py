"""Tests for sampler learning."""

import pytest
from gym.spaces import Box
import numpy as np
from predicators.src.nsrt_learning.sampler_learning import \
    _create_sampler_data, learn_samplers, _LearnedSampler
from predicators.src.structs import Type, Predicate, State, Action, \
    ParameterizedOption, LiftedAtom, Segment, LowLevelTrajectory
from predicators.src import utils
from predicators.src.torch_models import NeuralGaussianRegressor, MLPClassifier
from predicators.src.envs import ClutteredTablePlaceEnv


def test_create_sampler_data():
    """Tests for _create_sampler_data()."""
    utils.reset_config({
        "env": "cover",
        "min_data_for_nsrt": 0,
        "num_train_tasks": 15,
        "sampler_disable_classifier": False,
    })
    # Create two datastores
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    var_cup0 = cup_type("?cup0")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    predicates = {pred0}
    option = ParameterizedOption(
        "dummy", [], Box(0.1, 1, (1, )), lambda s, m, o, p: Action(p),
        lambda s, m, o, p: False,
        lambda s, m, o, p: False).ground([], np.array([0.3]))

    # Transition 1: adds pred0(cup0)
    state = State({cup0: [0.4]})
    action = option.policy(state)
    action.set_option(option)
    next_state = State({cup0: [0.9]})
    atoms = utils.abstract(state, predicates)
    next_atoms = utils.abstract(next_state, predicates)
    segment1 = Segment(LowLevelTrajectory([state, next_state], [action]),
                       atoms, next_atoms, option)
    var_to_obj1 = {var_cup0: cup0}

    # Transition 2: does nothing
    state = State({cup0: [0.4]})
    action = option.policy(state)
    action.set_option(option)
    next_state = state
    atoms = utils.abstract(state, predicates)
    next_atoms = utils.abstract(next_state, predicates)
    segment2 = Segment(LowLevelTrajectory([state, next_state], [action]),
                       atoms, next_atoms, option)
    var_to_obj2 = {var_cup0: cup0}

    datastores = [[(segment1, var_to_obj1)], [(segment2, var_to_obj2)]]
    variables = [var_cup0]
    preconditions = set()
    add_effects = {LiftedAtom(pred0, [var_cup0])}
    delete_effects = set()
    param_option = option.parent
    datastore_idx = 0

    positive_examples, negative_examples = _create_sampler_data(
        datastores, variables, preconditions, add_effects, delete_effects,
        param_option, datastore_idx)
    assert len(positive_examples) == 1
    assert len(negative_examples) == 1

    utils.reset_config({
        "env": "cover",
        "min_data_for_nsrt": 0,
        "num_train_tasks": 15,
        "sampler_disable_classifier": True,
    })
    positive_examples, negative_examples = _create_sampler_data(
        datastores, variables, preconditions, add_effects, delete_effects,
        param_option, datastore_idx)
    assert len(positive_examples) == 1
    assert len(negative_examples) == 0  # classifier disabled, no negatives
    utils.reset_config({
        "env": "cover",
        "min_data_for_nsrt": 0,
        "num_train_tasks": 15,
        "sampler_disable_classifier": False,
    })

    # When building data for a datastore with effects X, if we
    # encounter a transition with effects Y, and if Y is a superset
    # of X, then we do not want to include the transition as a
    # negative example, because if Y was achieved, then X was also
    # achieved. So for now, we just filter out such examples.
    #
    # In the example here, transition 1's effects are a superset
    # of transition 2's effects. So when creating the examples
    # for datastore 2, we do not want to inclue transition 1
    # in the negative effects.
    variables = []
    add_effects = set()
    datastore_idx = 1
    positive_examples, negative_examples = _create_sampler_data(
        datastores, variables, preconditions, add_effects, delete_effects,
        param_option, datastore_idx)
    assert len(positive_examples) == 1
    assert len(negative_examples) == 0


def test_learn_samplers_failure():
    """Tests for failure mode of learn_samplers()."""
    option = ParameterizedOption("dummy", [], Box(0.1, 1, (1, )),
                                 lambda s, m, o, p: Action(p),
                                 lambda s, m, o, p: False,
                                 lambda s, m, o, p: False)
    with pytest.raises(NotImplementedError):  # bad sampler_learner
        learn_samplers([None], None, [(option, [])], "bad sampler learner")


def test_learned_sampler_with_goal():
    """Tests _LearnedSampler() when goals are used."""
    utils.reset_config({
        "env": "cluttered_table_place",
        "sampler_learning_use_goals": True,
        "num_train_tasks": 15,
        "neural_gaus_regressor_max_itr": 3
    })
    env = ClutteredTablePlaceEnv()
    task = env.get_test_tasks()[0]
    goal = task.goal

    rng = np.random.default_rng(0)
    state = task.init
    goal = task.goal
    goal_obj = next(iter(goal)).objects[0]
    objects = [goal_obj]

    input_size = 11
    classifier = MLPClassifier(input_size, 10)

    input_size = 11
    output_size = 2
    num_samples = 5
    regressor = NeuralGaussianRegressor()
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    regressor.fit(X, Y)

    variables = [goal_obj]
    parameterized_option = ParameterizedOption(
        "Dummy", [], Box(0, 1,
                         (1, )), lambda s, m, o, p: Action(np.array([0.0])),
        utils.always_initiable, utils.onestep_terminal)

    ls = _LearnedSampler(classifier, regressor, variables,
                         parameterized_option).sampler
    params = ls(state, goal, rng, objects)
    assert not params is None

    # Should still work when the classifier is disabled.
    utils.update_config({"sampler_disable_classifier": True})
    params = ls(state, goal, rng, objects)
    assert not params is None
