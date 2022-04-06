"""Tests for models."""

import time

import numpy as np

from predicators.src import utils
from predicators.src.torch_models import ImplicitMLPRegressor, MLPClassifier, \
    MLPClassifierEnsemble, MLPRegressor, NeuralGaussianRegressor


def test_basic_mlp_regressor():
    """Tests for MLPRegressor."""
    utils.reset_config()
    input_size = 3
    output_size = 2
    num_samples = 5
    model = MLPRegressor(seed=123,
                         hid_sizes=[32, 32],
                         max_train_iters=100,
                         clip_gradients=True,
                         clip_value=5,
                         learning_rate=1e-3)
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = np.zeros(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-2)
    # Test with nonzero outputs.
    Y = 75 * np.ones((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = 75 * np.ones(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-2)


def test_implicit_mlp_regressor():
    """Tests for ImplicitMLPRegressor."""
    utils.reset_config()
    input_size = 3
    output_size = 1
    num_samples = 5
    model = ImplicitMLPRegressor(seed=123,
                                 hid_sizes=[32, 32],
                                 max_train_iters=100,
                                 clip_gradients=False,
                                 clip_value=5,
                                 learning_rate=1e-3,
                                 num_samples_per_inference=100,
                                 num_negative_data_per_input=5)
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = np.zeros(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-1)
    # Test with nonzero outputs.
    Y = 75 * np.ones((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    predicted_y = model.predict(x)
    expected_y = 75 * np.ones(output_size)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-1)


def test_neural_gaussian_regressor():
    """Tests for NeuralGaussianRegressor."""
    utils.reset_config()
    input_size = 3
    output_size = 2
    num_samples = 5
    model = NeuralGaussianRegressor(seed=123,
                                    hid_sizes=[32, 32],
                                    max_train_iters=100,
                                    clip_gradients=False,
                                    clip_value=5,
                                    learning_rate=1e-3)
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    mean = model.predict_mean(x)
    expected_y = np.zeros(output_size)
    assert mean.shape == expected_y.shape
    assert np.allclose(mean, expected_y, atol=1e-2)
    rng = np.random.default_rng(123)
    sample = model.predict_sample(x, rng)
    assert sample.shape == expected_y.shape


def test_mlp_classifier():
    """Tests for MLPClassifier."""
    utils.reset_config()
    input_size = 3
    num_class_samples = 5
    X = np.concatenate([
        np.zeros((num_class_samples, input_size)),
        np.ones((num_class_samples, input_size))
    ])
    y = np.concatenate(
        [np.zeros((num_class_samples)),
         np.ones((num_class_samples))])
    model = MLPClassifier(seed=123,
                          balance_data=True,
                          max_train_iters=100,
                          learning_rate=1e-3,
                          n_iter_no_change=1000000,
                          hid_sizes=[32, 32])
    model.fit(X, y)
    prediction = model.classify(np.zeros(input_size))
    assert not prediction
    assert model.predict_proba(np.zeros(input_size)) < 0.5
    prediction = model.classify(np.ones(input_size))
    assert prediction
    assert model.predict_proba(np.ones(input_size)) > 0.5
    # Test for early stopping
    start_time = time.time()
    model = MLPClassifier(seed=123,
                          balance_data=True,
                          max_train_iters=100000,
                          learning_rate=1e-2,
                          n_iter_no_change=1,
                          hid_sizes=[32, 32])
    model.fit(X, y)
    assert time.time() - start_time < 3, "Didn't early stop"
    # Test with no positive examples.
    num_class_samples = 1000
    X = np.concatenate([
        np.zeros((num_class_samples, input_size)),
        np.ones((num_class_samples, input_size))
    ])
    y = np.zeros(len(X))
    model = MLPClassifier(seed=123,
                          balance_data=True,
                          max_train_iters=100000,
                          learning_rate=1e-3,
                          n_iter_no_change=100000,
                          hid_sizes=[32, 32])
    start_time = time.time()
    model.fit(X, y)
    assert time.time() - start_time < 1, "Fitting was not instantaneous"
    prediction = model.classify(np.zeros(input_size))
    assert not prediction
    prediction = model.classify(np.ones(input_size))
    assert not prediction
    # Test with no negative examples.
    y = np.ones(len(X))
    model = MLPClassifier(seed=123,
                          balance_data=True,
                          max_train_iters=100000,
                          learning_rate=1e-3,
                          n_iter_no_change=100000,
                          hid_sizes=[32, 32])
    start_time = time.time()
    model.fit(X, y)
    assert time.time() - start_time < 1, "Fitting was not instantaneous"
    prediction = model.classify(np.zeros(input_size))
    assert prediction
    prediction = model.classify(np.ones(input_size))
    assert prediction


def test_mlp_classifier_ensemble():
    """Tests for MLPClassifierEnsemble."""
    utils.reset_config()
    input_size = 3
    num_class_samples = 5
    X = np.concatenate([
        np.zeros((num_class_samples, input_size)),
        np.ones((num_class_samples, input_size))
    ])
    y = np.concatenate(
        [np.zeros((num_class_samples)),
         np.ones((num_class_samples))])
    model = MLPClassifierEnsemble(seed=123,
                                  balance_data=True,
                                  max_train_iters=100,
                                  learning_rate=1e-3,
                                  n_iter_no_change=1000000,
                                  hid_sizes=[32, 32],
                                  ensemble_size=3)
    model.fit(X, y)
    prediction = model.classify(np.zeros(input_size))
    assert not prediction
    probas = model.predict_member_probas(np.zeros(input_size))
    assert all(p < 0.5 for p in probas)
    assert len(probas) == 3
    assert probas[0] != probas[1]  # there should be some variation
    prediction = model.classify(np.ones(input_size))
    assert prediction
    probas = model.predict_member_probas(np.ones(input_size))
    assert all(p > 0.5 for p in probas)
