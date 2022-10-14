"""Tests for models."""

import time

import numpy as np
import pytest

from predicators import utils
from predicators.ml_models import BinaryClassifierEnsemble, \
    DegenerateMLPDistributionRegressor, ImplicitMLPRegressor, \
    KNeighborsClassifier, KNeighborsRegressor, MLPBinaryClassifier, \
    MLPRegressor, NeuralGaussianRegressor


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
                                 num_negative_data_per_input=5,
                                 temperature=1.0,
                                 inference_method="sample_once",
                                 derivative_free_num_iters=3,
                                 derivative_free_sigma_init=0.33,
                                 derivative_free_shrink_scale=0.5,
                                 grid_num_ticks_per_dim=100)
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
    # Test other inference methods. Protected access is to avoid retraining.
    model._inference_method = "derivative_free"  # pylint: disable=protected-access
    predicted_y = model.predict(x)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-1)
    model._inference_method = "grid"  # pylint: disable=protected-access
    predicted_y = model.predict(x)
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-1)
    model._inference_method = "not a real inference method"  # pylint: disable=protected-access
    with pytest.raises(NotImplementedError):
        model.predict(x)


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


def test_degenerate_mlp_distribution_regressor():
    """Tests for DegenerateMLPDistributionRegressor."""
    utils.reset_config()
    input_size = 3
    output_size = 2
    num_samples = 5
    model = DegenerateMLPDistributionRegressor(seed=123,
                                               hid_sizes=[32, 32],
                                               max_train_iters=100,
                                               clip_gradients=True,
                                               clip_value=5,
                                               learning_rate=1e-3)
    X = np.ones((num_samples, input_size))
    Y = np.zeros((num_samples, output_size))
    model.fit(X, Y)
    x = np.ones(input_size)
    mean = model.predict(x)
    expected_y = np.zeros(output_size)
    assert mean.shape == expected_y.shape
    assert np.allclose(mean, expected_y, atol=1e-2)
    rng = np.random.default_rng(123)
    sample = model.predict_sample(x, rng)
    assert sample.shape == expected_y.shape
    assert np.allclose(sample, expected_y, atol=1e-2)
    assert np.allclose(sample, mean, atol=1e-6)


def test_mlp_classifier():
    """Tests for MLPBinaryClassifier."""
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
    model = MLPBinaryClassifier(seed=123,
                                balance_data=True,
                                max_train_iters=100,
                                learning_rate=1e-3,
                                n_iter_no_change=1000000,
                                hid_sizes=[32, 32],
                                n_reinitialize_tries=1,
                                weight_init="default")
    model.fit(X, y)
    prediction = model.classify(np.zeros(input_size))
    assert not prediction
    assert model.predict_proba(np.zeros(input_size)) < 0.5
    prediction = model.classify(np.ones(input_size))
    assert prediction
    assert model.predict_proba(np.ones(input_size)) > 0.5
    # Test for early stopping
    start_time = time.perf_counter()
    model = MLPBinaryClassifier(seed=123,
                                balance_data=True,
                                max_train_iters=100000,
                                learning_rate=1e-2,
                                n_iter_no_change=1,
                                hid_sizes=[32, 32],
                                n_reinitialize_tries=1,
                                weight_init="default")
    model.fit(X, y)
    assert time.perf_counter() - start_time < 3, "Didn't early stop"
    # Test with no positive examples.
    num_class_samples = 1000
    X = np.concatenate([
        np.zeros((num_class_samples, input_size)),
        np.ones((num_class_samples, input_size))
    ])
    y = np.zeros(len(X))
    model = MLPBinaryClassifier(seed=123,
                                balance_data=True,
                                max_train_iters=100000,
                                learning_rate=1e-3,
                                n_iter_no_change=100000,
                                hid_sizes=[32, 32],
                                n_reinitialize_tries=1,
                                weight_init="default")
    start_time = time.perf_counter()
    model.fit(X, y)
    assert time.perf_counter(
    ) - start_time < 1, "Fitting was not instantaneous"
    prediction = model.classify(np.zeros(input_size))
    assert not prediction
    prediction = model.classify(np.ones(input_size))
    assert not prediction
    # Test with no negative examples.
    y = np.ones(len(X))
    model = MLPBinaryClassifier(seed=123,
                                balance_data=True,
                                max_train_iters=100000,
                                learning_rate=1e-3,
                                n_iter_no_change=100000,
                                hid_sizes=[32, 32],
                                n_reinitialize_tries=1,
                                weight_init="default")
    start_time = time.perf_counter()
    model.fit(X, y)
    assert time.perf_counter(
    ) - start_time < 1, "Fitting was not instantaneous"
    prediction = model.classify(np.zeros(input_size))
    assert prediction
    prediction = model.classify(np.ones(input_size))
    assert prediction
    # Test with non-default weight initialization.
    X = np.concatenate([
        np.zeros((num_class_samples, input_size)),
        np.ones((num_class_samples, input_size))
    ])
    y = np.concatenate(
        [np.zeros((num_class_samples)),
         np.ones((num_class_samples))])
    model = MLPBinaryClassifier(seed=123,
                                balance_data=True,
                                max_train_iters=100,
                                learning_rate=1e-3,
                                n_iter_no_change=100000,
                                hid_sizes=[32, 32],
                                n_reinitialize_tries=1,
                                weight_init="normal")
    model.fit(X, y)
    # Test with invalid weight initialization.
    model = MLPBinaryClassifier(seed=123,
                                balance_data=True,
                                max_train_iters=100000,
                                learning_rate=1e-3,
                                n_iter_no_change=100000,
                                hid_sizes=[32, 32],
                                n_reinitialize_tries=1,
                                weight_init="foo")
    with pytest.raises(NotImplementedError):
        model.fit(X, y)
    # Test for reinitialization failure.
    model = MLPBinaryClassifier(seed=123,
                                balance_data=True,
                                max_train_iters=100000,
                                learning_rate=1e-3,
                                n_iter_no_change=100000,
                                hid_sizes=[32, 32],
                                n_reinitialize_tries=0,
                                weight_init="default")
    with pytest.raises(RuntimeError):
        model.fit(X, y)


def test_binary_classifier_ensemble():
    """Tests for BinaryClassifierEnsemble."""
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
    model = BinaryClassifierEnsemble(seed=123,
                                     ensemble_size=3,
                                     member_cls=MLPBinaryClassifier,
                                     balance_data=True,
                                     max_train_iters=100,
                                     learning_rate=1e-3,
                                     n_iter_no_change=1000000,
                                     hid_sizes=[32, 32],
                                     n_reinitialize_tries=1,
                                     weight_init="default")
    model.fit(X, y)
    with pytest.raises(Exception) as e:
        model.predict_proba(np.zeros(input_size))
    assert "Can't call predict_proba()" in str(e)
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
    assert len(probas) == 3
    # Test the KNN classifier with n_neighbors = num_class_samples.
    # Since there are num_class_samples data points of each class,
    # the probas should be all 0's or all 1's.
    model = BinaryClassifierEnsemble(seed=123,
                                     ensemble_size=3,
                                     member_cls=KNeighborsClassifier,
                                     n_neighbors=num_class_samples)
    model.fit(X, y)
    prediction = model.classify(np.zeros(input_size))
    assert not prediction
    probas = model.predict_member_probas(np.zeros(input_size))
    assert all(p == 0.0 for p in probas)
    assert len(probas) == 3
    prediction = model.classify(np.ones(input_size))
    assert prediction
    probas = model.predict_member_probas(np.ones(input_size))
    assert all(p == 1.0 for p in probas)
    assert len(probas) == 3
    # Test the KNN classifier with n_neighbors = 2 * num_class_samples.
    # Since there are num_class_samples data points of each class,
    # the probas should be all 0.5's.
    model = BinaryClassifierEnsemble(seed=123,
                                     ensemble_size=3,
                                     member_cls=KNeighborsClassifier,
                                     n_neighbors=(2 * num_class_samples))
    model.fit(X, y)
    probas = model.predict_member_probas(np.zeros(input_size))
    assert all(p == 0.5 for p in probas)
    assert len(probas) == 3
    probas = model.predict_member_probas(np.ones(input_size))
    assert all(p == 0.5 for p in probas)
    assert len(probas) == 3


def test_k_neighbors_regressor():
    """Tests for KNeighborsRegressor()."""
    utils.reset_config()
    input_size = 3
    output_size = 2
    num_samples = 5
    model = KNeighborsRegressor(seed=123, n_neighbors=1)
    rng = np.random.default_rng(123)
    X = rng.normal(size=(num_samples, input_size))
    Y = rng.normal(size=(num_samples, output_size))
    model.fit(X, Y)
    x = X[0]
    predicted_y = model.predict(x)
    expected_y = Y[0]
    assert predicted_y.shape == expected_y.shape
    assert np.allclose(predicted_y, expected_y, atol=1e-7)


def test_k_neighbors_classifier():
    """Tests for KNeighborsClassifier()."""
    utils.reset_config()
    input_size = 3
    num_samples = 5
    model = KNeighborsClassifier(seed=123, n_neighbors=1)
    rng = np.random.default_rng(123)
    X = rng.normal(size=(num_samples, input_size))
    Y = rng.choice(2, size=(num_samples, ))
    model.fit(X, Y)
    x = X[0]
    predicted_y = model.classify(x)
    expected_y = Y[0]
    assert isinstance(predicted_y, bool)
    assert predicted_y == expected_y
    assert model.predict_proba(x) == expected_y
