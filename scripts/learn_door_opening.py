"""A standalone script to experiment with learning the function to open doors
in the doors environment.

Notes:
    * Not using [1.0] in input. I don't think it matters.
    * Not using a binary classifier. Train the main method with
      --sampler_disable_classifier True for direct comparison.
"""
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from predicators.src import utils
from predicators.src.ml_models import DegenerateMLPDistributionRegressor, \
    DistributionRegressor, MLPRegressor, NeuralGaussianRegressor
from predicators.src.settings import CFG
from predicators.src.structs import Array

# Hardcoding these values for convenience.
# State dims: [x, y, theta, mass, friction, rot, target_rot, open, rx, ry]
NUM_STATE_DIMS = 10
MASS, FRICTION, ROT, TARGET_ROT, OPEN = range(3, 8)
# Param dims: [delta_rot, delta_open]
NUM_PARAM_DIMS = 2

NUM_DATA = [50, 100, 250, 500, 1000]
VALIDATION_FRAC = 0.1
START_SEED = 456
NUM_SEEDS = 10
# Uncomment to debug the pipeline.
# NUM_DATA = [5, 10]
# NUM_SEEDS = 2

SAMPLER_TYPES = ["gaussian", "degenerate"]
DATA_TYPE = "loaded"  # "synthetic"

OTHER_SETTINGS: Dict[str, Any] = {
    # Uncomment to debug the pipeline.
    # "mlp_regressor_max_itr": 10,
    # "neural_gaus_regressor_max_itr": 10,
    # "sesame_max_samples_per_step": 1,
}


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    all_results = []
    for seed in range(START_SEED, START_SEED + NUM_SEEDS):
        for num_data in NUM_DATA:
            # Run non-parameterized.
            results = _run_experiment(seed, num_data, parameterized=False)
            all_results.append(results)
            # Run parameterized with different sampler types.
            for sampler_type in SAMPLER_TYPES:
                results = _run_experiment(seed,
                                          num_data,
                                          parameterized=True,
                                          sampler_type=sampler_type)
                all_results.append(results)
    # Display and save all data.
    df = pd.DataFrame(all_results)
    logging.info(df)
    raw_outfile = os.path.join(outdir, "door_opening_raw.csv")
    df.to_csv(raw_outfile)
    logging.info(f"Saved dataframe to {raw_outfile}")
    # Make a plot.
    plt.figure()
    seed_key = "Seed"
    x_key = "Num Train Data"
    approach_key = "Approach"
    y_key = "Error"
    df = df[[seed_key, x_key, approach_key, y_key]]
    xs = sorted(np.unique(df[x_key]))
    approaches = sorted(np.unique(df[approach_key]))
    for approach in approaches:
        y_means, y_stds = [], []
        for x in xs:
            df_x = df[(df[x_key] == x) & (df[approach_key] == approach)][y_key]
            mean = np.mean(df_x)
            std = np.std(df_x)
            y_means.append(mean)
            y_stds.append(std)
        # Add a line to the plot.
        plt.errorbar(xs, y_means, yerr=y_stds, label=approach)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title("Standalone Door Learning")
    plt.legend()
    plt.tight_layout()
    outfile = os.path.join(outdir, "door_opening.png")
    plt.savefig(outfile)
    logging.info(f"Saved plot to {outfile}")


def _error_fn(yhat: Array, y: Array) -> float:
    return np.sum((yhat - y)**2)


def _run_experiment(seed: int,
                    num_data: int,
                    parameterized: bool,
                    sampler_type: Optional[str] = None) -> Dict[str, Any]:
    utils.reset_config({"seed": seed, **OTHER_SETTINGS})
    logging.info(f"\nRunning experiment for seed={seed}, "
                 f"num_data={num_data}")

    # Set up the results dict.
    if not parameterized:
        approach_name = "Nonparameterized"
    else:
        approach_name = f"Parameterized ({sampler_type})"
    results = {
        "Approach": approach_name,
        "Seed": seed,
        "Num Train Data": num_data,
    }
    logging.info(f"Starting experiment for {approach_name} with "
                 f"{num_data} data on seed {seed}.")

    # Generate data.
    X, Y = _generate_data(num_data)
    num_valid = max(1, int(num_data * VALIDATION_FRAC))
    train_X, full_test_X = X[num_valid:], X[:num_valid]
    train_Y, test_Y = Y[num_valid:], Y[:num_valid]

    # At test time, we don't have the parameterized dims.
    test_X = full_test_X[:, :NUM_STATE_DIMS]

    # Train the model.
    sample_fn = _train_model(train_X, train_Y, parameterized, sampler_type)

    # Evaluate the model by taking a min over multiple samples.
    num_samples = CFG.sesame_max_samples_per_step
    rng = np.random.default_rng(CFG.seed)
    errors = []
    for x, y in zip(test_X, test_Y):
        y_hats = [sample_fn(x, rng) for _ in range(num_samples)]
        errors_for_x = [_error_fn(yhat, y) for yhat in y_hats]
        min_error = min(errors_for_x)
        errors.append(min_error)
    error = np.mean(errors)
    logging.info(f"Error: {error}")
    results["Error"] = error
    return results


def _train_model(
    train_X: Array,
    train_Y: Array,
    parameterized: bool,
    sampler_type: Optional[str] = None
) -> Callable[[Array, np.random.Generator], Array]:

    if not parameterized:
        assert sampler_type is None
        return _train_nonparameterized_model(train_X, train_Y)

    assert sampler_type is not None
    return _train_parameterized_model(train_X, train_Y, sampler_type)


def _train_nonparameterized_model(
    train_X: Array,
    train_Y: Array,
) -> Callable[[Array, np.random.Generator], Array]:
    # Get the data for the nonparameterized model. This effectively removes
    # part of the inputs corresponding to the expected next state.
    nonparam_train_X = train_X[:, :NUM_STATE_DIMS]
    # Train a nonparameterized model.
    nonparam_model = MLPRegressor(
        seed=CFG.seed,
        hid_sizes=CFG.mlp_regressor_hid_sizes,
        max_train_iters=CFG.mlp_regressor_max_itr,
        clip_gradients=CFG.mlp_regressor_clip_gradients,
        clip_value=CFG.mlp_regressor_gradient_clip_value,
        learning_rate=CFG.learning_rate)
    nonparam_model.fit(nonparam_train_X, train_Y)

    # Construct a degenerate sampler.
    def _sample_fn(x: Array, rng: np.random.Generator) -> Array:
        del rng  # unused
        return nonparam_model.predict(x)

    return _sample_fn


def _train_parameterized_model(
        train_X: Array, train_Y: Array,
        sampler_type: str) -> Callable[[Array, np.random.Generator], Array]:
    # Get the data for the sampler for the parameterized model.
    sampler_train_X = train_X[:, :NUM_STATE_DIMS]  # same as nonparam_train_X
    sampler_train_Y = train_X[:, NUM_STATE_DIMS:]

    # Train the sampler for the parameterized model.
    if sampler_type == "gaussian":
        sampler: DistributionRegressor = NeuralGaussianRegressor(
            seed=CFG.seed,
            hid_sizes=CFG.neural_gaus_regressor_hid_sizes,
            max_train_iters=CFG.neural_gaus_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate)
    else:
        assert sampler_type == "degenerate"
        sampler = DegenerateMLPDistributionRegressor(
            seed=CFG.seed,
            hid_sizes=CFG.mlp_regressor_hid_sizes,
            max_train_iters=CFG.mlp_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate)

    sampler.fit(sampler_train_X, sampler_train_Y)

    # Train the parameterized model.
    param_model = MLPRegressor(
        seed=CFG.seed,
        hid_sizes=CFG.mlp_regressor_hid_sizes,
        max_train_iters=CFG.mlp_regressor_max_itr,
        clip_gradients=CFG.mlp_regressor_clip_gradients,
        clip_value=CFG.mlp_regressor_gradient_clip_value,
        learning_rate=CFG.learning_rate)
    param_model.fit(train_X, train_Y)

    # Construct a sample function from the parameterized model and sampler.
    def _sample_fn(x: Array, rng: np.random.Generator) -> Array:
        params = sampler.predict_sample(x, rng)
        return param_model.predict(np.hstack([x, params]))

    return _sample_fn


def _generate_data(num_data: int) -> Tuple[Array, Array]:
    if DATA_TYPE == "synthetic":
        return _generate_synthetic_data(num_data)
    assert DATA_TYPE == "loaded"
    data_dir = "saved_approaches"
    file_prefix = f"doors__nsrt_learning__{CFG.seed}____MoveToDoor,MoveThroughDoor__doors_main_{num_data}.saved"  # pylint: disable=line-too-long
    input_file = os.path.join(data_dir, file_prefix + ".option_X.npy")
    output_file = os.path.join(data_dir, file_prefix + ".option_Y.npy")
    X = np.load(input_file)
    Y = np.load(output_file)
    # Remove extra first dimension, which is all 1s.
    assert all(X[:, 0] == 1)
    X = X[:, 1:]
    return (X, Y)


def _generate_synthetic_data(num_data: int) -> Tuple[Array, Array]:
    rng = np.random.default_rng()
    # Generate random state inputs.
    state_inputs = rng.uniform(size=(num_data, NUM_STATE_DIMS))
    state_inputs[:, OPEN] = 0.0
    # Generate the targets based on the state inputs.
    mass = state_inputs[:, MASS]
    friction = state_inputs[:, FRICTION]
    target_rot = state_inputs[:, TARGET_ROT]
    ground_truth_outs = _ground_truth_function(mass, friction, target_rot)
    # Generate the parameterized inputs based on the outputs. For this example,
    # the parameterized outputs are the difference between the current rot and
    # the output, and then just 1.0 for the door opening.
    rot = state_inputs[:, ROT]
    delta_rot = ground_truth_outs - rot
    delta_open = np.ones_like(delta_rot)
    param_inputs = np.vstack([delta_rot, delta_open]).T
    concat_inputs = np.concatenate([state_inputs, param_inputs], axis=1)
    inputs = concat_inputs.astype(np.float32)
    outputs = np.reshape(ground_truth_outs, (num_data, 1)).astype(np.float32)
    return (inputs, outputs)


def _ground_truth_function(mass: NDArray, friction: NDArray,
                           target_rot: NDArray) -> NDArray:
    return np.tanh(target_rot) * (np.sin(mass) +
                                  np.cos(friction) * np.sqrt(mass))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s",
                        handlers=[logging.StreamHandler()])
    _main()
