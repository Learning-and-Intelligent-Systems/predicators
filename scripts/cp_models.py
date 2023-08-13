
from typing import List, Tuple
from numpy.typing import NDArray

from functools import partial
import numpy as np
from scipy.optimize import minimize
from jax.scipy.stats import beta as beta_distribution



def _run_inference(history: List[List[bool]], betas: List[Tuple[float, float]]) -> List[float]:
    assert len(history) == len(betas)
    map_competences: List[float] = []
    # NOTE: this is the mean rather than the mode, for simplicity...
    # TODO: maybe change
    for outcomes, (a, b) in zip(history, betas):
        n = len(outcomes)
        s = sum(outcomes)
        alpha_n = a + s
        beta_n = n - s + b
        mean = alpha_n / (alpha_n + beta_n)
        assert 0 < mean < 1
        map_competences.append(mean)
    return map_competences


def _run_learning(num_data_before_cycle: NDArray[np.float32], map_competences: List[float]) -> Tuple[NDArray[np.float32], float]:
    """Return parameters for mean prediction and constant variance."""
    fn = partial(_loss, num_data_before_cycle, map_competences)
    # Transform into unconstrained space.
    theta_0 = np.array([0.25, 0.75, 1.0])
    unconstrained_theta_0 = _transform_model_params_to_unconstrain(theta_0)
    res = minimize(fn, theta_0, method="L-BFGS-B", options=dict(maxiter=1000000, ftol=1e-1, eps=1e-3, verbose=True))
    unconstrained_theta_final = res.x
    theta_final = _invert_transform_model_params(unconstrained_theta_final)
    import ipdb; ipdb.set_trace()


def _validate_model_params(theta: NDArray[np.float32]) -> None:
    theta0, theta1, theta2 = theta
    assert 0 <= theta0 <= 1
    assert theta0 <= theta1 <= 1
    assert theta2 > 0


def _transform_model_params_to_unconstrain(theta: NDArray[np.float32]) -> NDArray[np.float32]:
    _validate_model_params(theta)
    theta0, theta1, theta2 = theta
    unconstrained_theta0 = np.log(theta0) - np.log(1 - theta0)  # logit
    unconstrained_theta1 = np.log(theta1) - np.log(1 - theta1)  # logit, will clip
    unconstrained_theta2 = theta2  # will clip
    return np.array([unconstrained_theta0, unconstrained_theta1, unconstrained_theta2])


def _invert_transform_model_params(transformed_theta: NDArray[np.float32]) -> NDArray[np.float32]:
    utheta0, utheta1, utheta2 = transformed_theta
    theta0 = 1 / (1 + np.exp(-utheta0))  # sigmoid, inverse of logit
    theta1 = max(1 / (1 + np.exp(-utheta1)), theta0)  # sigmoid + clip
    theta2 = max(0, utheta2)  # clip
    theta = np.array([theta0, theta1, theta2], dtype=np.float32)
    _validate_model_params(theta)
    return theta


def _loss(num_data_before_cycle: NDArray[np.float32], map_competences: List[float], model_params: NDArray[np.float32]) -> float:
    means = _model_predict(num_data_before_cycle, model_params)
    variance = np.var(map_competences - means)
    betas = [_beta_from_mean_and_variance(m, variance) for m in means]
    nlls = [-beta_distribution.logpdf(c, a, b) for c, (a, b) in zip(map_competences, betas)]
    return sum(nlls)


def _model_predict(x: NDArray[np.float32], transformed_params: NDArray[np.float32]) -> NDArray[np.float32]:
    params = _invert_transform_model_params(transformed_params)
    _validate_model_params(params)
    theta0, theta1, theta2 = params
    print(theta0, theta1, theta2)
    out = theta0 + (theta1 - theta0) * (1 - np.exp(-theta2 * x))
    assert np.all(out > 0) and np.all(out < 1)
    return out


def _beta_from_mean_and_variance(mean: float, variance: float) -> Tuple[float, float]:
    alpha = ((1 - mean) / variance  - 1 / mean) * (mean**2)
    beta = alpha * (1 / mean - 1)
    return (alpha, beta)


def _run_em(history: List[List[bool]], num_em_iters: int=10) -> Tuple[List[NDArray[np.float32]], List[Tuple[float, float]], List[float]]:
    num_cycles = len(history)
    num_data_after_cycle = list(np.cumsum([len(h) for h in history]))
    num_data_before_cycle = np.array([0] + num_data_after_cycle[:-1], dtype=np.float32)
    # Initialize betas with uniform distribution.
    betas = [(1.0, 1.0) for _ in range(num_cycles)]
    all_map_competences = []
    all_model_params = []
    all_betas = []
    for _ in range(num_em_iters):
        # Run inference.
        map_competences = _run_inference(history, betas)
        all_map_competences.append(map_competences)
        # Run learning.
        model_params, variance = _run_learning(num_data_before_cycle, map_competences)
        all_model_params.append(model_params)
        # Update betas by evaluating the model.
        mean = _model_predict(num_data_before_cycle, model_params)
        betas = _beta_from_mean_and_variance(mean, variance)
        all_betas.append(betas)
    return all_model_params, all_betas, all_map_competences


def _main():
    history = [
        [False, False, False],
        [True, False, False, True, False, False, False, False, False],
        [False, True, True, False, True, False, False, False],
        [False],
        [True, True, False, False, True, True],
        [True, True, True],
    ]
    all_model_params, all_betas, all_map_competences = _run_em(history)
    # _make_plots(history, all_model_params, all_map_competences, outfile = "pgmax_script_out_v1.mp4")

if __name__ == "__main__":
    _main()
