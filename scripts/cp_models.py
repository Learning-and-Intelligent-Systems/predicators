from functools import partial
from pathlib import Path
from typing import List, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributions.beta import Beta as TorchBeta


from predicators import utils
from predicators.structs import MaxTrainIters, Array
from predicators.ml_models import PyTorchRegressor


def _run_inference(history: List[List[bool]],
                   betas: List[Tuple[float, float]],
                   lb: float = 1e-3,
                   ub: float = 1.0 - 1e-3) -> List[float]:
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
        assert 0 <= mean <= 1
        mean = np.clip(mean, lb, ub)
        map_competences.append(mean)
    return map_competences


def _get_predict_variances(yhat: NDArray[np.float32], y: NDArray[np.float32]) -> NDArray[np.float32]:
    # variances = np.subtract(yhat, y)**2
    # max_variances = yhat * (1 - yhat) - 1e-3
    # return np.minimum(variances, max_variances)
    # TODO figure this out!!
    return np.ones_like(yhat) * 1e-6



class CompetenceModel(PyTorchRegressor):

    def __init__(self,
                 seed: int,
                 max_train_iters: MaxTrainIters,
                 clip_gradients: bool,
                 clip_value: float,
                 learning_rate: float,
                 weight_decay: float = 0,
                 use_torch_gpu: bool = False,
                 train_print_every: int = 1000,
                 n_iter_no_change: int = 10000000) -> None:
        super().__init__(seed,
                         max_train_iters,
                         clip_gradients,
                         clip_value,
                         learning_rate,
                         weight_decay=weight_decay,
                         n_iter_no_change=n_iter_no_change,
                         use_torch_gpu=use_torch_gpu,
                         do_normalize=False,
                         train_print_every=train_print_every)
        self.theta = torch.nn.Parameter(torch.randn(3), requires_grad=True)

    def interpret(self) -> None:
        print("Transformed weights:", self._transform_theta())
        print("At zero:", self.predict(np.array([0.0]))[0])
        print("At large number:", self.predict(np.array([1000000000.0]))[0])

    def _transform_theta(self) -> List[Tensor]:
        theta0 = self.theta[0]
        theta1 = self.theta[1]
        theta2 = self.theta[2]
        ctheta0 = F.sigmoid(theta0)
        ctheta1 = F.sigmoid(theta0 + (F.elu(theta1) + 1))
        ctheta2 = F.elu(theta2) + 1
        return [ctheta0, ctheta1, ctheta2]

    def forward(self, tensor_X: Tensor) -> Tensor:
        # Transform weights to obey constraints.
        ctheta0, ctheta1, ctheta2 = self._transform_theta()
        # Exponential saturation function.
        mean = ctheta0 + (ctheta1 - ctheta0) * (1 - torch.exp(-ctheta2 * tensor_X))
        # Clip mean to avoid numerical issues.
        mean = torch.clip(mean, 1e-3, 1.0 - 1e-3)
        return mean

    def _initialize_net(self) -> None:
        pass

    def _create_loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        return nn.MSELoss()


# class BetaLoss(nn.Module):
#     """Beta distribution loss.
    
#     Log likelihood is unstable, so just use MSE on means.
#     """

#     def __init__(self):
#         super(BetaLoss, self).__init__()

#     def forward(self, beta_params, targets):
#         # Clip targets to avoid numerical issues.
#         targets = torch.clip(targets, 1e-3, 1.0 - 1e-3)
#         alpha, beta = beta_params
#         dist = TorchBeta(alpha, beta, validate_args=True)
#         loss = ((dist.mean - targets)**2).mean()
#         return loss



def _run_learning(
        num_data_before_cycle: NDArray[np.float32],
        map_competences: List[float]) -> CompetenceModel:
    model = CompetenceModel(seed=0, max_train_iters=10000, clip_gradients=False, clip_value=1.0, learning_rate=1e-2)
    model.fit(np.reshape(num_data_before_cycle, (-1, 1)), np.reshape(map_competences, (-1, 1)))
    model.interpret()
    return model


def _validate_model_params(theta: NDArray[np.float32]) -> None:
    theta0, theta1, theta2 = theta
    assert 0 <= theta0 <= 1
    assert theta0 <= theta1 <= 1
    assert theta2 >= 0


# def _transform_model_params_to_unconstrain(
#         theta: NDArray[np.float32]) -> NDArray[np.float32]:
#     _validate_model_params(theta)
#     theta0, theta1, theta2 = theta
#     utheta0 = np.log(theta0) - np.log(1 - theta0)  # logit
#     utheta1 = np.log(theta1) - np.log(1 - theta1)  # logit / clip
#     utheta2 = theta2  # will clip
#     return np.array([utheta0, utheta1, utheta2], dtype=np.float32)


# def _invert_transform_model_params(
#         transformed_theta: NDArray[np.float32]) -> NDArray[np.float32]:
#     utheta0, utheta1, utheta2 = transformed_theta
#     theta0 = 1 / (1 + np.exp(-utheta0))  # sigmoid, inverse of logit
#     theta1 = max(1 / (1 + np.exp(-utheta1)), theta0)  # sigmoid + clip
#     theta2 = max(0, utheta2)  # clip
#     theta = np.array([theta0, theta1, theta2], dtype=np.float32)
#     _validate_model_params(theta)
#     return theta


# def _loss(cp_inputs: NDArray[np.float32], map_competences: List[float],
#           transformed_model_params: NDArray[np.float32]) -> float:
#     means = _model_predict_mean(cp_inputs, transformed_model_params)
#     variances = _get_predict_variances(means, np.array(map_competences))
#     betas = [_beta_from_mean_and_variance(m, v) for m, v in zip(means, variances)]
#     nlls = [
#         -beta_distribution.logpdf(c, a, b).item()
#         for c, (a, b) in zip(map_competences, betas)
#     ]
#     return sum(nlls)


# def _model_predict_mean(
#         x: NDArray[np.float32],
#         transformed_params: NDArray[np.float32]) -> NDArray[np.float32]:
#     params = _invert_transform_model_params(transformed_params)
#     _validate_model_params(params)
#     theta0, theta1, theta2 = params
#     out = theta0 + (theta1 - theta0) * (1 - np.exp(-theta2 * x))
#     assert np.all(out >= 0) and np.all(out <= 1)
#     out = np.clip(out, 1e-3, 1.0 - 1e-3)
#     return out


def _beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def _beta_variance(a: float, b: float) -> float:
    return (a * b) / ((a + b)**2 * (a + b + 1))


def _beta_from_mean_and_variance(mean: float,
                                 variance: float) -> Tuple[float, float]:
    alpha = ((1 - mean) / variance - 1 / mean) * (mean**2)
    beta = alpha * (1 / mean - 1)
    assert alpha > 0
    assert beta > 0
    assert abs(_beta_mean(alpha, beta) - mean) < 1e-6
    return (alpha, beta)


def _get_cp_model_inputs(history: List[List[bool]]) -> NDArray[np.float32]:
    num_data_after_cycle = list(np.cumsum([len(h) for h in history]))
    num_data_before_cycle = np.array([0] + num_data_after_cycle[:-1],
                                     dtype=np.float32)
    return num_data_before_cycle


def _run_em(
    history: List[List[bool]],
    num_em_iters: int = 10
) -> Tuple[List[NDArray[np.float32]], List[Tuple[float, float]], List[float]]:
    num_cycles = len(history)
    cp_inputs = _get_cp_model_inputs(history)
    # Initialize betas with uniform distribution.
    betas = [(1.0, 1.0) for _ in range(num_cycles)]
    all_map_competences = []
    all_model_params = []
    all_betas = []
    for it in range(num_em_iters):
        print(f"Starting EM cycle {it}")
        # Run inference.
        map_competences = _run_inference(history, betas)
        print("TODO TEMP OVERRIDE")
        map_competences = [1e-3 for _ in map_competences]
        print("MAP competences:", map_competences)
        all_map_competences.append(map_competences)
        # Run learning.
        model_params, variances = _run_learning(cp_inputs, map_competences)
        print("Model params:", model_params)
        print("Model variances:", variances)
        all_model_params.append(model_params)
        # Update betas by evaluating the model.
        means = _model_predict_mean(cp_inputs, _transform_model_params_to_unconstrain(model_params))
        betas = [_beta_from_mean_and_variance(m, v) for m, v in zip(means, variances)]
        print("Betas:", betas)
        print("Beta means:", [a / (a + b) for a, b in betas])
        all_betas.append(betas)
    return all_model_params, all_betas, all_map_competences


def _make_plots(history: List[List[bool]], all_betas: List[Tuple[float,
                                                                 float]],
                all_map_competences: List[float], outfile: Path) -> None:
    imgs: List[NDArray[np.uint8]] = []
    cp_inputs = _get_cp_model_inputs(history)
    for em_iter, (betas, map_competences) in enumerate(
            zip(all_betas, all_map_competences)):
        fig = plt.figure()
        plt.title(f"EM Iter {em_iter}")
        plt.xlabel("Skill Trial")
        plt.ylabel("Competence / Outcome")
        plt.xlim((min(cp_inputs) - 1, max(cp_inputs) + 1))
        plt.ylim((-0.25, 1.25))
        plt.yticks(np.linspace(0.0, 1.0, 5, endpoint=True))
        # Mark learning cycles.
        for i, x in enumerate(cp_inputs):
            label = "Learning Cycle" if i == 0 else None
            plt.plot((x, x), (-1.1, 2.1),
                     linestyle="--",
                     color="gray",
                     label=label)
        # Plot observation data.
        observations = [o for co in history for o in co]
        timesteps = np.arange(len(observations))
        plt.scatter(timesteps,
                    observations,
                    marker="o",
                    color="red",
                    label="Outcomes")
        # Plot competence progress model outputs (betas).
        means: List[float] = []
        stds: List[float] = []
        for a, b in betas:
            mean = _beta_mean(a, b)
            variance = _beta_variance(a, b)
            std = np.sqrt(variance)
            means.append(mean)
            stds.append(std)
        plt.plot(cp_inputs, means, color="blue", marker="+", label="CP Model")
        lb = np.subtract(means, stds)
        plt.plot(cp_inputs, lb, color="blue", linestyle="--")
        ub = np.add(means, stds)
        plt.plot(cp_inputs, ub, color="blue", linestyle="--")
        # Plot MAP competences.
        for cycle, cycle_map_competence in enumerate(map_competences):
            label = "MAP Competence" if cycle == 0 else None
            x_start = cp_inputs[cycle]
            if cycle == len(map_competences) - 1:
                x_end = x_start  # just a point
            else:
                x_end = cp_inputs[cycle + 1]
            y = cycle_map_competence
            plt.plot((x_start, x_end), (y, y),
                     color="green",
                     marker="*",
                     label=label)
        # Finish figure.
        plt.legend(loc="center right", framealpha=1.0)
        img = utils.fig2data(fig, dpi=300)
        imgs.append(img)
    utils.save_video(outfile, imgs)



def _test_inference() -> None:
    history = [
        [False, False, False, False, False],
        [False, False, False, False],
        [False, False, False, False, False, False, False],
        [False, False],
    ]
    betas = [(1.0, 1.0) for _ in history]
    map_competences =_run_inference(history, betas)
    assert all(p < 0.5 for p in map_competences)
    assert map_competences[0] < map_competences[1]

    history = [
        [True, True, True, True, True, True, True],
        [True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True],
    ]
    betas = [(1.0, 1.0) for _ in history]
    map_competences =_run_inference(history, betas)
    assert all(p > 0.5 for p in map_competences)
    assert map_competences[0] > map_competences[1]
    


# def _test_loss() -> None:

#     cp_inputs = np.array([0, 3, 10, 12])
#     map_competences = [1e-3, 1e-3, 1e-3, 1e-3]
#     good_model_params = np.array([1e-3, 1e-2, 1.0])
#     worse_model_params = np.array([1e-3, 0.99, 1.0])
#     u_good = _transform_model_params_to_unconstrain(good_model_params)
#     u_worse = _transform_model_params_to_unconstrain(worse_model_params)
#     good_loss = _loss(cp_inputs, map_competences, u_good)
#     worse_loss = _loss(cp_inputs, map_competences, u_worse)
#     assert good_loss < worse_loss

#     cp_inputs = np.array([0, 3, 10, 12])
#     map_competences = [0.999, 0.999, 0.999, 0.999]
#     good_model_params = np.array([0.99, 0.999, 1.0])
#     worse_model_params = np.array([1e-3, 0.99, 1.0])
#     u_good = _transform_model_params_to_unconstrain(good_model_params)
#     u_worse = _transform_model_params_to_unconstrain(worse_model_params)
#     good_loss = _loss(cp_inputs, map_competences, u_good)
#     worse_loss = _loss(cp_inputs, map_competences, u_worse)
#     assert good_loss < worse_loss


def _test_run_learning():

    cp_inputs = np.array([0, 3, 10, 12], dtype=np.float32)
    map_competences = [1e-3, 1e-3, 1e-3, 1e-3]
    model = _run_learning(cp_inputs, map_competences)
    predicted_competences = [model.predict(np.array([x])) for x in cp_inputs]
    assert np.allclose(predicted_competences, map_competences, atol=1e-2)

    cp_inputs = np.array([0, 3, 10, 12], dtype=np.float32)
    map_competences = [0.99, 0.99, 0.99, 0.99]
    model = _run_learning(cp_inputs, map_competences)
    predicted_competences = [model.predict(np.array([x])) for x in cp_inputs]
    assert np.allclose(predicted_competences, map_competences, atol=1e-2)

    cp_inputs = np.array([0, 3, 10, 12], dtype=np.float32)
    map_competences = [0.3, 0.4, 0.4, 0.7]
    model = _run_learning(cp_inputs, map_competences)
    predicted_competences = [model.predict(np.array([x])) for x in cp_inputs]
    assert predicted_competences[0] < predicted_competences[1]


def _main():

    # _test_inference()
    # _test_loss()
    _test_run_learning()


    # history = [
    #     [False, False, False, False, False],
    #     [False, False, False, False],
    #     [False, False, False, False, False, False, False],
    #     [False, False],
    # ]
    # _, all_betas, all_map_competences = _run_em(history)
    # _make_plots(history,
    #             all_betas,
    #             all_map_competences,
    #             outfile=Path("cp_model_all_false.mp4"))

    # history = [
    #     [True, True, True, True, True],
    #     [True, True, True, True, True, True, True],
    #     [True, True, True, True, True],
    #     [True, True, True, True, True, True, True, True, True],
    #     [True, True, True, True, True],
    # ]
    # _, all_betas, all_map_competences = _run_em(history)
    # _make_plots(history,
    #             all_betas,
    #             all_map_competences,
    #             outfile=Path("cp_model_all_true.mp4"))

    # history = [
    #     [False, False, False],
    #     [True, False, False, True, False, False, False, False, False],
    #     [False, True, True, False, True, False, False, False],
    #     [False],
    #     [True, True, False, False, True, True],
    #     [True, True, True],
    # ]
    # _, all_betas, all_map_competences = _run_em(history)
    # _make_plots(history,
    #             all_betas,
    #             all_map_competences,
    #             outfile=Path("cp_model_small_improve.mp4"))


if __name__ == "__main__":
    _main()
