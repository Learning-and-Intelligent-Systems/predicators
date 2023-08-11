from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import bernoulli
from scipy.optimize import minimize
from predicators import utils
    

def nondecreasing_model_predict(x, a, b, c, d):
    # https://arxiv.org/pdf/2103.10948.pdf
    # POW3 model
    # mu = a * (x + d) ** (-b) + c
    # out = 1.0 - np.exp(-mu)
    # if np.isnan(out):
    #     import ipdb; ipdb.set_trace()
    # return out

    # Logistic function
    return a / (1 + np.exp(-b * (x - d))) + c


def beta_model_predict(x, a, b, c, d, scale):
    """Returns alpha and beta for beta distribution.
    
    The mean of the distribution is guaranteed non-decreasing a function of x.
    """
    # scale = max(0, scale)  # scale needs to be nonnegative
    # Predict mean using a non-decreasing function.
    mean = np.clip(nondecreasing_model_predict(x, a, b, c, d), 1e-5, 1 - 1e-5)
    # mean = alpha / (alpha + beta)
    # beta = alpha * (1 / mean - 1)
    alpha = scale
    beta = alpha * (1 / mean - 1)
    return np.array([alpha, beta])


def loss(outcomes, rng, params, num_samples=100):
    if params[-1] <= 0:
        return np.inf
    cum_data = 0
    cum_loss = 0.0
    for cycle_outcomes in outcomes:
        a, b = beta_model_predict(cum_data, *params)
        for sample_comp in rng.beta(a, b, num_samples):
            sample_comp = np.clip(sample_comp, 1e-5, 1 - 1e-5)
            for o in cycle_outcomes:
                loss_o = -bernoulli.logpmf(float(o), sample_comp)
                assert not np.isnan(loss_o)
                cum_loss += loss_o
        cum_data += len(cycle_outcomes)
    print("Loss:", cum_loss)
    return cum_loss

    
def fit_beta_model(outcomes):
    rng = np.random.default_rng(0)
    f = partial(loss, outcomes, rng)
    x0 = np.array([
        1.0, 0.1, 0.0, 0.0, 1.0,
    ])
    res = minimize(f, x0, method="L-BFGS-B", bounds=[
        (1e-5, 100.0), (1e-5, 100.0), (-100.0, 100.0), (1e-5, 100.0), (1e-5, 100.0)
    ])
    model_params = res.x
    map_competences = 
    return model_params, map_competences





###############################################################################
#                                  Analysis                                   #
###############################################################################

def _make_plots(outcomes, model_params, outfile = "direct_script_out.png"):
    all_num_outcomes = [len(o) for o in outcomes]
    num_trials = sum(all_num_outcomes)
    cum_num_outcomes = np.cumsum(all_num_outcomes)
    plt.title(f"Direct Estimation")
    plt.xlabel("Skill Trial")
    plt.ylabel("Competence / Outcome")
    plt.xlim((-1, num_trials+1))
    plt.ylim((-0.25, 1.25))
    plt.yticks(np.linspace(0.0, 1.0, 5, endpoint=True))
    # Mark learning cycles.
    for i, x in enumerate(cum_num_outcomes):
        label = "Learning Cycle" if i == 0 else None
        plt.plot((x, x), (-1.1, 2.1), linestyle="--", color="gray", label=label)
    # Plot observation data.
    observations = [o for co in outcomes for o in co]
    timesteps = np.arange(len(observations))
    plt.scatter(timesteps, observations, marker="o", color="red", label="Outcomes")
    # Plot competence progress model.
    inputs = np.linspace(0, num_trials, 100)
    outputs = []
    lb = []
    ub = []
    for x in inputs:
        alpha, beta = beta_model_predict(x, *model_params)
        output = beta_distribution.mean(alpha, beta)
        var = beta_distribution.var(alpha, beta)
        outputs.append(output)
        lb.append(output - var)
        ub.append(output + var)
    plt.plot(inputs, outputs, color="blue", marker="+", label="CP Model")
    plt.plot(inputs, lb, color="blue", linestyle="--")
    plt.plot(inputs, ub, color="blue", linestyle="--")
    # # Plot MAP competences.
    # for cycle, cycle_map_competence in enumerate(map_competences):
    #     label = "MAP Competence" if cycle == 0 else None
    #     x_start = 0 if cycle == 0 else cum_num_outcomes[cycle-1]
    #     x_end = cum_num_outcomes[cycle]
    #     y = cycle_map_competence
    #     plt.plot((x_start, x_end), (y, y), color="green", label=label)
    # Finish figure.
    plt.legend(loc="center right", framealpha=1.0)
    plt.savefig(outfile)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    data = [
        [False, False, False],
        [True, False, False, True, False, False, False, False, False],
        [False, True, True, False, True, False, False, False],
        [False],
        [True, True, False, False, True, True],
        [True, True, True],
    ]
    mp_out, map_out = fit_beta_model(data)
    _make_plots(data, mp_out, map_out, outfile = "direct_script_out_v1.png")
    data = [
        [False, False, False, False, False],
        [False, False, False, False],
        [False, False, False, False, False, False, False],
        [False, False],
    ]
    mp_out, map_out = fit_beta_model(data)
    _make_plots(data, mp_out, map_out, outfile = "direct_script_out_v2.png")
    data = [
        [True, True, True, True, True],
        [True, True, True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True],

    ]
    mp_out, map_out = fit_beta_model(data)
    _make_plots(data, mp_out, map_out, outfile = "direct_script_out_v3.png")
    data = [
        [False, False, False],
        [True, False, False, True, False, False, False, False, False],
        [False, True, True, False, True, False, False, False],
        [False],
        [True, True, False, False, True, True],
        [True, True, False, False, True, True],
        [True, False, False, True, True, True, False, True, False],
        [False, True, True, True, False, True, True, True],
        [True, True, True, False, True],
        [True, True, True, True, False, True, True, False],
        [True, True, True, True, True, True, True, False, True],
    ]
    mp_out, map_out = fit_beta_model(data)
    _make_plots(data, mp_out, map_out, outfile = "direct_script_out_v4.png")
