from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pgmax import fgraph, fgroup, infer, vgroup
from scipy.special import logsumexp
from scipy.stats import bernoulli, beta

# Quantizing.
num_quants = 100


def _get_quant_centers(start=0.0, end=1.0, num_quants=num_quants):
    lefts = np.linspace(start, end, num_quants, endpoint=False)
    delta = lefts[1] - lefts[0]
    return (lefts + delta / 2).squeeze()


def _quantize(fn, start=0.0, end=1.0, num_quants=num_quants):
    centers = _get_quant_centers(start, end, num_quants)
    unnormed = np.vectorize(fn)(centers)
    z = logsumexp(unnormed)
    return unnormed - z


def _quant_to_value(quant, start=0.0, end=1.0, num_quants=num_quants):
    centers = _get_quant_centers(start, end, num_quants)
    return centers[quant]


def _quantize2d(fn, start=0.0, end=1.0, num_quants=num_quants):
    # TODO figure out way to unify this.
    unnormed = np.empty((num_quants, num_quants))
    centers0 = _get_quant_centers(start, end, num_quants)
    centers1 = _get_quant_centers(start, end, num_quants)
    for i, center0 in enumerate(centers0):
        for j, center1 in enumerate(centers1):
            unnormed[i, j] = fn(center0, center1)
    z = logsumexp(unnormed.flat)
    return unnormed - z


# Data
outcomes = [
    [False, False, False],
    [True, False, False, True, False, False, False, False, False],
    [False, True, True, False, True, False, False, False],
    [False],
    [True, True, False, False, True, True],
    [True, True, True],
]
num_cycles = len(outcomes)
all_num_outcomes = [len(o) for o in outcomes]
cum_num_outcomes = np.cumsum(all_num_outcomes)

# Create variables and initialize factor graph.
obs_variable_groups = []
for cycle, cycle_outcomes in enumerate(outcomes):
    num_outcomes = len(cycle_outcomes)
    obs_cycle_group = vgroup.NDVarArray(num_states=2, shape=(num_outcomes, ))
    obs_variable_groups.append(obs_cycle_group)
competence_variable_group = vgroup.NDVarArray(num_states=num_quants,
                                              shape=(num_cycles, ))
variable_groups = obs_variable_groups + [competence_variable_group]
fg = fgraph.FactorGraph(variable_groups=variable_groups)

# Create factors.
factors = []

# Create prior factor for initial competence.
a, b = 0.5, 0.5  # beta distribution parameters


def initial_competence_log_potential(competence):
    return beta.logpdf(competence, a, b)


quantized_initial_competence_log_potentials = _quantize(
    initial_competence_log_potential)

initial_competence_prior = fgroup.EnumFactorGroup(
    variables_for_factors=[[competence_variable_group[0]]],
    factor_configs=np.arange(num_quants)[:, None],
    log_potentials=quantized_initial_competence_log_potentials,
)
factors.append(initial_competence_prior)

# Create unary factors for observed values.
for cycle, cycle_outcomes in enumerate(outcomes):
    num_outcomes = len(cycle_outcomes)
    visible_log_potentials = np.full((num_outcomes, 2), -np.inf)
    for i, out in enumerate(cycle_outcomes):
        visible_log_potentials[i, int(out)] = 0
    cycle_visible_factor = fgroup.EnumFactorGroup(
        variables_for_factors=[[obs_variable_groups[cycle][i]]
                               for i in range(num_outcomes)],
        factor_configs=np.arange(2)[:, None],
        log_potentials=visible_log_potentials,
    )
    factors.append(cycle_visible_factor)


# Create observation model factors.
def observation_log_potential(outcome, competence):
    return bernoulli.logpmf(outcome, competence)


quantized_observation_log_potentials = np.array([
    _quantize(partial(observation_log_potential, False)),
    _quantize(partial(observation_log_potential, True)),
])

for cycle, cycle_outcomes in enumerate(outcomes):
    num_outcomes = len(cycle_outcomes)
    cycle_observation_factor = fgroup.PairwiseFactorGroup(
        variables_for_factors=[[
            obs_variable_groups[cycle][i], competence_variable_group[cycle]
        ] for i in range(num_outcomes)],
        log_potential_matrix=quantized_observation_log_potentials,
    )
    factors.append(cycle_observation_factor)

# Create competence transition factors.
# TODO make these hyperparameters RVs.
progress_max = 1.0
progress_scale = 1.0
progress_midpoint = 50
logistic = lambda x: progress_max / (1 + np.exp(-progress_scale *
                                                (x - progress_midpoint)))


def create_competence_transition_log_potential(current_cycle):
    current_cum_outcomes = cum_num_outcomes[current_cycle]
    next_cum_outcomes = cum_num_outcomes[current_cycle + 1]
    assert next_cum_outcomes >= current_cum_outcomes
    # Expected gain is less when we already have a lot of data.
    current_logistic = logistic(current_cum_outcomes)
    next_logistic = logistic(next_cum_outcomes)
    expected_gain = next_logistic - current_logistic
    assert expected_gain >= 0

    def competence_transition_log_potential(current_competence,
                                            next_competence):
        if next_competence < current_competence:  # can't get worse!
            return -np.inf
        gain = next_competence - current_competence
        return (expected_gain - gain)**2

    return competence_transition_log_potential


for cycle in range(num_cycles - 1):
    current_competence_var = competence_variable_group[cycle]
    next_competence_var = competence_variable_group[cycle + 1]
    competence_transition_log_potential = create_competence_transition_log_potential(
        cycle)
    quantized_competence_transition_log_potential = _quantize2d(
        competence_transition_log_potential)
    cycle_competence_transition_factor = fgroup.PairwiseFactorGroup(
        variables_for_factors=[[current_competence_var, next_competence_var]],
        log_potential_matrix=quantized_competence_transition_log_potential,
    )
    factors.append(cycle_competence_transition_factor)

# Finalize factor graph.
fg.add_factors(factors)

# Run MAP inference.
bp = infer.build_inferer(fg.bp_state, backend="bp")
bp_arrays = bp.run(bp.init(), num_iters=100, damping=0.5, temperature=0.0)
beliefs = bp.get_beliefs(bp_arrays)
map_states = infer.decode_map_states(beliefs)

# Print results.
print("Inference results...")
map_competences = [
    _quant_to_value(map_states[competence_variable_group][i])
    for i in range(num_cycles)
]
for cycle in range(num_cycles):
    print(f"Competence {cycle}: {map_competences[cycle]}")

import ipdb

ipdb.set_trace()
