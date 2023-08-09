import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from functools import partial

from scipy.stats import beta, bernoulli
from scipy.special import logsumexp

from pgmax import fgraph, fgroup, infer, vgroup



# Quantizing.
num_quants = 20

def _get_quant_centers(start=0.0, end=1.0, num_quants=num_quants):
    lefts = np.linspace(start, end, num_quants, endpoint=False)
    delta = lefts[1] - lefts[0]
    return lefts + delta / 2

def _quantize(fn, start=0.0, end=1.0, num_quants=num_quants):
    centers = _get_quant_centers(start, end, num_quants)
    unnormed = np.vectorize(fn)(centers)
    z = logsumexp(unnormed)
    return unnormed - z

def _quant_to_value(quant, start=0.0, end=1.0, num_quants=num_quants):
    centers = _get_quant_centers(start, end, num_quants)
    return centers[quant]


# Data
outcomes = [
    [False, False, False],
    [True, False, False, True, False, False, False, False, False],
    [False, True, True, False, True, False, False, False],
    [True, True, False, False, True, True],
    [True, True, True],
]
num_cycles = len(outcomes)

# Create variables and initialize factor graph.
obs_variable_groups = []
for cycle, cycle_outcomes in enumerate(outcomes):
    num_outcomes = len(cycle_outcomes)
    obs_cycle_group = vgroup.NDVarArray(num_states=2, shape=(num_outcomes,))
    obs_variable_groups.append(obs_cycle_group)
competence_variable_group = vgroup.NDVarArray(num_states=num_quants, shape=(num_cycles,))
variable_groups = obs_variable_groups + [competence_variable_group]
fg = fgraph.FactorGraph(variable_groups=variable_groups)

# Create factors.
factors = []

# Create prior factor for initial competence.
a, b = 0.5, 0.5  # beta distribution parameters
def initial_competence_log_potential(competence):
    return beta.logpdf(competence, a, b)

quantized_initial_competence_log_potentials = _quantize(initial_competence_log_potential)
        
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
        variables_for_factors=[[obs_variable_groups[cycle][i]] for i in range(num_outcomes)],
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
        variables_for_factors=[[obs_variable_groups[cycle][i], competence_variable_group[cycle]] for i in range(num_outcomes)],
        log_potential_matrix=quantized_observation_log_potentials,
    )
    factors.append(cycle_observation_factor)

# Finalize factor graph.
fg.add_factors(factors)

# Run MAP inference.
bp = infer.build_inferer(fg.bp_state, backend="bp")
bp_arrays = bp.run(bp.init(), num_iters=100, damping=0.5, temperature=0.0)
beliefs = bp.get_beliefs(bp_arrays)
map_states = infer.decode_map_states(beliefs)

# Print results.
print("Inference results...")
# for t, outcome in enumerate(outcomes0):
#     print(f"\nCycle 0 Time {t}\nOutcome={outcome}...\nObservation={map_states[obs0][t]}")

map_competences = [_quant_to_value(map_states[competence_variable_group][i]) for i in range(num_cycles)]
for cycle in range(num_cycles):
    print(f"Competence {cycle}: {map_competences[cycle]}")

import ipdb; ipdb.set_trace()
