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


# Just try to infer initial competence from some observations!

# Data
outcomes0 = [True, False, False, True, False, False, False, False, False]
# outcomes0 = [True, True, True, True]
num_obs0 = len(outcomes0)

# Create variables and initialize factor graph.
obs0 = vgroup.NDVarArray(num_states=2, shape=(num_obs0,))
competence0 = vgroup.NDVarArray(num_states=num_quants, shape=(1,))
fg = fgraph.FactorGraph(variable_groups=[obs0, competence0])

# Create prior factor.
a, b = 0.5, 0.5  # beta distribution parameters
def initial_competence_log_potential(competence):
    return beta.logpdf(competence, a, b)

quantized_initial_competence_log_potentials = _quantize(initial_competence_log_potential)
        
initial_competence_prior = fgroup.EnumFactorGroup(
    variables_for_factors=[[competence0[0]]],
    factor_configs=np.arange(num_quants)[:, None],
    log_potentials=quantized_initial_competence_log_potentials,
)

# Create unary factors for observed values.
visible_log_potentials = np.full((num_obs0, 2), -np.inf)
for i, out in enumerate(outcomes0):
    visible_log_potentials[i, int(out)] = 0
visible_unaries = fgroup.EnumFactorGroup(
    variables_for_factors=[[obs0[i]] for i in range(num_obs0)],
    factor_configs=np.arange(2)[:, None],
    log_potentials=visible_log_potentials,
)

# Create observation model factor.
def observation_log_potential(outcome, competence):
    return bernoulli.logpmf(outcome, competence)

quantized_observation_log_potentials = np.array([
    _quantize(partial(observation_log_potential, False)),
    _quantize(partial(observation_log_potential, True)),
])

observation_model0 = fgroup.PairwiseFactorGroup(
    variables_for_factors=[[obs0[i], competence0[0]] for i in range(num_obs0)],
    log_potential_matrix=quantized_observation_log_potentials,
)

# Finalize factor graph.
fg.add_factors([
    initial_competence_prior,
    visible_unaries,
    observation_model0
])

# Run MAP inference.
bp = infer.build_inferer(fg.bp_state, backend="bp")
bp_arrays = bp.run(bp.init(), num_iters=100, damping=0.5, temperature=0.0)
beliefs = bp.get_beliefs(bp_arrays)
map_states = infer.decode_map_states(beliefs)
map_competence0 = _quant_to_value(map_states[competence0][0])

# Print results.
print("Inference results...")
# for t, outcome in enumerate(outcomes0):
#     print(f"\nCycle 0 Time {t}\nOutcome={outcome}...\nObservation={map_states[obs0][t]}")
print(f"Competence 0: {map_competence0}")

import ipdb; ipdb.set_trace()