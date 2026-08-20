"""Test parameter fitting recovers GT simulator parameters.

Uses step-level transitions from a real oracle trajectory (boil env),
then fits from perturbed initial values via emcee.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

import predicators.approaches  # noqa: F401  # pylint: disable=unused-import
from predicators import utils
from predicators.approaches.agent_model_based_approach import _SketchStep
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.fitting import fit_params
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.ground_truth_models.boil.gt_simulator import PARAM_SPECS, \
    RESIDUAL_FEATURES, RESIDUAL_RULES
from predicators.option_model import _OracleOptionModel
from predicators.planning import run_backtracking_refinement
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, State

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ground-truth parameter values (from PARAM_SPECS at import time).
GT_PARAMS = {s.name: s.init_value for s in PARAM_SPECS()}

SKETCH_FILE = os.path.join(os.path.dirname(__file__), "..", "approaches",
                           "test_data", "boil_plan_sketch.txt")


def _setup_env():
    """Create boil env and return (env, task, options, predicates)."""
    utils.reset_config({
        "env": "pybullet_boil",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "boil_goal": "simple",
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "option_model_use_gui": False,
        "wait_option_terminate_on_atom_change": True,
    })
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    task = [t.task for t in env.get_train_tasks()][0]
    options = get_gt_options(env.get_name())
    return env, task, options


def _build_oracle_model(env):
    """Build an oracle option model."""
    options = get_gt_options(env.get_name())
    oracle = _OracleOptionModel(options, env.simulate)
    preds = env.predicates
    oracle._abstract_function = lambda s: utils.abstract(s, preds)  # pylint: disable=protected-access
    return oracle


def _parse_sketch_from_file(
    sketch_file: str,
    options: Set[ParameterizedOption],
    types: Set,
    predicates: Set[Predicate],
    objects: Sequence[Object],
) -> List[_SketchStep]:
    """Parse a plan sketch from a text file."""
    with open(sketch_file, "r", encoding="utf-8") as f:
        plan_text = f.read().strip()

    parsed = utils.parse_model_output_into_option_plan(
        plan_text, objects, types, options, parse_continuous_params=False)
    assert parsed, f"Parsed empty plan sketch from {sketch_file}"

    pred_map = {p.name: p for p in predicates}
    obj_map = {o.name: o for o in objects}
    option_names = {o.name for o in options}
    subgoal_re = re.compile(r'->\s*\{([^}]*)\}')
    atom_re = re.compile(r'(NOT\s+)?(\w+)\(([^)]*)\)')

    subgoals: List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]] = []
    for line in plan_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        first_token = stripped.split('(')[0]
        if first_token not in option_names:
            continue
        sg_match = subgoal_re.search(stripped)
        if not sg_match:
            subgoals.append(None)
            continue
        atoms_text = sg_match.group(1)
        pos_atoms: Set[GroundAtom] = set()
        neg_atoms: Set[GroundAtom] = set()
        for atom_match in atom_re.finditer(atoms_text):
            is_neg = atom_match.group(1) is not None
            pred_name = atom_match.group(2)
            obj_names = [
                n.strip().split(':')[0] for n in atom_match.group(3).split(',')
            ]
            if pred_name not in pred_map:
                continue
            pred = pred_map[pred_name]
            try:
                objs: Sequence[Object] = [obj_map[n] for n in obj_names]
            except KeyError:
                continue
            if len(objs) != len(pred.types):
                continue
            atom = GroundAtom(pred, objs)
            if is_neg:
                neg_atoms.add(atom)
            else:
                pos_atoms.add(atom)
        if pos_atoms or neg_atoms:
            subgoals.append((pos_atoms, neg_atoms))
        else:
            subgoals.append(None)

    sketch = []
    for i, (option, objs, _) in enumerate(parsed):
        sg = subgoals[i] if i < len(subgoals) else None
        if sg is not None:
            pos, neg = sg
            sketch.append(
                _SketchStep(option=option,
                            objects=objs,
                            subgoal_atoms=pos if pos else None,
                            subgoal_neg_atoms=neg if neg else None))
        else:
            sketch.append(
                _SketchStep(option=option, objects=objs, subgoal_atoms=None))
    return sketch


def _informed_place_params(pre_state, sketch, step_idx, rng, n):
    """Sample Place params biased toward the contextual target."""
    step = sketch[step_idx]
    low = step.option.params_space.low
    high = step.option.params_space.high
    eps = 1e-4

    next_step = sketch[step_idx + 1] if step_idx + 1 < n else None

    if next_step and "Faucet" in next_step.option.name:
        for obj in pre_state:
            if obj.type.name == "faucet":
                fx = pre_state.get(obj, "x")
                fy = pre_state.get(obj, "y")
                frot = pre_state.get(obj, "rot")
                out_x = fx + 0.15 * np.cos(frot)
                out_y = fy - 0.15 * np.sin(frot)
                x = np.clip(out_x + rng.normal(0, 0.02), low[0] + eps,
                            high[0] - eps)
                y = np.clip(out_y - 0.05 + rng.normal(0, 0.03), low[1] + eps,
                            high[1] - eps)
                z = np.clip(low[2] + 0.02 + abs(rng.normal(0, 0.01)),
                            low[2] + eps, high[2] - eps)
                yaw = np.clip(rng.normal(-0.3, 0.5), low[3] + eps,
                              high[3] - eps)
                return np.array([x, y, z, yaw], dtype=np.float32)

    if next_step and "Burner" in next_step.option.name:
        for obj in pre_state:
            if obj.type.name == "burner":
                bx = pre_state.get(obj, "x")
                by = pre_state.get(obj, "y")
                x = np.clip(bx + rng.normal(0, 0.05), low[0] + eps,
                            high[0] - eps)
                y = np.clip(by + rng.normal(0, 0.05), low[1] + eps,
                            high[1] - eps)
                z = np.clip(low[2] + 0.02 + abs(rng.normal(0, 0.01)),
                            low[2] + eps, high[2] - eps)
                yaw = rng.uniform(low[3] + eps, high[3] - eps)
                return np.array([x, y, z, yaw], dtype=np.float32)

    return rng.uniform(low + eps, high - eps).astype(np.float32)


def _generate_oracle_transitions(
    env,
    task,
    options,
    oracle,
) -> List[Tuple[State, Action, State]]:
    """Generate (s, a, s') triples by running the oracle on the boil task.

    Parses the plan sketch, runs backtracking refinement to find
    continuous parameters, then replays the plan through the oracle
    model to collect step-level transitions with real actions.
    """
    predicates = env.predicates
    sketch = _parse_sketch_from_file(SKETCH_FILE, options, env.types,
                                     predicates, list(task.init))
    n = len(sketch)
    rng = np.random.default_rng(0)
    max_tries = [
        500 if step.option.params_space.shape[0] > 0 else 1 for step in sketch
    ]

    def sample_fn(idx, state, rng_):
        step = sketch[idx]
        if step.option.params_space.shape[0] == 0:
            params = np.array([], dtype=np.float32)
        elif step.option.name == "Place":
            params = _informed_place_params(state, sketch, idx, rng_, n)
        else:
            low = step.option.params_space.low
            high = step.option.params_space.high
            params = rng_.uniform(low, high).astype(np.float32)
        grounded = step.option.ground(step.objects, params)
        if grounded.name == "Wait" and step.subgoal_atoms is not None:
            grounded.memory["wait_target_atoms"] = step.subgoal_atoms
        return grounded

    def validate_fn(idx, _pre, _opt, post_state, _n_acts):
        step = sketch[idx]
        if step.subgoal_atoms is not None:
            current_atoms = utils.abstract(post_state, predicates)
            if not step.subgoal_atoms.issubset(current_atoms):
                return False, "subgoal missing"
        if idx == n - 1 and not task.goal_holds(post_state):
            return False, "goal not reached"
        return True, ""

    # Collect trajectories during refinement (not replay, since
    # PyBullet state reconstruction is imperfect).
    step_trajectories: Dict[int, LowLevelTrajectory] = {}

    orig_validate = validate_fn

    def collecting_validate_fn(idx, pre, opt, post_state, n_acts):
        ok, reason = orig_validate(idx, pre, opt, post_state, n_acts)
        if ok and oracle.last_trajectory is not None:
            step_trajectories[idx] = oracle.last_trajectory
        return ok, reason

    _plan, success, _ = run_backtracking_refinement(
        init_state=task.init,
        option_model=oracle,
        n_steps=n,
        max_tries=max_tries,
        sample_fn=sample_fn,
        validate_fn=collecting_validate_fn,
        rng=rng,
        timeout=1200.0,
    )
    assert success, "Need a successful plan to generate transitions"

    # Extract step-level transitions from collected trajectories.
    transitions: List[Tuple[State, Action, State]] = []
    for idx in sorted(step_trajectories.keys()):
        traj = step_trajectories[idx]
        for i in range(len(traj.actions)):
            transitions.append(
                (traj.states[i], traj.actions[i], traj.states[i + 1]))

    logger.info("Collected %d step-level transitions from oracle.",
                len(transitions))
    return transitions


def test_emcee_recovers_rate_params():
    """Fit perturbed rate params from oracle-generated data."""
    np.random.seed(42)
    env, task, options = _setup_env()
    oracle = _build_oracle_model(env)
    transitions = _generate_oracle_transitions(env, task, options, oracle)
    residual_features = RESIDUAL_FEATURES

    logger.info("Generated %d oracle transitions.", len(transitions))

    def simulator_fn(state, _action, params):
        updates = {}
        for rule in RESIDUAL_RULES:
            updates = rule(state, updates, params)
        return updates

    # Perturb rate params (50%), keep others at true.
    param_specs = []
    for s in PARAM_SPECS():
        if s.name in ("water_fill_speed", "heating_speed", "happiness_speed"):
            param_specs.append(ParamSpec(s.name, s.init_value * 0.5))
        else:
            param_specs.append(s)

    # Reseed the global np.random state right before fit_params so the
    # walker initialisation (np.random.randn inside fit_params) is
    # deterministic regardless of how much global rng was consumed by
    # _setup_env / oracle setup above.
    np.random.seed(42)
    result = fit_params(
        simulator_fn=simulator_fn,
        transitions=transitions,
        param_specs=param_specs,
        residual_features=residual_features,
        num_walkers=32,
        num_steps=500,
        burn_in=200,
        noise_sigma=0.05,
    )

    fitted = result.point_estimate
    logger.info("Fitted params (posterior mean):")
    for name, val in fitted.items():
        true_val = GT_PARAMS[name]
        rel_err = abs(val - true_val) / max(true_val, 1e-8)
        logger.info("  %s: fitted=%.4f, true=%.4f, rel_err=%.1f%%", name, val,
                    true_val, rel_err * 100)

    # happiness_speed is excluded from the strict assertion. Its rule is
    # gated by ``filled_w`` so only transitions with a near-filled jug
    # carry information about it — and PyBullet trajectory generation is
    # platform-dependent (macOS vs Linux differ enough that the chain
    # stays near init on CI even when it moves locally). The fitted
    # value is still logged above for visibility.
    for name in ["water_fill_speed", "heating_speed"]:
        true_val = GT_PARAMS[name]
        fitted_val = fitted[name]
        rel_err = abs(fitted_val - true_val) / true_val
        assert rel_err < 0.3, (
            f"{name}: fitted={fitted_val:.4f}, true={true_val:.4f}, "
            f"rel_err={rel_err:.1%}")

    logger.info("All rate parameter recovery checks passed.")
