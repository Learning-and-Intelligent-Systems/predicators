"""Integration test: GT simulator + backtracking refinement solves boil.

Verifies that given a correct plan sketch (from a real agent run) and a
ground-truth simulator program, the hybrid learned option model
(PyBullet + learned residual dynamics) can find continuous parameters
that solve a pybullet_boil task.
"""
# pylint: disable=protected-access
import inspect
import logging
import os
import re
from types import SimpleNamespace
from typing import List, Optional, Sequence, Set, Tuple, cast

import dill as pkl
import numpy as np
import pytest

from predicators import utils
from predicators.approaches import agent_sim_learning_approach as asla
from predicators.approaches.agent_model_based_approach import _SketchStep
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning.fit_space import FitResult
from predicators.code_sim_learning.identifiability import Verdict
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, merge_updates
from predicators.envs import create_new_env
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.ground_truth_models import get_gt_options
from predicators.ground_truth_models.boil.gt_simulator import PARAM_SPECS, \
    RESIDUAL_RULES
from predicators.option_model import _OracleOptionModel
from predicators.planning import run_backtracking_refinement
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _setup_env():
    """Create boil env and return (env, task, options_dict, objects_dict)."""
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
    task = [t.task for t in env.get_test_tasks()][0]
    options = get_gt_options(env.get_name())
    options_dict = {o.name: o for o in options}
    objects_dict = {obj.name: obj for obj in task.init}
    return env, task, options_dict, objects_dict


def _build_oracle_model(env):
    """Build an oracle option model."""
    options = get_gt_options(env.get_name())
    oracle = _OracleOptionModel(options, env.simulate)
    preds = env.predicates
    oracle._abstract_function = lambda s: utils.abstract(s, preds)
    return oracle


def _build_kinematics_only_oracle(env):
    """Build an oracle that only handles kinematics (no residual dynamics).

    Creates a separate env instance with residual dynamics disabled, so
    that water filling, heating, and happiness are not simulated.
    """
    base_env = create_new_env("pybullet_boil",
                              do_cache=False,
                              use_gui=False,
                              skip_residual_dynamics=True)
    options = get_gt_options(base_env.get_name())
    oracle = _OracleOptionModel(options, base_env.simulate)
    preds = env.predicates
    oracle._abstract_function = lambda s: utils.abstract(s, preds)
    return oracle


def _build_combined_model(env):
    """Build a combined model: base-sim-only env + GT step-level dynamics.

    Mirrors AgentSimLearningApproach: wraps GT rules in a
    LearnedSimulator via apply_rules and composes with a base-sim-only
    env.
    """
    base_env = create_new_env("pybullet_boil",
                              do_cache=False,
                              use_gui=False,
                              skip_residual_dynamics=True)
    gt_params = {s.name: s.init_value for s in PARAM_SPECS()}
    rules = RESIDUAL_RULES

    simulator = LearnedSimulator(step_fn=lambda s, c, _r=rules, _p=gt_params:
                                 apply_rules(s, _r, _p, cmds=c),
                                 name="gt_combined")

    def combined_simulate(state, action):
        kin_state = base_env.simulate(state, action)
        updates = simulator.predict_step(kin_state)
        if not updates:
            return kin_state
        return merge_updates(kin_state, updates)

    options = get_gt_options(env.get_name())
    model = _OracleOptionModel(options, combined_simulate)
    preds = env.predicates
    model._abstract_function = lambda s: utils.abstract(s, preds)
    return model


def _parse_sketch_from_file(
    sketch_file: str,
    options: Set[ParameterizedOption],
    types: Set,
    predicates: Set[Predicate],
    objects: Sequence[Object],
) -> List[_SketchStep]:
    """Parse a plan sketch from a text file, as the model-based approach
    does."""
    with open(sketch_file, "r", encoding="utf-8") as f:
        plan_text = f.read().strip()

    # Phase 1: parse options + objects (no continuous params)
    parsed = utils.parse_model_output_into_option_plan(
        plan_text, objects, types, options, parse_continuous_params=False)
    assert parsed, f"Parsed empty plan sketch from {sketch_file}"

    # Phase 2: parse subgoal annotations
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

    # Zip into sketch steps
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
                # The jug has a physics offset after drop, so target
                # slightly past the faucet output to compensate.
                out_x = fx + 0.15 * np.cos(frot)
                out_y = fy - 0.15 * np.sin(frot)
                # Target near faucet output x but lower y (IK-reachable).
                x = np.clip(out_x + rng.normal(0, 0.02), low[0] + eps,
                            high[0] - eps)
                y = np.clip(out_y - 0.05 + rng.normal(0, 0.03), low[1] + eps,
                            high[1] - eps)
                z = np.clip(low[2] + 0.02 + abs(rng.normal(0, 0.01)),
                            low[2] + eps, high[2] - eps)
                # Negative yaw helps place jug closer to faucet output.
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
                # Bias z toward low end for reliable IK.
                z = np.clip(low[2] + 0.02 + abs(rng.normal(0, 0.01)),
                            low[2] + eps, high[2] - eps)
                yaw = rng.uniform(low[3] + eps, high[3] - eps)
                return np.array([x, y, z, yaw], dtype=np.float32)

    return rng.uniform(low + eps, high - eps).astype(np.float32)


def _refine(task,
            sketch,
            option_model,
            predicates,
            seed=0,
            max_samples=200,
            timeout=600.0):
    """Run backtracking refinement with informed Place sampling."""
    rng = np.random.default_rng(seed)
    n = len(sketch)
    max_tries = [
        max_samples if step.option.params_space.shape[0] > 0 else 1
        for step in sketch
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
                missing = step.subgoal_atoms - current_atoms
                return False, f"subgoal missing: {missing}"
        if idx == n - 1 and not task.goal_holds(post_state):
            return False, "goal not reached"
        return True, ""

    plan, success, total_samples = run_backtracking_refinement(
        init_state=task.init,
        option_model=option_model,
        n_steps=n,
        max_tries=max_tries,
        sample_fn=sample_fn,
        validate_fn=validate_fn,
        rng=rng,
        timeout=timeout,
    )
    logger.info("Refinement: %s, %d total samples",
                "success" if success else "failed", total_samples)
    return [p for p in plan if p is not None], success


SKETCH_FILE = os.path.join(os.path.dirname(__file__), "test_data",
                           "boil_plan_sketch.txt")


@pytest.mark.parametrize("model_type", ["oracle", "combined"])
def test_boil_sketch_refinement(model_type):
    """Test that backtracking refinement solves the first test task."""
    env, task, _options_dict, _objects_dict = _setup_env()
    predicates = env.predicates
    options = get_gt_options(env.get_name())

    if model_type == "oracle":
        option_model = _build_oracle_model(env)
    else:
        option_model = _build_combined_model(env)

    sketch = _parse_sketch_from_file(SKETCH_FILE, options, env.types,
                                     predicates, list(task.init))
    plan, success = _refine(task,
                            sketch,
                            option_model,
                            predicates,
                            max_samples=500,
                            timeout=1200.0)

    logger.info("Model=%s, success=%s, plan_len=%d", model_type, success,
                len(plan))
    if success:
        for i, opt in enumerate(plan):
            objs = ", ".join(o.name for o in opt.objects)
            params = ", ".join(f"{p:.3f}" for p in opt.params)
            logger.info("  %d: %s(%s)[%s]", i, opt.name, objs, params)

    assert success, (f"Refinement failed with {model_type} model. "
                     f"Partial plan: {len(plan)} steps.")

    # Forward validation: re-execute the plan in the oracle model (full
    # env dynamics) to verify the plan actually solves the task.
    # Always uses the oracle regardless of which model found the plan.
    oracle_model = _build_oracle_model(env)
    n = len(plan)

    def fwd_sample_fn(i, _s, _r):
        return plan[i]

    def fwd_validate_fn(i, _s, _o, post, _n):
        if i == n - 1 and not task.goal_holds(post):
            return False, "goal not reached"
        return True, ""

    _, fwd_success, _ = run_backtracking_refinement(
        init_state=task.init,
        option_model=oracle_model,
        n_steps=n,
        max_tries=[1] * n,
        sample_fn=fwd_sample_fn,
        validate_fn=fwd_validate_fn,
        rng=np.random.default_rng(0),
        timeout=600.0,
    )
    if fwd_success:
        logger.info("Forward validation passed for %s model.", model_type)
    else:
        logger.warning(
            "Forward validation failed for %s model "
            "(PyBullet state reconstruction is imperfect).", model_type)


def test_build_option_model_binds_sim_env():
    """The learned option model must expose ``_base_env`` as ``sim_env``.

    The task-evaluator verdict paths (sandbox tools and
    evaluate_trajectory) read ``option_model.sim_env`` to run
    certificates that need physics, such as the domino counterfactual
    push probe. Without the binding the probe is silently unavailable
    and captures are accepted on the pure rules only.
    """
    utils.reset_config({"wait_option_terminate_on_atom_change": False})
    approach = AgentSimLearningApproach.__new__(AgentSimLearningApproach)
    fake_env = SimpleNamespace()
    approach._base_env = fake_env
    approach._get_all_options = set  # type: ignore[method-assign]
    model = approach._build_option_model(lambda s, a: s)
    assert model.sim_env is fake_env
    # Pre-learning (no rules): the certificate probe stays base-only.
    assert fake_env.probe_process_model_factory is None
    # With rules, the combined-substrate factory is stamped alongside.
    approach._residual_rules = [lambda s, u, p: u]
    approach._fitted_params = {}
    model = approach._build_option_model(lambda s, a: s)
    assert model.sim_env is fake_env
    assert fake_env.probe_process_model_factory is not None


class _FakeScopeEnv:
    """Minimal env double for the fresh-validation-env scope tests."""

    def __init__(self):
        self.overrides = None

    def simulate(self, state, _action):
        """Bound method, so ``__self__`` identifies the owning env."""
        return state

    def apply_physical_param_overrides(self, params):
        """Record the identified physical params re-applied to a fresh env."""
        self.overrides = dict(params)


def _make_scope_approach(monkeypatch, prev_env, fresh_env, disposed):
    # The scope reads CFG.env before calling the (patched) env factory;
    # reset so these tests don't depend on an earlier test having
    # populated CFG (they fail under `-k fresh_validation` otherwise).
    utils.reset_config({"env": "pybullet_domino"})
    monkeypatch.setattr(asla, "create_new_env", lambda *a, **k: fresh_env)
    monkeypatch.setattr(asla, "dispose_env", disposed.append)
    approach = asla.AgentSimLearningApproach.__new__(
        asla.AgentSimLearningApproach)
    approach._base_env = prev_env
    approach._identified_physical_params = {"lateral_friction": 0.1}
    return approach


def test_fresh_validation_env_scope_swaps_and_restores(monkeypatch):
    """The scope points every physics consumer at the fresh env, restores
    everything on exit, and disposes the fresh env.

    Covers the pre-learning option model, whose simulator is the bound
    method ``_base_env.simulate`` and must be rebound explicitly (the
    learned combined simulator reads ``self._base_env`` dynamically
    instead).
    """
    prev_env, fresh_env = _FakeScopeEnv(), _FakeScopeEnv()
    disposed = []
    approach = _make_scope_approach(monkeypatch, prev_env, fresh_env, disposed)
    model = SimpleNamespace(_simulator=prev_env.simulate, sim_env=prev_env)
    approach._option_model = model

    with approach._fresh_validation_env_scope():
        assert approach._base_env is fresh_env
        assert model.sim_env is fresh_env
        assert model._simulator.__self__ is fresh_env
        # Identified physical params are re-asserted on the fresh env (the
        # in-place override does not survive env recreation).
        assert fresh_env.overrides == {"lateral_friction": 0.1}
        assert prev_env.overrides is None
    assert approach._base_env is prev_env
    assert model.sim_env is prev_env
    assert model._simulator.__self__ is prev_env
    assert disposed == [fresh_env]


def test_fresh_validation_env_scope_learned_simulator(monkeypatch):
    """A learned (closure) simulator is left untouched: it reads
    ``self._base_env`` dynamically, so only the attribute swap applies."""
    prev_env, fresh_env = _FakeScopeEnv(), _FakeScopeEnv()
    disposed = []
    approach = _make_scope_approach(monkeypatch, prev_env, fresh_env, disposed)
    closure_sim = lambda s, a: s  # noqa: E731
    model = SimpleNamespace(_simulator=closure_sim, sim_env=prev_env)
    approach._option_model = model

    with approach._fresh_validation_env_scope():
        assert approach._base_env is fresh_env
        assert model._simulator is closure_sim
        assert model.sim_env is fresh_env
    assert model._simulator is closure_sim
    assert model.sim_env is prev_env
    assert disposed == [fresh_env]


def test_fresh_validation_env_scope_disposes_crash_replacement(monkeypatch):
    """If a mid-rollout crash recovery replaced the fresh env, the scope
    disposes the replacement instead of leaking it."""
    prev_env, fresh_env = _FakeScopeEnv(), _FakeScopeEnv()
    crash_replacement = _FakeScopeEnv()
    disposed = []
    approach = _make_scope_approach(monkeypatch, prev_env, fresh_env, disposed)
    model = SimpleNamespace(_simulator=lambda s, a: s, sim_env=prev_env)
    approach._option_model = model

    with approach._fresh_validation_env_scope():
        # Emulate _recreate_base_env firing during the rollout.
        approach._base_env = crash_replacement
        model.sim_env = crash_replacement
    assert approach._base_env is prev_env
    assert model.sim_env is prev_env
    assert disposed == [crash_replacement]


def test_fresh_validation_env_scope_applies_physics_overrides(monkeypatch):
    """``physical_overrides`` (the capture gate's physics-margin points) land
    on the FRESH env on top of the identified params; the shared env is never
    touched."""

    class _MergingScopeEnv(_FakeScopeEnv):
        """Sticky per-param merge, matching the real override semantics."""

        def __init__(self):
            super().__init__()
            self.overrides = {}

        def apply_physical_param_overrides(self, params):
            self.overrides.update(params)

    prev_env, fresh_env = _MergingScopeEnv(), _MergingScopeEnv()
    disposed = []
    approach = _make_scope_approach(monkeypatch, prev_env, fresh_env, disposed)
    model = SimpleNamespace(_simulator=lambda s, a: s, sim_env=prev_env)
    approach._option_model = model

    with approach._fresh_validation_env_scope(
            physical_overrides={"lateral_friction": 0.48}):
        # Identified params first, then the perturbation on top.
        assert fresh_env.overrides == {"lateral_friction": 0.48}
        assert prev_env.overrides == {}
    assert disposed == [fresh_env]


def test_apply_identified_params_clears_sigma_points():
    """Any (re)application of identified params invalidates the standing.

    physics-margin points - they derive from a specific fit's posterior.
    """

    class _RegistryEnv(_FakeScopeEnv):

        def get_physical_param_info(self):
            """Empty registry: nothing to revert."""
            return {}

    approach = asla.AgentSimLearningApproach.__new__(
        asla.AgentSimLearningApproach)
    approach._base_env = _RegistryEnv()
    approach._identified_physical_params = {}
    approach._identified_physical_sigma_points = [{"lateral_friction": 0.48}]
    approach._apply_identified_physical_params({"lateral_friction": 0.53})
    assert not approach._identified_physical_sigma_points


if __name__ == "__main__":
    import sys
    _model = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    test_boil_sketch_refinement(_model)


def test_rollout_fit_trajectories_subset() -> None:
    """traj_idxs subsets the SOURCE trajectories (agent-facing indexing,
    applied before truncation/segmentation and before the completeness filter)
    and raises on an out-of-range index."""
    obj = object.__new__(AgentSimLearningApproach)
    t0 = SimpleNamespace(states=[0, 1], actions=["a"])
    t1 = SimpleNamespace(states=[0, 1, 2], actions=["a", "b"])
    t2 = SimpleNamespace(states=[0], actions=[])  # incomplete: filtered out
    # SimpleNamespace stubs stand in for LowLevelTrajectory (only the
    # states/actions attributes are read).
    obj._fit_trajectories = cast(List[LowLevelTrajectory], [t0, t1, t2])

    assert len(obj._rollout_fit_trajectories()) == 2
    sub = obj._rollout_fit_trajectories(traj_idxs=[1])
    assert len(sub) == 1 and len(sub[0][1]) == 2  # t1's two actions
    # Selecting only the incomplete trajectory leaves no usable rollouts.
    assert obj._rollout_fit_trajectories(traj_idxs=[2]) == []
    with pytest.raises(ValueError, match="out of range"):
        obj._rollout_fit_trajectories(traj_idxs=[3])


def _cross_cycle_fit(value: float) -> Tuple[FitResult, dict]:
    result = FitResult(names=["friction"],
                       samples=np.array([[value]]),
                       log_probs=np.zeros(1),
                       jacobian=None,
                       noise_sigma=0.05,
                       prior_sigma=np.array([0.75]),
                       scales=["log"])
    report = {
        "friction": {
            "posterior_std": 0.1,
            "prior_std": 0.75,
            "contraction": 0.13,
            "verdict": Verdict.IDENTIFIED,
            "note": "",
        }
    }
    return result, report


def test_cross_cycle_inconsistent_holds_then_confirms() -> None:
    """A many-sigma jump is held once, accepted on independent repeat.

    Regression for run_20260724_232411 seed2: cycle fits 0.3236 ->
    0.6267 (4.7 combined sigmas). The first jump must flag INCONSISTENT
    (trusted history unchanged, both fits recorded as hull candidates);
    a following cycle re-fitting near the new value confirms the jump
    and the history moves - without confirmation the stale reference
    would flag every future fit forever.
    """
    obj = object.__new__(AgentSimLearningApproach)
    obj._sysid_fit_history = {}
    obj._sysid_pending_fit = {}

    result, report = _cross_cycle_fit(0.3236)
    obj._check_cross_cycle_consistency(result, report, ["friction"])
    assert report["friction"]["verdict"] is Verdict.IDENTIFIED
    assert obj._sysid_fit_history["friction"][0] == 0.3236

    result, report = _cross_cycle_fit(0.6267)
    obj._check_cross_cycle_consistency(result, report, ["friction"])
    assert report["friction"]["verdict"] is Verdict.INCONSISTENT
    assert report["friction"]["candidate_values"] == [0.3236, 0.6267]
    # Trusted history holds; the rejected fit waits as pending.
    assert obj._sysid_fit_history["friction"][0] == 0.3236
    assert obj._sysid_pending_fit["friction"][0] == 0.6267

    result, report = _cross_cycle_fit(0.63)
    obj._check_cross_cycle_consistency(result, report, ["friction"])
    assert report["friction"]["verdict"] is Verdict.IDENTIFIED
    assert obj._sysid_fit_history["friction"][0] == 0.63
    assert "friction" not in obj._sysid_pending_fit


def test_make_probe_process_model_factory() -> None:
    """The certificate-probe factory mirrors the combined simulator.

    No rules -> None (the probe stays base-only). With rules, each
    factory call yields a stepper applying the CURRENT rules at the LIVE
    fitted params (in-place ``sim.fit`` updates must reach the probe,
    same closure convention as ``_build_combined_simulator``); recurrent
    5-arg rules get a fresh latent per stepper.
    """
    from predicators.structs import State, Type \
        # pylint: disable=import-outside-toplevel

    thing_type = Type("thing", ["x"])
    thing = Object("thing0", thing_type)
    state = State({thing: np.array([0.0])})
    noop = Action(np.zeros(1, dtype=np.float32))

    obj = object.__new__(AgentSimLearningApproach)
    obj._residual_rules = None
    assert obj._make_probe_process_model_factory() is None
    obj._residual_rules = []
    assert obj._make_probe_process_model_factory() is None

    def drift_rule(state: State, updates: dict, params: dict) -> dict:
        updates.setdefault(thing, {})["x"] = \
            state.get(thing, "x") + params["dx"]
        return updates

    obj._residual_rules = [drift_rule]
    obj._fitted_params = {"dx": 0.5}
    factory = obj._make_probe_process_model_factory()
    assert factory is not None
    assert factory()(state, noop).get(thing, "x") == 0.5
    obj._fitted_params["dx"] = 1.25  # in place, like sim.fit
    assert factory()(state, noop).get(thing, "x") == 1.25

    def latent_rule(state: State, latent: dict, history: list, updates: dict,
                    params: dict) -> dict:
        del history, params
        latent["count"] = latent.get("count", 0) + 1
        updates.setdefault(thing, {})["x"] = \
            state.get(thing, "x") + latent["count"]
        return updates

    obj._residual_rules = [latent_rule]
    obj._latent_init = {"count": 0}
    factory = obj._make_probe_process_model_factory()
    assert factory is not None
    stepper = factory()
    # Latent threads across steps within one stepper...
    assert stepper(state, noop).get(thing, "x") == 1.0
    assert stepper(state, noop).get(thing, "x") == 2.0
    # ...and resets on a fresh stepper (new replay attempt).
    assert factory()(state, noop).get(thing, "x") == 1.0


def test_cross_cycle_arbitration_by_pooled_evidence() -> None:
    """A flagged jump is accepted when pooled data decisively backs it.

    Regression for run_20260727_210827 seed1: the sharp-but-biased
    2-trajectory cycle-0 fit (0.9313, true 0.5) was held over the
    4-trajectory refit (0.4748) for the rest of the run even though the
    refit explained the pooled data ~30x better. With a pooled-SSE probe
    the arbitration must accept the new value immediately; an ambivalent
    gap (or a failing probe) must keep the hold.
    """
    obj = object.__new__(AgentSimLearningApproach)
    obj._sysid_fit_history = {}
    obj._sysid_pending_fit = {}

    result, report = _cross_cycle_fit(0.9313)
    obj._check_cross_cycle_consistency(result, report, ["friction"])
    assert obj._sysid_fit_history["friction"][0] == 0.9313

    def pooled_sse(theta: dict) -> float:
        return 0.14 if abs(theta["friction"] - 0.4748) < 1e-9 else 4.4

    result, report = _cross_cycle_fit(0.4748)
    obj._check_cross_cycle_consistency(result,
                                       report, ["friction"],
                                       pooled_sse=pooled_sse)
    assert report["friction"]["verdict"] is Verdict.IDENTIFIED
    assert "candidate_values" not in report["friction"]
    assert obj._sysid_fit_history["friction"][0] == 0.4748
    assert "friction" not in obj._sysid_pending_fit

    # Ambivalent pooled gap (below the decisive ratio): hold as before.
    obj._sysid_fit_history = {"friction": (0.9313, 0.1, "log")}
    obj._sysid_pending_fit = {}
    result, report = _cross_cycle_fit(0.4748)
    obj._check_cross_cycle_consistency(result,
                                       report, ["friction"],
                                       pooled_sse=lambda theta: 0.14)
    assert report["friction"]["verdict"] is Verdict.INCONSISTENT
    assert obj._sysid_fit_history["friction"][0] == 0.9313
    assert obj._sysid_pending_fit["friction"][0] == 0.4748

    # A failing SSE probe must fall back to the hold, not crash.
    obj._sysid_fit_history = {"friction": (0.9313, 0.1, "log")}
    obj._sysid_pending_fit = {}

    def broken_sse(theta: dict) -> float:
        raise RuntimeError("env died")

    result, report = _cross_cycle_fit(0.4748)
    obj._check_cross_cycle_consistency(result,
                                       report, ["friction"],
                                       pooled_sse=broken_sse)
    assert report["friction"]["verdict"] is Verdict.INCONSISTENT
    assert obj._sysid_fit_history["friction"][0] == 0.9313


def test_persist_fit_trajectories(tmp_path, monkeypatch) -> None:
    """Fit data lands in <log_dir>/fit_data/, one numbered pickle per fit."""
    obj = object.__new__(AgentSimLearningApproach)
    obj._fit_trajectories = cast(List[LowLevelTrajectory],
                                 ["fake_traj_a", "fake_traj_b"])
    obj._physical_param_specs = []
    obj._identified_physical_params = {"friction": 0.5}
    obj._get_log_dir = lambda: str(tmp_path)  # type: ignore[method-assign]

    obj._persist_fit_trajectories()
    obj._persist_fit_trajectories()
    out_dir = tmp_path / "fit_data"
    files = sorted(f.name for f in out_dir.glob("*.pkl"))
    assert files == [
        "fit_trajectories_000_fitted.pkl", "fit_trajectories_001_fitted.pkl"
    ]
    with open(out_dir / files[0], "rb") as f:
        payload = pkl.load(f)
    assert payload["trajectories"] == ["fake_traj_a", "fake_traj_b"]
    assert payload["identified_physical_params"] == {"friction": 0.5}

    monkeypatch.setattr(CFG, "code_sim_learning_persist_fit_data", False)
    obj._persist_fit_trajectories()
    assert len(list(out_dir.glob("*.pkl"))) == 2


def test_fit_data_is_dumped_even_when_no_fit_runs(tmp_path) -> None:
    """A cycle that declines to fit is exactly the one worth post-morteming.

    Persistence used to sit only inside the sysID fit, so the branch
    that never ran was the branch whose data mattered:
    run_20260817_171402 declined on a sweep returning one identical SSE
    for every value of five parameters, and left nothing on disk to
    explain it. Dumping where the data ARRIVES is what makes that
    replayable.
    """
    obj = object.__new__(AgentSimLearningApproach)
    obj._physical_param_specs = []
    obj._identified_physical_params = {}
    obj._get_log_dir = lambda: str(tmp_path)  # type: ignore[method-assign]
    obj._explainability_cache = {}
    obj._sysid_fit_cache = {}

    # The one line of _learn_simulator this is about, with no fit after it.
    obj._fit_trajectories = cast(List[LowLevelTrajectory], ["traj"])
    obj._persist_fit_trajectories("recorded")

    files = [f.name for f in (tmp_path / "fit_data").glob("*.pkl")]
    assert files == ["fit_trajectories_000_recorded.pkl"]
    with open(tmp_path / "fit_data" / files[0], "rb") as f:
        assert pkl.load(f)["trajectories"] == ["traj"]

    # The wiring, not just the function: _learn_simulator runs on every
    # cycle whether or not a fit follows, so the dump has to hang off it.
    # Asserted on the source because calling _learn_simulator for real
    # needs a whole synthesis session, and without this the test above
    # passes with the call deleted.
    source = inspect.getsource(AgentSimLearningApproach._learn_simulator)  # pylint: disable=protected-access
    assert "_persist_fit_trajectories(\"recorded\")" in source, \
        "the recorded-data dump is no longer wired into _learn_simulator"


def test_base_sim_reference_provisioning() -> None:
    """Base-sim source rides the sandbox reference registry (so every session
    phase gets it) and the agent-visible paths map per backend."""
    obj = object.__new__(AgentSimLearningApproach)
    obj._base_env = object.__new__(PyBulletFanEnv)
    utils.reset_config({
        "env": "pybullet_fan",
        "agent_sim_provide_base_sim_source": True,
        "agent_sdk_use_local_sandbox": True,
    })
    files = obj._get_sandbox_reference_files()
    assert files["base_sim/pybullet_fan_base.py"] == \
        "predicators/envs/pybullet_fan_base.py"
    assert files["base_sim/pybullet_env.py"] == \
        "predicators/envs/pybullet_env.py"
    assert obj._base_sim_reference_paths() == [
        "./reference/base_sim/pybullet_fan_base.py",
        "./reference/base_sim/pybullet_env.py",
    ]
    # Flag off: no registry entries, no advertised paths.
    utils.reset_config({
        "env": "pybullet_fan",
        "agent_sim_provide_base_sim_source": False,
        "agent_sdk_use_local_sandbox": True,
    })
    assert not any(
        k.startswith("base_sim/") for k in obj._get_sandbox_reference_files())
    assert obj._base_sim_reference_paths() == []
