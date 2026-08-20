"""Test that evaluate_option_plan produces correct results.

Validates that multi-step option plans (Pick→Place→Pick→Place→Push) produce
non-zero actions at every step, both in-process and in a subprocess that
simulates the Docker agent flow (pickle/unpickle + recreate option model).

The original bug: BiRRT motion planning trajectories include the start
position as the first waypoint, so the first action is a no-op.  The option
model's "option got stuck" check (option_model_terminate_on_repeat) then
immediately aborts with 0 actions.

Usage:
    python tests/test_docker_option_plan.py              # run all tests
    python tests/test_docker_option_plan.py --child PKL  # (internal) subprocess
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any

import dill as pkl
import numpy as np

# Bootstrap circular imports
import predicators.utils as pred_utils
from predicators.settings import CFG

# Config matching predicatorv3/predicator_v3.yaml (mf_agent approach)
_CFG_OVERRIDES = {
    "env": "pybullet_domino",
    "approach": "agent_planner",
    "seed": 0,
    "use_gui": False,
    "domino_restricted_push": True,
    "domino_use_continuous_place": True,
    "domino_use_skill_factories": True,
    "domino_use_domino_blocks_as_target": True,
    "domino_has_glued_dominos": False,
    "domino_initialize_at_finished_state": False,
    "num_train_tasks": 1,
    "num_test_tasks": 1,
    "skill_phase_use_motion_planning": True,
    "pybullet_ik_validate": False,
}

OPTION_PLAN: list[dict[str, Any]] = [
    {
        "option_name": "Pick",
        "object_names": ["robot", "domino_2"],
        "params": []
    },
    {
        "option_name": "Place",
        "object_names": ["robot"],
        "params": [0.76, 1.26, -1.57]
    },
    {
        "option_name": "Pick",
        "object_names": ["robot", "domino_1"],
        "params": []
    },
    {
        "option_name": "Place",
        "object_names": ["robot"],
        "params": [0.86, 1.26, -1.57]
    },
    {
        "option_name": "Push",
        "object_names": ["robot"],
        "params": []
    },
]


def _setup_env_and_context() -> Any:
    """Create environment, options, option model, and ToolContext."""
    pred_utils.reset_config(_CFG_OVERRIDES)

    from predicators.envs import \
        create_new_env  # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import \
        get_gt_options  # pylint: disable=import-outside-toplevel
    from predicators.option_model import \
        create_option_model  # pylint: disable=import-outside-toplevel

    env = create_new_env(CFG.env, do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    predicates = env.predicates
    train_tasks = list(env.get_train_tasks())
    types = env.types
    task = train_tasks[0]

    option_model = create_option_model(CFG.option_model_name)

    from predicators.agent_sdk.tools import \
        ToolContext  # pylint: disable=import-outside-toplevel
    ctx = ToolContext(
        types=types,
        predicates=predicates,
        processes=set(),
        options=options,
        train_tasks=[t.task for t in train_tasks],
        example_state=task.init,
        option_model=option_model,
        current_task=task.task,
    )
    return ctx


def _run_option_plan(ctx: Any,
                     plan: list[dict[str, Any]] | None = None,
                     label: str = "") -> list[tuple[int, bool]]:
    """Run option plan and return list of (num_actions, state_changed)
    tuples."""
    if plan is None:
        plan = OPTION_PLAN

    task = ctx.current_task
    all_options = ctx.options | ctx.iteration_proposals.proposed_options
    opt_map = {o.name: o for o in all_options}

    state = task.init
    results = []

    for step_idx, opt_spec in enumerate(plan):
        opt_name = opt_spec["option_name"]
        obj_names = opt_spec["object_names"]
        params = opt_spec["params"]

        param_opt = opt_map[opt_name]
        obj_map = {o.name: o for o in state}
        objects = [obj_map[n] for n in obj_names]
        params_arr = np.array(params, dtype=np.float32)
        option = param_opt.ground(objects, params_arr)

        assert option.initiable(state), \
            f"Step {step_idx} ({opt_name}): not initiable"

        next_state, num_actions = \
            ctx.option_model.get_next_state_and_num_actions(state, option)

        state_changed = not _states_feature_equal(state, next_state)
        print(f"  [{label}] Step {step_idx} ({opt_name}): "
              f"{num_actions} actions, state_changed={state_changed}")

        if num_actions > 0:
            atoms_before = pred_utils.abstract(state, ctx.predicates)
            atoms_after = pred_utils.abstract(next_state, ctx.predicates)
            added = atoms_after - atoms_before
            deleted = atoms_before - atoms_after
            if added:
                print(f"    Added:   {sorted(str(a) for a in added)}")
            if deleted:
                print(f"    Deleted: {sorted(str(a) for a in deleted)}")

        results.append((num_actions, state_changed))
        state = next_state

    return results


def _states_feature_equal(s1: Any, s2: Any, atol: float = 1e-3) -> bool:
    """Compare two states by feature values only (ignoring simulator_state)."""
    if sorted(s1.data) != sorted(s2.data):
        return False
    for obj in s1.data:
        if not np.allclose(s1.data[obj], s2.data[obj], atol=atol):
            return False
    return True


def _rehash_objects_after_unpickle(ctx: Any) -> None:
    """Fix stale Object hash caches after cross-process unpickling."""
    from predicators.structs import \
        State  # pylint: disable=import-outside-toplevel

    seen = set()

    def _clear(obj: Any) -> None:
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        obj.__dict__.pop("_hash", None)
        obj.__dict__.pop("_str", None)

    def _process_state(state: Any) -> None:
        if state is None or not isinstance(state, State):
            return
        for obj in list(state.data.keys()):
            _clear(obj)
        state.data = dict(state.data.items())

    def _process_atoms(atoms: Any) -> None:
        for atom in atoms:
            for obj in atom.objects:
                _clear(obj)

    def _process_task(task: Any) -> None:
        if hasattr(task, "init"):
            _process_state(task.init)
        if hasattr(task, "init_obs"):
            _process_state(task.init_obs)
        for attr in ("goal", "alt_goal", "goal_description", "alt_goal_desc"):
            atoms = getattr(task, attr, None)
            if atoms:
                _process_atoms(atoms)

    for task in getattr(ctx, "train_tasks", []):
        _process_task(task)
    if ctx.current_task is not None:
        _process_task(ctx.current_task)
    _process_state(getattr(ctx, "example_state", None))


def main_parent() -> None:
    """Host-side: create context, run test, then spawn subprocess."""
    ctx = _setup_env_and_context()

    # --- TEST 1: host-side (same process) ---
    print("=== HOST TEST ===")
    host_results = _run_option_plan(ctx, label="HOST")

    host_ok = all(n > 0 and changed for n, changed in host_results)
    assert host_ok, \
        f"HOST: some steps returned 0 actions: {host_results}"

    # --- TEST 2: pickle and run in SUBPROCESS (simulates Docker) ---
    print("\n=== SUBPROCESS TEST (simulates Docker) ===")
    cfg_snapshot = dict(CFG.__dict__)

    pkl_path = os.path.join(tempfile.mkdtemp(), "test_input.pkl")
    with open(pkl_path, "wb") as f:
        pkl.dump({"tool_context": ctx, "cfg_snapshot": cfg_snapshot}, f)

    proc = subprocess.run(
        [sys.executable, __file__, "--child", pkl_path],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    print(proc.stdout)
    if proc.stderr:
        # Only show last part of stderr (warnings, etc.)
        print("STDERR (last 1000 chars):", proc.stderr[-1000:])
    os.unlink(pkl_path)

    assert proc.returncode == 0, \
        f"SUBPROCESS exited with code {proc.returncode}"

    print("\nAll tests passed.")


def main_child(pkl_path: str) -> None:
    """Child process: simulate Docker agent runner flow."""
    from predicators.option_model import \
        create_option_model  # pylint: disable=import-outside-toplevel

    with open(pkl_path, "rb") as f:
        loaded = pkl.load(f)

    if "cfg_snapshot" in loaded:
        for k, v in loaded["cfg_snapshot"].items():
            setattr(CFG, k, v)

    ctx = loaded["tool_context"]
    _rehash_objects_after_unpickle(ctx)

    # Recreate option model (physics server can't survive pickling)
    ctx.option_model = create_option_model(CFG.option_model_name)

    results = _run_option_plan(ctx, label="SUBPROCESS")

    all_ok = all(n > 0 and changed for n, changed in results)
    if not all_ok:
        print(f"SUBPROCESS FAILED: {results}")
        sys.exit(1)


if __name__ == "__main__":
    if "--child" in sys.argv:
        main_child(sys.argv[sys.argv.index("--child") + 1])
    else:
        main_parent()
