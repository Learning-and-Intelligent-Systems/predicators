"""Render initial states of the domino test tasks for debugging.

Reproduces the exact test tasks from a run (same env, seed,
test_env_seed_offset and domino flags) and saves a PNG of each test
task's initial state so failed tasks can be visualized.

Usage:
    PYTHONPATH=. python scripts/domino_debug/render_domino_initial_states.py
"""
import os

import numpy as np
from PIL import Image

from predicators import utils
from predicators.envs import create_new_env

# Domino flags copied verbatim from the run namespace (info.log) so the
# generated test tasks match the run exactly.
_DOMINO_FLAGS = {
    "env": "pybullet_domino",
    "num_train_tasks": 1,
    "num_test_tasks": 5,
    "test_env_seed_offset": 10000,
    "pybullet_camera_width": 1340,
    "pybullet_camera_height": 720,
    "domino_test_num_dominos": [3],
    "domino_test_num_targets": [1, 2],
    "domino_test_num_pivots": [0],
    "domino_test_num_pos_x": 4,
    "domino_test_num_pos_y": 3,
    "domino_train_num_dominos": [2],
    "domino_train_num_targets": [1],
    "domino_train_num_pivots": [0],
    "domino_train_num_pos_x": 3,
    "domino_train_num_pos_y": 2,
    "domino_use_continuous_place": True,
    "domino_use_domino_blocks_as_target": True,
    "domino_restricted_push": True,
    "domino_only_straight_sequence_in_training": True,
    "domino_use_skill_factories": True,
    "domino_prune_actions": False,
    "domino_has_glued_dominos": False,
    "domino_some_dominoes_are_connected": False,
    "domino_include_connected_predicate": False,
    "domino_initialize_at_finished_state": False,
    "domino_debug_layout": False,
    "domino_domino_on_stairs": False,
}

# Which 1-indexed test tasks failed in each seed (for labeling).
_FAILED = {0: {2, 3}, 2: {1, 2, 3, 5}}

_OUT_DIR = ("logs/agent_sim_learning/"
            "domino-agent_oracle_hybrid_sim_oracle_samplers/initial_states")


def main() -> None:
    """Render and save initial states of the domino test tasks."""
    os.makedirs(_OUT_DIR, exist_ok=True)
    for seed in (0, 2):
        utils.reset_config({**_DOMINO_FLAGS, "seed": seed})
        # do_cache=False: a cached env keeps its seed-0 test tasks, so each
        # seed must build a fresh env to regenerate its own test tasks.
        env = create_new_env("pybullet_domino", do_cache=False)
        tasks = env.get_test_tasks()
        for idx in range(len(tasks)):
            env.reset("test", idx)
            rgb = np.asarray(env.render()[0], dtype=np.uint8)
            task_num = idx + 1  # 1-indexed to match the run logs
            status = "FAILED" if task_num in _FAILED.get(seed, set()) \
                else "solved"
            fname = f"seed{seed}_task{task_num}_{status}.png"
            path = os.path.join(_OUT_DIR, fname)
            Image.fromarray(rgb).save(  # type: ignore[no-untyped-call]
                path)
            goal = sorted(str(a) for a in tasks[idx].goal)
            print(f"seed{seed} task{task_num} [{status}] -> {path}")
            print(f"    goal: {goal}")


if __name__ == "__main__":
    main()
