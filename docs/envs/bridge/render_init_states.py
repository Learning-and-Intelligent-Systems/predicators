"""Render bridge init states (simple + full specs) to docs/envs/bridge."""

# Doc-asset generator; run from the repo root:
#     PYTHONPATH=. python docs/envs/bridge/render_init_states.py

import os

import imageio.v2 as imageio

from predicators import utils
from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.settings import CFG

OUT_DIR = "docs/envs/bridge"


def _main() -> None:
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 0,
        "num_train_tasks": 0,
        "num_test_tasks": 2,
        "pybullet_camera_width": 1674,
        "pybullet_camera_height": 900,
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    for spec in ("simple", "full"):
        CFG.bridge_task_spec_test = [spec]
        for seed in (0, 1, 2):
            CFG.seed = seed
            env = PyBulletBridgeEnv(use_gui=False)
            tasks = env._generate_test_tasks()  # pylint: disable=protected-access
            for task_idx, task in enumerate(tasks):
                env._set_state(task.init)  # pylint: disable=protected-access
                frame = env.render()[0]
                name = f"init_{spec}_seed{seed}_task{task_idx}.png"
                imageio.imwrite(os.path.join(OUT_DIR, name),
                                frame.astype("uint8"))
                print("wrote", name)
            del env


if __name__ == "__main__":
    _main()
