"""Environment rendering helpers shared by agent approaches and explorers.

Kept separate from the sketch/planning modules so those stay free of
``envs``/PIL dependencies.
"""
import logging
import os
from typing import Any, Optional

import numpy as np

from predicators.structs import EnvironmentTask, Task

logger = logging.getLogger(__name__)


def save_task_state_image(env: Any, task: Task, save_dir: str,
                          filename: str) -> Optional[str]:
    """Render ``task.init`` in ``env`` and save it under ``save_dir``.

    Shared by the solve-time approaches and the explorer so the agent
    can see the scene it is planning from. Best-effort by design: any
    rendering failure logs a warning and returns None instead of
    raising. Returns the absolute path of the saved image on success.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from PIL import Image as PILImage

        # For PyBullet envs, set state then use render() (render_state
        # raises NotImplementedError for arbitrary states). For other
        # envs, use render_state directly.
        try:
            from predicators.envs.pybullet_env import PyBulletEnv
            is_pybullet = isinstance(env, PyBulletEnv)
        except ImportError:
            is_pybullet = False

        if is_pybullet:
            env._set_state(task.init)  # pylint: disable=protected-access
            video = env.render()
        else:
            env_task = EnvironmentTask(task.init, task.goal)
            video = env.render_state(task.init, env_task)

        if not video:
            return None
        rgb_array = np.asarray(video[0], dtype=np.uint8)
        img = PILImage.fromarray(  # type: ignore[no-untyped-call]
            rgb_array)
        os.makedirs(save_dir, exist_ok=True)
        saved_path = os.path.join(save_dir, filename)
        img.save(saved_path)
        logger.info("Saved initial state image to %s", saved_path)
        return saved_path
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Failed to render initial state image: %s", e)
        return None
