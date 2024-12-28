from typing import Sequence, Optional

import pybullet as p
# import numpy as np

from predicators import utils
from predicators.envs.pybullet_env import PybulletEnv
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion

def create_object(asset_path: str,
                position: Pose3D, 
                orientation: Quaternion = PybulletEnv._default_orn, 
                color: Optional[Sequence[float, float, float, float]] = None,
                scale: float = 0.2, 
                use_fixed_base: bool = False,
                physics_client_id: int = 0) -> int:
    """Create a pot object in the environment.
    """
    obj_id = p.loadURDF(utils.get_env_asset_path(asset_path),
                        useFixedBase=use_fixed_base,
                        globalScaling=scale,
                        physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(obj_id, position, orientation, 
                                      physicsClientId=physics_client_id)
    if color is not None:
        # Change color of all links
        for link_id in range(p.getNumJoints(obj_id)):
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)

    return obj_id
