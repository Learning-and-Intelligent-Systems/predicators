import logging
from typing import Optional, Sequence, Tuple

import pybullet as p

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion

# import numpy as np
default_orn: Sequence[float] = [0.0, 0.0, 0.0, 1.0]

def create_object(asset_path: str,
                  position: Pose3D = (0, 0, 0),
                  orientation: Quaternion = default_orn,
                  color: Optional[Tuple[float, float, float, float]] = None,
                  scale: float = 0.2,
                  mass: Optional[float] = None,
                  use_fixed_base: bool = False,
                  physics_client_id: int = 0) -> int:
    """Create a pot object in the environment."""
    obj_id = p.loadURDF(utils.get_env_asset_path(asset_path),
                        useFixedBase=use_fixed_base,
                        globalScaling=scale,
                        physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(obj_id,
                                      position,
                                      orientation,
                                      physicsClientId=physics_client_id)
    if color is not None:
        # Change color of the base link (link_id = -1)
        p.changeVisualShape(obj_id, -1, rgbaColor=color)
        # Change color of all links
        for link_id in range(p.getNumJoints(obj_id)):
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)
    
    if mass is not None:
        p.changeDynamics(obj_id, -1, mass=mass, 
                         physicsClientId=physics_client_id)

    return obj_id


def update_object(obj_id: int,
                  position: Optional[Pose3D] = None,
                  orientation: Quaternion = default_orn,
                  color: Optional[Tuple[float, float, float, float]] = None,
                  physics_client_id: int = 0) -> None:
    """Update the position and orientation of an object."""
    if position is not None:
        p.resetBasePositionAndOrientation(obj_id,
                                          position,
                                          orientation,
                                          physicsClientId=physics_client_id)
    if color is not None:
        # Change color of all links
        for link_id in range(-1, p.getNumJoints(obj_id)):
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)
