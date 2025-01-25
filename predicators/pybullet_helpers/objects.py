import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.utils import _Geom2D

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
        p.changeDynamics(obj_id,
                         -1,
                         mass=mass,
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


def sample_collision_free_2d_positions(
        num_samples: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        shape_type: str,
        shape_params: Sequence[float],
        rng: np.random.Generator,
        max_tries_per_object: int = 10,
        max_tries_total: int = 1000) -> List[Tuple[float, float]]:
    """Sample collision-free 2D positions.

    This function supports two shape types:
    - "circle": requires shape_params=[radius].
    - "rectangle": requires shape_params=[width, height, theta],
      where `theta` is the rotation in radians (between -pi and pi).

    It will sample positions inside the given (x_range, y_range) such that
    none of the shapes overlap.

    Args:
        num_samples (int): Number of positions to sample.
        x_range (Tuple[float, float]): The min and max bounds for x.
        y_range (Tuple[float, float]): The min and max bounds for y.
        shape_type (str): "circle" or "rectangle".
        shape_params (Sequence[float]): Shape-specific parameters.
        rng (np.random.Generator): Random generator for reproducible sampling.
        max_tries_per_object (int, optional): Number of attempts per object
            before discarding the entire arrangement and restarting. Defaults to 10.
        max_tries_total (int, optional): Maximum total attempts before giving up.
            Defaults to 1000.

    Returns:
        List[Tuple[float, float]]: A list of (x, y) positions for the shapes,
        guaranteed to be collision-free.
    """
    from predicators.utils import Circle, Rectangle  # or from .utils if needed

    def create_geom(px: float, py: float) -> _Geom2D:
        """Create the geometry object based on shape_type and shape_params."""
        if shape_type == "circle":
            # shape_params = [radius]
            (radius, ) = shape_params
            return Circle(px, py, radius)
        elif shape_type == "rectangle":
            # shape_params = [width, height, theta]
            w, h, theta = shape_params
            return Rectangle(px, py, w, h, theta)
        else:
            raise ValueError(f"Unsupported shape_type: {shape_type}")

    positions: List[Tuple[float, float]] = []
    collision_geoms: List[_Geom2D] = []

    total_tries = 0
    while True:
        positions.clear()
        collision_geoms.clear()
        for _ in range(num_samples):
            for _ in range(max_tries_per_object):
                total_tries += 1
                if total_tries > max_tries_total:
                    raise RuntimeError("Max tries exceeded. Unable to sample "
                                       "collision-free positions.")

                # Sample random position
                px = rng.uniform(x_range[0], x_range[1])
                py = rng.uniform(y_range[0], y_range[1])

                new_geom = create_geom(px, py)
                # Check intersection with existing
                if not any(new_geom.intersects(g) for g in collision_geoms):
                    # Found a valid position
                    positions.append((px, py))
                    collision_geoms.append(new_geom)
                    break
            else:
                # Failed to place this shape, restart entire process
                break
        else:
            # We successfully placed all shapes
            return positions
