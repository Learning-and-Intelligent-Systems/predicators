"""Studio visuals for PyBullet environments.

Floor recolor, backdrop walls, GUI background/key-light/shadows, and table
textures -- the cosmetic "studio room" look shared by every PyBullet env. The
room geometry and key-light direction are derived from the env's camera when
not set explicitly, so the look adapts to each env automatically.

These helpers read the studio configuration straight off the env class (the
``_use_studio_visuals`` / ``floor_rgba`` / ``_camera_*`` / ``_gui_*`` ... class
vars defined on ``PyBulletEnv``). That keeps the per-env-overridable config on
the env while moving the rendering machinery out of the base class. ``env_cls``
is always a ``PyBulletEnv`` subclass.
"""
# These helpers deliberately read a PyBulletEnv subclass's (protected) studio
# config attributes -- that config lives on the env so subclasses can override
# it, and this module is just its rendering machinery split out for clarity.
# pylint: disable=protected-access
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.pybullet_helpers.camera import create_gui_connection
from predicators.pybullet_helpers.geometry import Pose3D


def wall_bounds(env_cls: Any) -> Optional[Dict[str, float]]:
    """Explicit ``_wall_bounds``, or a room derived from the camera.

    The derived room centers on the camera target and scales with the
    camera distance, sized so the camera sits comfortably inside.
    """
    if env_cls._wall_bounds is not None:
        return env_cls._wall_bounds
    tx, ty, _ = env_cls._camera_target
    half = env_cls._camera_distance * env_cls._studio_room_half_factor
    return {
        "x_min": tx - half,
        "x_max": tx + half,
        "y_min": ty - half,
        "y_max": ty + half,
        "height":
        env_cls._camera_distance * env_cls._studio_room_height_factor,
        "thickness": env_cls._studio_room_thickness,
    }


def light_direction(env_cls: Any) -> Pose3D:
    """Explicit ``_render_light_direction``, or a key derived from the camera.

    The derived light comes from the camera's horizontal side (so it
    lights camera-facing surfaces) and is elevated for a flattering top
    key.
    """
    if env_cls._render_light_direction is not None:
        return env_cls._render_light_direction
    theta = np.radians(env_cls._camera_yaw - 90.0)
    return (float(np.cos(theta)), float(np.sin(theta)),
            env_cls._studio_light_elevation)


def _gui_light_position(env_cls: Any) -> Tuple[float, float, float]:
    """Explicit ``_gui_light_position``, or a world point on the camera
    side."""
    if env_cls._gui_light_position is not None:
        return env_cls._gui_light_position
    tx, ty, tz = env_cls._camera_target
    theta = np.radians(env_cls._camera_yaw - 90.0)
    return (tx + 1.5 * float(np.cos(theta)), ty + 1.5 * float(np.sin(theta)),
            tz + 2.5)


def make_gui_connection(env_cls: Any) -> int:  # pragma: no cover
    """Open a GUI connection with the env's camera and studio look.

    The studio background / key light / shadow settings are forwarded
    only when ``_use_studio_visuals`` is set.
    """
    studio = env_cls._use_studio_visuals
    return create_gui_connection(
        camera_distance=env_cls._camera_distance,
        camera_yaw=env_cls._camera_yaw,
        camera_pitch=env_cls._camera_pitch,
        camera_target=env_cls._camera_target,
        background_rgb=env_cls._gui_background_rgb if studio else None,
        light_position=_gui_light_position(env_cls) if studio else None,
        shadow_map_resolution=(env_cls._gui_shadow_map_resolution
                               if studio else None),
        shadow_map_world_size=(env_cls._gui_shadow_map_world_size
                               if studio else None),
    )


def apply_floor(env_cls: Any, plane_id: int, physics_client_id: int) -> None:
    """Recolor the ground plane to ``floor_rgba`` (no-op if unset/disabled)."""
    if env_cls._use_studio_visuals and env_cls.floor_rgba is not None:
        p.changeVisualShape(plane_id,
                            -1,
                            rgbaColor=env_cls.floor_rgba,
                            physicsClientId=physics_client_id)


def create_walls(env_cls: Any, physics_client_id: int) -> List[int]:
    """Create visual-only backdrop walls (empty when disabled / no bounds).

    Walls carry no collision shape and are not part of the symbolic
    state; they exist purely so renders read like a room instead of an
    infinite plane. Four walls fully enclose the workspace (no ceiling,
    so overhead views still see in).
    """
    bounds = wall_bounds(env_cls)
    if not env_cls._use_studio_visuals or bounds is None:
        return []
    half_h = bounds["height"] / 2
    half_t = bounds["thickness"] / 2
    cx = (bounds["x_min"] + bounds["x_max"]) / 2
    cy = (bounds["y_min"] + bounds["y_max"]) / 2
    half_x = (bounds["x_max"] - bounds["x_min"]) / 2
    half_y = (bounds["y_max"] - bounds["y_min"]) / 2
    # (center, half_extents) for the back (+y), front (-y), left (-x) and
    # right (+x) walls -- a full enclosure with no ceiling.
    specs = [
        ((cx, bounds["y_max"], half_h), (half_x, half_t, half_h)),
        ((cx, bounds["y_min"], half_h), (half_x, half_t, half_h)),
        ((bounds["x_min"], cy, half_h), (half_t, half_y, half_h)),
        ((bounds["x_max"], cy, half_h), (half_t, half_y, half_h)),
    ]
    texture_id = None
    if env_cls.wall_texture_path is not None:
        texture_id = p.loadTexture(utils.get_env_asset_path(
            env_cls.wall_texture_path),
                                   physicsClientId=physics_client_id)
    base_color = (1, 1, 1, 1) if texture_id is not None else env_cls.wall_rgba
    wall_ids: List[int] = []
    for center, half_extents in specs:
        visual_id = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=half_extents,
                                        rgbaColor=base_color,
                                        physicsClientId=physics_client_id)
        body_id = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=-1,
                                    baseVisualShapeIndex=visual_id,
                                    basePosition=list(center),
                                    physicsClientId=physics_client_id)
        if texture_id is not None:
            p.changeVisualShape(body_id,
                                -1,
                                textureUniqueId=texture_id,
                                physicsClientId=physics_client_id)
        wall_ids.append(body_id)
    return wall_ids


def apply_table_textures(env_cls: Any, physics_client_id: int,
                         pybullet_bodies: Dict[str, Any]) -> None:
    """Texture every registered table body with the studio wood texture.

    Every env stores its table(s) under "table_id" (and "table_id2"), so
    this textures them all regardless of how the table was loaded
    (loadURDF, create_object, or a helper). No-op when disabled or no
    texture is set.
    """
    if not env_cls._use_studio_visuals or env_cls.table_texture_path is None:
        return
    texture_id = p.loadTexture(utils.get_env_asset_path(
        env_cls.table_texture_path),
                               physicsClientId=physics_client_id)
    for key, body_id in pybullet_bodies.items():
        if key.startswith("table_id") and isinstance(body_id, int):
            p.changeVisualShape(body_id,
                                -1,
                                textureUniqueId=texture_id,
                                rgbaColor=(1, 1, 1, 1),
                                physicsClientId=physics_client_id)
