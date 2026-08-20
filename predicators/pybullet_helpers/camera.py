"""PyBullet helpers for cameras and rendering."""

from typing import Any, Dict, Optional, Sequence

import pybullet as p

from predicators.pybullet_helpers.geometry import Pose3D

# A neutral, photo-studio background. Replaces PyBullet's default lavender
# clear color, which is the single biggest "this is a simulation" tell.
DEFAULT_BACKGROUND_RGB: Sequence[float] = (0.82, 0.83, 0.85)
# Off-axis, elevated key light (world frame). Gives objects form and a
# directional shadow instead of flat, ambient-only lighting.
DEFAULT_LIGHT_POSITION: Sequence[float] = (1.5, 0.5, 3.0)


def create_gui_connection(
        camera_distance: float = 0.8,
        camera_yaw: float = 90,
        camera_pitch: float = -24,
        camera_target: Pose3D = (1.65, 0.75, 0.42),
        disable_preview_windows: bool = True,
        background_rgb: Optional[Sequence[float]] = None,
        light_position: Optional[Sequence[float]] = None,
        shadow_map_resolution: Optional[int] = None,
        shadow_map_world_size: Optional[int] = None
) -> int:  # pragma: no cover
    """Creates a PyBullet GUI connection and initializes the camera.

    Returns the physics client ID for the connection.

    The optional visual arguments tune the look of the GUI window (they are
    opt-in so existing envs are unchanged unless they pass them):
    - ``background_rgb``: window clear color (defaults to PyBullet's when
      None). A neutral gray reads far more like a real scene than the
      default lavender.
    - ``light_position``: world-frame position of the GUI key light.
    - ``shadow_map_resolution`` / ``shadow_map_world_size``: shadow crispness.
      A *smaller* world size concentrates the shadow map on the workspace,
      giving sharper contact shadows so objects look seated, not floating.

    Not covered by unit tests because unit tests need to be headless.
    """
    # The GUI window clear color can only be set via connection options.
    if background_rgb is not None:
        options = (f"--background_color_red={background_rgb[0]} "
                   f"--background_color_green={background_rgb[1]} "
                   f"--background_color_blue={background_rgb[2]}")
        physics_client_id = p.connect(p.GUI, options=options)
    else:
        physics_client_id = p.connect(p.GUI)
    # Disable the PyBullet GUI preview windows for faster rendering.
    if disable_preview_windows:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,
                                   False,
                                   physicsClientId=physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                                   False,
                                   physicsClientId=physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                                   False,
                                   physicsClientId=physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                                   False,
                                   physicsClientId=physics_client_id)
    # Lighting and shadow tuning. Only forwarded when explicitly requested so
    # the default look is preserved for envs that don't opt in.
    light_kwargs: Dict[str, Any] = {}
    if light_position is not None:
        light_kwargs["lightPosition"] = light_position
    # PyBullet's C binding requires these be ints (a float raises TypeError).
    if shadow_map_resolution is not None:
        light_kwargs["shadowMapResolution"] = int(shadow_map_resolution)
    if shadow_map_world_size is not None:
        light_kwargs["shadowMapWorldSize"] = int(shadow_map_world_size)
    if light_kwargs:
        p.configureDebugVisualizer(physicsClientId=physics_client_id,
                                   **light_kwargs)
    p.resetDebugVisualizerCamera(camera_distance,
                                 camera_yaw,
                                 camera_pitch,
                                 camera_target,
                                 physicsClientId=physics_client_id)
    return physics_client_id
