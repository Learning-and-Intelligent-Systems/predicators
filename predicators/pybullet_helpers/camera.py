"""PyBullet helpers for cameras and rendering."""

import pybullet as p

from predicators.pybullet_helpers.geometry import Pose3D


def create_gui_connection(
        camera_distance: float = 0.8,
        camera_yaw: float = 90,
        camera_pitch: float = -24,
        camera_target: Pose3D = (1.65, 0.75, 0.42),
        disable_preview_windows: bool = True) -> int:  # pragma: no cover
    """Creates a PyBullet GUI connection and initializes the camera.

    Returns the physics client ID for the connection.

    Not covered by unit tests because unit tests need to be headless.
    """
    physics_client_id = p.connect(p.GUI, options='--background_color_red=0.0 --background_color_green=0.0 --background_color_blue=0.0')
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
    p.resetDebugVisualizerCamera(camera_distance,
                                 camera_yaw,
                                 camera_pitch,
                                 camera_target,
                                 physicsClientId=physics_client_id)
    return physics_client_id
