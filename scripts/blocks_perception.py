"""An extremely brittle and blocks/LIS/Panda-specific perception pipeline."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import pybullet as p
from matplotlib import pyplot as plt

from predicators import utils
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_env import create_pybullet_block
from predicators.structs import Image, State
from predicators.utils import LineSegment

################################### Structs ###################################


@dataclass
class Camera:
    name: str
    camera_distance: float
    camera_yaw: float
    camera_pitch: float
    camera_target: Tuple[float, float, float]


@dataclass
class CameraImage:
    rgb: Image
    camera: Camera

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def height(self) -> int:
        return self.rgb.shape[0]


@dataclass
class PyBulletScene:
    physics_client_id: int
    table_id: int
    block_ids: List[int] = field(init=False, default_factory=list)


################################## Constants ##################################

DOWNSCALE = 3
BLOCK_COLORS = {
    # RGB
    "red": (120, 50, 50),
    "purple": (60, 60, 100),
    "orange": (160, 90, 60),
    "yellow": (160, 120, 60),
    "blue": (75, 100, 120),
}

CAMERAS = [
    Camera(name="left",
           camera_distance=0.8,
           camera_yaw=90.0,
           camera_pitch=-24.0,
           camera_target=(1.65, 0.75, 0.42)),
    Camera(name="right",
           camera_distance=0.8,
           camera_yaw=-55.0,
           camera_pitch=-24.0,
           camera_target=(1.65, 0.75, 0.42)),
]

################################## Functions ##################################


def parse_state_from_images(camera_images: Sequence[CameraImage]) -> State:
    # Downscale the images.
    if DOWNSCALE > 1:
        camera_images = [
            downscale_camera_image(im, DOWNSCALE) for im in camera_images
        ]
    # Initialize a PyBullet scene.
    physics_client_id = initialize_pybullet()
    # Parse the state from each image.
    states = [
        parse_state_from_image(im, physics_client_id) for im in camera_images
    ]
    # TODO: Average states together.
    return states[0]


def downscale_camera_image(camera_image: CameraImage,
                           scale: int) -> CameraImage:
    width = camera_image.width // scale
    height = camera_image.height // scale
    dim = (width, height)
    new_rgb = cv2.resize(camera_image.rgb, dim, interpolation=cv2.INTER_AREA)
    return CameraImage(rgb=new_rgb, camera=camera_image.camera)


def initialize_pybullet() -> PyBulletScene:
    physics_client_id = p.connect(p.GUI)  # TODO offer non-GUI option

    # Set gravity.
    p.setGravity(0., 0., -10., physicsClientId=physics_client_id)

    # Load table. Might not be needed later.
    table_pose = PyBulletBlocksEnv._table_pose
    table_orientation = PyBulletBlocksEnv._table_orientation
    table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                          useFixedBase=True,
                          physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(table_id,
                                      table_pose,
                                      table_orientation,
                                      physicsClientId=physics_client_id)

    return PyBulletScene(physics_client_id=physics_client_id,
                         table_id=table_id)


def reset_pybullet(camera: Camera, scene: PyBulletScene) -> None:
    # Reset the camera.
    p.resetDebugVisualizerCamera(camera.camera_distance,
                                 camera.camera_yaw,
                                 camera.camera_pitch,
                                 camera.camera_target,
                                 physicsClientId=scene.physics_client_id)

    # Destroy any existing blocks.
    old_block_ids = list(scene.block_ids)
    for block_id in old_block_ids:
        p.removeBody(block_id, physicsClientId=scene.physics_client_id)
        scene.block_ids.remove(block_id)


def parse_state_from_image(camera_image: CameraImage,
                           scene: PyBulletScene) -> State:
    rgb = camera_image.rgb
    camera = camera_image.camera

    # Reset the Pybullet scene.
    reset_pybullet(camera, scene)

    import ipdb
    ipdb.set_trace()


#################################### Main ####################################

if __name__ == "__main__":
    color_imgs_path = Path("~/Desktop/blocks-images/color/")
    left_camera, right_camera = CAMERAS
    assert right_camera.name == "right"
    assert left_camera.name == "left"
    right_camera_id = 231122071284
    left_camera_id = 231122071283
    img_id = 0
    right_color_img_path = color_imgs_path / f"color-{img_id}-{right_camera_id}.png"
    left_color_img_path = color_imgs_path / f"color-{img_id}-{left_camera_id}.png"
    right_color_img = iio.imread(right_color_img_path)
    left_color_img = iio.imread(left_color_img_path)
    right_camera_img = CameraImage(right_color_img, right_camera)
    left_camera_img = CameraImage(left_color_img, left_camera)
    state = parse_state_from_images([right_camera_img, left_camera_img])
