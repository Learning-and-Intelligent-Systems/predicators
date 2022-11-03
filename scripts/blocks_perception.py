"""An extremely brittle and blocks/LIS/Panda-specific perception pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import pybullet as p
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from predicators import utils
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_env import create_pybullet_block
from predicators.structs import Image, State
from predicators.utils import LineSegment

################################### Structs ###################################


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
    done_button: int
    y_slider: Optional[int]
    block_ids: List[int] = field(init=False, default_factory=list)


@dataclass
class BlockState:
    x: float
    y: float
    z: float


@dataclass
class Camera:
    name: str
    camera_distance: float
    camera_yaw: float
    camera_pitch: float
    camera_target: Tuple[float, float, float]

    @property
    def view_matrix(self) -> NDArray[np.float32]:
        return p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2)

    @staticmethod
    def get_proj_matrix(width: int, height: int) -> NDArray[np.float32]:
        # TODO reconsider this
        return p.computeProjectionMatrixFOV(fov=90,
                                            aspect=float(width / height),
                                            nearVal=0.1,
                                            farVal=100.0)

    def sync(self, scene: PybulletScene) -> None:
        camera_state = p.getDebugVisualizerCamera(
            physicsClientId=scene.physics_client_id)
        yaw, pitch, dist, target = camera_state[-4:]
        self.camera_yaw = yaw
        self.camera_pitch = pitch
        self.camera_target = target
        self.camera_distance = dist

    def take_picture(self, width: int, height: int,
                     scene: PyBulletScene) -> Image:
        camera_out = p.getCameraImage(width=width,
                                      height=height,
                                      viewMatrix=self.view_matrix,
                                      projectionMatrix=self.get_proj_matrix(
                                          width, height),
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                      physicsClientId=scene.physics_client_id)
        px = camera_out[2]
        rgb_array = np.array(px).reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3].astype(np.uint8)
        return rgb_array

    def reset_pybullet(self, scene: PybulletScene) -> None:
        p.resetDebugVisualizerCamera(self.camera_distance,
                                     self.camera_yaw,
                                     self.camera_pitch,
                                     self.camera_target,
                                     physicsClientId=scene.physics_client_id)


################################## Constants ##################################

FIND_BLOCK_METHOD = "manual"
RUN_MANUAL_CAMERA_CALIBRATION = True
DOWNSCALE = 2
BLOCK_COLORS = {
    # RGB
    "red": (120, 50, 50),
    "purple": (60, 60, 100),
    "orange": (160, 90, 60),
    "yellow": (160, 120, 60),
    "blue": (75, 100, 120),
}

################################## Functions ##################################


def parse_state_from_images(camera_images: Sequence[CameraImage]) -> State:
    # Downscale the images.
    if DOWNSCALE > 1:
        camera_images = [
            downscale_camera_image(im, DOWNSCALE) for im in camera_images
        ]
    # Initialize a PyBullet scene.
    scene = initialize_pybullet()
    # Calibrate the cameras.
    if RUN_MANUAL_CAMERA_CALIBRATION:
        for camera_image in camera_images:
            calibrate_camera(camera_image, scene)
            print("Calibrated camera. New state:")
            print(camera_image.camera)
    # Parse the state from each image.
    states = [parse_state_from_image(im, scene) for im in camera_images]
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

    # Disable the preview windows for faster rendering.
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                               False,
                               physicsClientId=physics_client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                               False,
                               physicsClientId=physics_client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                               False,
                               physicsClientId=physics_client_id)

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

    # If we're using manual block finding, set up the slider.
    if FIND_BLOCK_METHOD == "manual":
        y_lb = PyBulletBlocksEnv.y_lb
        y_ub = PyBulletBlocksEnv.y_ub
        y_init = (y_lb + y_ub) / 2.0
        y_slider = p.addUserDebugParameter("block",
                                           y_lb,
                                           y_ub,
                                           y_init,
                                           physicsClientId=physics_client_id)
    else:
        y_slider = None

    # Add a done button for manual anything.
    done_button = p.addUserDebugParameter("done",
                                          1,
                                          0,
                                          1,
                                          physicsClientId=physics_client_id)

    return PyBulletScene(physics_client_id=physics_client_id,
                         table_id=table_id,
                         done_button=done_button,
                         y_slider=y_slider)


def calibrate_camera(camera_image: CameraImage, scene: PyBulletScene) -> None:
    camera = camera_image.camera
    # Reset the camera to the default.
    p.resetDebugVisualizerCamera(camera.camera_distance,
                                 camera.camera_yaw,
                                 camera.camera_pitch,
                                 camera.camera_target,
                                 physicsClientId=scene.physics_client_id)
    # Click and drag to calibrate.
    init_button_val = p.readUserDebugParameter(
        scene.done_button, physicsClientId=scene.physics_client_id)
    while p.readUserDebugParameter(
            scene.done_button,
            physicsClientId=scene.physics_client_id) <= init_button_val:
        time.sleep(0.01)
        update_overlay_view(camera_image, scene)


def update_overlay_view(image: CameraImage, scene: PyBulletScene) -> None:
    # Get the image from the camera.
    camera = image.camera
    camera.sync(scene)
    pybullet_img = camera.take_picture(image.width, image.height, scene)
    # Show overlaid image in a separate window.
    overlaid_image = cv2.addWeighted(image.rgb, 0.3, pybullet_img, 0.7, 0)
    bgr_img = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Overlay View", bgr_img)


def reset_pybullet(camera: Camera, scene: PyBulletScene) -> None:
    # Reset the camera.
    camera.reset_pybullet(scene)
    # Destroy any existing blocks.
    old_block_ids = list(scene.block_ids)
    for block_id in old_block_ids:
        p.removeBody(block_id, physicsClientId=scene.physics_client_id)
        scene.block_ids.remove(block_id)


def add_new_block_to_scene(scene: PyBulletScene) -> int:
    # Create the block.
    color = (1.0, 0.0, 0.0, 1.0)
    block_size = PyBulletBlocksEnv.block_size
    half_extents = (block_size / 2.0, block_size / 2.0, block_size / 2.0)
    mass = 1000.0
    friction = 1.0
    orientation = [0.0, 0.0, 0.0, 1.0]
    block_id = create_pybullet_block(color, half_extents, mass, friction,
                                     orientation, scene.physics_client_id)
    # Initialize the block position.
    x = (PyBulletBlocksEnv.x_lb + PyBulletBlocksEnv.x_ub) / 2.
    y = (PyBulletBlocksEnv.y_lb + PyBulletBlocksEnv.y_ub) / 2.
    z = PyBulletBlocksEnv.table_height + block_size / 2.
    p.resetBasePositionAndOrientation(block_id, [x, y, z],
                                      orientation,
                                      physicsClientId=scene.physics_client_id)
    # Add the block to the scene.
    scene.block_ids.append(block_id)
    return block_id


def change_block_color(block_id: int, block_color: Tuple[float, float, float],
                       scene: PyBulletScene) -> None:
    rgba = block_color + (255, )
    norm_rgba = np.array(rgba) / 255.0
    p.changeVisualShape(block_id,
                        -1,
                        rgbaColor=norm_rgba,
                        physicsClientId=scene.physics_client_id)


def find_block_in_image(block_id: int, image: CameraImage,
                        scene: PyBulletScene) -> Optional[BlockState]:
    if FIND_BLOCK_METHOD == "manual":
        return manual_find_block_in_image(block_id, image, scene)
    raise NotImplementedError


def manual_find_block_in_image(block_id: int, image: CameraImage,
                               scene: PyBulletScene) -> Optional[BlockState]:
    # Get the current position and orientation.
    (x, _, z), orientation = p.getBasePositionAndOrientation(
        block_id, physicsClientId=scene.physics_client_id)

    # Loop until the user quits.
    init_button_val = p.readUserDebugParameter(
        scene.done_button, physicsClientId=scene.physics_client_id)
    while p.readUserDebugParameter(
            scene.done_button,
            physicsClientId=scene.physics_client_id) <= init_button_val:
        time.sleep(0.01)
        y = p.readUserDebugParameter(scene.y_slider,
                                     physicsClientId=scene.physics_client_id)
        p.resetBasePositionAndOrientation(
            block_id, [x, y, z],
            orientation,
            physicsClientId=scene.physics_client_id)
        update_overlay_view(image, scene)


def parse_state_from_image(camera_image: CameraImage,
                           scene: PyBulletScene) -> State:
    camera = camera_image.camera
    block_states: List[BlockState] = []

    # Reset the PyBullet scene.
    reset_pybullet(camera, scene)

    # Add blocks one at a time until we don't find any new ones.
    while True:
        # Have we found any new block to add?
        found_new_block = False
        # Create a new block.
        block_id = add_new_block_to_scene(scene)
        # Loop through all possible block colors.
        for color_name, block_color in BLOCK_COLORS.items():
            # Change the block color.
            change_block_color(block_id, block_color, scene)
            # Search for blocks of this color.
            block_state = find_block_in_image(block_id, camera_image, scene)
            # If a block was found, restart this process.
            if block_state is not None:
                found_new_block = True
                block_states.append(block_state)
                break
        # If we didn't find any new blocks, stop and remove the last block,
        # just for good measure.
        if not found_new_block:
            scene.block_ids.remove(block_id)
            break

    # Create a state from the found block states.
    return block_states_to_state(block_states)


#################################### Main ####################################

if __name__ == "__main__":
    right_camera = Camera(name="right",
                          camera_distance=1.0,
                          camera_yaw=43,
                          camera_pitch=-40,
                          camera_target=(1.07, 0.95, -0.11637961119413376))
    left_camera = Camera(name="left",
                         camera_distance=0.8,
                         camera_yaw=-55.0,
                         camera_pitch=-24.0,
                         camera_target=(1.65, 0.75, 0.42))

    color_imgs_path = Path("~/Desktop/blocks-images/color/")
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
    state = parse_state_from_images([
        right_camera_img,
        # left_camera_img
    ])
