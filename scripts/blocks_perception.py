"""An extremely brittle and blocks/LIS/Panda-specific perception pipeline."""

from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np
import pybullet as p
import time
from typing import Sequence
import imageio.v3 as iio
from matplotlib import pyplot as plt
from pathlib import Path
from predicators import utils
from predicators.utils import LineSegment
from predicators.structs import Image, State
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_env import create_pybullet_block


@dataclass
class CameraParams:
    camera_distance: float
    camera_yaw: float
    camera_pitch: float
    camera_target: Tuple[float, float, float]

PHYSICS_CLIENT_ID = p.connect(p.GUI)

DOWNSCALE = 3

BLOCK_COLORS = {
    # RGB
    "red": (120, 50, 50),
    "purple": (60, 60, 100),
    "orange": (160, 90, 60),
    "yellow": (160, 120, 60),
    "blue": (75, 100, 120),
}
INIT_COLOR_THRESH = 25
COLOR_THRESH_INC = 5
MARKER_THRESH = 500

CAMERA_TO_LINE = {
    "left": LineSegment(1030, 700, 250, 230),
    "right": LineSegment(120, 705, 975, 190),
}

CAMERA_TO_PARAMS = {
    "left": CameraParams(
        camera_distance=0.8,
        camera_yaw=90.0,
        camera_pitch=-24.0,
        camera_target=(1.65, 0.75, 0.42),
    ),
    "right": CameraParams(
        camera_distance=0.8,
        camera_yaw=-55.0,
        camera_pitch=-24.0,
        camera_target=(1.65, 0.75, 0.42),
    ),
}

def _show_image(img: Image, title: str) -> None:
    # plt.figure()
    # plt.title(title)
    # plt.imshow(img)
    # plt.tight_layout()
    # plt.show()
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_state_from_images(right_color_img: Image, left_color_img: Image,
                            debug: bool = False) -> State:
    # Convert to RBG for cv2. 
    right_color_img = cv2.cvtColor(right_color_img, cv2.COLOR_BGR2RGB)
    left_color_img = cv2.cvtColor(left_color_img, cv2.COLOR_BGR2RGB)

    # Display raw images.
    if debug:
        _show_image(right_color_img, "Original Right Image")
        _show_image(left_color_img, "Original Left Image")
    
    right_state = parse_state_from_image(right_color_img, "right", debug=debug)
    left_state = parse_state_from_image(left_color_img, "left", debug=debug)

    # TODO: Average states together.
    return right_state


def parse_state_from_image(image: Image, camera: str, debug: bool = False) -> State:

    # Set gravity.
    p.setGravity(0., 0., -10., physicsClientId=PHYSICS_CLIENT_ID)

    # Reset camera.
    camera_params = CAMERA_TO_PARAMS[camera]
    p.resetDebugVisualizerCamera(
        camera_params.camera_distance,
        camera_params.camera_yaw,
        camera_params.camera_pitch,
        camera_params.camera_target,
        physicsClientId=PHYSICS_CLIENT_ID)

    # Load table. Might not be needed later.
    table_pose = PyBulletBlocksEnv._table_pose
    table_orientation = PyBulletBlocksEnv._table_orientation
    table_id = p.loadURDF(
        utils.get_env_asset_path("urdf/table.urdf"),
        useFixedBase=True,
        physicsClientId=PHYSICS_CLIENT_ID)
    p.resetBasePositionAndOrientation(
        table_id,
        table_pose,
        table_orientation,
        physicsClientId=PHYSICS_CLIENT_ID)

    # Create a block.
    color = (1.0, 0.0, 0.0, 1.0)
    block_size = PyBulletBlocksEnv.block_size
    half_extents = (block_size / 2.0, block_size / 2.0, block_size / 2.0)
    mass = 1000.0
    friction = 1.0
    orientation = [0.0, 0.0, 0.0, 1.0]
    block_id = create_pybullet_block(color, half_extents, mass, friction, orientation,
                                     PHYSICS_CLIENT_ID)
    x = 1.35
    y = 0.7
    z = PyBulletBlocksEnv.table_height + block_size / 2.
    p.resetBasePositionAndOrientation(block_id, [x, y, z],
                                      orientation,
                                      physicsClientId=PHYSICS_CLIENT_ID)

    # Take an image.
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_params.camera_target,
            distance=camera_params.camera_distance,
            yaw=camera_params.camera_yaw,
            pitch=camera_params.camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=PHYSICS_CLIENT_ID)

    width = 1280 // DOWNSCALE
    height = 720 // DOWNSCALE

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(width / height),
        nearVal=0.1,
        farVal=100.0,
        physicsClientId=PHYSICS_CLIENT_ID)

    y_slider = p.addUserDebugParameter("block", PyBulletBlocksEnv.y_lb, PyBulletBlocksEnv.y_ub, y)

    while True:
        time.sleep(0.01)
        y = p.readUserDebugParameter(y_slider)
        p.resetBasePositionAndOrientation(block_id, [x, y, z],
                                          orientation,
                                          physicsClientId=PHYSICS_CLIENT_ID)

        camera_state = p.getDebugVisualizerCamera(physicsClientId=PHYSICS_CLIENT_ID)
        viewMat = camera_state[2]
        projMat = camera_state[3]
        
        (_, _, px, _,
            _) = p.getCameraImage(width=width,
                                height=height,
                                viewMatrix=viewMat,
                                projectionMatrix=projMat,
                                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                physicsClientId=PHYSICS_CLIENT_ID)

        rgb_array = np.array(px).reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3].astype(np.uint8)
        cv2.imshow("Captured", cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB))



if __name__ == "__main__":
    color_imgs_path = Path("~/Desktop/blocks-images/color/")
    right_camera_id = 231122071284
    left_camera_id = 231122071283
    img_id = 0
    right_color_img_path = color_imgs_path / f"color-{img_id}-{right_camera_id}.png"
    left_color_img_path = color_imgs_path / f"color-{img_id}-{left_camera_id}.png"
    right_color_img = iio.imread(right_color_img_path)
    left_color_img = iio.imread(left_color_img_path)
    state = parse_state_from_images(right_color_img, left_color_img, debug=False)
