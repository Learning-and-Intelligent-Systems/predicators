"""Object-specific grasp selectors."""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from bosdyn.client import math_helpers
from scipy.ndimage import convolve

from predicators.spot_utils.perception.cv2_utils import \
    find_color_based_centroid
from predicators.spot_utils.perception.perception_structs import \
    AprilTagObjectDetectionID, LanguageObjectDetectionID, ObjectDetectionID, \
    RGBDImageWithContext

ball_prompt = "/".join([
    "small white ball", "ping-pong ball", "snowball", "cotton ball",
    "white button"
])
ball_obj = LanguageObjectDetectionID(ball_prompt)
cup_obj = LanguageObjectDetectionID("yellow hoop toy/yellow donut")


def _get_platform_grasp_pixel(
    rgbds: Dict[str, RGBDImageWithContext], artifacts: Dict[str, Any],
    camera_name: str, rng: np.random.Generator
) -> Tuple[Tuple[int, int], Optional[math_helpers.Quat]]:
    # This assumes that we have just navigated to the april tag and are now
    # looking down at the platform. We crop the top half of the image and
    # then use CV2 to find the blue handle inside of it.
    del artifacts, rng  # not used
    rgb = rgbds[camera_name].rgb
    half_height = rgb.shape[0] // 2

    # Crop the bottom half of the image.
    img = rgb[half_height:]

    # Use CV2 to find a pixel.
    lo, hi = ((0, 130, 130), (130, 255, 255))

    cropped_centroid = find_color_based_centroid(img, lo, hi)
    if cropped_centroid is None:
        raise RuntimeError("Could not find grasp for platform from image.")

    # Undo cropping.
    cropped_x, cropped_y = cropped_centroid
    x = cropped_x
    y = cropped_y + half_height

    return (x, y), None


def _get_ball_grasp_pixel(
    rgbds: Dict[str, RGBDImageWithContext], artifacts: Dict[str, Any],
    camera_name: str, rng: np.random.Generator
) -> Tuple[Tuple[int, int], Optional[math_helpers.Quat]]:
    del rgbds, rng
    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[ball_obj][camera_name]
    except KeyError:
        raise ValueError(f"{ball_obj} not detected in {camera_name}")
    # Select the last (bottom-most) pixel from the mask. We do this because the
    # back finger of the robot gripper might displace the ball during grasping
    # if we try to grasp at the center.
    mask = seg_bb.mask
    pixels_in_mask = np.where(mask)
    pixel = (pixels_in_mask[1][-1], pixels_in_mask[0][-1])
    # Force a forward top-down grasp.
    rot_quat = math_helpers.Quat.from_pitch(np.pi / 2)
    return pixel, rot_quat


def _get_cup_grasp_pixel(
    rgbds: Dict[str, RGBDImageWithContext], artifacts: Dict[str, Any],
    camera_name: str, rng: np.random.Generator
) -> Tuple[Tuple[int, int], Optional[math_helpers.Quat]]:
    """There are two main ideas in this grasp selector:

    1. We want to select a point on the object that is reasonably
       well-surrounded by other points. In other words, we shouldn't
       try to grasp the object near its edge, because that can lead
       to grasp failures when there is slight noise in the mask.
    2. We want to select a point that is towards the top of the cup.
       This part is specific to the cup and is due to us wanting to
       have a consistent grasp to prepare consistent placing.
    """
    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[cup_obj][camera_name]
    except KeyError:
        raise ValueError(f"{cup_obj} not detected in {camera_name}")
    mask = seg_bb.mask
    # Start by denoising the mask, "filling in" small gaps in it.
    convolved_mask = convolve(mask.astype(np.uint8),
                              np.ones((3, 3)),
                              mode="constant")
    smoothed_mask = (convolved_mask > 0)
    # Now select points that are well surrounded by others.
    convolved_smoothed_mask = convolve(smoothed_mask.astype(np.uint8),
                                       np.ones((10, 10)),
                                       mode="constant")
    surrounded_mask = (
        convolved_smoothed_mask == convolved_smoothed_mask.max())
    pixels_in_mask = np.where(surrounded_mask)

    # Randomly select whether to grasp on the right or the top.
    grasp_modality = rng.choice(["right", "top"])

    if grasp_modality == "top":
        # Select a pixel near the top of the ring.
        percentile_idx = int(len(pixels_in_mask[0]) / 20)  # 5th percentile
        idx = np.argsort(pixels_in_mask[0])[percentile_idx]
    else:
        # Finally, select a point in the upper percentile (towards the
        # right center of the cup).
        percentile_idx = int(len(pixels_in_mask[0]) /
                             1.0526)  # 95th percentile
        idx = np.argsort(pixels_in_mask[1])[percentile_idx]

    pixel = (pixels_in_mask[1][idx], pixels_in_mask[0][idx])

    center = np.mean(pixels_in_mask, axis=1)
    center_pixel = (int(center[1]), int(center[0]))

    dy = pixel[1] - center_pixel[1]
    dx = pixel[0] - center_pixel[0]
    angle = np.arctan2(dx, -dy)

    del rgbds  # not used, except for debugging
    # Uncomment for debugging. Make sure also to not del rgbds (above).
    # import cv2
    # rgbd = rgbds[camera_name]
    # bgr = cv2.cvtColor(rgbd.rgb, cv2.COLOR_RGB2BGR)
    # cv2.circle(bgr, pixel, 5, (0, 255, 0), -1)
    # cv2.circle(bgr, center_pixel, 5, (255, 0, 0), -1)
    # cv2.arrowedLine(bgr, center_pixel,
    #                 (center_pixel[0] + dx, center_pixel[1] + dy), (255, 0, 0),
    #                 5)
    # cv2.imshow("Selected grasp", bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    yaw = math_helpers.Quat.from_yaw(-angle)
    pitch = math_helpers.Quat.from_pitch(np.pi / 2)
    rot_quat = yaw * pitch

    return pixel, rot_quat


# Maps an object ID to a function from rgbds, artifacts and camera to pixel.
OBJECT_SPECIFIC_GRASP_SELECTORS: Dict[ObjectDetectionID, Callable[[
    Dict[str, RGBDImageWithContext], Dict[str, Any], str, np.random.Generator
], Tuple[Tuple[int, int], Optional[math_helpers.Quat]]]] = {
    # Platform-specific grasp selection.
    AprilTagObjectDetectionID(411): _get_platform_grasp_pixel,
    # Ball-specific grasp selection.
    ball_obj: _get_ball_grasp_pixel,
    # Cup-specific grasp selection.
    cup_obj: _get_cup_grasp_pixel
}
