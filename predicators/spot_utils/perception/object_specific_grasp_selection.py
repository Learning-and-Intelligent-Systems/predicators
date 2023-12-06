"""Object-specific grasp selectors."""

from typing import Any, Callable, Dict, Tuple

import numpy as np
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


def _get_platform_grasp_pixel(rgbds: Dict[str, RGBDImageWithContext],
                              artifacts: Dict[str, Any],
                              camera_name: str) -> Tuple[int, int]:
    # This assumes that we have just navigated to the april tag and are now
    # looking down at the platform. We crop the top half of the image and
    # then use CV2 to find the blue handle inside of it.
    del artifacts  # not used
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

    return (x, y)


def _get_ball_grasp_pixel(rgbds: Dict[str, RGBDImageWithContext],
                          artifacts: Dict[str, Any],
                          camera_name: str) -> Tuple[int, int]:
    del rgbds
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
    return (pixels_in_mask[1][-1], pixels_in_mask[0][-1])


def _get_cup_grasp_pixel(rgbds: Dict[str, RGBDImageWithContext],
                         artifacts: Dict[str, Any],
                         camera_name: str) -> Tuple[int, int]:
    """There are two main ideas in this grasp selector:

    1. We want to select a point on the object that is reasonably
       well-surrounded by other points. In other words, we shouldn't
       try to grasp the object near its edge, because that can lead
       to grasp failures when there is slight noise in the mask.
    2. We want to select a point that is towards the top of the cup.
       This part is specific to the cup and is due to us wanting to
       have a consistent grasp to prepare consistent placing.
    """
    del rgbds
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
    # Finally, select a point in the upper percentile (towards the
    # top center of the cup).
    pixels_in_mask = np.where(surrounded_mask)
    percentile_idx = int(len(pixels_in_mask[0]) / 20)  # 5th percentile
    idx = np.argsort(pixels_in_mask[0])[percentile_idx]
    pixel = (pixels_in_mask[1][idx], pixels_in_mask[0][idx])
    return pixel


# Maps an object ID to a function from rgbds, artifacts and camera to pixel.
OBJECT_SPECIFIC_GRASP_SELECTORS: Dict[ObjectDetectionID, Callable[
    [Dict[str,
          RGBDImageWithContext], Dict[str, Any], str], Tuple[int, int]]] = {
              # Platform-specific grasp selection.
              AprilTagObjectDetectionID(411): _get_platform_grasp_pixel,
              # Ball-specific grasp selection.
              ball_obj: _get_ball_grasp_pixel,
              # Cup-specific grasp selection.
              cup_obj: _get_cup_grasp_pixel
          }
