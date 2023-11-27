"""Object-specific grasp selectors."""

from typing import Any, Callable, Dict, Tuple

from predicators.spot_utils.perception.cv2_utils import \
    find_color_based_centroid
from predicators.spot_utils.perception.perception_structs import \
    AprilTagObjectDetectionID, LanguageObjectDetectionID, ObjectDetectionID, \
    RGBDImageWithContext

ball_prompt = "/".join([
    "small white ball",
    "ping-pong ball",
    "snowball",
    "cotton ball",
])
ball_obj = LanguageObjectDetectionID(ball_prompt)


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
    x1, y1, x2, y2 = seg_bb.bounding_box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# Maps an object ID to a function from rgbds, artifacts and camera to pixel.
OBJECT_SPECIFIC_GRASP_SELECTORS: Dict[ObjectDetectionID, Callable[
    [Dict[str,
          RGBDImageWithContext], Dict[str, Any], str], Tuple[int, int]]] = {
              # Platform-specific grasp selection.
              AprilTagObjectDetectionID(411):
              _get_platform_grasp_pixel,
              # Ball-specific grasp selection.
              LanguageObjectDetectionID(ball_prompt):
              _get_ball_grasp_pixel
          }
