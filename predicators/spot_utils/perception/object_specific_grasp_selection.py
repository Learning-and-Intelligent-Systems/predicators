"""Object-specific grasp selectors."""

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from bosdyn.client import math_helpers
from numpy.typing import NDArray
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
brush_prompt = "/".join(
    ["scrubbing brush", "hammer", "mop", "giant white toothbrush"])
brush_obj = LanguageObjectDetectionID(brush_prompt)
bucket_prompt = "/".join([
    "white plastic container with black handles",
    "white plastic tray with black handles",
    "white plastic bowl",
    "white storage bin with black handles",
])
bucket_obj = LanguageObjectDetectionID(bucket_prompt)
football_prompt = "/".join(["small orange basketball", "small orange"])
football_obj = LanguageObjectDetectionID(football_prompt)
train_toy_prompt = "/".join([
    "small white ambulance toy",
    "car_(automobile) toy",
    "egg",
])
train_toy_obj = LanguageObjectDetectionID(train_toy_prompt)
chair_prompt = "chair"
chair_obj = LanguageObjectDetectionID(chair_prompt)


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
    roll = math_helpers.Quat.from_roll(np.pi / 2)
    pitch = math_helpers.Quat.from_pitch(np.pi / 2)
    return pixel, pitch * roll  # NOTE: order is super important here!


def _get_chair_grasp_pixel(
    rgbds: Dict[str, RGBDImageWithContext], artifacts: Dict[str, Any],
    camera_name: str, rng: np.random.Generator
) -> Tuple[Tuple[int, int], Optional[math_helpers.Quat]]:
    del rng
    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[chair_obj][camera_name]
    except KeyError:
        raise ValueError(f"{chair_obj} not detected in {camera_name}")
    mask = seg_bb.mask
    rgbd = rgbds[camera_name]

    # Look for blue pixels in the isolated rgb.
    # Start by denoising the mask, "filling in" small gaps in it.
    convolved_mask = convolve(mask.astype(np.uint8),
                              np.ones((3, 3)),
                              mode="constant")
    smoothed_mask = (convolved_mask > 0)
    # Get copy of image with just the mask pixels in it.
    isolated_rgb = rgbd.rgb.copy()
    isolated_rgb[~smoothed_mask] = 0
    lo, hi = ((0, 0, 130), (130, 255, 255))
    centroid = find_color_based_centroid(isolated_rgb,
                                         lo,
                                         hi,
                                         min_component_size=10)
    if centroid is None:
        # Pick the topmost middle pixel, which should correspond to the top
        # of the chair.
        mask_args = np.argwhere(mask)
        mask_min_c = min(mask_args[:, 1])
        mask_max_c = max(mask_args[:, 1])
        c_len = mask_max_c - mask_min_c
        middle_c = mask_min_c + c_len // 2
        min_r = min(r for r, c in mask_args if c == middle_c)
        pixel = (middle_c, min_r)
    else:
        pixel = (centroid[0], centroid[1])

    # Uncomment for debugging.
    # rgbd = rgbds[camera_name]
    # bgr = cv2.cvtColor(rgbd.rgb, cv2.COLOR_RGB2BGR)
    # cv2.circle(bgr, pixel, 5, (0, 255, 0), -1)
    # cv2.circle(bgr, pixel, 5, (255, 0, 0), -1)
    # cv2.imshow("Selected grasp", bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Force a top-down grasp.
    pitch = math_helpers.Quat.from_pitch(np.pi / 2)
    return pixel, pitch


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
    roll = math_helpers.Quat.from_roll(angle)
    pitch = math_helpers.Quat.from_pitch(np.pi / 2)
    rot_quat = pitch * roll  # NOTE: order is super important here!

    del rgbds  # not used, except for debugging
    # Uncomment for debugging. Make sure also to not del rgbds (above).
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

    return pixel, rot_quat


def _get_brush_grasp_pixel(
    rgbds: Dict[str, RGBDImageWithContext], artifacts: Dict[str, Any],
    camera_name: str, rng: np.random.Generator
) -> Tuple[Tuple[int, int], Optional[math_helpers.Quat]]:
    """Grasp at the blue tape, i.e., blue pixels in the mask of the brush.

    Also, use the head of the brush to determine the grasp orientation.
    Grasp at a "9 o-clock" angle, if grasping toward the brush head is
    "12 o-clock", so that when the robot picks up the brush, the head is
    on the right.
    """
    del rng  # not used

    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[brush_obj][camera_name]
    except KeyError:
        raise ValueError(f"{brush_obj} not detected in {camera_name}")
    mask = seg_bb.mask
    rgb = rgbds[camera_name].rgb
    # Start by denoising the mask, "filling in" small gaps in it.
    convolved_mask = convolve(mask.astype(np.uint8),
                              np.ones((3, 3)),
                              mode="constant")
    mask = (convolved_mask > 0)
    # Get copy of image with just the mask pixels in it.
    isolated_rgb = rgb.copy()
    isolated_rgb[~mask] = 0
    # Look for blue pixels in the isolated rgb.
    lo, hi = ((0, 130, 180), (130, 255, 255))
    centroid = find_color_based_centroid(isolated_rgb,
                                         lo,
                                         hi,
                                         min_component_size=10)
    if centroid is None:
        raise RuntimeError("Could not find grasp for brush from image.")
    selected_pixel = (centroid[0], centroid[1])

    # Determine the rotation by considering a discrete number of possible
    # rolls and selecting the one that maximizes the number of mask pixels to
    # the right-hand-side of the grasp.

    # This part was extremely annoying to implement. If issues come up
    # again, it's helpful to dump these things and analyze separately.
    # import dill as pkl
    # with open("debug-brush-grasp.pkl", "wb") as f:
    #     pkl.dump(
    #         {
    #             "rgb": rgbds[camera_name].rgb,
    #             "mask": mask,
    #             "selected_pixel": selected_pixel,
    #         }, f)

    # Crop using the original mask, but then recompute the mask using color
    # because sometimes the top head of the brush gets cut off.
    crop_min_r, crop_min_c = np.min(np.argwhere(mask), axis=0)
    crop_max_r, crop_max_c = np.max(np.argwhere(mask), axis=0)
    # Widen the view to include the head.
    crop_min_r = max(0, crop_min_r - crop_min_r // 4)
    crop_min_c = max(0, crop_min_c - crop_min_c // 4)
    crop_max_r = min(mask.shape[0] - 1, crop_max_r + crop_max_r // 4)
    crop_max_c = min(mask.shape[1] - 1, crop_max_c + crop_max_c // 4)
    cropped_rgb = rgb[crop_min_r:crop_max_r + 1, crop_min_c:crop_max_c + 1]
    # Look for white because the brush is white.
    lower = np.array((220, 220, 220))
    upper = np.array((255, 255, 255))
    cropped_mask_uint8 = cv2.inRange(cropped_rgb, lower, upper)
    # Undo crop.
    cropped_mask = cropped_mask_uint8 > 0
    # Use the original mask as a starting point (NOTE: logical or).
    mask[crop_min_r:crop_max_r + 1, crop_min_c:crop_max_c + 1] |= cropped_mask

    # First find an angle that aligns with the handle of the brush.
    def _count_pixels_on_line(arr: NDArray, center: Tuple[int, int],
                              angle: float) -> float:
        y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
        mask = abs((y - center[1]) * np.cos(angle) -
                   (x - center[0]) * np.sin(angle)) < 10
        return np.sum(arr[mask])

    num_angle_candidates = 128
    candidates = [
        2 * np.pi * i / num_angle_candidates
        for i in range(num_angle_candidates)
    ]
    fn = lambda angle: _count_pixels_on_line(mask, selected_pixel, angle)
    aligned_angle = max(candidates, key=fn)

    # Now select among the two options based on which side has more pixels,
    # which is assumed to the side with the head of the brush.
    def _count_pixels_on_right(arr: NDArray, center: Tuple[int, int],
                               angle: float) -> float:
        y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
        mask = (y - center[1]) * np.cos(angle) - (
            x - center[0]) * np.sin(angle) > 0
        return np.sum(arr[mask])

    candidates = [aligned_angle + np.pi / 2, aligned_angle - np.pi / 2]
    fn = lambda angle: _count_pixels_on_right(mask, selected_pixel, angle)
    best_angle = max(candidates, key=fn)

    dy = int(50 * np.sin(best_angle))
    dx = int(50 * np.cos(best_angle))
    final_angle = np.arctan2(dx, -dy)

    # Uncomment for debugging.
    # bgr = cv2.cvtColor(rgbds[camera_name].rgb, cv2.COLOR_RGB2BGR)
    # cv2.circle(bgr, selected_pixel, 5, (0, 255, 0), -1)
    # cv2.arrowedLine(bgr, (selected_pixel[0], selected_pixel[1]),
    #                 (selected_pixel[0] + dx, selected_pixel[1] + dy),
    #                 (255, 0, 0), 5)
    # cv2.imshow("Selected grasp", bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    roll = math_helpers.Quat.from_roll(final_angle)
    pitch = math_helpers.Quat.from_pitch(np.pi / 2)
    rot_quat = pitch * roll  # NOTE: order is super important here!
    return selected_pixel, rot_quat


def _get_bucket_grasp_pixel(
    rgbds: Dict[str, RGBDImageWithContext], artifacts: Dict[str, Any],
    camera_name: str, rng: np.random.Generator
) -> Tuple[Tuple[int, int], Optional[math_helpers.Quat]]:
    """Select a blue pixel on the rim of the bucket to grasp."""
    del rng  # not used

    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[bucket_obj][camera_name]
    except KeyError:
        raise ValueError(f"{bucket_obj} not detected in {camera_name}")

    mask = seg_bb.mask
    rgbd = rgbds[camera_name]

    # Helpful to dump these things and analyze separately.
    # import dill as pkl
    # with open("debug.pkl", "wb") as f:
    #     pkl.dump(
    #         {
    #             "rgbd": rgbd,
    #             "mask": mask,
    #         }, f)

    # Look for blue pixels in the isolated rgb.
    # Start by denoising the mask, "filling in" small gaps in it.
    convolved_mask = convolve(mask.astype(np.uint8),
                              np.ones((3, 3)),
                              mode="constant")
    smoothed_mask = (convolved_mask > 0)
    # Get copy of image with just the mask pixels in it.
    isolated_rgb = rgbd.rgb.copy()
    isolated_rgb[~smoothed_mask] = 0
    lo, hi = ((0, 0, 130), (130, 255, 255))
    centroid = find_color_based_centroid(isolated_rgb,
                                         lo,
                                         hi,
                                         min_component_size=10)
    # This can happen sometimes if the rim of the bucket is separated from the
    # body of the bucket. If that happens, just pick the center bottom pixel in
    # the mask, which should be the rim.
    if centroid is None:
        mask_args = np.argwhere(mask)
        mask_min_c = min(mask_args[:, 1])
        mask_max_c = max(mask_args[:, 1])
        c_len = mask_max_c - mask_min_c
        middle_c = mask_min_c + c_len // 2
        max_r = max(r for r, c in mask_args if c == middle_c)
        selected_pixel = (middle_c, max_r)
    else:
        # NOTE! Testing
        selected_pixel = (centroid[0], centroid[1])

    # Uncomment for debugging.
    # bgr = cv2.cvtColor(rgbds[camera_name].rgb, cv2.COLOR_RGB2BGR)
    # cv2.circle(bgr, selected_pixel, 5, (0, 255, 0), -1)
    # cv2.imshow("Selected grasp", bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Specify a top-down grasp constraint.
    pitch = math_helpers.Quat.from_pitch(np.pi / 2)

    return selected_pixel, pitch


def _get_mask_center_grasp_pixel(
    detect_id: LanguageObjectDetectionID, rgbds: Dict[str,
                                                      RGBDImageWithContext],
    artifacts: Dict[str, Any], camera_name: str, rng: np.random.Generator
) -> Tuple[Tuple[int, int], Optional[math_helpers.Quat]]:
    """Select a pixel that's as close to the center of the mask as possible,
    while still being in the mask."""
    del rng
    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[detect_id][camera_name]
    except KeyError:
        raise ValueError(f"{detect_id} not detected in {camera_name}")
    mask = seg_bb.mask
    pixels_in_mask = np.where(mask)
    candidates = list(zip(*pixels_in_mask))
    center = np.mean(candidates, axis=0)
    dist_to_center = np.sum((candidates - center)**2, axis=1)
    idxs = list(range(len(candidates)))
    best_idx = min(idxs, key=lambda i: dist_to_center[i])
    best_r, best_c = candidates[best_idx]
    selected_pixel = (best_c, best_r)

    del rgbds
    # Uncomment for debugging.
    # bgr = cv2.cvtColor(rgbds[camera_name].rgb, cv2.COLOR_RGB2BGR)
    # cv2.circle(bgr, selected_pixel, 5, (0, 255, 0), -1)
    # cv2.imshow("Selected grasp", bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return selected_pixel, None


# Maps an object ID to a function from rgbds, artifacts and camera to pixel.
OBJECT_SPECIFIC_GRASP_SELECTORS: Dict[ObjectDetectionID, Callable[[
    Dict[str, RGBDImageWithContext], Dict[str, Any], str, np.random.Generator
], Tuple[Tuple[int, int], Optional[math_helpers.Quat]]]] = {
    # Platform-specific grasp selection.
    AprilTagObjectDetectionID(411): _get_platform_grasp_pixel,
    # Ball-specific grasp selection.
    ball_obj: _get_ball_grasp_pixel,
    # Cup-specific grasp selection.
    cup_obj: _get_cup_grasp_pixel,
    # Brush-specific grasp selection.
    brush_obj: _get_brush_grasp_pixel,
    # Bucket-specific grasp selection.
    bucket_obj: _get_bucket_grasp_pixel,
    # Chips-specific grasp selection.
    football_obj: partial(_get_mask_center_grasp_pixel, football_obj),
    # train_toy-specific grasp selection.
    train_toy_obj: partial(_get_mask_center_grasp_pixel, train_toy_obj),
    # Chair-specific grasp selection.
    chair_obj: _get_chair_grasp_pixel
}
