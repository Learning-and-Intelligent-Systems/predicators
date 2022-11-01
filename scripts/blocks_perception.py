"""An extremely brittle and blocks/LIS/Panda-specific perception pipeline."""

import cv2
import numpy as np
from typing import Sequence
import imageio.v3 as iio
from matplotlib import pyplot as plt
from pathlib import Path
from predicators.utils import LineSegment
from predicators.structs import Image, State


# TODO: maybe optimize over these hyperparameters for a few annotated images

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
    # Find line that intersects the bottom row of blocks.
    bottom_row_line = parse_bottom_row_line_from_image(image, camera, debug=debug)
    # Crop the image around that line.
    cropped_image = crop_from_bottom_row_line(image, bottom_row_line, debug=debug)
    # Segment by color.
    segmented_images = segment_image(cropped_image, debug=debug)


def parse_bottom_row_line_from_image(image: Image, camera: str, debug: bool = False) -> LineSegment:
    line = CAMERA_TO_LINE[camera]

    if debug:
        start_point = (line.x1, line.y1)
        end_point = (line.x2, line.y2)
        color = (255, 255, 255)
        thickness = 5
        line_image = image.copy()
        cv2.line(line_image, start_point, end_point, color, thickness)
        _show_image(line_image, f"Line for {camera}")

    return line


def crop_from_bottom_row_line(image: Image, bottom_row_line: LineSegment, debug: bool = False) -> Image:
    x1, y1 = bottom_row_line.x1, bottom_row_line.y1
    x2, y2 = bottom_row_line.x2, bottom_row_line.y2
    crop_height = 60
    bottom_left_corner = (x1, y1 - crop_height / 2)
    top_left_corner = (x1, y1 + crop_height / 2)
    bottom_right_corner = (x2, y2 - crop_height / 2)
    top_right_corner = (x2, y2 + crop_height / 2)
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([[
        bottom_left_corner,
        bottom_right_corner,
        top_right_corner,
        top_left_corner,
    ]], dtype=np.int32)
    channel_count = image.shape[2]
    ignore_mask_color = (255, ) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    cropped_image = cv2.bitwise_and(image, mask)
    if debug:
        _show_image(cropped_image, f"Cropped image")
    return cropped_image


def segment_image(image: Image, debug: bool = False) -> Image:
    img_blur = cv2.GaussianBlur(image, (3, 3), 0)

    for name, (r, g, b) in BLOCK_COLORS.items():
        # Repeat until number of components goes up.
        thresh = INIT_COLOR_THRESH
        components = find_ccs_by_color(img_blur, r, g, b, thresh)
        init_num_components = len(np.unique(components)) - 1
        while True:
            thresh += COLOR_THRESH_INC
            new_components = find_ccs_by_color(img_blur, r, g, b, thresh)
            new_num_components = len(np.unique(new_components)) - 1
            if new_num_components != init_num_components:
                if debug:
                    print(f"Converged at thresh={thresh}")
                break
            components = new_components
        # Isolate each component
        for component_id in np.unique(components):
            if component_id == 0:  # background
                continue
            component_mask = (255 * (components == component_id)).astype(np.uint8)
            if debug:
                _show_image(component_mask, f"Components for {name}")
        

def find_ccs_by_color(img: Image, r: float, g: float, b: float, color_thresh: float) -> Image:
    mean_color = np.array([b, g, r], dtype=np.uint8)
    lower_color = mean_color - color_thresh
    upper_color = mean_color + color_thresh
    detected_mask = cv2.inRange(img, lower_color, upper_color)
    detections = cv2.bitwise_and(img, img, mask=detected_mask)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(detected_mask, cv2.MORPH_OPEN, kernel, iterations = 2)
    # Marker labelling
    ret, markers = cv2.connectedComponents(opening)
    # Find the markers above a threshold size.
    for marker in np.unique(markers):
        if marker == 0:  # background
            continue
        if np.sum(markers == marker) < MARKER_THRESH:
            markers[markers == marker] = 0
    return markers


if __name__ == "__main__":
    color_imgs_path = Path("~/Desktop/blocks-images/color/")
    right_camera_id = 231122071284
    left_camera_id = 231122071283
    img_id = 0
    right_color_img_path = color_imgs_path / f"color-{img_id}-{right_camera_id}.png"
    left_color_img_path = color_imgs_path / f"color-{img_id}-{left_camera_id}.png"
    right_color_img = iio.imread(right_color_img_path)
    left_color_img = iio.imread(left_color_img_path)
    state = parse_state_from_images(right_color_img, left_color_img, debug=True)
