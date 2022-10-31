"""An extremely brittle and blocks/LIS/Panda-specific perception pipeline."""

import cv2
import numpy as np
from typing import Sequence
import imageio.v3 as iio
from matplotlib import pyplot as plt
from pathlib import Path
from predicators.utils import LineSegment
from predicators.structs import Image, State


BLOCK_COLORS = {
    # RGB
    "red": (120, 50, 50),
    "purple": (60, 60, 100),
    "orange": (160, 90, 60),
    "yellow": (160, 120, 60),
    "blue": (60, 90, 100),
}
COLOR_THRESH = 25


def _show_image(img: Image, title: str) -> None:
    # plt.figure()
    # plt.title(title)
    # plt.imshow(img)
    # plt.tight_layout()
    # plt.show()
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_state_from_images(images: Sequence[Image], debug: bool = False) -> State:
    # Convert to RBG for cv2. 
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

    # Display raw images.
    if debug:
        for i, img in enumerate(images):
            _show_image(img, f"Original Image {i}")
    
    states = [parse_state_from_image(img, debug=debug) for img in images]
    # TODO: Average states together.
    return states[0]


def parse_state_from_image(image: Image, debug: bool = False) -> State:
    # Find line that orients the table.
    table_line = parse_table_line_from_image(image, debug=debug)


def parse_table_line_from_image(image: Image, debug: bool = False) -> LineSegment:
    img_blur = cv2.GaussianBlur(image, (5, 5), 0)
    
    color_to_mask = {}
    for name, (r, g, b) in BLOCK_COLORS.items():
        mean_color = np.array([b, g, r], dtype=np.uint8)
        lower_color = mean_color - COLOR_THRESH
        upper_color = mean_color + COLOR_THRESH
        mask = cv2.inRange(img_blur, lower_color, upper_color)
        color_to_mask[name] = mask

    for name, mask in color_to_mask.items():
        full_mask = mask
        for other_name in color_to_mask:
            if name == other_name:
                continue
            other_mask = color_to_mask[other_name]
            full_mask = cv2.bitwise_and(full_mask, cv2.bitwise_not(other_mask))
        detections = cv2.bitwise_and(img_blur, img_blur, mask=full_mask)
        if debug:
            _show_image(detections, f"Detections for {name}")


if __name__ == "__main__":
    color_imgs_path = Path("~/Desktop/blocks-images/color/")
    right_camera_id = 231122071283
    left_camera_id = 231122071284
    img_id = 0
    right_color_img_path = color_imgs_path / f"color-{img_id}-{left_camera_id}.png"
    left_color_img_path = color_imgs_path / f"color-{img_id}-{right_camera_id}.png"
    right_color_img = iio.imread(right_color_img_path)
    left_color_img = iio.imread(left_color_img_path)
    state = parse_state_from_images([right_color_img, left_color_img], debug=True)
