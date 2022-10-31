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
    # Find line that orients the table.
    table_line = parse_table_line_from_image(image, camera, debug=debug)


def parse_table_line_from_image(image: Image, camera: str, debug: bool = False) -> LineSegment:
    
    line = CAMERA_TO_LINE[camera]

    if debug:
        start_point = (line.x1, line.y1)
        end_point = (line.x2, line.y2)
        color = (255, 255, 255)
        thickness = 5
        line_image = cv2.line(image, start_point, end_point, color, thickness)
        _show_image(line_image, f"Line for {camera}")


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
