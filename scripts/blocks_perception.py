"""An extremely brittle and blocks/LIS/Panda-specific perception pipeline."""

import cv2
import numpy as np
from typing import Sequence
import imageio.v3 as iio
from matplotlib import pyplot as plt
from pathlib import Path
from predicators.utils import LineSegment
from predicators.structs import Image, State




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
    # https://learnopencv.com/edge-detection-using-opencv/
    # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    if debug:
        _show_image(edges, "Canny Edge Detection")

    # https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    # Get mean direction of lines
    all_lines = [l for line in lines for l in line]
    angles = []
    lengths = []
    for x1, y1, x2, y2 in all_lines:
        theta = np.arctan2(y2 - y1, x2 - x1)
        angles.append(theta)
        lengths.append(np.linalg.norm([y2 - y1, x2 -x1]))
    # Weighted mean
    total_length = np.sum(lengths)
    angle_fracs = [length / total_length for angle, length in zip(angles, lengths)]
    table_angle = np.dot(angle_fracs, angles)

    # Draw the angle on the image for debugging
    if debug:
        height, width, _ = image.shape
        scale = height / 10
        start_point = (height // 2, width // 2)
        end_x = start_point[0] + scale * np.cos(table_angle)
        end_y = start_point[1] + scale * np.sin(table_angle)
        end_point = (int(end_x), int(end_y))
        thickness = 9
        color = (255, 255, 255)
        arrow_image = cv2.arrowedLine(image, start_point, end_point,
                                      color, thickness)
        _show_image(arrow_image, "Detected Orientation of Table")


    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    # if debug:
    #     lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    #     _show_image(lines_edges, "Detected Lines")


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
