"""CV2-based utilities."""
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


def find_color_based_centroid(
        img: NDArray[np.uint8],
        rgb_lower_bound: Tuple[int, int, int],
        rgb_upper_bound: Tuple[int, int, int],
        min_component_size: int = 1000) -> Optional[Tuple[int, int]]:
    """Find the centroid pixel of the largest connected component of a color,
    or return None if no large-enough component exists."""
    # Copy to make sure we don't modify the image.
    img = img.copy()
    # Mask color.
    lower = np.array(rgb_lower_bound)
    upper = np.array(rgb_upper_bound)
    mask = cv2.inRange(img, lower, upper)
    # Apply blur.
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # Connected components with stats.
    nb_components, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=4)
    # Fail if nothing found.
    if nb_components <= 1:
        return None
    # Find the largest non background component.
    # NOTE: range() starts from 1 since 0 is the background label.
    max_label, _ = max(
        ((i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)),
        key=lambda x: x[1])

    # Fail component too small.
    if stats[max_label][4] < min_component_size:
        return None

    x, y = map(int, centroids[max_label])

    return x, y
