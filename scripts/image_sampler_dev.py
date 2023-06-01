"""Hacky code for developing very object-specific image-based samplers."""

from pathlib import Path
import cv2
import numpy as np

from typing import Optional, Tuple


OBJECT_CROPS = {
    # min_x, max_x, min_y, max_y
    "hammer": (160, 350, 160, 350),
}

OBJECT_COLOR_BOUNDS = {
    # (min B, min G, min R), (max B, max G, max R)
    "hammer": ((0, 0, 50), (40, 40, 200)),
}


def _find_center(img_file: Path, obj_name: str, outfile: Optional[Path] = None) -> Tuple[int, int]:
    img = cv2.imread(str(img_file))
    
    # Crop
    crop_min_x, crop_max_x, crop_min_y, crop_max_y = OBJECT_CROPS[obj_name]
    cropped_img = img[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

    # # Uncomment for debugging
    # cv2.imshow("Cropped image", cropped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Mask color.
    lo, hi = OBJECT_COLOR_BOUNDS[obj_name]
    lower = np.array(lo)
    upper = np.array(hi)
    mask = cv2.inRange(cropped_img, lower, upper)

    # Apply blur.
    mask = cv2.GaussianBlur(mask,(5,5),0)

    # # Uncomment for debugging
    # cv2.imshow("Masked image", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find center.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # # Uncomment for debugging
    # cv2.drawContours(cropped_img, contours, 0, (0, 255, 0), 2)
    # cv2.imshow("Contour image", cropped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    M = cv2.moments(contours[0])
    cropped_x = round(M['m10'] / M['m00'])
    cropped_y = round(M['m01'] / M['m00'])

    x = cropped_x + crop_min_x
    y = cropped_y + crop_min_y

    if outfile is not None:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imwrite(str(outfile), img)

    return (x, y)


def _main() -> None:
    # Hammer
    obj_name = "hammer"
    img_nums = [2, 6, 7, 8, 9, 10]
    for n in img_nums:
        img_file = Path(f"sampler_images/wall/img{n}.png")
        outfile = Path(f"sampler_images/wall/labelled_img{n}.png")
        _find_center(img_file, obj_name, outfile)
        


if __name__ == "__main__":
    _main()
