"""adapted from SoM."""
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import cv2
import matplotlib.figure as mplfigure
import numpy as np
import torch as th
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from torchvision.transforms import ToPILImage  # type: ignore

from predicators import utils
from predicators.structs import Mask, Object

if TYPE_CHECKING:
    from predicators.utils import BoundingBox, PyBulletState


class VisImage:
    """A class to visualize an image using matplotlib."""

    def __init__(self, img: np.ndarray, scale: float = 1.0) -> None:
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in [0, 255].
            scale (float): scale the input image.
        """
        self.img: np.ndarray = img
        self.scale: float = scale
        self.height: int = img.shape[0]
        self.width: int = img.shape[1]
        self.dpi: float = 0.0
        self.fig: mplfigure.Figure
        self.ax: mplfigure.Axes
        self.canvas: FigureCanvasAgg
        self._setup_figure(img)

    def _setup_figure(self, img: np.ndarray) -> None:
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the
                image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the
                coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        fig.set_size_inches(self.width / self.dpi, self.height / self.dpi)

        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.ax.imshow(img, interpolation="nearest")

    def save(self, filepath: str) -> None:
        """
        Args:
            filepath (str): absolute path (including file name) to save the
            image.
        """
        self.fig.savefig(filepath)

    def get_image(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: (H, W, 3) uint8 image in RGB.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")
        img_rgba = buffer.reshape(height, width, 4)
        split_result = np.split(img_rgba, [3], axis=2)
        if len(split_result) == 2:
            rgb, _ = split_result  # pylint: disable=unbalanced-tuple-unpacking
        else:
            raise ValueError(f"Expected 2 elements from np.split, got "
                             f"{len(split_result)}.")
        return rgb.astype("uint8")


class ImagePatch:
    """A class to represent an image patch."""

    def __init__(
        self,
        state: 'PyBulletState',
        left: Optional[int] = None,
        lower: Optional[int] = None,
        right: Optional[int] = None,
        upper: Optional[int] = None,
        parent_left: int = 0,
        parent_lower: int = 0,
        parent_img_patch: Optional["ImagePatch"] = None,
        attn_objects: Optional[List[Object]] = None,
    ) -> None:
        """
        Args:
            img: An image, which can be a PIL Image, a NumPy array, or a torch
            Tensor.
            left, lower, right, upper: The bounding box coordinates for
            cropping.
            parent_left, parent_lower: Offsets if this patch is part of a larger
            parent image.
            parent_img_patch: If this patch has a parent patch.
            attn_objects: Optional list of `Object`s relevant to this patch.
        """
        self.attn_objects: Optional[List[Object]] = attn_objects
        self.state: 'PyBulletState' = state

        if state.labeled_image is None:
            img = state.state_image
        else:
            img = state.labeled_image

        image_tensor: th.Tensor

        # if isinstance(img, Image.Image):
        #     image_tensor = transforms.ToTensor()(img)
        if isinstance(img, np.ndarray):
            # If img is shape (H, W, C) or (C, H, W), adjust as needed
            if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                # (H, W, C)
                # Convert to shape (C, H, W)
                img = np.transpose(img, (2, 0, 1))
            image_tensor = th.tensor(
                img,
                dtype=th.float32  # pylint: disable=no-member
            ) / 255.0
        # elif isinstance(img, th.Tensor):
        #     # If dtype == uint8, convert to float
        #     if img.dtype == th.uint8:
        #         image_tensor = img.float() / 255.0
        #     else:
        #         image_tensor = img
        else:
            raise TypeError("Unsupported image type.")

        # For clarity in indexing, rename
        _, h, w = image_tensor.shape

        # If no bounding box is provided, take the full image
        if left is None and right is None and upper is None and lower is None:
            self.cropped_image: th.Tensor = image_tensor
            self.left: int = 0
            self.lower: int = 0
            self.right: int = w
            self.upper: int = h
        else:
            # Ensure these are not None via default 0
            _left = left if left is not None else 0
            _lower = lower if lower is not None else 0
            _right = right if right is not None else w
            _upper = upper if upper is not None else h

            # Crop indexing: image_tensor[:, row_slice, col_slice]
            # Remember that Torch uses [C, H, W] => indices: [C, y, x]
            # "upper" is the top, so the slice for the row dimension is
            # h - upper : h - lower
            cropped = image_tensor[:, (h - _upper):(h - _lower), _left:_right]
            if cropped.shape[1] == 0 or cropped.shape[2] == 0:
                raise ValueError("ImagePatch has zero area after cropping.")

            self.cropped_image = cropped
            self.left = _left + parent_left
            self.lower = _lower + parent_lower
            self.right = _right + parent_left
            self.upper = _upper + parent_lower

        self.height: int = self.cropped_image.shape[1]
        self.width: int = self.cropped_image.shape[2]

        self.parent_img_patch: Optional["ImagePatch"] = parent_img_patch
        self.horizontal_center: float = (self.left + self.right) / 2
        self.vertical_center: float = (self.lower + self.upper) / 2

    @property
    def cropped_image_in_PIL(self) -> Image.Image:
        """Return the cropped image as a PIL Image."""
        return ToPILImage()(self.cropped_image)

    def save(self, path: str) -> None:
        """Save the cropped image to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.cropped_image_in_PIL.save(path)

    def label_all_objects(
        self,
        obj_mask_dict: Dict[Object, Mask],
        #   alpha: float = 0.1,
        #   anno_mode: Optional[List[str]] = None
    ) -> None:
        """Label objects on the image patch."""
        # Make sure it's 3D: [C, H, W]
        if len(self.cropped_image.shape) != 3:
            raise ValueError("cropped_image must be 3-dimensional.")

        # Confirm all masks match the shape H x W
        _, h, w = self.cropped_image.shape
        for mask in obj_mask_dict.values():
            if mask.shape != (h, w):
                raise ValueError("Mask shape does not match patch dimensions.")

        img_np = self.cropped_image.permute(1, 2, 0).numpy()  # HWC
        vis_image = VisImage((img_np * 255).astype(np.uint8))

        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        for obj, mask in obj_mask_dict.items():
            if mask.sum() == 0:
                continue
            # distanceTransform requires uint8 or int
            mask_uint8 = mask.astype(np.uint8)
            mask_padded = np.pad(mask_uint8, ((1, 1), (1, 1)), 'constant')
            mask_dt = cv2.distanceTransform(mask_padded, cv2.DIST_L2, 0)  # pylint: disable=no-member
            mask_dt = mask_dt[1:-1, 1:-1]
            max_dist = np.max(mask_dt)
            coords_y, coords_x = np.where(mask_dt == max_dist)
            dy = 0
            if obj.type.name == 'jug':
                dy += 40
            elif obj.type.name == "cup":
                dy += 10
            elif obj.type.name == "target":
                dy += 15
            elif obj.type.name == "plate":
                dy += 30

            # Make sure draw_text gets string
            vis_image.ax.text(
                coords_x[len(coords_x) // 2],
                coords_y[len(coords_y) // 2] - 8 + dy,
                str(obj.id),  # convert to str
                size=14 * vis_image.scale,
                family="sans-serif",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.8,
                    "pad": 0.7,
                    "edgecolor": "none"
                },
                verticalalignment="top",
                horizontalalignment="center",
                color=tuple(color),  # sequence of floats
                zorder=10,
                rotation=0,
            )

        self.cropped_image = th.tensor(  # pylint: disable=no-member
            vis_image.get_image()).permute(2, 0, 1) / 255.0

    def crop_to_objects(self,
                        objects: Sequence[Object],
                        left_margin: int = 5,
                        lower_margin: int = 10,
                        right_margin: int = 10,
                        top_margin: int = 5) -> 'ImagePatch':
        """Crop the image patch to the smallest bounding box that contains all
        the masks of all the objects. The BBox origin (0, 0) is at the bottom-
        left corner.

        Parameters:
        -----------
        objects : List[Object]
            The objects whose bounding box is to be used for cropping.
        """
        masks = [self.state.get_obj_mask(obj) for obj in objects]
        # Assert the masks are not None
        bboxes = [utils.mask_to_bbox(mask) for mask in masks]

        bbox = utils.smallest_bbox_from_bboxes(bboxes)
        # Crop the image
        ip = self.crop(bbox.left - left_margin, bbox.lower - lower_margin,
                       bbox.right + right_margin, bbox.upper + top_margin)
        return ip

    def crop_to_bboxes(self, bboxes: Sequence['BoundingBox']) -> 'ImagePatch':
        """Crop the image patch to the smallest bounding box that contains
        all."""
        bbox = utils.smallest_bbox_from_bboxes(bboxes)
        return self.crop(bbox.left, bbox.lower, bbox.right, bbox.upper)

    def crop(self, left: int, lower: int, right: int,
             upper: int) -> 'ImagePatch':
        """Returns a new ImagePatch containing a crop of the original image at
        the given coordinates.
        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the
            given coordinates
        """
        left = max(0, left)
        lower = max(0, lower)
        right = min(self.width, right)
        upper = min(self.height, upper)

        return ImagePatch(self.state,
                          left,
                          lower,
                          right,
                          upper,
                          self.left,
                          self.lower,
                          parent_img_patch=self,
                          attn_objects=self.attn_objects)

    # def draw_text(
    #     self,
    #     fig: VisImage,
    #     text: str,
    #     position: Sequence[int],
    #     color: Sequence[float],
    #     font_size: int = 14,
    #     horizontal_alignment: str = "center",
    #     rotation: float = 0,
    # ) -> np.ndarray:
    #     """
    #     Draw text on a VisImage, then return the updated image as np.ndarray.
    #     """
    #     # A typed helper function for color contrast
    #     def contrasting_color(rgb: Tuple[float, float, float]) -> str:
    #         R, G, B = rgb
    #         # For typical 0-255 range, multiply by 255 if needed:
    #         # but here color is already float in [0..1], so scale to 255 for Y
    #         R_255, G_255, B_255 = R * 255, G * 255, B * 255
    #         Y = 0.299 * R_255 + 0.587 * G_255 + 0.114 * B_255
    #         return "black" if Y > 128 else "white"

    #     # Convert color to tuple if needed
    #     color_tuple = tuple(color)
    #     bbox_bg_color = contrasting_color(color_tuple)

    #     x, y = position
    #     fig.ax.text(
    #         x,
    #         y,
    #         text,
    #         size=font_size * fig.scale,
    #         family="sans-serif",
    #         bbox={
    #             "facecolor": bbox_bg_color,
    #             "alpha": 0.8,
    #             "pad": 0.7,
    #             "edgecolor": "none"
    #         },
    #         verticalalignment="top",
    #         horizontalalignment=horizontal_alignment,
    #         color=color_tuple,
    #         zorder=10,
    #         rotation=rotation,
    #     )

    #     return fig.get_image()
