import logging
import os
import random
from typing import List, Sequence, Tuple, Dict, Optional

import cv2
import matplotlib.colors as mcolors
import matplotlib.figure as mplfigure
import numpy as np
import torch as th
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.typing import NDArray
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

from predicators import utils
# from predicators.utils import BoundingBox
from predicators.settings import CFG
# from viper.image_patch import ImagePatch as ViperImagePatch
from predicators.structs import Mask, Object, State


class VisImage:

    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
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
        # add a small 1e-2 to avoid precision lost due to matplotlib's
        # truncation (https://github.com/matplotlib/matplotlib/issues/15363)
        # fig.set_size_inches(
        #     (self.width * self.scale + 1e-2) / self.dpi,
        #     (self.height * self.scale + 1e-2) / self.dpi,
        # )
        fig.set_size_inches(self.width / self.dpi, self.height / self.dpi)

        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        # self.reset_image(img)
        self.ax.imshow(
            img,  #extent=(0, self.width-1, self.height-1, 0), 
            interpolation="nearest")

    # def reset_image(self, img):
    #     """
    #     Args:
    #         img: same as in __init__
    #     """
    #     # img = img.astype("uint8")

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including 
                the file name, where the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given 
                `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        # return img_rgba
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class ImagePatch:
    # class ImagePatch:
    # def __init__(self, image: np.ndarray, *args, **kwargs):
    def __init__(self,
                 state: State,
                 left: int = None,
                 lower: int = None,
                 right: int = None,
                 upper: int = None,
                 parent_left: int = 0,
                 parent_lower: int = 0,
                 queues: Tuple = None,
                 parent_img_patch: 'ImagePatch' = None,
                 attn_objects: Optional[List[Object]] = None,
                 ) -> None:

        self.attn_objects = attn_objects
        if state.labeled_image is None:
            image = state.state_image
        else:
            image = state.labeled_image
        self.state = state
        # self.vlm = utils.create_vlm_by_name(CFG.vlm_model_name)
        # css4_colors = mcolors.CSS4_COLORS
        # self.color_proposals = [list(mcolors.hex2color(color)) for color in
        #                         css4_colors.values()]
        # ViperGPT init stuff
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = th.tensor(image).permute(1, 2, 0)
        elif isinstance(image, th.Tensor) and image.dtype == th.uint8:
            image = image / 255

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, image.shape[1] -
                                       upper:image.shape[1] - lower,
                                       left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

    @property
    def cropped_image_in_PIL(self):
        return ToPILImage()(self.cropped_image)

    def save(self, path: str):
        # save the cropped_image, assuming it's of type PIL.Image.Image
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pil_image = self.cropped_image_in_PIL
        pil_image.save(path)

    def label_all_objects(self,
                          obj_mask_dict: Dict[Object, Mask],
                          alpha: float = 0.1,
                          anno_mode: List[str] = ['Mark']):
        """Label an object, as indicated by mask, on the image.

        Parameters:
        -----------
        mask : Mask
            Mask of shape (H, W), where H is the image height and W is the image
            width. Each value in the array is either a True or False.
        label : str
            The label to be assigned to the object.
        alpha : float
            The transparency of the mask overlay.
        anno_mode : List[str]
            Weather to use text marker or mask overlay, or both.
        """
        # cropped_image: [4, 900, 900]
        assert len(self.cropped_image.shape) == 3
        masks = obj_mask_dict.values()
        for mask in masks:
            assert self.cropped_image.shape[1:] == mask.shape
        # Draw a color
        # randint = random.randint(0, len(self.color_proposals)-1)
        # color = self.color_proposals[randint]
        # color = mcolors.to_rgb(color)
        color = np.array([1, 1, 1])

        img_np = self.cropped_image.permute(1, 2, 0).numpy()
        vis_image = VisImage(img_np)
        # vis_image.save(f"images/vis_image_{label}.png")

        for obj, mask in obj_mask_dict.items():
            # Skip if the mask is empty
            if mask.sum() == 0:
                continue
            mask = mask.astype(np.uint8)
            mask = np.pad(mask, ((1, 1), (1, 1)), 'constant')
            # The distance from each pixel to the nearest 0 pixel.
            mask_dt = cv2.distanceTransform(mask, cv2.DIST_L2, 0)
            mask_dt = mask_dt[1:-1, 1:-1]
            max_dist = np.max(mask_dt)
            coords_y, coords_x = np.where(mask_dt == max_dist)
            dy = 0
            if obj.type.name == 'jug':
                dy += 30
            elif obj.type.name == "cup":
                dy += 10
            elif obj.type.name == "target":
                dy += 15

            img_np = self.draw_text(vis_image,
                                    obj.id, 
                                    (coords_x[len(coords_x) // 2],
                                    coords_y[len(coords_y) // 2] - 8 + dy),
                                    color=color)

        img_np = vis_image.get_image()
        self.cropped_image = th.tensor(img_np).permute(2, 0, 1)

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
        # try:
        bboxes = [utils.mask_to_bbox(mask) for mask in masks]
        # except Exception as e:
        #     breakpoint()

            # left = min(left, x_indices.min() - left_margin)
            # lower = min(lower, self.height - y_indices.max() - lower_margin - 1)
            # right = max(right, x_indices.max() + right_margin)
            # upper = max(upper, self.height - y_indices.min() + top_margin - 1)
        bbox = utils.smallest_bbox_from_bboxes(bboxes)
        # Crop the image
        try:
            ip = self.crop(bbox.left - left_margin, bbox.lower - lower_margin,
                         bbox.right + right_margin, bbox.upper + top_margin)
        except:
            breakpoint()
        return ip

    def crop_to_bboxes(self, bboxes) -> 'ImagePatch':
        bbox = utils.smallest_bbox_from_bboxes(bboxes)
        return self.crop(bbox.left, bbox.lower, bbox.right, bbox.upper)

    def crop(self, left: int, lower: int, right: int,
             upper: int) -> 'ImagePatch':
        """Returns a new ImagePatch containing a crop of the original image at
        the given coordinates.

        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the
            original image
        lower : int
            the position of the bottom border of the crop's bounding box in the
            original image
        right : int
            the position of the right border of the crop's bounding box in the
            original image
        upper : int
            the position of the top border of the crop's bounding box in the
            original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the
            given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        # if config.crop_larger_margin:
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
                          queues=self.queues,
                          parent_img_patch=self,
                          attn_objects=self.attn_objects)


    def draw_text(self, fig: VisImage, text: str, position: Sequence[int],
        color: Sequence[float], font_size: int=14,
        horizontal_alignment: str="center", rotation: float=0) ->\
            NDArray[np.uint8]:
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text 
                on image.
            font_size (int, optional): font of the text. If not provided, a font 
                size proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """

        def contrasting_color(rgb):
            """Returns 'white' or 'black' depending on which color contrasts
            more with the given RGB value."""

            # Decompose the RGB tuple
            R, G, B = rgb

            # Calculate the Y value
            Y = 0.299 * R + 0.587 * G + 0.114 * B

            # If Y value is greater than 128, it's closer to white so return
            # black. Otherwise, return white.
            return 'black' if Y > 128 else 'white'

        bbox_background = contrasting_color(color * 255)

        x, y = position
        fig.ax.text(
            x,
            y,
            text,
            size=font_size * fig.scale,
            family="sans-serif",
            bbox={
                "facecolor": bbox_background,
                "alpha": 0.8,
                "pad": 0.7,
                "edgecolor": "none"
            },
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )