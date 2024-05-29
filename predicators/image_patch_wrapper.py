from typing import List, Sequence
import random

import cv2
import numpy as np
from numpy.typing import NDArray

from viper.image_patch import ImagePatch as ViperImagePatch
from predicators.structs import Object, Mask
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.colors as mcolors
import matplotlib.figure as mplfigure

class ImagePatch(ViperImagePatch):
    # def __init__(self, image: np.ndarray, *args, **kwargs):
    #     super().__init__(image, *args, **kwargs)
    #     css4_colors = mcolors.CSS4_COLORS
    #     self.color_proposals = [list(mcolors.hex2color(color)) for color in 
    #                             css4_colors.values()]

    def label_object(self, mask: Mask, label: str, alpha: float = 0.1,
            anno_mode: List[str]=['Mark']):
        """
        Label an object, as indicated by mask, on the image.

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
        assert self.cropped_image.shape == mask.shape
        # Draw a color
        # randint = random.randint(0, len(self.color_proposals)-1)
        # color = self.color_proposals[randint]
        # color = mcolors.to_rgb(color)
        color = [1,1,1]

        mask = mask.astype(np.uint8)
        mask = np.pad(mask, ((1, 1), (1, 1)), 'constant')
        mask_dt = cv2.distanceTransform(mask, cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        coords_y, coords_x = np.where(mask_dt == max_dist)  

        self.cropped_image = self.draw_text(VisImage(self.cropped_image), label, 
            (coords_x[len(coords_x)//2] + 2, coords_y[len(coords_y)//2] - 6), 
            color=color)

    def crop_to_objects(self, masks: Sequence[Mask], left_margin: int = 0,
        lower_margin: int = 0, right_margin: int=0, top_margin: int=0) ->\
            ImagePatch:
        """
        Crop the image patch to the smallest bounding box that contains all the
        masks of all the objects.

        Parameters:
        -----------
        objects : List[Object]
            The objects whose bounding box is to be used for cropping.
        """
        # Initialize the bounding box coordinates
        left, lower, right, upper = np.inf, np.inf, -np.inf, -np.inf

        # Iterate over all masks
        for mask in masks:
            # Get the indices of the pixels that belong to the object
            y_indices, x_indices = np.where(mask)

            # Update the bounding box
            left = min(left, x_indices.min() - left_margin)
            lower = min(lower, y_indices.min() - lower_margin)
            right = max(right, x_indices.max() + right_margin)
            upper = max(upper, y_indices.max() + top_margin)

        # Crop the image
        return self.crop((left, lower, right, upper))

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
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

        return ImagePatch(self.cropped_image, left, lower, right, upper, 
                          self.left, self.lower, queues=self.queues,
                          parent_img_patch=self)
        

    def draw_text(self, fig: VisImage, text: str, position: Sequence[int], 
        color: Sequence[float], font_size: int=18, 
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

        bbox_background = contrasting_color(color*255)

        x, y = position
        fig.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": bbox_background, "alpha": 0.8, "pad": 0.7, 
                  "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.img_as_mpl_fig.get_image()

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
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Args:
            img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), 
                       interpolation="nearest")

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
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")
