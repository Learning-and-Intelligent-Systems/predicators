"""A set of utility functions for integrating deep-learning based perception
models with the Boston Dynamics Spot robot."""

import io
from typing import Dict, List, Optional, Tuple

import bosdyn.client
import bosdyn.client.util
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
from bosdyn.api import image_pb2
from PIL import Image
from scipy import ndimage

from predicators.settings import CFG

# NOTE: uncomment this line if trying to visualize stuff locally
# and matplotlib isn't displaying.
# matplotlib.use('TkAgg')

ROTATION_ANGLE = {
    'hand_color_image': 0,
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


def image_to_bytes(img: Image.Image) -> io.BytesIO:
    """Helper function to convert from a PIL image into a bytes object."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def visualize_output(im: Image.Image, masks: np.ndarray,
                     input_boxes: np.ndarray, classes: np.ndarray,
                     scores: np.ndarray) -> None:
    """Visualizes the output of SAM; useful for debugging.

    masks, input_boxes, and scores come from the output of SAM.
    Specifically, masks is an array of array bools, input_boxes is an
    array of array of 4 floats (corresponding to the 4 corners of the
    bounding box), classes is an array of strings, and scores is an
    array of floats corresponding to confidence values.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box, class_name, score in zip(input_boxes, classes, scores):
        show_box(box, plt.gca())
        x, y = box[:2]
        plt.gca().text(x,
                       y - 5,
                       class_name + f': {score:.2f}',
                       color='white',
                       fontsize=12,
                       fontweight='bold',
                       bbox=dict(facecolor='green',
                                 edgecolor='green',
                                 alpha=0.5))
    plt.axis('off')
    plt.show()


def show_mask(mask: np.ndarray,
              ax: matplotlib.axes.Axes,
              random_color: bool = False) -> None:
    """Helper function for visualization that displays a segmentation mask."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape((h, w, 1)) * color.reshape((1, 1, -1))
    ax.imshow(mask_image)


def show_box(box: np.ndarray, ax: matplotlib.axes.Axes) -> None:
    """Helper function for visualization that displays a bounding box."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0),
                      w,
                      h,
                      edgecolor='green',
                      facecolor=(0, 0, 0, 0),
                      lw=2))


def query_detic_sam(image_in: np.ndarray, classes: List[str],
                    viz: bool) -> Optional[Dict[str, List[np.ndarray]]]:
    """Send a query to SAM and return the response.

    The response is a dictionary that contains 4 keys: 'boxes',
    'classes', 'masks' and 'scores'.
    """
    image = Image.fromarray(image_in)
    buf = image_to_bytes(image)
    r = requests.post("http://localhost:5550/predict",
                      files={"file": buf},
                      data={"classes": ",".join(classes)})

    # If the status code is not 200, then fail.
    if r.status_code != 200:
        return None

    with io.BytesIO(r.content) as f:
        arr = np.load(f, allow_pickle=True)
        boxes = arr['boxes']
        ret_classes = arr['classes']
        masks = arr['masks']
        scores = arr['scores']

    d = {
        "boxes": boxes,
        "classes": ret_classes,
        "masks": masks,
        "scores": scores
    }

    if viz:
        # Optional visualization useful for debugging.
        visualize_output(image, d["masks"], d["boxes"], d["classes"],
                         d["scores"])

    # Filter out detections by confidence. We threshold detections
    # at a set confidence level minimum, and if there are multiple
    #, we only select the most confident one. This structure makes
    # it easy for us to select multiple detections if that's ever
    # necessary in the future.
    selected_idx = np.argmax(d['scores'])
    if d['scores'][selected_idx] < CFG.spot_vision_detection_threshold:
        return None
    d_filtered: Dict[str, List[np.ndarray]] = {
        "boxes": [],
        "classes": [],
        "masks": [],
        "scores": []
    }
    for key, value in d.items():
        d_filtered[key].append(value[selected_idx])

    return d_filtered


# NOTE: the below function is useful for visualization; uncomment
# if you'd like to convert depth to pointclouds and then visualize
# the entire pointcloud.
# def depth_image_to_pointcloud_custom(
#         image_response: bosdyn.api.image_pb2.ImageResponse,
#         masks: np.ndarray = None,
#         min_dist: float = 0.0,
#         max_dist: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
#     """Converts a depth image into a point cloud using the camera intrinsics.
#     The point cloud is represented as a numpy array of (x,y,z) values.
#     Requests can optionally filter the results based on the points distance
#     to the image plane. A depth image is represented with an unsigned 16 bit
#     integer and a scale factor to convert that distance to meters. In
#     addition, values of zero and 2^16 (uint 16 maximum) are used to
#     represent invalid indices.

#     A (min_dist * depth_scale) value that casts to an integer value <=0
#     will be assigned a value of 1 (the minimum representational distance).
#     Similarly, a (max_dist * depth_scale) value that casts to >= 2^16 will
#     be assigned a value of 2^16 - 1 (the maximum representational distance).

#     Args:
#         image_response (image_pb2.ImageResponse): An ImageResponse
#             containing a depth image.
#         min_dist (float): All points in the returned point cloud will be
#             greater than min_dist from the image plane [meters].
#         max_dist (float): All points in the returned point cloud will be
#             less than max_dist from the image plane [meters].

#     Returns:
#         A numpy stack of (x,y,z) values representing depth image as a point
#             cloud expressed in the sensor frame.
#     """
# from bosdyn.client.image import _depth_image_data_to_numpy, \
#     _depth_image_get_valid_indices
#     if image_response.source.image_type != \
#             image_pb2.ImageSource.IMAGE_TYPE_DEPTH:
#         raise ValueError('requires an image_type of IMAGE_TYPE_DEPTH.')

#     if image_response.shot.image.pixel_format != \
#             image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
#         raise ValueError(
#             'IMAGE_TYPE_DEPTH with an unsupported format, requires' +
#             'PIXEL_FORMAT_DEPTH_U16.')

#     if not image_response.source.HasField('pinhole'):
#         raise ValueError('Requires a pinhole camera_model.')

#     source_rows = image_response.source.rows
#     source_cols = image_response.source.cols
#     fx = image_response.source.pinhole.intrinsics.focal_length.x
#     fy = image_response.source.pinhole.intrinsics.focal_length.y
#     cx = image_response.source.pinhole.intrinsics.principal_point.x
#     cy = image_response.source.pinhole.intrinsics.principal_point.y
#     depth_scale = image_response.source.depth_scale

#     # Convert the proto representation into a numpy array.
#     depth_array = _depth_image_data_to_numpy(image_response)
#     # print(depth_array.shape)

#     # Determine which indices have valid data in the user requested range.
#     valid_inds = _depth_image_get_valid_indices(
#         depth_array, np.rint(min_dist * depth_scale),
#         np.rint(max_dist * depth_scale))

#     if masks is not None:
#         valid_inds = valid_inds & masks

#     # Compute the valid data.
#     rows, cols = np.mgrid[0:source_rows, 0:source_cols]
#     depth_array = depth_array[valid_inds]
#     rows = rows[valid_inds]
#     cols = cols[valid_inds]

#     # Convert the valid distance data to (x,y,z) values
#     # expressed in the sensor frame.
#     z = depth_array / depth_scale
#     x = np.multiply(z, (cols - cx)) / fx
#     y = np.multiply(z, (rows - cy)) / fy
#     return np.vstack((x, y, z)).T, valid_inds


def process_image_response(image_response: bosdyn.api.image_pb2.ImageResponse,
                           to_rgb: bool = False) -> np.ndarray:
    """Given a Boston Dynamics SDK image response, extract the correct np array
    corresponding to the image."""
    # pylint: disable=no-member
    num_bytes = 1  # Assume a default of 1 byte encodings.
    if image_response.shot.image.pixel_format == \
            image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
    else:
        if image_response.shot.image.pixel_format == \
                image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_bytes = 3
        elif image_response.shot.image.pixel_format == \
                image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_bytes = 4
        elif image_response.shot.image.pixel_format == \
                image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_bytes = 1
        elif image_response.shot.image.pixel_format == \
                image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_bytes = 2
        dtype = np.uint8  # type: ignore

    img = np.frombuffer(image_response.shot.image.data, dtype=dtype)
    if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image_response.shot.image.rows,
                               image_response.shot.image.cols, num_bytes))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    # Convert to RGB color, as some perception models assume RGB format
    # By default, still use BGR to keep backward compability
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_xyz_from_depth(image_response: bosdyn.api.image_pb2.ImageResponse,
                       depth_value: float, point_x: float,
                       point_y: float) -> Tuple[float, float, float]:
    """This is a function based on `depth_image_to_pointcloud`"""
    # pylint: disable=no-member
    # Make sure all the necessary fields exist.
    if image_response.source.image_type != \
            image_pb2.ImageSource.IMAGE_TYPE_DEPTH:
        raise ValueError('requires an image_type of IMAGE_TYPE_DEPTH.')
    if image_response.shot.image.pixel_format != \
            image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        raise ValueError(
            'IMAGE_TYPE_DEPTH with an unsupported format, requires ' +
            'PIXEL_FORMAT_DEPTH_U16.')
    if not image_response.source.HasField('pinhole'):
        raise ValueError('Requires a pinhole camera_model.')
    # Next, compute the xyz point.
    fx = image_response.source.pinhole.intrinsics.focal_length.x
    fy = image_response.source.pinhole.intrinsics.focal_length.y
    cx = image_response.source.pinhole.intrinsics.principal_point.x
    cy = image_response.source.pinhole.intrinsics.principal_point.y
    depth_scale = image_response.source.depth_scale

    # Convert the valid distance data to (x,y,z) values expressed in the
    # sensor frame.
    z = depth_value / depth_scale
    x = np.multiply(z, (point_x - cx)) / fx
    y = np.multiply(z, (point_y - cy)) / fy

    return x, y, z


def get_pixel_locations_with_detic_sam(
        classes: List[str],
        in_res_image: Dict[str, np.ndarray],
        plot: bool = False) -> List[Tuple[float, float]]:
    """Method to get the pixel locations of specific objects with class names
    listed in 'classes' within an input image."""
    res_segment = query_detic_sam(image_in=in_res_image['rgb'],
                                  classes=classes,
                                  viz=plot)
    # return: 'masks', 'boxes', 'classes'
    if res_segment is None:
        return []

    obj_num = len(res_segment['masks'])
    assert obj_num == 1

    pixel_locations = []

    # Detect multiple objects with their masks
    for i in range(obj_num):
        # Compute geometric center of object bounding box
        x1, y1, x2, y2 = res_segment['boxes'][i]
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2
        # Plot center and segmentation mask
        if plot:
            plt.imshow(res_segment['masks'][i][0])
            plt.show()
        pixel_locations.append((x_c, y_c))

    return pixel_locations


def get_object_locations_with_detic_sam(
        classes: List[str],
        res_image: Dict[str, np.ndarray],
        res_image_responses: Dict[str, bosdyn.api.image_pb2.ImageResponse],
        source_name: str,
        plot: bool = False) -> List[Tuple[float, float, float]]:
    """Given a list of string queries (classes), call SAM on these and return
    the positions of the centroids of these detections in the world frame.

    Importantly, note that a number of cameras on the Spot robot are
    rotated by various degrees. Since SAM doesn't do so well on rotated
    images, we first rotate these images to be upright, pass them to
    SAM, then rotate the result back so that we can correctly compute
    the 3D position in the world frame.
    """
    # First, rotate the rgb and depth images by the correct angle.
    # Importantly, DO NOT reshape the image, because this will
    # lead to a bunch of problems when we reverse the rotation later.
    rotated_rgb = ndimage.rotate(res_image['rgb'],
                                 ROTATION_ANGLE[source_name],
                                 reshape=False)
    rotated_depth = ndimage.rotate(res_image['depth'],
                                   ROTATION_ANGLE[source_name],
                                   reshape=False)

    # Plot the rotated image before querying SAM.
    if plot:
        plt.imshow(rotated_rgb)
        plt.show()

    # Start by querying SAM
    res_segment = query_detic_sam(image_in=rotated_rgb,
                                  classes=classes,
                                  viz=plot)
    if res_segment is None:
        return []

    # Detect multiple objects with their masks
    obj_num = len(res_segment['masks'])
    res_locations = []
    for i in range(obj_num):
        # Compute median value of depth
        depth_median = np.median(rotated_depth[res_segment['masks'][i][0]
                                               & (rotated_depth > 2)[:, :, 0]])
        # Compute geometric center of object bounding box
        x1, y1, x2, y2 = res_segment['boxes'][i]
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2
        # Create a transformation matrix for the rotation. Be very
        # careful to use radians, since np.cos and np.sin expect
        # angles in radians and not degrees.
        rotation_radians = np.radians(ROTATION_ANGLE[source_name])
        transform_matrix = np.array(
            [[np.cos(rotation_radians), -np.sin(rotation_radians)],
             [np.sin(rotation_radians),
              np.cos(rotation_radians)]])
        # Subtract the center of the image from the pixel location to
        # translate the rotation to the origin.
        center = np.array(
            [rotated_rgb.shape[1] / 2., rotated_rgb.shape[0] / 2.])
        pixel_centered = np.array([x_c, y_c]) - center
        # Apply the rotation
        rotated_pixel_centered = np.matmul(transform_matrix, pixel_centered)
        # Add the center of the image back to the pixel location to
        # translate the rotation back from the origin.
        rotated_pixel = rotated_pixel_centered + center
        # Now rotated_pixel is the location of the centroid pixel after the
        # inverse rotation.
        x_c_rotated = rotated_pixel[0]
        y_c_rotated = rotated_pixel[1]

        # Plot (1) the original RGB image, (2) the rotated
        # segmentation mask from SAM on top of it, (3) the
        # center of the image, (4) the centroid of the detected
        # object that comes from SAM, and (5) the centroid
        # after we rotate it back to align with the original
        # RGB image.
        if plot:
            inverse_rotation_angle = -ROTATION_ANGLE[source_name]
            plt.imshow(res_image['rgb'])
            plt.imshow(ndimage.rotate(res_segment['masks'][i][0],
                                      inverse_rotation_angle,
                                      reshape=False),
                       alpha=0.5,
                       cmap='Reds')
            plt.scatter(x=x_c_rotated,
                        y=y_c_rotated,
                        marker='*',
                        color='red',
                        zorder=3)
            plt.scatter(x=center[0],
                        y=center[1],
                        marker='.',
                        color='blue',
                        zorder=3)
            plt.scatter(x=x_c, y=y_c, marker='*', color='green', zorder=3)
            plt.show()

        # Get XYZ of the point at center of bounding box and median depth value.
        x0, y0, z0 = get_xyz_from_depth(res_image_responses['depth'],
                                        depth_value=depth_median,
                                        point_x=x_c_rotated,
                                        point_y=y_c_rotated)
        res_locations.append((x0, y0, z0))

    return res_locations
