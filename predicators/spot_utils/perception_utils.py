"""A set of utility functions for integrating deep-learning based perception
models with the Boston Dynamics Spot robot."""

import io
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import bosdyn.client
import bosdyn.client.util
import cv2
import dill as pkl
import imageio.v2 as iio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import requests
import skimage.exposure
from bosdyn.api import image_pb2
from numpy.typing import NDArray
from scipy import ndimage

from predicators import utils
from predicators.settings import CFG
from predicators.structs import Image

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
CAMERA_NAMES = [
    "hand_color_image", "left_fisheye_image", "right_fisheye_image",
    "frontleft_fisheye_image", "frontright_fisheye_image", "back_fisheye_image"
]
RGB_TO_DEPTH_CAMERAS = {
    "hand_color_image": "hand_depth_in_hand_color_frame",
    "left_fisheye_image": "left_depth_in_visual_frame",
    "right_fisheye_image": "right_depth_in_visual_frame",
    "frontleft_fisheye_image": "frontleft_depth_in_visual_frame",
    "frontright_fisheye_image": "frontright_depth_in_visual_frame",
    "back_fisheye_image": "back_depth_in_visual_frame"
}

# Define colors for plotting of depthmap
color1 = (0, 0, 255)  #red
color2 = (0, 165, 255)  #orange
color3 = (0, 255, 255)  #yellow
color4 = (255, 255, 0)  #cyan
color5 = (255, 0, 0)  #blue
color6 = (128, 64, 64)  #violet
colorArr = np.array([[color1, color2, color3, color4, color5, color6]],
                    dtype=np.uint8)
# resize lut to 256 (or more) values
lut = cv2.resize(colorArr, (256, 1), interpolation=cv2.INTER_LINEAR)


def image_to_bytes(img: PIL.Image.Image) -> io.BytesIO:
    """Helper function to convert from a PIL image into a bytes object."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def visualize_output(im: PIL.Image.Image,
                     masks: NDArray,
                     input_boxes: NDArray,
                     classes: NDArray,
                     scores: NDArray,
                     prefix: str,
                     plot: bool = False) -> None:
    """Visualizes the output of SAM; useful for debugging.

    masks, input_boxes, and scores come from the output of SAM.
    Specifically, masks is an array of array bools, input_boxes is an
    array of array of 4 floats (corresponding to the 4 corners of the
    bounding box), classes is an array of strings, and scores is an
    array of floats corresponding to confidence values.
    """
    fig = plt.figure(figsize=(10, 10))
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
    img = utils.fig2data(fig, dpi=150)
    _save_spot_perception_output(img, prefix, plot=plot)


def show_mask(mask: NDArray,
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


def show_box(box: NDArray, ax: matplotlib.axes.Axes) -> None:
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


def query_detic_sam(rgb_image_dict_in: Dict[str, Image], classes: List[str],
                    viz: bool) -> Dict[str, Dict[str, List[NDArray]]]:
    """Send a query to SAM and return the response.

    rgb_image_dict_in is expected to have keys corresponding to camera
    names and values corresponding to images taken from the particular
    camera named in the key. The DETIC-SAM model will send a batched
    query with this dict, and will receive a response per key in the
    dictionary (i.e, per input camera). The output will be a dictionary
    mapping the same cameras named in the input to a dict with 'boxes',
    'classes', 'masks' and 'scores' returned by DETIC-SAM. Importantly,
    we only keep the top-scoring result for each class seen by each
    camera.
    """
    buf_dict = {}
    for source_name in rgb_image_dict_in.keys():
        image = PIL.Image.fromarray(rgb_image_dict_in[source_name])
        buf_dict[source_name] = image_to_bytes(image)

    detic_sam_results: Dict[str, Dict[str, List[NDArray]]] = {}
    for source_name in rgb_image_dict_in.keys():
        detic_sam_results[source_name] = {
            "boxes": [],
            "classes": [],
            "masks": [],
            "scores": []
        }

    # Retry to handle possible wifi issues.
    for _ in range(5):
        try:
            r = requests.post("http://localhost:5550/batch_predict",
                              files=buf_dict,
                              data={"classes": ",".join(classes)})
            break
        except requests.exceptions.ConnectionError:
            continue
    else:
        print("WARNING: DETIC-SAM FAILED, POSSIBLE SERVER/WIFI ISSUE")
        return detic_sam_results

    # If the status code is not 200, then fail.
    if r.status_code != 200:
        print(f"WARNING: DETIC-SAM FAILED! STATUS CODE: {r.status_code}")
        return detic_sam_results

    with io.BytesIO(r.content) as f:
        try:
            arr = np.load(f, allow_pickle=True)
        except pkl.UnpicklingError:
            return detic_sam_results
        for source_name in rgb_image_dict_in.keys():
            curr_boxes = arr[source_name + '_boxes']
            curr_ret_classes = arr[source_name + '_classes']
            curr_masks = arr[source_name + '_masks']
            curr_scores = arr[source_name + '_scores']

            # If there were no detections (which means all the
            # returned values will be numpy arrays of shape (0, 0))
            # then just skip this source.
            if curr_ret_classes.size == 0:
                continue

            d = {
                "boxes": curr_boxes,
                "classes": curr_ret_classes,
                "masks": curr_masks,
                "scores": curr_scores
            }

            image = PIL.Image.fromarray(rgb_image_dict_in[source_name])
            # Optional visualization useful for debugging.
            prefix = f"detic_sam_{source_name}_raw_outputs"
            visualize_output(image,
                             d["masks"],
                             d["boxes"],
                             d["classes"],
                             d["scores"],
                             prefix,
                             plot=viz)

            # Filter out detections by confidence. We threshold detections
            # at a set confidence level minimum, and if there are multiple
            #, we only select the most confident one. This structure makes
            # it easy for us to select multiple detections if that's ever
            # necessary in the future.
            for obj_class in classes:
                class_mask = (d['classes'] == obj_class)
                if not np.any(class_mask):
                    continue
                max_score = np.max(d['scores'][class_mask])
                max_score_idx = np.where(d['scores'] == max_score)[0]
                if d['scores'][
                        max_score_idx] < CFG.spot_vision_detection_threshold:
                    continue
                for key, value in d.items():
                    # Sanity check to ensure that we're selecting a value from
                    # the class we're looking for.
                    if key == "classes":
                        assert value[max_score_idx] == obj_class
                    detic_sam_results[source_name][key].append(
                        value[max_score_idx])

    return detic_sam_results


# NOTE: the below function is useful for visualization; uncomment
# if you'd like to convert depth to pointclouds and then visualize
# the entire pointcloud.
# def depth_image_to_pointcloud_custom(
#         image_response: bosdyn.api.image_pb2.ImageResponse,
#         masks: NDArray = None,
#         min_dist: float = 0.0,
#         max_dist: float = 1000.0) -> Tuple[NDArray, NDArray]:
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
        obj_class: str,
        rgb_image_dict: Dict[str, NDArray],
        plot: bool = False) -> List[Tuple[float, float]]:
    """Method to get the pixel locations of specific objects with class names
    listed in 'classes' within an input image."""
    res_segment = query_detic_sam(rgb_image_dict_in=rgb_image_dict,
                                  classes=[obj_class],
                                  viz=plot)

    assert len(rgb_image_dict.keys()) == 1
    camera_name = list(rgb_image_dict.keys())[0]
    if len(res_segment[camera_name]['classes']) == 0:
        return []

    pixel_locations = []

    # Compute geometric center of object bounding box
    x1, y1, x2, y2 = res_segment[camera_name]['boxes'][0].squeeze()
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2
    pixel_locations.append((x_c, y_c))

    return pixel_locations


def get_object_locations_with_detic_sam(
        classes: List[str],
        rgb_image_dict: Dict[str, Image],
        depth_image_dict: Dict[str, Image],
        depth_image_response_dict: Dict[str,
                                        bosdyn.api.image_pb2.ImageResponse],
        plot: bool = False
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Given a list of string queries (classes), call SAM on these and return
    the positions of the centroids of these detections in the camera frame.

    If the same object is seen multiple times, take only the highest score one.

    Importantly, note that a number of cameras on the Spot robot are
    rotated by various degrees. Since SAM doesn't do so well on rotated
    images, we first rotate these images to be upright, pass them to
    SAM, then rotate the result back so that we can correctly compute
    the 3D position in the camera frame.
    """
    # First, rotate the rgb and depth images by the correct angle.
    # Importantly, DO NOT reshape the image, because this will
    # lead to a bunch of problems when we reverse the rotation later.
    rotated_rgb_image_dict = {}
    rotated_depth_image_dict = {}
    for source_name in CAMERA_NAMES:
        assert source_name in rgb_image_dict and source_name in depth_image_dict
        rotated_rgb = ndimage.rotate(rgb_image_dict[source_name],
                                     ROTATION_ANGLE[source_name],
                                     reshape=False)
        rotated_depth = ndimage.rotate(depth_image_dict[source_name],
                                       ROTATION_ANGLE[source_name],
                                       reshape=False)
        rotated_rgb_image_dict[source_name] = rotated_rgb
        rotated_depth_image_dict[source_name] = rotated_depth

        # Save/plot the rotated image before querying DETIC-SAM.
        _save_spot_perception_output(
            rotated_rgb,
            prefix=f"detic_sam_{source_name}_object_locs_inputs",
            plot=plot)

    # Start by querying the DETIC-SAM model.
    deticsam_results_all_cameras = query_detic_sam(
        rgb_image_dict_in=rotated_rgb_image_dict, classes=classes, viz=plot)

    ret_camera_to_obj_positions: Dict[str,
                                      Dict[str,
                                           Tuple[float, float, float]]] = {
                                               source_name: {}
                                               for source_name in CAMERA_NAMES
                                           }

    # Track the max scores found per object, assuming there is at most one.
    obj_class_to_max_score_and_source: Dict[str, Tuple[float, str]] = {}
    for source_name in CAMERA_NAMES:
        curr_res_segment = deticsam_results_all_cameras[source_name]
        for i, obj_class in enumerate(curr_res_segment['classes']):
            # Check that this particular class is one of the
            # classes we passed in, and that there was only one
            # instance of this class that was found.
            obj_cls_str = obj_class.item()
            assert obj_cls_str in classes
            assert curr_res_segment['classes'].count(obj_class) == 1
            score = curr_res_segment["scores"][i][0]
            if obj_class_to_max_score_and_source.get(obj_cls_str) is None:
                obj_class_to_max_score_and_source[obj_cls_str] = (score,
                                                                  source_name)
            else:
                if score > obj_class_to_max_score_and_source[obj_cls_str][0]:
                    obj_class_to_max_score_and_source[obj_cls_str] = (
                        score, source_name)

    for source_name in CAMERA_NAMES:
        curr_res_segment = deticsam_results_all_cameras[source_name]
        for i, obj_class in enumerate(curr_res_segment['classes']):
            # First, check if this is the highest scoring detection
            # for this class amongst all detections from all sources.
            score = curr_res_segment["scores"][i][0]
            obj_cls_str = obj_class.item()
            # Skip if we've already seen a higher-scoring detection
            # for this object class from a different source. The only
            # exception is if the source is the hand camera: we want to
            # remember all detections that we see from the hand camera,
            # because that is used for predicates like "InView".
            if (score, source_name) != obj_class_to_max_score_and_source[
                    obj_cls_str] and source_name != "hand_color_image":
                continue

            # Compute median value of depth
            curr_rotated_depth = rotated_depth_image_dict[source_name]
            depth_median = np.median(
                curr_rotated_depth[curr_res_segment['masks'][i][0].squeeze()
                                   & (curr_rotated_depth > 2)[:, :, 0]])
            # Compute geometric center of object bounding box
            x1, y1, x2, y2 = curr_res_segment['boxes'][i].squeeze()
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
            rotated_pixel_centered = np.matmul(transform_matrix,
                                               pixel_centered)
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
            inverse_rotation_angle = -ROTATION_ANGLE[source_name]
            fig = plt.figure()
            depth_image = depth_image_dict[source_name]
            stretch = skimage.exposure.rescale_intensity(
                depth_image, in_range='image',
                out_range=(0, 255)).astype(np.uint8)
            # Apply lut.
            result = cv2.LUT(stretch, lut)
            plt.imshow(result)
            plt.imshow(ndimage.rotate(
                curr_res_segment['masks'][i][0].squeeze(),
                inverse_rotation_angle,
                reshape=False),
                       alpha=0.3,
                       cmap='spring')
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
            debug_img = utils.fig2data(fig, dpi=150)
            _save_spot_perception_output(
                debug_img,
                prefix=
                f"detic_sam_{source_name}_{obj_class}_object_locs_outputs",
                plot=plot)

            # Get XYZ of the point at center of bounding box and median depth
            # value.
            x0, y0, z0 = get_xyz_from_depth(
                depth_image_response_dict[source_name],
                depth_value=depth_median,
                point_x=x_c_rotated,
                point_y=y_c_rotated)

            if not math.isnan(x0) and not math.isnan(y0) and not math.isnan(
                    z0):
                ret_camera_to_obj_positions[source_name][obj_cls_str] = (x0,
                                                                         y0,
                                                                         z0)

    return ret_camera_to_obj_positions


def _save_spot_perception_output(img: Image,
                                 prefix: str,
                                 plot: bool = False) -> None:
    if plot:
        plt.close()
        plt.figure()
        plt.axis("off")
        plt.imshow(img)
        plt.show()
    # Save image for debugging.
    time_str = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{time_str}_{prefix}.png"
    outfile = Path(CFG.spot_perception_outdir) / filename
    os.makedirs(CFG.spot_perception_outdir, exist_ok=True)
    iio.imsave(outfile, img)


def _run_offline_analysis() -> None:
    # Convenient script for identifying which classes might be best for a
    # group of images that all have the same object. The images should still
    # be manually inspected (in the debug dir).
    class_candidates = ["hammer", "hammer tool", "mallet"]
    # pylint:disable=line-too-long
    files = [
        "20230804-135530_detic_sam_hand_color_image_object_locs_inputs.png",
        "20230804-135448_detic_sam_hand_color_image_object_locs_inputs.png",
        "20230804-135310_detic_sam_right_fisheye_image_object_locs_inputs.png",
    ]
    root_dir = Path(__file__).parent / "../.."
    utils.reset_config({
        "spot_perception_outdir": root_dir / "spot_perception_debug_dir",
        "spot_vision_detection_threshold": 0.0,
    })

    class_candidate_to_scores: Dict[str, List[float]] = {
        c: []
        for c in class_candidates
    }
    for file in files:
        path = (root_dir / "spot_perception_outputs" / file).resolve()
        img = iio.imread(path)
        # NOTE: cannot batch class candidates for some strange reason, they
        # apparently interfere.
        for candidate in class_candidates:
            results = query_detic_sam({"debug": img}, [candidate], viz=True)
            scores = results["debug"]["scores"]
            assert len(scores) <= 1
            if len(scores) == 1:
                score = scores[0][0]
            else:
                print(f"Class {candidate} not found in {path}")
                score = 0.0
            class_candidate_to_scores[candidate].append(score)
    print("Class candidates in order from best to worst (and all scores):")
    for cc in sorted(class_candidate_to_scores,
                     key=lambda k: sum(class_candidate_to_scores[k]),
                     reverse=True):
        print(cc, class_candidate_to_scores[cc])


if __name__ == "__main__":
    _run_offline_analysis()
