import io
from dataclasses import dataclass
from typing import List, Tuple

import bosdyn.client
import bosdyn.client
import bosdyn.client.util
import bosdyn.client.util
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request, _depth_image_data_to_numpy, \
    _depth_image_get_valid_indices
from scipy import ndimage

import matplotlib
matplotlib.use('TkAgg')

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


def ask_sam(image, classes):
    buf = image_to_bytes(image)
    r = requests.post("http://localhost:5550/predict",
                      files={"file": buf},
                      data={"classes": ",".join(classes)}
                     )

    if r.status_code != 200:
        assert False, r.content

    with io.BytesIO(r.content) as f:
        arr = np.load(f, allow_pickle=True)
        boxes = arr['boxes']
        classes = arr['classes']
        masks = arr['masks']
        scores = arr['scores']

    return dict(boxes=boxes, classes=classes, masks=masks, scores=scores)


def image_to_bytes(img):

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def visualize_output(im, masks, input_boxes, classes, scores):
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box, class_name, score in zip(input_boxes, classes, scores):
        show_box(box, plt.gca())
        x, y = box[:2]
        plt.gca().text(x, y - 5, class_name + f': {score:.2f}', color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='green', edgecolor='green', alpha=0.5))
    plt.axis('off')
    plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def get_mask(image_in, classes):
    if isinstance(image_in, str):
        image = Image.open(image_in)
    elif isinstance(image_in, np.ndarray):
        image = Image.fromarray(image_in)
    else:
        raise NotImplementedError

    d = ask_sam(image, classes)

    if d is not None:
        visualize_output(image, d["masks"], d["boxes"], d["classes"], d["scores"])

    return d


# TODO for getting hand image
def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)


def depth_image_to_pointcloud_custom(image_response, masks=None, min_dist=0, max_dist=1000):
    """Converts a depth image into a point cloud using the camera intrinsics. The point
    cloud is represented as a numpy array of (x,y,z) values.  Requests can optionally filter
    the results based on the points distance to the image plane. A depth image is represented
    with an unsigned 16 bit integer and a scale factor to convert that distance to meters. In
    addition, values of zero and 2^16 (uint 16 maximum) are used to represent invalid indices.
    A (min_dist * depth_scale) value that casts to an integer value <=0 will be assigned a
    value of 1 (the minimum representational distance). Similarly, a (max_dist * depth_scale)
    value that casts to >= 2^16 will be assigned a value of 2^16 - 1 (the maximum
    representational distance).

    Args:
        image_response (image_pb2.ImageResponse): An ImageResponse containing a depth image.
        min_dist (double): All points in the returned point cloud will be greater than min_dist from the image plane [meters].
        max_dist (double): All points in the returned point cloud will be less than max_dist from the image plane [meters].

    Returns:
        A numpy stack of (x,y,z) values representing depth image as a point cloud expressed in the sensor frame.
    """

    if image_response.source.image_type != image_pb2.ImageSource.IMAGE_TYPE_DEPTH:
        raise ValueError('requires an image_type of IMAGE_TYPE_DEPTH.')

    if image_response.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        raise ValueError(
            'IMAGE_TYPE_DEPTH with an unsupported format, requires PIXEL_FORMAT_DEPTH_U16.')

    if not image_response.source.HasField('pinhole'):
        raise ValueError('Requires a pinhole camera_model.')

    source_rows = image_response.source.rows
    source_cols = image_response.source.cols
    fx = image_response.source.pinhole.intrinsics.focal_length.x
    fy = image_response.source.pinhole.intrinsics.focal_length.y
    cx = image_response.source.pinhole.intrinsics.principal_point.x
    cy = image_response.source.pinhole.intrinsics.principal_point.y
    depth_scale = image_response.source.depth_scale

    # Convert the proto representation into a numpy array.
    depth_array = _depth_image_data_to_numpy(image_response)
    # print(depth_array.shape)

    # Determine which indices have valid data in the user requested range.
    valid_inds = _depth_image_get_valid_indices(depth_array, np.rint(min_dist * depth_scale),
                                                np.rint(max_dist * depth_scale))

    if masks is not None:
        valid_inds = valid_inds & masks

    # Compute the valid data.
    rows, cols = np.mgrid[0:source_rows, 0:source_cols]
    depth_array = depth_array[valid_inds]
    rows = rows[valid_inds]
    cols = cols[valid_inds]

    # Convert the valid distance data to (x,y,z) values expressed in the sensor frame.
    z = depth_array / depth_scale
    x = np.multiply(z, (cols - cx)) / fx
    y = np.multiply(z, (rows - cy)) / fy
    return np.vstack((x, y, z)).T, valid_inds


def process_image_response(image, options):
    num_bytes = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = ".png"
    else:
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_bytes = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_bytes = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_bytes = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_bytes = 2
        dtype = np.uint8
        extension = ".jpg"

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    if options.auto_rotate:
        img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

    return img, extension


def get_hand_img(options):
    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(options.image_service)

    # image_sources = ["frontleft_fisheye_image", "frontleft_depth_in_visual_frame"]
    image_sources = ["hand_color_image", "hand_depth_in_hand_color_frame"]

    pixel_format = pixel_format_string_to_enum(options.pixel_format)
    image_request = [
        build_image_request(source, pixel_format=pixel_format)
        for source in image_sources
    ]
    image_responses = image_client.get_image(image_request)

    color_img, _ = process_image_response(image_responses[0], options)
    print(color_img.shape)
    # plt.imshow(color_img)
    # plt.show()

    x, valid_inds = depth_image_to_pointcloud_custom(image_responses[1])

    d = np.sqrt(np.sum(np.square(x), axis=1))
    c = (color_img[:, :, ::-1][valid_inds] / 255.).astype(np.float32)
    # c = np.tile((color_img[valid_inds] / 255.).astype(np.float32)[:, None], (1, 3))

    x = x[d < 2.]
    c = c[d < 2.]

    # print(x.shape, c.shape)

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(x)
    # pcd.colors = o3d.utility.Vector3dVector(c)
    # o3d.visualization.draw_geometries([pcd])
    # FIXME error: [Open3D WARNING] [DrawGeometries] Failed creating OpenGL window.

    res = {
        'rgb': process_image_response(image_responses[0], options)[0],
        'depth': process_image_response(image_responses[1], options)[0],
    }

    return res, image_responses


def get_xyz_from_depth(image_response, depth_value, point_x, point_y, min_dist=0, max_dist=1000):
    """
    This is a function based on `depth_image_to_pointcloud`
    """
    if image_response.source.image_type != image_pb2.ImageSource.IMAGE_TYPE_DEPTH:
        raise ValueError('requires an image_type of IMAGE_TYPE_DEPTH.')

    if image_response.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        raise ValueError(
            'IMAGE_TYPE_DEPTH with an unsupported format, requires PIXEL_FORMAT_DEPTH_U16.')

    if not image_response.source.HasField('pinhole'):
        raise ValueError('Requires a pinhole camera_model.')

    fx = image_response.source.pinhole.intrinsics.focal_length.x
    fy = image_response.source.pinhole.intrinsics.focal_length.y
    cx = image_response.source.pinhole.intrinsics.principal_point.x
    cy = image_response.source.pinhole.intrinsics.principal_point.y
    depth_scale = image_response.source.depth_scale

    # Convert the valid distance data to (x,y,z) values expressed in the sensor frame.
    z = depth_value / depth_scale
    x = np.multiply(z, (point_x - cx)) / fx
    y = np.multiply(z, (point_y - cy)) / fy
    # return np.vstack((x, y, z)).T, valid_inds
    return x, y, z


def get_pixel_locations_with_sam(
        args,
        classes: List,
        in_res_image=None, in_res_image_responses=None,
        plot: bool = False  # TODO for now
) -> List[Tuple[float, float]]:
    if in_res_image is None or in_res_image_responses is None:
        res_image, res_image_responses = get_hand_img(options=args)
        # return of res_image: 'rgb', 'depth'
        # return of res_image_responses: RGB and depth image responses with camera info
    else:
        res_image = in_res_image
        res_image_responses = in_res_image_responses

    if plot:
        plt.imshow(res_image['rgb'])
        plt.show()

    res_segment = get_mask(image_in=res_image['rgb'], classes=classes)
    # return: 'masks', 'boxes', 'classes'
    if res_segment is None:
        return []

    obj_num = len(res_segment['masks'])

    pixel_locations = []

    # Detect multiple objects with their masks
    for i in range(obj_num):
        # Compute median value of depth
        depth_median = np.median(
            res_image['depth'][res_segment['masks'][i][0] & (res_image['depth'] > 2)[:, :, 0]]
            # res_image['depth'][res_segment['masks'][i][0] & (res_image['depth'] > 2)]  # FIXME not sure why
        )

        # Compute geometric center of object bounding box
        x1, y1, x2, y2 = res_segment['boxes'][i]
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2

        # Plot center and segmentation mask
        if plot:
            plt.imshow(res_segment['masks'][i][0])
            # plt.scatter(x=x_c, y=y_c, marker='*', color='red', zorder=3)
            plt.show()
        pixel_locations.append((x_c, y_c))
    return pixel_locations    


def get_object_locations_with_sam(
        args,
        classes: list,
        in_res_image=None, in_res_image_responses=None,
        plot: bool = False  # TODO for now
):
    
    if in_res_image is None or in_res_image_responses is None:
        res_image, res_image_responses = get_hand_img(options=args)
        # return of res_image: 'rgb', 'depth'
        # return of res_image_responses: RGB and depth image responses with camera info
    else:
        res_image = in_res_image
        res_image_responses = in_res_image_responses

    if plot:
        plt.imshow(res_image['rgb'])
        plt.show()

    res_segment = get_mask(image_in=res_image['rgb'], classes=classes)
    # return: 'masks', 'boxes', 'classes'
    if res_segment is None:
        return []

    obj_num = len(res_segment['masks'])

    # TODO filter mask IDs by confidence scores
    k = 1
    topk_idx = np.argpartition(res_segment['scores'], -k)[-k:]
    threshold_idx = res_segment['scores'] > 0.5
    selected_idx = np.intersect1d(topk_idx, threshold_idx)

    print(f'Hardcode: select top-{k} from {obj_num} detected objects')

    res_locations = []

    # Detect multiple objects with their masks
    # for i in range(obj_num):
    for i in selected_idx:
        # Compute median value of depth
        depth_median = np.median(
            res_image['depth'][res_segment['masks'][i][0] & (res_image['depth'] > 2)[:, :, 0]]
            # res_image['depth'][res_segment['masks'][i][0] & (res_image['depth'] > 2)]  # FIXME not sure why
        )

        # Compute geometric center of object bounding box
        x1, y1, x2, y2 = res_segment['boxes'][i]
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2

        # Plot center and segmentation mask
        if plot:
            plt.imshow(res_segment['masks'][i][0])
            # plt.scatter(x=x_c, y=y_c, marker='*', color='red', zorder=3)
            plt.show()

        # Get XYZ of the point at center of bounding box and median depth value
        x0, y0, z0 = get_xyz_from_depth(
            res_image_responses['depth'],
            depth_value=depth_median,
            point_x=x_c,
            point_y=y_c
        )

        res_locations.append([x0, y0, z0])

        x, valid_inds = depth_image_to_pointcloud_custom(
            res_image_responses['depth'],
            masks=res_segment['masks'][i][0],
        )

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='blue', marker='.')

            ax.scatter(xs=x0, ys=y0, zs=z0, c='red', marker='*', s=100)
            plt.show()

    return res_locations

@dataclass
class TempArgs:
    # hostname: str = '10.17.30.30'  # 6th floor
    # hostname: str = '10.17.30.21'  #
    hostname: str = '10.17.30.29'  #
    list: bool = True
    auto_rotate: bool = False  # unnecessary for hand
    image_service: str = ImageClient.default_service_name
    pixel_format: list = tuple(pixel_format_type_strings())


if __name__ == '__main__':
    args = TempArgs()

    get_object_locations_with_sam(
        args,
        classes=['desk'],
        # in_res_image=res1, in_res_image_responses=image_responses1
    )
