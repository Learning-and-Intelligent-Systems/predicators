""""Interface for detecting objects with fiducials or pretrained models.

The fiducials are april tags. The size of the april tag is important and can be
configured via CFG.spot_fiducial_size.

The pretrained models are currently DETIC and SAM (used together). DETIC takes
a language description of an object (e.g., "brush") and an RGB image and finds
a bounding box. SAM segments objects within the bounding box (class-agnostic).
The associated depth image is then used to estimate the depth of the object
based on the median depth of segmented points. See the README in this directory
for instructions on setting up DETIC and SAM on a server.

Object detection returns SE3Poses in the world frame but only x, y, z positions
are currently detected. Rotations should be ignored.
"""

import io
import logging
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

import apriltag
import cv2
import dill as pkl
import numpy as np
import PIL.Image
import requests
from bosdyn.client import math_helpers
from matplotlib import pyplot as plt
from scipy import ndimage

from predicators.settings import CFG
from predicators.spot_utils.perception.perception_structs import \
    AprilTagObjectDetectionID, LanguageObjectDetectionID, ObjectDetectionID, \
    RGBDImageWithContext, SegmentedBoundingBox
from predicators.utils import rotate_point_in_image


def detect_objects(
    object_ids: Collection[ObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],  # camera name to RGBD
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Detect object poses (in the world frame!) from RGBD.

    Each object ID is assumed to exist at most once in each image, but can
    exist in multiple images.

    The second return value is a collection of artifacts that can be useful
    for debugging / analysis.
    """

    # Collect and dispatch.
    april_tag_object_ids: Set[AprilTagObjectDetectionID] = set()
    language_object_ids: Set[LanguageObjectDetectionID] = set()
    for object_id in object_ids:
        if isinstance(object_id, AprilTagObjectDetectionID):
            april_tag_object_ids.add(object_id)
        else:
            assert isinstance(object_id, LanguageObjectDetectionID)
            language_object_ids.add(object_id)
    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    artifacts: Dict[str, Any] = {"april": {}, "language": {}}

    # There is no batching over images for april tag detection.
    for rgbd in rgbds.values():
        img_detections, img_artifacts = detect_objects_from_april_tags(
            april_tag_object_ids, rgbd)
        # Possibly overrides previous detections.
        detections.update(img_detections)
        artifacts["april"].update(img_artifacts)

    # There IS batching over images here for efficiency.
    language_detections, language_artifacts = detect_objects_from_language(
        language_object_ids, rgbds)
    detections.update(language_detections)
    artifacts["language"] = language_artifacts

    return detections, artifacts


def detect_objects_from_april_tags(
    object_ids: Collection[AprilTagObjectDetectionID],
    rgbd: RGBDImageWithContext,
    fiducial_size: float = CFG.spot_fiducial_size,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Detect an object pose from an april tag.

    The rotation is currently not detected (set to default).
    """
    tag_num_to_object_id = {t.april_tag_number: t for t in object_ids}

    # Convert the RGB image to grayscale.
    image_grey = cv2.cvtColor(rgbd.rgb, cv2.COLOR_RGB2GRAY)

    # Create apriltag detector and get all apriltag locations.
    options = apriltag.DetectorOptions(families="tag36h11")
    options.refine_pose = 1
    detector = apriltag.Detector(options)
    apriltag_detections = detector.detect(image_grey)

    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    artifacts: Dict[str, Any] = {}

    # For every detection, find pose in world frame.
    for apriltag_detection in apriltag_detections:
        # Only include requested tags.
        if apriltag_detection.tag_id not in tag_num_to_object_id:
            continue
        obj_id = tag_num_to_object_id[apriltag_detection.tag_id]

        # Save the detection for external analysis.
        artifact_id = f"apriltag_{rgbd.camera_name}_{obj_id.april_tag_number}"
        artifacts[artifact_id] = apriltag_detection

        # Get the pose from the apriltag library.
        pose = detector.detection_pose(
            apriltag_detection,
            (rgbd.intrinsics.focal_length.x, rgbd.intrinsics.focal_length.y,
             rgbd.intrinsics.principal_point.x,
             rgbd.intrinsics.principal_point.y), fiducial_size)[0]
        tx, ty, tz, tw = pose[:, -1]
        assert np.isclose(tw, 1.0)

        # Detection is in meters, we want mm.
        camera_frame_pose = math_helpers.SE3Pose(
            x=float(tx) / 1000.0,
            y=float(ty) / 1000.0,
            z=float(tz) / 1000.0,
            rot=math_helpers.Quat(),
        )

        # Apply transforms.
        world_frame_pose = rgbd.world_tform_camera * camera_frame_pose
        world_frame_pose = obj_id.offset_transform * world_frame_pose

        # Save in detections.
        detections[obj_id] = world_frame_pose

    return detections, artifacts


def detect_objects_from_language(
    object_ids: Collection[LanguageObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Detect an object pose using a vision-language model."""

    object_id_to_img_detections = _query_detic_sam(object_ids, rgbds)

    # Aggregate detections over images.
    # Get the best-scoring image detections for each object ID.
    object_id_to_best_img: Dict[ObjectDetectionID, RGBDImageWithContext] = {}
    for obj_id, img_detections in object_id_to_img_detections.items():
        if not img_detections:
            continue
        best_score = -np.inf
        best_camera: Optional[str] = None
        for camera, detection in img_detections.items():
            if detection.score > best_score:
                best_score = detection.score
                best_camera = camera
        assert best_camera is not None
        object_id_to_best_img[obj_id] = rgbds[best_camera]

    # Convert the image detections into pose detections.
    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    for obj_id, rgbd in object_id_to_best_img.items():
        seg_bb = object_id_to_img_detections[obj_id][rgbd.camera_name]
        pose = _get_pose_from_segmented_bounding_box(seg_bb, rgbd)
        # Pose extraction can fail due to depth reading issues. See docstring
        # of _get_pose_from_segmented_bounding_box for more.
        if pose is None:
            continue
        detections[obj_id] = pose

    # Save artifacts for analysis and debugging.
    artifacts = {
        "rgbds": rgbds,
        "object_id_to_img_detections": object_id_to_img_detections
    }

    return detections, artifacts


def _query_detic_sam(
    object_ids: Collection[LanguageObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],
    max_server_retries: int = 5,
    detection_threshold: float = CFG.spot_vision_detection_threshold
) -> Dict[ObjectDetectionID, Dict[str, SegmentedBoundingBox]]:
    """Returns object ID to image ID (camera) to segmented bounding box."""

    object_id_to_img_detections: Dict[ObjectDetectionID,
                                      Dict[str, SegmentedBoundingBox]] = {
                                          obj_id: {}
                                          for obj_id in object_ids
                                      }

    # Create buffer dictionary to send to server.
    buf_dict = {}
    for camera_name, rgbd in rgbds.items():
        pil_rotated_img = PIL.Image.fromarray(rgbd.rotated_rgb)
        buf_dict[camera_name] = _image_to_bytes(pil_rotated_img)

    # Extract all the classes that we want to detect.
    classes = sorted(o.language_id for o in object_ids)

    # Query server, retrying to handle possible wifi issues.
    for _ in range(max_server_retries):
        try:
            r = requests.post("http://localhost:5550/batch_predict",
                              files=buf_dict,
                              data={"classes": ",".join(classes)})
            break
        except requests.exceptions.ConnectionError:
            continue
    else:
        logging.warning("DETIC-SAM FAILED, POSSIBLE SERVER/WIFI ISSUE")
        return object_id_to_img_detections

    # If the status code is not 200, then fail.
    if r.status_code != 200:
        logging.warning(f"DETIC-SAM FAILED! STATUS CODE: {r.status_code}")
        return object_id_to_img_detections

    # Querying the server succeeded; unpack the contents.
    with io.BytesIO(r.content) as f:
        try:
            server_results = np.load(f, allow_pickle=True)
        # Corrupted results.
        except pkl.UnpicklingError:
            logging.warning("DETIC-SAM FAILED DURING UNPICKLING!")
            return object_id_to_img_detections

        # Process the results and save all detections per object ID.
        for camera_name, rgbd in rgbds.items():
            rot_boxes = server_results[f"{camera_name}_boxes"]
            ret_classes = server_results[f"{camera_name}_classes"]
            rot_masks = server_results[f"{camera_name}_masks"]
            scores = server_results[f"{camera_name}_scores"]

            # Invert the rotation immediately so we don't need to worry about
            # them henceforth.
            h, w = rgbd.rgb.shape[:2]
            image_rot = rgbd.image_rot
            boxes = [
                _rotate_bounding_box(bb, -image_rot, h, w) for bb in rot_boxes
            ]
            masks = [
                ndimage.rotate(m.squeeze(), -image_rot, reshape=False)
                for m in rot_masks
            ]

            # Filter out detections by confidence. We threshold detections
            # at a set confidence level minimum, and if there are multiple,
            # we only select the most confident one. This structure makes
            # it easy for us to select multiple detections if that's ever
            # necessary in the future.
            for obj_id in object_ids:
                # If there were no detections (which means all the
                # returned values will be numpy arrays of shape (0, 0))
                # then just skip this source.
                if ret_classes.size == 0:
                    continue
                obj_id_mask = (ret_classes == obj_id.language_id)
                if not np.any(obj_id_mask):
                    continue
                max_score = np.max(scores[obj_id_mask])
                best_idx = np.where(scores == max_score)[0].item()
                if scores[best_idx] < detection_threshold:
                    continue
                # Save the detection.
                seg_bb = SegmentedBoundingBox(boxes[best_idx], masks[best_idx],
                                              scores[best_idx])
                object_id_to_img_detections[obj_id][rgbd.camera_name] = seg_bb

    return object_id_to_img_detections


def _image_to_bytes(img: PIL.Image.Image) -> io.BytesIO:
    """Helper function to convert from a PIL image into a bytes object."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _rotate_bounding_box(bb: Tuple[float, float, float,
                                   float], rot_degrees: float, height: int,
                         width: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bb
    ry1, rx1 = rotate_point_in_image(y1, x1, rot_degrees, height, width)
    ry2, rx2 = rotate_point_in_image(y2, x2, rot_degrees, height, width)
    return (rx1, ry1, rx2, ry2)


def _get_pose_from_segmented_bounding_box(
        seg_bb: SegmentedBoundingBox,
        rgbd: RGBDImageWithContext,
        min_depth_value: float = 2) -> Optional[math_helpers.SE3Pose]:
    """Returns None if the depth of the object cannot be estimated.

    The known case where this happens is when the robot's hand occludes
    the depth camera (which is physically above the RGB camera).
    """
    # Get the center of the bounding box.
    x1, y1, x2, y2 = seg_bb.bounding_box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Get the median depth value of segmented points.
    # Filter 0 points out of the depth map.
    seg_mask = seg_bb.mask & (rgbd.depth > min_depth_value)
    segmented_depth = rgbd.depth[seg_mask]
    # See docstring.
    if len(segmented_depth) == 0:
        # logging.warning doesn't work here because of poor spot logging.
        print("WARNING: depth reading failed. Is hand occluding?")
        return None
    depth_value = np.median(segmented_depth)

    # Convert to camera frame position.
    fx = rgbd.intrinsics.focal_length.x
    fy = rgbd.intrinsics.focal_length.y
    cx = rgbd.intrinsics.principal_point.x
    cy = rgbd.intrinsics.principal_point.y
    depth_scale = rgbd.depth_scale
    camera_z = depth_value / depth_scale
    camera_x = np.multiply(camera_z, (x_center - cx)) / fx
    camera_y = np.multiply(camera_z, (y_center - cy)) / fy
    camera_frame_pose = math_helpers.SE3Pose(x=camera_x,
                                             y=camera_y,
                                             z=camera_z,
                                             rot=math_helpers.Quat())

    # Convert camera to world.
    world_frame_pose = rgbd.world_tform_camera * camera_frame_pose

    return world_frame_pose


def _visualize_all_artifacts(artifacts: Dict[str,
                                             Any], detections_outfile: Path,
                             no_detections_outfile: Path) -> None:
    """Analyze the artifacts."""
    # At the moment, only language detection artifacts are visualized.
    rgbds = artifacts["language"]["rgbds"]
    detections = artifacts["language"]["object_id_to_img_detections"]
    flat_detections: List[Tuple[RGBDImageWithContext,
                                LanguageObjectDetectionID,
                                SegmentedBoundingBox]] = []
    for obj_id, img_detections in detections.items():
        for camera, seg_bb in img_detections.items():
            rgbd = rgbds[camera]
            flat_detections.append((rgbd, obj_id, seg_bb))

    # Visualize in subplots where columns are: rotated RGB, original RGB,
    # original depth, bounding box, mask. Each row is one detection, so if
    # there are multiple detections in a single image, then there will be
    # duplicate first cols.
    if flat_detections:
        _, axes = plt.subplots(len(flat_detections), 5, squeeze=False)
        plt.suptitle("Detections")
        for i, (rgbd, obj_id, seg_bb) in enumerate(flat_detections):
            ax_row = axes[i]
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
            ax_row[0].imshow(rgbd.rotated_rgb)
            ax_row[1].imshow(rgbd.rgb)
            ax_row[2].imshow(rgbd.depth, cmap='Greys_r', vmin=0, vmax=10000)

            # Bounding box.
            ax_row[3].imshow(rgbd.rgb)
            box = seg_bb.bounding_box
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax_row[3].add_patch(
                plt.Rectangle((x0, y0),
                              w,
                              h,
                              edgecolor='green',
                              facecolor=(0, 0, 0, 0),
                              lw=1))

            ax_row[4].imshow(seg_bb.mask, cmap="binary_r", vmin=0, vmax=1)

            # Labels.
            row_label = "\n".join([
                obj_id.language_id, f"[{rgbd.camera_name}]",
                f"[score: {seg_bb.score:.2f}]"
            ])
            ax_row[0].set_ylabel(row_label, fontsize=6)
            if i == len(flat_detections) - 1:
                ax_row[0].set_xlabel("Rotated RGB")
                ax_row[1].set_xlabel("Original RGB")
                ax_row[2].set_xlabel("Original Depth")
                ax_row[3].set_xlabel("Bounding Box")
                ax_row[4].set_xlabel("Mask")

        plt.tight_layout()
        plt.savefig(detections_outfile, dpi=300)
        print(f"Wrote out to {detections_outfile}.")

    # Visualize all of the images that have no detections.
    all_cameras = set(rgbds)
    cameras_with_detections = {r.camera_name for r, _, _ in flat_detections}
    cameras_without_detections = sorted(all_cameras - cameras_with_detections)

    if cameras_without_detections:
        _, axes = plt.subplots(len(cameras_without_detections),
                               3,
                               squeeze=False)
        plt.suptitle("Cameras without Detections")
        for i, camera in enumerate(cameras_without_detections):
            rgbd = rgbds[camera]
            ax_row = axes[i]
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
            ax_row[0].imshow(rgbd.rotated_rgb)
            ax_row[1].imshow(rgbd.rgb)
            ax_row[2].imshow(rgbd.depth, cmap='Greys_r', vmin=0, vmax=10000)

            # Labels.
            ax_row[0].set_ylabel(f"[{rgbd.camera_name}]", fontsize=6)
            if i == len(flat_detections) - 1:
                ax_row[0].set_xlabel("Rotated RGB")
                ax_row[1].set_xlabel("Original RGB")
                ax_row[2].set_xlabel("Original Depth")

        plt.tight_layout()
        plt.savefig(no_detections_outfile, dpi=300)
        print(f"Wrote out to {no_detections_outfile}.")


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: make sure the spot hand camera sees the 408 april tag, a brush,
    # and a drill. It is recommended to run this test a few times in a row
    # while moving the robot around, but keeping the objects in place.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.spot_utils.perception.spot_cameras import capture_images
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import verify_estop

    TEST_CAMERAS = [
        "hand_color_image", "frontleft_fisheye_image", "left_fisheye_image",
        "right_fisheye_image"
    ]
    TEST_APRIL_TAG_ID = 408
    # Assume the table is oriented such the tag is in the front with respect
    # to the world frame. In the 4th floor room, this is facing such that the
    # outside hall is on the left of the tag.
    TEST_APRIL_TAG_TRANSFORM = math_helpers.SE3Pose(0.0, 0.5, 0.0,
                                                    math_helpers.Quat())
    TEST_LANGUAGE_DESCRIPTIONS = ["brush", "drill"]

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip
        upload_dir = Path(__file__).parent.parent / "graph_nav_maps"
        path = upload_dir / CFG.spot_graph_nav_map

        # First, capture images.
        sdk = create_standard_sdk('SpotCameraTestClient')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_keepalive = LeaseKeepAlive(lease_client,
                                         must_acquire=True,
                                         return_at_exit=True)

        assert path.exists()
        localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
        rgbds = capture_images(robot, localizer, TEST_CAMERAS)

        # Detect the april tag and brush.
        april_tag_id: ObjectDetectionID = AprilTagObjectDetectionID(
            TEST_APRIL_TAG_ID, TEST_APRIL_TAG_TRANSFORM)
        language_ids: List[ObjectDetectionID] = [
            LanguageObjectDetectionID(d) for d in TEST_LANGUAGE_DESCRIPTIONS
        ]
        object_ids: List[ObjectDetectionID] = [april_tag_id] + language_ids
        detections, artifacts = detect_objects(object_ids, rgbds)
        for obj_id, detection in detections.items():
            print(f"Detected {obj_id} at {detection}")

        # Visualize the artifacts.
        detections_outfile = Path(".") / "object_detection_artifacts.png"
        no_detections_outfile = Path(".") / "no_detection_artifacts.png"
        _visualize_all_artifacts(artifacts, detections_outfile,
                                 no_detections_outfile)

    _run_manual_test()