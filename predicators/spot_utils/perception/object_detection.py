"""Interface for detecting objects with fiducials or pretrained models.

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

# TODO: refactor so that transform stuff happens externally to object detection

import io
import logging
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

import apriltag
import cv2
import dill as pkl
import numpy as np
import PIL.Image
import requests
from bosdyn.api import image_pb2
from bosdyn.client import math_helpers
from numpy.typing import NDArray

from predicators.settings import CFG
from predicators.spot_utils.perception.structs import ObjectDetectionID, LanguageObjectDetectionID, SegmentedBoundingBox, AprilTagObjectDetectionID, RGBDImageWithContext


def detect_objects(
    object_ids: Collection[ObjectDetectionID],
    rgbds: Collection[RGBDImageWithContext],
    world_tform_body: math_helpers.SE3Pose,
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
    artifacts: Dict[str, Any] = {}

    # There is no batching over images for april tag detection.
    for rgbd in rgbds:
        img_detections, img_artifacts = detect_objects_from_april_tags(
            april_tag_object_ids, rgbd, world_tform_body)
        # Possibly overrides previous detections.
        detections.update(img_detections)
        artifacts.update(img_artifacts)

    # There IS batching over images here for efficiency.
    language_detections, language_artifacts = detect_objects_from_language(
        language_object_ids, rgbds, world_tform_body)
    detections.update(language_detections)
    artifacts.update(language_artifacts)
    
    return detections, artifacts


def detect_objects_from_april_tags(
    object_ids: Collection[AprilTagObjectDetectionID],
    rgbd: RGBDImageWithContext,
    world_tform_body: math_helpers.SE3Pose,
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
             rgbd.intrinsics.principal_point.x, rgbd.intrinsics.principal_point.y),
            fiducial_size)[0]
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
        body_frame_pose = rgbd.body_tform_camera * camera_frame_pose
        world_frame_pose = world_tform_body * body_frame_pose
        world_frame_pose = obj_id.offset_transform * world_frame_pose

        # Save in detections
        detections[obj_id] = world_frame_pose

    return detections, artifacts


def image_to_bytes(img: PIL.Image.Image) -> io.BytesIO:
    """Helper function to convert from a PIL image into a bytes object."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def detect_objects_from_language(
    object_ids: Collection[LanguageObjectDetectionID],
    rgbds: List[RGBDImageWithContext],
    world_tform_body: math_helpers.SE3Pose,
    max_server_retries: int = 5,
    detection_threshold: float = CFG.spot_vision_detection_threshold,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Detect an object pose using a vision-language model."""

    # TODO refactor!

    # Initialize detections and artifacts.
    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    artifacts: Dict[str, Any] = {}

    # Create buffer dictionary to send to server.
    buf_dict = {}
    for rgbd in rgbds:
        pil_rotated_img = PIL.Image.fromarray(rgbd.rotated_rgb)
        buf_dict[rgbd.camera_name] = image_to_bytes(pil_rotated_img)

    # Extract all the classes that we want to detect.
    language_ids = sorted(o.language_id for o in object_ids)

    # Query server, retrying to handle possible wifi issues.
    for _ in range(max_server_retries):
        try:
            r = requests.post("http://localhost:5550/batch_predict",
                              files=buf_dict,
                              data={"classes": ",".join(language_ids)})
            break
        except requests.exceptions.ConnectionError:
            continue
    else:
        logging.warning("DETIC-SAM FAILED, POSSIBLE SERVER/WIFI ISSUE")
        return detections, artifacts

    # If the status code is not 200, then fail.
    if r.status_code != 200:
        logging.warning(f"DETIC-SAM FAILED! STATUS CODE: {r.status_code}")
        return detections, artifacts

    # Querying the server succeeded; unpack the contents.
    with io.BytesIO(r.content) as f:
        try:
            server_results = np.load(f, allow_pickle=True)
        # Corrupted results.
        except pkl.UnpicklingError:
            logging.warning("DETIC-SAM FAILED DURING UNPICKLING!")
            return detections, artifacts

    # Process the results and save all detections per object ID.
    object_id_to_img_detections: Dict[ObjectDetectionID,
                                      Dict[str, SegmentedBoundingBox]] = {
                                          obj_id: {}
                                          for obj_id in object_ids
                                      }

    for rgbd in rgbds:
        boxes = server_results[f"{rgbd.camera_name}_boxes"]
        ret_language_ids = server_results[f"{rgbd.camera_name}_classes"]
        masks = server_results[f"{rgbd.camera_name}_masks"]
        scores = server_results[f"{rgbd.camera_name}_scores"]

        # Filter out detections by confidence. We threshold detections
        # at a set confidence level minimum, and if there are multiple,
        # we only select the most confident one. This structure makes
        # it easy for us to select multiple detections if that's ever
        # necessary in the future.
        for language_id in language_ids:
            language_id_mask = (ret_language_ids['classes'] == language_id)
            if not np.any(language_id_mask):
                continue
            max_score = np.max(scores['scores'][language_id_mask])
            best_idx = np.where(scores['scores'] == max_score)[0]
            if scores['scores'][best_idx] < detection_threshold:
                continue
            # Save the detection.
            seg_bb = SegmentedBoundingBox(boxes[best_idx], masks[best_idx],
                                          scores[best_idx])
            object_id_to_img_detections[language_id][rgbd.camera_name] = seg_bb

    # Save all artifacts.
    artifacts = object_id_to_img_detections

    # Get the best-scoring image detections for each object ID.
    object_id_to_best_img_id: Dict[ObjectDetectionID, str] = {}
    for obj_id, img_detections in object_id_to_img_detections.items():
        if not img_detections:
            continue
        img_detections_lst = [img_detections[rgbd.camera_name] for rgbd in rgbds]
        best_score_idx = np.argmax([d.score for d in img_detections_lst])
        best_img_id = rgbds[best_score_idx].camera_name
        object_id_to_best_img_id[obj_id] = best_img_id

    # Convert the image detections into pose detections.
    import ipdb
    ipdb.set_trace()

    return detections, artifacts


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: make sure the spot hand camera sees the 408 april tag and a brush.
    # It is recommended to run this test a few times in a row while moving the
    # robot around, but keeping the object in place, to make sure that the
    # detections are consistent.

    # pylint: disable=ungrouped-imports
    from pathlib import Path

    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.spot_utils.perception.spot_cameras import capture_image
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import verify_estop

    TEST_CAMERA = "hand_color_image"
    TEST_APRIL_TAG_ID = 408
    # Assume the table is oriented such the tag is in the front with respect
    # to the world frame. In the 4th floor room, this is facing such that the
    # outside hall is on the left of the tag.
    TEST_APRIL_TAG_TRANSFORM = math_helpers.SE3Pose(0.0, 0.5, 0.0,
                                                    math_helpers.Quat())
    TEST_LANGUAGE_DESCRIPTION = "brush"

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
        world_tform_body = localizer.localize()
        rgbd = capture_image(robot, TEST_CAMERA)

        # Detect the april tag and brush.
        april_tag_id = AprilTagObjectDetectionID(TEST_APRIL_TAG_ID,
                                                 TEST_APRIL_TAG_TRANSFORM)
        language_id = LanguageObjectDetectionID(TEST_LANGUAGE_DESCRIPTION)
        detections, _ = detect_objects([april_tag_id, language_id], [rgbd],
                                       world_tform_body)
        for obj_id, detection in detections.items():
            print(f"Detected {obj_id} at {detection}")

    _run_manual_test()
