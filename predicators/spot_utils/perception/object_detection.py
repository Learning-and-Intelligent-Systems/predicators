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

from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

import apriltag
import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, \
    GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME, \
    get_a_tform_b, get_se2_a_tform_b, get_vision_tform_body
from numpy.typing import NDArray

from predicators.settings import CFG


@dataclass(frozen=True)
class ObjectDetectionID:
    """A unique identifier for an object that is to be detected."""


@dataclass(frozen=True)
class AprilTagObjectDetectionID(ObjectDetectionID):
    """An ID for an object to be detected from an april tag.

    The object center is defined to be the center of the april tag plus
    offset.
    """
    april_tag_number: int
    offset_transform: math_helpers.SE3Pose


@dataclass(frozen=True)
class LanguageObjectDetectionID(ObjectDetectionID):
    """An ID for an object to be detected with a vision-language model."""
    language_id: str


def detect_objects(
    object_ids: Collection[ObjectDetectionID],
    rgb_images: List[NDArray[np.uint8]],
    rgb_image_responses: List[image_pb2.ImageResponse],
    depth_images: List[NDArray[np.uint16]],
    depth_image_responses: List[image_pb2.ImageResponse],
    world_tform_body: math_helpers.SE3Pose,
    image_ids: Optional[List[str]] = None,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Detect object poses (in the world frame!) from RGBD.

    Each object ID is assumed to exist at most once in each image, but can
    exist in multiple images.

    The second return value is a collection of artifacts that can be useful
    for debugging / analysis.
    """
    assert len(rgb_images) == len(rgb_image_responses) == \
        len(depth_images) == len(depth_image_responses)
    if image_ids is None:
        image_ids = [f"img{i}" for i in range(len(rgb_images))]
    assert len(image_ids) == len(rgb_images)

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
    for img, resp, img_id in zip(rgb_images, rgb_image_responses, image_ids):
        img_detections, img_artifacts = detect_objects_from_april_tags(
            april_tag_object_ids, img, resp, img_id, world_tform_body)
        detections.update(img_detections)
        artifacts.update(img_artifacts)
    # There IS batching over images here for efficiency.
    language_detections, language_artifacts = detect_objects_from_language(
        language_object_ids, rgb_images, rgb_image_responses, depth_images,
        depth_image_responses, image_ids, world_tform_body)
    detections.update(language_detections)
    artifacts.update(language_artifacts)
    return detections, artifacts


def detect_objects_from_april_tags(
    object_ids: Collection[AprilTagObjectDetectionID],
    rgb_image: NDArray[np.uint8],
    rgb_image_response: image_pb2.ImageResponse,
    image_id: str,
    world_tform_body: math_helpers.SE3Pose,
    fiducial_size: float = CFG.spot_fiducial_size,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Detect an object pose from an april tag.

    The rotation is currently not detected (set to default).
    """
    tag_num_to_object_id = {t.april_tag_number: t for t in object_ids}

    # Camera body transform.
    camera_tform_body = get_a_tform_b(
        rgb_image_response.shot.transforms_snapshot,
        rgb_image_response.shot.frame_name_image_sensor, BODY_FRAME_NAME)

    # Camera intrinsics for the given source camera.
    intrinsics = rgb_image_response.source.pinhole.intrinsics

    # Convert the image to grayscale.
    image_grey = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

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
        object_id = tag_num_to_object_id[apriltag_detection.tag_id]

        # Save the detection for external analysis.
        artifact_id = f"apriltag_{image_id}_{object_id.april_tag_number}"
        artifacts[artifact_id] = apriltag_detection

        # Get the pose from the apriltag library.
        pose = detector.detection_pose(
            apriltag_detection,
            (intrinsics.focal_length.x, intrinsics.focal_length.y,
             intrinsics.principal_point.x, intrinsics.principal_point.y),
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
        body_tform_camera = camera_tform_body.inverse()
        body_frame_pose = body_tform_camera * camera_frame_pose
        world_frame_pose = world_tform_body * body_frame_pose
        world_frame_pose = object_id.offset_transform * world_frame_pose

        # Save in detections.
        detections[object_id] = world_frame_pose

    return detections, artifacts


def detect_objects_from_language(
    object_ids: Collection[LanguageObjectDetectionID],
    rgb_images: List[NDArray[np.uint8]],
    rgb_image_responses: List[image_pb2.ImageResponse],
    depth_images: List[NDArray[np.uint16]],
    depth_image_responses: List[image_pb2.ImageResponse],
    image_ids: List[str],
    world_tform_body: math_helpers.SE3Pose,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Detect an object pose using a vision-language model."""
    # TODO
    return {}, {}


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
    from predicators.spot_utils.perception.spot_cameras import \
        RGB_TO_DEPTH_CAMERAS, get_image_response, image_response_to_image
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import verify_estop

    # TODO add brush

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

        rgb_camera = TEST_CAMERA
        rgb_response = get_image_response(robot, rgb_camera)
        rgb_image = image_response_to_image(rgb_response)
        depth_camera = RGB_TO_DEPTH_CAMERAS[rgb_camera]
        depth_response = get_image_response(robot, depth_camera)
        depth_image = image_response_to_image(depth_response)

        # Detect the april tag.
        # TODO add language to the same call.
        april_tag_id = AprilTagObjectDetectionID(TEST_APRIL_TAG_ID,
                                                 TEST_APRIL_TAG_TRANSFORM)
        detections, _ = detect_objects([april_tag_id], [rgb_image],
                                       [rgb_response], [depth_image],
                                       [depth_response], world_tform_body)
        detection = detections[april_tag_id]
        print(f"Detected tag {april_tag_id.april_tag_number} at {detection}")

    _run_manual_test()
