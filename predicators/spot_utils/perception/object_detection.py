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
from functools import partial
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Set, Tuple

try:
    import apriltag
    _APRILTAG_IMPORTED = True
except ModuleNotFoundError:
    _APRILTAG_IMPORTED = False
import cv2
import dill as pkl
import numpy as np
import PIL.Image
import requests
from bosdyn.client import math_helpers
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.spatial import Delaunay

from predicators.settings import CFG
from predicators.spot_utils.perception.object_specific_grasp_selection import \
    OBJECT_SPECIFIC_GRASP_SELECTORS
from predicators.spot_utils.perception.perception_structs import \
    AprilTagObjectDetectionID, KnownStaticObjectDetectionID, \
    LanguageObjectDetectionID, ObjectDetectionID, PythonicObjectDetectionID, \
    RGBDImageWithContext, RGBDImage, SegmentedBoundingBox
from predicators.spot_utils.utils import get_april_tag_transform, \
    get_graph_nav_dir
from predicators.utils import rotate_point_in_image

# Hack to avoid double image capturing when we want to (1) get object states
# and then (2) use the image again for pixel-based grasping.
_LAST_DETECTED_OBJECTS: Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose],
                              Dict[str, Any]] = ({}, {})


def get_last_detected_objects(
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Return the last output from detect_objects(), ignoring inputs."""
    return _LAST_DETECTED_OBJECTS


def detect_objects(
    object_ids: Collection[ObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],  # camera name to RGBD
    allowed_regions: Optional[Collection[Delaunay]] = None,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Detect object poses (in the world frame!) from RGBD.

    Each object ID is assumed to exist at most once in each image, but can
    exist in multiple images.

    The second return value is a collection of artifacts that can be useful
    for debugging / analysis.
    """
    global _LAST_DETECTED_OBJECTS  # pylint: disable=global-statement

    # Collect and dispatch.
    april_tag_object_ids: Set[AprilTagObjectDetectionID] = set()
    language_object_ids: Set[LanguageObjectDetectionID] = set()
    pythonic_object_ids: Set[PythonicObjectDetectionID] = set()
    known_static_object_ids: Set[KnownStaticObjectDetectionID] = set()
    for object_id in object_ids:
        if isinstance(object_id, AprilTagObjectDetectionID):
            april_tag_object_ids.add(object_id)
        elif isinstance(object_id, KnownStaticObjectDetectionID):
            known_static_object_ids.add(object_id)
        elif isinstance(object_id, PythonicObjectDetectionID):
            pythonic_object_ids.add(object_id)
        else:
            assert isinstance(object_id, LanguageObjectDetectionID)
            language_object_ids.add(object_id)
    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    artifacts: Dict[str, Any] = {"april": {}, "language": {}}

    # Read off known objects directly.
    for known_obj_id in known_static_object_ids:
        detections[known_obj_id] = known_obj_id.pose

    # There is no batching over images for april tag detection.
    for rgbd in rgbds.values():
        img_detections, img_artifacts = detect_objects_from_april_tags(
            april_tag_object_ids, rgbd)
        # Possibly overrides previous detections.
        detections.update(img_detections)
        artifacts["april"][rgbd.camera_name] = img_artifacts

    # There IS batching over images here for efficiency.
    language_detections, language_artifacts = detect_objects_from_language(
        language_object_ids, rgbds, allowed_regions)
    detections.update(language_detections)
    artifacts["language"] = language_artifacts

    # Handle pythonic object detection.
    for object_id in pythonic_object_ids:
        detection = object_id.fn(rgbds)
        if detection is not None:
            detections[object_id] = detection
            break

    _LAST_DETECTED_OBJECTS = (detections, artifacts)

    return detections, artifacts


def detect_objects_from_april_tags(
    object_ids: Collection[AprilTagObjectDetectionID],
    rgbd: RGBDImageWithContext,
    fiducial_size: float = CFG.spot_fiducial_size,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict]:
    """Detect an object pose from an april tag.

    The rotation is currently not detected (set to default).

    The second return value is a dictionary of "artifacts", which include
    the raw april tag detection results. These are primarily useful for
    debugging / analysis.
    """
    if not object_ids:
        return {}, {}

    if not _APRILTAG_IMPORTED:
        raise ModuleNotFoundError("Need to install 'apriltag' package")

    tag_num_to_object_id = {t.april_tag_number: t for t in object_ids}

    # Convert the RGB image to grayscale.
    image_grey = cv2.cvtColor(rgbd.rgb, cv2.COLOR_RGB2GRAY)

    # Create apriltag detector and get all apriltag locations.
    options = apriltag.DetectorOptions(families="tag36h11")
    options.refine_pose = 1
    detector = apriltag.Detector(options)
    apriltag_detections = detector.detect(image_grey)

    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    artifacts: Dict = {}

    # For every detection, find pose in world frame.
    for apriltag_detection in apriltag_detections:
        # Only include requested tags.
        if apriltag_detection.tag_id not in tag_num_to_object_id:
            continue
        obj_id = tag_num_to_object_id[apriltag_detection.tag_id]

        # Save the detection for external analysis.
        artifacts[obj_id] = apriltag_detection

        # Get the pose from the apriltag library.
        intrinsics = rgbd.camera_model.intrinsics
        pose = detector.detection_pose(
            apriltag_detection,
            (intrinsics.focal_length.x, intrinsics.focal_length.y,
             intrinsics.principal_point.x, intrinsics.principal_point.y),
            fiducial_size)[0]
        tx, ty, tz, tw = pose[:, -1]
        assert np.isclose(tw, 1.0)

        # Detection is in meters, we want mm.
        camera_tform_tag = math_helpers.SE3Pose(
            x=float(tx) / 1000.0,
            y=float(ty) / 1000.0,
            z=float(tz) / 1000.0,
            rot=math_helpers.Quat(),
        )

        # Look up transform.
        world_object_tform_tag = get_april_tag_transform(
            obj_id.april_tag_number)

        # Apply transforms.
        world_frame_pose = rgbd.world_tform_camera * camera_tform_tag
        world_frame_pose = world_object_tform_tag * world_frame_pose

        # Save in detections.
        detections[obj_id] = world_frame_pose

    return detections, artifacts


def detect_objects_from_language(
    object_ids: Collection[LanguageObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],
    allowed_regions: Optional[Collection[Delaunay]] = None,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict]:
    """Detect an object pose using a vision-language model.

    The second return value is a dictionary of "artifacts", which
    include the raw vision-language detection results. These are
    primarily useful for debugging / analysis. See
    visualize_all_artifacts().
    """

    object_id_to_img_detections = _query_detic_sam(object_ids, rgbds)

    # Convert the image detections into pose detections. Use the best scoring
    # image for which a pose can be successfully extracted.

    def _get_detection_score(img_detections: Dict[str, SegmentedBoundingBox],
                             camera: str) -> float:
        return img_detections[camera].score

    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    for obj_id, img_detections in object_id_to_img_detections.items():
        # Consider detections from best (highest) to worst score.
        for camera in sorted(img_detections,
                             key=partial(_get_detection_score, img_detections),
                             reverse=True):
            seg_bb = img_detections[camera]
            rgbd = rgbds[camera]
            pose = _get_pose_from_segmented_bounding_box(seg_bb, rgbd)
            # Pose extraction can fail due to depth reading issues. See
            # docstring of _get_pose_from_segmented_bounding_box for more.
            if pose is None:
                continue
            # If the detected pose is outside the allowed bounds, skip.
            pose_xy = np.array([pose.x, pose.y])
            if allowed_regions is not None:
                in_allowed_region = False
                for region in allowed_regions:
                    if region.find_simplex(pose_xy).item() >= 0:
                        in_allowed_region = True
                        break
                if not in_allowed_region:
                    logging.info("WARNING: throwing away detection for " +\
                                 f"{obj_id} because it's out of bounds. " + \
                                 f"(pose = {pose_xy})")
                    continue
            # Pose extraction succeeded.
            detections[obj_id] = pose
            break

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
        pil_rotated_img = PIL.Image.fromarray(rgbd.rotated_rgb)  # type: ignore
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

def _query_detic_sam2(
    object_ids: Collection[LanguageObjectDetectionID],
    rgbds: Dict[str, RGBDImage],
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
        pil_rotated_img = PIL.Image.fromarray(rgbd.rotated_rgb)  # type: ignore
        buf_dict[camera_name] = _image_to_bytes(pil_rotated_img)

    # Extract all the classes that we want to detect.
    classes = sorted(o.language_id for o in object_ids)

    # Query server, retrying to handle possible wifi issues.
    # import pdb; pdb.set_trace()
    # imgs = [v.rotated_rgb for _, v in rgbds.items()]
    # pil_img = PIL.Image.fromarray(imgs[0])
    # import pdb; pdb.set_trace()

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
            # h, w = rgbd.rgb.shape[:2]
            # image_rot = rgbd.image_rot
            # boxes = [
            #     _rotate_bounding_box(bb, -image_rot, h, w) for bb in rot_boxes
            # ]
            # masks = [
            #     ndimage.rotate(m.squeeze(), -image_rot, reshape=False)
            #     for m in rot_masks
            # ]
            boxes = rot_boxes
            masks = rot_masks

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

    import pdb; pdb.set_trace()
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
    fx = rgbd.camera_model.intrinsics.focal_length.x
    fy = rgbd.camera_model.intrinsics.focal_length.y
    cx = rgbd.camera_model.intrinsics.principal_point.x
    cy = rgbd.camera_model.intrinsics.principal_point.y
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

    # The angles are not meaningful, so override them.
    final_pose = math_helpers.SE3Pose(x=world_frame_pose.x,
                                      y=world_frame_pose.y,
                                      z=world_frame_pose.z,
                                      rot=math_helpers.Quat())

    return final_pose


def get_grasp_pixel(
    rgbds: Dict[str, RGBDImageWithContext], artifacts: Dict[str, Any],
    object_id: ObjectDetectionID, camera_name: str, rng: np.random.Generator
) -> Tuple[Tuple[int, int], Optional[math_helpers.Quat]]:
    """Select a pixel for grasping in the given camera image.

    NOTE: for april tag detections, the pixel returned will correspond to the
    center of the april tag, which may not always be ideal for grasping.
    Consider using OBJECT_SPECIFIC_GRASP_SELECTORS in this case.
    """

    if object_id in OBJECT_SPECIFIC_GRASP_SELECTORS:
        selector = OBJECT_SPECIFIC_GRASP_SELECTORS[object_id]
        return selector(rgbds, artifacts, camera_name, rng)

    pixel = get_random_mask_pixel_from_artifacts(artifacts, object_id,
                                                 camera_name, rng)
    return (pixel[0], pixel[1]), None


def get_random_mask_pixel_from_artifacts(
        artifacts: Dict[str, Any], object_id: ObjectDetectionID,
        camera_name: str, rng: np.random.Generator) -> Tuple[int, int]:
    """Extract the pixel in the image corresponding to the center of the object
    with object ID.

    The typical use case is to get the pixel to pass into the grasp
    controller. This is a fairly hacky way to go about this, but since
    the grasp controller parameterization is a special case (most users
    of object detection shouldn't need to care about the pixel), we do
    this.
    """
    if isinstance(object_id, AprilTagObjectDetectionID):
        try:
            april_detection = artifacts["april"][camera_name][object_id]
        except KeyError:
            raise ValueError(f"{object_id} not detected in {camera_name}")
        pr, pc = april_detection.center
        return int(pr), int(pc)

    assert isinstance(object_id, LanguageObjectDetectionID)
    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[object_id][camera_name]
    except KeyError:
        raise ValueError(f"{object_id} not detected in {camera_name}")

    # Select a random valid pixel from the mask.
    mask = seg_bb.mask
    pixels_in_mask = np.where(mask)
    mask_idx = rng.choice(len(pixels_in_mask))
    pixel_tuple = (pixels_in_mask[1][mask_idx], pixels_in_mask[0][mask_idx])
    # Uncomment to plot the grasp pixel being selected!
    # rgb_img = artifacts["language"]["rgbds"][camera_name].rgb
    # _, axes = plt.subplots()
    # axes.imshow(rgb_img)
    # axes.add_patch(
    #     plt.Rectangle((pixel_tuple[0], pixel_tuple[1]), 5, 5, color='red'))
    # plt.tight_layout()
    # outdir = Path(CFG.spot_perception_outdir)
    # plt.savefig(outdir / "grasp_pixel.png", dpi=300)
    # plt.close()
    return pixel_tuple


def visualize_all_artifacts(artifacts: Dict[str,
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
    fig_scale = 2
    if flat_detections:
        _, axes = plt.subplots(len(flat_detections),
                               5,
                               squeeze=False,
                               figsize=(5 * fig_scale,
                                        len(flat_detections) * fig_scale))
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
            # ax_row[3].imshow(rgbd.rgb)
            ax_row[3].imshow(rgbd.rotated_rgb)
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

            # ax_row[4].imshow(seg_bb.mask, cmap="binary_r", vmin=0, vmax=1)

            # Labels.
            abbreviated_name = obj_id.language_id
            max_abbrev_len = 24
            if len(abbreviated_name) > max_abbrev_len:
                abbreviated_name = abbreviated_name[:max_abbrev_len] + "..."
            row_label = "\n".join([
                abbreviated_name, f"[{rgbd.camera_name}]",
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
        plt.close()

    # Visualize all of the images that have no detections.
    all_cameras = set(rgbds)
    cameras_with_detections = {r.camera_name for r, _, _ in flat_detections}
    cameras_without_detections = sorted(all_cameras - cameras_with_detections)

    if cameras_without_detections:
        _, axes = plt.subplots(len(cameras_without_detections),
                               3,
                               squeeze=False,
                               figsize=(3 * fig_scale,
                                        len(cameras_without_detections) *
                                        fig_scale))
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
        plt.close()


def display_camera_detections(artifacts: Dict[str, Any],
                              axes: plt.Axes) -> None:
    """Plot per-camera detections on the given axes.

    The axes are given as input because we might want to update the same
    axes repeatedly, e.g., during object search.
    """

    # At the moment, only language detection artifacts are visualized.
    rgbds = artifacts["language"]["rgbds"]
    detections = artifacts["language"]["object_id_to_img_detections"]
    # Organize detections by camera.
    camera_to_rgbd = {rgbd.camera_name: rgbd for rgbd in rgbds.values()}
    camera_to_detections: Dict[str, List[Tuple[LanguageObjectDetectionID,
                                               SegmentedBoundingBox]]] = {
                                                   c: []
                                                   for c in camera_to_rgbd
                                               }
    for obj_id, img_detections in detections.items():
        for camera, seg_bb in img_detections.items():
            camera_to_detections[camera].append((obj_id, seg_bb))

    # Plot per-camera.
    box_colors = ["green", "red", "blue", "purple", "gold", "brown", "black"]
    camera_order = sorted(camera_to_rgbd)
    for ax, camera in zip(axes.flat, camera_order):
        ax.clear()
        ax.set_title(camera)
        ax.set_xticks([])
        ax.set_yticks([])

        # Display the RGB image.
        rgbd = camera_to_rgbd[camera]
        ax.imshow(rgbd.rotated_rgb)

        for i, (obj_id, seg_bb) in enumerate(camera_to_detections[camera]):

            color = box_colors[i % len(box_colors)]

            # Display the bounding box.
            box = seg_bb.bounding_box
            # Rotate.
            image_rot = rgbd.image_rot
            h, w = rgbd.rgb.shape[:2]
            box = _rotate_bounding_box(box, image_rot, h, w)
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(
                plt.Rectangle((x0, y0),
                              w,
                              h,
                              edgecolor=color,
                              facecolor=(0, 0, 0, 0),
                              lw=1))
            # Label with the detection and score.
            ax.text(
                -250,  # off to the left side
                50 + 60 * i,
                f'{obj_id.language_id}: {seg_bb.score:.2f}',
                color='white',
                fontsize=12,
                fontweight='bold',
                bbox=dict(facecolor=color, edgecolor=color, alpha=0.5))


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
    from predicators.spot_utils.perception.cv2_utils import \
        find_color_based_centroid
    from predicators.spot_utils.perception.spot_cameras import capture_images
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import verify_estop

    TEST_CAMERAS = [
        "hand_color_image",
        "frontleft_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "frontright_fisheye_image",
    ]
    TEST_APRIL_TAG_ID = 408
    TEST_LANGUAGE_DESCRIPTIONS = [
        "small basketball toy/stuffed toy basketball/small orange ball",
        "small football toy/stuffed toy football/small brown ball",
    ]

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()

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
            TEST_APRIL_TAG_ID)
        language_ids: List[ObjectDetectionID] = [
            LanguageObjectDetectionID(d) for d in TEST_LANGUAGE_DESCRIPTIONS
        ]
        known_static_id: ObjectDetectionID = KnownStaticObjectDetectionID(
            "imaginary_box",
            math_helpers.SE3Pose(-5, 0, 0, rot=math_helpers.Quat()))
        object_ids: List[ObjectDetectionID] = [april_tag_id, known_static_id
                                               ] + language_ids
        detections, artifacts = detect_objects(object_ids, rgbds)
        for obj_id, detection in detections.items():
            print(f"Detected {obj_id} at {detection}")

        # Visualize the artifacts.
        detections_outfile = Path(".") / "object_detection_artifacts.png"
        no_detections_outfile = Path(".") / "no_detection_artifacts.png"
        visualize_all_artifacts(artifacts, detections_outfile,
                                no_detections_outfile)

    def _run_pythonic_bowl_test() -> None:
        # Test for using an arbitrary python function to detect objects,
        # which in this case uses a combination of vision-language and
        # colored-based detection to find a bowl that has blue tape on the
        # bottom. The tape is used to crudely orient the bowl. Like the
        # previous test, this one assumes that the bowl is within view.
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()

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
        rgbds = capture_images(robot, localizer)

        def _detect_bowl(
            rgbds: Dict[str, RGBDImageWithContext]
        ) -> Optional[math_helpers.SE3Pose]:
            # ONLY use the hand camera (which we assume is looking down)
            # because otherwise it's impossible to see the top/bottom.
            hand_camera = "hand_color_image"
            assert hand_camera in rgbds
            rgbds = {hand_camera: rgbds[hand_camera]}
            # Start by using vision-language.
            language_id = LanguageObjectDetectionID("large cup")
            detections, artifacts = detect_objects([language_id], rgbds)
            if not detections:
                return None
            # Crop using the bounding box. If there were multiple detections,
            # choose the highest scoring one.
            obj_id_to_img_detections = artifacts["language"][
                "object_id_to_img_detections"]
            img_detections = obj_id_to_img_detections[language_id]
            assert len(img_detections) > 0
            best_seg_bb: Optional[SegmentedBoundingBox] = None
            best_seg_bb_score = -np.inf
            best_camera: Optional[str] = None
            for camera, seg_bb in img_detections.items():
                if seg_bb.score > best_seg_bb_score:
                    best_seg_bb_score = seg_bb.score
                    best_seg_bb = seg_bb
                    best_camera = camera
            assert best_camera is not None
            assert best_seg_bb is not None
            x1, y1, x2, y2 = best_seg_bb.bounding_box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            best_rgb = rgbds[best_camera].rgb
            height, width = best_rgb.shape[:2]
            r_min = min(max(int(y_min), 0), height)
            r_max = min(max(int(y_max), 0), height)
            c_min = min(max(int(x_min), 0), width)
            c_max = min(max(int(x_max), 0), width)
            cropped_img = best_rgb[r_min:r_max, c_min:c_max]
            # Look for the blue tape inside the bounding box.
            lo, hi = ((0, 130, 130), (130, 255, 255))
            centroid = find_color_based_centroid(cropped_img, lo, hi)
            blue_tape_found = (centroid is not None)
            # If the blue tape was found, assume that the bowl is oriented
            # upside-down; otherwise, it's right-side up.
            if blue_tape_found:
                roll = np.pi
                print("Detected blue tape; bowl is upside-down!")
            else:
                roll = 0.0
                print("Did NOT detect blue tape; bowl is right side-up!")
            rot = math_helpers.Quat.from_roll(roll)
            # Use the x, y, z from vision-language.
            vision_language_pose = detections[language_id]
            pose = math_helpers.SE3Pose(x=vision_language_pose.x,
                                        y=vision_language_pose.y,
                                        z=vision_language_pose.z,
                                        rot=rot)
            return pose

        bowl_id = PythonicObjectDetectionID("bowl", _detect_bowl)
        detections, artifacts = detect_objects([bowl_id], rgbds)
        for obj_id, detection in detections.items():
            print(f"Detected {obj_id} at {detection}")

        # Visualize the artifacts.
        detections_outfile = Path(".") / "object_detection_artifacts.png"
        no_detections_outfile = Path(".") / "no_detection_artifacts.png"
        visualize_all_artifacts(artifacts, detections_outfile,
                                no_detections_outfile)

    _run_manual_test()
    _run_pythonic_bowl_test()
