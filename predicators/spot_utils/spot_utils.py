"""Utility functions to interface with the Boston Dynamics Spot robot."""

import functools
import logging
import os
import sys
import time
from collections import OrderedDict
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple

import apriltag
import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.api import arm_command_pb2, basic_command_pb2, estop_pb2, \
    geometry_pb2, image_pb2, manipulation_api_pb2, robot_command_pb2, \
    synchronized_command_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, \
    GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME, \
    get_a_tform_b, get_se2_a_tform_b, get_vision_tform_body
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, \
    RobotCommandClient, block_until_arm_arrives, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot
from gym.spaces import Box

from predicators import utils
from predicators.settings import CFG
from predicators.spot_utils.helpers.graph_nav_command_line import \
    GraphNavInterface
from predicators.spot_utils.perception_utils import CAMERA_NAMES, \
    RGB_TO_DEPTH_CAMERAS, get_object_locations_with_detic_sam, \
    get_pixel_locations_with_detic_sam
from predicators.structs import Array, Image, Object

ARM_6DOF_NAMES = [
    "arm0.sh0",
    "arm0.sh1",
    "arm0.el0",
    "arm0.el1",
    "arm0.wr0",
    "arm0.wr1",
]

g_image_click = None
g_image_display = None


def get_memorized_waypoint(obj_name: str) -> Optional[Tuple[str, Array]]:
    """Returns None if the location of the object is unknown.

    Returns a waypoint ID (str) and a (x, y, yaw) offset otherwise.
    """
    graph_nav_loc_to_id = {
        "front_tool_room": "dented-marlin-HZHTzO56529oo0oGfAFHdg==",
        "tool_room_table": "lumpen-squid-p9fT8Ui8TYI7JWQJvfQwKw==",
        "bucket": "seared-hare-0JBmyRiYHfbxn58ymEwPaQ==",
        "tool_room_tool_stand": "roving-gibbon-3eduef4VV0itZzkpHZueNQ==",
        "tool_room_platform": "comfy-auk-W0iygJ1WJyKR1eB3qe2mlg==",
        "low_wall_rack": "alight-coyote-Nvl0i02Mk7Ds8ax0sj0Hsw==",
        "high_wall_rack": "alight-coyote-Nvl0i02Mk7Ds8ax0sj0Hsw==",
        "extra_room_table": "alight-coyote-Nvl0i02Mk7Ds8ax0sj0Hsw==",
    }
    offsets = {"extra_room_table": np.array([0.0, -0.3, np.pi / 2])}
    if obj_name not in graph_nav_loc_to_id:
        return None
    waypoint_id = graph_nav_loc_to_id[obj_name]
    offset = offsets.get(obj_name, np.zeros(3, dtype=np.float32))
    return (waypoint_id, offset)


obj_name_to_apriltag_id = {
    "hammer": 401,
    "brush": 402,
    "measuring_tape": 403,
    "bucket": 405,
    "low_wall_rack": 406,
    "front_tool_room": 407,
    "tool_room_table": 408,
    "extra_room_table": 409,
    "cube": 410,
    "platform": 411,
    "high_wall_rack": 412,
}
obj_name_to_vision_prompt = {
    "hammer": "hammer",
    "brush": "brush",
    "measuring_tape": "measuring tape",
    "platform": "red t-shaped dolly handle",
    "bucket": "bucket",
}
vision_prompt_to_obj_name = {
    value: key
    for key, value in obj_name_to_vision_prompt.items()
}

OBJECT_CROPS = {
    # min_x, max_x, min_y, max_y
    "hammer": (160, 450, 160, 350),
    "brush": (100, 400, 350, 480),
}

OBJECT_COLOR_BOUNDS = {
    # (min B, min G, min R), (max B, max G, max R)
    "hammer": ((0, 0, 50), (40, 40, 200)),
    "brush": ((0, 100, 200), (80, 255, 255)),
}

OBJECT_GRASP_OFFSET = {
    # dx, dy
    "hammer": (0, 0),
    "brush": (0, 0),
}

COMMAND_TIMEOUT = 20.0


def _find_object_center(img: Image,
                        obj_name: str) -> Optional[Tuple[int, int]]:
    # Copy to make sure we don't modify the image.
    img = img.copy()

    # Crop
    crop_min_x, crop_max_x, crop_min_y, crop_max_y = OBJECT_CROPS[obj_name]
    cropped_img = img[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

    # Mask color.
    lo, hi = OBJECT_COLOR_BOUNDS[obj_name]
    lower = np.array(lo)
    upper = np.array(hi)
    mask = cv2.inRange(cropped_img, lower, upper)

    # Apply blur.
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Connected components with stats.
    nb_components, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=4)

    # Fail if nothing found.
    if nb_components <= 1:
        return None

    # Find the largest non background component.
    # NOTE: range() starts from 1 since 0 is the background label.
    max_label, _ = max(
        ((i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)),
        key=lambda x: x[1])

    cropped_x, cropped_y = map(int, centroids[max_label])

    x = cropped_x + crop_min_x
    y = cropped_y + crop_min_y

    # Apply offset.
    dx, dy = OBJECT_GRASP_OFFSET[obj_name]
    x = np.clip(x + dx, 0, img.shape[1])
    y = np.clip(y + dy, 0, img.shape[0])

    return (x, y)


# pylint: disable=no-member
class _SpotInterface():
    """Implementation of interface with low-level controllers and sensor data
    grabbing for the Spot robot."""

    def __init__(self) -> None:
        self._hostname = CFG.spot_robot_ip
        self._verbose = False
        self._force_45_angle_grasp = False
        self._force_horizontal_grasp = False
        self._force_squeeze_grasp = False
        self._force_top_down_grasp = False
        self._force_forward_grasp = False
        self._grasp_image_source = "hand_color_image"

        self.hand_x, self.hand_y, self.hand_z = (0.80, 0, 0.45)
        self.hand_x_bounds = (0.3, 0.9)
        self.hand_y_bounds = (-0.5, 0.5)
        self.hand_z_bounds = (0.09, 0.7)
        self.localization_timeout = 10

        self._find_controller_move_queue_idx = 0

        # Try to connect to the robot. If this fails, still maintain the
        # instance for testing, but assert that it succeeded within the
        # controller calls.
        self._connected_to_spot = False
        try:
            self._connect_to_spot()
            self._connected_to_spot = True
        except (bosdyn.client.exceptions.ProxyConnectionError,
                bosdyn.client.exceptions.UnableToConnectToRobotError,
                RuntimeError):
            logging.warning("Could not connect to Spot!")

    def _connect_to_spot(self) -> None:
        # See hello_spot.py for an explanation of these lines.
        bosdyn.client.util.setup_logging(self._verbose)

        self.sdk = bosdyn.client.create_standard_sdk('SesameClient')
        self.robot: Robot = self.sdk.create_robot(self._hostname)
        if not os.environ.get('BOSDYN_CLIENT_USERNAME') or not os.environ.get(
                'BOSDYN_CLIENT_PASSWORD'):
            raise RuntimeError("Spot environment variables unset.")
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()

        assert self.robot.has_arm(
        ), "Robot requires an arm to run this example."

        # Verify the robot is not estopped and that an external application has
        # registered and holds an estop endpoint.
        self.verify_estop(self.robot)

        self.lease_client = self.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name)
        self.robot_state_client: RobotStateClient = self.robot.ensure_client(
            RobotStateClient.default_service_name)
        self.robot_command_client: RobotCommandClient = \
            self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.image_client = self.robot.ensure_client(
            ImageClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(
            ManipulationApiClient.default_service_name)
        self.lease_client.take()
        self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(
            self.lease_client, must_acquire=True, return_at_exit=True)

        # Create Graph Nav Command Line
        self.upload_filepath = "predicators/spot_utils/bike_env/" + \
            "downloaded_graph/"
        self.graph_nav_command_line = GraphNavInterface(
            self.robot, self.upload_filepath, self.lease_client,
            self.lease_keepalive)

        # Initializing Spot
        self.robot.logger.info(
            "Powering on robot... This may take a several seconds.")
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), "Robot power on failed."

        self.robot.logger.info("Commanding robot to stand...")
        blocking_stand(self.robot_command_client, timeout_sec=10)
        self.robot.logger.info("Robot standing.")

    def get_camera_images(
        self
    ) -> Tuple[Dict[str, Image], Dict[str, bosdyn.api.image_pb2.ImageResponse],
               Dict[str, Image], Dict[str,
                                      bosdyn.api.image_pb2.ImageResponse]]:
        """Get rgb and depth camera images + responses from all of spot's
        cameras.

        Returns rgb images, rgb image responses, depth images, depth
        image responses.
        """
        rgb_imgs: Dict[str, Image] = {}
        rgb_img_responses: Dict[str, bosdyn.api.image_pb2.ImageResponse] = {}
        depth_imgs: Dict[str, Image] = {}
        depth_img_responses: Dict[str, bosdyn.api.image_pb2.ImageResponse] = {}
        for source_name in CAMERA_NAMES:
            rgb_img, rgb_img_response = self.get_single_camera_image(
                source_name, to_rgb=True)
            depth_img, depth_img_response = self.get_single_camera_image(
                RGB_TO_DEPTH_CAMERAS[source_name], to_rgb=True)
            rgb_imgs[source_name] = rgb_img
            rgb_img_responses[source_name] = rgb_img_response
            depth_imgs[source_name] = depth_img
            depth_img_responses[source_name] = depth_img_response
        return (rgb_imgs, rgb_img_responses, depth_imgs, depth_img_responses)

    def get_single_camera_image(
        self,
        source_name: str,
        to_rgb: bool = False
    ) -> Tuple[Image, bosdyn.api.image_pb2.ImageResponse]:
        """Get a single source camera image and image response."""
        img_req = build_image_request(
            source_name,
            quality_percent=100,
            pixel_format=(None if
                          ('hand' in source_name or 'depth' in source_name)
                          else image_pb2.Image.PIXEL_FORMAT_RGB_U8))
        image_response = self.image_client.get_image([img_req])

        # Format image before detecting apriltags.
        if image_response[0].shot.image.pixel_format == image_pb2.Image. \
                PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16  # type: ignore
        else:
            dtype = np.uint8  # type: ignore
        img = np.fromstring(image_response[0].shot.image.data,
                            dtype=dtype)  # type: ignore
        if image_response[0].shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image_response[0].shot.image.rows,
                              image_response[0].shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        # Convert to RGB color, as some perception models assume RGB format
        # By default, still use BGR to keep backward compability
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return (img, image_response[0])

    def get_objects_in_view_by_camera(
        self, from_apriltag: bool, rgb_image_dict: Dict[str, Image],
        depth_image_dict: Dict[str, Image],
        rgb_image_response_dict: Dict[str, bosdyn.api.image_pb2.ImageResponse],
        depth_image_response_dict: Dict[str,
                                        bosdyn.api.image_pb2.ImageResponse]
    ) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        """Get objects currently in view for each camera."""
        tag_to_pose: Dict[str, Dict[int, Tuple[float, float, float]]] = {
            source_name: {}
            for source_name in CAMERA_NAMES
        }
        if from_apriltag:
            tag_to_pose = self.get_apriltag_poses_from_imgs(
                rgb_image_dict, rgb_image_response_dict)
        else:
            # First, get a dictionary mapping vision prompts
            # to the corresponding location of that object in the
            # scene by camera
            detic_sam_pose_results = self.get_deticsam_pose_from_imgs(
                classes=list(obj_name_to_vision_prompt.values()),
                rgb_image_dict=rgb_image_dict,
                depth_image_dict=depth_image_dict,
                rgb_image_response_dict=rgb_image_response_dict,
                depth_image_response_dict=depth_image_response_dict)
            # Next, convert the keys of this dictionary to be april
            # tag id's instead.
            for source_name, obj_pose_dict in detic_sam_pose_results.items():
                viewable_obj_poses: Dict[int, Tuple[float, float, float]] = {}
                for k, v in obj_pose_dict.items():
                    viewable_obj_poses[obj_name_to_apriltag_id[
                        vision_prompt_to_obj_name[k]]] = v
                tag_to_pose[source_name].update(viewable_obj_poses)

        apriltag_id_to_obj_name = {
            v: k
            for k, v in obj_name_to_apriltag_id.items()
        }
        camera_to_obj_names_to_poses: Dict[str, Dict[str, Tuple[float, float,
                                                                float]]] = {}
        for source_name in tag_to_pose.keys():
            camera_to_obj_names_to_poses[source_name] = {
                apriltag_id_to_obj_name[t]: p
                for t, p in tag_to_pose[source_name].items()
            }
        return camera_to_obj_names_to_poses

    def get_robot_pose(self) -> Tuple[float, float, float, float]:
        """Get the x, y, z position of the robot body."""
        state = self.get_localized_state()
        gn_origin_tform_body = math_helpers.SE3Pose.from_obj(
            state.localization.seed_tform_body)
        x, y, z = gn_origin_tform_body.transform_point(0.0, 0.0, 0.0)
        yaw = gn_origin_tform_body.rotation.to_yaw()
        return (x, y, z, yaw)

    def actively_construct_initial_object_views(
            self,
            object_names: Set[str]) -> Dict[str, Tuple[float, float, float]]:
        """Walk around and build object views."""
        object_views: Dict[str, Tuple[float, float, float]] = {}
        if CFG.spot_initialize_surfaces_to_default:
            object_views = {
                "tool_room_table": (6.63041, -6.35143, 0.179613),
                "extra_room_table": (8.27387, -6.23233, -0.0678132),
                "low_wall_rack":
                (10.049931203338616, -6.9443170697742, 0.27881268568327966),
                "high_wall_rack":
                (10.049931203338616, -6.9443170697742, 0.757881268568327966),
                "bucket":
                (7.043112552148553, -8.198686802340527, -0.18750694527153725),
                "platform": (8.79312, -7.8821, -0.100635)
            }
        waypoints = ["tool_room_table", "low_wall_rack"]
        objects_to_find = object_names - set(object_views.keys())
        obj_name_to_loc = self._scan_for_objects(waypoints, objects_to_find)
        for obj_name in objects_to_find:
            assert obj_name in obj_name_to_loc, \
                f"Did not locate object {obj_name}!"
            object_views[obj_name] = obj_name_to_loc[obj_name]
            logging.info(f"Located object {obj_name}")
        return object_views

    def get_localized_state(self) -> Any:
        """Get localized state from GraphNav client."""
        exec_start, exec_sec = time.perf_counter(), 0.0
        # This needs to be a while loop because get_localization_state
        # sometimes returns null pose if poorly localized. We assert JIC.
        while exec_sec < self.localization_timeout:
            # Localizes robot from larger graph fiducials.
            self.graph_nav_command_line.set_initial_localization_fiducial()
            state = self.graph_nav_command_line.graph_nav_client.\
                get_localization_state()
            exec_sec = time.perf_counter() - exec_start
            if str(state.localization.seed_tform_body) != '':
                break
        if str(state.localization.seed_tform_body) == '':
            logging.warning("WARNING: Localization timed out!")
        return state

    def get_apriltag_poses_from_imgs(
        self,
        rgb_image_dict: Dict[str, Image],
        rgb_image_response_dict: Dict[str, bosdyn.api.image_pb2.ImageResponse],
        fiducial_size: float = CFG.spot_fiducial_size,
    ) -> Dict[str, Dict[int, Tuple[float, float, float]]]:
        """Get the poses of all fiducials in camera view.

        The fiducial size has to be correctly defined in arguments (in mm).
        Also, it only works for tags that start with "40" in their ID.

        Returns a dict mapping the integer of the tag id to an (x, y, z)
        position tuple in the map frame.
        """
        tag_to_pose: Dict[str, Dict[int, Tuple[float, float, float]]] = {
            source_name: {}
            for source_name in CAMERA_NAMES
        }
        for source_name in CAMERA_NAMES:
            assert source_name in rgb_image_dict
            assert source_name in rgb_image_response_dict
            rgb_image = rgb_image_dict[source_name]
            rgb_image_response = rgb_image_response_dict[source_name]

            # Camera body transform.
            camera_tform_body = get_a_tform_b(
                rgb_image_response.shot.transforms_snapshot,
                rgb_image_response.shot.frame_name_image_sensor,
                BODY_FRAME_NAME)

            # Camera intrinsics for the given source camera.
            intrinsics = rgb_image_response.source.pinhole.intrinsics

            # Convert the image to grayscale.
            image_grey = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

            # Create apriltag detector and get all apriltag locations.
            options = apriltag.DetectorOptions(families="tag36h11")
            options.refine_pose = 1
            detector = apriltag.Detector(options)
            detections = detector.detect(image_grey)
            obj_poses: Dict[int, Tuple[float, float, float]] = {}
            # For every detection find location in graph_nav frame.
            for detection in detections:
                pose = detector.detection_pose(
                    detection,
                    (intrinsics.focal_length.x, intrinsics.focal_length.y,
                     intrinsics.principal_point.x,
                     intrinsics.principal_point.y), fiducial_size)[0]
                tx, ty, tz, tw = pose[:, -1]
                assert np.isclose(tw, 1.0)
                fiducial_rt_camera_frame = np.array([
                    float(tx) / 1000.0,
                    float(ty) / 1000.0,
                    float(tz) / 1000.0
                ])

                body_tform_fiducial = (
                    camera_tform_body.inverse()).transform_point(
                        fiducial_rt_camera_frame[0],
                        fiducial_rt_camera_frame[1],
                        fiducial_rt_camera_frame[2])

                # Get graph_nav to body frame.
                state = self.get_localized_state()
                gn_origin_tform_body = math_helpers.SE3Pose.from_obj(
                    state.localization.seed_tform_body)

                # Apply transform to fiducial to body location
                fiducial_rt_gn_origin = gn_origin_tform_body.transform_point(
                    body_tform_fiducial[0], body_tform_fiducial[1],
                    body_tform_fiducial[2])

                # This only works for small fiducials because of initial size.
                if detection.tag_id in obj_name_to_apriltag_id.values():
                    obj_poses[detection.tag_id] = fiducial_rt_gn_origin

            tag_to_pose[source_name].update(obj_poses)

        return tag_to_pose

    def get_deticsam_pose_from_imgs(
        self,
        classes: List[str],
        rgb_image_dict: Dict[str, Image],
        depth_image_dict: Dict[str, Image],
        rgb_image_response_dict: Dict[str, bosdyn.api.image_pb2.ImageResponse],
        depth_image_response_dict: Dict[str,
                                        bosdyn.api.image_pb2.ImageResponse],
    ) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        """Get object location in 3D (no orientation) estimated using
        pretrained DETIC-SAM model."""
        res_location_dict = get_object_locations_with_detic_sam(
            classes=classes,
            rgb_image_dict=rgb_image_dict,
            depth_image_dict=depth_image_dict,
            depth_image_response_dict=depth_image_response_dict,
            plot=CFG.spot_visualize_vision_model_outputs)

        transformed_location_dict: Dict[str, Dict[str, Tuple[
            float, float,
            float]]] = {source_name: {}
                        for source_name in CAMERA_NAMES}
        for source_name in CAMERA_NAMES:
            assert source_name in res_location_dict
            for obj_class in classes:
                if obj_class in res_location_dict[source_name]:
                    camera_tform_body = get_a_tform_b(
                        rgb_image_response_dict[source_name].shot.
                        transforms_snapshot,
                        rgb_image_response_dict[source_name].shot.
                        frame_name_image_sensor, BODY_FRAME_NAME)
                    x, y, z = res_location_dict[source_name][obj_class]
                    object_rt_gn_origin = self.convert_obj_location(
                        camera_tform_body, x, y, z)
                    transformed_location_dict[source_name][
                        obj_class] = object_rt_gn_origin

        # Use the input class name as the identifier for object(s) and
        # their positions.
        return transformed_location_dict

    def convert_obj_location(
            self, camera_tform_body: bosdyn.client.math_helpers.SE3Pose,
            x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Given an x, y, z position in the camera frame, transform it into the
        map frame."""
        body_tform_object = (camera_tform_body.inverse()).transform_point(
            x, y, z)
        # Get graph_nav to body frame.
        state = self.get_localized_state()
        gn_origin_tform_body = math_helpers.SE3Pose.from_obj(
            state.localization.seed_tform_body)
        # Apply transform to object to body location
        gn_origin_tform_object = gn_origin_tform_body.transform_point(
            body_tform_object[0], body_tform_object[1], body_tform_object[2])
        return gn_origin_tform_object

    def get_gripper_obs(self) -> float:
        """Grabs the current observation of relevant quantities from the
        gripper."""
        robot_state = self.robot_state_client.get_robot_state()
        return float(robot_state.manipulator_state.gripper_open_percentage)

    @property
    def params_spaces(self) -> Dict[str, Box]:
        """The parameter spaces for each of the controllers."""
        return {
            "navigate": Box(-5.0, 5.0, (3, )),
            "grasp": Box(-1.0, 2.0, (4, )),
            "grasp_from_platform": Box(-1.0, 2.0, (4, )),
            "placeOnTop": Box(-5.0, 5.0, (3, )),
            "drag": Box(-12.0, 12.0, (2, )),
            "noop": Box(0, 1, (0, ))
        }

    def execute(self, name: str, objects: Sequence[Object],
                params: Array) -> None:
        """Run the controller based on the given name."""
        assert self._connected_to_spot
        if name == "find":
            self._find_controller_move_queue_idx += 1
            return self.findController()
        # Just finished finding.
        self._find_controller_move_queue_idx = 0
        if name == "stow":
            return self.stow_arm()
        if name == "navigate":
            return self.navigateToController(objects, params)
        if name == "grasp":
            return self.graspController(objects,
                                        params,
                                        move_while_grasping=True)
        if name == "grasp_from_platform":
            return self.graspController(objects,
                                        params,
                                        move_while_grasping=False)
        if name == "placeOnTop":
            return self.placeOntopController(objects, params)
        assert name == "drag"
        return self.dragController(objects, params)

    def findController(self) -> None:
        """Execute look around."""
        # Execute a hard-coded sequence of movements and hope that one of them
        # puts the lost object in view. This is very specifically designed for
        # the case where an object has fallen in the immediate vicinity.

        # Start by stowing.
        self.stow_arm()

        # First move way back and don't move the hand. This is useful when the
        # object has not actually fallen, but wasn't grasped.
        if self._find_controller_move_queue_idx == 1:
            self.relative_move(-0.75, 0.0, 0.0)
            time.sleep(0.75)
            return

        # Now just look down.
        if self._find_controller_move_queue_idx == 2:
            pass

        # Move to the right.
        elif self._find_controller_move_queue_idx == 3:
            self.relative_move(0.0, 0.0, np.pi / 6)

        # Move to the left.
        elif self._find_controller_move_queue_idx == 4:
            self.relative_move(0.0, 0.0, -np.pi / 6)

        # Soon we should implement asking for help here instead of crashing.
        else:
            prompt = """Please take control of the robot and make the
            object become in its view. Hit the 'Enter' key
            when you're done!"""
            utils.prompt_user(prompt)
            self._find_controller_move_queue_idx = 0
            self.lease_client.take()
            return

        # Move the hand to get a view of the floor.
        self.hand_movement(np.array([0.0, 0.0, 0.0]),
                           keep_hand_pose=False,
                           angle=(np.cos(np.pi / 6), 0, np.sin(np.pi / 6), 0))

        # Sleep for longer to make sure that there is no shaking.
        time.sleep(0.75)

    def navigateToController(self, objs: Sequence[Object],
                             params: Array) -> None:
        """Controller that navigates to specific pre-specified locations.

        Params are [dx, dy, d-yaw (in radians)]
        """
        # Always start by stowing the arm.
        self.stow_arm()

        print("NavigateTo", objs)
        assert len(params) == 3
        assert len(objs) in [2, 3, 4]

        waypoint = ("", np.zeros(3, dtype=np.float32))  # default
        for obj in objs[1:]:
            possible_waypoint = get_memorized_waypoint(obj.name)
            if possible_waypoint is not None:
                waypoint = possible_waypoint
                break
        waypoint_id, offset = waypoint

        params = np.add(params, offset)

        if len(objs) == 3 and objs[2].name == "floor":
            self.navigate_to_position(params)
        else:
            self.navigate_to(waypoint_id, params)

        # Set arm view pose
        if len(objs) == 3:
            if "_table" in objs[2].name:
                self.hand_movement(np.array([0.0, 0.0, 0.0]),
                                   keep_hand_pose=False,
                                   angle=(np.cos(np.pi / 8), 0,
                                          np.sin(np.pi / 8), 0),
                                   open_gripper=False)
                return
            if "floor" in objs[2].name:
                self.hand_movement(np.array([-0.2, 0.0, -0.25]),
                                   keep_hand_pose=False,
                                   angle=(np.cos(np.pi / 7), 0,
                                          np.sin(np.pi / 7), 0),
                                   open_gripper=False)
                return
        if "platform" in objs[1].name:
            self.hand_movement(np.array([-0.2, 0.0, -0.25]),
                               keep_hand_pose=False,
                               angle=(np.cos(np.pi / 7), 0, np.sin(np.pi / 7),
                                      0),
                               open_gripper=False)
            time.sleep(1.0)
            return
        self.stow_arm()

    def graspController(self,
                        objs: Sequence[Object],
                        params: Array,
                        move_while_grasping: bool,
                        grasp_dist_in_hand: Optional[float] = None) -> None:
        """Wrapper method for grasp controller.

        Params are 4 dimensional corresponding to a top-down grasp (1),
        side grasp (-1) or any (0), and dx, dy, dz of post grasp
        position.
        """
        print("Grasp", objs)
        assert len(params) == 4
        assert params[3] in [0, 1, -1, 2]
        if params[3] == 1:
            self._force_horizontal_grasp = False
            self._force_top_down_grasp = True
            self._force_forward_grasp = False
        elif params[3] == -1:
            self._force_horizontal_grasp = True
            self._force_top_down_grasp = False
            self._force_forward_grasp = False
        elif params[3] == 2:
            self._force_horizontal_grasp = False
            self._force_top_down_grasp = False
            self._force_forward_grasp = True

        self.arm_object_grasp(objs[1],
                              move_while_grasping=move_while_grasping,
                              grasp_dist_in_hand=grasp_dist_in_hand)
        if not np.allclose(params[:3], [0.0, 0.0, 0.0]):
            self.hand_movement(params[:3], open_gripper=False)
        if objs[1].name == "platform":
            # Make sure the gripper is closed.
            gripper_close_command = RobotCommandBuilder.\
            claw_gripper_open_fraction_command(0.0)
            gripper_close_command_id = self.robot_command_client.robot_command(
                gripper_close_command)
            self.robot.logger.debug("Stow command issued.")
            block_until_arm_arrives(self.robot_command_client,
                                    gripper_close_command_id, 3.0)
        else:
            self.stow_arm()

    def placeOntopController(self, objs: Sequence[Object],
                             params: Array) -> None:
        """Wrapper method for placeOnTop controller.

        Params is dx, dy, and dz corresponding to the location of the
        arm from the robot when placing.
        """
        print("PlaceOntop", objs)
        angle = (np.cos(np.pi / 6), 0, np.sin(np.pi / 6), 0)
        assert len(params) == 3
        self.hand_movement(params,
                           keep_hand_pose=False,
                           relative_to_default_pose=False,
                           angle=angle)
        # Look down to see if the object landed where we hoped.
        self.hand_movement(params,
                           keep_hand_pose=False,
                           relative_to_default_pose=False,
                           open_gripper=False,
                           angle=(np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0))
        # Longer sleep necessary to prevent blurry images.
        time.sleep(2.0)

    def dragController(self, objs: Sequence[Object], params: Array) -> None:
        """Drag Controller."""
        print("Drag", objs)
        assert len(params) == 2  # [x, y] vector for direction
        self.drag_arm_control(params)
        time.sleep(0.5)
        self.stow_arm()

    def _scan_for_objects(
        self, waypoints: Sequence[str], objects_to_find: Collection[str]
    ) -> Dict[str, Tuple[float, float, float]]:
        """Walks around and spins around to find object poses."""
        # Stow arm before
        self.stow_arm()
        obj_poses: Dict[str, Tuple[float, float, float]] = {
            "floor": (0.0, 0.0, -1.0)
        }
        for waypoint_name in waypoints:
            if set(objects_to_find).issubset(set(obj_poses)):
                logging.info("All objects located!")
                break
            waypoint = get_memorized_waypoint(waypoint_name)
            assert waypoint is not None
            waypoint_id, offset = waypoint
            self.navigate_to(waypoint_id, offset)
            for i in range(8):
                if ('tool_room_table' in waypoint_name
                        and i > 5) or ('low_wall_rack' in waypoint_name
                                       and i in [0, 1, 6, 7]):
                    # Lift arm to pose where it can see things that are high.
                    # We only want to do this in situations where (1) we won't
                    # collide with an obstacle, and (2) we are likely to
                    # actually see something.
                    self.hand_movement(np.array([0.0, 0.0, 0.1]),
                                       keep_hand_pose=False,
                                       angle=(np.cos(0), 0, np.sin(0), 0),
                                       open_gripper=False)

                objects_in_view: Dict[str, Tuple[float, float, float]] = {}
                rgb_img_dict, rgb_img_response_dict, \
                    depth_img_dict, depth_img_response_dict = \
                        self.get_camera_images()
                # We want to get objects in view both using AprilTags and
                # using SAM potentially.
                objects_in_view_by_camera = {}
                objects_in_view_by_camera_apriltag = \
                    self.get_objects_in_view_by_camera(from_apriltag=True,
                                                rgb_image_dict=rgb_img_dict,
                                                rgb_image_response_dict=\
                                                    rgb_img_response_dict,
                                                depth_image_dict=\
                                                    depth_img_dict,
                                                depth_image_response_dict=\
                                                    depth_img_response_dict)
                objects_in_view_by_camera.update(
                    objects_in_view_by_camera_apriltag)
                if CFG.spot_grasp_use_sam:
                    objects_in_view_by_camera_sam = \
                        self.get_objects_in_view_by_camera(from_apriltag=False,
                                                    rgb_image_dict=rgb_img_dict,
                                                    rgb_image_response_dict=\
                                                        rgb_img_response_dict,
                                                    depth_image_dict=\
                                                        depth_img_dict,
                                                    depth_image_response_dict=\
                                                        depth_img_response_dict)
                    # Combine these together to get all objects in view.
                    for k, v in objects_in_view_by_camera.items():
                        v.update(objects_in_view_by_camera_sam[k])
                # Now update the seen objects vs. objects still
                # being searched for.
                for v in objects_in_view_by_camera.values():
                    objects_in_view.update(v)
                obj_poses.update(objects_in_view)
                logging.info("Seen objects:")
                logging.info(set(obj_poses))
                remaining_objects = set(objects_to_find) - set(obj_poses)
                if not remaining_objects:
                    break
                logging.info("Still searching for objects:")
                logging.info(remaining_objects)
                self.stow_arm()
                self.relative_move(0.0, 0.0, np.pi / 4)
        return obj_poses

    def verify_estop(self, robot: Any) -> None:
        """Verify the robot is not estopped."""

        client = robot.ensure_client(EstopClient.default_service_name)
        if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
            error_message = "Robot is estopped. Please use an external" + \
                " E-Stop client, such as the estop SDK example, to" + \
                " configure E-Stop."
            robot.logger.error(error_message)
            raise Exception(error_message)

    # NOTE: We want to deprecate this over the long-term!
    def cv_mouse_callback(self, event, x, y, flags, param):  # type: ignore
        """Callback for the click-to-grasp functionality with the Spot API's
        grasping interface."""
        del flags, param
        # pylint: disable=global-variable-not-assigned
        global g_image_click, g_image_display
        clone = g_image_display.copy()
        if event == cv2.EVENT_LBUTTONUP:
            g_image_click = (x, y)
        else:
            # Draw some lines on the image.
            color = (30, 30, 30)
            thickness = 2
            image_title = 'Click to grasp'
            height = clone.shape[0]
            width = clone.shape[1]
            cv2.line(clone, (0, y), (width, y), color, thickness)
            cv2.line(clone, (x, 0), (x, height), color, thickness)
            cv2.imshow(image_title, clone)

    def add_grasp_constraint(
        self, grasp: manipulation_api_pb2.PickObjectInImage,
        robot_state_client: RobotStateClient
    ) -> manipulation_api_pb2.PickObjectInImage:
        """Method to constrain desirable grasps."""
        # There are 3 types of constraints:
        #   1. Vector alignment
        #   2. Full rotation
        #   3. Squeeze grasp
        #
        # You can specify more than one if you want and they will be
        # OR'ed together.

        # For these options, we'll use a vector alignment constraint.
        use_vector_constraint = self._force_top_down_grasp or \
            self._force_horizontal_grasp or self._force_forward_grasp

        # Specify the frame we're using.
        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

        if use_vector_constraint:
            if self._force_top_down_grasp:
                # Add a constraint that requests that the x-axis of the
                # gripper is pointing in the negative-z direction in the
                # vision frame.

                # The axis on the gripper is the x-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

                # The axis in the vision frame is the negative z-axis
                axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

            if self._force_horizontal_grasp:
                # Add a constraint that requests that the y-axis of the
                # gripper is pointing in the positive-z direction in the
                # vision frame.  That means that the gripper is
                # constrained to be rolled 90 degrees and pointed at the
                # horizon.

                # The axis on the gripper is the y-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

                # The axis in the vision frame is the positive z-axis
                axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

            if self._force_forward_grasp:
                # Add a constraint that requests that the z-axis of the gripper
                # is pointing in the positive z direction in the vision frame.
                # This means that the gripper is constrained to be flat, as
                # it usually is in the standard sow position.

                # The axis on the gripper is the z-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=0, z=1)

                # The axis in the vision frame is the positive z-axis
                axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

            # Add the vector constraint to our proto.
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.vector_alignment_with_tolerance.\
                axis_on_gripper_ewrt_gripper.\
                    CopyFrom(axis_on_gripper_ewrt_gripper)
            constraint.vector_alignment_with_tolerance.\
                axis_to_align_with_ewrt_frame.\
                    CopyFrom(axis_to_align_with_ewrt_vo)

            # We'll take anything within about 10 degrees for top-down or
            # horizontal grasps.
            constraint.vector_alignment_with_tolerance.\
                threshold_radians = 0.17

        elif self._force_45_angle_grasp:
            # Demonstration of a RotationWithTolerance constraint.
            # This constraint allows you to specify a full orientation you
            # want the hand to be in, along with a threshold.
            # You might want this feature when grasping an object with known
            # geometry and you want to make sure you grasp a specific part
            # of it. Here, since we don't have anything in particular we
            # want to grasp,  we'll specify an orientation that will have the
            # hand aligned with robot and rotated down 45 degrees as an
            # example.

            # First, get the robot's position in the world.
            robot_state = robot_state_client.get_robot_state()
            vision_T_body = get_vision_tform_body(
                robot_state.kinematic_state.transforms_snapshot)

            # Rotation from the body to our desired grasp.
            body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
            vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

            # Turn into a proto
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
                vision_Q_grasp.to_proto())

            # We'll accept anything within +/- 10 degrees
            constraint.rotation_with_tolerance.threshold_radians = 0.17

        elif self._force_squeeze_grasp:
            # Tell the robot to just squeeze on the ground at the given point.
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.squeeze_grasp.SetInParent()

        return grasp

    def arm_object_grasp(self,
                         obj: Object,
                         move_while_grasping: bool = True,
                         grasp_dist_in_hand: Optional[float] = None) -> None:
        """A helper function (largely copied from SDK example) to grasp an
        object. We assume the object is already in the view of the hand camera
        before calling this function.

        The `move_while_grasping` param dictates whether we're allowing
        the robot to automatically move its feet while grasping or not.
        """
        assert self.robot.is_powered_on(), "Robot power on failed."
        assert basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING

        # Take a picture with a camera
        self.robot.logger.debug(
            f'Getting an image from: {self._grasp_image_source}')
        rgb_img, rgb_img_response = self.get_single_camera_image(
            self._grasp_image_source)

        # pylint: disable=global-variable-not-assigned, global-statement
        global g_image_click, g_image_display

        if CFG.spot_grasp_use_apriltag:
            # Convert Image to grayscale
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

            # Define the AprilTags detector options and then detect the tags.
            self.robot.logger.info("[INFO] detecting AprilTags...")
            options = apriltag.DetectorOptions(families="tag36h11")
            detector = apriltag.Detector(options)
            results = detector.detect(gray)
            self.robot.logger.info(f"[INFO] {len(results)} AprilTags detected")
            for result in results:
                if result.tag_id == obj_name_to_apriltag_id[obj.name]:
                    g_image_click = result.center

        elif CFG.spot_grasp_use_cv2:
            if obj.name in ["hammer", "brush"]:
                g_image_click = _find_object_center(rgb_img, obj.name)

        elif CFG.spot_grasp_use_sam:
            results = get_pixel_locations_with_detic_sam(
                obj_class=obj_name_to_vision_prompt[obj.name],
                rgb_image_dict={self._grasp_image_source: rgb_img},
                plot=CFG.spot_visualize_vision_model_outputs)

            if len(results) > 0:
                # We only want the most likely sample (for now).
                # NOTE: we make the hard assumption here that
                # we will only see one instance of a particular object
                # type. We can relax this later.
                assert len(results) == 1
                g_image_click = (int(results[0][0]), int(results[0][1]))

        if g_image_click is None:
            # Show the image to the user and wait for them to click on a pixel
            self.robot.logger.info('Click on an object to start grasping...')
            image_title = 'Click to grasp'
            cv2.namedWindow(image_title)
            cv2.setMouseCallback(image_title, self.cv_mouse_callback)
            g_image_display = rgb_img
            cv2.imshow(image_title, g_image_display)

        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                # Quit
                print('"q" pressed, exiting.')
                sys.exit()

        # Uncomment to debug.
        # g_image_display = img.copy()
        # image_title = "Selected grasp"
        # cv2.namedWindow(image_title)
        # cv2.circle(g_image_display, g_image_click, 3, (0, 255, 0), 3)
        # cv2.imshow(image_title, g_image_display)
        # cv2.waitKey(0)

        # pylint: disable=unsubscriptable-object
        self.robot.\
            logger.info(f"Object at ({g_image_click[0]}, {g_image_click[1]})")
        # pylint: disable=unsubscriptable-object
        pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

        # Build the proto. Note that the possible settings for walk_gaze_mode
        # can be found here:
        # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html#walkgazemode
        walk_gaze_mode = 1
        if not move_while_grasping:
            walk_gaze_mode = 2
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=rgb_img_response.shot.
            transforms_snapshot,
            frame_name_image_sensor=rgb_img_response.shot.
            frame_name_image_sensor,
            camera_model=rgb_img_response.source.pinhole,
            walk_gaze_mode=walk_gaze_mode)

        # We can specify where in the gripper we want to grasp.
        # About halfway is generally good for small objects like this.
        # For a bigger object like a shoe, 0 is better (use the entire
        # gripper)
        if grasp_dist_in_hand is not None:
            grasp.grasp_params.grasp_palm_to_fingertip = grasp_dist_in_hand

        # Optionally add a grasp constraint.  This lets you tell the robot you
        # only want top-down grasps or side-on grasps.
        grasp = self.add_grasp_constraint(grasp, self.robot_state_client)

        if move_while_grasping:
            # Stow Arm first (only if robot is allowed to move while grasping)
            self.stow_arm()

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp)

        # Send the request
        cmd_response = self.manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        # Get feedback from the robot and execute grasping.
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) <= COMMAND_TIMEOUT:
            feedback_request = manipulation_api_pb2.\
                ManipulationApiFeedbackRequest(manipulation_cmd_id=\
                    cmd_response.manipulation_cmd_id)

            # Send the request
            response = self.manipulation_api_client.\
                manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)
            if response.current_state in [manipulation_api_pb2.\
                MANIP_STATE_GRASP_SUCCEEDED, manipulation_api_pb2.\
                MANIP_STATE_GRASP_FAILED]:
                break
        if (time.perf_counter() - start_time) > COMMAND_TIMEOUT:
            logging.info("Timed out waiting for grasp to execute!")

        time.sleep(0.25)
        g_image_click = None
        g_image_display = None
        self.robot.logger.debug('Finished grasp.')

    def stow_arm(self) -> None:
        """A simple example of using the Boston Dynamics API to stow Spot's
        arm."""

        # Allow Stowing and Stow Arm
        grasp_carry_state_override = manipulation_api_pb2.\
            ApiGraspedCarryStateOverride(override_request=3)
        grasp_override_request = manipulation_api_pb2.\
            ApiGraspOverrideRequest(
            carry_state_override=grasp_carry_state_override)
        self.manipulation_api_client.\
            grasp_override_command(grasp_override_request)

        stow_cmd = RobotCommandBuilder.arm_stow_command()
        gripper_close_command = RobotCommandBuilder.\
            claw_gripper_open_fraction_command(0.0)
        # Combine the arm and gripper commands into one RobotCommand
        stow_and_close_command = RobotCommandBuilder.build_synchro_command(
            gripper_close_command, stow_cmd)
        stow_and_close_command_id = self.robot_command_client.robot_command(
            stow_and_close_command)
        self.robot.logger.debug("Stow command issued.")
        block_until_arm_arrives(self.robot_command_client,
                                stow_and_close_command_id, 4.5)

    def hand_movement(
        self,
        params: Array,
        open_gripper: bool = True,
        relative_to_default_pose: bool = True,
        keep_hand_pose: bool = True,
        keep_body_pose: bool = False,
        clip_z: bool = True,
        angle: Tuple[float, float, float,
                     float] = (np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0)
    ) -> None:
        """Move arm to infront of robot an open gripper."""
        # Move the arm to a spot in front of the robot, and open the gripper.
        assert self.robot.is_powered_on(), "Robot power on failed."
        assert basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING

        if keep_hand_pose:
            # Get current hand quaternion.
            robot_state = self.robot_state_client.get_robot_state()
            body_T_hand = get_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot,
                BODY_FRAME_NAME, "hand")
            qw, qx, qy, qz = body_T_hand.rot.w, body_T_hand.rot.x,\
                body_T_hand.rot.y, body_T_hand.rot.z
        else:
            # Set downward place rotation as a quaternion.
            qw, qx, qy, qz = angle
        flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        if not relative_to_default_pose:
            x = params[0]  # dx hand
            y = params[1]
            z = params[2]
        else:
            x = self.hand_x + params[0]  # dx hand
            y = self.hand_y + params[1]
            z = self.hand_z + params[2]

        # Here we are making sure the hand pose is within our range and
        # if it is not we are clipping it to be, and then displacing the
        # robot respectively.
        x_clipped = np.clip(x, self.hand_x_bounds[0], self.hand_x_bounds[1])
        y_clipped = np.clip(y, self.hand_y_bounds[0], self.hand_y_bounds[1])
        if clip_z:
            z_clipped = np.clip(z, self.hand_z_bounds[0],
                                self.hand_z_bounds[1])
        else:
            z_clipped = z
        if not keep_body_pose:
            self.relative_move((x - x_clipped), (y - y_clipped), 0.0)
        x = x_clipped
        y = y_clipped
        z = z_clipped

        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        robot_state = self.robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME,
            GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(
            flat_body_T_hand)

        # duration in seconds
        seconds = 2

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w,
            odom_T_hand.rot.x, odom_T_hand.rot.y, odom_T_hand.rot.z,
            ODOM_FRAME_NAME, seconds)

        # Make the close gripper RobotCommand
        gripper_command = RobotCommandBuilder.\
            claw_gripper_open_fraction_command(0.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(
            gripper_command, arm_command)

        # Send the request
        cmd_id: int = self.robot_command_client.robot_command(command)
        self.robot.logger.debug('Moving arm to position.')

        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(self.robot_command_client, cmd_id, 3.0)
        time.sleep(0.5)

        if not open_gripper:
            gripper_command = RobotCommandBuilder.\
                claw_gripper_open_fraction_command(0.0)
        else:
            gripper_command = RobotCommandBuilder.\
                claw_gripper_open_fraction_command(1.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(
            gripper_command, arm_command)

        # Send the request
        cmd_id = self.robot_command_client.robot_command(command)
        self.robot.logger.debug('Moving arm to position.')

        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(self.robot_command_client, cmd_id, 3.0)
        time.sleep(0.5)

    def navigate_to(self, waypoint_id: str, params: Array) -> None:
        """Use GraphNavInterface to localize robot and go to a location."""
        # pylint: disable=broad-except
        try:
            # (1) Initialize location
            self.graph_nav_command_line.set_initial_localization_fiducial()
            self.graph_nav_command_line.graph_nav_client.get_localization_state(
            )

            # (2) Navigate to
            self.graph_nav_command_line.navigate_to([waypoint_id])

            # (3) Offset by params
            if not np.allclose(params, [0.0, 0.0, 0.0]):
                self.relative_move(params[0], params[1], params[2])

        except Exception as e:
            logging.info(e)

    def relative_move(
        self,
        dx: float,
        dy: float,
        dyaw: float,
        max_xytheta_vel: Tuple[float, float, float] = (2.0, 2.0, 1.0),
        min_xytheta_vel: Tuple[float, float, float] = (-2.0, -2.0, -1.0)
    ) -> bool:
        """Move to relative robot position in body frame."""
        transforms = self.robot_state_client.get_robot_state(
        ).kinematic_state.transforms_snapshot

        # Build the transform for where we want the robot to be
        # relative to where the body currently is.
        body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
        # We do not want to command this goal in body frame because
        # the body will move, thus shifting our goal. Instead, we
        # transform this offset to get the goal position in the output
        # frame (which will be either odom or vision).
        out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME,
                                           BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal

        # Command the robot to go to the goal point in the specified
        # frame. The command will stop at the new position.
        # Constrain the robot not to turn, forcing it to strafe laterally.
        speed_limit = SE2VelocityLimit(
            max_vel=SE2Velocity(linear=Vec2(x=max_xytheta_vel[0],
                                            y=max_xytheta_vel[1]),
                                angular=max_xytheta_vel[2]),
            min_vel=SE2Velocity(linear=Vec2(x=min_xytheta_vel[0],
                                            y=min_xytheta_vel[1]),
                                angular=min_xytheta_vel[2]))
        mobility_params = spot_command_pb2.MobilityParams(
            vel_limit=speed_limit)

        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x,
            goal_y=out_tform_goal.y,
            goal_heading=out_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME,
            params=mobility_params)
        cmd_id = self.robot_command_client.robot_command(
            lease=None,
            command=robot_cmd,
            end_time_secs=time.time() + COMMAND_TIMEOUT)
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) <= COMMAND_TIMEOUT:
            feedback = self.robot_command_client.\
                robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.\
                synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != \
                RobotCommandFeedbackStatus.STATUS_PROCESSING:
                logging.info("Failed to reach the goal")
                return False
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL
                    and traj_feedback.body_movement_status
                    == traj_feedback.BODY_STATUS_SETTLED):
                logging.info("Arrived at the goal.")
                return True
        if (time.perf_counter() - start_time) > COMMAND_TIMEOUT:
            logging.info("Timed out waiting for movement to execute!")
        return False

    def navigate_to_position(self, params: Array) -> None:
        """Use GraphNavInterface to localize robot and go to a position."""
        # pylint: disable=broad-except

        try:
            # (1) Initialize location
            self.graph_nav_command_line.set_initial_localization_fiducial()
            self.graph_nav_command_line.graph_nav_client.\
                get_localization_state()

            # (2) Just move
            self.relative_move(params[0], params[1], params[2])

        except Exception as e:
            logging.info(e)

    def get_arm_proprioception(
        self,
        robot_state: Optional[bosdyn.api.robot_state_pb2.RobotState] = None
    ) -> Dict[str, bosdyn.api.robot_state_pb2.JointState]:
        """Return state of each of the 6 joints of the arm."""
        if robot_state is None:
            robot_state = self.robot_state_client.get_robot_state()
        arm_joint_states = OrderedDict({
            i.name[len("arm0."):]: i
            for i in robot_state.kinematic_state.joint_states
            if i.name in ARM_6DOF_NAMES
        })

        return arm_joint_states

    def lock_arm(self) -> None:
        """Helper function to lock arm in body frame when moving."""
        arm_proprioception = self.get_arm_proprioception()
        positions = np.array(
            [v.position.value for v in arm_proprioception.values()])
        sh0, sh1, el0, el1, wr0, wr1 = positions

        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1)
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(
            points=[traj_point])
        # Make a RobotCommand
        joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(
            trajectory=arm_joint_traj)
        arm_command = arm_command_pb2.ArmCommand.Request(
            arm_joint_move_command=joint_move_command)
        sync_arm = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command)
        arm_sync_robot_cmd = robot_command_pb2.RobotCommand(
            synchronized_command=sync_arm)
        command = RobotCommandBuilder.build_synchro_command(arm_sync_robot_cmd)

        # Send the request
        self.robot_command_client.robot_command(command)
        self.robot.logger.info('Locking Arm.')

    def drag_arm_control(self, params: Array) -> None:
        """Simple drag controller by locking arm position in body frame.

        then moving to a location given by params.
        """
        # NOTE: Core idea here is we first move the arm to be centered with the
        # robot, then lock the arm to the robot, then move in the y direction
        # in the world (which is horizontally in the room), then move
        # in the x direction in the world (which is forwards)
        # This may require some modification as we start dragging to different
        # locations.

        assert len(params) == 2
        x, y, _, yaw = self.get_robot_pose()
        robot_to_world = math_helpers.SE2Pose(x, y, yaw).inverse()
        world_to_desiredplatform = math_helpers.Vec2(
            params[0],
            params[1],
        )

        # Move Hand to Front of Body
        robot_state = self.robot_state_client.get_robot_state()
        body_T_hand = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            GRAV_ALIGNED_BODY_FRAME_NAME, "hand")
        self.hand_movement(np.array([body_T_hand.x, 0.0, body_T_hand.z]),
                           keep_hand_pose=True,
                           keep_body_pose=True,
                           clip_z=False,
                           relative_to_default_pose=False,
                           open_gripper=False)

        # Move Body Horizontally in the world.
        robot_state = self.robot_state_client.get_robot_state()
        robot_T_hand = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            GRAV_ALIGNED_BODY_FRAME_NAME, "hand")
        state = self.get_localized_state()
        gn_origin_T_robot = math_helpers.SE3Pose.from_obj(
            state.localization.seed_tform_body)
        gn_origin_T_hand = gn_origin_T_robot * robot_T_hand
        # IMPORTANT: we want to keep the x pose of the platform the same,
        # and we can get this by computing the location of the
        # hand in the world frame.
        world_to_desiredplatform_y_with_robot_x = math_helpers.Vec2(
            gn_origin_T_hand.x, world_to_desiredplatform[1])
        robot_to_desiredplatform_y_with_robot_x = robot_to_world * \
            world_to_desiredplatform_y_with_robot_x
        self.lock_arm()
        self.relative_move(
            dx=robot_to_desiredplatform_y_with_robot_x[0] - body_T_hand.x,
            dy=robot_to_desiredplatform_y_with_robot_x[1] - body_T_hand.y,
            dyaw=0.0,
            max_xytheta_vel=(0.25, 0.25, 0.5),
            min_xytheta_vel=(-0.25, -0.25, -0.5))

        # Move Body remaining
        robot_state = self.robot_state_client.get_robot_state()
        body_T_hand = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            GRAV_ALIGNED_BODY_FRAME_NAME, "hand")
        x, y, _, yaw = self.get_robot_pose()
        robot_to_world = math_helpers.SE2Pose(x, y, yaw).inverse()
        robot_to_desiredplatform = robot_to_world * world_to_desiredplatform
        self.lock_arm()
        self.relative_move(dx=robot_to_desiredplatform[0] - body_T_hand.x,
                           dy=robot_to_desiredplatform[1] - body_T_hand.y,
                           dyaw=0.0,
                           max_xytheta_vel=(0.25, 0.25, 0.5),
                           min_xytheta_vel=(-0.25, -0.25, -0.5))

        # Open Gripper
        gripper_command = RobotCommandBuilder.\
            claw_gripper_open_fraction_command(1.0)

        # Send the request
        self.robot_command_client.robot_command(gripper_command)
        self.robot.logger.debug('Opening Gripper.')
        time.sleep(1)


@functools.lru_cache(maxsize=None)
def get_spot_interface() -> _SpotInterface:
    """Ensure that _SpotControllers is only created once."""
    return _SpotInterface()
