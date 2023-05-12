# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

# mypy: ignore-errors
"""Command line interface for graph nav with options to download/upload a map
and to navigate a map."""

import time

from bosdyn.api import robot_state_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client import ResponseError, RpcError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient, power_on, safe_power_off
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient

from predicators.spot_utils.helpers import graph_nav_util


# pylint: disable=no-member
class GraphNavInterface():
    """GraphNav service command line interface."""

    def __init__(self, robot, upload_path, lease_client,
                 lease_keepalive) -> None:
        self._robot = robot

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create robot state and command clients.
        self._robot_command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name)
        self._robot_state_client = self._robot.ensure_client(
            RobotStateClient.default_service_name)

        # Create clients -- do not use the for communication yet.
        self._lease_client = lease_client
        self._lease_keepalive = lease_keepalive

        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(
            GraphNavClient.default_service_name)

        # Create a power client for the robot.
        self._power_client = self._robot.ensure_client(
            PowerClient.default_service_name)

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (
            power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        # Number of attempts to wait before trying to re-power on.
        self._max_attempts_to_wait = 50

        # Store the most recent state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = {}  # maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = {}  # maps id to waypoint snapshot
        self._current_edge_snapshots = {}  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = {}

        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == "/":
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path

        # Upload graph and snapshots on start.
        self._upload_graph_and_snapshots()

        self._command_dictionary = {
            '1': self.set_initial_localization_fiducial,
            '2': self.get_localization_state,
            '3': self.list_graph_waypoint_and_edge_ids,
            '4': self.navigate_to
        }

        # Stuff that is set in start()
        self._robot_id = None

    def start(self) -> None:
        """Begin communication with the robot."""
        self._robot_id = self._robot.get_id()

    def get_localization_state(self) -> None:
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        print(f'Got localization: \n{str(state.localization)}')
        odom_tform_body = get_odom_tform_body(
            state.robot_kinematics.transforms_snapshot)
        print(f'Got robot state in odometry frame: \n{str(odom_tform_body)}')

    def set_initial_localization_fiducial(self) -> None:
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are
        # asking it to localize based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            ko_tform_body=current_odom_tform_body)

    def list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the
        robot."""
        del args

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state(
        ).localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = \
            graph_nav_util.update_waypoints_and_edges(graph, localization_id)

    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        del args
        print("Loading the graph from disk into local storage...")
        with open(self._upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print(
                f"Loaded graph has {len(self._current_graph.waypoints)} " + \
                f"waypoints and {len(self._current_graph.edges)} edges"
            )
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(
                    self._upload_filepath +
                    f"/waypoint_snapshots/{(waypoint.snapshot_id)}",
                    "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[
                    waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(
                    self._upload_filepath +
                    f"/edge_snapshots/{edge.snapshot_id}",
                    "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(
            graph=self._current_graph, generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print(f"Uploaded {waypoint_snapshot.id}")
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print(f"Uploaded {edge_snapshot.id}")

        # The upload is complete! Check that the robot is localized to the
        # graph, and if it is not, prompt the user to localize the robot
        # before attempting any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print("\n")
            print("Upload complete! The robot is currently not localized to", \
                "the map; please localize the robot using command (1).")

    def _navigate_to_anchor(self, *args):
        """Navigate to a pose in seed frame, using anchors."""
        # The following options are accepted for arguments:
        # [x, y], [x, y, yaw], [x, y, z, yaw], [x, y, z, qw, qx, qy, qz].
        # When a value for z is not specified, we use the current z height.
        # When only yaw is specified, the quaternion is constructed from
        # the yaw. When yaw is not specified, an identity quaternion is used.

        if len(args) < 1 or len(args[0]) not in [2, 3, 4, 7]:
            print("Invalid arguments supplied.")
            return

        seed_T_goal = SE3Pose(float(args[0][0]), float(args[0][1]), 0.0,
                              Quat())

        if len(args[0]) in [4, 7]:
            seed_T_goal.z = float(args[0][2])
        else:
            localization_state = self._graph_nav_client.get_localization_state(
            )
            if not localization_state.localization.waypoint_id:
                print("Robot not localized")
                return
            seed_T_goal.z = \
                localization_state.localization.seed_tform_body.position.z

        if len(args[0]) == 3:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][2]))
        elif len(args[0]) == 4:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][3]))
        elif len(args[0]) == 7:
            seed_T_goal.rot = Quat(w=float(args[0][3]),
                                   x=float(args[0][4]),
                                   y=float(args[0][5]),
                                   z=float(args[0][6]))

        if not self.toggle_power(should_power_on=True):
            print("Failed to power on, cannot complete navigation request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is
            # easy to terminate the navigation command (with estop or killing
            # the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to_anchor(
                    seed_T_goal.to_proto(), 1.0, command_id=nav_to_cmd_id)
            except ResponseError as e:
                print(f"Error while navigating {e}")
                break
            time.sleep(
                .5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation
            # command is complete.
            is_finished = self._check_success(nav_to_cmd_id)

    def navigate_to(self, *args) -> None:
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without
            # requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return

        destination_waypoint = args[0][0]
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the
            # navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on, cannot complete navigation request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that
            # it is easy to terminate the navigation command (with estop or
            # killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(
                    destination_waypoint, 1.0, command_id=nav_to_cmd_id)
            except ResponseError as e:
                print(f"Error while navigating {e}")
                break
            time.sleep(
                .5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation
            # command is complete.
            is_finished = self._check_success(nav_to_cmd_id)

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a
            # powered-off state.
            power_on(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(
                    timeout=10
                )  # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == \
                    robot_state_pb2.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off(self._robot_command_client,
                           self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (
            power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == \
            graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        if status.status == \
            graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print("Robot got lost when navigating the route.")
            return True
        if status.status == \
            graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print("Robot got stuck when navigating the route.")
            return True
        if status.status == \
            graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print("Robot is impaired.")
            return True
        # Navigation command is not complete yet.
        return False

    def _on_quit(self) -> None:
        """Cleanup on quit from the command line interface."""
        return None

    def _try_grpc(self, desc, thunk):
        try:
            return thunk()
        except (ResponseError, RpcError, LeaseBaseError) as err:
            self.add_message(f"Failed {desc}: {err}")
            return None
