# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

# mypy: ignore-errors
"""Command line interface for graph nav with options to download/upload a map
and to navigate a map."""

import time

from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client import ResponseError, RpcError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient


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
        self.graph_nav_client = self._robot.ensure_client(
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

        # Stuff that is set in start()
        self._robot_id = None

    def start(self) -> None:
        """Begin communication with the robot."""
        self._robot_id = self._robot.get_id()

    def set_initial_localization_fiducial(self) -> None:
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are
        # asking it to localize based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self.graph_nav_client.set_localization(
            initial_guess_localization=localization,
            ko_tform_body=current_odom_tform_body)

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
        response = self.graph_nav_client.upload_graph(
            graph=self._current_graph, generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self.graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print(f"Uploaded {waypoint_snapshot.id}")
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self.graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print(f"Uploaded {edge_snapshot.id}")

        # The upload is complete! Check that the robot is localized to the
        # graph, and if it is not, prompt the user to localize the robot
        # before attempting any navigation commands.
        localization_state = self.graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print("\n")
            print("Upload complete! The robot is currently not localized to", \
                "the map; please localize the robot using command (1).")

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

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that
            # it is easy to terminate the navigation command (with estop or
            # killing the program).
            try:
                nav_to_cmd_id = self.graph_nav_client.navigate_to(
                    destination_waypoint, 1.0, command_id=nav_to_cmd_id)
            except ResponseError as e:
                print(f"Error while navigating {e}")
                break
            time.sleep(
                .5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation
            # command is complete.
            is_finished = self._check_success(nav_to_cmd_id)

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self.graph_nav_client.navigation_feedback(command_id)
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
