"""Converts an RGBD image and goal into a blocks task JSON specification.

Note that this script has an open3d dependency that we exclude from the rest of
predicators.

Usage example:

    python scripts/run_blocks_perception.py \
        --rgb blocks_vision_data/color-0-821212060083.png \
        --depth blocks_vision_data/depth-0-821212060083.png \
        --goal blocks_vision_data/goal-0.json \
        --extrinsics blocks_vision_data/extrinsics.json \
        --intrinsics blocks_vision_data/intrinsics.json \
        --output blocks_vision_data/blocks-vision-task0.json \
        --debug_viz

## Inputs to this script

The RGBD image should be collected from a top-down view of the Panda workspace.

The camera should be calibrated following these instructions:
https://docs.google.com/document/d/1iaI8RAZeAaFOTRQvhc_L-l-4tdkOECSA5X5dGqCTUdk/

The output of that calibration process should be an instrinsics metadata JSON
and an extrinsics metadata JSON. The intrinsics metadata should include:

    {
        "depth_scale": float  // converts depth into meters
        "intrinsics": {
            "width": int,
            "height": int,
            "intrinsic_matrix": List[List[float]],  // 3x3
        }
    }

The extrinsics metadata should include:

    {
        "c2w": List[List[float]], // 4x4 camera to world transform
        "depth_trunc": float,  // ignore points beyond this distance in meters
        "workspace_min_bounds": [float, float, float],
        "workspace_max_bounds": [float, float, float],
    }

The task goal should be specified in another JSON file, with block names
ordered from left to right and then bottom to top, following this convention:

    {
        "goal": {
            "On": [
                ["block4", "block2"],
                ["block2", "block1"]
            ],
            "OnTable": [
                ["block1"]
            ]
        }
    }

## Outputs of this script

This script outputs a new JSON file in the form

    {
        "problem_name": str,
        "blocks": {
            block_id_1: {
                "position": [x, y, z],
                "color": [r, g, b]
            },
            block_id_2: { ... }
        },
        "block_size": float,
        "goal": {
            "On": [
                ["block1", "block2"],
                ["block2", "block3"]
            ],
            "OnTable": [
                ["block3"]
            ]
        }
    }
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d  # pylint: disable=import-error
import pybullet as p
from numpy.typing import NDArray

from predicators import utils
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_env import create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose3D
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot
from predicators.structs import Image


def _main(rgb_path: Path,
          depth_path: Path,
          goal_path: Path,
          extrinsics_path: Path,
          intrinsics_path: Path,
          output_path: Path,
          debug_viz: bool = False,
          dbscan_eps: float = 0.02,
          dbscan_min_points: int = 50) -> None:
    # Load images.
    rgb = iio.imread(rgb_path)
    depth = iio.imread(depth_path)

    # Load goal.
    with open(goal_path, "r", encoding="utf-8") as f:
        goal_dict = json.load(f)

    # Load extrinsics.
    with open(extrinsics_path, "r", encoding="utf-8") as f:
        extrinsics_dict = json.load(f)
    c2w = np.array(extrinsics_dict["c2w"])
    # We have to give open3d w2c matrix so invert c2w.
    w2c = np.linalg.inv(c2w)
    depth_trunc = extrinsics_dict["depth_trunc"]
    workspace_min_bounds = extrinsics_dict["workspace_min_bounds"]
    workspace_max_bounds = extrinsics_dict["workspace_max_bounds"]

    # Load intrinsics.
    with open(intrinsics_path, "r", encoding="utf-8") as f:
        intrinsics_metadata_dict = json.load(f)
    intrinsics_dict = intrinsics_metadata_dict["intrinsics"]
    depth_scale = intrinsics_metadata_dict["depth_scale"]
    intrinsic_matrix = np.array(intrinsics_dict["intrinsic_matrix"])
    width = intrinsics_dict["width"]
    height = intrinsics_dict["height"]

    # Validation.
    assert rgb.shape == (height, width, 3)
    assert depth.shape == (height, width)
    assert "goal" in goal_dict
    assert w2c.shape == (4, 4)
    assert intrinsic_matrix.shape == (3, 3)

    # Create open3d RGBD.
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        # Convert depth to meters
        depth_scale=depth_scale,
        # Remove points farther than this length (in meters)
        depth_trunc=depth_trunc,
        # Don't use RGB to determine depth
        convert_rgb_to_intensity=False,
    )

    # Create open3d pinhole camera model using intrinsics.
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )

    # Create point cloud given RGBD, intrinsics and extrinsics.
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d_intrinsic,
        w2c,
    )

    if debug_viz:
        _visualize_point_cloud(pcd)

    # Crop the point clouds to the workspace we care about
    # (i.e., the line the blocks lie across).
    workspace_bounds = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=workspace_min_bounds, max_bound=workspace_max_bounds)
    cropped_pcd = pcd.crop(workspace_bounds)

    if debug_viz:
        _visualize_point_cloud(cropped_pcd)

    # Mask out the gray floor (to the extent possible).
    def non_grey_mask(rgb: Image, thresh: float = 10 / 255) -> NDArray:
        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]
        return np.logical_and(np.abs(r - g) > thresh, np.abs(r - b) > thresh)

    mask = non_grey_mask(np.asarray(cropped_pcd.colors))
    indices = np.where(mask)[0]
    masked_pcd = cropped_pcd.select_by_index(indices)

    if debug_viz:
        _visualize_point_cloud(masked_pcd)

    # Cluster the points into piles.
    cluster_labels = np.array(
        masked_pcd.cluster_dbscan(eps=dbscan_eps,
                                  min_points=dbscan_min_points))
    max_label = cluster_labels.max()
    if debug_viz:
        cmap = plt.get_cmap("tab20")
        colors = cmap(cluster_labels / (max_label if max_label > 0 else 1))
        colors[cluster_labels < 0] = 0
        clusters_pcd = o3d.geometry.PointCloud(masked_pcd)
        clusters_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        _visualize_point_cloud(clusters_pcd)

    # Infer the pile (x, y) and heights.
    piles_data: List[Dict[str, float]] = []
    for cluster_id in range(max_label + 1):
        indices = np.where(cluster_labels == cluster_id)[0]
        cluster_pcd = masked_pcd.select_by_index(indices)
        _, _, max_z = cluster_pcd.get_max_bound()
        center_x, center_y, _ = cluster_pcd.get_center()
        piles_data.append({
            "center_x": center_x,
            "center_y": center_y,
            "max_z": max_z
        })
        if debug_viz:
            _visualize_point_cloud(cluster_pcd)

    # Get the translation for the robot base in the PyBullet environment.
    w2b = _get_world_to_base()

    # Create the blocks from the pile data.
    blocks_data: Dict[str, Dict[str, Any]] = {}
    # Assume the block size in the PyBullet environment is correct.
    block_size = PyBulletBlocksEnv.block_size
    # Sorting convention (must agree with goal specification): left to right
    # then bottom to top.
    block_name_count = 1
    for pile in sorted(piles_data, key=lambda pd: pd["center_y"]):
        center_x = pile["center_x"]
        center_y = pile["center_y"]
        max_z = pile["max_z"]
        num_blocks = int(max_z / block_size + 0.5)
        for i in range(num_blocks):
            block_name = f"block{block_name_count}"
            z = i * block_size + block_size / 2
            bx, by, bz = np.add(w2b, [center_x, center_y, z])
            block_data = {"position": [bx, by, bz]}
            blocks_data[block_name] = block_data
            block_name_count += 1

    # Guess block colors based on RGB of points within the bounding boxes of
    # the blocks. This doesn't work that well right now because the vertical
    # calibration is poor, but it doesn't actually matter for anything.
    translated_pcd = o3d.geometry.PointCloud(masked_pcd).translate(w2b)
    for block_data in blocks_data.values():
        x, y, z = block_data["position"]
        hs = block_size / 2
        block_min_bounds = (x - hs, y - hs, z - hs)
        block_max_bounds = (x + hs, y + hs, z + hs)
        # Create bounding box.
        block_bounds = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=block_min_bounds, max_bound=block_max_bounds)
        block_pcd = translated_pcd.crop(block_bounds)
        block_rgb = np.asarray(block_pcd.colors)
        r, g, b = np.mean(block_rgb, axis=0)
        block_data["color"] = [r, g, b]

    # Create and save the output JSON.
    # Infer a problem name from the color file name.
    problem_name = f"problem-{rgb_path.stem}"
    output_dict = {
        "problem_name": problem_name,
        "blocks": blocks_data,
        "block_size": block_size,
        **goal_dict,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4)
    print(f"\nDumped to {output_path}")

    # Create a PyBullet visualization for debugging.
    if debug_viz:
        _visualize_pybullet(blocks_data, translated_pcd)


def _visualize_point_cloud(pcd: o3d.geometry.PointCloud) -> None:
    # Show point cloud, press 'q' to close.
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, frame])  # pylint: disable=no-member


def _visualize_pybullet(blocks_data: Dict[str, Dict[str, Any]],
                        pcd: o3d.geometry.PointCloud) -> None:
    utils.reset_config()
    physics_client_id = p.connect(p.GUI)
    # Disable the preview windows for faster rendering.
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                               False,
                               physicsClientId=physics_client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                               False,
                               physicsClientId=physics_client_id)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                               False,
                               physicsClientId=physics_client_id)
    # Load table.
    table_pose = PyBulletBlocksEnv._table_pose  # pylint: disable=protected-access
    table_orientation = PyBulletBlocksEnv._table_orientation  # pylint: disable=protected-access
    table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                          useFixedBase=True,
                          physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(table_id,
                                      table_pose,
                                      table_orientation,
                                      physicsClientId=physics_client_id)
    # Create the robot.
    ee_home = (PyBulletBlocksEnv.robot_init_x, PyBulletBlocksEnv.robot_init_y,
               PyBulletBlocksEnv.robot_init_z)
    create_single_arm_pybullet_robot("panda", physics_client_id, ee_home)
    # Show the point cloud. Downsample because PyBullet has a limit on points.
    pcd = pcd.farthest_point_down_sample(5000)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    p.addUserDebugPoints(points,
                         colors,
                         pointSize=10,
                         physicsClientId=physics_client_id)
    # Show the inferred blocks.
    block_size = PyBulletBlocksEnv.block_size
    half_extents = (block_size / 2.0, block_size / 2.0, block_size / 2.0)
    mass = -1
    friction = 1.0
    orientation = [0.0, 0.0, 0.0, 1.0]
    for block_data in blocks_data.values():
        bx, by, bz = block_data["position"]
        r, g, b = block_data["color"]
        color = (r, g, b, 1.0)
        block_id = create_pybullet_block(color, half_extents, mass, friction,
                                         orientation, physics_client_id)
        p.resetBasePositionAndOrientation(block_id, [bx, by, bz],
                                          orientation,
                                          physicsClientId=physics_client_id)

    while True:
        time.sleep(0.01)
        p.stepSimulation(physicsClientId=physics_client_id)


def _get_world_to_base() -> Pose3D:
    """Get the translation for the Panda robot in PyBullet blocks env."""
    utils.reset_config()
    physics_client_id = p.connect(p.DIRECT)
    ee_home = (PyBulletBlocksEnv.robot_init_x, PyBulletBlocksEnv.robot_init_y,
               PyBulletBlocksEnv.robot_init_z)
    robot = create_single_arm_pybullet_robot("panda", physics_client_id,
                                             ee_home)
    dx, dy, dz = p.getBasePositionAndOrientation(
        robot.robot_id, physicsClientId=physics_client_id)[0]
    return (dx, dy, dz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--goal", required=True, type=Path)
    parser.add_argument("--extrinsics", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--debug_viz", action="store_true")
    args = parser.parse_args()
    _main(args.rgb, args.depth, args.goal, args.extrinsics, args.intrinsics,
          args.output, args.debug_viz)
