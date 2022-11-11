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
ordered from left to right and then top to bottom, following this convention:

    {
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
from pathlib import Path
import glob
import json
from typing import List

import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import time

from numpy.typing import NDArray

import pybullet as p

from predicators import utils

from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_env import create_pybullet_block
from predicators.pybullet_helpers.link import get_link_state
from predicators.structs import Image, State
from predicators.utils import LineSegment
from predicators.pybullet_helpers.robots import create_single_arm_pybullet_robot


def _main(rgb_path: Path, depth_path: Path, goal_path: Path, extrinsics_path: Path, intrinsics_path: Path, output_path: Path, debug_viz: bool = False) -> None:
    # Load images.
    rgb = imageio.imread(rgb_path)
    depth = imageio.imread(depth_path)

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
    with open(intrinsics_path, "r") as f:
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
        min_bound=workspace_min_bounds, max_bound=workspace_max_bounds
    )
    cropped_pcd = pcd.crop(workspace_bounds)

    if debug_viz:
        _visualize_point_cloud(cropped_pcd)

    # Mask out the gray floor (to the extent possible).
    def non_grey_mask(rgb: Image, thresh: float = 10 / 255) -> NDArray[bool]:
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
    import ipdb; ipdb.set_trace()    



def _visualize_point_cloud(pcd: o3d.geometry.PointCloud) -> None:
    # Show point cloud, press 'q' to close.
    o3d.visualization.draw_geometries(
        [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]
    )


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
    _main(args.rgb, args.depth, args.goal, args.extrinsics, args.intrinsics, args.output, args.debug_viz)
