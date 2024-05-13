from copy import copy
import json
from typing import Dict, List, Tuple
import numpy as np
import numpy.typing as npt
import cv2
from cv2 import aruco
import pickle as pkl
import os, sys
import itertools
from experiments.envs.pybullet_packing.env import PyBulletPackingEnv
from predicators.pybullet_helpers.geometry import Pose
import pybullet as p

from predicators.pybullet_helpers.robots.panda import PandaPyBulletRobot
from predicators.settings import CFG

# Creating a detector for aruco tags
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Fetching assets
asset_path = os.path.join(os.path.dirname(__file__), "assets")

## Per block data
block_id_to_dimensions = {}
block_id_to_tags = {}
tags_path = os.path.join(asset_path, "tags")
for fname in os.listdir(tags_path):
    assert fname.endswith('info.pkl')
    block_id = int(fname.split('_')[1])
    block_info = pkl.load(open(os.path.join(tags_path, fname), 'rb'))
    block_id_to_dimensions[block_id] = np.array(block_info['dimensions'])
    block_id_to_tags[block_id] = {tag_id: tag_info for tag_id, tag_info in block_info.items() if type(tag_id) == int}

## Camera intrinsics
intrinsics = np.load(os.path.join(asset_path, "intrinsics.npz"))
intrinsics_K = intrinsics['K']
intrinsics_D = intrinsics['D']

## Camera extrinsics
ee_from_camera = Pose(*pkl.load(open(os.path.join(asset_path, "extrinsics.pkl"), 'rb')))

def process_tags(color_image):
    corners, ids, _ = detector.detectMarkers(
        cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    )
    if ids is None:
        return np.empty((0,1)), ()
    return ids, corners

def pnp_block_poses(ids, corners, min_tags=1):
    block_id_to_corners = {}
    corner_coeffs = np.array([[-1,1], [1, 1], [1, -1], [-1, -1]])/2
    for i in range(0, ids.size):
        # pull out the info corresponding to this block
        tag_id, = ids[i]
        block_id = tag_id // 6
        img_corners, = corners[i]
        marker_info = block_id_to_tags[block_id][tag_id]
        tag_length = marker_info["marker_size_cm"]/100 # in meters
        X_OT = marker_info["X_OT"]

        # create tag-frame poses of the corners
        X_TCs = np.zeros([4,4,4])
        X_TCs[:] = np.eye(4)
        X_TCs[:, :2, 3] = tag_length * corner_coeffs

        # and get the object frame poses
        X_OCs = np.einsum('ij,njk->nik', X_OT, X_TCs)
        obj_corners = X_OCs[:, :3, 3]

        if block_id not in block_id_to_corners:
            block_id_to_corners[block_id] = {'obj': obj_corners, 'img': img_corners}
        else:
            block_id_to_corners[block_id]['obj'] = np.vstack([block_id_to_corners[block_id]['obj'], obj_corners])
            block_id_to_corners[block_id]['img'] = np.vstack([block_id_to_corners[block_id]['img'], img_corners])

    block_id_to_camera_from_block = {}
    for block_id, corners in block_id_to_corners.items():
        if corners['img'].shape[0] < min_tags*4:
            continue
        _, rvec, tvec = cv2.solvePnP(corners['obj'], corners['img'], intrinsics_K, intrinsics_D)
        block_id_to_camera_from_block[block_id] = Pose(
            tvec,
            p.getQuaternionFromAxisAngle(rvec.flatten(), np.linalg.norm(rvec)) # type: ignore
        )

    return block_id_to_camera_from_block

def apply_extrinsics(block_id_to_camera_from_block, world_from_ee):
    return {
        block_id: world_from_ee.multiply(ee_from_camera, camera_from_block)
        for block_id, camera_from_block in block_id_to_camera_from_block.items()
    }

def get_block_corners(dimensions):
    return list(itertools.product(*np.vstack([dimensions/2, -dimensions/2]).T))

def reposition_tall(dimensions):
    d, w, h = dimensions
    if d >= w and d >= h:
        return Pose.from_rpy((0, 0, 0), (0, np.pi/2, 0)), (h, w, d)
    if w >= d and w >= h:
        return Pose.from_rpy((0, 0, 0), (np.pi/2., 0, 0)), (d, h, w)
    return Pose.identity(), (d, w, h)

def adjust_block(world_from_block: Pose, dimensions, floor_z=0.0):
    # Snap orientation
    best_z_axis, cos_angle = max([
        (rot_axis, rot_axis[2])
        for z_axis in np.vstack([np.eye(3), -np.eye(3)])
        for rot_axis in [Pose((0, 0, 0), world_from_block.orientation).multiply(Pose(z_axis)).position]
    ], key=lambda x: x[1])
    rotation = p.getQuaternionFromAxisAngle(np.cross(best_z_axis, np.array([0, 0, 1])), np.arccos(cos_angle))
    world_from_block = Pose(world_from_block.position, rotation).multiply(Pose((0, 0, 0), world_from_block.orientation))
    best_z_axis, cos_angle = max([
        (rot_axis, rot_axis[2])
        for z_axis in np.vstack([np.eye(3), -np.eye(3)])
        for rot_axis in [Pose((0, 0, 0), world_from_block.orientation).multiply(Pose(z_axis)).position]
    ], key=lambda x: x[1])

    # Snap position
    block_from_corners = list(map(Pose, get_block_corners(dimensions)))
    min_z = min(world_from_block.multiply(block_from_corner).position[2] for block_from_corner in block_from_corners)
    world_from_block = Pose(np.hstack([world_from_block.position[:2], world_from_block.position[2] - min_z + floor_z]), world_from_block.orientation) # type: ignore

    # Make the block tall
    new_block_from_block, new_dimensions = reposition_tall(dimensions)

    return world_from_block.multiply(new_block_from_block), new_dimensions

def draw_block(camera_from_object: Pose, dimensions, color_image, draw_axis=True):
    signed_corners = np.array([c for c in itertools.product([-1, 1], repeat=3)])

    # get the translation from the COG to the box corner points
    corner_poss = get_block_corners(dimensions)
    # and convert the corner points to camera frame
    t_CP = np.array([camera_from_object.multiply(Pose(corner_pos)).position for corner_pos in corner_poss])
    # project the points into the image coordinates
    image_points, _ = cv2.projectPoints(t_CP, np.eye(3), np.zeros(3), intrinsics_K, intrinsics_D)
    # and draw them into the image
    for corner, image_pt in zip(signed_corners, image_points):
        color = np.array([1.0,0.0,0.7])*100 + (corner[2] > 0) * 155
        cv2.circle(color_image, tuple(image_pt[0].astype(int)), 5, color, -1)

    if draw_axis:
        axis, angle = p.getAxisAngleFromQuaternion(camera_from_object.orientation)
        cv2.drawFrameAxes(
            color_image, intrinsics_K, intrinsics_D,
            np.array(axis) * angle, np.array(camera_from_object.position), dimensions.min()/2
        )

def draw_debug(color_image):
    ids, corners = process_tags(color_image)
    block_id_to_block_pose = pnp_block_poses(ids, corners)

    color_image = color_image.copy()
    aruco.drawDetectedMarkers(color_image, corners, ids)
    for block_id in block_id_to_block_pose.keys():
        block_from_camera = block_id_to_block_pose[block_id]
        dimensions = block_id_to_dimensions[block_id]
        draw_block(block_from_camera, dimensions, color_image, intrinsics)

    cv2.imshow('Blocks', color_image)

def get_blocks_data(color_image: npt.NDArray, world_from_ee: Pose) -> List[Tuple[int, Pose]]:
    ids, corners = process_tags(color_image)
    block_id_to_camera_from_block = pnp_block_poses(ids, corners)
    block_id_to_world_from_block = apply_extrinsics(block_id_to_camera_from_block, world_from_ee)
    return list(block_id_to_world_from_block.items())

def merge_blocks_data(blocks_data: List[Tuple[int, Pose]], floor_z: float=0.0) -> List[Tuple[Pose, Tuple[float, float, float]]]:
    blocks_info: List[Tuple[Pose, Tuple[float, float, float]]] = []
    for block_id, block_data in itertools.groupby(sorted(blocks_data, key=lambda x: x[0]), lambda x: x[0]):
        _, poses = zip(*block_data)
        position = np.mean([pose.position for pose in poses], axis=0)
        orientation = np.mean([pose.orientation for pose in poses], axis=0)
        orientation /= np.linalg.norm(orientation)
        blocks_info.append(adjust_block(Pose(position, orientation), block_id_to_dimensions[block_id], floor_z)) # type: ignore
    return blocks_info

def get_task_data(fname: str) -> Pose:
    data = json.load(open(os.path.join(asset_path, fname)))
    joint_angles = data['parameter']['poses'][0]['relative_trajectories'][0][0]['joint_angles']
    return get_gripper_pose(joint_angles)

def get_gripper_pose(joint_angles):
    CFG.seed = 0
    robot = PandaPyBulletRobot(PyBulletPackingEnv.robot_ee_init_pose, p.connect(p.DIRECT), Pose(PyBulletPackingEnv.robot_base_pos, PyBulletPackingEnv._default_orn))
    return robot.forward_kinematics(joint_angles + robot.get_joints()[7:])

if __name__ == "__main__":
    _, img_data_pkl, arg = sys.argv
    image_data = pkl.load(open(img_data_pkl, 'rb'))

    # image_data = [
    #     (np.array(cv2.imread(os.path.join(asset_path, "color_image_1.tiff"))), get_task_data(os.path.join(asset_path, "BARTEK-GRIPPER1.task"))),
    #     (np.array(cv2.imread(os.path.join(asset_path, "color_image_2.tiff"))), get_task_data(os.path.join(asset_path, "BARTEK-GRIPPER2.task"))),
    # ]

    if arg == '--debug':
        for color_image, _ in image_data:
            draw_debug(color_image)
        cv2.waitKey(10000)
    else:
        blocks_info = merge_blocks_data(
            sum([get_blocks_data(color_image, get_gripper_pose(joint_angles)) for color_image, joint_angles in image_data], start=[]), # type: ignore
        )
        pkl.dump(blocks_info, open(arg, 'wb'))