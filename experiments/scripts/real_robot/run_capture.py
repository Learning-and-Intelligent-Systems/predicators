import sys
import os
import time
import pickle as pkl

import rospy
from franka_interface import ArmInterface, GripperInterface
import pyrealsense2 as rs

import numpy as np

from experiments.envs.pybullet_packing.env import PyBulletPackingEnv
from predicators.pybullet_helpers.geometry import Pose
from predicators.settings import CFG
from predicators.pybullet_helpers.link import get_link_state
import pybullet as p
from predicators.pybullet_helpers.robots.panda import PandaPyBulletRobot

# Constants
gripper_open_thresh = 0.035
rough_joint_angle_threshold = 0.1
precise_joint_angle_threshold = 0.002
joint_speed = 0.2

# Setting up the panda connection
# rospy.init_node("run_capture_node")
# arm = ArmInterface()
# gripper = GripperInterface()

# Setting up IK
CFG.seed = 0
robot = PandaPyBulletRobot(PyBulletPackingEnv.robot_ee_init_pose, p.connect(p.GUI), Pose(PyBulletPackingEnv.robot_base_pos, PyBulletPackingEnv._default_orn))


panda_hand_from_tool_link = get_link_state(
	robot.robot_id,
	robot.link_from_name("panda_hand"),
	physics_client_id = robot.physics_client_id
).pose.invert().multiply(get_link_state(
	robot.robot_id,
	robot.link_from_name("tool_link"),
	physics_client_id = robot.physics_client_id
).pose)
panda_hand_from_camera = Pose(*pkl.load(open("experiments/envs/pybullet_packing/assets/extrinsics.pkl", 'rb')))

neutral_joints = None
def move_to_coords(x, y, z, camera_forward, joint_angle_thresh=0.001):
	assert z >= 0.1
	# To initialize the later IK
	robot.set_joints(neutral_joints + [0.04, 0.04])

	# Proper IK
	robot.go_home()
	# joint_positions = robot.inverse_kinematics(Pose((x, y, z), (0, 1, 0, 0) if camera_forward else (1, 0, 0, 0)), True)
	joint_positions = robot.inverse_kinematics(Pose((x, y, z)).multiply(Pose.from_rpy((0, 0, 0), (0, 0, np.arctan2(y, x)+np.pi/2)), Pose((0, 0, 0), (0, 1, 0, 0) if camera_forward else (1, 0, 0, 0)), panda_hand_from_camera.invert(), panda_hand_from_tool_link), True)
	joint_spec = {f'panda_joint{idx + 1}': joint_position for idx, joint_position in enumerate(joint_positions[:7])}
	# arm.move_to_joint_positions(joint_spec, threshold=joint_angle_thresh)

# Fetching the desired poses
# _, in_file, out_file = sys.argv
_, out_file = sys.argv
# height, camera_forward, capture_positions = pkl.load(open(in_file, 'rb'))

# Moving the arm to neutral
# arm.set_joint_position_speed(joint_speed)
# arm.move_to_joint_positions(arm._neutral_pose_joints, threshold=rough_joint_angle_threshold)
# neutral_joints = arm.joint_ordered_angles()

# Initializing the camera 
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 6)
pipeline.start(config)

frame = pipeline.wait_for_frames()
color_img = np.array(frame.get_color_frame().get_data())
pkl.dump([(color_img, robot.get_joints()[:7])], open(out_file, 'wb'))

# Running data gathering
img_data = []
for x, y in capture_positions:
	if height < 0.5:
		move_to_coords(x, y, 0.5, camera_forward, rough_joint_angle_threshold)
	if height < 0.3:
		move_to_coords(x, y, 0.3, camera_forward, rough_joint_angle_threshold)
	move_to_coords(x, y, height, camera_forward, precise_joint_angle_threshold)
	
	frame = pipeline.wait_for_frames()
	color_img = np.array(frame.get_color_frame().get_data())
	img_data.append((color_img, arm.joint_ordered_angles()))
	
	if height < 0.3:
		move_to_coords(x, y, 0.3, camera_forward, rough_joint_angle_threshold)
	if height < 0.5:
		move_to_coords(x, y, 0.5, camera_forward, rough_joint_angle_threshold)

pkl.dump(img_data, open(out_file, 'wb'))
