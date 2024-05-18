import sys
import rospy
from franka_interface import ArmInterface, GripperInterface
import pickle as pkl

# Constants
gripper_open_thresh = 0.035
joint_angle_threshold = 0.1
joint_speed = 0.25

# Initializing the arm controls
rospy.init_node("run_trajectory_node")
arm = ArmInterface()
gripper = GripperInterface()

def generate_joints_dict(joint_positions):
	return {f'panda_joint{idx + 1}': joint_position for idx, joint_position in enumerate(joint_positions)}

def execute_rrt_path(arm, position_path, timeout=5.0,
                threshold=0.00085, test=None):
    if len(position_path) <= 1:
        return
    #assert len(position_path) > 40
    # Compute the timing using joint velocity limits, speed_ratio and min_traj_dur
    # Start at the second waypoint because robot is already at first waypoint
    min_traj_dur = 0.05
    time_so_far = 0
    total_times = [0]
    interval_lengths = [0]
    for q_prev, q in zip(position_path[:-1], position_path[1:]):
        dur = []
        for j in range(len(arm._joint_names)):
            dur.append(max(abs(q[arm._joint_names[j]] - q_prev[arm._joint_names[j]]) / arm._joint_limits.velocity[j], min_traj_dur))
        interval = max(dur)/arm._speed_ratio
        interval_lengths.append(interval)
        time_so_far += interval
        total_times.append(time_so_far)
    arm.execute_position_trajectory(position_path, interval_lengths, max(timeout, time_so_far), threshold, test)
    

if __name__ == "__main__":
	_, in_file = sys.argv[:2]
	if '--fast' in sys.argv[2:]:
		arm.set_joint_position_speed(0.3)
	else:
		arm.set_joint_position_speed(0.15)
	trajectories = pkl.load(open(in_file, 'rb'))

	# Initializing the arm position
	arm.move_to_joint_positions(arm._neutral_pose_joints, threshold=joint_angle_threshold)
	
	# Initializing the gripper
	gripper.open()
	gripper.close()
	gripper.open()
	current_gripper_open = True

	# Running the trajectories
	for idx, (trajectory, gripper_open) in enumerate(trajectories):
		position_path = [generate_joints_dict(joint_positions) for joint_positions in trajectory]
		if idx == 0:
			position_path = [arm.joint_angles()] + position_path
		print(len(position_path))
		if len(position_path) <= 40:
			execute_rrt_path(arm, position_path, timeout=3)
		else:
			execute_rrt_path(arm, position_path, timeout=3)
		if not current_gripper_open and gripper_open:
			gripper.open()
		if current_gripper_open and not gripper_open:
			gripper.grasp(0.035, 25, epsilon_outer=0.04) # 50 N of grip force
		current_gripper_open = gripper_open

