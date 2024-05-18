import sys
import pickle as pkl
import numpy as np
from experiments.envs.pybullet_packing.env import PyBulletPackingEnv
from predicators.settings import CFG

# Constants
gripper_open_thresh = 0.039

# Helper Functions
def is_gripper_open(joint_positions):
	global gripper_open_thresh
	print(joint_positions[-2:])
	left_finger = joint_positions[-2] > gripper_open_thresh
	right_finger = joint_positions[-1] > gripper_open_thresh
	assert not (left_finger ^ right_finger)
	return left_finger

# Parsing args and setup
## Setting predicators stuff
CFG.seed = 0
print(CFG.pybullet_birrt_smooth_amt)

## Parsing args
_, in_fname, out_fname = sys.argv

## Loading the input trajectory
data = pkl.load(open(in_fname, 'rb'))
states, _ = data['trajectory']
assert states

# Calculating the trajectories
trajectories = [([states[0].joint_positions[:7]], is_gripper_open(states[0].joint_positions))]
for state, next_state in zip(states, states[1:]):
	robot_moved = np.allclose(state.joint_positions[:7], next_state.joint_positions[:7])

	trajectory = []
	if not np.allclose(state.joint_positions[:7], next_state.joint_positions[:7]):
		trajectory = PyBulletPackingEnv.run_motion_planning(state, next_state.joint_positions, use_gui=True)
	trajectories.append(([joint_positions[:7] for joint_positions in trajectory], is_gripper_open(next_state.joint_positions)))
	print("CONVERTED TRAJECTORY", is_gripper_open(next_state.joint_positions))
	print(trajectory)
	print(len(trajectory))

pkl.dump(trajectories, open(out_fname, 'wb'))
