import sys
import pickle as pkl
import numpy as np
from predicators.pybullet_helpers.geometry import Pose

_, in_file, out_file = sys.argv

extrinsics_pose = Pose(*pkl.load(open("experiments/envs/pybullet_packing/assets/extrinsics.pkl", 'rb')))
dx, dy = np.array(extrinsics_pose.invert().position)[:2]

block_info = pkl.load(open(in_file, 'rb'))
positions = [pose.position for pose, _ in block_info]
coords = [(x, y) for x, y, _ in positions]
pkl.dump((0.2, True, sorted(coords, key=lambda v: np.arctan2(v[1], v[0]))), open(out_file, 'wb'))




