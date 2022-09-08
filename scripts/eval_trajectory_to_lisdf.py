"""Script for converting a saved eval trajectory into a List[JointSpacePath].

First run the pipeline to collect a trajectory. E.g.:

    python predicators/main.py --approach oracle --env pybullet_blocks \
        --seed 0 --num_test_tasks 1

Then run this script:

    python scripts/eval_trajectory_to_lisdf.py \
        --input eval_trajectories/pybullet_blocks__oracle__0________task1.traj \
        --output /tmp/pybullet_blocks__oracle__0________task1.jsp
"""
import argparse

import dill as pkl


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        type=str,
                        help="File path with a saved eval trajectory",
                        required=True)
    parser.add_argument("--output",
                        type=str,
                        help="File path to save the JointSpacePath",
                        required=True)
    parser.add_argument("--pybullet_robot",
                        type=str,
                        help="File path to save the JointSpacePath",
                        required=True)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        traj_data = pkl.load(f)

    ll_traj = traj_data["trajectory"]

    # For now, we just convert the trajectory into one big JointSpacePath. In
    # the future, we may want to chunk it up.


if __name__ == "__main__":
    _main()
