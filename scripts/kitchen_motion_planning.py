"""Testing the feasibility of motion planning in kitchen environment."""

from typing import Iterator

import numpy as np

from predicators import utils
from predicators.envs import create_new_env
from predicators.structs import Array


def _main() -> None:
    utils.reset_config({
        "env": "kitchen",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
    })
    use_gui = True
    env = create_new_env("kitchen", use_gui=use_gui)
    gym_env = env._gym_env
    init_obs = env.reset("train", 0)
    target = np.add(init_obs["state_info"]["knob3_site"], [0.0, -0.1, 0.0])
    target_eps = 0.01
    start = gym_env.get_robot_joint_position()

    cspace = gym_env.robot.robot_pos_bound[:7]
    sample_lo, sample_hi = cspace.T

    # Set up RRT in 7 DOF space.
    rng = np.random.default_rng(0)
    action_magnitude = 0.1
    num_attempts = 10
    num_iters = 250
    smooth_amt = 0  # TODO

    def _sample_fn(_: Array) -> Array:
        return rng.uniform(low=sample_lo, high=sample_hi)

    def _extend_fn(pt1: Array, pt2: Array) -> Iterator[Array]:
        distance = np.linalg.norm(pt2 - pt1)
        num = int(distance / action_magnitude) + 1
        for i in range(1, num + 1):
            yield pt1 * (1 - i / num) + pt2 * i / num

    def _collision_fn(pt: Array) -> bool:
        return False  # TODO

    def _distance_fn(pt1: Array, pt2: Array) -> float:
        return np.sum(np.subtract(pt2, pt1)**2)
    
    def _sample_goal_fn() -> Array:
        raise NotImplementedError("IK not implemented.")
    
    def _check_goal_fn(pt: Array) -> bool:
        gym_env.set_robot_joint_position(pt)
        ee_pose = gym_env.get_ee_pose()
        print(ee_pose)
        goal_reached = np.sum(np.subtract(ee_pose, target)**2) < target_eps
        if goal_reached:
            if use_gui:
                gym_env.render()
            import ipdb; ipdb.set_trace()
        return goal_reached
    

    rrt = utils.RRT(_sample_fn, _extend_fn, _collision_fn, _distance_fn, rng,
                    num_attempts, num_iters, smooth_amt)
    
    result = rrt.query_to_goal_fn(start, _sample_goal_fn, _check_goal_fn)
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    _main()
