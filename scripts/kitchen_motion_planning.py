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
    env = create_new_env("kitchen", use_gui=False)
    gym_env = env._gym_env

    # Set up RRT in 7 DOF space.
    rng = np.random.default_rng(0)
    action_magnitude = 0.1
    num_attempts = 10
    num_iters = 100
    smooth_amt = 0  # TODO

    def _sample_fn(_: Array) -> Array:
        import ipdb
        ipdb.set_trace()

    def _extend_fn(pt1: Array, pt2: Array) -> Iterator[Array]:
        distance = np.linalg.norm(pt2 - pt1)
        num = int(distance / action_magnitude) + 1
        for i in range(1, num + 1):
            yield pt1 * (1 - i / num) + pt2 * i / num

    def _collision_fn(pt: Array) -> bool:
        return False  # TODO

    def _distance_fn(pt1: Array, pt2: Array) -> float:
        return np.sum(np.subtract(pt2, pt1)**2)

    rrt = utils.RRT(_sample_fn, _extend_fn, _collision_fn, _distance_fn, rng,
                    num_attempts, num_iters, smooth_amt)


if __name__ == "__main__":
    _main()
