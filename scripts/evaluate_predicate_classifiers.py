"""Script to evaluate learned predicate classifiers on held-out test cases."""

from typing import Callable, List, Set

import numpy as np

from predicators.src import utils
from predicators.src.approaches import create_approach
from predicators.src.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.src.envs import BaseEnv, create_new_env
from predicators.src.settings import CFG
from predicators.src.structs import Action, Predicate, Task


def main(
    evaluate_fn: Callable[[BaseEnv, InteractiveLearningApproach],
                          None]) -> None:
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    # Create classes.
    env = create_new_env(CFG.env, do_cache=True)
    preds, _ = utils.parse_config_excluded_predicates(env)
    # Don't need actual train tasks.
    train_tasks: List[Task] = []
    # Create the agent (approach).
    approach = create_approach(CFG.approach, preds, env.options, env.types,
                               env.action_space, train_tasks)
    assert approach.is_learning_based
    assert isinstance(approach, InteractiveLearningApproach)
    _run_pipeline(env, approach, evaluate_fn)


def _run_pipeline(
    env: BaseEnv, approach: InteractiveLearningApproach,
    evaluate_fn: Callable[[BaseEnv, InteractiveLearningApproach],
                          None]) -> None:
    approach.load(online_learning_cycle=None)
    evaluate_fn(env, approach)
    for i in range(CFG.num_online_learning_cycles):
        print(f"\n\nONLINE LEARNING CYCLE {i}\n")
        try:
            approach.load(online_learning_cycle=i)
            evaluate_fn(env, approach)
        except FileNotFoundError:
            break


def _evaluate_preds(env: BaseEnv,
                    approach: InteractiveLearningApproach) -> None:
    if CFG.env == "cover":
        return _evaluate_preds_cover(
            approach._get_current_predicates(),  # pylint: disable=protected-access
            env)
    raise NotImplementedError(
        f"Held out predicate test set not yet implemented for {CFG.env}")


def _evaluate_preds_cover(preds: Set[Predicate], env: BaseEnv) -> None:
    Holding = [p for p in preds if p.name == "Holding"][0]
    Covers = [p for p in preds if p.name == "Covers"][0]
    HoldingGT = [p for p in env.predicates if p.name == "Holding"][0]
    CoversGT = [p for p in env.predicates if p.name == "Covers"][0]
    # Create initial state
    task = env.get_test_tasks()[0]
    state = task.init
    block0 = [b for b in state if b.name == "block0"][0]
    block1 = [b for b in state if b.name == "block1"][0]
    target0 = [b for b in state if b.name == "target0"][0]
    target1 = [b for b in state if b.name == "target1"][0]
    robot = [b for b in state if b.name == "robby"][0]
    blocks = [block0, block1]
    targets = [target0, target1]
    block_poses = [0.15, 0.605]
    target_poses = [0.375, 0.815]
    for block, pose in zip(blocks, block_poses):
        # [is_block, is_target, width, pose, grasp]
        state.set(block, "pose", pose)
        # Make sure blocks are not held
        state.set(block, "grasp", -1)
    for target, pose in zip(targets, target_poses):
        # [is_block, is_target, width, pose]
        state.set(target, "pose", pose)
    state.set(robot, "hand", 0.0)
    # Test 1: no blocks overlap any targets, none are held
    atoms = utils.abstract(state, (Holding, Covers))
    atoms_gt = utils.abstract(state, (HoldingGT, CoversGT))
    print(f"False positives: {atoms - atoms_gt}\n"
          f"False negatives: {atoms_gt - atoms}")
    # Pick up block0 and place it over target0
    action = Action(np.array([block_poses[0]], dtype=np.float32))
    state = env.simulate(state, action)
    action = Action(np.array([target_poses[0]], dtype=np.float32))
    state = env.simulate(state, action)
    # Test 2: block0 covers target0
    atoms = utils.abstract(state, (Holding, Covers))
    atoms_gt = utils.abstract(state, (HoldingGT, CoversGT))
    print(f"False positives: {atoms - atoms_gt}\n"
          f"False negatives: {atoms_gt - atoms}")
    # Pick and place block1 so it partially overlaps target1
    action = Action(np.array([0.63], dtype=np.float32))
    state = env.simulate(state, action)
    action = Action(np.array([target_poses[1]], dtype=np.float32))
    state = env.simulate(state, action)
    # Test 3: block1 does not completely cover target1
    atoms = utils.abstract(state, (Holding, Covers))
    atoms_gt = utils.abstract(state, (HoldingGT, CoversGT))
    print(f"False positives: {atoms - atoms_gt}\n"
          f"False negatives: {atoms_gt - atoms}")


if __name__ == "__main__":
    main(_evaluate_preds)
