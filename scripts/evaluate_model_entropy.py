"""Script to evaluate learned predicate classifiers on held-out test cases."""

from typing import Sequence

import numpy as np

from predicators.scripts.evaluate_predicate_classifiers import main
from predicators.src import utils
from predicators.src.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Object, State


def evaluate_pred_ensemble(env: BaseEnv,
                           approach: InteractiveLearningApproach) -> None:
    """Prints entropy and BALD scores of predicate classifier ensembles."""
    if CFG.env == "cover":
        return _evaluate_pred_ensemble_cover(env, approach)
    raise NotImplementedError(
        f"Held out predicate test set not yet implemented for {CFG.env}")


def _evaluate_pred_ensemble_cover(
        env: BaseEnv, approach: InteractiveLearningApproach) -> None:
    preds = approach._get_current_predicates()  # pylint: disable=protected-access
    Covers = [p for p in preds if p.name == "Covers"][0]
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
    # Test 1: block and target far apart
    _calculate(approach, state, Covers.name, [block0, target1])
    # Test 2: block and target closer together but no overlap
    _calculate(approach, state, Covers.name, [block0, target0])
    # Test 3: block and target overlap a bit
    state.set(block0, "pose", 0.31)
    _calculate(approach, state, Covers.name, [block0, target0])
    # Test 4: block and target overlap more
    state.set(block0, "pose", 0.33)
    _calculate(approach, state, Covers.name, [block0, target0])
    # Test 5: block covers target and right edges align
    state.set(block0, "pose", 0.35)
    _calculate(approach, state, Covers.name, [block0, target0])
    # Test 6: block covers target, centered
    state.set(block0, "pose", target_poses[0])
    _calculate(approach, state, Covers.name, [block0, target0])


def _calculate(approach: InteractiveLearningApproach, state: State,
               pred_name: str, objects: Sequence[Object]) -> None:
    x = state.vec(objects)
    ps = approach._pred_to_ensemble[pred_name].predict_member_probas(x)  # pylint: disable=protected-access
    entropy = utils.entropy(np.mean(ps))
    bald_score = entropy - np.mean([utils.entropy(p) for p in ps])
    print(f"Entropy: {entropy}, BALD score: {bald_score}")


if __name__ == "__main__":
    main(evaluate_pred_ensemble)
