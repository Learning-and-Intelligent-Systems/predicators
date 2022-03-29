"""Script to calculate entropies and BALD scores of interactively learned
predicate classifier ensembles in certain states."""

from typing import Sequence, cast

import numpy as np

from predicators.scripts.evaluate_interactive_approach_classifiers import \
    create_states_cover, evaluate_approach
from predicators.src import utils
from predicators.src.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.src.envs import BaseEnv
from predicators.src.envs.cover import CoverEnv
from predicators.src.settings import CFG
from predicators.src.structs import Object, State


def evaluate_pred_ensemble(env: BaseEnv,
                           approach: InteractiveLearningApproach) -> None:
    """Prints entropy and BALD scores of predicate classifier ensembles."""
    if CFG.env == "cover":
        cover_env = cast(CoverEnv, env)
        return _evaluate_pred_ensemble_cover(cover_env, approach)
    raise NotImplementedError(
        f"Held out predicate test set not yet implemented for {CFG.env}")


def _evaluate_pred_ensemble_cover(
        env: CoverEnv, approach: InteractiveLearningApproach) -> None:
    preds = approach._get_current_predicates()  # pylint: disable=protected-access
    Covers = [p for p in preds if p.name == "Covers"][0]
    states, blocks, targets = create_states_cover(env)
    assert len(blocks) == 2
    assert len(targets) == 2
    block0, _ = blocks
    target0, target1 = targets
    # Test 1: block and target far apart
    _calculate(approach, states[0], Covers.name, [block0, target1])
    # Test 2: block and target closer together but no overlap
    _calculate(approach, states[0], Covers.name, [block0, target0])
    # Test 3: block and target overlap a bit
    _calculate(approach, states[1], Covers.name, [block0, target0])
    # Test 4: block and target overlap more
    _calculate(approach, states[2], Covers.name, [block0, target0])
    # Test 5: block covers target and right edges align
    _calculate(approach, states[3], Covers.name, [block0, target0])
    # Test 6: block covers target, centered
    _calculate(approach, states[4], Covers.name, [block0, target0])


def _calculate(approach: InteractiveLearningApproach, state: State,
               pred_name: str, objects: Sequence[Object]) -> None:
    x = state.vec(objects)
    ps = approach._pred_to_ensemble[pred_name].predict_member_probas(x)  # pylint: disable=protected-access
    entropy = utils.entropy(np.mean(ps))
    bald_score = entropy - np.mean([utils.entropy(p) for p in ps])
    print(f"Entropy: {entropy}, BALD score: {bald_score}")


if __name__ == "__main__":
    evaluate_approach(evaluate_pred_ensemble)
