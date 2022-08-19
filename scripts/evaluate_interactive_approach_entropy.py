"""Script to calculate entropies and BALD scores of interactively learned
predicate classifier ensembles in certain states."""

from typing import List, Optional, Sequence

import numpy as np

from scripts.evaluate_interactive_approach_classifiers import \
    create_states_cover, evaluate_approach
from predicators import utils
from predicators.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.envs import BaseEnv
from predicators.envs.cover import CoverEnv
from predicators.settings import CFG
from predicators.structs import Object, State


def evaluate_pred_ensemble(env: BaseEnv, approach: InteractiveLearningApproach,
                           cycle_num: Optional[int], data: List) -> None:
    """Prints entropy and BALD scores of predicate classifier ensembles."""
    if CFG.env == "cover":
        assert isinstance(env, CoverEnv)
        return _evaluate_pred_ensemble_cover(env, approach, cycle_num, data)
    raise NotImplementedError(
        f"Held out predicate test set not yet implemented for {CFG.env}")


def _evaluate_pred_ensemble_cover(env: CoverEnv,
                                  approach: InteractiveLearningApproach,
                                  cycle_num: Optional[int],
                                  data: List) -> None:
    preds = approach._get_current_predicates()  # pylint: disable=protected-access
    Covers = [p for p in preds if p.name == "Covers"][0]
    states, blocks, targets = create_states_cover(env)
    assert len(blocks) == 2
    assert len(targets) == 2
    block0, _ = blocks
    target0, target1 = targets
    # Test 0: block and target far apart
    # Test 1: block and target closer together but no overlap
    # Test 2: block and target overlap a bit
    # Test 3: block and target overlap more
    # Test 4: block covers target and right edges align
    # Test 5: block covers target, centered
    test_states = [
        states[0], states[0], states[1], states[2], states[3], states[4]
    ]
    test_objs = [[block0, target1]]
    test_objs.extend([[block0, target0] for _ in range(5)])
    _calculate(approach, test_states, Covers.name, test_objs, cycle_num, data)


def _calculate(approach: InteractiveLearningApproach, states: List[State],
               pred_name: str, objects_lst: Sequence[Sequence[Object]],
               cycle_num: Optional[int], data: List) -> None:
    assert len(states) == len(objects_lst)
    for i in range(len(states)):
        state = states[i]
        x = state.vec(objects_lst[i])
        ps = approach._pred_to_ensemble[pred_name].predict_member_probas(x)  # pylint: disable=protected-access
        entropy = utils.entropy(float(np.mean(ps)))
        mean_entropy = float(np.mean([utils.entropy(p) for p in ps]))
        bald_score = entropy - mean_entropy
        print(f"Entropy: {entropy}, BALD score: {bald_score}")
        info = {
            "TEST_ID": f"entropy_{i}",
            "CYCLE": cycle_num,
            "SCORE": entropy
        }
        data.append(info)
        info = {
            "TEST_ID": f"BALD_{i}",
            "CYCLE": cycle_num,
            "SCORE": bald_score
        }
        data.append(info)


if __name__ == "__main__":
    evaluate_approach(evaluate_pred_ensemble, [])
