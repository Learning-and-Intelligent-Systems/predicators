"""Test cases for planning algorithms.
"""

from predicators.src.approaches.oracle_approach import get_gt_ops
from predicators.src.envs import CoverEnv
from predicators.src.planning import sesame_plan
from predicators.src import utils


def test_sesame_plan():
    """Tests for sesame_plan().
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    operators = get_gt_ops(env.predicates, env.options)
    task = env.get_train_tasks()[0]
    plan = sesame_plan(task, env.simulate, operators,
                       env.predicates, 1, 123)
    assert len(plan) == 2
