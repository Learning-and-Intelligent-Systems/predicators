"""Test cases for the Maple Q approach."""
import pytest

from predicators import utils
from predicators.approaches.maple_q_approach import MapleQApproach
from predicators.cogman import CogMan
from predicators.envs.cover import RegionalBumpyCoverEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.main import _generate_interaction_results
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Dataset, State, Task
from predicators.teacher import Teacher


@pytest.mark.parametrize("cover_num_blocks,cover_num_targets", [(1, 1),
                                                                (2, 2)])
def test_maple_q_approach(cover_num_blocks, cover_num_targets):
    """Test for MapleQApproach class, entire pipeline."""

    cover_block_widths = [0.01 for _ in range(cover_num_blocks)]
    cover_target_widths = [0.008 for _ in range(cover_num_targets)]

    utils.reset_config({
        "env": "regional_bumpy_cover",
        "cover_num_blocks": cover_num_blocks,
        "cover_block_widths": cover_block_widths,
        "cover_num_targets": cover_num_targets,
        "cover_target_widths": cover_target_widths,
        "approach": "maple_q",
        "timeout": 10,
        "strips_learner": "oracle",
        "sampler_learner": "oracle",
        "bilevel_plan_without_sim": True,
        "num_online_learning_cycles": 1,
        "max_num_steps_interaction_request": 5,
        "online_nsrt_learning_requests_per_cycle": 1,
        "sampler_mlp_classifier_max_itr": 10,
        "mlp_regressor_max_itr": 10,
        "num_train_tasks": 3,
        "max_initial_demos": 0,
        "num_test_tasks": 1,
        "explorer": "maple_q",
        "active_sampler_learning_num_samples": 5,
        "active_sampler_learning_num_lookahead_samples": 2,
    })
    env = RegionalBumpyCoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = get_gt_options(env.get_name())
    approach = MapleQApproach(env.predicates, options, env.types,
                              env.action_space, train_tasks)
    assert approach.is_learning_based
    assert approach.get_name() == "maple_q"
    approach.learn_from_offline_dataset(Dataset([]))
    approach.load(online_learning_cycle=None)
    interaction_requests = approach.get_interaction_requests()
    teacher = Teacher(train_tasks)
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    interaction_results, _ = _generate_interaction_results(
        cogman, env, teacher, interaction_requests)
    approach.learn_from_interaction_results(interaction_results)
    approach.load(online_learning_cycle=0)
    with pytest.raises(FileNotFoundError):
        approach.load(online_learning_cycle=1)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=CFG.timeout)
        # We won't fully check the policy here because we don't want
        # tests to have to train very good models, since that would
        # be slow. But we will test that the policy at least produces
        # an action.
        action = policy(task.init)
        assert env.action_space.contains(action.arr)

    # Test case where a task is presented with only a subset of objects.
    if cover_num_blocks == 2:
        state = train_tasks[0].init
        goal = train_tasks[0].goal
        assert len(goal) == 1
        _, robot_type, _ = sorted(env.types)
        assert len(goal) == 1
        b, t = next(iter(goal)).objects
        r, = state.get_objects(robot_type)
        init_state = State({b: state[b], t: state[t], r: state[r]})
        new_task = Task(init_state, goal)
        # Policy should not crash.
        policy = approach.solve(new_task, timeout=CFG.timeout)
        action = policy(new_task.init)
        assert env.action_space.contains(action.arr)
