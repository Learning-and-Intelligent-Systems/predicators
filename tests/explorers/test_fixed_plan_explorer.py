"""Test cases for the fixed plan explorer class."""
import pytest

from predicators import utils
from predicators.envs.cover import CoverEnv
from predicators.explorers import create_explorer
from predicators.explorers.fixed_plan_explorer import _parse_plan
from predicators.ground_truth_models import get_gt_options


def _write_plan(tmp_path, text):
    path = tmp_path / "plan.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_parse_plan_matches_replay_plans_format():
    """The explorer reads what probe_real_scene --dump-plan writes, so a plan
    verified through replay_plan can be handed straight to the loop.

    That includes the '#' header those files carry and the '-> {...}'
    subgoal tail a solved plan may carry.
    """
    steps = _parse_plan("# scene : whatever.json\n"
                        "# solved : False\n"
                        "\n"
                        "Pick(robot:robot, domino_1:domino)[0.0657]\n"
                        "Place(robot:robot)[0.70, 1.16] -> {Holding(x)}\n"
                        "Wait(robot:robot)[]\n")
    assert steps == [
        ("Pick", ["robot", "domino_1"], [0.0657]),
        ("Place", ["robot"], [0.70, 1.16]),
        ("Wait", ["robot"], []),
    ]


def test_fixed_plan_explorer_replays_the_file(tmp_path):
    """The explorer's policy executes the plan on the file, not a search."""
    # cover's only option is PickPlace(), one param and no objects.
    plan_path = _write_plan(tmp_path, "PickPlace()[0.75]\n")
    utils.reset_config({
        "env": "cover",
        "seed": 0,
        "explorer": "fixed_plan",
        "fixed_plan_explorer_path": plan_path,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    explorer = create_explorer("fixed_plan", env.predicates,
                               get_gt_options("cover"), env.types,
                               env.action_space, train_tasks)
    policy, termination_function = explorer.get_exploration_strategy(0, 500)

    # Never terminates on its own; the plan running out is what ends it.
    assert not termination_function(train_tasks[0].init)
    act = policy(train_tasks[0].init)
    assert env.action_space.contains(act.arr)
    assert act.get_option().name == "PickPlace"


def test_fixed_plan_explorer_needs_a_path():
    """A missing path fails loudly rather than exploring some other way."""
    utils.reset_config({
        "env": "cover",
        "seed": 0,
        "explorer": "fixed_plan",
        "fixed_plan_explorer_path": "",
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    explorer = create_explorer("fixed_plan", env.predicates,
                               get_gt_options("cover"), env.types,
                               env.action_space, train_tasks)
    with pytest.raises(AssertionError, match="fixed_plan_explorer_path"):
        explorer.get_exploration_strategy(0, 500)
