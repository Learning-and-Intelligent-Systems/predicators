"""Test cases for the option-renaming open-loop LLM approach."""
from predicators import utils
from predicators.approaches.llm_option_renaming_approach import \
    LLMOptionRenamingApproach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options


def test_llm_option_renaming_approach():
    """Tests for LLMOptionRenamingApproach()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "llm_option_renaming",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
    })
    env = create_new_env(env_name)
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = LLMOptionRenamingApproach(env.predicates,
                                         get_gt_options(env.get_name()),
                                         env.types, env.action_space,
                                         train_tasks)
    assert approach.get_name() == "llm_option_renaming"
    assert approach._renaming_prefixes == [" ", "\n"]  # pylint: disable=protected-access
    assert approach._renaming_suffixes == ["("]  # pylint: disable=protected-access
    subs = approach._orig_to_replace  # pylint: disable=protected-access
    assert set(subs) == {o.name for o in get_gt_options(env.get_name())}
    assert all(len(k) == len(v) for k, v in subs.items())
