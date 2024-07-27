"""Tests for methods in the BaseSTRIPSLearner class."""

from predicators import utils
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning.llm_strips_learner import \
    LLMStripsLearner
from predicators.pretrained_model_interface import LargeLanguageModel


class _DummyLLM(LargeLanguageModel):

    def get_id(self):
        return "dummy"

    def _sample_completions(self,
                            prompt,
                            imgs,
                            temperature,
                            seed,
                            stop_token=None,
                            num_completions=1):
        del imgs  # unused.
        completions = []
        for _ in range(num_completions):
            # Hardcoded completion for 'cover'.
            completion = """```
Operators:
PickPlaceOp1(?b - block ?t - target)
preconditions: (and (Holding ?b) (IsTarget ?t))
add effects: (and (Covers ?b ?t) (HandEmpty))
delete effects: (and (Holding ?b))
action: PickPlace()

PickPlaceOp2(?b - block)
preconditions: (and (HandEmpty) (IsBlock ?b))
add effects: (and (Holding ?b))
delete effects: (and (HandEmpty))
action: PickPlace()
```
"""
            completions.append(completion)
        return completions


def test_llm_op_learning():
    """Test the operator learning approach using a dummy llm."""
    utils.reset_config({
        "env": "cover",
        "strips_learner": "llm",
        "num_train_tasks": 3
    })
    env = create_new_env("cover")
    train_tasks = [t.task for t in env.get_train_tasks()]
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                             predicates)
    segmented_trajs = [
        segment_trajectory(t, env.predicates) for t in dataset.trajectories
    ]
    learner = LLMStripsLearner(dataset.trajectories,
                               train_tasks,
                               env.predicates,
                               segmented_trajs,
                               verify_harmlessness=True,
                               annotations=None)
    learner._llm = _DummyLLM()  # pylint:disable=protected-access
    pnads = learner._learn()  # pylint:disable=protected-access
    assert len(pnads) == 2
