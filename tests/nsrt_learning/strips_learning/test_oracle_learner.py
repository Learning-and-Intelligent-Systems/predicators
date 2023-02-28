"""Tests for oracle STRIPS operator learning."""
from predicators import utils
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import learn_strips_operators


def test_oracle_strips_learner():
    """Tests for OracleSTRIPSLearner."""
    utils.reset_config({
        "env": "blocks",
        "strips_learner": "oracle",
        "num_train_tasks": 3
    })
    # Without any data, no operators should be learned.
    pnads = learn_strips_operators([],
                                   None,
                                   None, [],
                                   verify_harmlessness=True)
    assert not pnads
    # With sufficiently representative data, all operators should be learned.
    env = create_new_env("blocks")
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
    segmented_trajs = [
        segment_trajectory(traj) for traj in ground_atom_dataset
    ]
    pnads = learn_strips_operators(dataset.trajectories,
                                   train_tasks,
                                   env.predicates,
                                   segmented_trajs,
                                   verify_harmlessness=True)
    assert str(sorted(pnads, key=str)) == """[STRIPS-PickFromTable:
    Parameters: [?block:block, ?robot:robot]
    Preconditions: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Add Effects: [Holding(?block:block)]
    Delete Effects: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Ignore Effects: []
    Option Spec: Pick(?robot:robot, ?block:block), STRIPS-PutOnTable:
    Parameters: [?block:block, ?robot:robot]
    Preconditions: [Holding(?block:block)]
    Add Effects: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Delete Effects: [Holding(?block:block)]
    Ignore Effects: []
    Option Spec: PutOnTable(?robot:robot), STRIPS-Stack:
    Parameters: [?block:block, ?otherblock:block, ?robot:robot]
    Preconditions: [Clear(?otherblock:block), Holding(?block:block)]
    Add Effects: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Delete Effects: [Clear(?otherblock:block), Holding(?block:block)]
    Ignore Effects: []
    Option Spec: Stack(?robot:robot, ?otherblock:block), STRIPS-Unstack:
    Parameters: [?block:block, ?otherblock:block, ?robot:robot]
    Preconditions: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Add Effects: [Clear(?otherblock:block), Holding(?block:block)]
    Delete Effects: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Ignore Effects: []
    Option Spec: Pick(?robot:robot, ?block:block)]"""  # pylint: disable=line-too-long
    # Test with unknown options. Expected behavior is that the operators should
    # be identical, except the option specs will be dummies.
    utils.reset_config({
        "env": "blocks",
        "strips_learner": "oracle",
        "option_learner": "oracle",
        "num_train_tasks": 3,
        "segmenter": "atom_changes",
    })
    env = create_new_env("blocks")
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, set())
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
    segmented_trajs = [
        segment_trajectory(traj) for traj in ground_atom_dataset
    ]
    pnads = learn_strips_operators(dataset.trajectories,
                                   train_tasks,
                                   env.predicates,
                                   segmented_trajs,
                                   verify_harmlessness=True)
    assert str(sorted(pnads, key=str)) == """[STRIPS-PickFromTable:
    Parameters: [?block:block, ?robot:robot]
    Preconditions: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Add Effects: [Holding(?block:block)]
    Delete Effects: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Ignore Effects: []
    Option Spec: DummyOption(), STRIPS-PutOnTable:
    Parameters: [?block:block, ?robot:robot]
    Preconditions: [Holding(?block:block)]
    Add Effects: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Delete Effects: [Holding(?block:block)]
    Ignore Effects: []
    Option Spec: DummyOption(), STRIPS-Stack:
    Parameters: [?block:block, ?otherblock:block, ?robot:robot]
    Preconditions: [Clear(?otherblock:block), Holding(?block:block)]
    Add Effects: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Delete Effects: [Clear(?otherblock:block), Holding(?block:block)]
    Ignore Effects: []
    Option Spec: DummyOption(), STRIPS-Unstack:
    Parameters: [?block:block, ?otherblock:block, ?robot:robot]
    Preconditions: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Add Effects: [Clear(?otherblock:block), Holding(?block:block)]
    Delete Effects: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Ignore Effects: []
    Option Spec: DummyOption()]"""  # pylint: disable=line-too-long
