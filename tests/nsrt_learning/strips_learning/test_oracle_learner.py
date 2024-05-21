"""Tests for oracle STRIPS operator learning."""
from predicators import utils
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.structs import Dataset, LowLevelTrajectory


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
                                   verify_harmlessness=True,
                                   annotations=None)
    assert not pnads
    # With sufficiently representative data, all operators should be learned.
    env = create_new_env("blocks")
    train_tasks = [t.task for t in env.get_train_tasks()]
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                             predicates)
    segmented_trajs = [
        segment_trajectory(t, env.predicates) for t in dataset.trajectories
    ]
    pnads = learn_strips_operators(dataset.trajectories,
                                   train_tasks,
                                   env.predicates,
                                   segmented_trajs,
                                   verify_harmlessness=True,
                                   annotations=None)
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
    train_tasks = [t.task for t in env.get_train_tasks()]
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, set(), predicates)
    segmented_trajs = [
        segment_trajectory(t, env.predicates) for t in dataset.trajectories
    ]
    pnads = learn_strips_operators(dataset.trajectories,
                                   train_tasks,
                                   env.predicates,
                                   segmented_trajs,
                                   verify_harmlessness=True,
                                   annotations=None)
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
    # Test case where the data includes a trajectory with a single option that
    # never terminates. The expected behavior is that no NSRTs are learned.
    utils.reset_config({
        "env": "cover_multistep_options",
        "strips_learner": "oracle",
        "num_train_tasks": 1,
    })
    env = create_new_env("cover_multistep_options")
    train_tasks = [t.task for t in env.get_train_tasks()]
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                             predicates)
    # Truncate dataset.
    state, next_state = dataset.trajectories[0].states[:2]
    action = dataset.trajectories[0].actions[0]
    assert not action.get_option().terminal(next_state)
    truncated_traj = LowLevelTrajectory([state, next_state], [action])
    dataset = Dataset([truncated_traj])
    segmented_trajs = [
        segment_trajectory(t, env.predicates) for t in dataset.trajectories
    ]
    assert len(segmented_trajs[0]) == 0
    pnads = learn_strips_operators(dataset.trajectories,
                                   train_tasks,
                                   env.predicates,
                                   segmented_trajs,
                                   verify_harmlessness=True,
                                   annotations=None)
    assert not pnads
    # Test the same case, but where the option appears to have terminated due
    # to the max horizon being exceeded. In this case, there should be one
    # segment kept.
    utils.reset_config({
        "env": "cover_multistep_options",
        "strips_learner": "oracle",
        "num_train_tasks": 1,
        "max_num_steps_option_rollout": 1,
    })
    segmented_trajs = [
        segment_trajectory(t, env.predicates) for t in dataset.trajectories
    ]
    assert len(segmented_trajs[0]) == 1  # was 0 before
