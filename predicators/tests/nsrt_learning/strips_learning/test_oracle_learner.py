"""Tests for oracle STRIPS operator learning."""

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import \
    learn_strips_operators


def test_oracle_strips_learner():
    """Tests for OracleSTRIPSLearner."""
    utils.reset_config({"env": "blocks", "strips_learner": "oracle"})
    pnads = learn_strips_operators([],
                                   None,
                                   None, [],
                                   verify_harmlessness=True)
    assert str(sorted(pnads, key=str)) == """[STRIPS-PickFromTable:
    Parameters: [?block:block, ?robot:robot]
    Preconditions: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Add Effects: [Holding(?block:block)]
    Delete Effects: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Side Predicates: []
    Option Spec: Pick(?robot:robot, ?block:block), STRIPS-PutOnTable:
    Parameters: [?block:block, ?robot:robot]
    Preconditions: [Holding(?block:block)]
    Add Effects: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Delete Effects: [Holding(?block:block)]
    Side Predicates: []
    Option Spec: PutOnTable(?robot:robot), STRIPS-Stack:
    Parameters: [?block:block, ?otherblock:block, ?robot:robot]
    Preconditions: [Clear(?otherblock:block), Holding(?block:block)]
    Add Effects: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Delete Effects: [Clear(?otherblock:block), Holding(?block:block)]
    Side Predicates: []
    Option Spec: Stack(?robot:robot, ?otherblock:block), STRIPS-Unstack:
    Parameters: [?block:block, ?otherblock:block, ?robot:robot]
    Preconditions: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Add Effects: [Clear(?otherblock:block), Holding(?block:block)]
    Delete Effects: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Side Predicates: []
    Option Spec: Pick(?robot:robot, ?block:block)]"""  # pylint: disable=line-too-long
    # Test with unknown options. Expected behavior is that the operators should
    # be identical, except the option specs will be dummies.
    utils.reset_config({
        "env": "blocks",
        "strips_learner": "oracle",
        "option_learner": "oracle"
    })
    pnads = learn_strips_operators([],
                                   None,
                                   None, [],
                                   verify_harmlessness=True)
    assert str(sorted(pnads, key=str)) == """[STRIPS-PickFromTable:
    Parameters: [?block:block, ?robot:robot]
    Preconditions: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Add Effects: [Holding(?block:block)]
    Delete Effects: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Side Predicates: []
    Option Spec: DummyOption(), STRIPS-PutOnTable:
    Parameters: [?block:block, ?robot:robot]
    Preconditions: [Holding(?block:block)]
    Add Effects: [Clear(?block:block), GripperOpen(?robot:robot), OnTable(?block:block)]
    Delete Effects: [Holding(?block:block)]
    Side Predicates: []
    Option Spec: DummyOption(), STRIPS-Stack:
    Parameters: [?block:block, ?otherblock:block, ?robot:robot]
    Preconditions: [Clear(?otherblock:block), Holding(?block:block)]
    Add Effects: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Delete Effects: [Clear(?otherblock:block), Holding(?block:block)]
    Side Predicates: []
    Option Spec: DummyOption(), STRIPS-Unstack:
    Parameters: [?block:block, ?otherblock:block, ?robot:robot]
    Preconditions: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Add Effects: [Clear(?otherblock:block), Holding(?block:block)]
    Delete Effects: [Clear(?block:block), GripperOpen(?robot:robot), On(?block:block, ?otherblock:block)]
    Side Predicates: []
    Option Spec: DummyOption()]"""  # pylint: disable=line-too-long
