"""Tests for clustering-based STRIPS operator learning.

Note that most of the coverage is provided by
test_nsrt_learning_approach.py, which runs end-to-end tests of the
algorithms on actual domains.
"""

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import \
    learn_strips_operators
from predicators.src.structs import Action, LowLevelTrajectory, Predicate, \
    Segment, State, Task, Type
from predicators.tests.nsrt_learning.test_segmentation import \
    test_segment_trajectory


def test_cluster_and_intersect_strips_learner():
    """Tests for ClusterAndIntersectSTRIPSLearner."""
    known_option_ll_traj, known_option_segments, unknown_option_ll_traj, \
        unknown_option_segments = test_segment_trajectory()
    utils.reset_config({"strips_learner": "cluster_and_intersect"})
    known_option_pnads = learn_strips_operators([known_option_ll_traj],
                                                None,
                                                None, [known_option_segments],
                                                verify_harmlessness=True)
    known_option_ops = [pnad.op for pnad in known_option_pnads]
    assert len(known_option_ops) == 1
    assert str((known_option_ops[0])) == """STRIPS-Op0:
    Parameters: [?x0:cup_type]
    Preconditions: []
    Add Effects: []
    Delete Effects: []
    Side Predicates: []"""
    unknown_option_pnads = learn_strips_operators([unknown_option_ll_traj],
                                                  None,
                                                  None,
                                                  [unknown_option_segments],
                                                  verify_harmlessness=True)
    unknown_option_ops = [pnad.op for pnad in unknown_option_pnads]
    assert len(unknown_option_ops) == 1
    assert str(unknown_option_ops[0]) == """STRIPS-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x0:cup_type), Pred0(?x2:cup_type), Pred1(?x0:cup_type, ?x0:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred1(?x2:cup_type, ?x2:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Side Predicates: []"""  # pylint: disable=line-too-long


def test_cluster_and_search_strips_learner():
    """Tests for ClusterAndSearchSTRIPSLearner."""
    # In this test, we want to highlight the deficiency of the default
    # algorithm, cluster_and_intersect. So, we will first run that and
    # show that it yields suboptimal operators.

    # Set up the data.
    obj_type = Type("obj_type", ["red", "green", "blue", "happy", "sad"])
    obj = obj_type("obj")
    IsRed = Predicate("IsRed", [obj_type], lambda s, o: s[o[0]][0] > 0.5)
    IsGreen = Predicate("IsGreen", [obj_type], lambda s, o: s[o[0]][1] > 0.5)
    IsBlue = Predicate("IsBlue", [obj_type], lambda s, o: s[o[0]][2] > 0.5)
    IsHappy = Predicate("IsHappy", [obj_type], lambda s, o: s[o[0]][3] > 0.5)
    IsSad = Predicate("IsSad", [obj_type], lambda s, o: s[o[0]][4] > 0.5)
    preds = {IsRed, IsGreen, IsBlue, IsHappy, IsSad}
    Interact = utils.SingletonParameterizedOption(
        "Interact", lambda s, m, o, p: None).ground([], [])
    # We give three demonstrations. When the object is red or green, it
    # becomes happy. When the object is blue, it becomes sad.
    s1 = State({obj: [1.0, 0.0, 0.0, 0.0, 0.0]})
    a1 = Action([], Interact)
    ns1 = State({obj: [1.0, 0.0, 0.0, 1.0, 0.0]})
    g1 = {IsHappy([obj])}
    traj1 = LowLevelTrajectory([s1, ns1], [a1], True, 0)
    task1 = Task(s1, g1)
    segment1 = Segment(traj1, utils.abstract(s1, preds),
                       utils.abstract(ns1, preds), Interact)
    s2 = State({obj: [0.0, 1.0, 0.0, 0.0, 0.0]})
    a2 = Action([], Interact)
    ns2 = State({obj: [0.0, 1.0, 0.0, 1.0, 0.0]})
    g2 = {IsHappy([obj])}
    traj2 = LowLevelTrajectory([s2, ns2], [a2], True, 1)
    task2 = Task(s2, g2)
    segment2 = Segment(traj2, utils.abstract(s2, preds),
                       utils.abstract(ns2, preds), Interact)
    s3 = State({obj: [0.0, 0.0, 1.0, 0.0, 0.0]})
    a3 = Action([], Interact)
    ns3 = State({obj: [0.0, 0.0, 1.0, 0.0, 1.0]})
    g3 = {IsSad([obj])}
    traj3 = LowLevelTrajectory([s3, ns3], [a3], True, 2)
    task3 = Task(s3, g3)
    segment3 = Segment(traj3, utils.abstract(s3, preds),
                       utils.abstract(ns3, preds), Interact)

    # Run cluster_and_intersect. The learned PNAD for making the object happy
    # will have trivial preconditions, which is undesirable. Ideally, we want
    # the demonstration that makes the object sad to provide signal for
    # producing two different PNADs for making the object happy, one with
    # IsRed and another with IsGreen as the precondition.
    utils.reset_config({"strips_learner": "cluster_and_intersect"})
    pnads = learn_strips_operators([traj1, traj2, traj3],
                                   [task1, task2, task3],
                                   preds, [[segment1], [segment2], [segment3]],
                                   verify_harmlessness=True)
    assert len(pnads) == 2
    op0, op1 = sorted(pnads, key=str)
    assert str(op0) == """STRIPS-Op0:
    Parameters: [?x0:obj_type]
    Preconditions: []
    Add Effects: [IsHappy(?x0:obj_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Interact()"""
    assert len(op0.datastore) == 2
    assert str(op1) == """STRIPS-Op1:
    Parameters: [?x0:obj_type]
    Preconditions: [IsBlue(?x0:obj_type)]
    Add Effects: [IsSad(?x0:obj_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Interact()"""
    assert len(op1.datastore) == 1

    # Run cluster_and_search. This should produce the desired operators.
    # For this test, we make false positives very costly.
    utils.reset_config({
        "strips_learner": "cluster_and_search",
        "clustering_learner_false_pos_weight": 100
    })
    pnads = learn_strips_operators([traj1, traj2, traj3],
                                   [task1, task2, task3],
                                   preds, [[segment1], [segment2], [segment3]],
                                   verify_harmlessness=True)
    assert len(pnads) == 3
    op0, op1, op2 = sorted(pnads, key=str)
    assert str(op0) == """STRIPS-Op0-0:
    Parameters: [?x0:obj_type]
    Preconditions: [IsRed(?x0:obj_type)]
    Add Effects: [IsHappy(?x0:obj_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Interact()"""
    assert len(op0.datastore) == 1
    assert str(op1) == """STRIPS-Op0-1:
    Parameters: [?x0:obj_type]
    Preconditions: [IsGreen(?x0:obj_type)]
    Add Effects: [IsHappy(?x0:obj_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Interact()"""
    assert len(op1.datastore) == 1
    assert str(op2) == """STRIPS-Op1-0:
    Parameters: [?x0:obj_type]
    Preconditions: [IsBlue(?x0:obj_type)]
    Add Effects: [IsSad(?x0:obj_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Interact()"""
    assert len(op2.datastore) == 1

    # If we run cluster_and_search without allowing any node expansions
    # during GBFS, we should recover the cluster_and_intersect operators.
    utils.reset_config({
        "strips_learner": "cluster_and_search",
        "clustering_learner_false_pos_weight": 100,
        "cluster_and_search_inner_search_max_expansions": 0,
    })
    pnads = learn_strips_operators([traj1, traj2, traj3],
                                   [task1, task2, task3],
                                   preds, [[segment1], [segment2], [segment3]],
                                   verify_harmlessness=True)
    assert len(pnads) == 2
    op0, op1 = sorted(pnads, key=str)
    assert str(op0) == """STRIPS-Op0-0:
    Parameters: [?x0:obj_type]
    Preconditions: []
    Add Effects: [IsHappy(?x0:obj_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Interact()"""
    assert len(op0.datastore) == 2
    assert str(op1) == """STRIPS-Op1-0:
    Parameters: [?x0:obj_type]
    Preconditions: [IsBlue(?x0:obj_type)]
    Add Effects: [IsSad(?x0:obj_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Interact()"""
    assert len(op1.datastore) == 1
