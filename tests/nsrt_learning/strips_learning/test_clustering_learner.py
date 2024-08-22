"""Tests for clustering-based STRIPS operator learning.

Note that most of the coverage is provided by
test_nsrt_learning_approach.py, which runs end-to-end tests of the
algorithms on actual domains.
"""

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.structs import Action, LowLevelTrajectory, \
    ParameterizedOption, Predicate, Segment, State, Task, Type


def test_cluster_and_intersect_strips_learner():
    """Tests for ClusterAndIntersectSTRIPSLearner."""
    utils.reset_config({"segmenter": "option_changes"})
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    pred1 = Predicate("Pred1", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    preds = {pred0, pred1, pred2}
    state0 = State({cup0: [0.4], cup1: [0.7], cup2: [0.1]})
    atoms0 = utils.abstract(state0, preds)
    state1 = State({cup0: [0.8], cup1: [0.3], cup2: [1.0]})
    atoms1 = utils.abstract(state1, preds)
    # Tests with known options.
    param_option = utils.SingletonParameterizedOption(
        "Dummy",
        lambda s, m, o, p: Action(p),
        types=[cup_type],
        params_space=Box(0.1, 1, (1, )),
    )
    option0 = param_option.ground([cup0], np.array([0.2]))
    assert option0.initiable(state0)
    action0 = option0.policy(state0)
    # The option changes, but the option spec stays the same. Want to segment.
    # Note that this is also a test for the case where the final option
    # terminates in the final state.
    option1 = param_option.ground([cup0], np.array([0.1]))
    assert option1.initiable(state0)
    action1 = option1.policy(state0)
    option2 = param_option.ground([cup1], np.array([0.1]))
    assert option2.initiable(state0)
    action2 = option2.policy(state0)
    known_option_ll_traj = LowLevelTrajectory(
        [state0.copy() for _ in range(5)],
        [action0, action1, action2, action0])
    known_option_atom_seq = [atoms0, atoms0, atoms0, atoms0, atoms0]
    known_option_segments = segment_trajectory(known_option_ll_traj, preds,
                                               known_option_atom_seq)
    assert len(known_option_segments) == 4
    # Test case where the final option does not terminate in the final state.
    infinite_param_option = ParameterizedOption(
        "InfiniteDummy",
        types=[cup_type],
        params_space=Box(0.1, 1, (1, )),
        policy=lambda s, m, o, p: Action(p),
        initiable=lambda s, m, o, p: True,
        terminal=lambda s, m, o, p: False,
    )
    infinite_option = infinite_param_option.ground([cup0], np.array([0.2]))
    states = [state0.copy() for _ in range(5)]
    infinite_option.initiable(states[0])
    actions = [infinite_option.policy(s) for s in states[:-1]]
    atom_seq = [atoms0, atoms0, atoms0, atoms0, atoms1]
    assert len(
        segment_trajectory(LowLevelTrajectory(states, actions), preds,
                           atom_seq)) == 0

    # More tests for temporally extended options.
    def _initiable(s, m, o, p):
        del s, o, p  # unused
        m["steps_remaining"] = 3
        return True

    def _policy(s, m, o, p):
        del s, o  # unused
        m["steps_remaining"] -= 1
        return Action(p)

    def _terminal(s, m, o, p):
        del s, o, p  # unused
        return m["steps_remaining"] <= 0

    three_step_param_option = ParameterizedOption(
        "ThreeStepDummy",
        types=[cup_type],
        params_space=Box(0.1, 1, (1, )),
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
    )

    def _simulate(s, a):
        del a  # unused
        return s.copy()

    three_option0 = three_step_param_option.ground([cup0], np.array([0.2]))
    three_option1 = three_step_param_option.ground([cup0], np.array([0.2]))
    policy = utils.option_plan_to_policy([three_option0, three_option1])
    traj = utils.run_policy_with_simulator(
        policy,
        _simulate,
        state0,
        termination_function=lambda s: False,
        max_num_steps=6)
    atom_traj = [atoms0] * 3 + [atoms1] * 3 + [atoms0]
    segments = segment_trajectory(traj, preds, atom_traj)
    assert len(segments) == 2
    segment0 = segments[0]
    segment1 = segments[1]
    assert segment0.has_option()
    assert segment0.get_option() == three_option0
    assert segment0.init_atoms == atoms0
    assert segment0.final_atoms == atoms1
    assert segment1.has_option()
    assert segment1.get_option() == three_option1
    assert segment1.init_atoms == atoms1
    assert segment1.final_atoms == atoms0

    # Tests without known options.
    action0 = option0.policy(state0)
    action0.unset_option()
    action1 = option0.policy(state0)
    action1.unset_option()
    action2 = option1.policy(state0)
    action2.unset_option()
    ll_traj = LowLevelTrajectory([state0.copy() for _ in range(5)],
                                 [action0, action1, action2, action0])
    atom_seq = [atoms0, atoms0, atoms0, atoms0, atoms0]
    # changes segmenter.
    utils.reset_config({"segmenter": "oracle"})
    known_option_segments = segment_trajectory(known_option_ll_traj, preds,
                                               known_option_atom_seq)
    assert len(known_option_segments) == 4
    # Segment with atoms changes instead.
    utils.reset_config({"segmenter": "atom_changes"})
    assert len(segment_trajectory(ll_traj, preds, atom_seq)) == 0
    unknown_option_ll_traj = LowLevelTrajectory(
        [state0.copy() for _ in range(5)] + [state1],
        [action0, action1, action2, action0, action1])
    atom_seq = [atoms0, atoms0, atoms0, atoms0, atoms0, atoms1]
    unknown_option_segments = segment_trajectory(unknown_option_ll_traj, preds,
                                                 atom_seq)

    utils.reset_config({"strips_learner": "cluster_and_intersect"})
    known_option_pnads = learn_strips_operators([known_option_ll_traj],
                                                None,
                                                None, [known_option_segments],
                                                verify_harmlessness=True,
                                                annotations=None)
    known_option_ops = [pnad.op for pnad in known_option_pnads]
    assert len(known_option_ops) == 1
    assert str((known_option_ops[0])) == """STRIPS-Op0:
    Parameters: [?x0:cup_type]
    Preconditions: []
    Add Effects: []
    Delete Effects: []
    Ignore Effects: []"""
    unknown_option_pnads = learn_strips_operators([unknown_option_ll_traj],
                                                  None,
                                                  None,
                                                  [unknown_option_segments],
                                                  verify_harmlessness=True,
                                                  annotations=None)
    unknown_option_ops = [pnad.op for pnad in unknown_option_pnads]
    assert len(unknown_option_ops) == 1
    assert str(unknown_option_ops[0]) == """STRIPS-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x0:cup_type), Pred0(?x2:cup_type), Pred1(?x0:cup_type, ?x0:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred1(?x2:cup_type, ?x2:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Ignore Effects: []"""  # pylint: disable=line-too-long

    # Test pruning pnads with small datastores. The set up is copied from the
    # next test, test_cluster_and_search_strips_learner().
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
    utils.reset_config({
        "strips_learner": "cluster_and_intersect",
        "cluster_and_intersect_prune_low_data_pnads": True,
        "cluster_and_intersect_min_datastore_fraction": 0.0
    })
    pnads = learn_strips_operators([traj1, traj2, traj3],
                                   [task1, task2, task3],
                                   preds, [[segment1], [segment2], [segment3]],
                                   verify_harmlessness=False,
                                   annotations=None)
    assert len(pnads) == 2
    utils.reset_config({
        "strips_learner": "cluster_and_intersect",
        "cluster_and_intersect_prune_low_data_pnads": True,
        "cluster_and_intersect_min_datastore_fraction": 0.4
    })
    pnads = learn_strips_operators([traj1, traj2, traj3],
                                   [task1, task2, task3],
                                   preds, [[segment1], [segment2], [segment3]],
                                   verify_harmlessness=False,
                                   annotations=None)
    assert len(pnads) == 1
    assert len(pnads[0].datastore) == 2
    utils.reset_config({
        "strips_learner": "cluster_and_intersect",
        "cluster_and_intersect_prune_low_data_pnads": True,
        "cluster_and_intersect_min_datastore_fraction": 0.9
    })
    pnads = learn_strips_operators([traj1, traj2, traj3],
                                   [task1, task2, task3],
                                   preds, [[segment1], [segment2], [segment3]],
                                   verify_harmlessness=False,
                                   annotations=None)
    assert len(pnads) == 0

    # Test cluster and intersect soft intersection.
    utils.reset_config({
        "strips_learner":
        "cluster_and_intersect",
        "cluster_and_intersect_soft_intersection_for_preconditions":
        True,
        "precondition_soft_intersection_threshold_percent":
        0.6
    })
    s4 = State({obj: [0.0, 1.0, 0.0, 0.0, 0.0]})
    a4 = Action([], Interact)
    ns4 = State({obj: [0.0, 1.0, 0.0, 1.0, 0.0]})
    g4 = {IsHappy([obj])}
    traj4 = LowLevelTrajectory([s4, ns4], [a4], True, 1)
    task4 = Task(s4, g4)
    segment4 = Segment(traj4, utils.abstract(s4, preds),
                       utils.abstract(ns4, preds), Interact)
    pnads = learn_strips_operators(
        [traj1, traj2, traj3, traj4], [task1, task2, task3, task4],
        preds, [[segment1], [segment2], [segment3], [segment4]],
        verify_harmlessness=False,
        annotations=None)
    assert len(pnads) == 2
    assert len(pnads[0].op.preconditions) == 1
    # When we take atoms that appear >= 60% of the time as preconditions,
    # IsGreen will show up, because it occurs 2/3 times.
    assert list(pnads[0].op.preconditions)[0].predicate == IsGreen

    utils.reset_config({
        "strips_learner":
        "cluster_and_intersect",
        "cluster_and_intersect_soft_intersection_for_preconditions":
        True,
        "precondition_soft_intersection_threshold_percent":
        0.7
    })
    pnads = learn_strips_operators(
        [traj1, traj2, traj3, traj4], [task1, task2, task3, task4],
        preds, [[segment1], [segment2], [segment3], [segment4]],
        verify_harmlessness=False,
        annotations=None)
    assert len(pnads) == 2
    # Now that the threshold is 70%, there will be no chosen preconditions.
    assert len(pnads[0].op.preconditions) == 0


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
                                   verify_harmlessness=True,
                                   annotations=None)
    assert len(pnads) == 2
    op0, op1 = sorted(pnads, key=str)
    assert str(op0) == """STRIPS-Op0:
    Parameters: [?x0:obj_type]
    Preconditions: []
    Add Effects: [IsHappy(?x0:obj_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Interact()"""
    assert len(op0.datastore) == 2
    assert str(op1) == """STRIPS-Op1:
    Parameters: [?x0:obj_type]
    Preconditions: [IsBlue(?x0:obj_type)]
    Add Effects: [IsSad(?x0:obj_type)]
    Delete Effects: []
    Ignore Effects: []
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
                                   verify_harmlessness=True,
                                   annotations=None)
    assert len(pnads) == 3
    op0, op1, op2 = sorted(pnads, key=str)
    assert str(op0) == """STRIPS-Op0-0:
    Parameters: [?x0:obj_type]
    Preconditions: [IsRed(?x0:obj_type)]
    Add Effects: [IsHappy(?x0:obj_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Interact()"""
    assert len(op0.datastore) == 1
    assert str(op1) == """STRIPS-Op0-1:
    Parameters: [?x0:obj_type]
    Preconditions: [IsGreen(?x0:obj_type)]
    Add Effects: [IsHappy(?x0:obj_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Interact()"""
    assert len(op1.datastore) == 1
    assert str(op2) == """STRIPS-Op1-0:
    Parameters: [?x0:obj_type]
    Preconditions: [IsBlue(?x0:obj_type)]
    Add Effects: [IsSad(?x0:obj_type)]
    Delete Effects: []
    Ignore Effects: []
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
                                   verify_harmlessness=True,
                                   annotations=None)
    assert len(pnads) == 2
    op0, op1 = sorted(pnads, key=str)
    assert str(op0) == """STRIPS-Op0-0:
    Parameters: [?x0:obj_type]
    Preconditions: []
    Add Effects: [IsHappy(?x0:obj_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Interact()"""
    assert len(op0.datastore) == 2
    assert str(op1) == """STRIPS-Op1-0:
    Parameters: [?x0:obj_type]
    Preconditions: [IsBlue(?x0:obj_type)]
    Add Effects: [IsSad(?x0:obj_type)]
    Delete Effects: []
    Ignore Effects: []
    Option Spec: Interact()"""
    assert len(op1.datastore) == 1
