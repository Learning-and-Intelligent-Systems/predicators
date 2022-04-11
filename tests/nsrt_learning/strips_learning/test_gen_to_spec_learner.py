"""Tests for general-to-specific STRIPS operator learning."""

from predicators.src.nsrt_learning.strips_learning.general_to_specific_learner import \
    BackchainingSTRIPSLearner
from predicators.src.structs import Action, LowLevelTrajectory, \
    PartialNSRTAndDatastore, Predicate, Segment, State, STRIPSOperator, Task, \
    Type
from predicators.src.utils import SingletonParameterizedOption


class MockBackchainingSTRIPSLearner(BackchainingSTRIPSLearner):
    """Mock class that exposes private methods for testing."""

    def try_specializing_pnad(self, necessary_add_effects, pnad, segment):
        """Exposed for testing."""
        return self._try_specializing_pnad(necessary_add_effects, pnad,
                                           segment)

    @staticmethod
    def find_unification(necessary_add_effects,
                         pnad,
                         segment,
                         find_partial_grounding=True):
        """Exposed for testing."""
        return (BackchainingSTRIPSLearner._find_unification(
            necessary_add_effects, pnad, segment, find_partial_grounding))


def test_backchaining_strips_learner():
    """Test the BackchainingSidePredicateLearner."""
    # Set up the PNADs.
    human_type = Type("human_type", ["feat1", "feat2"])
    Asleep = Predicate("Asleep", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    Sad = Predicate("Sad", [human_type], lambda s, o: s[o[0]][1] < 0.5)
    opt_name_to_opt = {}
    for opt_name in ["Cry", "Eat"]:
        opt = SingletonParameterizedOption(opt_name, lambda s, m, o, p: None)
        opt_name_to_opt[opt_name] = opt
    # Set up the data.
    bob = human_type("bob")
    state_awake_and_sad = State({bob: [0.0, 0.0]})
    state_awake_and_happy = State({bob: [0.0, 1.0]})
    state_asleep_and_sad = State({bob: [1.0, 0.0]})
    assert not Asleep.holds(state_awake_and_sad, [bob])
    assert Sad.holds(state_awake_and_sad, [bob])
    assert not Asleep.holds(state_awake_and_happy, [bob])
    assert not Sad.holds(state_awake_and_happy, [bob])
    assert Asleep.holds(state_asleep_and_sad, [bob])
    assert Sad.holds(state_asleep_and_sad, [bob])
    Cry = opt_name_to_opt["Cry"].ground([], [])
    Eat = opt_name_to_opt["Eat"].ground([], [])
    goal1 = {Asleep([bob])}
    act1 = Action([], Cry)
    traj1 = LowLevelTrajectory([state_awake_and_sad, state_asleep_and_sad],
                               [act1], True, 0)
    task1 = Task(state_awake_and_sad, goal1)
    segment1 = Segment(traj1, set(), goal1, Cry)
    goal2 = set()
    act2 = Action([], Eat)
    traj2 = LowLevelTrajectory([state_awake_and_sad, state_awake_and_sad],
                               [act2], True, 1)
    task2 = Task(state_awake_and_sad, set())
    segment2 = Segment(traj2, set(), goal2, Eat)
    learner = MockBackchainingSTRIPSLearner([traj1, traj2], [task1, task2],
                                            {Asleep}, [[segment1], [segment2]])
    pnads = learner.learn()
    # Verify the results are as expected.
    expected_strs = [
        """STRIPS-Cry0:
    Parameters: [?x0:human_type]
    Preconditions: []
    Add Effects: [Asleep(?x0:human_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Cry()""", """STRIPS-Eat0:
    Parameters: []
    Preconditions: []
    Add Effects: []
    Delete Effects: []
    Side Predicates: []
    Option Spec: Eat()"""
    ]
    for pnad, exp_str in zip(sorted(pnads, key=lambda pnad: pnad.op.name),
                             expected_strs):
        assert str(pnad) == repr(pnad) == exp_str

    # Test sidelining where an existing operator needs to
    # be specialized.
    goal3 = {Asleep([bob]), Sad([bob])}
    act3 = Action([], Cry)
    traj3 = LowLevelTrajectory([state_awake_and_sad, state_asleep_and_sad],
                               [act3], True, 0)
    traj4 = LowLevelTrajectory([state_awake_and_happy, state_asleep_and_sad],
                               [act3], True, 0)
    task3 = Task(state_awake_and_sad, goal3)
    segment3 = Segment(traj3, set([Sad([bob])]), goal3, Cry)
    segment4 = Segment(traj4, set(), goal3, Cry)
    # Create and run the sidelining approach.
    learner = MockBackchainingSTRIPSLearner([traj3, traj4], [task3],
                                            {Asleep, Sad},
                                            [[segment3], [segment4]])
    pnads = learner.learn()
    assert len(pnads) == 1
    expected_str = """STRIPS-Cry0:
    Parameters: [?x0:human_type]
    Preconditions: []
    Add Effects: [Asleep(?x0:human_type), Sad(?x0:human_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Cry()"""
    assert str(pnads[0]) == repr(pnads[0]) == expected_str


def test_find_unification_and_try_specializing_pnad():
    """Test the find_unification() and try_specializing_pnad() methods in the
    BackchainingSidePredicateLearner."""

    human_type = Type("human_type", ["feat"])
    Asleep = Predicate("Asleep", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    Happy = Predicate("Happy", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    opt = SingletonParameterizedOption("Move", lambda s, m, o, p: None)
    human_var = human_type("?x0")
    params = [human_var]
    add_effects = {Asleep([human_var])}
    op = STRIPSOperator("MoveOp", params, set(), add_effects, set(), set())
    pnad = PartialNSRTAndDatastore(op, [], (opt, []))
    bob = human_type("bob")
    state = State({bob: [0.0]})
    task = Task(state, set())
    Move = opt.ground([], [])
    traj = LowLevelTrajectory([state], [])
    segment = Segment(traj, {Happy([bob])}, {Asleep([bob])}, Move)
    # Create the sidelining approach.
    learner = MockBackchainingSTRIPSLearner([traj], [task], {Asleep, Happy},
                                            [[segment]])
    # Normal usage: the PNAD add effects can capture a subset of
    # the necessary_add_effects.
    ground_op = learner.find_unification(
        {Asleep([bob]), Happy([bob])}, pnad, Segment(traj, set(), set(), Move))
    assert ground_op is not None
    assert str(ground_op) == repr(ground_op) == """GroundSTRIPS-MoveOp:
    Parameters: [bob:human_type]
    Preconditions: []
    Add Effects: [Asleep(bob:human_type)]
    Delete Effects: []
    Side Predicates: []"""
    # The necessary_add_effects is empty, but the PNAD has an add effect,
    # so no grounding is possible.
    ground_op = learner.find_unification(set(), pnad,
                                         Segment(traj, set(), set(), Move))
    assert ground_op is None
    ground_op = learner.find_unification(set(), pnad,
                                         Segment(traj, set(), set(), Move),
                                         False)
    assert ground_op is None
    # Change the PNAD to have non-trivial preconditions.
    pnad.op = pnad.op.copy_with(preconditions={Happy([human_var])})
    # The new preconditions are not satisfiable in the segment's init_atoms,
    # so no grounding is possible.
    ground_op = learner.find_unification(
        set(), pnad, Segment(traj, {Asleep([bob])}, set(), Move))
    assert ground_op is None
    new_pnad = learner.try_specializing_pnad(
        set(), pnad, Segment(traj, {Asleep([bob])}, set(), Move))
    assert new_pnad is None
    # Make the preconditions be satisfiable in the segment's init_atoms.
    # Now, we are back to normal usage.
    ground_op = learner.find_unification({Asleep([bob])}, pnad,
                                         Segment(traj, {Happy([bob])}, set(),
                                                 Move))
    assert ground_op is not None
    assert str(ground_op) == repr(ground_op) == """GroundSTRIPS-MoveOp:
    Parameters: [bob:human_type]
    Preconditions: [Happy(bob:human_type)]
    Add Effects: [Asleep(bob:human_type)]
    Delete Effects: []
    Side Predicates: []"""
    new_pnad = learner.try_specializing_pnad({Asleep([bob])}, pnad,
                                             Segment(traj, {Happy([bob])},
                                                     set(), Move))
    assert str(new_pnad) == repr(new_pnad) == """STRIPS-MoveOp:
    Parameters: [?x0:human_type]
    Preconditions: [Happy(?x0:human_type)]
    Add Effects: [Asleep(?x0:human_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Move()"""
