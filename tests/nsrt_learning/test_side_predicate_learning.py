"""Tests to cover side_predicate_learning.py.

Note that most of the coverage is provided by
test_nsrt_learning_approach.py, which runs end-to-end tests of the
algorithms on an actual domain.
"""

from predicators.src.nsrt_learning.side_predicate_learning import \
    BackchainingSidePredicateLearner
from predicators.src.structs import Action, LowLevelTrajectory, \
    PartialNSRTAndDatastore, Predicate, Segment, State, STRIPSOperator, Task, \
    Type
from predicators.src.utils import SingletonParameterizedOption


class MockBackchainingSPL(BackchainingSidePredicateLearner):
    """Mock class that exposes private methods for testing."""

    @staticmethod
    def get_partially_satisfying_grounding(necessary_add_effects, pnad,
                                           segment):
        """Exposed for testing."""
        return (BackchainingSidePredicateLearner.
                _get_partially_satisfying_grounding(necessary_add_effects,
                                                    pnad, segment))

    @staticmethod
    def try_refining_pnad(necessary_add_effects, pnad, segment):
        """Exposed for testing."""
        return (BackchainingSidePredicateLearner._try_refining_pnad(
            necessary_add_effects, pnad, segment))


def test_backchaining():
    """Test the BackchainingSidePredicateLearner."""

    # Set up the PNADs.
    human_type = Type("human_type", ["feat"])
    Asleep = Predicate("Asleep", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    initial_pnads = set()
    opt_name_to_opt = {}
    for opt_name in ["Cry", "Eat"]:
        opt = SingletonParameterizedOption(opt_name, lambda s, m, o, p: None)
        opt_name_to_opt[opt_name] = opt
        human_var = human_type("?human")
        params = [human_var]
        add_effects = {Asleep([human_var])}
        op = STRIPSOperator(f"{opt_name}Op", params, set(), add_effects, set(),
                            set())
        initial_pnads.add(PartialNSRTAndDatastore(op, [], (opt, [])))
    # Set up the data.
    bob = human_type("bob")
    state_awake = State({bob: [0.0]})
    state_asleep = State({bob: [1.0]})
    assert not Asleep.holds(state_awake, [bob])
    assert Asleep.holds(state_asleep, [bob])
    Cry = opt_name_to_opt["Cry"].ground([], [])
    Eat = opt_name_to_opt["Eat"].ground([], [])
    goal1 = {Asleep([bob])}
    act1 = Action([], Cry)
    traj1 = LowLevelTrajectory([state_awake, state_asleep], [act1], True, 0)
    task1 = Task(state_awake, goal1)
    segment1 = Segment(traj1, set(), goal1, Cry)
    goal2 = set()
    act2 = Action([], Eat)
    traj2 = LowLevelTrajectory([state_awake, state_awake], [act2], True, 1)
    task2 = Task(state_awake, set())
    segment2 = Segment(traj2, set(), goal2, Eat)
    # Create and run the sidelining approach.
    spl = MockBackchainingSPL(initial_pnads, [traj1, traj2], [task1, task2],
                              {Asleep}, [[segment1], [segment2]])
    pnads = spl.sideline()
    # Verify the results are as expected.
    expected_strs = [
        """STRIPS-Cry:
    Parameters: [?x0:human_type]
    Preconditions: []
    Add Effects: [Asleep(?x0:human_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Cry()""", """STRIPS-Eat:
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


def test_get_partially_satisfying_grounding_and_try_refining_pnad():
    """Test the _get_partially_satisfying_grounding() and try_refining_pnad()
    methods in the BackchainingSidePredicateLearner."""

    human_type = Type("human_type", ["feat"])
    Asleep = Predicate("Asleep", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    Happy = Predicate("Happy", [human_type], lambda s, o: s[o[0]][0] > 0.5)
    opt = SingletonParameterizedOption("Move", lambda s, m, o, p: None)
    human_var = human_type("?human")
    params = [human_var]
    add_effects = {Asleep([human_var])}
    op = STRIPSOperator("MoveOp", params, set(), add_effects, set(), set())
    pnad = PartialNSRTAndDatastore(op, [], (opt, []))
    bob = human_type("bob")
    state = State({bob: [0.0]})
    Move = opt.ground([], [])
    traj = LowLevelTrajectory([state], [])
    # Normal usage: the PNAD add effects can capture a subset of
    # the necessary_add_effects.
    ground_op = MockBackchainingSPL.get_partially_satisfying_grounding(
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
    ground_op = MockBackchainingSPL.get_partially_satisfying_grounding(
        set(), pnad, Segment(traj, set(), set(), Move))
    assert ground_op is None
    # Change the PNAD to have non-trivial preconditions.
    pnad.op = pnad.op.copy_with(preconditions={Happy([human_var])})
    # The new preconditions are not satisfiable in the segment's init_atoms,
    # so no grounding is possible.
    ground_op = MockBackchainingSPL.get_partially_satisfying_grounding(
        set(), pnad, Segment(traj, {Asleep([bob])}, set(), Move))
    assert ground_op is None
    pnad_refinable, _, _ = MockBackchainingSPL.try_refining_pnad(
        set(), pnad, Segment(traj, {Asleep([bob])}, set(), Move))
    assert not pnad_refinable
    # Make the preconditions be satisfiable in the segment's init_atoms.
    # Now, we are back to normal usage.
    ground_op = MockBackchainingSPL.get_partially_satisfying_grounding(
        {Asleep([bob])}, pnad, Segment(traj, {Happy([bob])}, set(), Move))
    assert ground_op is not None
    assert str(ground_op) == repr(ground_op) == """GroundSTRIPS-MoveOp:
    Parameters: [bob:human_type]
    Preconditions: [Happy(bob:human_type)]
    Add Effects: [Asleep(bob:human_type)]
    Delete Effects: []
    Side Predicates: []"""
