"""Tests to cover side_predicate_learning.py.

Note that most of the coverage is provided by
test_nsrt_learning_approach.py, which runs end-to-end tests of the
algorithms on an actual domain.
"""

# This unused import prevents circular import issues when running this
# test as a standalone file.
from predicators.src import approaches  # pylint:disable=unused-import
from predicators.src.nsrt_learning.side_predicate_learning import \
    BackchainingSidePredicateLearner
from predicators.src.structs import Action, LowLevelTrajectory, \
    PartialNSRTAndDatastore, Predicate, Segment, State, STRIPSOperator, Task, \
    Type
from predicators.src.utils import SingletonParameterizedOption


def test_backchaining_normal_behavior():
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
    spl = BackchainingSidePredicateLearner(initial_pnads, [traj1, traj2],
                                           [task1, task2], {Asleep},
                                           [[segment1], [segment2]])
    pnads = spl.sideline()
    for pnad in pnads:
        print(pnad)
