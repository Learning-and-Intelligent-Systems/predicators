"""Test cases for the PG3 approach."""

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, create_approach
from predicators.src.approaches.pg3_approach import _PolicyGuidedPG3Heuristic
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_new_env
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.settings import CFG
from predicators.src.structs import LDLRule, LiftedDecisionList


def test_policy_guided_heuristic():
    """Tests for _PolicyGuidedPG3Heuristic."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "pg3",
        "num_train_tasks": 10,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
        "pg3_heuristic": "policy_guided",
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    ground_atom_trajectories = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
    abstract_train_tasks = [
        utils.create_abstract_task(t, env.predicates) for t in train_tasks
    ]
    nsrts = get_gt_nsrts(env.predicates, env.options)
    heuristic = _PolicyGuidedPG3Heuristic(abstract_train_tasks,
                                          ground_atom_trajectories,
                                          env.predicates, nsrts)
    name_to_nsrt = {nsrt.name: nsrt for nsrt in nsrts}
    deliver_nsrt = name_to_nsrt["deliver"]
    pick_up_nsrt = name_to_nsrt["pick-up"]
    move_nsrt = name_to_nsrt["move"]
    name_to_pred = {pred.name: pred for pred in env.predicates}
    satisfied = name_to_pred["satisfied"]
    wantspaper = name_to_pred["wantspaper"]

    pick_up_rule = LDLRule(name="PickUp",
                           parameters=pick_up_nsrt.parameters,
                           pos_state_preconditions=set(
                               pick_up_nsrt.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions=set(),
                           nsrt=pick_up_nsrt)

    paper, loc = deliver_nsrt.parameters
    assert "paper" in str(paper)
    assert "loc" in str(loc)

    deliver_rule1 = LDLRule(name="Deliver",
                            parameters=[loc, paper],
                            pos_state_preconditions=set(
                                deliver_nsrt.preconditions),
                            neg_state_preconditions=set(),
                            goal_preconditions=set(),
                            nsrt=deliver_nsrt)

    deliver_rule2 = LDLRule(
        name="Deliver",
        parameters=[loc, paper],
        pos_state_preconditions=set(deliver_nsrt.preconditions),
        neg_state_preconditions={satisfied([loc])},  # different
        goal_preconditions=set(),
        nsrt=deliver_nsrt)

    from_loc, to_loc = move_nsrt.parameters
    assert "from" in str(from_loc)
    assert "to" in str(to_loc)

    move_rule1 = LDLRule(name="Move",
                         parameters=[from_loc, to_loc],
                         pos_state_preconditions=set(move_nsrt.preconditions),
                         neg_state_preconditions=set(),
                         goal_preconditions=set(),
                         nsrt=move_nsrt)

    move_rule2 = LDLRule(
        name="Move",
        parameters=[from_loc, to_loc],
        pos_state_preconditions=set(move_nsrt.preconditions) | \
                                {wantspaper([to_loc])},  # different
        neg_state_preconditions=set(),
        goal_preconditions=set(),
        nsrt=move_nsrt
    )

    # The final policy should get a perfect score.
    ldl = LiftedDecisionList([pick_up_rule, deliver_rule2, move_rule2])
    score = heuristic(ldl)
    assert score == 0

    # Scores should monotonically decrease.
    policy_sequence = [
        LiftedDecisionList([]),
        LiftedDecisionList([pick_up_rule]),
        LiftedDecisionList([pick_up_rule, deliver_rule1]),
        LiftedDecisionList([pick_up_rule, deliver_rule2]),
        LiftedDecisionList([pick_up_rule, deliver_rule2, move_rule1]),
        LiftedDecisionList([pick_up_rule, deliver_rule2, move_rule2]),
    ]

    score_sequence = [heuristic(ldl) for ldl in policy_sequence]

    # TODO
    import ipdb
    ipdb.set_trace()
