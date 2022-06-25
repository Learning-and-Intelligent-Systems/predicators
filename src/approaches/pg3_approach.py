"""Policy-guided planning for generalized policy generation (PG3).

Example command line:
    python src/main.py --approach pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 1
"""

from typing import List, Optional, Set

from predicators.src import utils
from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.nsrt_metacontroller_approach import \
    NSRTMetacontrollerApproach
from predicators.src.structs import Box, Dataset, GroundAtom, LDLRule, \
    LiftedAtom, LiftedDecisionList, LowLevelTrajectory, ParameterizedOption, \
    Predicate, State, Task, Type, _GroundNSRT


class PG3Approach(NSRTMetacontrollerApproach):
    """Policy-guided planning for generalized policy generation (PG3)."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._current_ldl = LiftedDecisionList("LDL0", [])

    @classmethod
    def get_name(cls) -> str:
        return "pg3"

    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom]) -> _GroundNSRT:
        del state  # unused
        ground_nsrt = utils.query_ldl(self._current_ldl, atoms, goal)
        if ground_nsrt is None:
            raise ApproachFailure("PG3 policy was not applicable!")
        return ground_nsrt

    def _learn_ldl(self, trajectories: List[LowLevelTrajectory]) -> None:
        """Learn a lifted decision list policy."""
        del trajectories  # TODO use

        predicates = {p.name: p for p in self._initial_predicates}
        at = predicates["at"]
        carrying = predicates["carrying"]
        is_home_base = predicates["ishomebase"]
        safe = predicates["safe"]
        satisfied = predicates["satisfied"]
        wants_paper = predicates["wantspaper"]
        unpacked = predicates["unpacked"]

        nsrts = {n.name: n for n in self._get_current_nsrts()}
        deliver = nsrts["deliver"]
        move = nsrts["move"]
        pick_up = nsrts["pick-up"]

        paper_var, loc_var = deliver.parameters
        from_var, to_var = move.parameters
        papers_var, at_var = pick_up.parameters
        rules = [
            LDLRule(
                "Pick-up",
                parameters=[papers_var, at_var],
                pos_state_preconditions={
                    LiftedAtom(at, [at_var]),
                    LiftedAtom(is_home_base, [at_var]),
                    LiftedAtom(unpacked, [papers_var])
                },
                neg_state_preconditions={LiftedAtom(carrying, [papers_var])},
                goal_preconditions=set(),
                nsrt=pick_up),
            LDLRule(
                "Move",
                parameters=[from_var, to_var],
                pos_state_preconditions={
                    LiftedAtom(at, [from_var]),
                    LiftedAtom(safe, [from_var])
                    #LiftedAtom(wants_paper, [to_var])
                },
                neg_state_preconditions={
                    LiftedAtom(satisfied, [to_var]),
                    LiftedAtom(wants_paper, [from_var])
                },
                goal_preconditions={LiftedAtom(satisfied, [to_var])},
                nsrt=move),
            LDLRule("Deliver",
                    parameters=[paper_var, loc_var],
                    pos_state_preconditions={
                        LiftedAtom(at, [loc_var]),
                        LiftedAtom(wants_paper, [loc_var]),
                        LiftedAtom(carrying, [paper_var])
                    },
                    neg_state_preconditions={LiftedAtom(satisfied, [loc_var])},
                    goal_preconditions={LiftedAtom(satisfied, [loc_var])},
                    nsrt=deliver)
        ]

        self._current_ldl = LiftedDecisionList("DeliveryPolicy", rules)

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First, learn NSRTs.
        self._learn_nsrts(dataset.trajectories, online_learning_cycle=None)
        # Now, learn the LDL policy.
        self._learn_ldl(dataset.trajectories)
        # TODO save the LDL policy.

    def load(self, online_learning_cycle: Optional[int]) -> None:
        # TODO
        import ipdb
        ipdb.set_trace()
