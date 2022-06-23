"""An approach that implements a delivery-specific policy.

Example command line:
    python src/main.py --approach delivery_policy --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 1
"""

from typing import Callable, cast

from predicators.src import utils
from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.envs.pddl_env import _PDDLEnvState
from predicators.src.structs import Action, LDLRule, LiftedAtom, \
    LiftedDecisionList, State, Task


class DeliverySpecificApproach(NSRTLearningApproach):
    """Implements a delivery-specific policy."""

    @classmethod
    def get_name(cls) -> str:
        return "delivery_policy"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        goal = task.goal

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

        # TODO: Fix the code here!

        rules = [
            LDLRule("Deliver",
                    parameters=[paper_var, loc_var],
                    pos_state_preconditions={
                        LiftedAtom(at, [loc_var]),
                        LiftedAtom(wants_paper, [loc_var]),
                        LiftedAtom(carrying, [paper_var])
                    },
                    neg_state_preconditions={LiftedAtom(satisfied, [loc_var])},
                    goal_preconditions=set(),
                    nsrt=deliver),
            LDLRule("PickUp",
                    parameters=[paper_var, loc_var],
                    pos_state_preconditions={
                        LiftedAtom(at, [loc_var]),
                        LiftedAtom(is_home_base, [loc_var]),
                        LiftedAtom(unpacked, [paper_var])
                    },
                    neg_state_preconditions=set(),
                    goal_preconditions=set(),
                    nsrt=pick_up),
            LDLRule("Move",
                    parameters=[from_var, to_var],
                    pos_state_preconditions={
                        LiftedAtom(at, [from_var]),
                        LiftedAtom(safe, [from_var])
                    },
                    neg_state_preconditions={LiftedAtom(satisfied, [to_var])},
                    goal_preconditions={LiftedAtom(satisfied, [to_var])},
                    nsrt=move),
        ]

        # Below this line should not need changing.

        ldl_policy = LiftedDecisionList("DeliveryPolicy", rules)

        def _policy(state: State) -> Action:
            state = cast(_PDDLEnvState, state)
            ground_atoms = state.get_ground_atoms()
            ground_nsrt = utils.query_ldl(ldl_policy, ground_atoms, goal)
            if ground_nsrt is None:
                raise ApproachFailure("LDL policy was not applicable!")
            ground_option = ground_nsrt.sample_option(state, goal, self._rng)
            assert ground_option.initiable(state)
            return ground_option.policy(state)

        return _policy
