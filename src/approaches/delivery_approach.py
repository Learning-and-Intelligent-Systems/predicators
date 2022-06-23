"""An approach that implements a delivery-specific policy.

Example command line:
    python src/main.py --approach delivery_policy --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 1
"""

from typing import Callable, cast

import numpy as np

from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.src.envs.pddl_env import _PDDLEnvState
from predicators.src.structs import Action, GroundAtom, State, Task, LiftedDecisionList, LDLRule
from predicators.src import utils


class DeliverySpecificApproach(NSRTLearningApproach):
    """Implements a delivery-specific policy."""

    @classmethod
    def get_name(cls) -> str:
        return "delivery_policy"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        goal = task.goal
        options = {o.name: o for o in self._initial_options}
        predicates = {p.name: p for p in self._initial_predicates}
        types = {t.name: t for t in self._types}
        nsrts = {n.name: n for n in self._get_current_nsrts()}
        deliver = nsrts["deliver"]
        move = nsrts["move"]
        pick_up = nsrts["pick_up"]
        paper_var, loc_var = deliver.parameters
        from_var, to_var = move.parameters

        # TODO: Fix the code here!

        rules = [
            LDLRule(
                "Deliver",
                parameters=[paper_var, loc_var],
                state_preconditions={at([loc_var]),
                                     wants_paper([loc_var])},
                goal_preconditions=set(),
                nsrt=pick_nsrt
            ),
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
