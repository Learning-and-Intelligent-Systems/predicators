"""An approach that implements a delivery-specific policy.

Example command line:
    python predicators/main.py --approach deliver y_policy --seed 0 \
        --env pddl_easy_delivery_procedural_tasks
"""

from typing import Callable, cast

import numpy as np

from predicators.approaches import BaseApproach
from predicators.envs.pddl_env import _PDDLEnvState
from predicators.structs import Action, GroundAtom, State, Task


class DeliverySpecificApproach(BaseApproach):
    """Implements a delivery-specific policy."""

    @classmethod
    def get_name(cls) -> str:
        return "delivery_policy"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self.home_base = None
        self.want_locations = None

        def commit_action(state, action, object_args):
            params = np.zeros(0, dtype=np.float32)
            selected_option = action
            ground_option = selected_option.ground(
                object_args, params)

            assert ground_option.initiable(state)
            return ground_option.policy(state)

        def _policy(state: State) -> Action:
            # Extract the predicators and options from the state.
            options = {o.name: o for o in self._initial_options}
            predicates = {p.name: p for p in self._initial_predicates}

            types = {t.name: t for t in self._types}
            state = cast(_PDDLEnvState, state)
            ground_atoms = state.get_ground_atoms()
            locations = state.get_objects(types["loc"])
            papers = state.get_objects(types["paper"])

            satisfied = predicates["satisfied"]
            at = predicates["at"]
            wants_paper = predicates["wantspaper"]
            is_home_base = predicates["ishomebase"]
            unpacked = predicates["unpacked"]
            carrying = predicates["carrying"]
            safe = predicates["safe"]

            '''
            Strategy: If at homebase: Pick up papers, move to location that needs paper, deliver paper,
                      return to homebase, repeat
            '''

            # define home-base
            if self.home_base is None:
                for loc in locations:
                    if GroundAtom(is_home_base, [loc]) in ground_atoms:
                        self.home_base = loc
                        break

            # Find list of locations that want paper
            if self.want_locations is None:
                self.want_locations = []

                for loc in locations:
                    if GroundAtom(safe, [loc]) in ground_atoms and GroundAtom(wants_paper, [loc]) in ground_atoms:
                        self.want_locations.append(loc)

            for loc in locations:
                if GroundAtom(at, [loc]) in ground_atoms:
                    if GroundAtom(is_home_base, [loc]) in ground_atoms:
                        for paper in papers:

                            # Pick-up paper if at home-base
                            if GroundAtom(unpacked, [paper]) in ground_atoms:
                                return commit_action(state, options["pick-up"], [paper, loc])

                        # Once all papers picked, move to safe location that wants papers
                        loc_to_move = self.want_locations.pop(0)
                        return commit_action(state, options["move"], [self.home_base, loc_to_move])

                    # If location the robot is at wants paper
                    if GroundAtom(wants_paper, [loc]) in ground_atoms:
                        for paper in papers:

                            # deliver paper
                            if GroundAtom(carrying, [paper]) in ground_atoms:
                                return commit_action(state, options["deliver"], [paper, loc])

                    # Else move to home_base
                    return commit_action(state, options["move"], [loc, self.home_base])

            # Shouldn't reach here
            return

        return _policy
