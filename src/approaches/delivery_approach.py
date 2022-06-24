"""An approach that implements a delivery-specific policy.

Example command line:
    python src/main.py --approach delivery_policy --seed 0 \
        --env pddl_easy_delivery_procedural_tasks
"""

from typing import Callable, cast

import numpy as np

from predicators.src.approaches import BaseApproach
from predicators.src.envs.pddl_env import _PDDLEnvState
from predicators.src.structs import Action, GroundAtom, State, Task


class DeliverySpecificApproach(BaseApproach):
    """Implements a delivery-specific policy."""

    @classmethod
    def get_name(cls) -> str:
        return "delivery_policy"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        def _policy(state: State) -> Action:
            options = {o.name: o for o in self._initial_options}
            predicates = {p.name: p for p in self._initial_predicates}
            types = {t.name: t for t in self._types}
            state = cast(_PDDLEnvState, state)
            ground_atoms = state.get_ground_atoms()
            locations = state.get_objects(types["loc"])
            papers = state.get_objects(types["paper"])
            at = predicates["at"]
            wants_paper = predicates["wantspaper"]
            move = options["move"]
            homebase = predicates["ishomebase"]
            satisfied = predicates["satisfied"]
            safe = predicates["safe"]
            unpacked = predicates["unpacked"]
            carrying = predicates["carrying"]

            deliver = options["deliver"]
            pickup = options["pick-up"]

            move_to_loc = None
            move_from_loc = None

            for loc in locations:

                if GroundAtom(at, [loc]) in ground_atoms:

                    move_from_loc = loc

                    if GroundAtom(homebase, [loc]) in ground_atoms:
                        for paper in papers:
                            if GroundAtom(unpacked, [paper]) in ground_atoms:
                                selected_option = pickup
                                object_args = [paper, loc]
                                params = np.zeros(0, dtype=np.float32)
                                ground_option = selected_option.ground(
                                    object_args, params)
                                assert ground_option.initiable(state)
                                return ground_option.policy(state)

                    elif GroundAtom(wants_paper, [loc]) in ground_atoms:
                        for paper in papers:
                            if GroundAtom(carrying, [paper]) in ground_atoms:
                                selected_option = deliver
                                object_args = [paper, loc]
                                params = np.zeros(0, dtype=np.float32)
                                ground_option = selected_option.ground(
                                    object_args, params)
                                assert ground_option.initiable(state)
                                return ground_option.policy(state)

                elif GroundAtom(wants_paper, [loc]) in ground_atoms:
                    selected_option = move
                    move_to_loc = loc
                    object_args = [move_from_loc, move_to_loc]
                    params = np.zeros(0, dtype=np.float32)
                    ground_option = selected_option.ground(object_args, params)
                    assert ground_option.initiable(state)
                    return ground_option.policy(state)

        return _policy