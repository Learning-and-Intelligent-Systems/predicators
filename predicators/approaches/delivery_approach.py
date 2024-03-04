"""An approach that implements a delivery-specific policy.

Example command line:
    python predicators/main.py --approach delivery_policy --seed 0 \
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

        def _policy(state: State) -> Action:
            # Extract the predicators and options from the state.
            options = {o.name: o for o in self._initial_options}
            predicates = {p.name: p for p in self._initial_predicates}

            types = {t.name: t for t in self._types}
            state = cast(_PDDLEnvState, state)
            ground_atoms = state.get_ground_atoms()
            locations = state.get_objects(types["loc"])
            papers = state.get_objects(types["paper"])
    
            at = predicates["at"]
            wants_paper = predicates["wantspaper"]
            is_home_base = predicates["ishomebase"]
            unpacked = predicates["unpacked"]
            carrying = predicates["carrying"]

            for loc in locations:
                if GroundAtom(at, [loc]) in ground_atoms:
                    if GroundAtom(is_home_base, [loc]) in ground_atoms:
                        for paper in papers:
                            if GroundAtom(unpacked, [paper]) in ground_atoms:
                                pack = options["pick-up"]
                                selected_option = pack
                                object_args = [paper, loc]
                                params = np.zeros(0, dtype=np.float32)
                                ground_option = selected_option.ground(
                                    object_args, params)
                                assert ground_option.initiable(state)
                                return ground_option.policy(state)
            # raise NotImplementedError("Finish me!")

        return _policy
