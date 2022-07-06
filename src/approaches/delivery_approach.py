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
        i = 0
        used_locs = []

        def _policy(state: State) -> Action:
            nonlocal i

            options = {o.name: o for o in self._initial_options}
            predicates = {p.name: p for p in self._initial_predicates}
            types = {t.name: t for t in self._types}
            state = cast(_PDDLEnvState, state)
            ground_atoms = state.get_ground_atoms()
            locations = state.get_objects(types["loc"])
            used_locs.append(locations[i])
            papers = state.get_objects(types["paper"])
            at = predicates["at"]
            isbase = predicates["ishomebase"]
            wants_paper = predicates["wantspaper"]
            safe = predicates["safe"]
            unpacked = predicates["unpacked"]
            carrying = predicates["carrying"]
            move = options["move"]
            pick = options["pick-up"]
            deliver = options["deliver"]

            assert GroundAtom(at, [locations[i]]) in ground_atoms
            if GroundAtom(isbase, [locations[i]]) in ground_atoms:
                selected_option = pick
                for paper in papers:
                    if GroundAtom(unpacked, [paper]) in ground_atoms:
                        object_args = [paper, locations[i]]
                        selected_option = pick
                        params = np.zeros(0, dtype=np.float32)
                        ground_option = selected_option.ground(
                            object_args, params)
                        return ground_option.policy(state)
            if GroundAtom(wants_paper, [locations[i]]) in ground_atoms:
                for paper in papers:
                    if GroundAtom(carrying, [paper]) in ground_atoms:
                        object_args = [paper, locations[i]]
                        selected_option = deliver
                        params = np.zeros(0, dtype=np.float32)
                        ground_option = selected_option.ground(
                            object_args, params)
                        return ground_option.policy(state)

            for loc in locations:
                if GroundAtom(safe,
                              [loc]) in ground_atoms and not loc in used_locs:
                    object_args = [locations[i], loc]
                    selected_option = move
                    params = np.zeros(0, dtype=np.float32)
                    ground_option = selected_option.ground(object_args, params)
                    i = locations.index(loc)
                    return ground_option.policy(state)

            return ground_option.policy(state)

        return _policy
