"""An approach that implements a delivery-specific policy.

Example command line:
    python src/main.py --approach delivery_policy --seed 0 \
        --env pddl_easy_delivery_procedural_tasks
"""

from typing import Callable, cast

import numpy as np

from predicators.src.approaches import BaseApproach
from predicators.src.envs.pddl_env import _PDDLEnvState
from predicators.src.structs import Action, GroundAtom, State, Task, Object


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
            isHomeBase = predicates["ishomebase"]
            unpacked = predicates["unpacked"]
            carrying = predicates["carrying"]
            safe = predicates["safe"]
            move_to_loc = None
            move_from_loc = None
            for loc in locations:
                if GroundAtom(at, [loc]) in ground_atoms:
                    move_from_loc = loc
                    if GroundAtom(isHomeBase, [loc]) in ground_atoms:
                        for pape in papers:
                            if GroundAtom(unpacked, [pape]) in ground_atoms:
                                pack = options["pick-up"]
                                selected_option = pack
                                object_args = [pape, loc]
                                params = np.zeros(0, dtype=np.float32)
                                ground_option = selected_option.ground(
                                    object_args, params)
                                assert ground_option.initiable(state)
                                return ground_option.policy(state)
                    elif GroundAtom(wants_paper, [loc]) in ground_atoms:
                        for pape in papers:
                            if GroundAtom(carrying, [pape]) in ground_atoms:
                                deliver = options["deliver"]
                                selected_option = deliver
                                object_args = [pape, loc]
                                params = np.zeros(0, dtype=np.float32)
                                ground_option = selected_option.ground(
                                    object_args, params)
                                assert ground_option.initiable(state)
                                return ground_option.policy(state)
                elif GroundAtom(wants_paper, [loc]) in ground_atoms:
                    if GroundAtom(safe, [loc]) in ground_atoms:
                        move_to_loc = loc
            move = options["move"]
            selected_option = move
            object_args = [cast(Object, move_from_loc), cast(Object,
            move_to_loc)]
            assert move_to_loc is not None
            assert move_from_loc is not None
            # Below this line should not need changing.
            params = np.zeros(0, dtype=np.float32)
            ground_option = selected_option.ground(object_args, params)
            assert ground_option.initiable(state)
            return ground_option.policy(state)
        return _policy
