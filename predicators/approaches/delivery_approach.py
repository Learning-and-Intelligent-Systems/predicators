"""An approach that implements a delivery-specific policy.

Example command line:
    python predicators/main.py --approach delivery_policy --seed 0 \
        --env pddl_easy_delivery_procedural_tasks
"""

from typing import Any, Callable, List, cast

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
        # print("task", task)

        def _policy(state: State) -> Action:
            # Extract the predicators and options from the state.
            #option=action
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
            satisfied = predicates["satisfied"]
            print(f'{ground_atoms=}')
            # print(f'{at=}')
            # print(f'{wants_paper=}')
            # print(f'{is_home_base=}')
            # print(f'{unpacked=}')
            # print(f'{carrying=}')
            for loc in locations:
                if GroundAtom(is_home_base, [loc]) in ground_atoms:
                    home_base = loc
            for paper in papers:
                if GroundAtom(carrying, [paper]) in ground_atoms:
                    carried_paper = paper
                    # print("carrying paper", carried_paper)

            for loc in locations:
                if GroundAtom(at, [loc]) in ground_atoms:
                    # print("we here", loc)

                    #if we're at home:
                    #loop thru all papers to see if we're carrying that paper.
                    #if we r carrying a paper:
                    #loop thru all locations to see if any want paper
                    #move to this location
                    #if we r not carrying a paper, pick one up!
                    #if we're not at home:
                    #if location wants paper
                    #deliver the paper
                    #if location is satisfied
                    #go back home

                    #if we're at home:
                    if GroundAtom(is_home_base, [loc]) in ground_atoms:
                        for paper in papers:

                            #if we r carrying a paper:
                            if GroundAtom(carrying, [paper]) in ground_atoms:
                                for destination in locations:
                                    #move to location that wants paper
                                    if GroundAtom(
                                            wants_paper,
                                        [destination]) in ground_atoms:
                                        selected_option = options["move"]
                                        object_args = [loc, destination]
                                        params = np.zeros(0, dtype=np.float32)
                                        ground_option = selected_option.ground(
                                            object_args, params)
                                        assert ground_option.initiable(state)
                                        # print(f'{ground_option.policy(state)=}')
                                        return ground_option.policy(state)

                        #if we r not carrying a paper, pick one up!
                        for paper in papers:
                            if GroundAtom(unpacked, [paper]) in ground_atoms:
                                pack = options["pick-up"]
                                selected_option = pack
                                object_args = [paper, loc]
                                params = np.zeros(0, dtype=np.float32)
                                ground_option = selected_option.ground(
                                    object_args, params)
                                assert ground_option.initiable(state)
                                # print(f'{ground_option.policy(state)=}')
                                return ground_option.policy(state)

                    else:
                        #if location wants paper, deliver
                        if GroundAtom(wants_paper, [loc]) in ground_atoms:
                            selected_option = options["deliver"]
                            object_args = [carried_paper, loc]
                            params = np.zeros(0, dtype=np.float32)
                            ground_option = selected_option.ground(
                                object_args, params)
                            assert ground_option.initiable(state)
                            # print(f'{ground_option.policy(state)=}')
                            return ground_option.policy(state)

                        #if location is satisfied, go back home
                        elif GroundAtom(satisfied, [loc]) in ground_atoms:
                            selected_option = options["move"]
                            object_args = [loc, home_base]
                            params = np.zeros(0, dtype=np.float32)
                            ground_option = selected_option.ground(
                                object_args, params)
                            assert ground_option.initiable(state)
                            # print(f'{ground_option.policy(state)=}')

            return ground_option.policy(state)


        return _policy
