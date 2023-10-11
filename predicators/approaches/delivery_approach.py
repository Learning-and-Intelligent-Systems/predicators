"""An approach that implements a delivery-specific policy.

Example command line:
    python predicators/main.py --approach delivery_approach --seed 0 \
        --env pddl_easy_delivery_procedural_tasks
"""

from typing import Callable, Iterable, List, Optional, Set, cast

import numpy as np
from gym.spaces.box import Box

from predicators.approaches import ApproachFailure, BaseApproach
from predicators.envs.pddl_env import _PDDLEnvState
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, _Option

__all__ = ["DeliverySpecificApproach"]


def filter_predicated_objs(objs: Iterable[Object], pred: Predicate,
                           ground_atoms: Iterable[GroundAtom]) -> Set[Object]:
    """Filters the objects based on whether they fulfill a unitary predicate in
    a given set of ground atoms.

    The predicate has to be unitary
    """
    return set(
        filter(lambda obj: GroundAtom(pred, [obj]) in ground_atoms, objs))


def check_option(ground_option: _Option, state: _PDDLEnvState) -> None:
    """Makes sure that the ground option is initiable on the state.

    If not, throws an exception.
    """
    if not ground_option.initiable(state):
        raise ApproachFailure("Could not execute option")  # pragma: no cover


class DeliverySpecificApproach(BaseApproach):
    """Implements a policy for the delivery domain.

    See envs/assets/pddl/delivery/domain.pddl for the domain definition
    """

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]):
        super().__init__(initial_predicates=initial_predicates,
                         initial_options=initial_options,
                         types=types,
                         action_space=action_space,
                         train_tasks=train_tasks)

        # Extracts domain-specific class instances for the delivery domain
        type_dict = {t.name: t for t in types}
        self.type_loc = type_dict["loc"]
        self.type_paper = type_dict["paper"]

        options = {o.name: o for o in initial_options}
        self.opt_pick_up = options["pick-up"]
        self.opt_move = options["move"]
        self.opt_deliver = options["deliver"]

        predicates = {p.name: p for p in initial_predicates}
        self.pred_is_home_base = predicates["ishomebase"]
        self.pred_safe = predicates["safe"]
        self.pred_at = predicates["at"]
        self.pred_wants_paper = predicates["wantspaper"]
        self.pred_satisfied = predicates["satisfied"]
        self.pred_unpacked = predicates["unpacked"]
        self.pred_carrying = predicates["carrying"]

    @classmethod
    def get_name(cls) -> str:
        return "delivery_approach"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _extract_at_loc_optional(
            self, locs: Iterable[Object],
            ground_atoms: Iterable[GroundAtom]) -> Optional[Object]:
        at_locs = filter_predicated_objs(locs, self.pred_at, ground_atoms)
        if len(at_locs) > 1:
            raise ApproachFailure(
                "Cannot be at multiple locations at the same time"
            )  # pragma: no cover
        return min(at_locs, default=None)

    def _extract_at_loc(self, locs: Iterable[Object],
                        ground_atoms: Iterable[GroundAtom]) -> Object:
        at_loc = self._extract_at_loc_optional(locs, ground_atoms)
        if at_loc is None:
            raise ApproachFailure(
                "Expected a concrete location")  # pragma: no cover
        return at_loc

    def _move(self, state: _PDDLEnvState, from_loc: Object,
              to_loc: Object) -> Action:
        ground_option = self.opt_move.ground([from_loc, to_loc],
                                             np.empty(0, dtype=np.float32))
        check_option(ground_option, state)
        return ground_option.policy(state)

    def _pick_up(self, state: _PDDLEnvState, paper: Object,
                 loc: Object) -> Action:
        ground_option = self.opt_pick_up.ground([paper, loc],
                                                np.empty(0, dtype=np.float32))
        check_option(ground_option, state)
        return ground_option.policy(state)

    def _deliver(self, state: _PDDLEnvState, paper: Object,
                 loc: Object) -> Action:
        ground_option = self.opt_deliver.ground([paper, loc],
                                                np.empty(0, dtype=np.float32))
        check_option(ground_option, state)
        return ground_option.policy(state)

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        # Extracting task-specific information
        assert isinstance(task.init, _PDDLEnvState)
        init_ground_atoms = task.init.get_ground_atoms()
        obj_locs = set(task.init.get_objects(self.type_loc))
        obj_papers = set(task.init.get_objects(self.type_paper))

        # Extracting initial predicates
        obj_safe_locs = filter_predicated_objs(obj_locs, self.pred_safe,
                                               init_ground_atoms)
        obj_home_bases = filter_predicated_objs(obj_locs,
                                                self.pred_is_home_base,
                                                init_ground_atoms)
        obj_home_base = min(obj_home_bases, default=None)

        # Extracting goal objects
        goal_at_loc = self._extract_at_loc_optional(obj_locs, task.goal)
        goal_satisfied = filter_predicated_objs(obj_locs, self.pred_satisfied,
                                                task.goal)
        goal_unpacked = filter_predicated_objs(obj_papers, self.pred_unpacked,
                                               task.goal)
        goal_carrying = filter_predicated_objs(obj_papers, self.pred_carrying,
                                               task.goal)

        # Joint object extraction
        obj_available_papers = obj_papers - goal_unpacked
        obj_satisfying_papers = obj_available_papers - goal_carrying

        # Sanity checks (satisfies all the "safe" and "home_base" goals)
        if not filter_predicated_objs(obj_locs, self.pred_safe,
                                      task.goal) <= obj_safe_locs:
            raise ApproachFailure(
                "Cannot make new safe locations")  # pragma: no cover
        if not filter_predicated_objs(obj_locs, self.pred_is_home_base,
                                      task.goal) <= obj_home_bases:
            raise ApproachFailure(
                "Cannot create new bases")  # pragma: no cover
        if goal_carrying:
            raise ApproachFailure(
                "Case not handled when" +
                "robot has to carry papers in the end")  # pragma: no cover

        def _policy(state: State) -> Action:
            state = cast(_PDDLEnvState, state)
            ground_atoms = state.get_ground_atoms()

            obj_at_loc = self._extract_at_loc(obj_locs, ground_atoms)
            obj_carried_papers = filter_predicated_objs(
                obj_papers, self.pred_carrying, ground_atoms)
            obj_locs_to_satisfy = goal_satisfied - filter_predicated_objs(
                obj_locs, self.pred_satisfied, ground_atoms)

            # Pick up enough satisfying papers
            if len(obj_carried_papers) < len(obj_locs_to_satisfy):
                if obj_home_base is None:
                    raise ApproachFailure(
                        "Need a home base")  # pragma: no cover
                if obj_at_loc != obj_home_base:
                    return self._move(state, obj_at_loc, obj_home_base)

                obj_satisfying_paper = min(obj_satisfying_papers -
                                           obj_carried_papers,
                                           default=None)
                if obj_satisfying_paper is None:
                    raise ApproachFailure(
                        "Too little paper that can satisfy locations"
                    )  # pragma: no cover
                return self._pick_up(state, obj_satisfying_paper,
                                     obj_home_base)

            # If already at a location to satisfy, satisfy it
            if obj_at_loc in obj_locs_to_satisfy:
                carried_paper = min(obj_carried_papers)
                return self._deliver(state, carried_paper, obj_at_loc)

            # Go to some location to satisfy if needed
            # (satisfies all the "satisfied" goals)
            for obj_loc_to_satisfy in obj_locs_to_satisfy:
                return self._move(state, obj_at_loc, obj_loc_to_satisfy)

            # Finally go to the desired goal location
            # (satisfies the "at_loc" goal)
            if goal_at_loc and goal_at_loc != obj_at_loc:
                return self._move(state, obj_at_loc, goal_at_loc)

            raise ApproachFailure(
                "Ran out of things to do")  # pragma: no cover

        return _policy
