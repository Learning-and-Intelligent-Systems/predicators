"""A hand-written LDL bridge policy."""

from typing import Callable, Set

from predicators.bridge_policies.base_bridge_policy import BaseBridgePolicy
from predicators.ground_truth_models import get_gt_ldl_bridge_policy
from predicators.settings import CFG
from predicators.structs import NSRT, LiftedDecisionList, \
    ParameterizedOption, Predicate, Type, State, _Option
from predicators import utils


class ReeceBridgePolicy(BaseBridgePolicy):
    """A hand-written LDL bridge policy."""

    @classmethod
    def get_name(cls) -> str:
        return "reece_bridge_policy"

    @property
    def is_learning_based(self) -> bool:
        return False

    def get_option_policy(self) -> Callable[[State], _Option]:
        
        def _option_policy(state: State) -> _Option:
            print("State:")
            print(state.pretty_str())
            atoms = utils.abstract(state, self._predicates)
            print("Abstract state:", atoms)
            print("Failed option history:", self._failed_options)
            # Do something here...

            # Temporary crap
            nsrt_name_to_nsrt = {n.name: n for n in self._nsrts}
            PickStickFromButton = nsrt_name_to_nsrt["PickStickFromButton"]
            robot_above_button_atoms = {a for a in atoms if a.predicate.name == "RobotAboveButton"}
            assert len(robot_above_button_atoms) == 1
            robot, above_button = next(iter(robot_above_button_atoms)).objects
            stick_type = next(iter({t for t in self._types if t.name == "stick"}))
            stick, = state.get_objects(stick_type)
            ground_nsrt = PickStickFromButton.ground([robot, stick, above_button])
            option = ground_nsrt.sample_option(state, set(), self._rng)

            # Return option
            print(option)
            return option

        return _option_policy
    