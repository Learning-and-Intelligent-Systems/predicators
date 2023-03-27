"""A hand-written bridge policy."""

import logging
from typing import Callable, List, Set, Dict
import numpy as np

from predicators.bridge_policies import BaseBridgePolicy, BridgePolicyDone
from predicators.envs import BaseEnv, get_or_create_env
from predicators.settings import CFG
from predicators.structs import GroundAtom, State, Task, _GroundNSRT, _Option, Action, BridgePolicy, NSRT, DummyOption, Predicate
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators import utils


class OracleBridgePolicy(BaseBridgePolicy):
    """A hand-written bridge policy."""

    def __init__(self) -> None:
        # TODO initialize with predicates and NSRTs instead
        super().__init__()
        env = get_or_create_env(CFG.env)
        options = get_gt_options(CFG.env)
        self._predicates = set(env.predicates)
        self._nsrts = get_gt_nsrts(CFG.env, self._predicates, options)
        self._rng = np.random.default_rng(CFG.seed)
        self._oracle_bridge_policy = _create_oracle_bridge_policy(CFG.env, self._nsrts, self._predicates, self._rng)

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    @property
    def is_learning_based(self) -> bool:
        return False

    def get_policy(self, failed_nsrt: _GroundNSRT) -> Callable[[State], Action]:

        # Create action policy.
        cur_option = DummyOption

        def _policy(state: State) -> Action:
            nonlocal cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                atoms = utils.abstract(state, self._predicates)
                cur_option = self._oracle_bridge_policy(state, atoms, failed_nsrt)
                if not cur_option.initiable(state):
                    raise OptionExecutionFailure("Bridge option not initiable.")
            act = cur_option.policy(state)
            return act

        return _policy


def _create_oracle_bridge_policy(env_name: str, nsrts: Set[NSRT], predicates: Set[Predicate], rng: np.random.Generator) -> BridgePolicy:
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    pred_name_to_pred = {p.name: p for p in predicates}

    if env_name == "painting":
        return _create_painting_oracle_bridge_policy(nsrt_name_to_nsrt, pred_name_to_pred, rng)
    raise NotImplementedError(f"No oracle bridge policy for {env_name}")


def _create_painting_oracle_bridge_policy(nsrt_name_to_nsrt: Dict[str, NSRT], pred_name_to_pred: Dict[str, Predicate], rng: np.random.Generator) -> BridgePolicy:

    PlaceOnTable = nsrt_name_to_nsrt["PlaceOnTable"]
    OpenLid = nsrt_name_to_nsrt["OpenLid"]

    GripperOpen = pred_name_to_pred["GripperOpen"]

    def _bridge_policy(state: State, atoms: Set[GroundAtom], failed_nsrt: _GroundNSRT) -> _Option:
        assert failed_nsrt.name == "PlaceInBox"
        held_obj, box, robot = failed_nsrt.objects
        lid = next(o for o in state if o.type.name == "lid")

        # If the box lid is already open, the bridge policy is done.
        if state.get(lid, "is_open") > 0.5:
            raise BridgePolicyDone()

        if GripperOpen.holds(state, [robot]):
            next_nsrt = OpenLid.ground([lid, robot])
        else:
            next_nsrt = PlaceOnTable.ground([held_obj, robot])

        logging.debug(f"Using NSRT {next_nsrt.name}{next_nsrt.objects} "
                      "from bridge policy.")

        goal = set()  # goal assumed not used by sampler
        return next_nsrt.sample_option(state, goal, rng)

    return _bridge_policy
