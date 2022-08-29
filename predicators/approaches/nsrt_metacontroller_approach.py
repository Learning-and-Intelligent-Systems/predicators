"""An abstract approach that implements a policy in the abstract transition
space instead of task planning.

For each ground NSRT, we sample continuous parameters until the expected
atoms check is satisfied, and use those to produce an option. The option
policy is executed in the environment, and the process repeats.
"""

import abc
from typing import Callable, Dict, Set

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.settings import CFG
from predicators.structs import Action, DummyOption, GroundAtom, State, Task, \
    _GroundNSRT, _Option


class NSRTMetacontrollerApproach(NSRTLearningApproach):
    """NSRTMetacontrollerApproach definition."""

    @abc.abstractmethod
    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom], memory: Dict) -> _GroundNSRT:
        """Get a next ground NSRT to refine.

        This is the main method defining the metacontroller.
        """
        raise NotImplementedError("Override me!")

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        cur_option = DummyOption
        memory: Dict = {}  # optionally updated by predict()

        def _policy(state: State) -> Action:
            atoms = utils.abstract(state, self._initial_predicates)
            nonlocal cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                ground_nsrt = self._predict(state, atoms, task.goal, memory)
                cur_option = self._sample_option_from_nsrt(
                    ground_nsrt, state, atoms, task.goal)
            act = cur_option.policy(state)
            return act

        return _policy

    def _sample_option_from_nsrt(self, ground_nsrt: _GroundNSRT, state: State,
                                 atoms: Set[GroundAtom],
                                 goal: Set[GroundAtom]) -> _Option:
        """Given a ground NSRT, try invoking its sampler repeatedly until we
        find an option that produces the expected next atoms under the ground
        NSRT."""
        for _ in range(CFG.metacontroller_max_samples):
            # Invoke the ground NSRT's sampler to produce an option.
            opt = ground_nsrt.sample_option(state, goal, self._rng)
            if not opt.initiable(state):
                # The option is not initiable. Continue on to the next sample.
                continue
            try:
                next_state, _ = \
                    self._option_model.get_next_state_and_num_actions(state,
                                                                      opt)
            except utils.EnvironmentFailure:
                continue
            expected_next_atoms = utils.apply_operator(ground_nsrt, atoms)
            if not all(a.holds(next_state) for a in expected_next_atoms):
                # Some expected atom is not achieved. Continue on to the
                # next sample.
                continue
            return opt
        raise ApproachFailure("Metacontroller could not sample an option")
