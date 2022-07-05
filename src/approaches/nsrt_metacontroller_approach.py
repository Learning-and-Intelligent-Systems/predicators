"""An abstract approach that implements a policy in the abstract transition
space instead of task planning.

For each ground NSRT, we sample continuous parameters until the expected
atoms check is satisfied, and use those to produce an option. The option
policy is executed in the environment, and the process repeats.
"""

import abc
from typing import Callable, Set, FrozenSet
import time
from predicators.src import utils
from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.settings import CFG
from predicators.src.structs import Action, DummyOption, GroundAtom, State, \
    Task, _GroundNSRT, _Option
from predicators.src import planning


class NSRTMetacontrollerApproach(NSRTLearningApproach):
    """NSRTMetacontrollerApproach definition."""

    @abc.abstractmethod
    def _predict(self, state: State, atoms: Set[GroundAtom],
                 goal: Set[GroundAtom]) -> _GroundNSRT:
        """Get a next ground NSRT to refine.

        This is the main method defining the metacontroller.
        """
        raise NotImplementedError("Override me!")

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        cur_option = DummyOption
        _S: TypeAlias = FrozenSet[GroundAtom]
        _A: TypeAlias = _GroundNSRT
        skeleton = []
        atoms_sequence = []
        expected_atoms = []

        def get_next_state(atoms: _S, ground_nsrt: _A) -> _S:
            return frozenset(utils.apply_operator(ground_nsrt, set(atoms)))

        state = task.init
        atoms = utils.abstract(state, self._initial_predicates)
        goal = task.goal.copy().pop()
        atoms_sequence.append(atoms)
        start_time = time.time()

        if cur_option is DummyOption or cur_option.terminal(state):
            while goal not in expected_atoms:
                ground_nsrt = self._predict(state, atoms, task.goal)
                expected_atoms = utils.apply_operator(ground_nsrt, atoms)
                skeleton.append(ground_nsrt)
                atoms_sequence.append(expected_atoms)
                state = get_next_state(atoms, ground_nsrt)
                atoms = expected_atoms
            option_list, _ = planning._run_low_level_search(task,
                self._option_model, skeleton, atoms_sequence,
                self._seed, timeout - (time.time() - start_time),
                CFG.horizon)
        policy = utils.option_plan_to_policy(option_list)
        return policy

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
        #print("\n")
        #print(ground_nsrt)
        raise ApproachFailure("Metacontroller could not sample an option")
