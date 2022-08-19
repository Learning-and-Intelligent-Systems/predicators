"""An NSRT learning approach that collects and learns from data online.

Example command:
    python predicators/main.py --approach online_nsrt_learning --seed 0 \
        --env cover \
        --explorer glib \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 10 \
        --min_data_for_nsrt 10
"""
from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from typing import DefaultDict, FrozenSet, List, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.explorers import create_explorer
from predicators.settings import CFG
from predicators.structs import Dataset, GroundAtom, InteractionRequest, \
    InteractionResult, LiftedAtom, LowLevelTrajectory, Object, \
    ParameterizedOption, Predicate, Task, Type


class OnlineNSRTLearningApproach(NSRTLearningApproach):
    """OnlineNSRTLearningApproach implementation."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._dataset = Dataset([])
        self._online_learning_cycle = 0
        # Used for the novelty score function (in GLIB).
        self._novelty_counts: DefaultDict[FrozenSet[LiftedAtom], int] = \
            defaultdict(int)

    @classmethod
    def get_name(cls) -> str:
        return "online_nsrt_learning"

    def get_interaction_requests(self) -> List[InteractionRequest]:
        # Explore in the train tasks. The number of train tasks that are
        # explored at each timestep is a hyperparameter. The train task
        # is randomly selected.

        # Create the explorer. Note that greedy lookahead is not yet supported.
        preds = self._get_current_predicates()
        explorer = create_explorer(CFG.explorer,
                                   preds,
                                   self._initial_options,
                                   self._types,
                                   self._action_space,
                                   self._train_tasks,
                                   self._get_current_nsrts(),
                                   self._option_model,
                                   babble_predicates=preds,
                                   atom_score_fn=self._score_atoms_novelty)

        # Create the interaction requests.
        requests = []
        for _ in range(CFG.online_nsrt_learning_requests_per_cycle):
            # Select a random task (with replacement).
            task_idx = self._rng.choice(len(self._train_tasks))
            # Set up the explorer policy and termination function.
            policy, termination_function = explorer.get_exploration_strategy(
                task_idx, CFG.timeout)
            # Create the interaction request.
            req = InteractionRequest(train_task_idx=task_idx,
                                     act_policy=policy,
                                     query_policy=lambda s: None,
                                     termination_function=termination_function)
            requests.append(req)
        return requests

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Update the dataset with the offline data.
        for traj in dataset.trajectories:
            self._update_dataset(traj)
        super().learn_from_offline_dataset(dataset)

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        # Add the new data to the cumulative dataset.
        for result in results:
            traj = LowLevelTrajectory(result.states, result.actions)
            self._update_dataset(traj)
        # Re-learn the NSRTs.
        self._learn_nsrts(self._dataset.trajectories,
                          self._online_learning_cycle)
        # Advance the online learning cycle.
        self._online_learning_cycle += 1

    def _update_dataset(self, trajectory: LowLevelTrajectory) -> None:
        """Add a new low-level trajectory to the dataset."""
        # Update the dataset.
        self._dataset.append(trajectory)
        # Update the atom counts for the novelty score function (for GLIB).
        # NOTE: this assumes that predicates are not changing because we are
        # just running the predicate classifiers once per state for efficiency.
        assert not CFG.excluded_predicates  # make sure not predicate learning
        preds = self._get_current_predicates()
        assert preds == self._initial_predicates
        for state in trajectory.states:
            atoms = utils.abstract(state, preds)
            for s in range(CFG.glib_min_goal_size, CFG.glib_max_goal_size + 1):
                for atom_tup in itertools.combinations(atoms, s):
                    atom_set = set(atom_tup)
                    can_atom_set = self._get_canonical_lifted_atoms(atom_set)
                    self._novelty_counts[can_atom_set] += 1
        logging.debug(f"Novelty counts: {self._novelty_counts}")

    def _score_atoms_novelty(self, atoms: Set[GroundAtom]) -> float:
        """Score the novelty of a ground atom set, with higher better.

        Score based on the number of times that this atom set has been seen in
        the data, with object identities ignored (i.e., this is lifted).

        Assumes that the size of the atom set is between CFG.glib_min_goal_size
        and CFG.glib_max_goal_size (inclusive).
        """
        assert CFG.glib_min_goal_size <= len(atoms) <= CFG.glib_max_goal_size
        canonical_atoms = self._get_canonical_lifted_atoms(atoms)
        # Note minus sign: less frequent is better.
        return -self._novelty_counts[canonical_atoms]

    @staticmethod
    def _get_canonical_lifted_atoms(
            atoms: Set[GroundAtom]) -> FrozenSet[LiftedAtom]:
        """Create a canonical lifted atoms set.

        This is a helper for novelty scoring for GLIB.

        This is an efficient approximation of what we really care about, which
        is whether two atom sets unify. It's an approximation because there are
        tricky cases where the sorting procedure is ambiguous.
        """
        # Create a "signature" for each object, which will be used to break
        # ties when sorting based on predicates alone is not enough.
        objs = {o for a in atoms for o in a.objects}
        obj_to_sig = {
            o: tuple(sorted(a.predicate for a in atoms if o in a.objects))
            for o in objs
        }
        # Sort the atom set based first on predicates, then based on object
        # signature.
        key = lambda a: (a.predicate, tuple(obj_to_sig[o] for o in a.objects))
        sorted_atom_set = sorted(atoms, key=key)
        # Replace the objects with variables in order.
        sorted_objs: List[Object] = []
        for atom in sorted_atom_set:
            for obj in atom.objects:
                if obj not in sorted_objs:
                    sorted_objs.append(obj)
        variables = utils.create_new_variables([o.type for o in sorted_objs])
        sub = dict(zip(sorted_objs, variables))
        # Lift the atoms.
        lifted_atoms = frozenset(a.lift(sub) for a in sorted_atom_set)
        return lifted_atoms
