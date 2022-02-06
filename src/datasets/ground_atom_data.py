"""Create offline data by annotating low-level trajectories with ground atoms
that hold in the respective states."""

import numpy as np
from typing import DefaultDict, List, Set, Dict, Tuple
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Predicate, GroundAtom
from predicators.src import utils


def create_ground_atom_data(env: BaseEnv, base_dataset: Dataset,
                            annotating_predicates: Set[Predicate],
                            num_examples: int) -> Dataset:
    """Create offline data by annotating low-level trajectories with ground
    atoms. The ground atoms must hold in the respective states. Only the
    predicates in annotating_predicates are used in the annotations.

    The base_dataset is the initial dataset (e.g. demos) of unannotated
    low-level trajectories.
    """
    # Do not leak the classifier in the dataset. We want the names and args of
    # the predicates only in the annotations.
    assert annotating_predicates.issubset(env.predicates)
    predicates_to_stripped = {
        p: utils.strip_predicate(p)
        for p in annotating_predicates
    }
    # Generate all positive and negative examples
    pos_examples = DefaultDict(list)  # predicate: [(i, j, ground_atom)]
    neg_examples = DefaultDict(list)
    for i, traj in enumerate(base_dataset.trajectories):
        for j, s in enumerate(traj.states):
            ground_atoms = utils.abstract(s, annotating_predicates)
            ground_atom_universe = utils.all_possible_ground_atoms(s, annotating_predicates)
            for ga in ground_atom_universe:
                if ga in ground_atoms:  # positive example
                    pos_examples[ga.predicate].append((i, j, ga))
                else:
                    neg_examples[ga.predicate].append((i, j, ga))
    # Sample `num_examples` from each list
    pos_picks: List[Tuple] = []
    neg_picks: List[Tuple] = []
    for p in annotating_predicates:
        for examples, picks in zip((pos_examples, neg_examples), (pos_picks, neg_picks)):
            idxs = np.random.choice(len(examples[p]), size=num_examples, replace=False)
            picks.extend([examples[p][idx] for idx in idxs])
    annotations: List[List[List[Set[GroundAtom]]]] = [[[set(), set()] for _ in traj.states] for traj in base_dataset.trajectories]
    for label, picks in enumerate([neg_picks, pos_picks]):
        for (i, j, ga) in picks:
            stripped_pred = predicates_to_stripped[ga.predicate]
            stripped_ga = GroundAtom(stripped_pred, ga.objects)
            annotations[i][j][label].add(stripped_ga)
    return Dataset(base_dataset.trajectories, annotations)
