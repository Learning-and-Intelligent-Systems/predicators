"""Create offline data by annotating low-level trajectories with ground atoms
that hold in the respective states."""

from typing import List, Set
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Predicate, GroundAtom
from predicators.src import utils


def create_ground_atom_data(env: BaseEnv, base_dataset: Dataset,
                            annotating_predicates: Set[Predicate],
                            annotating_ratio: float) -> Dataset:
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
    labeleds = {p: 0 for p in annotating_predicates}
    totals = {p: 0 for p in annotating_predicates}
    annotations: List[List[Set[GroundAtom]]] = []
    for traj in base_dataset.trajectories:
        ground_atoms_traj: List[Set[GroundAtom]] = []
        for s in traj.states:
            ground_atoms = sorted(utils.abstract(s, annotating_predicates))
            subset_atoms = set()
            if annotating_ratio > 0:
                for ga in ground_atoms:
                    pred = ga.predicate
                    if (totals[pred] == 0 or \
                            labeleds[pred] / totals[pred] <= annotating_ratio):
                        # Teacher comments on this atom
                        stripped_pred = predicates_to_stripped[pred]
                        stripped_ga = GroundAtom(stripped_pred, ga.objects)
                        subset_atoms.add(stripped_ga)
                        labeleds[pred] += 1
                    totals[pred] += 1
            ground_atoms_traj.append(subset_atoms)
        assert len(traj.states) == len(ground_atoms_traj)
        annotations.append(ground_atoms_traj)
    assert len(annotations) == len(base_dataset.trajectories)
    return Dataset(base_dataset.trajectories, annotations)
