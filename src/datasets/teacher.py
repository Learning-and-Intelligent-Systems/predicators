"""Create teacher dataset for interactive learning.
"""

from typing import List, Set
import numpy as np
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, GroundAtom
from predicators.src.settings import CFG
from predicators.src import utils


def create_teacher_dataset(env: BaseEnv,
                           dataset: Dataset) -> List[List[Set[GroundAtom]]]:
    """Create sparse dataset of GroundAtoms for interactive learning.
    """
    frac = CFG.frac
    rng = np.random.default_rng(CFG.seed)
    ground_atoms_dataset = []
    for (ss, _) in dataset:
        ground_atoms_traj = []
        for s in ss:
            ground_atoms = list(utils.abstract(s, env.predicates))
            # select random subset to keep
            n_samples = int(len(ground_atoms) * frac)
            subset = rng.choice(np.arange(len(ground_atoms)),
                                size=(n_samples,),
                                replace=False)
            subset_atoms = {ground_atoms[j] for j in subset}
            ground_atoms_traj.append(subset_atoms)
        ground_atoms_dataset.append(ground_atoms_traj)
    assert len(ground_atoms_dataset) == len(dataset)
    return ground_atoms_dataset
