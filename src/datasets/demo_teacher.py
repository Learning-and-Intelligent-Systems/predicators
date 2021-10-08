"""Create offline datasets by TODO
"""

import numpy as np
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset
from predicators.src.datasets.demo_only import create_demo_data
from predicators.src.settings import CFG
from predicators.src import utils


def create_demo_teacher_data(env: BaseEnv, frac: float=0.5) -> Dataset:
    """Create offline datasets by TODO
    """
    demo_dataset = create_demo_data(env)    
    ground_atoms_dataset = []
    for (ss, _) in demo_dataset:  # ss is List[State]
        ground_atoms_traj = []
        for s in ss:
            ground_atoms = utils.abstract(s, env.predicates)
            # select random subset to keep
            rng = np.random.default_rng(CFG.seed)
            n_samples = len(ground_atoms) * frac
            subset = rng.choice(ground_atoms, size=(n_samples,), replace=False)
            ground_atoms_traj.append(subset)
        ground_atoms_dataset.append(ground_atoms_traj)    
    assert len(ground_atoms_dataset) == len(demo_dataset)

    return ground_atoms_dataset
