"""Create offline datasets for training, given a set of training tasks for an
environment."""

from typing import List
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Task
from predicators.src.datasets.demo_only import create_demo_data
from predicators.src.datasets.demo_replay import create_demo_replay_data
from predicators.src.datasets.ground_atom_data import create_ground_atom_data
from predicators.src.settings import CFG


def create_dataset(env: BaseEnv, train_tasks: List[Task]) -> Dataset:
    """Create offline datasets for training, given a set of training tasks for
    an environment."""
    if CFG.offline_data_method == "demo":
        return create_demo_data(env, train_tasks)
    if CFG.offline_data_method == "demo+replay":
        return create_demo_replay_data(env, train_tasks)
    if CFG.offline_data_method == "demo+nonoptimalreplay":
        return create_demo_replay_data(env, train_tasks, nonoptimal_only=True)
    if CFG.offline_data_method == "demo+ground_atoms":
        base_dataset = create_demo_data(env, train_tasks)
        known_predicate_names = CFG.interactive_known_predicates.split(",")
        predicates_to_learn = {
            p
            for p in env.predicates if p.name not in known_predicate_names
        }
        ratio = CFG.teacher_dataset_label_ratio
        return create_ground_atom_data(env, base_dataset, predicates_to_learn,
                                       ratio)
    raise NotImplementedError("Unrecognized dataset method.")
