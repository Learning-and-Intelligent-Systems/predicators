"""Create offline datasets for training, given a set of training tasks for an
environment."""

from typing import List, Set

from predicators.src import utils
from predicators.src.datasets.demo_only import create_demo_data
from predicators.src.datasets.demo_replay import create_demo_replay_data
from predicators.src.datasets.ground_atom_data import create_ground_atom_data
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Dataset, ParameterizedOption, Task


def create_dataset(env: BaseEnv, train_tasks: List[Task],
                   known_options: Set[ParameterizedOption]) -> Dataset:
    """Create offline datasets for training, given a set of training tasks for
    an environment.

    Some or all of this data may be loaded from disk.
    """
    if CFG.offline_data_method == "demo":
        return create_demo_data(env, train_tasks, known_options)
    if CFG.offline_data_method == "demo+replay":
        return create_demo_replay_data(env, train_tasks, known_options)
    if CFG.offline_data_method == "demo+ground_atoms":
        base_dataset = create_demo_data(env, train_tasks, known_options)
        _, excluded_preds = utils.parse_config_excluded_predicates(env)
        n = int(CFG.teacher_dataset_num_examples)
        assert n >= 1, "Must have at least 1 example of each predicate"
        return create_ground_atom_data(env, base_dataset, excluded_preds, n)
    if CFG.offline_data_method == "empty":
        return Dataset([])
    raise NotImplementedError("Unrecognized dataset method.")
