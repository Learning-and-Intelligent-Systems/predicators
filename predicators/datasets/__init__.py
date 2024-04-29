"""Create offline datasets for training, given a set of training tasks for an
environment."""

from typing import List, Set

from predicators import utils
from predicators.datasets.demo_only import create_demo_data
from predicators.datasets.demo_replay import create_demo_replay_data
from predicators.datasets.generate_atom_trajs_with_vlm import \
    create_ground_atom_data_from_img_trajs, \
    create_ground_atom_data_from_labelled_txt
from predicators.datasets.ground_atom_data import create_ground_atom_data
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Dataset, ParameterizedOption, Task


def create_dataset(env: BaseEnv, train_tasks: List[Task],
                   known_options: Set[ParameterizedOption]) -> Dataset:
    """Create offline datasets for training, given a set of training tasks for
    an environment.

    Some or all of this data may be loaded from disk.
    """
    if CFG.offline_data_method == "demo":
        return create_demo_data(env,
                                train_tasks,
                                known_options,
                                annotate_with_gt_ops=False)
    if CFG.offline_data_method == "demo+gt_operators":
        return create_demo_data(env,
                                train_tasks,
                                known_options,
                                annotate_with_gt_ops=True)
    if CFG.offline_data_method == "demo+replay":
        return create_demo_replay_data(env, train_tasks, known_options)
    if CFG.offline_data_method == "demo+ground_atoms":
        base_dataset = create_demo_data(env,
                                        train_tasks,
                                        known_options,
                                        annotate_with_gt_ops=False)
        _, excluded_preds = utils.parse_config_excluded_predicates(env)
        n = int(CFG.teacher_dataset_num_examples)
        assert n >= 1, "Must have at least 1 example of each predicate"
        return create_ground_atom_data(env, base_dataset, excluded_preds, n)
    if CFG.offline_data_method == "demo+labelled_atoms":
        return create_ground_atom_data_from_labelled_txt(
            env, train_tasks, known_options)
    if CFG.offline_data_method == "img_demos":  # pragma: no cover.
        # NOTE: this below method is tested separately; it's just that testing
        # it by calling the above function is painful because a VLM is
        # instantiated and called from inside this method, but when testing,
        # we want to instantiate our own 'dummy' VLM.
        return create_ground_atom_data_from_img_trajs(env, train_tasks,
                                                      known_options)
    if CFG.offline_data_method == "empty":
        return Dataset([])
    raise NotImplementedError("Unrecognized dataset method.")
