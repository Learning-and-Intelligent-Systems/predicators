"""Create offline datasets for training, given a set of training tasks for an
environment."""

from typing import List, Set

from predicators import utils
from predicators.datasets.demo_only import create_demo_data
from predicators.datasets.demo_replay import create_demo_replay_data
from predicators.datasets.generate_atom_trajs_with_vlm import \
    create_ground_atom_data_from_generated_demos, \
    create_ground_atom_data_from_labelled_txt, \
    create_ground_atom_data_from_saved_img_trajs
from predicators.datasets.ground_atom_data import create_ground_atom_data
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Dataset, ParameterizedOption, Predicate, Task


def create_dataset(env: BaseEnv, train_tasks: List[Task],
                   known_options: Set[ParameterizedOption],
                   known_predicates: Set[Predicate]) -> Dataset:
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
    if CFG.offline_data_method in ["demo_with_vlm_imgs", "geo_and_vlm"]:  # pragma: no cover  # pylint:disable=line-too-long
        # NOTE: this below method is tested separately; it's just that testing
        # it by calling the above function is painful because a VLM is
        # instantiated and called from inside this method, but when testing,
        # we want to instantiate our own 'dummy' VLM.
        # NOTE: this data generation method is currently not compatible with
        # option learning because it will modify dataset trajectories to
        # remove a number of intermediate states when an option was being
        # executed. Thus, we assert this before doing anything further.
        assert CFG.option_learner == "no_learning", \
            ("offline data method demo_with_vlm_imgs only compatible with the"
            "'no_learning' option learner.")
        # First, we call create_demo_data to create a dataset.
        demo_data = create_demo_data(env,
                                     train_tasks,
                                     known_options,
                                     annotate_with_gt_ops=False)
        assert len(demo_data.trajectories) == len(train_tasks), (
            "Cannot run "
            "VLM-based predicate invention if we don't have one demo per "
            "training task; ensure there are no failures in demonstration "
            "generation.")
        # Second, we add annotations to these trajectories by leveraging
        # a VLM.
        return create_ground_atom_data_from_generated_demos(
            demo_data, env, known_predicates, train_tasks)
    if CFG.offline_data_method == "demo+labelled_atoms":
        return create_ground_atom_data_from_labelled_txt(
            env, train_tasks, known_options)
    if CFG.offline_data_method == "saved_vlm_img_demos_folder":  # pragma: no cover  # pylint:disable=line-too-long
        # NOTE: this below method is tested separately; it's just that testing
        # it by calling the above function is painful because a VLM is
        # instantiated and called from inside this method, but when testing,
        # we want to instantiate our own 'dummy' VLM.
        return create_ground_atom_data_from_saved_img_trajs(
            env, train_tasks, known_predicates, known_options)
    if CFG.offline_data_method == "empty":
        return Dataset([])
    raise NotImplementedError("Unrecognized dataset method.")
