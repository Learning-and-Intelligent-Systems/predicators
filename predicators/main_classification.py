"""Example command:"""
import glob
import logging
import os
import sys
import time
from collections import defaultdict
from typing import List, Optional, Union

import dill as pkl
from PIL import Image

from predicators import utils
from predicators.classification_approaches import DinoSimilarityApproach, \
    VLMClassificationApproach
from predicators.settings import CFG
from predicators.structs import ClassificationDataset, Metrics, Video


def main() -> None:
    """Main entry point for running classification approaches."""
    script_start = time.perf_counter()

    # Parse args
    args = utils.parse_args()
    utils.update_config(args)
    str_args = " ".join(sys.argv)

    # Set up logging
    if CFG.log_file:
        CFG.log_file = os.path.join(CFG.log_dir, CFG.approach,
                                    f"seed{CFG.seed}", "")
    utils.configure_logging()
    os.makedirs(CFG.results_dir, exist_ok=True)
    os.makedirs(CFG.eval_trajectories_dir, exist_ok=True)

    # Log initial info
    utils.log_initial_info(str_args)

    # # Setup environment
    # env, approach_train_tasks, train_tasks = setup_environment()

    # # Setup predicates
    # included_preds, excluded_preds = \
    #     utils.parse_config_excluded_predicates(env)
    # preds = utils.replace_goals_with_agent_specific_goals(
    #     included_preds, excluded_preds, env
    #     ) if CFG.approach != "oracle" else included_preds

    # --- Create dataset
    # In a meta learning setting, we have meta-train and
    # meta-test datasets but we only have meta-test now.
    # Each dataset contains multiple tasks. Each task
    # contains a support and query set.
    # For now, there are 1-2 support videos and 2 query
    # videos per task.
    # ---
    # Alternatively, with the current design, there is
    # just 1 kind of counterfactual per env. So we only
    # have 1 task in the meta-test split.
    # In each task, we will have 1 or more training
    # samples and multiple test samples.
    # Each sample will have a (state, action) traj and a
    # label for whether it's from the standard world.
    test_dataset = create_dataset()

    # Create approach
    # approach = setup_approach(env, preds, approach_train_tasks)
    approach: Union[VLMClassificationApproach, DinoSimilarityApproach]
    if CFG.approach == "vlm_classification":
        approach = VLMClassificationApproach()
    elif CFG.approach == "dino_similarity":
        approach = DinoSimilarityApproach()
    else:
        raise ValueError(f"Unknown approach: {CFG.approach}")

    _run_pipeline(approach, test_dataset)

    # Log completion
    script_time = time.perf_counter() - script_start
    logging.info(f"\n\nMain script completed in {script_time:.2f} seconds.")


def create_dataset() -> ClassificationDataset:
    """Create training and test datasets for classification.

    A dataset has many episodes. Each is 1-2 support videos with labels
    and 2 query videos with labels.
    """
    all_task_names: List[str] = []
    all_support_videos: List[List[Video]] = []
    all_support_labels: List[List[int]] = []
    all_query_videos: List[List[Video]] = []
    all_query_labels: List[List[int]] = []
    max_video_len = 0

    env_names = [
        "cover",
        "blocks",
        "coffee",
        "balance",
        "grow",
        "circuit",
        "float",
        "domino",
        "laser",
        "ants",
        "fan",
    ]
    for env in env_names:
        episode_support_videos: List[Video] = []
        episode_support_labels: List[int] = []
        episode_query_videos: List[Video] = []
        episode_query_labels: List[int] = []
        for support_split in [True, False]:
            if CFG.classification_has_counterfactual_support or \
                    not support_split:
                is_counterfactual_list = [False, True]
            else:
                is_counterfactual_list = [False]

            for is_counterfactual in is_counterfactual_list:
                split = "support" if support_split else "query"
                base_env_name = env if not is_counterfactual else f"{env}_cf"
                # Note: so far only have seed0
                dataset_base_dir = os.path.join(CFG.image_dir, base_env_name,
                                                "seed0", split)

                logging.debug(f"Loading the {split} set for "
                              f"{base_env_name}...")
                for task_dir in glob.glob(
                        os.path.join(dataset_base_dir, 'task*')):
                    # Get all images
                    img_paths = sorted(glob.glob(
                        os.path.join(task_dir, '*.png')),
                                       key=os.path.basename)

                    video_len = len(img_paths)
                    if video_len > max_video_len:
                        max_video_len = video_len
                    video = []

                    for img_path in img_paths:
                        with Image.open(img_path) as img:
                            video.append(img.copy())
                    if support_split:
                        episode_support_videos.append(video)
                        episode_support_labels.append(
                            int(not is_counterfactual))
                    else:
                        episode_query_videos.append(video)
                        episode_query_labels.append(int(not is_counterfactual))
        assert len(episode_support_videos) == 1, \
                "Currently assume only 1 support video."
        all_task_names.append(env)
        all_support_videos.append(episode_support_videos)
        all_support_labels.append(episode_support_labels)
        all_query_videos.append(episode_query_videos)
        all_query_labels.append(episode_query_labels)

    logging.debug(f"Max video length: {max_video_len}")
    return ClassificationDataset(all_task_names, all_support_videos,
                                 all_support_labels, all_query_videos,
                                 all_query_labels, CFG.seed)


def _run_testing(approach: Union[VLMClassificationApproach,
                                 DinoSimilarityApproach],
                 test_dataset: ClassificationDataset) -> Metrics:
    num_correct = 0
    num_episodes = len(test_dataset)
    metrics: Metrics = defaultdict(float)

    for i, episode in enumerate(test_dataset):
        (episode_name, support_videos, support_labels, query_videos,
         query_labels) = episode

        pred_labels = approach.predict(episode_name,
                                       support_videos,
                                       support_labels,
                                       query_videos,
                                       task_id=i)
        correct = pred_labels == query_labels
        num_correct += int(correct)
        logging.debug(f"Ep. {i}: {episode_name}: pred: {pred_labels}, "
                      f"true: {query_labels}. "
                      f"Correct: {correct}")
        # Can either do the average here or during plotting
        metrics[f"{episode_name}_accuracy"] = float(correct)

    metrics["num_correct"] = num_correct
    metrics["num_episodes"] = num_episodes
    accuracy = num_correct / num_episodes
    metrics["avg_accuracy"] = accuracy
    logging.info(f"Accuracy: {num_correct}/{num_episodes} ({accuracy:.2f})")

    return metrics


def _save_test_results(results: Metrics,
                       online_learning_cycle: Optional[int] = None) -> None:
    """Save the test results."""
    outfile = (f"{CFG.results_dir}/{utils.get_config_path_str()}__"
               f"{online_learning_cycle}.pkl")
    outdata = {
        "config": CFG,
        "results": results.copy(),
        "git_commit_hash": utils.get_git_commit_hash()
    }
    with open(outfile, "wb") as f:
        pkl.dump(outdata, f)

    logging.info("-------------------")
    logging.info(f"Test results: {results}")
    logging.info(f"Wrote out test results to {outfile}")


def _run_pipeline(approach: Union[VLMClassificationApproach,
                                  DinoSimilarityApproach],
                  test_dataset: ClassificationDataset) -> None:
    """Run the classification pipeline."""
    results = _run_testing(approach, test_dataset)
    _save_test_results(results)


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except Exception as _err:  # pylint: disable=broad-except
        logging.exception("main_classification.py crashed")
        raise _err
