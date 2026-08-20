"""VLM-based classification approach."""
import logging
import os
import random
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import PIL

from predicators import utils
from predicators.settings import CFG
from predicators.structs import Video


class VLMClassificationApproach:
    """Vision Language Model (VLM) based classification approach.

    This class implements a classification approach using a Vision
    Language Model to compare query videos against support videos
    and determine their similarity.

    Attributes:
        _vlm: The vision language model instance
        _max_video_len: Maximum number of frames to use from each video
        log_dir: Directory for saving intermediate results and model outputs
    """

    def __init__(self) -> None:
        """Initialize the VLM classification approach."""
        self._vlm = utils.create_vlm_by_name(CFG.vlm_model_name)
        self._max_video_len = 10  # Maximum frames to process per video
        self.log_dir: Optional[str] = None

    @classmethod
    def approach_name(cls) -> str:
        """Return the name of this classification approach."""
        return "vlm_classification"

    def predict(self, episode_name: str, support_videos: List[Video],
                support_labels: List[int], query_videos: List[Video],
                task_id: int) -> List[int]:
        """Predict labels for query videos based on support videos.

        Args:
            support_videos: Reference videos for comparison
            support_labels: Labels corresponding to support videos
            query_videos: Videos to be classified
            task_id: Unique identifier for the classification task

        Returns:
            List of predicted labels (0 or 1) for query videos
        """
        # Setup logging directory for this task
        self._setup_logging_dir(task_id)

        # Preprocess videos to manageable length
        support_videos, query_videos = self._preprocess_videos(
            support_videos, query_videos)

        # Generate VLM prompt and prepare images
        prompt, imgs = self._prepare_prompt(episode_name, support_videos,
                                            support_labels, query_videos)

        # Get and parse VLM response
        response = self._vlm.sample_completions(prompt, imgs,
                                                CFG.vlm_temperature,
                                                CFG.seed)[0]
        parsed_response = self._save_and_parse_vlm_response(response)

        # Convert parsed response to classification labels
        return self._convert_response_to_labels(parsed_response, query_videos)

    def _setup_logging_dir(self, task_id: int) -> None:
        """Setup and clean the logging directory for the current task."""
        self.log_dir = os.path.join(CFG.log_dir, self.approach_name(),
                                    f"seed{CFG.seed}", f"task{task_id}")
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

    def _convert_response_to_labels(self, response: Dict[str, Any],
                                    query_videos: List[Video]) -> List[int]:
        """Convert VLM response to binary classification labels."""
        clf_answer = [1, 0
                      ] if response["matching_video"] == "query_1" else [0, 1]
        assert len(clf_answer) == len(query_videos), "Answer length mismatch."
        return clf_answer

    def _prepare_prompt(
        self,
        episode_name: str,
        support_videos: List[Video],
        support_labels: List[int],
        query_videos: List[Video],
    ) -> Tuple[str, List[PIL.Image.Image]]:
        """Prepare the prompt for the VLM by:

        1. Load the prompt from file
        2. add labels to the videos
        """
        del support_labels

        # --- Prepare the prompt ---
        prompt_path = os.path.join("prompts", "classification.outline")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # Replace placeholders in the prompt
        prompt = prompt_template.format(ENV_NAME=episode_name)

        # save prompt
        self._save_text_to_logdir(prompt, "prompt.txt")

        # --- Prepare the images ---
        # Create a directory to save the images.
        imgs_dir = os.path.join(self.log_dir or "", "imgs")

        # Save for later inspection
        support_videos = [
            utils.add_label_to_video(video,
                                     prefix="ref_",
                                     imgs_dir=imgs_dir,
                                     save=True) for video in support_videos
        ]
        assert len(
            support_videos) == 1, "Currently assume only 1 support video."
        query_videos = [
            utils.add_label_to_video(video,
                                     prefix=f"query{i+1}_",
                                     imgs_dir=imgs_dir,
                                     save=True)
            for i, video in enumerate(query_videos)
        ]
        imgs = [
            img for video in support_videos + query_videos for img in video
        ]

        return prompt, imgs  # type: ignore[return-value]

    def _save_and_parse_vlm_response(self,
                                     response_text: str) -> Dict[str, str]:
        """Parse and save the VLM's response.

        Args:
            response_text: Raw response from the VLM

        Returns:
            Dictionary containing:
                - matching_video: Either 'query_1' or 'query_2'
                - reasoning: Explanation for the choice

        Raises:
            ValueError: If response cannot be parsed correctly
        """
        # Save response for debugging
        self._save_text_to_logdir(response_text, "response.txt")

        # Extract matching video and reasoning using regex
        match_video = re.search(r"%% Matching Video:\s*(query_1|query_2)",
                                response_text)
        match_reasoning = re.search(
            r"%% Reasoning:\s*(.*?)(?=\n%% Matching Video:|$)", response_text,
            re.DOTALL)

        if not match_video:
            logging.warning("Could not find matching video in response.")
            answer = random.choice(["query_1", "query_2"])
        else:
            answer = match_video.group(1)

        return {
            "matching_video":
            answer,
            "reasoning": (match_reasoning.group(1).strip()
                          if match_reasoning else "No reasoning provided.")
        }

    def _save_text_to_logdir(self, response_text: str, fname: str) -> None:
        """Save VLM response to file for debugging purposes."""
        response_path = os.path.join(self.log_dir or "", fname)
        os.makedirs(os.path.dirname(response_path), exist_ok=True)
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(response_text)

    def _preprocess_videos(
            self, support_videos: List[Video],
            query_videos: List[Video]) -> Tuple[List[Video], List[Video]]:
        """Preprocess the support and query videos.

        Subsample the videos to the max_video_len.
        """

        # Subsample the frames of the videos to the max_video_len.
        def subsample_video(video: Video) -> Video:
            if len(video) <= self._max_video_len:
                return video
            # Always include first and last frame, sample the rest
            step = (len(video) - 1) / (self._max_video_len - 1)
            sampled = [
                video[int(i * step)] for i in range(self._max_video_len - 1)
            ]
            return sampled + [video[-1]]

        support_videos = [subsample_video(video) for video in support_videos]
        query_videos = [subsample_video(video) for video in query_videos]

        return support_videos, query_videos
