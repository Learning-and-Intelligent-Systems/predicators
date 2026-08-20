"""DINO similarity-based classification approach."""
import logging
import os
import shutil
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from predicators.settings import CFG
from predicators.structs import Video


###############################################################################
# Example DTW implementation (for completeness).
# If you already have a DTW function or library, you can plug that in directly.
###############################################################################
def _dtw_distance(seqA: np.ndarray, seqB: np.ndarray) -> float:
    """Compute a simple DTW distance between two sequences of embeddings.

    seqA and seqB should be numpy arrays of shape [T, E], where:
      - T is the number of frames/timesteps
      - E is the dimensionality of the embedding.

    Returns:
      The DTW distance (smaller = more similar).
    """
    # Let N and M be the time dimensions of seqA and seqB, respectively.
    N, _ = seqA.shape
    M, _ = seqB.shape

    # Cost matrix
    dist_matrix = np.zeros((N, M))

    # Initialize distance with L2 norms
    for i in range(N):
        for j in range(M):
            dist_matrix[i, j] = np.linalg.norm(seqA[i] - seqB[j])

    # DP matrix for DTW
    dp = np.zeros((N + 1, M + 1)) + np.inf
    dp[0, 0] = 0

    # Populate the DP table
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = dist_matrix[i - 1, j - 1]
            dp[i, j] = cost + min(
                dp[i - 1, j],  # insertion
                dp[i, j - 1],  # deletion
                dp[i - 1, j - 1])  # match

    return dp[N, M]


def _chamfer_distance(seqA: np.ndarray, seqB: np.ndarray) -> float:
    """Compute Chamfer distance between two sequences of embeddings.

    The Chamfer distance is defined as:

        sum_{x in seqA} min_{y in seqB} d(x, y)
      + sum_{y in seqB} min_{x in seqA} d(x, y)

    where d(x, y) is typically the L2 distance ||x - y||.

    Args:
        seqA: [N, E] array of N embeddings (dimension E) for sequence A.
        seqB: [M, E] array of M embeddings (dimension E) for sequence B.

    Returns:
        A scalar (float) representing the Chamfer distance.
    """
    # seqA: shape [N, E]
    # seqB: shape [M, E]
    # Construct an [N, M] matrix of pairwise L2 distances.
    dist_matrix = np.linalg.norm(seqA[:, None] - seqB[None, :], axis=-1)

    # For each embedding in seqA, find the closest embedding in seqB.
    # dist_matrix.min(axis=1) -> shape [N]
    # Summation of minima across seqA.
    sum_A_to_B = dist_matrix.min(axis=1).sum()

    # For each embedding in seqB, find the closest embedding in seqA.
    # dist_matrix.min(axis=0) -> shape [M]
    # Summation of minima across seqB.
    sum_B_to_A = dist_matrix.min(axis=0).sum()

    return sum_A_to_B + sum_B_to_A


def _distance(seqA: np.ndarray,
              seqB: np.ndarray,
              method: str = "dtw") -> float:
    """Compute distance between two sequences of embeddings."""
    if method == "dtw":
        return _dtw_distance(seqA, seqB)
    if method == "chamfer":
        return _chamfer_distance(seqA, seqB)
    raise ValueError(f"Unknown distance method: {method}")


def crop_to_multiple_of_patch_size(image: Any, patch_size: int = 14) -> Any:
    """Crops the image to the largest possible multiple of `patch_size` while
    keeping the aspect ratio."""
    _, _, H, W = image.shape  # Get original height and width
    new_H = (H // patch_size) * patch_size  # Closest smaller multiple of 14
    new_W = (W // patch_size) * patch_size  # Closest smaller multiple of 14

    # Center crop the image
    transform = transforms.CenterCrop((new_H, new_W))
    cropped_image = transform(image)

    return cropped_image


class DinoSimilarityApproach:
    """Classification approach using a pretrained DINO model for feature
    extraction and DTW for sequence similarity.

    This class mirrors the structure of the VLM-based approach but
    replaces the vision-language model with a pretrained DINO feature
    extractor.
    """

    def __init__(self) -> None:
        """Load a pretrained DINO model and set internal parameters."""
        # 1) Load the DINO model
        self._dino = torch.hub.load(  # type: ignore[no-untyped-call]
            "facebookresearch/dinov2", CFG.dino_model_name)
        self._max_video_len = 10  # Maximum frames to process per video
        self.log_dir: Optional[str] = None

    @classmethod
    def approach_name(cls) -> str:
        """Return the name of this classification approach."""
        return "dino_similarity"

    def predict(self, _episode_name: str, support_videos: List[Video],
                _support_labels: List[int], query_videos: List[Video],
                task_id: int) -> List[int]:
        """Predict labels for query videos based on a single support video.

        1. Preprocess videos (subsample frames).
        2. Extract DINO feature embeddings for each frame.
        3. Compute similarity (via DTW) between the support video embeddings
           and each query video.
        4. Assign labels to query videos based on whichever has the smaller
           DTW distance (i.e., more similar).

        Args:
            support_videos: Reference videos (we assume only 1 in this demo).
            support_labels: Labels for support videos (not used here).
            query_videos: Videos to be classified.
            task_id: Task identifier for logging/debugging.

        Returns:
            A list of predicted labels for the query videos (e.g., [1,0]).
        """
        # Setup logging directory for this task
        self._setup_logging_dir(task_id)

        # Preprocess videos (subsample frames)
        support_videos, query_videos = self._preprocess_videos(
            support_videos, query_videos)
        assert len(support_videos
                   ) == 1, "Currently we assume exactly 1 support video."

        # 2) Extract DINO feature embeddings
        logging.debug("Extracting features...")
        support_features = self._extract_features(support_videos)
        query_features = self._extract_features(query_videos)

        # We assume just 1 support video, so index 0
        support_seq = support_features[0]

        # 3) Compute similarity for each query
        logging.debug("Computing distances...")
        distances = []
        for qseq in query_features:
            dist = _distance(support_seq, qseq, CFG.distance_function)
            distances.append(dist)

        # 4) Label assignment based on relative similarity:
        #    - The "closest" query video gets label 1, the other gets 0.
        #    - If you prefer threshold-based classification, adapt here.
        if distances[0] < distances[1]:
            # First query is more similar
            return [1, 0]
        # Second query is more similar
        return [0, 1]

    def _setup_logging_dir(self, task_id: int) -> None:
        """Set up and clean the logging directory for the current task."""
        self.log_dir = os.path.join(CFG.log_dir, self.approach_name(),
                                    f"seed{CFG.seed}", f"task{task_id}")
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

    def _preprocess_videos(
            self, support_videos: List[Video],
            query_videos: List[Video]) -> Tuple[List[Video], List[Video]]:
        """Subsample frames to a maximum length (_max_video_len)."""

        def _subsample(video: Video) -> Video:
            if len(video) <= self._max_video_len:
                return video
            step = (len(video) - 1) / (self._max_video_len - 1)
            sampled = [
                video[int(i * step)] for i in range(self._max_video_len - 1)
            ]
            sampled.append(video[-1])
            return sampled

        support_videos = [_subsample(vid) for vid in support_videos]
        query_videos = [_subsample(vid) for vid in query_videos]
        return support_videos, query_videos

    def _extract_features(self, videos: List[Video]) -> List[np.ndarray]:
        """Extract one DINO feature embedding per frame in each video.

        Args:
            videos: A list of videos, each a sequence of frames in PIL.Image
                   or numpy array format.

        Returns:
            A list (length = #videos) of numpy arrays. Each array has shape
            [T, E], where T is the number of frames in the video and E is the
            embedding dimension output by the DINO model.
        """
        all_features: List[np.ndarray] = []
        for vid in tqdm(videos, desc="Processing videos", leave=False):
            embeddings_for_vid = []
            # Add progress bar for frames within each video
            for frame in tqdm(
                    vid, desc="Processing frames",
                    leave=False):  # Convert frame to tensor if necessary
                if not isinstance(frame, torch.Tensor):
                    # Convert PIL Image to tensor
                    if "PIL" in str(type(frame)):
                        frame_tensor = torch.from_numpy(
                            np.array(frame)).permute(2, 0,
                                                     1).float().unsqueeze(0)
                    # Convert numpy array to tensor
                    elif isinstance(frame, np.ndarray):
                        frame_tensor = torch.from_numpy(frame).permute(
                            2, 0, 1).float().unsqueeze(0)
                    else:
                        raise ValueError(
                            f"Unsupported frame type: {type(frame)}")
                else:
                    frame_tensor = frame.unsqueeze(0)

                # DINO expects normalization, resizing, etc.
                frame_tensor = crop_to_multiple_of_patch_size(frame_tensor)

                # In many official DINO repos, the forward
                # pass is something like:
                with torch.no_grad():
                    # Return the final-layer features (depends on the repo)
                    # This snippet is conceptual; adapt it to your loaded model.
                    feat = self._dino(frame_tensor)

                # Suppose feat has shape [1, E] or [E]; convert to 1D numpy
                feat_np = feat.squeeze(0).cpu().numpy()
                embeddings_for_vid.append(feat_np)

            # Convert to [T, E]
            stacked: np.ndarray = np.stack(embeddings_for_vid, axis=0)
            all_features.append(stacked)

        return all_features
