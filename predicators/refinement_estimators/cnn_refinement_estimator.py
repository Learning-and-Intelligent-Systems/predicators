"""A learning-based refinement cost estimator that, for each skeleton, trains a
CNN regression model mapping initial state render to cost."""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import dill as pkl
import numpy as np

from predicators.ml_models import CNNRegressor
from predicators.refinement_estimators import BaseRefinementEstimator
from predicators.settings import CFG
from predicators.structs import GroundAtom, ImageInput, RefinementDatapoint, \
    Task, _GroundNSRT

# Type of the (skeleton, atoms_sequence) key for model dictionary
# which converts both of them to be immutable
ModelDictKey = Tuple[Tuple[_GroundNSRT, ...],  # skeleton converted to tuple
                     Tuple[FrozenSet[GroundAtom], ...]  # atoms_sequence
                     ]


class CNNRefinementEstimator(BaseRefinementEstimator):
    """A refinement cost estimator that uses a CNN to predict refinement cost
    from an initial state render."""

    def __init__(self) -> None:
        super().__init__()
        # _model_dict maps immutable skeleton atoms_sequence pair to model
        self._model_dict: Optional[Dict[ModelDictKey, CNNRegressor]] = None

    @classmethod
    def get_name(cls) -> str:
        return "cnn"

    @property
    def is_learning_based(self) -> bool:
        return True

    def get_cost(self, initial_task: Task, skeleton: List[_GroundNSRT],
                 atoms_sequence: List[Set[GroundAtom]]) -> float:
        assert self._model_dict is not None, "Need to train"
        key = self._immutable_model_dict_key(skeleton, atoms_sequence)
        # If key isn't in dictionary (no data for skeleton), cost is infinite
        if key not in self._model_dict:
            return float('inf')
        model = self._model_dict[key]
        # Get render image of initial state and pass into model
        input_img = self._get_rendered_initial_state(initial_task)
        cost = model.predict(input_img)
        return cost[0]

    def train(self, data: List[RefinementDatapoint]) -> None:
        """Train the CNN regressors on the data points for that skeleton,
        atoms_sequence pair."""

        # Go through data and group them by skeleton
        grouped_input_imgs = defaultdict(list)
        grouped_targets = defaultdict(list)
        for task, skeleton, atoms_sequence, succeeded, refinement_time in data:
            # Convert skeleton and atoms_sequence into an immutable dict key
            key = self._immutable_model_dict_key(skeleton, atoms_sequence)
            # Render the initial state for use as an input image matrix
            img = self._get_rendered_initial_state(task)
            grouped_input_imgs[key].append(img)
            # Compute target value from refinement time and possible failure
            value = refinement_time
            if not succeeded:
                value += CFG.refinement_data_failed_refinement_penalty
            grouped_targets[key].append([value])

        # For each (skeleton, atoms_sequence) key, fit a CNNRegressor
        self._model_dict = {}
        total_num_keys = len(grouped_input_imgs)
        for i, key in enumerate(grouped_input_imgs):
            X = np.stack(grouped_input_imgs[key])
            assert len(X.shape) == 4  # expect (N, 3, H, W)
            Y = np.array(grouped_targets[key])
            assert Y.shape == (X.shape[0], 1)
            model = self._create_regressor()
            logging.info(f"Training CNN for skeleton {i}/{total_num_keys} "
                         f"using {X.shape[0]} data points...")
            t0 = time.perf_counter()
            model.fit(X, Y)
            logging.info(f"Fit model in {time.perf_counter() - t0:.2f}s")
            self._model_dict[key] = model

    @staticmethod
    def _immutable_model_dict_key(
            skeleton: List[_GroundNSRT],
            atoms_sequence: List[Set[GroundAtom]]) -> ModelDictKey:
        """Converts a skeleton and atoms_sequence into immutable types to use
        as a key for the model dictionary."""
        return (tuple(skeleton),
                tuple(frozenset(atoms) for atoms in atoms_sequence))

    @staticmethod
    def _create_regressor() -> CNNRegressor:
        return CNNRegressor(
            seed=CFG.seed,
            conv_channel_nums=CFG.cnn_regressor_conv_channel_nums,
            conv_kernel_sizes=CFG.cnn_regressor_conv_kernel_sizes,
            linear_hid_sizes=CFG.cnn_regressor_linear_hid_sizes,
            max_train_iters=CFG.cnn_regressor_max_itr,
            clip_gradients=CFG.cnn_regressor_clip_gradients,
            clip_value=CFG.cnn_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate)

    def _get_rendered_initial_state(self, task: Task) -> ImageInput:
        """Render the initial state of the task using the given environment's
        method, and pre-process image as necessary."""
        # Render initial state
        img = self._env.render_state(task.init, task)[0]

        # Crop and downsample the image if needed
        if CFG.cnn_refinement_estimator_crop:
            h_crop_lb, h_crop_ub, w_crop_lb, w_crop_ub = \
                CFG.cnn_refinement_estimator_crop_bounds
            img = img[h_crop_lb:h_crop_ub, w_crop_lb:w_crop_ub]
        if CFG.cnn_refinement_estimator_downsample > 1:
            img = img[::CFG.cnn_refinement_estimator_downsample, ::CFG.
                      cnn_refinement_estimator_downsample]

        # Move the channels dimension to front and convert array to float
        img = np.moveaxis(img, -1, 0)
        float_img = img.astype(np.float32)
        return float_img

    def save_model(self, filepath: Path) -> None:
        with open(filepath, "wb") as f:
            pkl.dump(self._model_dict, f)

    def load_model(self, filepath: Path) -> None:
        with open(filepath, "rb") as f:
            self._model_dict = pkl.load(f)
