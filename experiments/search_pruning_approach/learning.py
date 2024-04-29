from abc import ABC, abstractmethod
from contextlib import nullcontext
import copy
from functools import cache, cached_property
import time
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union
from experiments.search_pruning_approach.dataset import FeasibilityDataset, FeasibilityInputBatch, SkeletonNSRT
from predicators.ml_models import DeviceTrackingModule, _get_torch_device
from torch import nn, Tensor, tensor
import torch
import logging
import sys
import os
import cProfile

from predicators.structs import NSRT, _GroundNSRT, Object, State, Variable

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("tkagg")

np.set_printoptions(linewidth=np.inf, precision=4,
                    floatmode='fixed', threshold=sys.maxsize)


def tensor_stats(x: Tensor, full_tensor: bool = False):
    if x.numel() == 0:
        return f"shape: {x.shape}"
    if full_tensor:
        return f"min: {x.min(dim=0).values.tolist()}; max: {x.max(dim=0).values.tolist()}; mean: {x.mean(dim=0).tolist()}; std: {x.std(dim=0).tolist()}; shape: {x.shape}"
    return f"min: {x.min().item():.4f}; max: {x.max().item():.4f}; mean: {x.mean().item():.4f}; std: {x.std().item():.4f}; shape: {x.shape}"


def l1_regularization(models: Union[nn.Module, Iterable[nn.Module]]) -> Tensor:
    if isinstance(models, nn.Module):
        models = [models]
    return torch.cat([torch.abs(param).flatten() for model in models for param in model.parameters()]).sum()


class FeasibilityFeaturizer(DeviceTrackingModule):
    """Featurizer that turns a state vector into a uniformly sized feature vector.
    """

    def __init__(
        self,
        name: str,
        sizes: int,
        dropout: float,
        device: Optional[str] = None
    ):
        """Creates a new featurizer

        Params:
            state_vector_size - size of the state vector for the corresponding NSRT
            hidden_sizes - the sizes of the hidden layers in the DNN
            feature_size - output feature vector size
            device (optional) - what device to place the module and generated vectors on
                (uses the globally default device if unspecified)
        """
        super().__init__(device)

        self._name = name

        assert sizes
        self.register_buffer("_min_state", torch.zeros(
            (sizes[0],), device=device))
        self.register_buffer("_max_state", torch.ones(
            (sizes[0],), device=device))

        self._layers = nn.ModuleList([
            nn.Linear(input_size, output_size, device=device)
            for input_size, output_size in zip(sizes, sizes[1:])
        ])
        self._dropout = nn.Dropout(dropout)

    def update_range(self, min_state: npt.NDArray, max_state: npt.NDArray) -> None:
        self._min_state = torch.minimum(
            self._min_state, torch.tensor(min_state, device=self.device))
        self._max_state = torch.maximum(
            self._max_state, torch.tensor(max_state, device=self.device))

    def forward(self, state: npt.NDArray) -> Tensor:
        """Runs the featurizer

        Params:
            state - numpy array of shape (batch_size, state_vector_size) of batched state vectors
        """
        state = tensor(state, device=self.device)

        if not self.training:
            logging.info(
                f"-- FORWARD featurizer {self._name} state stats: {tensor_stats(state, full_tensor=True)}")

        state -= self._min_state
        state /= torch.clamp(self._max_state - self._min_state, min=0.1)
        state = (state - 1) * 2

        for layer in self._layers[:-1]:
            state = self._dropout(nn.functional.elu(layer(state)))
        return self._layers[-1](state)


class Tokenizer(DeviceTrackingModule):  # TODO add start and end tokens
    """Adds a positional embedding and optionally a cls token to the feature vectors"""

    def __init__(
        self,
        feature_size: int,
        embedding_size: int,
        token_size: int,
        num_nsrts: int,
        concat: bool,
        mark_last_token: bool,
        include_cls: Optional[str],
        horizon: int,
        device: Optional[str] = None
    ):
        """Creates a new positional embedding layer.

        Params:
            feature_size - size of the output feature vector
            embedding_size - size of the embedding vector (if it's concatenated)
            concat - whether the positional embedding should be concatenated with the input vector
                or added to it
            include_cls - whether and how the embedding vector should be included
                before all the feature vectors. Possible values are: None (no cls), 'learned'
                (as a learnable parameter), 'marked' (each feature vector has an additional 0 added to it
                and the cls token has a 1 in that space)

        """
        super().__init__(device)
        assert include_cls in {'learned', 'marked', None}
        self._concat = concat
        self._horizon = horizon

        self._num_nsrts = num_nsrts

        feature_vector_size = feature_size

        if concat:
            self._embedding_size = embedding_size
            feature_vector_size += embedding_size
        else:
            self._embedding_size = feature_size

        feature_vector_size += num_nsrts

        self._mark_last_token = mark_last_token
        feature_vector_size += mark_last_token

        if include_cls == 'learned':
            self._cls = nn.Parameter(torch.randn(
                (1, feature_vector_size), device=device))
            self._cls_marked = False
        elif include_cls == 'marked':
            feature_vector_size += 1
            self.register_buffer(
                "_cls", torch.concat([torch.zeros(
                    (1, feature_vector_size - 1), device=device), torch.ones((1, 1), device=device)], dim=1)
            )
            self._cls_marked = True
        else:
            self._cls = None
            self._cls_marked = False

        self._linear = nn.Linear(
            feature_vector_size, token_size, device=device)

    def forward(
        self,
        tokens: Tensor,
        invalid_mask: Tensor,
        last_token_mask: Tensor,
        nsrt_indices: Tensor,
        pos_offset: Optional[Tensor] = None
    ) -> Tensor:
        """Runs the positional embeddings.

        Params:
            tokens - tensor of shape (batch_size, max_sequence_length, feature_size) of outputs from featurizers
            invalid_mask - tensor of shape (batch_size, max_sequence_length) of which tokens are invalid
            last_token_mask - tensor of shape (batch_size, max_sequence_length) of whether to mark the token if last_token_mark is True
            nsrt_indices - tensor of shape (batch_size, max_sequence_length) of which nsrt is at a specific spot
            pos_offset - tensor of shape (batch_size,) of positions offsets per batch
        """

        batch_size, max_len, input_feature_size = tokens.shape

        # Calculating per-token positions and in-token indices
        indices = torch.arange(self._embedding_size,
                               device=self.device).unsqueeze(0)
        positions = torch.arange(max_len, device=self.device).unsqueeze(
            0).unsqueeze(-1).expand(batch_size, -1, -1)
        if pos_offset is not None:
            positions = positions + pos_offset.unsqueeze(-1).unsqueeze(-1)

        # Calculating the embeddings
        freq = 1 / \
            self._horizon ** ((indices - (indices % 2)) / self._embedding_size)
        embeddings = torch.sin(positions.float() @ freq +
                               torch.pi / 2 * (indices % 2))

        # Concateanting/adding embeddings
        if self._concat:
            tokens = torch.cat([tokens, embeddings], dim=-1)
        else:
            tokens += embeddings

        # Marking the nsrts
        nsrt_marks = torch.zeros(
            (batch_size, max_len, self._num_nsrts), device=self.device)
        nsrt_marks.scatter_(2, nsrt_indices.unsqueeze(2), 1)
        tokens = torch.cat([tokens, nsrt_marks], dim=2)

        # Marking the last token
        if self._mark_last_token:
            last_token_marks = torch.zeros(
                (batch_size, max_len), device=self.device)
            last_token_marks[last_token_mask] = 1.0
            tokens = torch.cat([tokens, last_token_marks.unsqueeze(2)], dim=2)

        # Adding the cls token
        if self._cls is not None:
            if self._cls_marked:
                tokens = torch.cat([tokens, torch.zeros(
                    (batch_size, max_len, 1), device=self.device)], dim=2)
            tokens = torch.cat([self._cls.unsqueeze(
                0).expand(batch_size, -1, -1), tokens], dim=1)
            invalid_mask = torch.cat(
                [torch.full((batch_size, 1), False, device=self.device), invalid_mask], dim=1)

        # Mapping to the token size
        return self._linear(tokens), invalid_mask


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = -1,
    gamma_pos: float = 1.0,
    gamma_neg: float = 1.0,
    reduction: str = "none",
) -> Tensor:
    """Adapted from torchvision.ops.focal_loss"""
    ce_loss = nn.functional.binary_cross_entropy(
        inputs, targets, reduction="none")

    loss = torch.zeros_like(ce_loss)
    is_positive = torch.abs(targets - 1.0) < 0.0001
    is_negative = torch.logical_not(is_positive)
    loss[is_positive] = ce_loss[is_positive] * \
        ((1 - inputs[is_positive]) ** gamma_pos)
    loss[is_negative] = ce_loss[is_negative] * \
        (inputs[is_negative] ** gamma_neg)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    assert reduction in {"none", "mean", "sum"}
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class FeasibilityClassifier(ABC):
    """Abstract class to be able to share functionality between a constant and a learned feasibility classifier
    """
    @abstractmethod
    def classify(self, states: Sequence[State], skeleton: Sequence[_GroundNSRT]) -> Tuple[bool, float]:
        raise NotImplementedError()


class ConstFeasibilityClassifier(FeasibilityClassifier):
    def classify(self, states: Sequence[State], skeleton: Sequence[_GroundNSRT]) -> Tuple[bool, float]:
        return True, 1.0


class StaticFeasibilityClassifier(FeasibilityClassifier):
    def __init__(self, classify: Callable[[Sequence[State], Sequence[_GroundNSRT]], Tuple[bool, float]]):
        self._classify = classify

    def classify(self, states: Sequence[State], skeleton: Sequence[_GroundNSRT]) -> Tuple[bool, float]:
        return self._classify(states, skeleton)


class NeuralFeasibilityClassifier(DeviceTrackingModule, FeasibilityClassifier):
    # +--------+     +--------+  +----------+     +--------+
    # |   ENC  |     |   ENC  |  |    DEC   |     |   DEC  |
    # \ FEAT 1 / ... \ FEAT N /  \ FEAT N+1 / ... \ FEAT M /
    #  \______/       \______/    \________/       \______/
    #     ||             ||           ||              ||
    #     \/             \/           \/              \/
    # +-----------------------+  +-------------------------+
    # |        ENCODER        |->|         DECODER         |
    # +-----------------------+  +-------------------------+
    #                               ||   ||   ||   ||   ||
    #                               \/   \/   \/   \/   \/
    #                            +-------------------------+
    #                             \     MEAN POLLING      /
    #                              +---------------------+
    #                                       |  |
    #                                       \__/
    #                              +---------------------+
    #                               \    CLASSIFIER     /
    #                                +-----------------+
    def __init__(
        self,
        nsrts: Set[NSRT],
        featurizer_sizes: List[int],
        positional_embedding_size: int,
        positional_embedding_concat: bool,
        mark_failing_nsrt: bool,
        token_size: int,
        transformer_num_heads: int,
        transformer_num_layers: int,
        transformer_ffn_hidden_size: int,
        cls_style: str,
        embedding_horizon: int,
        max_train_iters: int,
        general_lr: float,
        transformer_lr: float,
        min_inference_prefix: int,
        threshold_recalibration_percentile: float,
        max_num_objects: int,
        optimizer_name: str = 'adam',
        dropout: int = 0.2,
        l1_penalty: float = 0.05,
        l2_penalty: float = 0.0,
        num_threads: int = 8,
        classification_threshold: float = 0.5,
        use_torch_gpu: bool = False,
        check_nans: bool = False,
    ):
        torch.set_num_threads(num_threads)
        DeviceTrackingModule.__init__(self, _get_torch_device(use_torch_gpu))

        assert cls_style in {'learned', 'marked', 'mean'}
        self._cls_style = cls_style

        self.min_inference_prefix = min_inference_prefix
        self._thresh = classification_threshold
        self._max_num_objects = max_num_objects

        self._num_iters = max_train_iters
        self._general_lr = general_lr
        self._transformer_lr = transformer_lr

        skeleton_nsrts = [(nsrt, ran_flag) for nsrt in sorted(nsrts)
                          for ran_flag in [True, False]]
        self._skeleton_nsrt_indices = {
            skeleton_nsrt: idx for idx, skeleton_nsrt in enumerate(skeleton_nsrts)}
        self._featurizer_output_size = featurizer_sizes[-1]
        self._featurizers: Dict[SkeletonNSRT, FeasibilityFeaturizer] = {
            (nsrt, ran_flag): self.create_featurizer(
                f"featurizer_{self._skeleton_nsrt_indices[(nsrt, ran_flag)]}_{nsrt.name}_{'ran' if ran_flag else 'not_ran'}",
                nsrt, featurizer_sizes, dropout,
            ) for nsrt in nsrts for ran_flag in [True, False]
        }

        self._positional_encoding = Tokenizer(
            feature_size=featurizer_sizes[-1],
            embedding_size=positional_embedding_size,
            token_size=token_size,
            num_nsrts=len(skeleton_nsrts),
            concat=positional_embedding_concat,
            mark_last_token=mark_failing_nsrt,
            include_cls={
                'mean': None, 'learned': 'learned', 'marked': 'marked'
            }[cls_style],
            horizon=embedding_horizon,
            device=self.device,
        )
        self._transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=token_size,
                nhead=transformer_num_heads,
                dim_feedforward=transformer_ffn_hidden_size,
                dropout=dropout,
                batch_first=True,
                device=self.device
            ), norm=nn.LayerNorm(
                normalized_shape=token_size,
                device=self.device
            ), num_layers=transformer_num_layers,
        )
        self._classifier_head = nn.Sequential(
            nn.Linear(token_size, 1, device=self.device),
            nn.Sigmoid(),
        )

        self._l1_penalty = l1_penalty
        self._optimizer = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'rmsprop': torch.optim.RMSprop
        }[optimizer_name]([
            {'params': [
                parameter for featurizer in self._featurizers.values()
                for parameter in featurizer.parameters()
            ], 'lr': self._general_lr, "weight_decay": l2_penalty * self._general_lr},
            {'params': self._positional_encoding.parameters(
            ), 'lr': self._general_lr, "weight_decay": l2_penalty * self._general_lr},
            {'params': self._classifier_head.parameters(), 'lr': self._general_lr,
             "weight_decay": l2_penalty * self._general_lr},
            {'params': self._transformer.parameters(), 'lr': self._transformer_lr,
             "weight_decay": l2_penalty * self._transformer_lr},
        ])
        self._check_nans = check_nans
        self._threshold_recalibration_frac = threshold_recalibration_percentile
        self._unsure_confidence = classification_threshold
        self._trained_failing_nsrts: Set[NSRT] = set()

    def move_to_device(self, device: str) -> 'NeuralFeasibilityClassifier':
        return self.to(device).share_memory()

    @torch.no_grad()
    def classify(self, states: Sequence[State], skeleton: Sequence[_GroundNSRT]) -> Tuple[bool, float]:
        """Classifies a single datapoint
        """
        if len(states) == len(skeleton) + 1 or len(states) < 2:  # Make sure there is at least one not-run nsrt
            return True, 1.0

        failing_nsrt = skeleton[len(states) - 2].parent
        # Make sure we don't have too big of a horizon to predict
        if len(states) <= self.min_inference_prefix:
            return True, self._unsure_confidence
        # Make sure we don't run on an untrained failing nsrt
        if failing_nsrt not in self._trained_failing_nsrts:
            return True, self._unsure_confidence

        self.eval()
        confidence, logits = self(FeasibilityDataset.transform_input(
            self._skeleton_nsrt_indices, skeleton, states, self._max_num_objects))
        confidence = confidence.cpu()
        logits = logits.tolist()
        logging.info(f"Confidence {float(confidence)}; Logits {logits}")
        return confidence >= self._thresh, confidence

    def fit(
        self,
        training_dataset: FeasibilityDataset,
        validation_dataset: FeasibilityDataset,
        training_snapshot_directory: str = "",
        delete_optimizer: bool = True
    ) -> None:
        if self._optimizer is None:
            raise RuntimeError("The classifier has been fitted")

        if not training_dataset.trainable_failing_nsrts:
            return
        self._trained_failing_nsrts = training_dataset.trainable_failing_nsrts

        # Diagnostics
        logging.info("Training dataset statistics:")
        logging.info(training_dataset.diagnostics)

        logging.info("Validation dataset statistics:")
        logging.info(validation_dataset.diagnostics)

        # Initializing per-nsrt featurizers if not initialized already
        logging.info("Rescaling state featurizers")
        self.rescale_featurizers(training_dataset)
        self.rescale_featurizers(validation_dataset)

        # Creating loss functions
        def training_loss_fn(inputs, targets): return sigmoid_focal_loss(
            inputs, targets, reduction="sum") / len(training_dataset)
        def validation_loss_fn(inputs, targets): return sigmoid_focal_loss(
            inputs, targets, reduction="sum") / len(validation_dataset)

        # Training loop
        logging.info("Running training")
        best_params = self.state_dict()
        best_params["iter"] = -1
        best_loss = float('inf')
        with (torch.autograd.detect_anomaly(True) if self._check_nans else nullcontext()):
            for itr, (x_train_batch, y_train_batch) in zip(
                reversed(range(self._num_iters, -1, -1)),
                itertools.chain.from_iterable(
                    itertools.repeat(training_dataset))
            ):
                if itr:  # To make sure we're better than the original network
                    self.train()
                    self._optimizer.zero_grad()
                    outputs, logits = self(x_train_batch)
                    train_loss = training_loss_fn(
                        outputs, torch.from_numpy(y_train_batch).to(self.device))
                    train_loss.backward()

                # l1_loss = l1_regularization(self) * self._l1_penalty
                # l1_loss.backward()

                self._optimizer.step()

                if itr % 100 == 0:
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            logging.info(
                                f"-- PARAM NAME {(name + ' '*70)[:70]} gradient stats: {tensor_stats(param.grad)} value stats: {tensor_stats(param)}")

                    if itr % self._num_iters == 0:
                        training_debug_str, _, training_acc, _, _ = self.report_performance(
                            training_dataset, training_loss_fn)
                        validation_debug_str, validation_loss, validation_acc, _, _ = self.report_performance(
                            validation_dataset, validation_loss_fn)
                        logging.info(f"Iteration {itr}/{self._num_iters}")
                        logging.info(
                            f"\tTraining performance: {training_debug_str}")
                        logging.info(
                            f"\tValidation performance: {validation_debug_str}")
                    else:
                        validation_debug_str, validation_loss, validation_acc, _, _ = self.report_performance(
                            validation_dataset, validation_loss_fn)
                        logging.info(f"Iteration {itr}/{self._num_iters}")
                        logging.info(
                            f"\tValidation performance: {validation_debug_str}")

                    if best_loss > validation_loss:
                        del best_params
                        best_loss = validation_loss
                        best_params = copy.deepcopy(self.state_dict())
                        best_params["iter"] = itr

                    if training_snapshot_directory:
                        torch.save(self, os.path.join(
                            training_snapshot_directory, f"model-{itr}.pt"))

                    # if training_acc >= 0.9999: #validation_acc >= 0.95 and (itr - best_params["iter"]) >= 2000:
                    #     break

        # Loading the best params
        logging.info(f"Best params from iter {best_params['iter']}")
        del best_params["iter"]
        self.load_state_dict(best_params)

        # Threshold recalibration and final metrics
        info_str, _, _, _, _ = self.report_performance(
            validation_dataset, validation_loss_fn)
        logging.info(f"Final Training Performance: {info_str}")
        info_str, _, _, false_confidence, _ = self.report_performance(
            validation_dataset, validation_loss_fn)
        logging.info(f"Final Validation Performance: {info_str}")

        self._unsure_confidence = max(self._thresh, false_confidence)
        logging.info(f"Unsure confidence set to {self._unsure_confidence}")

        if delete_optimizer:
            # Making sure we don't fit twice
            self._optimizer = None

    def forward(
        self,
        batch: FeasibilityInputBatch,
    ) -> float:
        """Runs the core of the classifier.
        """

        # Calculating feature vectors
        tokens, invalid_mask, skeleton_nsrt_indices, last_token_mask = batch.run_featurizers(
            self._featurizers, self._skeleton_nsrt_indices, self._featurizer_output_size, self.device
        )
        sequence_lengths = torch.logical_not(invalid_mask).sum(dim=1)

        # Calculating offsets for encoding
        pos_offsets = torch.zeros((invalid_mask.shape[0],), device=self.device)
        # if self.training:
        #     pos_offsets += (torch.rand_like(pos_offsets) * (self._positional_encoding._horizon - sequence_lengths - 1)).long()

        # Adding positional encoding and cls token
        tokens, invalid_mask = self._positional_encoding(
            tokens=tokens,
            invalid_mask=invalid_mask,
            last_token_mask=last_token_mask,
            nsrt_indices=skeleton_nsrt_indices,
            pos_offset=pos_offsets,
        )
        if not self.training:
            logging.info(f"-- FORWARD tokens stats: {tensor_stats(tokens)}")

        assert tokens.shape[:2] == invalid_mask.shape
        # Running the core transformer
        transformer_outputs = self._transformer(
            src=tokens,
            src_key_padding_mask=invalid_mask,
            # is_causal=False,
        )

        # Preparing for the classifier head
        if self._cls_style == 'mean':
            transformer_outputs[invalid_mask.unsqueeze(
                -1).expand(-1, -1, transformer_outputs.shape[2])] = 0
            classifier_tokens = transformer_outputs.sum(
                dim=1) / sequence_lengths.unsqueeze(-1)
            if not self.training:
                logging.info(
                    f"-- FORWARD classifier outputs stats: {tensor_stats(self._classifier_head[0](transformer_outputs))}")
        else:
            classifier_tokens = transformer_outputs[:, 0]

        # Running the classifier head
        output = self._classifier_head(classifier_tokens).flatten()

        return output, self._classifier_head[0](transformer_outputs).flatten()

    def create_featurizer(
        self,
        name: str,
        nsrt: NSRT,
        featurizer_sizes: List[int],
        dropout: float,
    ) -> FeasibilityFeaturizer:
        """ Initializes a featurizer for a single ground nsrt.

        NOTE: The assumption is that the concatentated features of all objects that
        are passed to a given NSRT always have the same data layout and total length
        equal to the dimensionalities of their types.
        """
        dim = sum(v.type.dim for v in nsrt.parameters) + len(nsrt.parameters) * \
            self._max_num_objects  # Object vectors + object ids
        featurizer = FeasibilityFeaturizer(
            name=name,
            sizes=[dim] + featurizer_sizes,
            dropout=dropout,
            device=self.device,
        )
        self.add_module(name, featurizer)
        return featurizer

    def rescale_featurizers(self, dataset: FeasibilityDataset) -> None:
        """Rescales the featurizers based on the states
        """
        for skeleton_nsrt, (min_state, max_state) in dataset.state_ranges.items():
            self._featurizers[skeleton_nsrt].update_range(min_state, max_state)

    @torch.no_grad()
    def report_performance(self, dataset: FeasibilityDataset, loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tuple[str, float, float, float, float]:
        self.eval()
        y_pred_batches, y_true_batches = zip(*(
            (self(x_batch)[0], torch.from_numpy(y_batch).to(self.device))
            for x_batch, y_batch in dataset
        ))
        y_pred, y_true = torch.concatenate(
            y_pred_batches), torch.concatenate(y_true_batches)
        true_positives_mask, true_negatives_mask = torch.abs(
            y_true - 1.0) < 0.0001, torch.abs(y_true - 0.0) < 0.0001
        pred_positives_mask, pred_negatives_mask = y_pred >= self._thresh, y_pred < self._thresh

        num_true_positives = torch.logical_and(
            true_positives_mask, pred_positives_mask).sum().item()
        num_true_negatives = torch.logical_and(
            true_negatives_mask, pred_negatives_mask).sum().item()

        loss = loss_fn(y_pred, y_true).item()
        acc = torch.logical_or(
            torch.logical_and(true_positives_mask, pred_positives_mask),
            torch.logical_and(true_negatives_mask, pred_negatives_mask)
        ).float().mean().item()

        false_confidence = torch.kthvalue(
            torch.cat([y_pred[true_negatives_mask].flatten(),
                      tensor([0], device=self.device)]).cpu(),
            int(dataset.num_positive_datapoints *
                self._threshold_recalibration_frac) + 1
        ).values.item()
        acceptance_rate = (
            torch.logical_and(true_positives_mask, y_pred >= false_confidence).sum(
            ).item() / dataset.num_positive_datapoints
        )

        return (
            f"Loss: {loss:.6}, Acc: {acc:.2%}, "
            f"%True+: {num_true_positives/len(dataset):.2%}, "
            f"%True-: {num_true_negatives/len(dataset):.2%}, "
            f"%False+: {(dataset.num_negative_datapoints-num_true_negatives)/len(dataset):.2%}, "
            f"%False-: {(dataset.num_positive_datapoints-num_true_positives)/len(dataset):.2%}, "
            f"{self._threshold_recalibration_frac:.0%} Positive Thresh: {false_confidence:.4}, Acceptance rate: {acceptance_rate:.2%}",
            loss, acc, false_confidence, acceptance_rate
        )
