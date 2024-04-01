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

from predicators.structs import NSRT, _GroundNSRT, Object, State, Variable

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("tkagg")

np.set_printoptions(linewidth=np.inf, precision=4, floatmode='fixed', threshold=sys.maxsize)

def tensor_stats(x: Tensor, full_tensor: bool = False):
    x = x.detach().cpu()
    if x.numel() == 0:
        return f"shape: {x.shape}"
    if full_tensor:
        return f"min: {x.min(dim=0).values.numpy()}; max: {x.max(dim=0).values.numpy()}; mean: {x.mean(dim=0).numpy()}; std: {x.std(dim=0).numpy()}; shape: {x.shape}"
    return f"min: {float(x.min()):.4f}; max: {float(x.max()):.4f}; mean: {float(x.mean()):.4f}; std: {float(x.std()):.4f}; shape: {x.shape}"

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
        self.register_buffer("_min_state", torch.zeros((sizes[0],), device=device))
        self.register_buffer("_max_state", torch.ones((sizes[0],), device=device))

        self._layers = nn.ModuleList([
            nn.Linear(input_size, output_size, device=device)
            for input_size, output_size in zip(sizes, sizes[1:])
        ])
        self._dropout = nn.Dropout(dropout)

    def update_range(self, min_state: npt.NDArray, max_state: npt.NDArray) -> None:
        self._min_state = torch.minimum(self._min_state, torch.tensor(min_state, device=self.device))
        self._max_state = torch.maximum(self._max_state, torch.tensor(max_state, device=self.device))

    def forward(self, state: npt.NDArray) -> Tensor:
        """Runs the featurizer

        Params:
            state - numpy array of shape (batch_size, state_vector_size) of batched state vectors
        """
        state = tensor(state, device=self.device)

        if not self.training:
            logging.info(f"-- FORWARD featurizer {self._name} state stats: {tensor_stats(state, full_tensor=True)}")

        state -= self._min_state
        state /= torch.clamp(self._max_state - self._min_state, min=0.1)

        for layer in self._layers[:-1]:
            state = self._dropout(nn.functional.elu(layer(state)))
        return self._layers[-1](state)

class Tokenizer(DeviceTrackingModule): # TODO add start and end tokens
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
            self._cls = nn.Parameter(torch.randn((1, feature_vector_size), device=device))
            self._cls_marked = False
        elif include_cls == 'marked':
            feature_vector_size += 1
            self.register_buffer(
                "_cls", torch.concat([torch.zeros((1, feature_vector_size - 1), device=device), torch.ones((1, 1), device=device)], dim=1)
            )
            self._cls_marked = True
        else:
            self._cls = None
            self._cls_marked = False

        self._linear = nn.Linear(feature_vector_size, token_size, device=device)

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
        indices = torch.arange(self._embedding_size, device=self.device).unsqueeze(0)
        positions = torch.arange(max_len, device=self.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
        if pos_offset is not None:
            positions = positions + pos_offset.unsqueeze(-1).unsqueeze(-1)

        # Calculating the embeddings
        freq = 1 / self._horizon ** ((indices - (indices % 2)) / self._embedding_size)
        embeddings = torch.sin(positions.float() @ freq + torch.pi / 2 * (indices % 2))

        # Concateanting/adding embeddings
        if self._concat:
            tokens = torch.cat([tokens, embeddings], dim=-1)
        else:
            tokens += embeddings

        # Marking the nsrts
        nsrt_marks = torch.zeros((batch_size, max_len, self._num_nsrts), device=self.device)
        nsrt_marks.scatter_(2, nsrt_indices.unsqueeze(2), 1)
        tokens = torch.cat([tokens, nsrt_marks], dim=2)

        # Marking the last token
        if self._mark_last_token:
            last_token_marks = torch.zeros((batch_size, max_len), device=self.device)
            last_token_marks[last_token_mask] = 1.0
            tokens = torch.cat([tokens, last_token_marks.unsqueeze(2)], dim=2)

        # Adding the cls token
        if self._cls is not None:
            if self._cls_marked:
                tokens = torch.cat([tokens, torch.zeros((batch_size, max_len, 1), device=self.device)], dim=2)
            tokens = torch.cat([self._cls.unsqueeze(0).expand(batch_size, -1, -1), tokens], dim=1)
            invalid_mask = torch.cat([torch.full((batch_size, 1), False, device=self.device), invalid_mask], dim=1)

        # Mapping to the token size
        return self._linear(tokens), invalid_mask

def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = -1,
    gamma_pos: float = 2.0,
    gamma_neg: float = 2.0,
    reduction: str = "none",
) -> Tensor:
    """Adapted from torchvision.ops.focal_loss"""
    ce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")

    loss = torch.zeros_like(ce_loss)
    is_positive = torch.abs(targets - 1.0) < 0.0001
    is_negative = torch.logical_not(is_positive)
    loss[is_positive] = ce_loss[is_positive] * ((1 - inputs[is_positive]) ** gamma_pos)
    loss[is_negative] = ce_loss[is_negative] * (inputs[is_negative] ** gamma_neg)

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
        optimizer_name: str = 'adam',
        test_split: float = 0.10,
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

        self._min_inference_prefix = min_inference_prefix
        self._thresh = classification_threshold

        self._num_iters = max_train_iters
        self._test_split = test_split
        self._general_lr = general_lr
        self._transformer_lr = transformer_lr

        skeleton_nsrts = [(nsrt, ran_flag) for nsrt in sorted(nsrts) for ran_flag in [True, False]]
        self._skeleton_nsrt_indices = {skeleton_nsrt: idx for idx, skeleton_nsrt in enumerate(skeleton_nsrts)}
        self._featurizer_output_size = featurizer_sizes[-1]
        self._featurizers: Dict[SkeletonNSRT, FeasibilityFeaturizer] = {
            (nsrt, ran_flag): self.create_featurizer(
                f"featurizer_{self._skeleton_nsrt_indices[(nsrt, ran_flag)]}_{nsrt.name}_{'ran' if ran_flag else 'not_ran'}",
                nsrt, featurizer_sizes, 0.0,
            ) for nsrt in nsrts for ran_flag in [True, False]
        }

        self._positional_encoding = Tokenizer(
            feature_size = featurizer_sizes[-1],
            embedding_size = positional_embedding_size,
            token_size = token_size,
            num_nsrts = len(skeleton_nsrts),
            concat = positional_embedding_concat,
            mark_last_token = mark_failing_nsrt,
            include_cls = {
                'mean': None, 'learned': 'learned', 'marked': 'marked'
            }[cls_style],
            horizon = embedding_horizon,
            device = self.device,
        )
        self._transformer = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = token_size,
                nhead = transformer_num_heads,
                dim_feedforward = transformer_ffn_hidden_size,
                dropout = dropout,
                batch_first = True,
                device = self.device
            ), norm = nn.LayerNorm(
                normalized_shape = token_size,
                device = self.device
            ), num_layers = transformer_num_layers,
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
            {'params': self._positional_encoding.parameters(), 'lr': self._general_lr, "weight_decay": l2_penalty * self._general_lr},
            {'params': self._classifier_head.parameters(), 'lr': self._general_lr, "weight_decay": l2_penalty * self._general_lr},
            {'params': self._transformer.parameters(), 'lr': self._transformer_lr, "weight_decay": l2_penalty * self._transformer_lr},
        ])
        self._check_nans = check_nans
        self._threshold_recalibration_frac = threshold_recalibration_percentile
        self._unsure_confidence = classification_threshold

    def move_to_device(self, device: str) -> 'NeuralFeasibilityClassifier':
        return self.to(device).share_memory()

    def classify(self, states: Sequence[State], skeleton: Sequence[_GroundNSRT]) -> Tuple[bool, float]:
        """Classifies a single datapoint
        """
        if len(states) == len(skeleton) + 1: # Make sure there is at least one not-run nsrt
            return True, 1.0
        if len(states) <= self._min_inference_prefix: # Make sure we don't have too big of a horizon to predict
            return True, self._unsure_confidence
        if self._optimizer is not None: # Making sure the classifier is not used if not trained
            return True, 1.0

        self.eval()
        confidence, logits = self(FeasibilityDataset.transform_input(self._skeleton_nsrt_indices, skeleton, states))
        confidence = confidence.cpu()
        logits = logits.detach().cpu().numpy()
        logging.info(f"Confidence {float(confidence)}; Logits {logits}")
        return confidence >= self._thresh, confidence

    def fit(
        self,
        train_dataset: FeasibilityDataset,
        validation_dataset: FeasibilityDataset,
        training_snapshot_directory: str = "",
    ) -> None:
        if self._optimizer is None:
            raise RuntimeError("The classifier has been fitted")

        if not train_dataset.num_positive_datapoints or not train_dataset.num_negative_datapoints:
            return

        if train_dataset.num_positive_datapoints + train_dataset.num_negative_datapoints < 400:
            return

        # Diagnostics
        logging.info("Training dataset statistics:")
        logging.info(train_dataset.diagnostics)

        logging.info("Validation dataset statistics:")
        logging.info(validation_dataset.diagnostics)

        # Initializing per-nsrt featurizers if not initialized already
        logging.info("Rescaling state featurizers")
        self.rescale_featurizers(train_dataset)
        self.rescale_featurizers(validation_dataset)

        # Creating loss functions
        train_loss_fn = lambda inputs, logits, targets: (sigmoid_focal_loss(inputs, targets, reduction="sum")) / len(train_dataset)
        test_loss_fn = lambda inputs, targets: sigmoid_focal_loss(inputs, targets, reduction="sum")

        # Training loop
        logging.info("Running training")
        best_params = self.state_dict()
        best_params["iter"] = -1
        best_loss = float('inf')
        best_acceptance_rate = 0.0
        with (torch.autograd.detect_anomaly(True) if self._check_nans else nullcontext()):
            for itr in reversed(range(self._num_iters, 0, -1)):
                self.train()
                for x_train_batch, y_train_batch in train_dataset:
                    self._optimizer.zero_grad()
                    outputs, logits = self(x_train_batch)
                    train_loss = train_loss_fn(outputs, logits, torch.from_numpy(y_train_batch).to(self.device))
                    train_loss.backward()

                    l1_loss = l1_regularization(self) * self._l1_penalty
                    l1_loss.backward()

                    self._optimizer.step()

                if itr % 50 == 0:
                    # Reporting gradient statistics
                    for name, param in self.named_parameters():
                        logging.info(f"-- PARAM NAME {(name + ' '*70)[:70]} gradient stats: {tensor_stats(param.grad)}")

                    # Evaluating on a test dataset
                    self.eval()
                    y_pred_batches, y_true_batches = zip(*(
                        (self(x_test_batch)[0], torch.from_numpy(y_test_batch).to(self.device))
                        for x_test_batch, y_test_batch in validation_dataset
                    ))
                    y_pred, y_true = torch.concatenate(y_pred_batches), torch.concatenate(y_true_batches)

                    for name, param in self.state_dict().items():
                        logging.info(f"-- PARAM NAME {(name + ' '*70)[:70]} stats: {tensor_stats(param)}")

                    # Calculating the loss and accuracy
                    test_loss = float(test_loss_fn(y_pred, y_true).cpu().detach()) / len(validation_dataset)
                    matches = torch.logical_or(
                        torch.logical_and(torch.abs(y_true - 0.0) < 0.0001, y_pred <= self._thresh),
                        torch.logical_and(torch.abs(y_true - 1.0) < 0.0001, y_pred >= self._thresh)
                    ).cpu().detach().numpy()

                    # Calculating additional metrics
                    num_false_positives = torch.logical_and(torch.abs(y_true - 0.0) < 0.0001, y_pred >= self._thresh).cpu().detach().numpy().sum()
                    num_false_negatives = torch.logical_and(torch.abs(y_true - 1.0) < 0.0001, y_pred <= self._thresh).cpu().detach().numpy().sum()
                    num_positives = (torch.abs(y_true - 0.0) < 0.0001).cpu().detach().numpy().sum()

                    positive_confidence = float(torch.kthvalue(
                        torch.cat([y_pred[torch.abs(y_true - 0.0) < 0.0001].flatten(), tensor([0], device=self.device)]).cpu(),
                        int(num_positives * self._threshold_recalibration_frac) + 1
                    ).values)

                    acceptance_rate = (
                        torch.logical_and(torch.abs(y_true - 1.0) < 0.0001, y_pred >= positive_confidence).cpu().detach().numpy().sum() /
                        (torch.abs(y_true - 1.0) < 0.0001).cpu().detach().numpy().sum()
                    )

                    # Updating the best parameters
                    if best_loss > test_loss or (acceptance_rate >= best_acceptance_rate * 2):
                        del best_params
                        best_loss = test_loss
                        best_acceptance_rate = acceptance_rate
                        best_params = copy.deepcopy(self.state_dict())
                        best_params["iter"] = itr

                    logging.info(f"Train Loss: {float(train_loss.cpu().detach())}, Test Loss: {test_loss}, Acc: {matches.mean():.1%}, "
                                f"%False+: {num_false_positives/len(y_true):.1%}, %False-: {num_false_negatives/len(y_true):.1%}, "
                                f"{self._threshold_recalibration_frac:.0%} Positive Thresh: {positive_confidence:.4}, Acceptance rate: {acceptance_rate:.1%}, "
                                f"Training Iter {itr}/{self._num_iters}")

                    # Calculating accuracy per plan length
                    # per_length_matches = {}
                    # for y, ((encoder_skeleton, decoder_skeleton, _1, _2), label) in zip(y_pred, test_dataset):
                    #     n = len(encoder_skeleton) + len(decoder_skeleton)
                    #     if n not in per_length_matches:
                    #         per_length_matches[n] = ([], [])
                    #     per_length_matches[n][0].append(float(y.cpu()))
                    #     per_length_matches[n][1].append(label)
                    # for n, (sub_y_pred, sub_y_true) in per_length_matches.items():
                    #     sub_y_pred = np.array(sub_y_pred)
                    #     sub_y_true = np.array(sub_y_true)
                    #     sub_matches = np.logical_or(
                    #         np.logical_and(np.abs(sub_y_true - 0.0) < 0.0001, sub_y_pred <= self._thresh),
                    #         np.logical_and(np.abs(sub_y_true - 1.0) < 0.0001, sub_y_pred >= self._thresh)
                    #     )
                    #     sub_test_loss = float(test_loss_fn(torch.tensor(sub_y_pred), torch.tensor(sub_y_true)).cpu().detach()) / sub_y_pred.size
                    #     logging.info(f"Plan length {n}: Test Loss {sub_test_loss}, Acc: {sub_matches.mean():.1%}")
                    if training_snapshot_directory:
                        torch.save(self, os.path.join(training_snapshot_directory, f"model-{itr}.pt"))
                    if matches.mean() >= 0.95 and (itr - best_params["iter"]) >= 500:
                        break

        # Loading the best params
        logging.info(f"Best params from iter {best_params['iter']}")
        del best_params["iter"]
        self.load_state_dict(best_params)

        # Threshold recalibration and final metrics
        self.eval()
        y_pred_batches, y_true_batches = zip(*(
            (self(x_test_batch)[0], torch.from_numpy(y_test_batch).to(self.device))
            for x_test_batch, y_test_batch in validation_dataset
        ))
        y_pred, y_true = torch.concatenate(y_pred_batches), torch.concatenate(y_true_batches)

        test_loss = float(test_loss_fn(y_pred, y_true).cpu().detach()) / len(validation_dataset)
        matches = torch.logical_or(
            torch.logical_and(torch.abs(y_true - 0.0) < 0.0001, y_pred <= self._thresh),
            torch.logical_and(torch.abs(y_true - 1.0) < 0.0001, y_pred >= self._thresh)
        ).cpu().detach().numpy()

        logging.info(f"Final metrics: Test Loss: {test_loss}, Acc: {matches.mean():.1%}")

        num_positives = (torch.abs(y_true - 0.0) < 0.0001).cpu().detach().numpy().sum()
        positive_confidence = float(torch.kthvalue(torch.cat(
            [y_pred[torch.abs(y_true - 0.0) < 0.0001].flatten(), tensor([0], device=self.device)]
        ).cpu(), int(num_positives * self._threshold_recalibration_frac) + 1).values)

        self._unsure_confidence = max(self._thresh, positive_confidence)
        logging.info(f"Unsure confidence set to {self._unsure_confidence}")

        # Making sure we don't fit twice and turn on classification
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
            tokens = tokens,
            invalid_mask = invalid_mask,
            last_token_mask = last_token_mask,
            nsrt_indices = skeleton_nsrt_indices,
            pos_offset = pos_offsets,
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
            transformer_outputs[invalid_mask.unsqueeze(-1).expand(-1, -1, transformer_outputs.shape[2])] = 0
            classifier_tokens = transformer_outputs.sum(dim=1) / sequence_lengths.unsqueeze(-1)
            if not self.training:
                logging.info(f"-- FORWARD classifier outputs stats: {tensor_stats(self._classifier_head[0](transformer_outputs))}")
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
        dim = sum(v.type.dim for v in nsrt.parameters) + len(nsrt.parameters) # Object vectors + object ids
        featurizer = FeasibilityFeaturizer(
            name = name,
            sizes = [dim] + featurizer_sizes,
            dropout = dropout,
            device = self.device,
        )
        self.add_module(name, featurizer)
        return featurizer

    def rescale_featurizers(self, dataset: FeasibilityDataset) -> None:
        """Rescales the featurizers based on the states
        """
        for skeleton_nsrt, (min_state, max_state) in dataset.state_ranges.items():
            self._featurizers[skeleton_nsrt].update_range(min_state, max_state)