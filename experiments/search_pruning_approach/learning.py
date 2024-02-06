from functools import cache
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from torch import nn, Tensor, tensor
import torch
from predicators.settings import CFG
import logging

from predicators.structs import NSRT, _GroundNSRT, State, Variable

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("tkagg")

@dataclass(frozen=True)
class FeasibilityDatapoint:
    states: Sequence[State]
    skeleton: Sequence[_GroundNSRT]

    def __post_init__(self):
        assert 0 <= len(self.states) - 1 <= len(self.skeleton)

class FeasibilityDataset(torch.utils.data.Dataset):
    def __init__(self, positive_examples: Sequence[FeasibilityDatapoint], negative_examples: Sequence[FeasibilityDatapoint]):
        super().__init__()
        assert all(
            2 <= len(datapoint.states) <= len(datapoint.skeleton) # should be at least one encoder and decoder nsrt
            for datapoint in positive_examples + negative_examples
        )
        self._positive_examples = positive_examples
        self._negative_examples = negative_examples
        self._total_label_examples = max(len(positive_examples), len(negative_examples))

    def __len__(self) -> int:
        return self._total_label_examples * 2

    @cache
    def __getitem__(self, idx: int) -> Tuple[Tuple[List[NSRT], List[NSRT], List[npt.NDArray], List[npt.NDArray]], int]:
        if idx < self._total_label_examples:
            return self.transform_datapoint(self._positive_examples[idx % len(self._positive_examples)]), 1.0
        elif idx < self._total_label_examples * 2:
            return self.transform_datapoint(self._negative_examples[(idx - self._total_label_examples) % len(self._negative_examples)]), 0.0
        else:
            raise IndexError()

    @classmethod
    def transform_datapoint(
        cls, datapoint: FeasibilityDatapoint
    ) -> Tuple[Tuple[List[NSRT], List[NSRT], List[npt.NDArray], List[npt.NDArray]], int]:
        prefix_length = len(datapoint.states) - 1

        return [ground_nsrt.parent for ground_nsrt in datapoint.skeleton[:prefix_length]], \
            [ground_nsrt.parent for ground_nsrt in datapoint.skeleton[prefix_length:]], [
                state[ground_nsrt.objects]
                for state, ground_nsrt in zip(datapoint.states[1:], datapoint.skeleton)
            ], [
                datapoint.states[-1][ground_nsrt.objects]
                for ground_nsrt in datapoint.skeleton[prefix_length:]
            ]

class FeasibilityFeaturizer(nn.Module):
    def __init__(self, state_vector_size: int, hidden_sizes: List[int], feature_size: int, device: Optional[str] = None):
        super().__init__()
        self._device = device
        self._range = None

        sizes = [state_vector_size] + hidden_sizes + [feature_size]
        self._layers = nn.ModuleList([
            nn.Linear(input_size, output_size, device=device)
            for input_size, output_size in zip(sizes, sizes[1:])
        ])

    def update_range(self, state: npt.NDArray):
        state = tensor(state, device=self._device)
        if self._range is None:
            self._range = (state, state)
        else:
            min_state, max_state = self._range
            self._range = (torch.minimum(min_state, state), torch.maximum(max_state, state))

    def forward(self, state: npt.NDArray):
        state = tensor(state, device=self._device)

        if self._range is not None:
            min_state, max_state = self._range
            state -= min_state
            state /= torch.clamp(max_state - min_state, min=0.1)

        for layer in self._layers[:-1]:
            state = nn.functional.elu(layer(state))
        return self._layers[-1](state)

class PositionalEmbeddingLayer(nn.Module):
    def __init__(self, feature_size: int, embedding_size: int, concat: bool, include_cls: Optional[str], horizon: int, device: Optional[str] = None):
        super().__init__()
        assert include_cls in ['learned', 'marked', None]
        self._device = device
        self._concat = concat
        self._horizon = horizon

        if concat:
            self._input_size = feature_size - embedding_size - (include_cls == 'marked')
            self._embedding_size = embedding_size
        else:
            self._input_size = feature_size - (include_cls == 'marked')
            self._embedding_size = self._input_size

        if include_cls == 'learned':
            self._cls = nn.Parameter(torch.randn((1, feature_size), device=device))
            self._cls_marked = False
        elif include_cls == 'marked':
            self._cls = torch.concat([torch.zeros((1, feature_size - 1), device=device), torch.ones((1, 1), device=device)], dim=1)
            self._cls_marked = True
        else:
            self._cls = None
            self._cls_marked = False

    @property
    def input_size(self):
        return self._input_size

    def forward(self, tokens: Tensor) -> Tensor: # Assuming tokens.shape is (batch, max_len, size)
        batch_size, max_len, input_feature_size = tokens.shape
        assert input_feature_size == self._input_size
        positions = torch.arange(max_len, device=self._device).unsqueeze(-1)
        indices = torch.arange(self._embedding_size, device=self._device).unsqueeze(0)

        freq = 1 / self._horizon ** ((indices - (indices % 2)) / self._embedding_size)
        embeddings = torch.sin(positions.float() @ freq + torch.pi / 2 * (indices % 2))

        if self._concat:
            embedded_tokens = torch.cat([tokens, embeddings.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)
        else:
            embedded_tokens = tokens + embeddings

        if self._cls is None:
            return embedded_tokens
        elif self._cls_marked:
            marked_tokens = torch.cat([embedded_tokens, torch.zeros((batch_size, max_len, 1), device=self._device)], dim=2)
            return torch.cat([self._cls.unsqueeze(0).expand(batch_size, -1, -1), marked_tokens], dim=1)
        else:
            return torch.cat([self._cls.unsqueeze(0).expand(batch_size, -1, -1), embedded_tokens], dim=1)

    @classmethod
    def calculate_input_size(cls, feature_size: int, embedding_size: int, concat: bool, include_cls: Optional[str]) -> int:
        assert not concat or feature_size > embedding_size
        return (feature_size - embedding_size if concat else feature_size) - (include_cls == 'marked')

    def recalculate_mask(self, mask: Tensor) -> Tensor: # Assuming mask.shape is (batch, max_len)
        if self._cls is None:
            return mask
        return torch.cat([torch.full((mask.shape[0], 1), False, device=self._device), mask], dim=1)

FeasibilityClassifier = Callable[[Sequence[State], Sequence[_GroundNSRT]], bool]

# def get_soft_binning_ece_tensor(predictions, labels, soft_binning_bins,
#                                 soft_binning_use_decay,
#                                 soft_binning_decay_factor, soft_binning_temp):
#     soft_binning_anchors = torch.tensor(
#         np.arange(1.0 / (2.0 * soft_binning_bins), 1.0, 1.0 / soft_binning_bins),
#         dtype=torch.float32)

#     predictions_tile = predictions.unsqueeze(1).expand(-1, soft_binning_anchors.size(0))
#     predictions_tile = predictions_tile.unsqueeze(2)
#     bin_anchors_tile = soft_binning_anchors.unsqueeze(0).expand(predictions.size(0), -1)
#     bin_anchors_tile = bin_anchors_tile.unsqueeze(2)

#     if soft_binning_use_decay:
#         soft_binning_temp = 1 / (
#             math.log(soft_binning_decay_factor) * soft_binning_bins *
#             soft_binning_bins)

#     predictions_bin_anchors_product = torch.cat([predictions_tile, bin_anchors_tile], dim=2)

#     predictions_bin_anchors_differences = torch.sum(
#         torch.scan(
#             lambda _, row: torch.scan(
#                 lambda _, x: torch.tensor(
#                     [-((x[0] - x[1])**2) / soft_binning_temp, 0.],
#                     dtype=torch.float32),
#                 elems=row,
#                 initializer=torch.zeros(predictions_bin_anchors_product.size()[2])
#             ),
#             elems=predictions_bin_anchors_product,
#             initializer=torch.zeros(predictions_bin_anchors_product.size()[1:])
#         ),
#         dim=2
#     )

#     predictions_soft_binning_coeffs = F.softmax(predictions_bin_anchors_differences, dim=1)

#     sum_coeffs_for_bin = torch.sum(predictions_soft_binning_coeffs, dim=[0])

#     intermediate_predictions_reshaped_tensor = predictions.repeat(soft_binning_anchors.size())
#     intermediate_predictions_reshaped_tensor = intermediate_predictions_reshaped_tensor.view(
#         predictions_soft_binning_coeffs.size())
#     net_bin_confidence = torch.sum(
#         intermediate_predictions_reshaped_tensor * predictions_soft_binning_coeffs,
#         dim=[0]
#     ) / torch.max(sum_coeffs_for_bin, EPS * torch.ones_like(sum_coeffs_for_bin))

#     intermediate_labels_reshaped_tensor = labels.repeat(soft_binning_anchors.size())
#     intermediate_labels_reshaped_tensor = intermediate_labels_reshaped_tensor.view(
#         predictions_soft_binning_coeffs.size())
#     net_bin_accuracy = torch.sum(
#         intermediate_labels_reshaped_tensor * predictions_soft_binning_coeffs,
#         dim=[0]
#     ) / torch.max(sum_coeffs_for_bin, EPS * torch.ones_like(sum_coeffs_for_bin))

#     bin_weights = sum_coeffs_for_bin / torch.norm(sum_coeffs_for_bin, p=1)
#     soft_binning_ece = torch.sqrt(
#         torch.tensordot(
#             (net_bin_confidence - net_bin_accuracy).pow(2),
#             bin_weights,
#             dims=1,
#         )
#     )

#     return soft_binning_ece

class NeuralFeasibilityClassifier(nn.Module):
    def __init__(self,
        featurizer_hidden_sizes: List[int],
        classifier_feature_size: int,
        positional_embedding_size: int,
        positional_embedding_concat: bool,
        transformer_num_heads: int,
        transformer_encoder_num_layers: int,
        transformer_decoder_num_layers: int,
        transformer_ffn_hidden_size: int,
        max_train_iters: int,
        general_lr: float,
        transformer_lr: float,
        max_inference_suffix: int,
        cls_style: str,
        embedding_horizon: int,
        batch_size: int,
        early_stopping_loss_thresh: float = 0.0001,
        validate_split: float = 0.125,
        dropout: int = 0.25,
        num_threads: int = 8,
        classification_threshold: float = 0.8,
        device: Optional[str] = 'cuda'
    ):
        super().__init__()
        torch.set_num_threads(num_threads)
        self._device = device

        assert cls_style in {'learned', 'marked', 'mean'}
        self._cls_style = cls_style

        self._max_inference_suffix = max_inference_suffix
        self._thresh = classification_threshold

        self._early_stopping_thresh = early_stopping_loss_thresh
        self._num_iters = max_train_iters
        self._batch_size = batch_size
        self._validate_split = validate_split
        self._general_lr = general_lr
        self._transformer_lr = transformer_lr

        self._encoder_featurizers: Dict[NSRT, FeasibilityFeaturizer] = {} # Initialized with self._init_featurizer
        self._decoder_featurizers: Dict[NSRT, FeasibilityFeaturizer] = {} # Initialized with self._init_featurizer
        self._featurizer_hidden_sizes = featurizer_hidden_sizes
        self._featurizer_count: int = 0 # For naming the module when adding it in self._init_featurizer

        self._encoder_positional_encoding = PositionalEmbeddingLayer(
            feature_size = classifier_feature_size,
            embedding_size = positional_embedding_size,
            concat = positional_embedding_concat,
            include_cls = None,
            horizon = embedding_horizon,
            device = device,

        )
        self._decoder_positional_encoding = PositionalEmbeddingLayer(
            feature_size = classifier_feature_size,
            embedding_size = positional_embedding_size,
            concat = positional_embedding_concat,
            include_cls = {
                'mean': None, 'learned': 'learned', 'marked': 'marked'
            }[cls_style],
            horizon = embedding_horizon,
            device = device,
        )
        self._transformer = nn.Transformer(
            d_model = classifier_feature_size,
            nhead = transformer_num_heads,
            num_encoder_layers = transformer_encoder_num_layers,
            num_decoder_layers = transformer_decoder_num_layers,
            dim_feedforward = transformer_ffn_hidden_size,
            dropout = dropout,
            batch_first = True,
            device = device,
        )
        self._classifier_head = nn.Sequential(
            nn.Linear(classifier_feature_size, 1, device=device),
            nn.Sigmoid(),
        )

        self._optimizer: Optional[torch.optim.Optimizer] = None

    def classify(self, states: Sequence[State], skeleton: Sequence[_GroundNSRT]) -> bool:
        if len(states) == len(skeleton) + 1: # Make sure there is at least one decoder nsrt
            return True
        if len(skeleton) - self._max_inference_suffix >= len(states): # Make sure we don't have too big of a horizon to predict
            return True

        encoder_nsrts, decoder_nsrts, encoder_states, decoder_states = \
            FeasibilityDataset.transform_datapoint(FeasibilityDatapoint(states, skeleton))
        self._init_featurizers_datapoint(encoder_nsrts, decoder_nsrts, encoder_states, decoder_states)

        self.eval()
        confidence = self([encoder_nsrts], [decoder_nsrts], [encoder_states], [decoder_states]).cpu()
        print(f"Confidence {float(confidence)}")
        return confidence >= self._thresh

    def fit(self, positive_examples: Sequence[FeasibilityDatapoint], negative_examples: Sequence[FeasibilityDatapoint]) -> None:
        self._create_optimizer()
        if not positive_examples and not negative_examples:
            return

        logging.info("Training Feasibility Classifier...")


        # Creating datasets
        logging.info(f"Creating datasets (validate split {int(self._validate_split*100)}%)")

        positive_examples, negative_examples = positive_examples.copy(), negative_examples.copy()
        np.random.shuffle(positive_examples)
        np.random.shuffle(negative_examples)

        positive_train_size = len(positive_examples) - int(len(positive_examples) * self._validate_split)
        negative_train_size = len(negative_examples) - int(len(negative_examples) * self._validate_split)
        train_dataset = FeasibilityDataset(positive_examples[:positive_train_size], negative_examples[:negative_train_size])
        validate_dataset = FeasibilityDataset(positive_examples[positive_train_size:], negative_examples[negative_train_size:])


        # Initializing per-nsrt featurizers if not initialized already
        logging.info("Initializing state featurizers")
        self._init_featurizers_dataset(train_dataset)
        self._init_featurizers_dataset(validate_dataset)

        # Setting up dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=self._collate_batch
        )
        validate_dataloader = torch.utils.data.DataLoader(
            validate_dataset,
            batch_size=self._batch_size,
            collate_fn=self._collate_batch
        )

        # Training loop
        logging.info("Running training")
        train_loss_fn = nn.BCELoss()
        validate_loss_fn = nn.BCELoss()
        # with torch.autograd.detect_anomaly(False):
        for itr, (x_train_batch, y_train_batch) in zip(range(self._num_iters), itertools.cycle(train_dataloader)):
            self.train()
            self._optimizer.zero_grad()
            train_loss = train_loss_fn(self(*x_train_batch), y_train_batch)
            train_loss.backward()
            self._optimizer.step()
            if itr % 100 == 0:
                self.eval()
                y_pred_batches, y_true_batches = zip(*(
                    (self(*x_validate_batch), y_validate_batch)
                    for x_validate_batch, y_validate_batch in validate_dataloader
                ))
                y_pred, y_true = torch.concatenate(y_pred_batches), torch.concatenate(y_true_batches)

                validate_loss = float(validate_loss_fn(y_pred, y_true).cpu().detach())
                matches = torch.logical_or(
                    torch.logical_and(torch.abs(y_true - 0.0) < 0.0001, y_pred <= self._thresh),
                    torch.logical_and(torch.abs(y_true - 1.0) < 0.0001, y_pred >= self._thresh)
                ).cpu().detach().numpy()

                num_false_positives = torch.logical_and(torch.abs(y_true - 0.0) < 0.0001, y_pred >= self._thresh).cpu().detach().numpy().sum()

                false_positive_max_confidence = float(torch.cat([y_pred[
                    torch.logical_and(torch.abs(y_true - 0.0) < 0.0001, y_pred >= self._thresh)
                ].flatten(), tensor([0], device=self._device)]).max().cpu().detach())

                # false_positive_max_confidence_matches = torch.logical_or(
                #     torch.logical_and(torch.abs(y_true - 0.0) < 0.0001, y_pred <= false_positive_max_confidence),
                #     torch.logical_and(torch.abs(y_true - 1.0) < 0.0001, y_pred >= false_positive_max_confidence)
                # ).cpu().detach().numpy()

                acceptance_rate = (
                    torch.logical_and(torch.abs(y_true - 1.0) < 0.0001, y_pred >= false_positive_max_confidence).cpu().detach().numpy().sum() /
                    (torch.abs(y_true - 1.0) < 0.0001).cpu().detach().numpy().sum()
                )

                logging.info(f"Loss: {validate_loss}, Acc: {matches.mean():.1%}"
                             f"Worst False+: {false_positive_max_confidence:.4}, Acceptance rate: {acceptance_rate:.1%}, "
                             f"Training Iter {itr}/{self._num_iters}")
                if validate_loss <= self._early_stopping_thresh:
                    break

        if CFG.feasibility_loss_output_file:
            validate_loss = np.concatenate([
                validate_loss_fn(self(*x_validate_batch), y_validate_batch).detach().numpy()
                for x_validate_batch, y_validate_batch in validate_dataloader
            ]).mean()
            print(validate_loss, file=open(CFG.feasibility_loss_output_file, "w"))
            raise RuntimeError()

        # Threshold recalibration TODO: revisit this and add training-time calibration
        y_pred_batches, y_true_batches = zip(*(
            (self(*x_validate_batch), y_validate_batch)
            for x_validate_batch, y_validate_batch in validate_dataloader
        ))
        y_pred, y_true = torch.concatenate(y_pred_batches), torch.concatenate(y_true_batches)

        false_positive_max_confidence = float(torch.cat([y_pred[
            torch.logical_and(torch.abs(y_true - 0.0) < 0.0001, y_pred >= self._thresh)
        ].flatten(), tensor([0], device=self._device)]).max().cpu().detach())

        self._thresh = false_positive_max_confidence


    def _collate_batch(
        self, batch: Sequence[Tuple[Tuple[Sequence[NSRT], Sequence[NSRT], Sequence[npt.NDArray], Sequence[npt.NDArray]], int]]
    ) -> Tuple[Tuple[Sequence[Sequence[NSRT]], Sequence[Sequence[NSRT]], Sequence[Sequence[npt.NDArray]], Sequence[Sequence[npt.NDArray]]], Tensor]:
        """ Convert a batch of datapoints to batched datapoints
        """
        return (
            [dp[0][0] for dp in batch], [dp[0][1] for dp in batch],
            [dp[0][2] for dp in batch], [dp[0][3] for dp in batch]
        ), tensor([dp[1] for dp in batch], device=self._device)

    def _create_optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam([
                {'params':
                    list(self._encoder_positional_encoding.parameters()) +
                    list(self._decoder_positional_encoding.parameters())
                , 'lr': self._general_lr},
                {'params': self._classifier_head.parameters(), 'lr': self._general_lr},
                {'params': self._transformer.parameters(), 'lr': self._transformer_lr},
            ])

    def forward(
        self,
        encoder_nsrts_batch: Sequence[Sequence[NSRT]],
        decoder_nsrts_batch: Sequence[Sequence[NSRT]],
        encoder_states_batch: Sequence[Sequence[npt.NDArray]],
        decoder_states_batch: Sequence[Sequence[npt.NDArray]],
    ) -> float:
        assert len(encoder_nsrts_batch) == len(decoder_nsrts_batch) and\
            len(decoder_nsrts_batch) == len(encoder_states_batch) and\
            len(encoder_states_batch) == len(decoder_states_batch)
        assert all(
            len(encoder_states) == len(encoder_nsrts)
            for encoder_states, encoder_nsrts in zip(encoder_states_batch, encoder_nsrts_batch)
        )
        encoder_tokens, encoder_mask = self._run_featurizers(
            self._encoder_featurizers, self._encoder_positional_encoding.input_size, encoder_states_batch, encoder_nsrts_batch
        )
        decoder_tokens, decoder_mask = self._run_featurizers(
            self._decoder_featurizers, self._decoder_positional_encoding.input_size, decoder_states_batch, decoder_nsrts_batch
        )

        encoder_mask = self._encoder_positional_encoding.recalculate_mask(encoder_mask)
        decoder_mask = self._decoder_positional_encoding.recalculate_mask(decoder_mask)
        transformer_outputs = self._transformer(
            src=self._encoder_positional_encoding(encoder_tokens),
            tgt=self._decoder_positional_encoding(decoder_tokens),
            src_key_padding_mask=encoder_mask,
            tgt_key_padding_mask=decoder_mask,
            memory_key_padding_mask=encoder_mask,
            # src_is_causal=False,
            # tgt_is_causal=False,
        )

        if self._cls_style == 'mean':
            classifier_tokens = torch.stack([
                sequence[torch.logical_not(mask)].mean(dim=0)
                for sequence, mask in zip(transformer_outputs, decoder_mask)
            ])
        else:
            classifier_tokens = transformer_outputs[:, 0]
        output = self._classifier_head(classifier_tokens).flatten()

        return output

    def _run_featurizers(
        self,
        featurizers: Dict[NSRT, nn.Module],
        output_size: int,
        states_batch: Iterable[Sequence[npt.NDArray]],
        nsrts_batch: Iterable[Iterable[NSRT]],
    ) -> Tuple[Tensor, Tensor]:
        """ Runs state featurizers that are executed before passing the states into the main transformer

        Outputs the transformer inputs and the padding mask for them
        """

        # All of those shenenigans are to batch the execution of featurizers
        batch_size = len(states_batch)
        max_len = max(len(states) for states in states_batch)

        tokens = torch.zeros((batch_size, max_len, output_size), device=self._device)
        mask = torch.full((batch_size, max_len), True) # Not on device for fast assignment

        grouped_data: Dict[NSRT, List[Tuple[Tensor, int, int]]] = {nsrt: [] for nsrt in featurizers.keys()}
        for batch_idx, states, nsrts in zip(range(batch_size), states_batch, nsrts_batch):
            for seq_idx, state, nsrt in zip(range(max_len), states, nsrts):
                grouped_data[nsrt].append((state, batch_idx, seq_idx))
                mask[batch_idx, seq_idx] = False

        for nsrt, data in grouped_data.items():
            if not data:
                continue
            states, batch_indices, seq_indices = zip(*data)
            tokens[batch_indices, seq_indices, :] = featurizers[nsrt](np.stack(states))

        if mask.device != tokens.device:
            mask = mask.to(self._device)
        return tokens, mask

    def _init_featurizers_dataset(self, dataset: FeasibilityDataset) -> None:
        """ Initializes featurizers that should be learned from that datapoint
        """
        for idx in range(len(dataset)):
            (encoder_nsrts, decoder_nsrts, encoder_states, decoder_states), _ = dataset[idx]
            self._init_featurizers_datapoint(encoder_nsrts, decoder_nsrts, encoder_states, decoder_states)

    def _init_featurizers_datapoint(
        self,
        encoder_nsrts: Sequence[NSRT],
        decoder_nsrts: Sequence[NSRT],
        encoder_states: Sequence[npt.NDArray],
        decoder_states: Sequence[npt.NDArray]
    ):
        for nsrt, state in zip(encoder_nsrts, encoder_states):
            self._init_featurizer(self._encoder_featurizers, self._encoder_positional_encoding.input_size, state, nsrt)
        for nsrt, state in zip(decoder_nsrts, decoder_states):
            self._init_featurizer(self._decoder_featurizers, self._decoder_positional_encoding.input_size, state, nsrt)

    def _init_featurizer(self, featurizers: Dict[NSRT, nn.Module], output_size: int, state: npt.NDArray, nsrt: NSRT) -> None:
        """ Initializes a featurizer for a single ground nsrt.

        NOTE: The assumption is that the concatentated features of all objects that
        are passed to a given NSRT always have the same data layout and total length.
        """
        assert len(state.shape) == 1 and self._optimizer
        if nsrt not in featurizers:
            featurizer = FeasibilityFeaturizer(
                state.size,
                hidden_sizes = self._featurizer_hidden_sizes,
                feature_size = output_size,
                device = self._device
            )
            self._featurizer_count += 1

            self.add_module(f"featurizer_{self._featurizer_count}", featurizer)
            self._optimizer.add_param_group(
                {'params': featurizer.parameters(), 'lr': self._general_lr}
            )

            featurizers[nsrt] = featurizer
        else:
            featurizer = featurizers[nsrt]
        featurizer.update_range(state)