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
matplotlib.use("tkagg")

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
    def __init__(self, feature_size: int, embedding_size: int, concat: bool, include_cls: bool, horizon: int, device: Optional[str] = None):
        super().__init__()
        self._device = device
        self._expected_input_size = self.calculate_input_size(feature_size, embedding_size, concat)
        self._size = embedding_size if concat else feature_size
        self._concat = concat
        self._horizon = horizon
        self._cls = nn.Parameter(torch.randn((1, feature_size), device=device)) if include_cls else None

    def forward(self, tokens: Tensor) -> Tensor: # Assuming tokens.shape is (batch, max_len, size)
        batch_size, max_len, input_feature_size = tokens.shape
        assert input_feature_size == self._expected_input_size
        positions = torch.arange(max_len, device=self._device).unsqueeze(-1)
        indices = torch.arange(self._size, device=self._device).unsqueeze(0)

        freq = 1 / self._horizon ** ((indices - (indices % 2)) / self._size)
        embeddings = torch.sin(positions.float() @ freq + torch.pi / 2 * (indices % 2))

        if self._concat:
            embedded_tokens = torch.cat([tokens, embeddings.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)
        else:
            embedded_tokens = tokens + embeddings

        if self._cls is None:
            return embedded_tokens
        return torch.cat([self._cls.unsqueeze(0).expand(batch_size, -1,-1), embedded_tokens], dim=1)

    @classmethod
    def calculate_input_size(cls, feature_size: int, embedding_size: int, concat: bool) -> int:
        assert not concat or feature_size > embedding_size
        return feature_size - embedding_size if concat else feature_size

    def recalculate_mask(self, mask: Tensor) -> Tensor: # Assuming mask.shape is (batch, max_len)
        if self._cls is None:
            return mask
        return torch.cat([torch.full((mask.shape[0], 1), False, device=self._device), mask], dim=1)

FeasibilityClassifier = Callable[[Sequence[State], Sequence[_GroundNSRT]], bool]

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
        lr: float,
        max_inference_suffix: int,
        batch_size: int = 128,
        validate_split: float = 0.2,
        dropout: int = 0,
        num_threads: int = 8,
        classification_threshold: float = 0.5,
        device: Optional[str] = 'cuda'
    ):
        super().__init__()
        torch.set_num_threads(num_threads)
        self._device = device

        self._thresh = classification_threshold
        self._num_iters = max_train_iters
        self._batch_size = batch_size
        self._validate_split = validate_split
        self._lr = lr

        self._featurizer_output_size = PositionalEmbeddingLayer.calculate_input_size(
            classifier_feature_size, positional_embedding_size, positional_embedding_concat
        )
        self._featurizer_hidden_sizes = featurizer_hidden_sizes

        self._encoder_featurizers: Dict[NSRT, FeasibilityFeaturizer] = {} # Initialized with self._init_featurizer
        self._decoder_featurizers: Dict[NSRT, FeasibilityFeaturizer] = {} # Initialized with self._init_featurizer
        self._featurizer_count: int = 0 # For naming the module when adding it in self._init_featurizer

        self._encoder_positional_encoding = PositionalEmbeddingLayer(
            feature_size = classifier_feature_size,
            embedding_size = positional_embedding_size,
            concat = positional_embedding_concat,
            include_cls = False,
            horizon = CFG.horizon,
            device = device,

        )
        self._decoder_positional_encoding = PositionalEmbeddingLayer(
            feature_size = classifier_feature_size,
            embedding_size = positional_embedding_size,
            concat = positional_embedding_concat,
            include_cls = True,
            horizon = CFG.horizon,
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
            nn.Sigmoid()
        )
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._max_inference_suffix = max_inference_suffix

    def classify(self, states: Sequence[State], skeleton: Sequence[_GroundNSRT]) -> bool:
        if len(states) == len(skeleton) + 1: # Make sure there is at least one decoder nsrt
            return True
        if len(skeleton) - self._max_inference_suffix < len(states): # Make sure we don't have too big of a horizon to predict
            return True

        encoder_nsrts, decoder_nsrts, encoder_states, decoder_states = \
            FeasibilityDataset.transform_datapoint(FeasibilityDatapoint(states, skeleton))
        self._init_featurizers_datapoint(encoder_nsrts, decoder_nsrts, encoder_states, decoder_states)

        self.eval()
        confidence = self([encoder_nsrts], [decoder_nsrts], [encoder_states], [decoder_states])
        print(confidence)
        return confidence >= self._thresh

    def fit(self, positive_examples: Sequence[FeasibilityDatapoint], negative_examples: Sequence[FeasibilityDatapoint]) -> None:
        self._create_optimizer()
        if not positive_examples and not negative_examples == 0:
            return

        logging.info("Training Feasibility Classifier...")


        # Creating datasets
        logging(f"Creating datasets (validate split {int(self._validate_split*100)}%)")

        positive_examples, negative_examples = positive_examples.copy(), negative_examples.copy()
        np.random.shuffle(positive_examples)
        np.random.shuffle(negative_examples)

        positive_train_size = len(positive_examples) - int(len(positive_examples) * self._validate_split)
        negative_train_size = len(negative_examples) - int(len(negative_examples) * self._validate_split)
        train_dataset = FeasibilityDataset(positive_examples[:positive_train_size], negative_examples[:negative_train_size])
        validate_dataset = FeasibilityDataset(positive_examples[positive_train_size:], negative_examples[negative_train_size:])


        # Initializing per-nsrt featurizers if not initialized already
        logging("Initializing state featurizers")
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
        logging("Running training")
        train_loss_fn = nn.BCELoss()
        validate_loss_fn = nn.BCELoss(reduction='none')
        with torch.autograd.detect_anomaly(False):
            for itr, (x_train_batch, y_train_batch) in zip(range(self._num_iters), itertools.cycle(train_dataloader)):
                self.train()
                self._optimizer.zero_grad()
                train_loss = train_loss_fn(self(*x_train_batch), y_train_batch)
                train_loss.backward()
                self._optimizer.step()
                if itr % 100 == 0:
                    self.eval()
                    validate_loss = np.concatenate([
                        validate_loss_fn(self(*x_validate_batch), y_validate_batch).detach().cpu().numpy()
                        for x_validate_batch, y_validate_batch in validate_dataloader
                    ]).mean()

                    logging(f"Loss: {validate_loss}, Training Iteration {itr}/{self._num_iters}")
                    # print("-"*10)
                    # print(y_test_batch.min(), y_test_batch.max(), y_test_batch.mean(), y_test_batch.std(), y_test_batch.shape)
                    # print(y_test_batch)
                    # print("-"*10)
                    # print(y_test_pred)
                    # print(y_test_pred.min(), y_test_pred.max(), y_test_pred.mean(), y_test_pred.std(), y_test_pred.shape)
                    # print("-"*10)
                    # print([[(param.grad.min(), param.grad.max(), param.grad.mean(), param.grad.std()) if param.grad is not None else None for param in featurizer.parameters() if param.numel()] for featurizer in list(self._encoder_featurizers.values()) + list(self._decoder_featurizers.values())])
                    # print("-"*10)
                    # print([(param.grad.min(), param.grad.max(), param.grad.mean(), param.grad.std()) if param.grad is not None else None for param in list(self._encoder_positional_encoding.parameters()) + list(self._decoder_positional_encoding.parameters()) if param.numel()])
                    # print("-"*10)
                    # print([(param.grad.min(), param.grad.max(), param.grad.mean(), param.grad.std()) if param.grad is not None else None for param in self._transformer.parameters() if param.numel()])
                    # print("-"*10)
                    # print([(param.grad.min(), param.grad.max(), param.grad.mean(), param.grad.std()) if param.grad is not None else None for param in self._classifier_head.parameters() if param.numel()])
                    # print("-"*10)
                    # raise RuntimeError()
        validate_loss = np.concatenate([
            validate_loss_fn(self(*x_validate_batch), y_validate_batch).detach().numpy()
            for x_validate_batch, y_validate_batch in validate_dataloader
        ]).mean()
        print(validate_loss, file=open("w", CFG.feasibility_loss_output_file))
        raise RuntimeError()

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
            assert list(self._encoder_positional_encoding.parameters()) + list(self._decoder_positional_encoding.parameters())
            self._optimizer = torch.optim.Adam([
                {'params':
                    list(self._encoder_positional_encoding.parameters()) +
                    list(self._decoder_positional_encoding.parameters())
                , 'lr': CFG.feasibility_general_lr},
                {'params': self._classifier_head.parameters(), 'lr': CFG.feasibility_general_lr},
                {'params': self._transformer.parameters(), 'lr': CFG.feasibility_transformer_lr},
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
        encoder_tokens, encoder_mask = self._run_featurizers(self._encoder_featurizers, encoder_states_batch, encoder_nsrts_batch)
        decoder_tokens, decoder_mask = self._run_featurizers(self._decoder_featurizers, decoder_states_batch, decoder_nsrts_batch)

        encoder_mask = self._encoder_positional_encoding.recalculate_mask(encoder_mask)
        decoder_mask = self._decoder_positional_encoding.recalculate_mask(decoder_mask)
        transformer_outputs = self._transformer(
            src=self._encoder_positional_encoding(encoder_tokens),
            tgt=self._decoder_positional_encoding(decoder_tokens),
            src_key_padding_mask=encoder_mask,
            tgt_key_padding_mask=decoder_mask,
            memory_key_padding_mask=encoder_mask,
            src_is_causal=False,
            tgt_is_causal=False,
        )
        return self._classifier_head(transformer_outputs[:, 0]).flatten()

    def _run_featurizers(
        self,
        featurizers: Dict[NSRT, nn.Module],
        states_batch: Iterable[Sequence[npt.NDArray]],
        nsrts_batch: Iterable[Iterable[NSRT]],
    ) -> Tuple[Tensor, Tensor]:
        """ Runs state featurizers that are executed before passing the states into the main transformer

        Outputs the transformer inputs and the padding mask for them
        """
        # print(states_batch, [[nsrt.name for nsrt in nsrts] for nsrts in nsrts_batch])
        tokens_batch = [
            torch.stack([
                featurizers[nsrt](state)
                for state, nsrt in zip(states, nsrts)
            ]) for states, nsrts in zip(states_batch, nsrts_batch)
        ]
        mask = [
            torch.full((len(tokens),), False, device=self._device)
            for tokens in tokens_batch
        ]
        return nn.utils.rnn.pad_sequence(tokens_batch, batch_first=True),\
            nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=True)

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
            self._init_featurizer(self._encoder_featurizers, state, nsrt)
        for nsrt, state in zip(decoder_nsrts, decoder_states):
            self._init_featurizer(self._decoder_featurizers, state, nsrt)

    def _init_featurizer(self, featurizers: Dict[NSRT, nn.Module], state: npt.NDArray, nsrt: NSRT) -> None:
        """ Initializes a featurizer for a single ground nsrt.

        NOTE: The assumption is that the concatentated features of all objects that
        are passed to a given NSRT always have the same data layout and total length.
        """
        assert len(state.shape) == 1 and self._optimizer
        if nsrt not in featurizers:
            featurizer = FeasibilityFeaturizer(
                state.size,
                hidden_sizes = self._featurizer_hidden_sizes,
                feature_size = self._featurizer_output_size,
                device = self._device
            )
            self._featurizer_count += 1

            self.add_module(f"featurizer_{self._featurizer_count}", featurizer)
            self._optimizer.add_param_group(
                {'params': featurizer.parameters(), 'lr': CFG.feasibility_general_lr}
            )

            featurizers[nsrt] = featurizer
        else:
            featurizer = featurizers[nsrt]
        featurizer.update_range(state)