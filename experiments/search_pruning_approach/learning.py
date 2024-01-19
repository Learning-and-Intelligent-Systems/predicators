from dataclasses import dataclass
import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from torch import nn, Tensor
import torch

from predicators.structs import NSRT, _GroundNSRT, State, Variable

@dataclass(frozen=True)
class FeasibilityDatapoint:
    previous_states: Sequence[State]
    next_state: State
    encoder_nsrts: Sequence[_GroundNSRT]
    decoder_nsrts: Sequence[_GroundNSRT]

    def __post_init__(self):
        assert len(self.previous_states) == len(self.encoder_nsrts)

class FeasibilityDataset(torch.utils.data.Dataset):
    def __init__(self, positive_examples: Sequence[FeasibilityDatapoint], negative_examples: Sequence[FeasibilityDatapoint]):
        assert all(
            len(datapoint.previous_states) == len(datapoint.encoder_nsrts)
            for datapoint in positive_examples + negative_examples
        )
        self._positive_examples = positive_examples
        self._negative_examples = negative_examples
        self._total_label_examples = max(len(positive_examples), len(negative_examples))

    def __len__(self):
        return len(self._positive_examples) + len(self._negative_examples)

    def __getitem__(self, idx):
        if idx < self._total_label_examples:
            return self.transform_datapoint(self._positive_examples[idx % len(self._positive_examples)]), 1
        else:
            return self.transform_datapoint(self._negative_examples[(idx - self._total_label_examples) % len(self._negative_examples)]), 0

    @classmethod
    def transform_datapoint(
        cls, datapoint: FeasibilityDatapoint
    ) -> Tuple[Tuple[Sequence[NSRT], Sequence[NSRT], Tensor, Tensor], int]:
        return [ground_nsrt.parent for ground_nsrt in datapoint.encoder_nsrts],\
            [ground_nsrt.parent for ground_nsrt in datapoint.decoder_nsrts],\
            torch.nested.nested_tensor([
                Tensor(previous_state[ground_nsrt.objects])
                for previous_state, ground_nsrt in zip(datapoint.previous_states, datapoint.encoder_nsrts)
            ]), torch.nested.nested_tensor([
                Tensor(datapoint.next_state[ground_nsrt.objects])
                for ground_nsrt in datapoint.decoder_nsrts
            ])

class PositionalEmbeddingLayer(nn.Module):
    @classmethod
    def __init__(self, feature_size: int, embedding_size: int, concat: bool, include_cls: bool, horizon: int = 1e2):
        self._size = embedding_size if concat else feature_size
        self._concat = concat
        self._horizon = horizon
        self._cls_list = [nn.Parameter(torch.randn(self._size))] if include_cls else []
    def forward(self, tokens_batch: Tensor): # Assuming tokens.size is (batch, ?, size)
        max_len = max(tokens.shape[0] for tokens in tokens_batch)
        positions = torch.arange(max_len).unsqueeze(-1)
        indices = torch.arange(self._size).unsqueeze(0)

        freq = 1 / self._horizon ** ((indices - (indices % 2)) / self._size)
        embeddings = torch.sin((freq + torch.pi / 2 * (indices % 2)) @ positions)

        return torch.nested.as_nested_tensor([
            torch.cat([tokens + embeddings[:tokens.shape[0]]] + self._cls_list)
            for tokens in tokens_batch
        ])

    @classmethod
    def calculate_input_size(cls, feature_size: int, embedding_size: int, concat: bool):
        assert concat or feature_size > embedding_size
        return feature_size if concat else feature_size - embedding_size

FeasibilityClassifier = Callable[[Sequence[State], Sequence[_GroundNSRT]], bool]

class NeuralFeasibilityClassifier(nn.Module):
    def __init__(self,
        encoder_hidden_sizes: List[int],
        classifier_feature_size: int,
        positional_embedding_size: int,
        positional_embedding_concat: bool,
        transformer_num_heads: int,
        transformer_encoder_num_layers: int,
        transformer_decoder_num_layers: int,
        transformer_ffn_hidden_size: int,
        max_train_iters: int,
        lr: float,
        max_suffix: int,
        batch_size: int = 1024,
        dropout: int = 0,
        num_threads: int = 8,
        classification_threshold: float = 0.5,
    ):
        super().__init__()
        torch.set_num_threads(num_threads)
        self._thresh = classification_threshold
        self._num_iters = max_train_iters
        self._num_threads = num_threads
        self._batch_size = batch_size
        self._lr = lr

        self._encoder_output_size = PositionalEmbeddingLayer.calculate_input_size(
            classifier_feature_size, positional_embedding_size, positional_embedding_concat
        )
        self._encoder_hidden_sizes = encoder_hidden_sizes

        self._encoder_encoders: Dict[NSRT, nn.Module] = {} # Initialized with self._init_encoder
        self._decoder_encoders: Dict[NSRT, nn.Module] = {} # Initialized with self._init_encoder
        self._encoder_positional_encoding = PositionalEmbeddingLayer(
            feature_size = classifier_feature_size,
            embedding_size = positional_embedding_size,
            concat = positional_embedding_concat,
            include_cls = False
        )
        self._decoder_positional_encoding = PositionalEmbeddingLayer(
            feature_size = classifier_feature_size,
            embedding_size = positional_embedding_size,
            concat = positional_embedding_concat,
            include_cls = True
        )
        self._transformer = nn.Transformer(
            d_model = classifier_feature_size,
            nhead = transformer_num_heads,
            num_encoder_layers = transformer_encoder_num_layers,
            num_decoder_layers = transformer_decoder_num_layers,
            dim_feedforward = transformer_ffn_hidden_size,
            dropout = dropout,
            batch_first = True
        )
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._max_suffix = max_suffix

    def classify(self, states: Sequence[State], skeleton: Sequence[_GroundNSRT]) -> bool:
        assert states and len(states) <= len(skeleton) + 1
        if len(states) == len(skeleton) + 1: # Make sure there is at least one decoder nsrt
            return True

        previous_states = states[:-1]
        next_state = states[-1]
        encoder_nsrts = skeleton[:len(states) - 1]
        decoder_nsrts = skeleton[len(states) - 1:]
        datapoint = FeasibilityDatapoint(previous_states, next_state, encoder_nsrts, decoder_nsrts)
        if len(decoder_nsrts) > self._max_suffix: # Make sure we don't have too big of a horizon to predict
            return True

        self._init_encoders_datapoint(datapoint)
        self.eval()
        encoder_nsrts, decoder_nsrts, encoder_states, decoder_state = FeasibilityDataset.transform_datapoint(datapoint)
        return self(previous_states, next_state, encoder_nsrts, decoder_nsrts) >= self._thresh

    def fit(self, positive_examples: Sequence[FeasibilityDatapoint], negative_examples: Sequence[FeasibilityDatapoint]) -> None:
        self._create_optimizer()

        # Initializing per-nsrt encoders if not initialized already
        for datapoint in positive_examples + negative_examples:
            self._init_encoders_datapoint(datapoint)

        # Setting up dataset
        dataset = FeasibilityDataset(positive_examples, negative_examples)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_threads,
            collate_fn=self._collate_batch
        )

        # Training loop
        loss_fn = nn.BCELoss()
        self.train()
        for itr, (x_batch, y_batch) in zip(range(self._num_iters), itertools.cycle(dataloader)):
            self._optimizer.zero_grad()
            feasibility = self()
            loss = loss_fn(self(*x_batch), y_batch)
            loss.backward()
            self._optimizer.step()

    @classmethod
    def _collate_batch(
        cls, batch: Sequence[Tuple[Tuple[Sequence[NSRT], Sequence[NSRT], Tensor, Tensor], int]]
    ) -> Tuple[Tuple[Sequence[Sequence[NSRT]], Sequence[Sequence[NSRT]], Sequence[Tensor], Sequence[Tensor]], Tensor]:
        """ Convert a batch of datapoints to batched datapoints
        """
        return (
            [dp[0][0] for dp in batch], [dp[0][1] for dp in batch],
            [dp[0][2] for dp in batch], [dp[0][3] for dp in batch]
        ), Tensor([dp[1] for dp in batch])

    def _create_optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)

    def forward(
        self,
        encoder_nsrts_batch: Sequence[Sequence[NSRT]],
        decoder_nsrts_batch: Sequence[Sequence[NSRT]],
        encoder_states_batch: Sequence[Tensor],
        decoder_states_batch: Sequence[Tensor],
    ) -> float:
        assert all(
            len(encoder_states) == len(encoder_nsrts)
            for encoder_states, encoder_nsrts in zip(encoder_states_batch, encoder_nsrts_batch)
        )
        assert len(encoder_nsrts_batch) == len(decoder_nsrts_batch) and\
            len(decoder_nsrts_batch) == len(encoder_states_batch) and\
            len(encoder_states_batch) == len(decoder_states_batch)

        encoder_tokens = self._run_encoders(self._encoder_encoders, encoder_states_batch, encoder_nsrts_batch, False)
        decoder_tokens = self._run_encoders(self._decoder_encoders, decoder_states_batch, decoder_states_batch, True)
        transformer_outputs = self._transformer(
            src=self._encoder_positional_encoding(encoder_tokens),
            tgt=self._decoder_positional_encoding(decoder_tokens),
            src_is_causal=False,
            tgt_is_causal=False,
        )
        return torch.stack([tokens[-1] for tokens in transformer_outputs])

    @classmethod
    def _run_encoders(
        cls,
        encoders: Dict[NSRT, nn.Module],
        states_batch: Iterable[Tensor],
        nsrts_batch: Iterable[Iterable[NSRT]],
        replicate_state: bool,
    ) -> Tensor:
        """ Runs state encoders that are executed before passing the states into the main transformer
        """
        return torch.nested.nested_tensor([Tensor([
                encoders[nsrt](state)
                for state, nsrt in zip(itertools.repeat(states) if replicate_state else states, nsrts)
            ]) for states, nsrts in zip(states_batch, nsrts_batch)
        ])

    def _init_encoders_datapoint(self, datapoint: FeasibilityDatapoint):
        """ Initializes encoders that should be learned from that datapoint
        """
        for prev_state, encoder_nsrt in zip(datapoint.previous_states, datapoint.encoder_nsrts):
            self._init_encoder(prev_state, encoder_nsrt)
        for decoder_nsrt in datapoint.decoder_nsrts:
            self._init_encoder(datapoint.decoder_nsrts, decoder_nsrt)

    def _init_encoder(self, state: State, ground_nsrt: _GroundNSRT):
        """ Initializes an encoder for a single ground nsrt.

        NOTE: The assumption is that the concatentated features of all objects that
        are passed to a given NSRT always have the same data layout and total length.
        """
        nsrt = ground_nsrt.parent
        if nsrt in self._encoder_encoders:
            assert nsrt in self._decoder_encoders
            return

        state_vector_size = state[ground_nsrt.objects].size
        encoder_encoder = self._create_encoder(
            state_vector_size, self._encoder_hidden_sizes, self._encoder_output_size
        )
        decoder_encoder = self._create_encoder(
            state_vector_size, self._encoder_hidden_sizes, self._encoder_output_size
        )
        self.add_module(f"{nsrt.name}_encoder", encoder_encoder)
        self.add_module(f"{nsrt.name}_decoder", decoder_encoder)
        self._encoder_encoders[nsrt] = encoder_encoder
        self._decoder_encoders[nsrt] = decoder_encoder

    @classmethod
    def _create_encoder(cls, state_vector_size: int, hidden_sizes: List[int], feature_size: int) -> nn.Module:
        """ Creates a single state encoder
        """
        input_sizes = [state_vector_size] + hidden_sizes
        return Sequence([
            layer
            for input_size, output_size in zip(input_sizes, hidden_sizes)
            for layer in [nn.Linear(input_size, output_size), nn.ReLU()]
        ] + [nn.Linear(input_sizes[-1], feature_size)])