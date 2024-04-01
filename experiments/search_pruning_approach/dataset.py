from dataclasses import dataclass, field
from functools import cached_property
from itertools import cycle, repeat, groupby, chain
from collections import defaultdict
import logging
import pickle
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import numpy as np
from predicators.structs import NSRT, _GroundNSRT, Object, State
import numpy.typing as npt
import torch
from torch import Tensor

SkeletonNSRT = Tuple[NSRT, bool] # (NSRT, was_ran)

@dataclass(frozen=True)
class FeasibilityInputBatch:
    max_len: int
    batch_size: int

    states: Dict[SkeletonNSRT, npt.NDArray[np.float32]]
    seq_indices: Dict[SkeletonNSRT, npt.NDArray[np.int64]]
    batch_indices: Dict[SkeletonNSRT, npt.NDArray[np.int64]]

    def run_featurizers(
        self,
        featurizers: Dict[SkeletonNSRT, torch.nn.Module],
        skeleton_nsrt_indices_mapping: Dict[SkeletonNSRT, int],
        featurizer_output_size: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        tokens = torch.zeros((self.batch_size, self.max_len, featurizer_output_size), device=device)
        mask = np.full((self.batch_size, self.max_len), True) # Not on device for fast assignment
        skeleton_nsrt_indices = np.zeros((self.batch_size, self.max_len), dtype=np.int64) # Not on device for fast assignment
        ran_mask = np.full((self.batch_size, self.max_len), False) # Not on device for fast assignment

        for skeleton_nsrt, featurizer in featurizers.items():
            if len(self.states[skeleton_nsrt]) == 0:
                continue
            assert self.seq_indices[skeleton_nsrt].shape == self.batch_indices[skeleton_nsrt].shape == (self.states[skeleton_nsrt].shape[0],)
            tokens[self.batch_indices[skeleton_nsrt], self.seq_indices[skeleton_nsrt]] = featurizer(self.states[skeleton_nsrt].astype(np.float32))
            mask[self.batch_indices[skeleton_nsrt], self.seq_indices[skeleton_nsrt]] = False
            skeleton_nsrt_indices[self.batch_indices[skeleton_nsrt], self.seq_indices[skeleton_nsrt]] = skeleton_nsrt_indices_mapping[skeleton_nsrt]

            _, ran_flag = skeleton_nsrt
            if ran_flag:
                ran_mask[self.batch_indices[skeleton_nsrt], self.seq_indices[skeleton_nsrt]] = True

        last_ran_mask = np.full((self.batch_size, self.max_len), False)
        last_ran_mask[np.arange(self.batch_size), self.max_len - 1 - np.argmax(ran_mask[:, ::-1], axis=1)] = True

        mask = torch.from_numpy(mask).to(device)
        skeleton_nsrt_indices = torch.from_numpy(skeleton_nsrt_indices).to(device)
        last_ran_mask = torch.from_numpy(last_ran_mask).to(device)

        return tokens, mask, skeleton_nsrt_indices, last_ran_mask

class FeasibilityDataset:
    @dataclass(frozen=True)
    class SkeletonNSRTDatapoint:
        states: List[npt.NDArray[np.float32]] = field(default_factory=list)
        objects: List[List[int]] = field(default_factory=list)
        seq_indices: List[int] = field(default_factory=list)

    @dataclass(frozen=True)
    class PicklableDatapoint:
        total_length: int
        num_objects: int
        inputs: Dict[str, 'FeasibilityDataset.SkeletonNSRTDatapoint']

    @dataclass(frozen=True)
    class Datapoint:
        total_length: int
        num_objects: int
        inputs: Dict[SkeletonNSRT, 'FeasibilityDataset.SkeletonNSRTDatapoint']

    @dataclass(frozen=True)
    class DatasetSkeletonNSRTCache:
        state_counts: npt.NDArray[np.int64] # (num_datapoints,)
        datapoint_ranges: npt.NDArray[np.int64] #(num_datapoints + 1,)
        states: npt.NDArray[np.float32] #(sum(state_counts), state_width)
        objects: npt.NDArray[np.int64] #(sum(state_counts), num_objects_per_nsrt)
        seq_indices: npt.NDArray[np.int64] #(sum(state_counts),)

    @dataclass(frozen=True)
    class DatasetCache:
        num_datapoints: int
        num_objects: int

        skeleton_lengths: npt.NDArray[np.int64] #(num_datapoints,)
        labels: npt.NDArray[np.float32] #(num_datapoints,)

        skeleton_nsrt_cache: Dict[SkeletonNSRT, 'FeasibilityDataset.DatasetSkeletonNSRTCache']

    def __init__(self, nsrts: Iterable[NSRT], batch_size: int, equalize_categories: bool = False):
        super().__init__()
        self._equalize_categories = equalize_categories
        self._batch_size = batch_size
        self._skeleton_nsrts = [(nsrt, ran_flag) for nsrt in nsrts for ran_flag in [True, False]]

        self._positive_datapoints: List[FeasibilityDataset.Datapoint] = []
        self._negative_datapoints: List[FeasibilityDataset.Datapoint] = []

        self._positive_nsrt_statistics: Dict[NSRT, int] = {nsrt: 0 for nsrt in nsrts}
        self._negative_nsrt_statistics: Dict[NSRT, int] = {nsrt: 0 for nsrt in nsrts}

    @property
    def diagnostics(self) -> str:
        num_datapoints = len(self._positive_datapoints) + len(self._negative_datapoints)
        return f"""------------
Positive datapoints: {len(self._positive_datapoints)}/{num_datapoints}
Negative datapoints: {len(self._negative_datapoints)}/{num_datapoints}

Positive datapoints per failing nsrt: {[(nsrt.name, count) for nsrt, count in self._positive_nsrt_statistics.items()]}
Negative datapoints per failing nsrt: {[(nsrt.name, count) for nsrt, count in self._negative_nsrt_statistics.items()]}
------------"""

    @property
    def num_positive_datapoints(self) -> int:
        return len(self._positive_datapoints)

    @property
    def num_negative_datapoints(self) -> int:
        return len(self._negative_datapoints)

    def __len__(self) -> int:
        if self._equalize_categories:
            return max(len(self._positive_datapoints), len(self._negative_datapoints)) * 2
        return len(self._positive_datapoints) + len(self._negative_datapoints)

    def add_positive_datapoint(self, skeleton: Sequence[_GroundNSRT], states: Sequence[State]) -> None:
        assert 2 <= len(states) <= len(skeleton)
        self._invalidate_cache()
        self._positive_datapoints.append(self._create_datapoint(skeleton, states))
        self._positive_nsrt_statistics[skeleton[len(states) - 2].parent] += 1

    def add_negative_datapoint(self, skeleton: Sequence[_GroundNSRT], states: Sequence[State]) -> None:
        assert 2 <= len(states) <= len(skeleton)
        self._invalidate_cache()
        self._negative_datapoints.append(self._create_datapoint(skeleton, states))
        self._negative_nsrt_statistics[skeleton[len(states) - 2].parent] += 1

    def _invalidate_cache(self) -> None:
        if "state_ranges" in self.__dict__:
            del self.state_ranges
        if "_dataset" in self.__dict__:
            del self._dataset

    @cached_property
    def state_ranges(self) -> Dict[SkeletonNSRT, Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
        dataset = self._dataset
        return {
            skeleton_nsrt: (
                np.hstack([dataset.skeleton_nsrt_cache[skeleton_nsrt].states.min(0)] + [0]*len(nsrt.parameters)),
                np.hstack([dataset.skeleton_nsrt_cache[skeleton_nsrt].states.max(0)] + [1]*len(nsrt.parameters)),
            ) for skeleton_nsrt in self._skeleton_nsrts
            for nsrt, _ in [skeleton_nsrt]
            if dataset.skeleton_nsrt_cache[skeleton_nsrt].states.shape[0]
        }

    @classmethod
    def transform_input(
        cls, skeleton_nsrts: Iterable[SkeletonNSRT], skeleton: Sequence[_GroundNSRT], states: Sequence[State]
    ) -> Tuple[FeasibilityInputBatch, FeasibilityInputBatch]:
        assert 2 <= len(states) <= len(skeleton)
        datapoint = cls._create_datapoint(skeleton, states)
        object_indices = cls._create_random_object_indices(datapoint.num_objects)
        return FeasibilityInputBatch(
            max_len = datapoint.total_length, batch_size = 1,
            states = {
                skeleton_nsrt: np.hstack([
                    np.vstack(datapoint.inputs[skeleton_nsrt].states),
                    object_indices[np.array(datapoint.inputs[skeleton_nsrt].objects, dtype=np.int64)]
                ]) if datapoint.inputs[skeleton_nsrt].states else np.empty((0, sum(p.type.dim + 1 for p in nsrt.parameters)), dtype=np.float32)
                for skeleton_nsrt in skeleton_nsrts
                for nsrt, _ in [skeleton_nsrt]
            },
            seq_indices = {
                skeleton_nsrt: np.hstack([np.empty((0,), dtype=np.int64)] + datapoint.inputs[skeleton_nsrt].seq_indices)
                for skeleton_nsrt in skeleton_nsrts
            },
            batch_indices = {
                skeleton_nsrt: np.zeros(len(datapoint.inputs[skeleton_nsrt].states), dtype=np.int64)
                for skeleton_nsrt in skeleton_nsrts
            }
        )

    @classmethod
    def _create_datapoint(
        cls,
        skeleton: Sequence[_GroundNSRT],
        states: Sequence[State]
    ) -> Datapoint:
        object_indices = {obj:idx for idx, obj in enumerate(states[0])}

        inputs = defaultdict(lambda: cls.SkeletonNSRTDatapoint())
        for idx, state, ran_flag, ground_nsrt in zip(
            range(len(skeleton)), chain(states[1:], repeat(states[-1])),
            chain([True] * (len(states) - 1), repeat(False)), skeleton
        ):
            nsrt_datapoint = inputs[(ground_nsrt.parent, ran_flag)]
            nsrt_datapoint.states.append(state.vec(ground_nsrt.objects))
            nsrt_datapoint.objects.append([object_indices[obj] for obj in ground_nsrt.objects])
            nsrt_datapoint.seq_indices.append(idx)

        return cls.Datapoint(
            total_length = len(skeleton),
            num_objects = len(object_indices),
            inputs = inputs,
        )

    def __iter__(self) -> Iterator[Tuple[FeasibilityInputBatch, npt.NDArray]]:
        dataset = self._dataset

        datapoint_permutation = np.random.permutation(dataset.num_datapoints)
        object_indices = self._create_random_object_indices(dataset.num_objects)

        premuted_skeleton_lengths = dataset.skeleton_lengths[datapoint_permutation]
        batches_indices = [
            (FeasibilityInputBatch(
                max_len = skeleton_lengths_batch.max(initial=0),
                batch_size = skeleton_lengths_batch.shape[0],
                states = {},
                seq_indices = {},
                batch_indices = {}
            ), batch_idx, batch_idx + skeleton_lengths_batch.shape[0])
            for batch_idx in range(0, dataset.num_datapoints, self._batch_size)
            for skeleton_lengths_batch in [premuted_skeleton_lengths[batch_idx:batch_idx + self._batch_size]]
        ]

        for skeleton_nsrt in self._skeleton_nsrts:
            skeleton_nsrt_cache = dataset.skeleton_nsrt_cache[skeleton_nsrt]

            # Calculating various index information about the permuted states
            ## Ranges of datapoints in permuted states
            permuted_datapoint_ranges = np.cumsum(np.hstack([0, skeleton_nsrt_cache.state_counts[datapoint_permutation]]))

            ## What indicies in original states should the new states come from
            permuted_states_original_indices = np.ones(permuted_datapoint_ranges[-1] + 1, dtype=np.int64)
            np.add.at(permuted_states_original_indices, permuted_datapoint_ranges[:-1], skeleton_nsrt_cache.datapoint_ranges[datapoint_permutation])
            np.subtract.at(permuted_states_original_indices, permuted_datapoint_ranges[1:], skeleton_nsrt_cache.datapoint_ranges[datapoint_permutation + 1])
            permuted_states_original_indices = np.cumsum(permuted_states_original_indices)[:-1] - 1

            ## Indicies of datapoints scattered across permuted states
            permuted_datapoint_indices = np.zeros(permuted_datapoint_ranges[-1] + 1, dtype=np.int64)
            np.add.at(permuted_datapoint_indices, permuted_datapoint_ranges[1:], 1)
            permuted_datapoint_indices = np.cumsum(permuted_datapoint_indices)[:-1]

            # Calculating permuted states and indices
            permuted_states = np.hstack([
                skeleton_nsrt_cache.states, object_indices[skeleton_nsrt_cache.objects]
            ])[permuted_states_original_indices]
            permuted_seq_indices = skeleton_nsrt_cache.seq_indices[permuted_states_original_indices]
            batch_indices = permuted_datapoint_indices % self._batch_size

            for input_batch, batch_start_idx, batch_end_idx in batches_indices:
                batch_slice = slice(permuted_datapoint_ranges[batch_start_idx], permuted_datapoint_ranges[batch_end_idx])
                input_batch.states[skeleton_nsrt] = permuted_states[batch_slice]
                input_batch.seq_indices[skeleton_nsrt] = permuted_seq_indices[batch_slice]
                input_batch.batch_indices[skeleton_nsrt] = batch_indices[batch_slice]

        permuted_labels = dataset.labels[datapoint_permutation]
        for input_batch, batch_start_idx, batch_end_idx in batches_indices:
            yield (input_batch, permuted_labels[batch_start_idx:batch_end_idx])

    @classmethod
    def _create_random_object_indices(cls, num_objects: int) -> npt.NDArray[np.float32]:
        # return (np.random.permutation(num_objects) / (num_objects - 1)).astype(np.float32)
        return np.zeros(num_objects, dtype=np.float32)

    @cached_property
    def _dataset(self) -> DatasetCache:
        positive_datapoints = self._positive_datapoints
        negative_datapoints = self._negative_datapoints
        if self._equalize_categories:
            positive_datapoints = [datapoint for datapoint, _ in zip(cycle(positive_datapoints), negative_datapoints)]
            negative_datapoints = [datapoint for _, datapoint in zip(positive_datapoints, cycle(negative_datapoints))]
        all_main_datapoints = positive_datapoints + negative_datapoints

        object_offsets = np.cumsum([0] + [datapoint.num_objects for datapoint in all_main_datapoints])

        return self.DatasetCache(
            num_datapoints = len(all_main_datapoints),
            num_objects = object_offsets[-1],

            skeleton_lengths = np.hstack([np.empty((0,), dtype=np.int64)] + [datapoint.total_length for datapoint in all_main_datapoints]),
            labels = np.hstack([np.ones(len(positive_datapoints), dtype=np.float32), np.zeros(len(negative_datapoints), dtype=np.float32)]),

            skeleton_nsrt_cache = {
                skeleton_nsrt: self.DatasetSkeletonNSRTCache(
                    state_counts = state_counts,
                    datapoint_ranges = np.cumsum(np.hstack([0, state_counts])),
                    states = np.vstack([np.empty((0, sum(v.type.dim for v in nsrt.parameters)), dtype=np.float32)] + [
                        state for datapoint in all_main_datapoints for state in datapoint.inputs[skeleton_nsrt].states
                    ]),
                    objects = np.vstack([np.empty((0, num_objects), dtype=np.int64)] + [
                        np.array(datapoint.inputs[skeleton_nsrt].objects, dtype=np.int64).reshape(-1, num_objects) + object_offset
                        for object_offset, datapoint in zip(object_offsets, all_main_datapoints)
                    ]),
                    seq_indices = np.hstack([np.empty((0,), dtype=np.int64)] + [
                        seq_index for datapoint in all_main_datapoints for seq_index in datapoint.inputs[skeleton_nsrt].seq_indices
                    ])
                )
                for skeleton_nsrt in self._skeleton_nsrts
                for (nsrt, _), state_counts in [(
                    skeleton_nsrt,
                    np.hstack([np.empty((0,), dtype=np.int64)] + [
                        len(datapoint.inputs[skeleton_nsrt].states) for datapoint in all_main_datapoints
                    ]),
                )]
                for num_objects in [len(nsrt.parameters)]
            }
        )

    def dumps(self) -> bytes:
        return pickle.dumps((
            list(map(self._pickle_datapoint, self._positive_datapoints)),
            list(map(self._pickle_datapoint, self._negative_datapoints)),
        ))

    @classmethod
    def _pickle_datapoint(cls, datapoint: Datapoint) -> PicklableDatapoint:
        return cls.PicklableDatapoint(datapoint.total_length, datapoint.num_objects, {
            (str(nsrt), ran_flag): skeleton_nsrt_input for (nsrt, ran_flag), skeleton_nsrt_input in datapoint.inputs.items()
        })

    def loads(self, data: bytes, nsrts: Iterable[NSRT]) -> None:
        nsrt_map = {str(nsrt): nsrt for nsrt in nsrts}
        unpickle_datapoint = lambda dp: self._unpickle_datapoint(dp, nsrt_map)

        pickled_positive_datapoints, pickled_negative_datapoints = pickle.loads(data)
        self._positive_datapoints = list(map(unpickle_datapoint, pickled_positive_datapoints))
        self._negative_datapoints = list(map(unpickle_datapoint, pickled_negative_datapoints))

    @classmethod
    def _unpickle_datapoint(cls, datapoint: PicklableDatapoint, nsrt_map: Dict[str, NSRT]) -> Datapoint:
        return cls.Datapoint(datapoint.total_length, datapoint.num_objects, {
            (nsrt_map[nsrt_str], ran_flag): skeleton_nsrt_input
            for (nsrt_str, ran_flag), skeleton_nsrt_input in datapoint.inputs.items()
        })