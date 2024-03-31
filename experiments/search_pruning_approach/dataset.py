from dataclasses import dataclass, field
from functools import cached_property
from itertools import cycle, groupby
import logging
import pickle
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from predicators.structs import NSRT, _GroundNSRT, Object, State
import numpy.typing as npt
import torch
from torch import Tensor

@dataclass(frozen=True)
class FeasibilityInputBatch:
    max_len: int
    batch_size: int
    nsrt_indices: Dict[NSRT, int]

    states: Dict[NSRT, npt.NDArray[np.float32]]
    seq_indices: Dict[NSRT, npt.NDArray[np.int64]]
    batch_indices: Dict[NSRT, npt.NDArray[np.int64]]

    def run_featurizers(
        self,
        featurizers: Dict[NSRT, torch.nn.Module],
        featurizer_output_size: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        tokens = torch.zeros((self.batch_size, self.max_len, featurizer_output_size), device=device)
        mask = np.full((self.batch_size, self.max_len), True) # Not on device for fast assignment
        nsrt_indices = np.zeros((self.batch_size, self.max_len), dtype=np.int64) # Not on device for fast assignment

        for nsrt, idx in self.nsrt_indices.items():
            if len(self.states[nsrt]) == 0:
                continue
            assert self.seq_indices[nsrt].shape == self.batch_indices[nsrt].shape == (self.states[nsrt].shape[0],)
            tokens[self.batch_indices[nsrt], self.seq_indices[nsrt]] = featurizers[nsrt](self.states[nsrt].astype(np.float32))
            mask[self.batch_indices[nsrt], self.seq_indices[nsrt]] = False
            nsrt_indices[self.batch_indices[nsrt], self.seq_indices[nsrt]] = idx

        mask = torch.from_numpy(mask).to(device)
        nsrt_indices = torch.from_numpy(nsrt_indices).to(device)

        return tokens, mask, nsrt_indices

class FeasibilityDataset:
    @dataclass(frozen=True)
    class PicklableDatapoint:
        encoder_inputs: Dict[str, 'FeasibilityDataset.NSRTDatapoint']
        decoder_inputs: Dict[str, 'FeasibilityDataset.NSRTDatapoint']
        total_encoder_length: int
        total_decoder_length: int
        num_objects: int

    @dataclass(frozen=True)
    class NSRTDatapoint:
        states: List[npt.NDArray[np.float32]] = field(default_factory=list)
        objects: List[List[int]] = field(default_factory=list)
        seq_indices: List[int] = field(default_factory=list)

    @dataclass(frozen=True)
    class Datapoint:
        encoder_inputs: Dict[NSRT, 'FeasibilityDataset.NSRTDatapoint']
        decoder_inputs: Dict[NSRT, 'FeasibilityDataset.NSRTDatapoint']
        total_encoder_length: int
        total_decoder_length: int
        num_objects: int

    @dataclass(frozen=True)
    class DatasetNSRTCache:
        state_counts: npt.NDArray[np.int64] # (num_datapoints,)
        datapoint_ranges: npt.NDArray[np.int64] #(num_datapoints + 1,)
        states: npt.NDArray[np.float32] #(sum(state_counts), state_width)
        objects: npt.NDArray[np.int64] #(sum(state_counts), num_objects_per_nsrt)
        seq_indices: npt.NDArray[np.int64] #(sum(state_counts),)

    @dataclass(frozen=True)
    class DatasetSubCache:
        skeleton_lengths: npt.NDArray[np.int64] #(num_datapoints,)
        nsrt_cache: Dict[NSRT, 'FeasibilityDataset.DatasetNSRTCache']

    @dataclass(frozen=True)
    class AugDatasetNSRTCache:
        state_counts: npt.NDArray[np.int64] # (num_datapoints,)
        states: npt.NDArray[np.float32] #(sum(state_counts), state_width)
        objects: npt.NDArray[np.int64] #(sum(state_counts), num_objects_per_nsrt)
        seq_indices: npt.NDArray[np.int64] #(sum(state_counts),)
        datapoint_indices: npt.NDArray[np.int64] #(sum(state_counts),)

    @dataclass(frozen=True)
    class AugDatasetSubCache:
        skeleton_lengths: npt.NDArray[np.int64] #(num_datapoints,)
        nsrt_cache: Dict[NSRT, 'FeasibilityDataset.AugDatasetNSRTCache']

    @dataclass(frozen=True)
    class DatasetCache:
        num_datapoints: int
        num_aug_datapoints:int

        num_main_objects: int
        num_aug_objects: int
        labels: npt.NDArray[np.float32]

        main_encoder_data: 'FeasibilityDataset.DatasetSubCache'
        main_decoder_data: 'FeasibilityDataset.DatasetSubCache'
        aug_encoder_data: 'FeasibilityDataset.AugDatasetSubCache'
        aug_decoder_data: 'FeasibilityDataset.AugDatasetSubCache'

    def __init__(self, nsrts: Iterable[NSRT], batch_size: int, equalize_categories: bool = False):
        super().__init__()
        self._equalize_categories = equalize_categories
        self._batch_size = batch_size
        self._nsrt_indices = {nsrt: idx for idx, nsrt in enumerate(sorted(nsrts))}
        self.empty()

    @property
    def diagnostics(self) -> str:
        num_datapoints = len(self._positive_datapoints) + len(self._negative_datapoints)
        return f"""------------
Positive datapoints: {len(self._positive_datapoints)}/{num_datapoints}
Negative datapoints: {len(self._negative_datapoints)}/{num_datapoints}
Augmentation datapoints: {len(self._augmentation_datapoints)}

Positive datapoints per failing nsrt: {[(nsrt.name, count) for nsrt, count in self._positive_nsrt_statistics.items()]}
Negative datapoints per failing nsrt: {[(nsrt.name, count) for nsrt, count in self._negative_nsrt_statistics.items()]}
------------"""

    def empty(self) -> None:
        self._invalidate_cache()

        self._positive_datapoints: List[FeasibilityDataset.Datapoint] = []
        self._negative_datapoints: List[FeasibilityDataset.Datapoint] = []
        self._augmentation_datapoints: List[FeasibilityDataset.Datapoint] = []

        self._positive_nsrt_statistics: Dict[NSRT, int] = {nsrt: 0 for nsrt in self._nsrt_indices}
        self._negative_nsrt_statistics: Dict[NSRT, int] = {nsrt: 0 for nsrt in self._nsrt_indices}

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
        self._positive_datapoints.append(self._create_main_datapoint(self._nsrt_indices.keys(), skeleton, states))
        self._positive_nsrt_statistics[skeleton[len(states) - 2].parent] += 1

    def add_negative_datapoint(self, skeleton: Sequence[_GroundNSRT], states: Sequence[State]) -> None:
        assert 2 <= len(states) <= len(skeleton)
        self._invalidate_cache()
        self._negative_datapoints.append(self._create_main_datapoint(self._nsrt_indices.keys(), skeleton, states))
        self._negative_nsrt_statistics[skeleton[len(states) - 2].parent] += 1

    def add_augmentation_datapoint(self, skeleton: Sequence[_GroundNSRT], states: Sequence[State]) -> None:
        assert len(states) == len(skeleton) + 1
        # self._invalidate_cache()
        # object_indices = self._create_object_indices(states[0])
        # self._augmentation_datapoints.append(self.Datapoint(
        #     encoder_inputs = self._create_feasibility_datapoint_inputs(
        #         self._nsrt_indices.keys(), skeleton, states[1:], object_indices
        #     ),
        #     decoder_inputs = self._create_feasibility_datapoint_inputs(
        #         self._nsrt_indices.keys(), skeleton, [states[-1]] * len(skeleton), object_indices
        #     ),
        #     total_encoder_length = len(skeleton),
        #     total_decoder_length = len(skeleton),
        #     num_objects = len(object_indices),
        # ))

    def _invalidate_cache(self) -> None:
        if "state_ranges" in self.__dict__:
            del self.state_ranges
        if "_dataset" in self.__dict__:
            del self._dataset

    @cached_property
    def state_ranges(self) -> Dict[NSRT, Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
        dataset = self._dataset
        return {
            nsrt: (
                np.hstack([states.min(0)] + [0]*len(nsrt.parameters)),
                np.hstack([states.max(0)] + [1]*len(nsrt.parameters)),
            ) for nsrt in self._nsrt_indices
            for states in [np.vstack([
                dataset.main_encoder_data.nsrt_cache[nsrt].states,
                dataset.main_decoder_data.nsrt_cache[nsrt].states,
                dataset.aug_encoder_data.nsrt_cache[nsrt].states,
                dataset.aug_decoder_data.nsrt_cache[nsrt].states,
            ])] if states.shape[0]
        }

    @classmethod
    def transform_input(
        cls, nsrt_indices: Dict[NSRT, int], skeleton: Sequence[_GroundNSRT], states: Sequence[State]
    ) -> Tuple[FeasibilityInputBatch, FeasibilityInputBatch]:
        assert 2 <= len(states) <= len(skeleton)
        datapoint = cls._create_main_datapoint(nsrt_indices.keys(), skeleton, states)
        object_indices = np.zeros(datapoint.num_objects, dtype=np.float32)
        # object_indices = np.linspace(0, 1, datapoint.num_objects, endpoint=True)
        return cls._transform_datapoint(
            datapoint.total_encoder_length, nsrt_indices, object_indices, datapoint.encoder_inputs
        ), cls._transform_datapoint(
            datapoint.total_decoder_length, nsrt_indices, object_indices, datapoint.decoder_inputs
        )

    @classmethod
    def _transform_datapoint(
        cls,
        total_length: int,
        nsrt_indices: Dict[NSRT, int],
        object_indices: npt.NDArray[np.float32],
        datapoint_inputs: Dict[NSRT, NSRTDatapoint]
    ) -> FeasibilityInputBatch:
        return FeasibilityInputBatch(
            max_len = total_length, batch_size = 1, nsrt_indices = nsrt_indices,
            states = {
                nsrt: np.hstack([
                    np.vstack(datapoint_inputs[nsrt].states),
                    object_indices[np.array(datapoint_inputs[nsrt].objects, dtype=np.int64)]
                ]) if datapoint_inputs[nsrt].states else np.empty((0, sum(p.type.dim + 1 for p in nsrt.parameters)), dtype=np.float32)
                for nsrt in nsrt_indices
            },
            seq_indices = {
                nsrt: np.hstack([np.empty((0,), dtype=np.int64)] + datapoint_inputs[nsrt].seq_indices)
                for nsrt in nsrt_indices
            },
            batch_indices = {
                nsrt: np.zeros(len(datapoint_inputs[nsrt].states), dtype=np.int64)
                for nsrt in nsrt_indices
            }
        )

    @classmethod
    def _create_main_datapoint(
        cls,
        nsrts: Sequence[NSRT],
        skeleton: Sequence[_GroundNSRT],
        states: Sequence[State]
    ) -> Datapoint:
        object_indices = cls._create_object_indices(states[0])

        return cls.Datapoint(
            encoder_inputs = cls._create_feasibility_datapoint_inputs(
                nsrts, skeleton[:len(states) - 1], states[1:], object_indices
            ),
            decoder_inputs = cls._create_feasibility_datapoint_inputs(
                nsrts, skeleton[len(states) - 1:], [states[-1]] * (len(skeleton) - len(states) + 1), object_indices
            ),
            total_encoder_length = len(states) - 1,
            total_decoder_length = len(skeleton) - len(states) + 1,
            num_objects = len(object_indices),
        )

    @classmethod
    def _create_object_indices(cls, state: State) -> Dict[Object, int]:
        return {obj:idx for idx, obj in enumerate(state)}

    @classmethod
    def _create_feasibility_datapoint_inputs(
        cls,
        nsrts: Iterable[NSRT],
        skeleton: Sequence[_GroundNSRT],
        states: Sequence[State],
        object_indices: Dict[Object, int],
    ) -> Dict[NSRT, NSRTDatapoint]:
        assert len(skeleton) == len(states)
        inputs = {nsrt: cls.NSRTDatapoint() for nsrt in nsrts}
        for idx, state, ground_nsrt in zip(range(len(skeleton)), states, skeleton):
            nsrt_datapoint = inputs[ground_nsrt.parent]
            nsrt_datapoint.states.append(state.vec(ground_nsrt.objects))
            nsrt_datapoint.objects.append([object_indices[obj] for obj in ground_nsrt.objects])
            nsrt_datapoint.seq_indices.append(idx)
        return inputs

    def __iter__(self) -> Iterator[Tuple[Tuple[FeasibilityInputBatch, FeasibilityInputBatch], npt.NDArray]]:
        dataset = self._dataset

        num_selected_aug_datapoints = min(np.random.binomial(dataset.num_datapoints, 0.5), dataset.num_aug_datapoints * 2)
        aug_mask = np.full((dataset.num_aug_datapoints * 2), False)
        aug_mask[np.random.choice(dataset.num_aug_datapoints * 2, (num_selected_aug_datapoints,), replace=False)] = True
        aug_mask = aug_mask.reshape((2, -1))

        aug_encoder_sub_cache = self._choose_aug_datapoints(dataset.num_datapoints, aug_mask[0], dataset.aug_encoder_data)
        aug_decoder_sub_cache = self._choose_aug_datapoints(dataset.num_datapoints, aug_mask[1], dataset.aug_decoder_data)

        main_permutation = np.random.permutation(dataset.num_datapoints)
        aug_permutation = np.random.permutation(dataset.num_datapoints)
        # object_indices = np.random.permutation(dataset.num_objects) / (dataset.num_objects - 1)
        object_indices = np.hstack([np.zeros(dataset.num_main_objects, dtype=np.float32), np.ones(dataset.num_aug_objects, dtype=np.float32)])

        return iter(zip(zip(self._construct_input_batches(
                    num_datapoints = dataset.num_datapoints,
                    batch_size = self._batch_size,
                    nsrt_indices = self._nsrt_indices,
                    object_indices = object_indices,
                    prefix_permutation = aug_permutation,
                    suffix_permutation = main_permutation,
                    prefix_sub_cache = aug_encoder_sub_cache,
                    suffix_sub_cache = dataset.main_encoder_data
                ),
                self._construct_input_batches(
                    num_datapoints = dataset.num_datapoints,
                    batch_size = self._batch_size,
                    nsrt_indices = self._nsrt_indices,
                    object_indices = object_indices,
                    prefix_permutation = main_permutation,
                    suffix_permutation = aug_permutation,
                    prefix_sub_cache = dataset.main_decoder_data,
                    suffix_sub_cache = aug_decoder_sub_cache,
                )
            ),
            self._construct_label_batches(
                num_datapoints = dataset.num_datapoints,
                batch_size = self._batch_size,
                permutation = main_permutation,
                labels = dataset.labels,
            )
        ))

    @classmethod
    def _choose_aug_datapoints(
        cls,
        num_datapoints: int,
        aug_mask: npt.NDArray[np.bool_], #(num_aug_datapoints,)
        sub_cache: AugDatasetSubCache,
    ) -> DatasetSubCache:
        skeleton_lengths = sub_cache.skeleton_lengths[aug_mask]
        return cls.DatasetSubCache(
            skeleton_lengths = np.pad(skeleton_lengths, (0, num_datapoints - skeleton_lengths.size)),
            nsrt_cache = {
                nsrt: cls.DatasetNSRTCache(
                    state_counts = state_counts,
                    datapoint_ranges = np.cumsum(np.hstack([0, state_counts])),
                    states = nsrt_cache.states[aug_states_mask],
                    objects = nsrt_cache.objects[aug_states_mask],
                    seq_indices = nsrt_cache.seq_indices[aug_states_mask],
                ) for nsrt, nsrt_cache in sub_cache.nsrt_cache.items()
                for aug_states_mask, state_counts in [(
                    aug_mask[nsrt_cache.datapoint_indices],
                    np.pad(nsrt_cache.state_counts[aug_mask], (0, num_datapoints - skeleton_lengths.size))
                )]
            }
        )

    @classmethod
    def _construct_label_batches(
        cls,
        num_datapoints: int,
        batch_size: int,
        permutation: npt.NDArray[np.int64], #(num_datapoints,)
        labels: npt.NDArray[np.float32], #(num_datapoints,)
    ) -> Iterator[npt.NDArray[np.float32]]:
        permuted_labels = labels[permutation]
        for batch_idx in range(0, num_datapoints, batch_size):
            yield permuted_labels[batch_idx:batch_idx + batch_size]

    @classmethod
    def _construct_input_batches(
        cls,
        num_datapoints: int,
        batch_size: int,
        nsrt_indices: Dict[NSRT, int],
        object_indices: npt.NDArray[np.float32], #(num_objects,)

        prefix_permutation: npt.NDArray[np.int64], # (num_datapoints,)
        suffix_permutation: npt.NDArray[np.int64], # (num_datapoints,)

        prefix_sub_cache: DatasetSubCache,
        suffix_sub_cache: DatasetSubCache
    ) -> Iterator[FeasibilityInputBatch]:
        final_lengths = prefix_sub_cache.skeleton_lengths[prefix_permutation] + \
            suffix_sub_cache.skeleton_lengths[suffix_permutation]
        batches_indices = [
            (FeasibilityInputBatch(
                max_len = batch_final_lengths.max(initial=0),
                batch_size = batch_final_lengths.shape[0],
                nsrt_indices = nsrt_indices,
                states = {},
                seq_indices = {},
                batch_indices = {}
            ), batch_idx, batch_idx + batch_final_lengths.shape[0]) for batch_idx in range(0, num_datapoints, batch_size)
            for batch_final_lengths in [final_lengths[batch_idx:batch_idx + batch_size]]
        ]
        for nsrt in nsrt_indices:
            prefix_full_states = cls._construct_full_state(
                object_indices, prefix_sub_cache.nsrt_cache[nsrt].states,
                prefix_sub_cache.nsrt_cache[nsrt].objects,
            )
            suffix_full_states = cls._construct_full_state(
                object_indices, suffix_sub_cache.nsrt_cache[nsrt].states,
                suffix_sub_cache.nsrt_cache[nsrt].objects,
            )
            datapoint_ranges, states, seq_indices, batch_indices = cls._construct_data_single_nsrt(
                num_datapoints = num_datapoints,
                batch_size = batch_size,
                prefix_permutation = prefix_permutation,
                suffix_permutation = suffix_permutation,

                prefix_skeleton_lengths = prefix_sub_cache.skeleton_lengths,
                prefix_state_counts = prefix_sub_cache.nsrt_cache[nsrt].state_counts,
                prefix_datapoint_ranges = prefix_sub_cache.nsrt_cache[nsrt].datapoint_ranges,
                prefix_states = prefix_full_states,
                prefix_seq_indices = prefix_sub_cache.nsrt_cache[nsrt].seq_indices,

                suffix_state_counts = suffix_sub_cache.nsrt_cache[nsrt].state_counts,
                suffix_datapoint_ranges = suffix_sub_cache.nsrt_cache[nsrt].datapoint_ranges,
                suffix_states = suffix_full_states,
                suffix_seq_indices = suffix_sub_cache.nsrt_cache[nsrt].seq_indices,
            )

            for input_batch, start_batch_idx, end_batch_idx in batches_indices:
                batch_slice = slice(datapoint_ranges[start_batch_idx], datapoint_ranges[end_batch_idx])
                input_batch.states[nsrt] = states[batch_slice]
                input_batch.seq_indices[nsrt] = seq_indices[batch_slice]
                input_batch.batch_indices[nsrt] = batch_indices[batch_slice]

        for input_batch, _1, _2 in batches_indices:
            yield input_batch

    @classmethod
    def _construct_full_state(
        cls,
        object_indices: npt.NDArray[np.float32], #(num_objects,)
        states: npt.NDArray[np.float32], #(n, state_width)
        objects: npt.NDArray[np.int64], #(n, num_objects)
    ) -> npt.NDArray:
        assert len(states.shape) == len(objects.shape) == 2
        assert states.shape[0] == objects.shape[0]
        return np.hstack([states, object_indices[objects]])

    @classmethod
    def _construct_data_single_nsrt(
        cls,
        num_datapoints: int,
        batch_size: int,
        prefix_permutation: npt.NDArray[np.int64], # (num_datapoints,)
        suffix_permutation: npt.NDArray[np.int64], # (num_datapoints,)

        prefix_skeleton_lengths: npt.NDArray[np.int64], #(num_datapoints,)
        prefix_state_counts: npt.NDArray[np.int64], # (num_datapoints,)
        prefix_datapoint_ranges: npt.NDArray[np.int64], # (num_datapoints + 1,)
        prefix_states: npt.NDArray[np.float32], # (sum(prefix_state_counts), state_width)
        prefix_seq_indices: npt.NDArray[np.int64], # (sum(prefix_state_counts),)

        suffix_state_counts: npt.NDArray[np.int64], # (num_datapoints,)
        suffix_datapoint_ranges: npt.NDArray[np.int64], # (num_datapoints + 1,)
        suffix_states: npt.NDArray[np.float32], # (sum(suffix_state_counts), state_width)
        suffix_seq_indices: npt.NDArray[np.int64], # (sum(suffix_state_counts),)
    ) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        assert prefix_permutation.shape == suffix_permutation.shape == prefix_skeleton_lengths.shape == prefix_state_counts.shape == suffix_state_counts.shape == (num_datapoints,)
        assert prefix_datapoint_ranges.shape == suffix_datapoint_ranges.shape == (num_datapoints + 1,)
        assert prefix_states.shape[0] == prefix_seq_indices.shape[0] == prefix_datapoint_ranges[-1]
        assert suffix_states.shape[0] == suffix_seq_indices.shape[0] == suffix_datapoint_ranges[-1]

        # Combined permutation for converting from concatenated prefix and suffix states
        permutation = np.hstack([prefix_permutation[:, None], (suffix_permutation + num_datapoints)[:, None]]).reshape(num_datapoints * 2)

        # Concatenated data for the permutation to convert from
        state_counts = np.hstack([prefix_state_counts, suffix_state_counts])
        states = np.vstack([prefix_states, suffix_states])
        seq_indices = np.hstack([prefix_seq_indices, suffix_seq_indices])
        datapoint_ranges = np.hstack([prefix_datapoint_ranges, suffix_datapoint_ranges[1:] + prefix_datapoint_ranges[-1]])

        # Permutation indicies calculation
        ## Ranges of datapoints in permuted states
        permuted_datapoint_ranges = np.cumsum(np.hstack([0, state_counts[permutation]]))
        assert state_counts.shape == (num_datapoints * 2,)

        ## Indicies of datapoints scattered across permuted states
        permuted_datapoint_indices = np.zeros(permuted_datapoint_ranges[-1] + 1, dtype=np.int64)
        np.add.at(permuted_datapoint_indices, permuted_datapoint_ranges[1:], 1)
        permuted_datapoint_indices = np.cumsum(permuted_datapoint_indices)[:-1]

        ## What indicies in original states should the new states come from
        permuted_states_original_indices = np.ones(permuted_datapoint_ranges[-1] + 1, dtype=np.int64)
        np.add.at(permuted_states_original_indices, permuted_datapoint_ranges[:-1], datapoint_ranges[permutation])
        np.subtract.at(permuted_states_original_indices, permuted_datapoint_ranges[1:], datapoint_ranges[permutation + 1])
        permuted_states_original_indices = np.cumsum(permuted_states_original_indices)[:-1] - 1

        # Permuted data calculation
        permuted_states = states[permuted_states_original_indices]
        seq_offsets = np.hstack([
            np.zeros((num_datapoints, 1), dtype=np.int64), prefix_skeleton_lengths[prefix_permutation, None]
        ]).reshape(num_datapoints * 2)
        permuted_seq_indices = seq_indices[permuted_states_original_indices] + seq_offsets[permuted_datapoint_indices]
        batch_indices = (permuted_datapoint_indices // 2) % batch_size

        return permuted_datapoint_ranges[::2], permuted_states, permuted_seq_indices, batch_indices

    @cached_property
    def _dataset(self) -> DatasetCache:
        positive_datapoints = self._positive_datapoints
        negative_datapoints = self._negative_datapoints
        if self._equalize_categories:
            positive_datapoints = [datapoint for datapoint, _ in zip(cycle(positive_datapoints), negative_datapoints)]
            negative_datapoints = [datapoint for _, datapoint in zip(positive_datapoints, cycle(negative_datapoints))]
        all_main_datapoints = positive_datapoints + negative_datapoints

        object_offsets = np.cumsum([0] + [
            datapoint.num_objects for datapoint in all_main_datapoints + self._augmentation_datapoints
        ])

        return self.DatasetCache(
            num_datapoints = len(positive_datapoints) + len(negative_datapoints),
            num_aug_datapoints = len(self._augmentation_datapoints),
            num_main_objects = object_offsets[len(all_main_datapoints)],
            num_aug_objects = object_offsets[-1] - object_offsets[len(all_main_datapoints)],
            labels = np.hstack([np.ones(len(positive_datapoints), dtype=np.float32), np.zeros(len(negative_datapoints), dtype=np.float32)]),
            main_encoder_data = self._construct_dataset_sub_cache(
                self._nsrt_indices.keys(), [dp.total_encoder_length for dp in all_main_datapoints],
                object_offsets[:len(all_main_datapoints)], [dp.encoder_inputs for dp in all_main_datapoints]
            ),
            main_decoder_data = self._construct_dataset_sub_cache(
                self._nsrt_indices.keys(), [dp.total_decoder_length for dp in all_main_datapoints],
                object_offsets[:len(all_main_datapoints)], [dp.decoder_inputs for dp in all_main_datapoints]
            ),
            aug_encoder_data = self._construct_aug_dataset_sub_cache(
                self._nsrt_indices.keys(), [dp.total_encoder_length for dp in self._augmentation_datapoints],
                object_offsets[len(all_main_datapoints):-1], [dp.encoder_inputs for dp in self._augmentation_datapoints]
            ),
            aug_decoder_data = self._construct_aug_dataset_sub_cache(
                self._nsrt_indices.keys(), [dp.total_decoder_length for dp in self._augmentation_datapoints],
                object_offsets[len(all_main_datapoints):-1], [dp.decoder_inputs for dp in self._augmentation_datapoints]
            ),
        )

    @classmethod
    def _construct_dataset_sub_cache(
        cls,
        nsrts: Sequence[NSRT],
        total_lengths: List[int],
        object_offsets: Iterable[int],
        datapoint_inputs: Sequence[Dict[NSRT, NSRTDatapoint]],
    ) -> DatasetSubCache:
        return cls.DatasetSubCache(
            skeleton_lengths = np.hstack([np.empty((0,), dtype=np.int64)] + total_lengths),
            nsrt_cache = {
                nsrt: cls.DatasetNSRTCache(
                    state_counts = state_counts,
                    datapoint_ranges = np.cumsum(np.hstack([0, state_counts])),
                    states = np.vstack([np.empty((0, sum(v.type.dim for v in nsrt.parameters)), dtype=np.float32)] + [
                        state for datapoint_input in datapoint_inputs for state in datapoint_input[nsrt].states
                    ]),
                    objects = np.vstack([np.empty((0, num_objects), dtype=np.int64)] + [
                        np.array(datapoint_input[nsrt].objects, dtype=np.int64).reshape(-1, num_objects) + object_offset
                        for object_offset, datapoint_input in zip(object_offsets, datapoint_inputs)
                    ]),
                    seq_indices = np.hstack([np.empty((0,), dtype=np.int64)] + [
                        seq_index for datapoint_input in datapoint_inputs
                        for seq_index in datapoint_input[nsrt].seq_indices
                    ])
                ) for nsrt in nsrts
                for num_objects, state_counts in [(len(nsrt.parameters), np.hstack([np.empty((0,), dtype=np.int64)] + [
                    len(datapoint_input[nsrt].states) for datapoint_input in datapoint_inputs
                ]))]
            }
        )

    @classmethod
    def _construct_aug_dataset_sub_cache(
        cls,
        nsrts: Sequence[NSRT],
        total_lengths: List[int],
        object_offsets: Iterable[int],
        datapoint_inputs: Sequence[Dict[NSRT, NSRTDatapoint]],
    ) -> AugDatasetSubCache:
        return cls.AugDatasetSubCache(
            skeleton_lengths = np.hstack([np.empty((0,), dtype=np.int64)] + total_lengths),
            nsrt_cache = {
                nsrt: cls.AugDatasetNSRTCache(
                    state_counts = np.hstack([np.empty((0,), dtype=np.int64)] + [
                        len(datapoint_input[nsrt].states) for datapoint_input in datapoint_inputs
                    ]),
                    states = np.vstack([np.empty((0, sum(v.type.dim for v in nsrt.parameters)), dtype=np.float32)] + [
                        state for datapoint_input in datapoint_inputs for state in datapoint_input[nsrt].states
                    ]),
                    objects = np.vstack([np.empty((0, num_objects), dtype=np.int64)] + [
                        np.array(datapoint_input[nsrt].objects, dtype=np.int64).reshape(-1, num_objects) + object_offset
                        for object_offset, datapoint_input in zip(object_offsets, datapoint_inputs)
                    ]),
                    seq_indices = np.hstack([np.empty((0,), dtype=np.int64)] + [
                        seq_index for datapoint_input in datapoint_inputs
                        for seq_index in datapoint_input[nsrt].seq_indices
                    ]),
                    datapoint_indices = np.hstack([np.empty((0,), dtype=np.int64)] + [
                        np.full_like(datapoint_input[nsrt].seq_indices, idx, dtype=np.int64)
                        for idx, datapoint_input in enumerate(datapoint_inputs)
                    ]),
                ) for nsrt in nsrts
                for num_objects in [len(nsrt.parameters)]
            }
        )

    def dumps(self) -> bytes:
        return pickle.dumps((
            list(map(self._pickle_datapoint, self._positive_datapoints)),
            list(map(self._pickle_datapoint, self._negative_datapoints)),
            list(map(self._pickle_datapoint, self._augmentation_datapoints)),
        ))

    @classmethod
    def _pickle_datapoint(cls, datapoint: Datapoint) -> PicklableDatapoint:
        return cls.PicklableDatapoint({
            str(nsrt): encoder_input for nsrt, encoder_input in datapoint.encoder_inputs.items()
        }, {
            str(nsrt): decoder_input for nsrt, decoder_input in datapoint.decoder_inputs.items()
        }, datapoint.total_encoder_length, datapoint.total_decoder_length, datapoint.num_objects)

    def loads(self, data: bytes, nsrts: Iterable[NSRT]) -> None:
        nsrt_map = {str(nsrt): nsrt for nsrt in nsrts}
        unpickle_datapoint = lambda dp: self._unpickle_datapoint(dp, nsrt_map)

        pickled_positive_datapoints, pickled_negative_datapoints, pickled_augmentation_datapoints = pickle.loads(data)
        self._positive_datapoints = list(map(unpickle_datapoint, pickled_positive_datapoints))
        self._negative_datapoints = list(map(unpickle_datapoint, pickled_negative_datapoints))
        self._augmentation_datapoints = list(map(unpickle_datapoint, pickled_augmentation_datapoints))

    @classmethod
    def _unpickle_datapoint(cls, datapoint: PicklableDatapoint, nsrt_map: Dict[str, NSRT]) -> Datapoint:
        return cls.Datapoint({
            nsrt_map[nsrt_str]: encoder_input for nsrt_str, encoder_input in datapoint.encoder_inputs.items()
        }, {
            nsrt_map[nsrt_str]: decoder_input for nsrt_str, decoder_input in datapoint.decoder_inputs.items()
        }, datapoint.total_encoder_length, datapoint.total_decoder_length, datapoint.num_objects)