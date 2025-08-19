"""Class implementing PyTorch loaders."""

from typing import Callable, Optional

import torch
from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.types import FeatureIdentifier
from torch.utils.data import DataLoader

# from torch.utils.data import Dataset as TorchDataset

# TODO: maybe create a class with transform/inverse transform with these two functions ? (see the FNO notebook)


# -----------------------------------------------------------------------------------------------------
class PlaidRawDataLoader(DataLoader):
    """PlaidSampleDataLoader."""

    def __init__(
        self,
        dataset: Dataset,
        **kwargs,
    ):
        super().__init__(
            [sample for sample in dataset],
            collate_fn=lambda batch: batch,
            **kwargs,
        )


# -----------------------------------------------------------------------------------------------------


class PlaidDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        collate_fn: Callable,
        **kwargs,
    ):
        super().__init__(
            [sample for sample in dataset],
            collate_fn=collate_fn,
            **kwargs,
        )


# class PlaidDataLoader(DataLoader):
#     """PlaidDataLoader."""

#     def __init__(
#         self,
#         dataset: Dataset,
#         collate_fn: Callable,
#         in_feature_identifiers: list[FeatureIdentifier],
#         out_feature_identifiers: Optional[list[FeatureIdentifier]] = None,
#         **kwargs,
#     ):
#         super().__init__(
#             dataset,
#             collate_fn=collate_fn(
#                 dataset, in_feature_identifiers, out_feature_identifiers
#             ),
#             **kwargs,
#         )


class BaseCollater:
    """Collater."""

    def __init__(
        self,
        in_feature_identifiers: list[FeatureIdentifier],
        out_feature_identifiers: Optional[list[FeatureIdentifier]] = None,
    ):
        self.in_feature_identifiers = in_feature_identifiers
        self.out_feature_identifiers = out_feature_identifiers or []

    @staticmethod
    def _can_stack_features(batch_features: list):
        """Check if all elements have the same shape for stacking."""
        shapes = [torch.as_tensor(f).shape for f in batch_features]
        return len(set(shapes)) == 1

    def _batch_features(self, batch: list[Sample]):
        """_batch_features."""
        batch_in_features = [[] for _ in range(len(self.in_feature_identifiers))]
        batch_out_features = [[] for _ in range(len(self.out_feature_identifiers))]

        # Group samples by features
        for sample in batch:
            for j, feat_id in enumerate(self.in_feature_identifiers):
                batch_in_features[j].append(sample.get_feature_from_identifier(feat_id))
            for j, feat_id in enumerate(self.out_feature_identifiers):
                batch_out_features[j].append(
                    sample.get_feature_from_identifier(feat_id)
                )

        return batch_in_features, batch_out_features


# -----------------------------------------------------------------------------------------------------


class HeterogeneousCollater(BaseCollater):
    """Collater."""

    def __call__(self, batch: list[Sample]):
        return self._batch_features(batch)


# -----------------------------------------------------------------------------------------------------


class HomogeneousCollater(BaseCollater):
    """Collater."""

    def __call__(self, batch: list[Sample]):
        """Collater's __call__."""
        batch_in_features, batch_out_features = self._batch_features(batch)

        # Stack features
        for i, _ in enumerate(self.in_feature_identifiers):
            assert self._can_stack_features(batch_in_features[i]), (
                f"features {self.in_feature_identifiers[i]} are of different sizes in batch"
            )
            batch_in_features[i] = torch.stack(
                [torch.as_tensor(feat) for feat in batch_in_features[i]]
            )

        for i, _ in enumerate(self.out_feature_identifiers):
            assert self._can_stack_features(batch_out_features[i]), (
                f"features {self.out_feature_identifiers[i]} are of different sizes in batch"
            )
            batch_out_features[i] = torch.stack(
                [torch.as_tensor(feat) for feat in batch_out_features[i]]
            )

        return batch_in_features, batch_out_features


# # -----------------------------------------------------------------------------------------------------
# class HeterogeneousPlaidDataLoader(DataLoader):
#     """PlaidDataLoader."""

#     def __init__(
#         self,
#         dataset: Dataset,
#         batch_size: int,
#         shuffle: bool,
#         in_feature_identifiers: list[FeatureIdentifier],
#         out_feature_identifiers: Optional[list[FeatureIdentifier]] = None,
#         **kwargs,
#     ):
#         super().__init__(
#             dataset,
#             batch_size,
#             shuffle,
#             collate_fn=HeterogeneousCollater(
#                 dataset, in_feature_identifiers, out_feature_identifiers
#             ),
#             **kwargs,
#         )


# class HeterogeneousCollater:
#     """Collater."""

#     def __init__(
#         self,
#         dataset: Dataset,
#         in_feature_identifiers: list[FeatureIdentifier],
#         out_feature_identifiers: Optional[list[FeatureIdentifier]] = None,
#     ):
#         self.dataset = dataset
#         self.in_feature_identifiers = in_feature_identifiers
#         self.out_feature_identifiers = out_feature_identifiers or []

#     def __call__(self, batch: list[Sample]):
#         """Collater's __call__."""
#         batch_in_features = {
#             _make_feat_id_hashable(feat): [] for feat in self.in_feature_identifiers
#         }
#         batch_out_features = {
#             _make_feat_id_hashable(feat): [] for feat in self.out_feature_identifiers
#         }

#         # Group samples by features
#         for sample in batch:
#             for feat in self.in_feature_identifiers:
#                 key = _make_feat_id_hashable(feat)
#                 batch_in_features[key].append(sample.get_feature_from_identifier(feat))
#             for feat in self.out_feature_identifiers:
#                 key = _make_feat_id_hashable(feat)
#                 batch_out_features[key].append(sample.get_feature_from_identifier(feat))

#         return batch_in_features, batch_out_features


# # -----------------------------------------------------------------------------------------------------


# class HomogeneousPlaidDataLoader(DataLoader):
#     """PlaidDataLoader."""

#     def __init__(
#         self,
#         dataset: Dataset,
#         batch_size: int,
#         shuffle: bool,
#         in_feature_identifiers: list[FeatureIdentifier],
#         out_feature_identifiers: Optional[list[FeatureIdentifier]] = None,
#         **kwargs,
#     ):
#         super().__init__(
#             dataset,
#             batch_size,
#             shuffle,
#             collate_fn=HomogeneousCollater(
#                 dataset, in_feature_identifiers, out_feature_identifiers
#             ),
#             **kwargs,
#         )


# class HomogeneousCollater:
#     """HomogeneousCollater."""

#     def __init__(
#         self,
#         dataset,
#         in_feature_identifiers: list[dict],
#         out_feature_identifiers: Optional[list[dict]] = None,
#     ):
#         self.dataset = dataset
#         self.in_feature_identifiers = in_feature_identifiers
#         self.out_feature_identifiers = out_feature_identifiers or []

#     def __call__(self, batch: list):
#         """SafeCollater's __call__."""
#         batch_in_features = {
#             _make_feat_id_hashable(feat): [] for feat in self.in_feature_identifiers
#         }
#         batch_out_features = {
#             _make_feat_id_hashable(feat): [] for feat in self.out_feature_identifiers
#         }

#         # Group samples by features
#         for sample in batch:
#             for feat in self.in_feature_identifiers:
#                 key = _make_feat_id_hashable(feat)
#                 batch_in_features[key].append(sample.get_feature_from_identifier(feat))
#             for feat in self.out_feature_identifiers:
#                 key = _make_feat_id_hashable(feat)
#                 batch_out_features[key].append(sample.get_feature_from_identifier(feat))

#         # Convert to tensors and stack only if shapes match
#         print("batch_in_features.keys() =", list[batch_in_features.keys()])
#         for key in batch_in_features.keys():
#             assert _can_stack_features(batch_in_features[key]), (
#                 f"features {key} are of different sizes in batch"
#             )
#             batch_in_features[key] = torch.stack(
#                 [torch.as_tensor(v) for v in batch_in_features[key]]
#             )
#         print("batch_out_features.keys() =", list[batch_out_features.keys()])
#         for key in batch_out_features.keys():
#             assert _can_stack_features(batch_out_features[key]), (
#                 f"features {key} are of different sizes in batch"
#             )
#             batch_out_features[key] = torch.stack(
#                 [torch.as_tensor(v) for v in batch_out_features[key]]
#             )

#         return batch_in_features, batch_out_features
