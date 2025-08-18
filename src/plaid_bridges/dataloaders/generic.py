"""Class implementing PyTorch loaders."""

from typing import Optional

import torch
from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.types import FeatureIdentifier
from torch.utils.data import DataLoader

# from torch.utils.data import Dataset as TorchDataset

# TODO: maybe create a class with transform/inverse transform with these two functions ? (see the FNO notebook)


class PlaidSampleDataLoader(DataLoader):
    """PlaidSampleDataLoader."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda batch: batch,
            **kwargs,
        )


class PlaidDataLoader(DataLoader):
    """PlaidDataLoader."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        in_feature_identifiers: list[FeatureIdentifier],
        out_feature_identifiers: Optional[list[FeatureIdentifier]] = None,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=SafeCollater(
                dataset, in_feature_identifiers, out_feature_identifiers
            ),
            **kwargs,
        )


class Collater:
    """Collater."""

    def __init__(
        self,
        dataset: Dataset,
        in_feature_identifiers: list[FeatureIdentifier],
        out_feature_identifiers: Optional[list[FeatureIdentifier]] = None,
    ):
        self.dataset = dataset
        self.in_feature_identifiers = in_feature_identifiers
        self.out_feature_identifiers = out_feature_identifiers or []

    @staticmethod
    def _make_hashable(feat_dict: dict):
        return tuple(sorted(feat_dict.items()))

    def __call__(self, batch: list[Sample]):
        """Collater's __call__."""
        in_features = {
            self._make_hashable(feat): [] for feat in self.in_feature_identifiers
        }
        out_features = {
            self._make_hashable(feat): [] for feat in self.out_feature_identifiers
        }

        for sample in batch:
            for feat in self.in_feature_identifiers:
                key = self._make_hashable(feat)
                in_features[key].append(sample.get_feature_from_identifier(feat))
            for feat in self.out_feature_identifiers:
                key = self._make_hashable(feat)
                out_features[key].append(sample.get_feature_from_identifier(feat))

        # Convert all elements to torch.Tensor and stack
        for key in in_features.keys():
            in_features[key] = torch.stack(
                [torch.as_tensor(x) for x in in_features[key]]
            )
        for key in out_features.keys():
            out_features[key] = torch.stack(
                [torch.as_tensor(x) for x in out_features[key]]
            )

        return in_features, out_features


class SafeCollater:
    """SafeCollater."""

    def __init__(
        self,
        dataset,
        in_feature_identifiers: list[dict],
        out_feature_identifiers: Optional[list[dict]] = None,
    ):
        self.dataset = dataset
        self.in_feature_identifiers = in_feature_identifiers
        self.out_feature_identifiers = out_feature_identifiers or []

    @staticmethod
    def _make_hashable(feat_dict: dict):
        return tuple(sorted(feat_dict.items()))

    @staticmethod
    def _can_stack(values: list):
        """Check if all elements have the same shape for stacking."""
        shapes = [torch.as_tensor(v).shape for v in values]
        return len(set(shapes)) == 1

    def _stack_or_list(self, values: list):
        if self._can_stack(values):
            return torch.stack([torch.as_tensor(v) for v in values])
        else:
            return values  # heterogeneous, return list

    def __call__(self, batch: list):
        """SafeCollater's __call__."""
        in_features = {
            self._make_hashable(feat): [] for feat in self.in_feature_identifiers
        }
        out_features = {
            self._make_hashable(feat): [] for feat in self.out_feature_identifiers
        }

        for sample in batch:
            for feat in self.in_feature_identifiers:
                key = self._make_hashable(feat)
                in_features[key].append(sample.get_feature_from_identifier(feat))
            for feat in self.out_feature_identifiers:
                key = self._make_hashable(feat)
                out_features[key].append(sample.get_feature_from_identifier(feat))

        # Convert to tensors and stack only if shapes match
        for key in in_features:
            in_features[key] = self._stack_or_list(in_features[key])
        for key in out_features:
            out_features[key] = self._stack_or_list(out_features[key])

        return in_features, out_features


# from plaid_bridges.dataloader import DataLoader

# for batch in dataloader:
# 	batch = batch.to(device)

# 	nodes = batch.nodes
# 	scalars = batch.scalars
# 	fields = batch.fields

# 	pred = model(nodes, scalars, fields)
