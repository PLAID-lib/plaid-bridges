"""Class implementing PyTorch loaders."""

from typing import List, Optional, Sequence

import numpy as np
import torch
from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.containers.utils import get_feature_type_and_details_from
from plaid.types import FeatureIdentifier
from torch.utils.data import DataLoader

# from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset as TorchTensorDataset

from plaid_bridges.dataloaders.generic import BaseCollater

# TODO: maybe create a class with transform/inverse transform with these two functions ? (see the FNO notebook)


def structured_grid_with_scalars_loader(
    grid_dataset: Dataset,
    dimensions: Sequence[int],
    in_features_identifiers: List[FeatureIdentifier],
    out_features_identifiers: List[FeatureIdentifier],
    training: Optional[bool] = True,
    batch_size: Optional[int] = 1,
) -> DataLoader:
    """Initialize a :class:`StructuredGridWithScalars <plaid_bridges.torch_data_loaders.StructuredGridWithScalars>`.

    Work for dimensions 1, 2 and 3 (and upper).
    """
    dims = tuple(dimensions)

    in_scalars = []
    in_fields = []
    out_scalars = []
    out_fields = []

    for in_feat in in_features_identifiers:
        _type, _details = get_feature_type_and_details_from(in_feat)
        if _type == "scalar":
            in_scalars.append(_details)
        elif _type == "field":
            in_fields.append(_details)
        else:
            raise Exception(
                f"feature type {_type} not compatible with `structured_grid_with_scalars_loader`"
            )

    for out_feat in out_features_identifiers:
        _type, _details = get_feature_type_and_details_from(out_feat)
        if _type == "scalar":
            out_scalars.append(_details)
        elif _type == "field":
            out_fields.append(_details)
        else:
            raise Exception(
                f"feature type {_type} not compatible with `structured_grid_with_scalars_loader`"
            )

    n_samples = len(grid_dataset)
    sample_ids = grid_dataset.get_sample_ids()

    inputs = np.empty(
        (n_samples, len(in_scalars) + len(in_fields), *dims), dtype=np.float32
    )
    for i, id_sample in enumerate(sample_ids):
        for k, sn in enumerate(in_scalars):
            inputs[i, k, ...] = grid_dataset[id_sample].get_scalar(**sn)
        for k, fn in enumerate(in_fields):
            inputs[i, k + len(in_scalars), ...] = (
                grid_dataset[id_sample].get_field(**fn).reshape(dims)
            )

    inputs_torch = torch.from_numpy(inputs)

    if training:
        outputs = np.empty(
            (n_samples, len(out_scalars) + len(out_fields), *dims), dtype=np.float32
        )
        for i, id_sample in enumerate(sample_ids):
            for k, sn in enumerate(out_scalars):
                outputs[i, k, ...] = grid_dataset[id_sample].get_scalar(**sn)
            for k, fn in enumerate(out_fields):
                outputs[i, k + len(out_scalars), ...] = (
                    grid_dataset[id_sample].get_field(**fn).reshape(dims)
                )
        outputs_torch = torch.from_numpy(outputs)
        _torch_dataset = TorchTensorDataset(inputs_torch, outputs_torch)
        shuffle = True
    else:
        _torch_dataset = TorchTensorDataset(inputs_torch)
        shuffle = False

    return DataLoader(_torch_dataset, batch_size=batch_size, shuffle=shuffle)


def prediction_to_structured_grid(
    grid_dataset: Dataset,
    predictions: List,
    out_features_identifiers: List[FeatureIdentifier],
) -> Dataset:
    """Set prediction from FNO to dataset."""
    pred_features_dict = {id: [] for id in grid_dataset.get_sample_ids()}

    for out_feat in out_features_identifiers:
        _type = out_feat["type"]
        for i, id in enumerate(grid_dataset.get_sample_ids()):
            if _type == "scalar":
                pred_features_dict[id].append(np.mean(predictions[i].flatten()))
            elif _type == "field":
                pred_features_dict[id].append(predictions[i].flatten())
            else:
                raise Exception(
                    f"feature type {_type} not compatible with `prediction_to_structured_grid`"
                )

    return grid_dataset.update_features_from_identifier(
        out_features_identifiers, pred_features_dict
    )


# -----------------------------------------------------------------------------------------------------


class GridFieldsAndScalarsCollater(BaseCollater):
    def __init__(
        self,
        dimensions: Sequence[int],
        in_feature_identifiers: List[FeatureIdentifier],
        out_feature_identifiers: Optional[List[FeatureIdentifier]] = None,
    ):
        self.dims = tuple(dimensions)
        super().__init__(
            in_feature_identifiers,
            out_feature_identifiers,
        )

    def _treat_feature(self, sample, feat_id):
        feature = sample.get_feature_from_identifier(feat_id)

        _type = feat_id["type"]
        if _type == "scalar":
            treated_feature = feature
        elif _type == "field":
            treated_feature = feature.reshape(self.dims)
        else:
            raise Exception(
                f"feature type {_type} not compatible with `GridFieldsAndScalarsCollater`"
            )

        return torch.tensor(treated_feature, dtype=torch.float32)

    def __call__(self, batch: List[Sample]):
        """Collater's __call__."""
        batch_in_features = torch.empty(
            (len(batch), len(self.in_feature_identifiers), *self.dims),
            dtype=torch.float32,
        )
        batch_out_features = torch.empty(
            (len(batch), len(self.out_feature_identifiers), *self.dims),
            dtype=torch.float32,
        )

        # Group samples by features
        for i, sample in enumerate(batch):
            for j, feat_id in enumerate(self.in_feature_identifiers):
                batch_in_features[i, j, ...] = self._treat_feature(sample, feat_id)
            for j, feat_id in enumerate(self.out_feature_identifiers):
                batch_out_features[i, j, ...] = self._treat_feature(sample, feat_id)

        return batch_in_features, batch_out_features


# ----------------------------------------------------------------------------------------------------------------


class GridFieldsAndScalarsTransformer:
    """Transforms raw Samples to tensors and back."""

    def __init__(
        self,
        dimensions: Sequence[int],
        in_feature_identifiers: List,
        out_feature_identifiers: Optional[List] = None,
    ):
        self.dims = tuple(dimensions)
        self.in_feature_identifiers = in_feature_identifiers
        self.out_feature_identifiers = out_feature_identifiers or []

    def transform_sample(self, sample, feature_ids):
        feature = sample.get_feature_from_identifier(feature_ids)

        _type = feature_ids["type"]
        if _type == "scalar":
            treated_feature = feature
        elif _type == "field":
            treated_feature = feature.reshape(self.dims)
        else:
            raise Exception(
                f"feature type {_type} not compatible with `GridFieldsAndScalarsTransformer`"
            )

        return torch.tensor(treated_feature, dtype=torch.float32)

    def transform_batch(self, batch):
        """Transform a batch of samples into tensors for DataLoader."""
        batch_in_features = torch.empty(
            (len(batch), len(self.in_feature_identifiers), *self.dims),
            dtype=torch.float32,
        )
        batch_out_features = torch.empty(
            (len(batch), len(self.out_feature_identifiers), *self.dims),
            dtype=torch.float32,
        )

        # Group samples by features
        for i, sample in enumerate(batch):
            for j, feat_id in enumerate(self.in_feature_identifiers):
                batch_in_features[i, j, ...] = self.transform_sample(sample, feat_id)
            for j, feat_id in enumerate(self.out_feature_identifiers):
                batch_out_features[i, j, ...] = self.transform_sample(sample, feat_id)

        return batch_in_features, batch_out_features

    def inverse_transform_batch(self, batch_in, batch_out):
        """Convert tensors back to original NumPy-like sample dicts."""
        inv_in = []
        inv_out = []

        for i in range(batch_in.shape[0]):
            sample_in = {}
            sample_out = {}
            for j, feat_id in enumerate(self.in_feature_identifiers):
                t = batch_in[i, j]
                if feat_id["type"] == "scalar":
                    sample_in[feat_id["name"]] = t.item()
                else:
                    sample_in[feat_id["name"]] = t.numpy()
            for j, feat_id in enumerate(self.out_feature_identifiers):
                t = batch_out[i, j]
                if feat_id["type"] == "scalar":
                    sample_out[feat_id["name"]] = t.item()
                else:
                    sample_out[feat_id["name"]] = t.numpy()
            inv_in.append(sample_in)
            inv_out.append(sample_out)

        return inv_in, inv_out

    def inverse_transform_batch(self, batch_out):

        """Set prediction from FNO to dataset."""
        features_out = []

        for i, batched_sample in enumerate(batch_out):

            features_out.append([])

            for j, out_feat in enumerate(self.out_feature_identifiers):
                _type = out_feat["type"]

                if _type == "scalar":
                    features_out[i].append(np.mean(batched_sample[j].flatten()).detach().cpu().numpy())
                elif _type == "field":
                    features_out[i].append(batched_sample[j].flatten().detach().cpu().numpy())
                else:
                    raise Exception(
                        f"feature type {_type} not compatible with `GridFieldsAndScalarsTransformer`"
                    )

        return features_out
