"""Class implementing PyTorch loaders."""

from typing import Optional, Sequence

import numpy as np
import torch
from plaid.containers.dataset import Dataset
from plaid.containers.utils import get_feature_type_and_details_from
from plaid.types import FeatureIdentifier
from torch.utils.data import DataLoader

# from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset as TorchTensorDataset

# TODO: maybe create a class with transform/inverse transform with these two functions ? (see the FNO notebook)


def structured_grid_with_scalars_loader(
    grid_dataset: Dataset,
    dimensions: Sequence[int],
    in_features_identifiers: list[FeatureIdentifier],
    out_features_identifiers: list[FeatureIdentifier],
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
    predictions: list,
    out_features_identifiers: list[FeatureIdentifier],
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
