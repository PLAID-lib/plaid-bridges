"""File implementing Grid-like TorchRegressionDatasets."""

from typing import Optional, Sequence

import numpy as np
import torch
from plaid.containers.dataset import Dataset
from plaid.types import FeatureIdentifier, FeatureType, ScalarType

from plaid_bridges.common import (
    BaseRegressionDataset,
)
from plaid_bridges.common.base_regression_dataset import feature_transform


class GridFieldsAndScalarsDataset(BaseRegressionDataset):
    """GridFieldsAndScalarsDataset.

    Args:
        dataset (Dataset): PLAID dataset.
        in_feature_identifiers (list[FeatureIdentifier]): List of input feature identifiers.
        out_feature_identifiers (list[FeatureIdentifier]): List of output feature identifiers.
        online_transform (featuture_transform, optional): Transformation applied to the samples through the `__getitem__` function.
    """

    def __init__(
        self,
        dataset: Dataset,
        dimensions: Sequence[int],
        in_feature_identifiers: list[FeatureIdentifier],
        out_feature_identifiers: list[FeatureIdentifier],
        online_transform: Optional[feature_transform] = None,
    ):
        super().__init__(
            dataset, in_feature_identifiers, out_feature_identifiers, online_transform
        )

        self.dims = tuple(dimensions)

        self.in_features = self._create_tensor(  # pyright: ignore[reportAttributeAccessIssue]  # overwritting self.in_features
            self.in_features, self.in_feature_identifiers
        )
        self.out_features = self._create_tensor(  # pyright: ignore[reportAttributeAccessIssue]  # overwritting self.in_features
            self.out_features, self.out_feature_identifiers
        )

    def _transform_sample(self, feature: FeatureType, feature_ids: FeatureIdentifier):
        _type = feature_ids["type"]
        if _type == "scalar" and isinstance(feature, ScalarType):
            treated_feature = feature
        elif _type == "field" and isinstance(feature, np.ndarray):
            treated_feature = feature.reshape(self.dims)
        else:
            raise Exception(
                f"feature type {_type} not compatible with `GridFieldsAndScalarsTransformer`"
            )  # pragma: no cover
        return torch.tensor(treated_feature)

    def _create_tensor(
        self,
        features: list[list[FeatureType]],
        feature_identifiers: list[FeatureIdentifier],
    ) -> torch.Tensor:
        tensor = torch.empty((len(features), len(feature_identifiers), *self.dims))
        for i, feature in enumerate(features):
            for j, feat_id in enumerate(feature_identifiers):
                tensor[i, j, ...] = self._transform_sample(feature[j], feat_id)
        return tensor

    @staticmethod
    def inverse_transform_single_feature(
        feat_id: FeatureIdentifier, predicted_feature: FeatureType
    ) -> FeatureType:
        """inverse_transform single feature."""
        _type = feat_id["type"]
        if _type == "scalar" and isinstance(predicted_feature, ScalarType):
            return np.mean(predicted_feature)
        elif _type == "field" and isinstance(predicted_feature, np.ndarray):
            return predicted_feature.flatten()
        else:
            raise Exception(
                f"feature type {_type} not compatible with `prediction_to_structured_grid`"
            )  # pragma: no cover
