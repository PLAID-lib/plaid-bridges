"""File implementing Grid-like TorchRegressionDatasets."""

from typing import Sequence

import numpy as np
import torch
from plaid.containers.dataset import Dataset
from plaid.types import Feature, FeatureIdentifier

from plaid_bridges.common import BaseTransformer


class GridFieldsAndScalarsOfflineTransformer(BaseTransformer):
    """GridFieldsAndScalarsOfflineTransformer."""

    def __init__(
        self,
        dimensions: Sequence[int],
        features_identifiers: list[FeatureIdentifier],
    ):
        self.dimensions = dimensions

        super().__init__(
            features_identifiers,
        )

    def transform_single_feature(self, feature: Feature, feature_id: FeatureIdentifier):
        """Transform_single_feature."""
        assert feature is not None
        _type = feature_id["type"]
        if (
            _type == "scalar"
        ):  # and isinstance(feature, Scalar): # `isinstance` not adapted to complex type aliases
            treated_feature = feature
        elif _type == "field":  # and isinstance(feature, np.ndarray):
            treated_feature = feature.reshape(self.dimensions)
        else:
            raise Exception(
                f"feature type {_type} not compatible with `GridFieldsAndScalarsTransformer`"
            )  # pragma: no cover
        return torch.tensor(treated_feature)

    def transform(self, dataset: Dataset) -> torch.Tensor:
        """Transform."""
        tensor = torch.empty(
            (len(dataset), len(self.features_identifiers), *self.dimensions)
        )
        for i, sample in enumerate(dataset):
            for j, feat_id in enumerate(self.features_identifiers):
                feature = sample.get_feature_from_identifier(feat_id)
                tensor[i, j, ...] = self.transform_single_feature(feature, feat_id)

        return tensor

    @staticmethod
    def inverse_transform_single_feature(
        feat_id: FeatureIdentifier, predicted_feature: torch.Tensor
    ) -> Feature:
        """inverse_transform single feature."""
        assert isinstance(predicted_feature, torch.Tensor)
        _type = feat_id["type"]
        if _type == "scalar":
            return np.mean(predicted_feature.numpy())
        elif _type == "field":
            return predicted_feature.numpy().flatten()
        else:
            raise Exception(
                f"feature type {_type} not compatible with `prediction_to_structured_grid`"
            )  # pragma: no cover
