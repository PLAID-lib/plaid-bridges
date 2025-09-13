"""File implementing Grid-like TorchRegressionDatasets."""

from typing import Sequence

import numpy as np
import torch
from plaid.containers.dataset import Dataset
from plaid.types import Feature, FeatureIdentifier

from plaid_bridges.common import BaseBridge


class GridFieldsAndScalarsBridge(BaseBridge):
    """GridFieldsAndScalarsBridge."""

    def __init__(
        self,
        dimensions: Sequence[int],
    ):
        self.dimensions = dimensions

    def transform_single_feature(
        self, feature: Feature, feature_id: FeatureIdentifier
    ) -> torch.Tensor:
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

    def transform(
        self, dataset: Dataset, features_ids: list[FeatureIdentifier]
    ) -> torch.Tensor:
        """Transform."""
        tensor = torch.empty((len(dataset), len(features_ids), *self.dimensions))
        for i, sample in enumerate(dataset):
            for j, feat_id in enumerate(features_ids):
                feature = sample.get_feature_from_identifier(feat_id)
                tensor[i, j, ...] = self.transform_single_feature(feature, feat_id)

        return tensor

    def inverse_transform(
        self,
        features_ids: list[FeatureIdentifier],
        all_transformed_features: list[list[torch.Tensor]],
    ) -> list[list[Feature]]:
        """Inverse transform."""
        all_features = []
        for transformed_features in all_transformed_features:
            features = []
            for feat_id, transf_feature in zip(features_ids, transformed_features):
                assert isinstance(transf_feature, torch.Tensor)
                _type = feat_id["type"]
                if _type == "scalar":
                    feature = np.mean(transf_feature.numpy())
                elif _type == "field":
                    feature = transf_feature.numpy().flatten()
                else:
                    raise Exception(
                        f"feature type {_type} not compatible with `prediction_to_structured_grid`"
                    )  # pragma: no cover
                features.append(feature)
            all_features.append(features)

        return all_features
