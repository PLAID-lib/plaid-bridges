"""File implementing Grid-like TorchRegressionDatasets."""

from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from plaid.containers.dataset import Dataset
from plaid.types import FeatureIdentifier

from plaid_bridges.torch.base import BaseRegressionDataset


class GridFieldsAndScalarsDataset(BaseRegressionDataset):
    """GridFieldsAndScalarsDataset."""

    def __init__(
        self,
        dataset: Dataset,
        dimensions: Sequence[int],
        in_feature_identifiers: List[FeatureIdentifier],
        out_feature_identifiers: List[FeatureIdentifier],
        online_transform: Optional[Callable] = None,
    ):
        super().__init__(
            dataset, in_feature_identifiers, out_feature_identifiers, online_transform
        )

        self.dims = tuple(dimensions)

        self.in_features = self._create_tensor(self.in_features, self.in_feature_identifiers)
        self.out_features = self._create_tensor(
            self.out_features, self.out_feature_identifiers
        )

    def _transform_sample(self, feature, feature_ids):
        _type = feature_ids["type"]
        if _type == "scalar":
            treated_feature = feature
        elif _type == "field":
            treated_feature = feature.reshape(self.dims)
        else:
            raise Exception(
                f"feature type {_type} not compatible with `GridFieldsAndScalarsTransformer`"
            )  # pragma: no cover
        return torch.tensor(treated_feature)


    def _create_tensor(self, features, feature_identifiers):
        tensor = torch.empty((len(features), len(feature_identifiers), *self.dims))
        for i, feature in enumerate(features):
            for j, feat_id in enumerate(feature_identifiers):
                tensor[i, j, ...] = self._transform_sample(feature[j], feat_id)
        return tensor


    @staticmethod
    def inverse_transform_single_feature(feat_id, predicted_feature):
        """inverse_transform single feature."""
        _type = feat_id["type"]
        if _type == "scalar":
            return np.mean(predicted_feature)
        elif _type == "field":
            return predicted_feature.flatten()
        else:
            raise Exception(
                f"feature type {_type} not compatible with `prediction_to_structured_grid`"
            )  # pragma: no cover
