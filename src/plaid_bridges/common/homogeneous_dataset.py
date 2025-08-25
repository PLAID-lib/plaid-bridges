"""Implement the `HomogeneousDataset` class for regression problems."""

from typing import Optional

import numpy as np
from plaid.containers.dataset import Dataset
from plaid.types import FeatureIdentifier, FeatureType

from plaid_bridges.common.base_regression_dataset import (
    BaseRegressionDataset,
    feature_transform,
)


class HomogeneousDataset(BaseRegressionDataset):
    """Implement a regression dataset for homogeneous data.

    The input and output features can all be stacked into tensors.

    Args:
        dataset (Dataset): PLAID dataset.
        in_feature_identifiers (list[FeatureIdentifier]): List of input feature identifiers.
        out_feature_identifiers (list[FeatureIdentifier]): List of output feature identifiers.
        online_transform (featuture_transform, optional): Transformation applied to the samples through the `__getitem__` function.

    """

    def __init__(
        self,
        dataset: Dataset,
        in_feature_identifiers: list[FeatureIdentifier],
        out_feature_identifiers: list[FeatureIdentifier],
        online_transform: Optional[feature_transform] = None,
    ):
        super().__init__(
            dataset, in_feature_identifiers, out_feature_identifiers, online_transform
        )

        assert len(set([feat_id["type"] for feat_id in self.in_feature_identifiers])), (
            "input features not of same type"
        )
        assert len(
            set([feat_id["type"] for feat_id in self.out_feature_identifiers])
        ), "input features not of same type"

        self.in_features = np.stack(  # pyright: ignore[reportAttributeAccessIssue]  # overwritting self.in_features
            [
                np.stack([feat for feat in sample_features])
                for sample_features in self.in_features
            ]
        )
        self.out_features = np.stack(  # pyright: ignore[reportAttributeAccessIssue]  # overwritting self.in_features
            [
                np.stack([feat for feat in sample_features])
                for sample_features in self.out_features
            ]
        )

    @staticmethod
    def inverse_transform_single_feature(
        feat_id: FeatureIdentifier,  # noqa: ARG004  # ignore unused argument
        predicted_feature: FeatureType,
    ) -> FeatureType:
        """inverse_transform single feature."""
        return predicted_feature
