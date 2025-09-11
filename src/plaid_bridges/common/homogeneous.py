"""Implement the `HomogeneousOfflineTransformer` class."""

from typing import Optional

import numpy as np
from plaid.containers.dataset import Dataset
from plaid.types import Feature, FeatureIdentifier

from plaid_bridges.common.base_regression import (
    BaseTransformer,
)


class HomogeneousOfflineTransformer(BaseTransformer):
    """HomogeneousOfflineTransformer."""

    def __init__(
        self,
        features_identifiers: list[FeatureIdentifier],
    ):
        super().__init__(
            features_identifiers,
        )

        assert len(set([feat_id["type"] for feat_id in features_identifiers])), (
            "input features not of same type"
        )

    def transform(self, dataset: Dataset) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform."""
        stacked_features = np.stack(
            [
                [
                    (f if f is not None else 0.0)
                    for f in (
                        sample.get_feature_from_identifier(fid)
                        for fid in self.features_identifiers
                    )
                ]
                for sample in dataset
            ]
        )

        return stacked_features

    @staticmethod
    def inverse_transform_single_feature(
        feat_id: FeatureIdentifier,  # noqa: ARG004  # ignore unused argument
        predicted_feature: Feature,
    ) -> Feature:
        """inverse_transform single feature."""
        return predicted_feature
