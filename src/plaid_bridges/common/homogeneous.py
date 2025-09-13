"""Implement the `HomogeneousOfflineTransformer` class."""

import numpy as np
from plaid.containers.dataset import Dataset
from plaid.types import Feature, FeatureIdentifier, Scalar

from plaid_bridges.common import BaseBridge


class HomogeneousBridge(BaseBridge):
    """HomogeneousBridge."""

    def transform(
        self, dataset: Dataset, features_ids: list[FeatureIdentifier]
    ) -> np.ndarray:
        """Transform."""
        assert len(set([feat_id["type"] for feat_id in features_ids])), (
            "input features not of same type"
        )

        stacked_features = np.stack(
            [
                [
                    feature
                    for feature in (
                        sample.get_feature_from_identifier(feat_id)
                        for feat_id in features_ids
                    )
                ]
                for sample in dataset
            ]
        )

        return stacked_features

    def inverse_transform(
        self,
        features_ids: list[FeatureIdentifier],
        all_transformed_features: list[list[np.ndarray]],
    ) -> list[list[Feature]]:
        """Inverse transform."""
        del features_ids
        all_features = []
        for transformed_features in all_transformed_features:
            features = []
            for transf_feature in transformed_features:
                assert isinstance(transf_feature, (np.ndarray, Scalar))
                features.append(transf_feature)
            all_features.append(features)

        return all_features
