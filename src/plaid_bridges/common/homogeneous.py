"""Implement the `HomogeneousBridge` class for transforming homogeneous features.

This module provides a bridge for handling datasets where all features
are of the same type. It enables efficient transformation of homogeneous
features into NumPy arrays for machine learning workflows.
"""

import numpy as np
from plaid.containers.dataset import Dataset
from plaid.types import Feature, FeatureIdentifier, Scalar

from plaid_bridges.common import BaseBridge


class HomogeneousBridge(BaseBridge):
    """Bridge for transforming homogeneous features in a dataset.

    This bridge handles datasets where all features are of the same type,
    transforming them into NumPy arrays for efficient processing in ML pipelines.
    It supports both forward transformation of features for model input and
    inverse transformation of predicted features back to their original format.
    """

    def transform(
        self, dataset: Dataset, features_ids: list[FeatureIdentifier]
    ) -> np.ndarray:
        """Transform homogeneous features into a NumPy array.

        Converts features of the same type from a dataset into a stacked NumPy array
        suitable for machine learning models.

        Args:
            dataset: The input dataset containing the features to transform.
            features_ids: List of feature identifiers to transform. All features
                         must be of the same type.

        Returns:
            A NumPy array of shape (n_samples, n_features) containing the
            transformed features.

        Raises:
            AssertionError: If the features are not all of the same type.
        """
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
        """Inverse transform predicted features to original format.

        Converts predicted NumPy arrays back to their original feature format.

        Args:
            features_ids: List of feature identifiers that were transformed.
            all_transformed_features: List of transformed features (NumPy arrays)
                                     to convert back to original format.

        Returns:
            List of features in their original format.
        """
        del features_ids
        all_features = []
        for transformed_features in all_transformed_features:
            features = []
            for transf_feature in transformed_features:
                assert isinstance(transf_feature, (np.ndarray, Scalar))
                features.append(transf_feature)
            all_features.append(features)

        return all_features
