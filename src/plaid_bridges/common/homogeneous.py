# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Homogeneous feature transformers for PLAID bridges.

This module provides the `HomogeneousBridge` class and the `ArrayDataset` wrapper,
enabling efficient transformation of datasets where all features are of the same type
(e.g., all scalars or all fields). These tools convert homogeneous features from
PLAID datasets into stacked NumPy arrays or tensors, making them suitable for
machine learning workflows. The bridge supports both forward transformation for
model input and inverse transformation for converting predictions back to the
original data format.
"""

from typing import TYPE_CHECKING, Union

import numpy as np
from plaid.containers.dataset import Dataset
from plaid.types import Feature, FeatureIdentifier, Scalar

from plaid_bridges.common.base import BaseBridge

if TYPE_CHECKING:
    import torch

ArrayType = Union[np.ndarray, "torch.Tensor"]


class ArrayDataset:
    """Machine Learning Dataset wrapper for handling multiple data sources.

    This class wraps multiple data arrays and provides a unified interface
    for accessing samples across all data sources by index. It's designed to
    work with transformed features from the BaseBridge class.
    """

    def __init__(self, all_data: tuple[ArrayType, ...]) -> None:
        """Initialize the ArrayDataset with multiple data sources.

        Args:
            all_data: Tuple of data arrays/tensors to be wrapped.
                      All data sources must have the same length.
        """
        self.all_data = all_data

    def __getitem__(self, index: int) -> tuple[ArrayType, ...]:
        """Get a sample from all data sources by index.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            A tuple containing the sample data from each data source at the given index.
        """
        return tuple(data[index] for data in self.all_data)

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            The length of the dataset (based on the first data source).
        """
        return len(self.all_data[0])


class HomogeneousBridge(BaseBridge):
    """Bridge for transforming homogeneous features in a dataset.

    This bridge handles datasets where all features are of the same type,
    transforming them into NumPy arrays for efficient processing in ML pipelines.
    It supports both forward transformation of features for model input and
    inverse transformation of predicted features back to their original format.
    """

    def __init__(self):
        """Initialize the HomogeneousBridge."""
        super().__init__(ArrayDataset)

    def transform(
        self, dataset: Dataset, *features_ids: list[FeatureIdentifier]
    ) -> tuple[np.ndarray, ...]:
        """Transform homogeneous features into NumPy arrays.

        Converts features of the same type from a dataset into stacked NumPy arrays
        suitable for machine learning models. Each list of feature identifiers
        produces a separate array in the returned tuple.

        Args:
            dataset: The input dataset containing the features to transform.
            features_ids: Variable number of lists of feature identifiers to transform.
                         All features within each list must be of the same type.

        Returns:
            A tuple of NumPy arrays, each of shape (n_samples, n_features) containing
            the transformed features for each list of feature identifiers.

        Raises:
            AssertionError: If features within any list are not all of the same type.
        """
        for feat_ids in features_ids:
            assert len(set([feat_id["type"] for feat_id in feat_ids])), (
                f"input features not of same type in {feat_ids}"
            )

        return tuple(
            np.stack(
                [
                    [
                        feature
                        for feature in (
                            sample.get_feature_from_identifier(feat_id)
                            for feat_id in feat_ids
                        )
                    ]
                    for sample in dataset
                ]
            )
            for feat_ids in features_ids
        )

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
