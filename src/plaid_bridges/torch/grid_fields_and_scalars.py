# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Implementation of Grid-like TorchRegressionDatasets.

This module provides a bridge for transforming grid-based features (fields) and
scalar features into PyTorch tensors for machine learning workflows. It handles
both field data that needs to be reshaped into grid dimensions and scalar values
that can be used directly.
"""

from typing import Sequence

import numpy as np
import torch
from plaid.containers.dataset import Dataset
from plaid.types import Feature, FeatureIdentifier

from plaid_bridges.common import BaseBridge


class GridFieldsAndScalarsBridge(BaseBridge):
    """Bridge for transforming grid fields and scalar features into PyTorch tensors.

    This bridge handles datasets containing two types of features:
    1. Fields: Multi-dimensional data that needs to be reshaped into grid dimensions
    2. Scalars: Single numerical values that can be used directly

    The bridge transforms these features into PyTorch tensors suitable for
    deep learning models and can also inverse transform predictions back to
    their original format.
    """

    def __init__(
        self,
        dimensions: Sequence[int],
    ):
        """Initialize the GridFieldsAndScalarsBridge.

        Args:
            dimensions: The grid dimensions to reshape field features into.
                       For example, (height, width) for 2D fields.
        """
        self.dimensions = dimensions

    def transform_single_feature(
        self, feature: Feature, feature_id: FeatureIdentifier
    ) -> torch.Tensor:
        """Transform a single feature into a PyTorch tensor.

        Converts a feature to a PyTorch tensor, reshaping field features
        according to the specified dimensions while leaving scalar features unchanged.

        Args:
            feature: The feature to transform.
            feature_id: Identifier containing the feature type information.

        Returns:
            A PyTorch tensor representation of the feature.

        Raises:
            Exception: If the feature type is not compatible with this transformer.
        """
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
        """Transform dataset features into a PyTorch tensor.

        Converts features from a dataset into a multi-dimensional PyTorch tensor
        where field features are reshaped according to the specified dimensions
        and scalar features are kept as-is.

        Args:
            dataset: The input dataset containing features to transform.
            features_ids: List of feature identifiers to transform.

        Returns:
            A PyTorch tensor of shape (n_samples, n_features, *dimensions)
            containing the transformed features.
        """
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
        """Inverse transform predicted features to original format.

        Converts predicted PyTorch tensors back to their original feature format,
        flattening field features and averaging scalar features as needed.

        Args:
            features_ids: List of feature identifiers that were transformed.
            all_transformed_features: List of transformed features (PyTorch tensors)
                                     to convert back to original format.

        Returns:
            List of features in their original format.

        Raises:
            Exception: If the feature type is not compatible with this transformer.
        """
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
