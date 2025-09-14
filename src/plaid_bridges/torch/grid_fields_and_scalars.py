# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""GridFieldsAndScalarsBridge: Transforming grid and scalar features for PyTorch.

This module implements the `GridFieldsAndScalarsBridge` class, which enables
conversion of PLAID datasets containing both grid-based field features and scalar
features into PyTorch tensors. It supports reshaping field data into specified
grid dimensions and direct handling of scalar values, making the data suitable
for deep learning workflows. The bridge also provides inverse transformation to
convert model predictions back to their original PLAID feature format.
"""

from typing import Sequence

import numpy as np
import torch
from plaid.containers.dataset import Dataset
from plaid.types import Feature, FeatureIdentifier

from plaid_bridges.common import ArrayDataset, BaseBridge


class GridFieldsAndScalarsBridge(BaseBridge):
    """Bridge for transforming grid fields and scalar features into PyTorch tensors.

    This bridge processes datasets containing two types of features:
      - Fields: Multi-dimensional arrays that are reshaped into grid dimensions.
      - Scalars: Single numerical values used directly.

    It transforms these features into PyTorch tensors for use in deep learning
    models, and can also convert predictions back to the original PLAID format.
    """

    def __init__(
        self,
        dimensions: Sequence[int],
    ):
        """Initialize the GridFieldsAndScalarsBridge.

        Args:
            dimensions: The target grid dimensions for reshaping field features,
                e.g., (height, width) for 2D fields.
        """
        self.dimensions = dimensions
        super().__init__(ArrayDataset)

    def transform_single_feature(
        self, feature: Feature, feature_id: FeatureIdentifier
    ) -> torch.Tensor:
        """Transform a single feature into a PyTorch tensor.

        Converts a feature to a PyTorch tensor. Field features are reshaped
        according to the specified grid dimensions; scalar features are used as-is.

        Args:
            feature: The feature value to transform.
            feature_id: The feature identifier, including type information.

        Returns:
            A PyTorch tensor representation of the feature.

        Raises:
            Exception: If the feature type is not supported by this transformer.
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
                f"feature type {_type} not compatible with `GridFieldsAndScalarsBridge`"
            )  # pragma: no cover
        return torch.tensor(treated_feature)

    def transform(
        self, dataset: Dataset, *features_ids: list[FeatureIdentifier]
    ) -> tuple[torch.Tensor, ...]:
        """Transform dataset features into PyTorch tensors.

        Converts features from a dataset into one or more multi-dimensional
        PyTorch tensors. Each tensor corresponds to a list of feature identifiers,
        with field features reshaped to the specified grid dimensions and scalar
        features kept as-is.

        Args:
            dataset: The input PLAID dataset containing features to transform.
            features_ids: One or more lists of feature identifiers to transform.

        Returns:
            A tuple of PyTorch tensors, each of shape
            ``(n_samples, n_features, *dimensions)``, containing the transformed features.
        """
        tensors = []
        for feat_ids in features_ids:
            tensor = torch.empty((len(dataset), len(feat_ids), *self.dimensions))
            for i, sample in enumerate(dataset):
                for j, feat_id in enumerate(feat_ids):
                    feature = sample.get_feature_from_identifier(feat_id)
                    tensor[i, j, ...] = self.transform_single_feature(feature, feat_id)
            tensors.append(tensor)

        return tuple(tensors)

    def inverse_transform(
        self,
        features_ids: list[FeatureIdentifier],
        all_transformed_features: list[list[torch.Tensor]],
    ) -> list[list[Feature]]:
        """Inverse transform predicted features to original PLAID format.

        Converts predicted PyTorch tensors back to their original feature format.
        Field features are flattened; scalar features are averaged.

        Args:
            features_ids: List of feature identifiers that were transformed.
            all_transformed_features: List of lists of PyTorch tensors to convert back.

        Returns:
            A list of lists of features in their original PLAID format.

        Raises:
            Exception: If a feature type is not supported by this transformer.
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
                        f"feature type {_type} not compatible with `GridFieldsAndScalarsBridge`"
                    )  # pragma: no cover
                features.append(feature)
            all_features.append(features)

        return all_features
