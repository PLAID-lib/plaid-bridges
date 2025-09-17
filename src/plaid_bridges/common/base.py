# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Implement `BaseBridge` for transforming datasets in ML pipelines.

This module provides base classes for transforming datasets and creating
ML-ready datasets. The BaseBridge handles feature transformation
and inverse transformation for machine learning workflows.
"""

from typing import Any, Generic, Type, TypeVar, Union

from plaid.containers import Dataset
from plaid.types import Feature, FeatureIdentifier

DatasetType = TypeVar("DatasetType")


class BaseBridge(Generic[DatasetType]):
    """Base class for transforming features in a dataset for ML pipelines.

    This bridge handles both forward transformation of features for
    model input and inverse transformation of predicted features back
    to their original format. It serves as a foundation for creating
    ML-ready datasets from PLAID datasets.

    The class is generic over DatasetType, allowing different implementations
    to specify their preferred dataset wrapper type.
    """

    def __init__(self, dataset_cls: Type[DatasetType]):
        """Initialize the BaseBridge with a dataset class.

        Args:
            dataset_cls: The class to use for wrapping transformed data.
        """
        self.dataset_cls = dataset_cls

    def transform(
        self, dataset: Dataset, features_ids: list[FeatureIdentifier]
    ) -> Union[Any, tuple[Any, ...]]:
        """Transform dataset features for model input.

        This method must be implemented by subclasses to define how
        features are transformed for model input.

        Args:
            dataset: The input dataset to transform.
            features_ids: List of feature identifiers to transform.

        Returns:
            Transformed features in a format suitable for ML models.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    def inverse_transform(
        self,
        features_ids: list[FeatureIdentifier],
        all_transformed_features: list[list[Any]],
    ) -> list[list[Feature]]:
        """Inverse transform predicted features to original format.

        Converts predicted features back to their original format.

        Args:
            features_ids: List of feature identifiers that were transformed.
            all_transformed_features: List of transformed features to convert back.

        Returns:
            List of features in their original format.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("This method must be implemented by subclasses")

    def convert(
        self, dataset: Dataset, *features_ids: list[FeatureIdentifier]
    ) -> DatasetType:
        """Convert a dataset into an ML-ready format.

        Transforms features from a dataset and wraps them in the specified
        dataset class for ML training/inference.

        Args:
            dataset: The input dataset to convert.
            features_ids: Feature identifier lists to transform.

        Returns:
            A dataset of type DatasetType containing the transformed features.
        """
        transformed_data = self.transform(dataset, *features_ids)

        return self.dataset_cls(transformed_data)

    def restore(
        self,
        dataset: Dataset,
        all_transformed_features: list[list[Any]],
        features_ids: list[FeatureIdentifier],
    ) -> Dataset:
        """Restore transformed features back to a dataset.

        Converts predicted features back to their original format and
        updates the dataset with these values.

        Args:
            dataset: The original dataset to update.
            all_transformed_features: List of transformed features to restore.
            features_ids: List of feature identifiers that were transformed.

        Returns:
            Updated dataset with restored features.
        """
        all_features = self.inverse_transform(features_ids, all_transformed_features)

        pred_features_dict: dict[int, list[Feature]] = {
            id: all_features[i] for i, id in enumerate(dataset.get_sample_ids())
        }

        return dataset.update_features_from_identifier(features_ids, pred_features_dict)
