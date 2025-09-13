"""Implement `BaseBridge` for transforming datasets in ML pipelines.

This module provides base classes for transforming datasets and creating
ML-ready datasets. The BaseBridge handles feature transformation
and inverse transformation for machine learning workflows.
"""

from typing import Any

from plaid.containers import Dataset
from plaid.types import Feature, FeatureIdentifier


class MLDataset:
    """Machine Learning Dataset wrapper for handling multiple data sources.

    This class wraps multiple data arrays and provides a unified interface
    for accessing samples across all data sources by index. It's designed to
    work with transformed features from the BaseBridge class.
    """

    all_data: tuple[Any, ...]

    def __init__(self, *all_data: Any) -> None:
        """Initialize the MLDataset with multiple data sources.

        Args:
            *all_data: Variable number of data arrays/tensors to be wrapped.
                      All data sources must have the same length.
        """
        self.all_data = all_data

    def __getitem__(self, index: int) -> tuple[Any, ...]:
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


class BaseBridge:
    """Base class for transforming features in a dataset for ML pipelines.

    This bridge handles both forward transformation of features for
    model input and inverse transformation of predicted features back
    to their original format. It serves as a foundation for creating
    ML-ready datasets from PLAID datasets.
    """

    def transform(self, dataset: Dataset, features_ids: list[FeatureIdentifier]) -> Any:
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
        self, dataset: Dataset, features_ids_list: list[list[FeatureIdentifier]]
    ) -> MLDataset:
        """Convert a dataset into an ML-ready format.

        Transforms multiple sets of features from a dataset and wraps
        them in an MLDataset for ML training/inference.

        Args:
            dataset: The input dataset to convert.
            features_ids_list: List of feature identifier lists to transform.

        Returns:
            An MLDataset containing the transformed features.
        """
        transf_data_list = tuple(
            self.transform(dataset, feat_ids) for feat_ids in features_ids_list
        )

        return MLDataset(*transf_data_list)

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
