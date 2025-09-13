"""Implement `BaseTransformer` and `BaseRegressionDataset`."""

from typing import Any

from plaid.containers import Dataset
from plaid.types import Feature, FeatureIdentifier


class MLDataset:  # add online_transform later ?
    """MLDataset."""

    all_data: tuple[Any, ...]

    def __init__(self, *all_data: Any) -> None:
        self.all_data = all_data

    def __getitem__(self, index: int) -> tuple[Any]:
        """Get item."""
        return tuple(data[index] for data in self.all_data)

    def __len__(self) -> int:
        """Returns length of MLDataset."""
        return len(self.all_data[0])


class BaseBridge:
    """BaseTransformer."""

    def transform(self, dataset: Dataset, features_ids: list[FeatureIdentifier]) -> Any:
        """Transform."""
        raise NotImplementedError("This method must be implemented by subclasses")

    def inverse_transform(
        self,
        features_ids: list[FeatureIdentifier],
        all_transformed_features: list[list[Any]],
    ) -> list[list[Feature]]:
        """Inverse transform."""
        raise NotImplementedError("This method must be implemented by subclasses")

    def convert(
        self, dataset: Dataset, features_ids_list: list[list[FeatureIdentifier]]
    ) -> MLDataset:
        """Convert."""
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
        """Restore."""
        all_features = self.inverse_transform(features_ids, all_transformed_features)

        pred_features_dict: dict[int, list[Feature]] = {
            id: all_features[i] for i, id in enumerate(dataset.get_sample_ids())
        }

        return dataset.update_features_from_identifier(features_ids, pred_features_dict)
