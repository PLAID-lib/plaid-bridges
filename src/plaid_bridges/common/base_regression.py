"""Implement `BaseTransformer` and `BaseRegressionDataset`."""

from typing import Any, Optional, Union

from plaid.containers import Dataset
from plaid.types import Feature, FeatureIdentifier


class BaseTransformer:
    """BaseTransformer."""

    def __init__(
        self,
        features_identifiers: list[FeatureIdentifier],
    ):
        self.features_identifiers = features_identifiers

    def transform(self, dataset: Dataset) -> tuple[Any, Optional[Any]]:
        """Transform."""
        raise NotImplementedError("This method must be implemented by subclasses")

    @staticmethod
    def inverse_transform_single_feature(
        feat_id: FeatureIdentifier, predicted_feature: Any
    ) -> Feature:
        """Inverse_transform a single feature."""
        raise NotImplementedError("This method must be implemented by subclasses")

    def inverse_transform(
        self,
        dataset: Dataset,
        predicted_features: list[Any],
    ) -> Dataset:
        """Inverse_transform multiple features and returns them into a PLAID Dataset."""
        assert len(predicted_features) == len(dataset)

        pred_features_dict: dict[int, list[Feature]] = {
            id: [] for id in dataset.get_sample_ids()
        }

        for i, id in enumerate(dataset.get_sample_ids()):
            for j, feat_id in enumerate(self.features_identifiers):
                pred_features_dict[id].append(
                    self.inverse_transform_single_feature(
                        feat_id, predicted_features[i][j]
                    )
                )

        return dataset.update_features_from_identifier(
            self.features_identifiers, pred_features_dict
        )

    def __str__(self) -> str:
        """Function to return synthetic description of the BaseTransformer."""
        return f"Transformer ({len(self.features_identifiers)} input features)"

    def show_details(self) -> str:
        """Function to return extended description of the BaseTransformer."""
        return "Transformer: features=" + str(
            [f"{feat['name']} ({feat['type']})" for feat in self.features_identifiers]
        )


class BaseRegressionDataset:
    """BaseRegressionDataset."""

    def __init__(
        self,
        dataset: Dataset,
        offline_in_transformer: BaseTransformer,
        offline_out_transformer: Optional[BaseTransformer] = None,
    ):
        self.dataset = dataset

        self.inputs = offline_in_transformer.transform(self.dataset)

        if offline_out_transformer:
            self.outputs = offline_out_transformer.transform(self.dataset)
            self.has_outputs = True
        else:
            self.has_outputs = False

    def __getitem__(self, id) -> Union[Any, tuple[Any, Any]]:
        """Get item."""
        if self.has_outputs:
            return self.inputs[id], self.outputs[id]
        else:
            return self.inputs[id]

    def __len__(self) -> int:
        """Returns length of BaseRegressionDataset."""
        return len(self.dataset)

    def __str__(self) -> str:
        """Function to return synthetic description of the BaseRegressionDataset."""
        return f"RegressionDataset, initialized from plaid:{self.dataset}"
