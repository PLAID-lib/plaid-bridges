"""File implementing aseRegressionDataset."""

from typing import Callable, Optional

from plaid.containers.dataset import Dataset
from plaid.types import FeatureIdentifier, FeatureType


class BaseRegressionDataset:
    """BaseRegressionDataset that provides minimal accessors and transforms mechanics."""

    def __init__(
        self,
        dataset: Dataset,
        in_feature_identifiers: list[FeatureIdentifier],
        out_feature_identifiers: list[FeatureIdentifier],
        online_transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.in_feature_identifiers = in_feature_identifiers
        self.out_feature_identifiers = out_feature_identifiers
        self.online_transform = online_transform

        self.in_features: list[list[FeatureType]] = [[] for _ in dataset]
        self.out_features: list[list[FeatureType]] = [[] for _ in dataset]

        for i, sample in enumerate(dataset):
            for _, feat_id in enumerate(self.in_feature_identifiers):
                self.in_features[i].append(sample.get_feature_from_identifier(feat_id))
            for _, feat_id in enumerate(self.out_feature_identifiers):
                self.out_features[i].append(sample.get_feature_from_identifier(feat_id))

    def __len__(self) -> int:
        """Returns length of BaseRegressionDataset."""
        return len(self.dataset)

    def __str__(self):
        """Function to return synthetic description of the BaseRegressionDataset."""
        return f"RegressionDataset ({len(self)} sample, {len(self.in_feature_identifiers)} input features, {len(self.out_feature_identifiers)}) output features)"

    def __getitem__(self, index):
        """Retrieves indexed element of BaseRegressionDataset."""
        if self.online_transform:
            return self.online_transform(
                self.in_features[index], self.out_features[index]
            )
        else:
            return self.in_features[index], self.out_features[index]

    @staticmethod
    def inverse_transform_single_feature(feat_id, predicted_feature):
        """inverse_transform single feature."""
        raise NotImplementedError(
            "inverse_transform_single_feature not implemented in base class."
        )

    def inverse_transform(self, predictions):
        """inverse_transform."""
        assert len(predictions) == len(self.dataset)
        pred_features_dict = {id: [] for id in self.dataset.get_sample_ids()}

        for i, id in enumerate(self.dataset.get_sample_ids()):
            for j, feat_id in enumerate(self.out_feature_identifiers):
                pred_features_dict[id].append(
                    self.inverse_transform_single_feature(feat_id, predictions[i][j])
                )

        return self.dataset.update_features_from_identifier(
            self.out_feature_identifiers, pred_features_dict
        )  # pragma: no cover

    def show_details(self):
        """Function to return details on the BaseRegressionDataset."""
        print(self)
        print(
            "Input features : "
            + str(
                [
                    f"{feat['name']} ({feat['type']})"
                    for feat in self.in_feature_identifiers
                ]
            )
        )
        print(
            "Output features: "
            + str(
                [
                    f"{feat['name']} ({feat['type']})"
                    for feat in self.out_feature_identifiers
                ]
            )
        )
