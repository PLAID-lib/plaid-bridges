"""Implement the `BaseRegressionDataset`."""

from typing import Any, Optional

from plaid.containers import Dataset
from plaid.types import Feature, FeatureIdentifier


class BaseTransformer:
    def __init__(
        self,
        in_features_identifiers: list[FeatureIdentifier],
        out_features_identifiers: list[FeatureIdentifier],
    ):
        self.in_features_identifiers = in_features_identifiers
        self.out_features_identifiers = out_features_identifiers

    def transform(self, dataset: Dataset) -> tuple[Any, Optional[Any]]:
        raise NotImplementedError("This method must be implemented by subclasses")

    @staticmethod
    def inverse_transform_single_feature(
        feat_id: FeatureIdentifier, predicted_feature: Any
    ) -> Feature:
        """Inverse_transform a single feature."""
        raise NotImplementedError("This method must be implemented by subclasses")

    def inverse_transform(
        self, dataset: Dataset, predicted_features: list[Any]
    ) -> Dataset:
        """Inverse_transform multiple features and returns them into a PLAID Dataset."""
        assert len(predicted_features) == len(dataset)

        pred_features_dict: dict[int, list[Feature]] = {
            id: [] for id in dataset.get_sample_ids()
        }

        for i, id in enumerate(dataset.get_sample_ids()):
            for j, feat_id in enumerate(self.out_features_identifiers):
                pred_features_dict[id].append(
                    self.inverse_transform_single_feature(
                        feat_id, predicted_features[i][j]
                    )
                )

        return dataset.update_features_from_identifier(
            self.out_features_identifiers, pred_features_dict
        )  # pragma: no cover

    def __str__(self) -> str:
        """Function to return synthetic description of the BaseTransformer."""
        return (
            f"Transformer ({len(self.in_features_identifiers)} input features, "
            + f"{len(self.out_features_identifiers)} output features)"
        )

    def show_details(self) -> str:
        """Function to return extended description of the BaseTransformer."""
        return (
            "Transformer: input features="
            + str(
                [
                    f"{feat['name']} ({feat['type']})"
                    for feat in self.in_features_identifiers
                ]
            )
            + ", output features="
            + str(
                [
                    f"{feat['name']} ({feat['type']})"
                    for feat in self.out_features_identifiers
                ]
            )
        )


class BaseRegressionDataset:
    def __init__(
        self,
        dataset: Dataset,
        offline_transformer: BaseTransformer,
        online_transformer: Optional[BaseTransformer] = None,
    ):
        self.dataset = dataset
        self.online_transformer = online_transformer

        self.inputs, self.outputs = offline_transformer.transform(self.dataset)

    def __getitem__(self, id) -> tuple[Any, Optional[Any]]:
        if self.online_transformer:
            return self.online_transformer.transform(
                self.inputs[id], self.outputs[id]
            )  # pragma: no cover
        else:
            return self.inputs[id], self.outputs[id]

    def __len__(self) -> int:
        """Returns length of BaseRegressionDataset."""
        return len(self.dataset)

    def __str__(self) -> str:
        """Function to return synthetic description of the BaseRegressionDataset."""
        return f"RegressionDataset, initialized from plaid:{self.dataset}"


# try:
#     from typing import TypeAlias  # Python 3.10+
# except ImportError:  # pragma: no cover
#     from typing_extensions import TypeAlias
# from typing import Callable
# from plaid.containers import Dataset, Sample
# from plaid.types import Feature, FeatureIdentifier
# feature_transform: TypeAlias = Callable[
#     [list[Feature], list[Feature]], tuple[list[Feature], list[Feature]]
# ]

# class BaseRegressionDataset:
#     """BaseRegressionDataset that provides minimal accessors and transforms mechanics.

#     Args:
#         dataset (Dataset): PLAID dataset.
#         in_feature_identifiers (list[FeatureIdentifier]): List of input feature identifiers.
#         out_feature_identifiers (list[FeatureIdentifier]): List of output feature identifiers.
#         train (bool, optional): if True, out_features are initialized for the later regressor fit.
#         online_transform (featuture_transform, optional): Transformation applied to the samples through the `__getitem__` function.
#     """

#     def __init__(
#         self,
#         dataset: Dataset,
#         in_features_identifiers: list[FeatureIdentifier],
#         out_features_identifiers: list[FeatureIdentifier],
#         train: Optional[bool] = True,
#         online_transform: Optional[feature_transform] = None,
#     ):
#         self.dataset = dataset
#         self.in_features_identifiers = in_features_identifiers
#         self.out_features_identifiers = out_features_identifiers
#         self.train = train
#         self.online_transform = online_transform

#         self.in_features: list[list[Feature]] = [[] for _ in dataset]
#         if train:
#             self.out_features: list[list[Feature]] = [[] for _ in dataset]

#         for i, sample in enumerate(dataset):
#             for _, feat_id in enumerate(self.in_features_identifiers):
#                 self.in_features[i].append(sample.get_feature_from_identifier(feat_id))
#             if train:
#                 for _, feat_id in enumerate(self.out_features_identifiers):
#                     self.out_features[i].append(
#                         sample.get_feature_from_identifier(feat_id)
#                     )

#     def __len__(self) -> int:
#         """Returns length of BaseRegressionDataset."""
#         return len(self.dataset)

#     def __str__(self) -> str:
#         """Function to return synthetic description of the BaseRegressionDataset."""
#         return f"RegressionDataset ({len(self)} sample, {len(self.in_features_identifiers)} input features, {len(self.out_features_identifiers)}) output features)"

#     def __getitem__(self, index: int) -> tuple[list[Feature], list[Feature]]:
#         """Retrieve an element from the dataset.

#         Applies `online_transform` if defined. Returns a tuple `(input, target)` during training,
#         or just `input` during evaluation.
#         """
#         # Get input features
#         in_feat = self.in_features[index]
#         if self.online_transform:
#             in_feat = self.online_transform(in_feat)

#         # Get output features if training
#         if self.train:
#             out_feat = self.out_features[index]
#             if self.online_transform:
#                 out_feat = self.online_transform(out_feat)
#             return in_feat, out_feat

#         return in_feat

#     @staticmethod
#     def inverse_transform_single_feature(
#         feat_id: FeatureIdentifier, predicted_feature: Feature
#     ) -> Feature:
#         """Inverse_transform a single feature."""
#         raise NotImplementedError(
#             "inverse_transform_single_feature not implemented in base class."
#         )

#     def inverse_transform(self, predictions) -> Dataset:
#         """Inverse_transform multiple features and returns them into a PLAID Dataset."""
#         assert len(predictions) == len(self.dataset)
#         pred_features_dict: dict[int, list[Feature]] = {
#             id: [] for id in self.dataset.get_sample_ids()
#         }

#         for i, id in enumerate(self.dataset.get_sample_ids()):
#             for j, feat_id in enumerate(self.out_features_identifiers):
#                 pred_features_dict[id].append(
#                     self.inverse_transform_single_feature(feat_id, predictions[i][j])
#                 )

#         return self.dataset.update_features_from_identifier(
#             self.out_features_identifiers, pred_features_dict
#         )  # pragma: no cover

#     def show_details(self):
#         """Function to return details on the BaseRegressionDataset."""
#         print(self)
#         print(
#             "Input features : "
#             + str(
#                 [
#                     f"{feat['name']} ({feat['type']})"
#                     for feat in self.in_features_identifiers
#                 ]
#             )
#         )
#         print(
#             "Output features: "
#             + str(
#                 [
#                     f"{feat['name']} ({feat['type']})"
#                     for feat in self.out_features_identifiers
#                 ]
#             )
#         )
