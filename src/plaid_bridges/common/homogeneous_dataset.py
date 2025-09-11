"""Implement the `HomogeneousDataset` class for regression problems."""

from typing import Optional

import numpy as np
from plaid.containers.dataset import Dataset
from plaid.types import Feature, FeatureIdentifier

from plaid_bridges.common.base_regression_dataset import (
    # BaseRegressionDataset,
    BaseTransformer,
    # feature_transform,
)


class HomogeneousOfflineTransformer(BaseTransformer):
    def __init__(
        self,
        in_features_identifiers: list[FeatureIdentifier],
        out_features_identifiers: list[FeatureIdentifier],
    ):
        super().__init__(
            in_features_identifiers,
            out_features_identifiers,
        )

        assert len(set([feat_id["type"] for feat_id in in_features_identifiers])), (
            "input features not of same type"
        )
        assert len(set([feat_id["type"] for feat_id in out_features_identifiers])), (
            "output features not of same type"
        )

    def transform(self, dataset: Dataset) -> tuple[np.ndarray, Optional[np.ndarray]]:
        in_features = np.stack(
            [
                [
                    (f if f is not None else 0.0)
                    for f in (
                        sample.get_feature_from_identifier(fid)
                        for fid in self.in_features_identifiers
                    )
                ]
                for sample in dataset
            ]
        )

        out_features = np.stack(
            [
                [
                    (f if f is not None else 0.0)
                    for f in (
                        sample.get_feature_from_identifier(fid)
                        for fid in self.out_features_identifiers
                    )
                ]
                for sample in dataset
            ]
        )

        return in_features, out_features

    @staticmethod
    def inverse_transform_single_feature(
        feat_id: FeatureIdentifier,  # noqa: ARG004  # ignore unused argument
        predicted_feature: Feature,
    ) -> Feature:
        """inverse_transform single feature."""
        return predicted_feature


# class HomogeneousDataset(BaseRegressionDataset):
#     """Implement a regression dataset for homogeneous data.

#     The input and output features can all be stacked into tensors.

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
#         online_transform: Optional[Any] = None,
#     ):
#         super().__init__(
#             dataset,
#             in_features_identifiers,
#             out_features_identifiers,
#             train,
#             online_transform,
#         )

#         assert len(
#             set([feat_id["type"] for feat_id in self.in_features_identifiers])
#         ), "input features not of same type"
#         assert len(
#             set([feat_id["type"] for feat_id in self.out_features_identifiers])
#         ), "input features not of same type"

#         self.in_features = np.stack(  # pyright: ignore[reportAttributeAccessIssue]  # overwritting self.in_features
#             [
#                 np.stack([feat for feat in sample_features])
#                 for sample_features in self.in_features
#             ]
#         )
#         if train:
#             self.out_features = np.stack(  # pyright: ignore[reportAttributeAccessIssue]  # overwritting self.in_features
#                 [
#                     np.stack([feat for feat in sample_features])
#                     for sample_features in self.out_features
#                 ]
#             )

#     @staticmethod
#     def inverse_transform_single_feature(
#         feat_id: FeatureIdentifier,  # noqa: ARG004  # ignore unused argument
#         predicted_feature: Feature,
#     ) -> Feature:
#         """inverse_transform single feature."""
#         return predicted_feature
