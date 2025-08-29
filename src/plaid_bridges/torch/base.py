"""File implementing base TorchRegressionDatasets."""

from typing import Callable, List, Optional

import numpy as np
from plaid.containers.dataset import Dataset
from plaid.types import FeatureIdentifier

from plaid_bridges.common.base import BaseRegressionDataset


class HomogeneousDataset(BaseRegressionDataset):
    """HomogeneousDataset."""

    def __init__(
        self,
        dataset: Dataset,
        in_features_identifiers: List[FeatureIdentifier],
        out_features_identifiers: List[FeatureIdentifier],
        train: Optional[bool] = True,
        online_transform: Optional[Callable] = None,
    ):
        super().__init__(
            dataset,
            in_features_identifiers,
            out_features_identifiers,
            train,
            online_transform,
        )

        assert len(
            set([feat_id["type"] for feat_id in self.in_features_identifiers])
        ), "input features not of same type"
        assert len(
            set([feat_id["type"] for feat_id in self.out_features_identifiers])
        ), "input features not of same type"

        self.in_features = np.stack(
            [
                np.stack([feat for feat in sample_features])
                for sample_features in self.in_features
            ]
        )
        if train:
            self.out_features = np.stack(
                [
                    np.stack([feat for feat in sample_features])
                    for sample_features in self.out_features
                ]
            )

    @staticmethod
    def inverse_transform_single_feature(_feat_id, predicted_feature):
        """inverse_transform single feature."""
        return predicted_feature
