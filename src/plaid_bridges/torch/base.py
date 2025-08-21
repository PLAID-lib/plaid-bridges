from typing import List, Optional, Callable

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.types import FeatureIdentifier, FeatureType
import torch
from plaid_bridges.common.base import BaseRegressionDataset
import numpy as np


class HomogeneousDataset(BaseRegressionDataset):

    def __init__(
        self,
        dataset:Dataset,
        in_feature_identifiers: List[FeatureIdentifier],
        out_feature_identifiers: List[FeatureIdentifier],
        online_transform: Optional[Callable] = None,
    ):
        super().__init__(
            dataset,
            in_feature_identifiers,
            out_feature_identifiers,
            online_transform)


        assert len(set([feat_id['type'] for feat_id in self.in_feature_identifiers])), "input features not of same type"
        assert len(set([feat_id['type'] for feat_id in self.out_feature_identifiers])), "input features not of same type"

        self.in_features = np.stack([np.stack([feat for feat in sample_features]) for sample_features in self.in_features])
        self.out_features = np.stack([np.stack([feat for feat in sample_features]) for sample_features in self.out_features])

    @staticmethod
    def inverse_transform_single_feature(feat_id, predicted_feature):
        """inverse_transform single feature."""
        return predicted_feature