from typing import List, Optional, Callable

from plaid.containers.dataset import Dataset
from plaid.types import FeatureIdentifier, FeatureType
import numpy as np


class BaseRegressionDataset:
    def __init__(
        self,
        dataset:Dataset,
        in_feature_identifiers: List[FeatureIdentifier],
        out_feature_identifiers: List[FeatureIdentifier],
        online_transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.in_feature_identifiers = in_feature_identifiers
        self.out_feature_identifiers = out_feature_identifiers
        self.online_transform = online_transform

        self.in_features = [[] for _ in self.in_feature_identifiers]
        self.out_features = [[] for _ in self.out_feature_identifiers]

        for sample in dataset:
            for j, feat_id in enumerate(self.in_feature_identifiers):
                self.in_features[j].append(sample.get_feature_from_identifier(feat_id))
            for j, feat_id in enumerate(self.out_feature_identifiers):
                self.out_features[j].append(sample.get_feature_from_identifier(feat_id))

    @staticmethod
    def _can_stack_features(features: list[FeatureType]):
        """Check if the feature for all sample have same size."""
        shapes = [() if np.isscalar(obj) else np.shape(obj) for obj in features]
        return len(set(shapes)) == 1

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        if self.online_transform:
            return self.online_transform(self.in_features[index], self.out_features[index])
        else:
            return self.in_features[index], self.out_features[index]


    @staticmethod
    def inverse_transform_single_feature(feat_id, predicted_feature):
        """inverse_transform single feature."""
        raise NotImplementedError("inverse_transform_single_feature not implemented in base class.")

    def inverse_transform(self, predictions):
        """inverse_transform."""

        assert len(predictions) == len(self.dataset)
        pred_features_dict = {id: [] for id in self.dataset.get_sample_ids()}

        for i, id in enumerate(self.dataset.get_sample_ids()):
            for j, feat_id in enumerate(self.out_feature_identifiers):
                pred_features_dict[id].append(self.inverse_transform_single_feature(feat_id, predictions[i,j]))

        return self.dataset.update_features_from_identifier(
            self.out_feature_identifiers, pred_features_dict
        )
