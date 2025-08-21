from typing import List, Optional, Callable

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.types import FeatureIdentifier, FeatureType
import torch
from plaid_bridges.common.base import BaseRegressionDataset



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

        for i, _ in enumerate(self.in_feature_identifiers):
            assert self._can_stack_features(self.in_features[i]), (
                f"features {self.in_feature_identifiers[i]} are of different sizes in batch"
            )
            self.in_features[i] = torch.stack(
                [torch.as_tensor(feat) for feat in self.in_features[i]]
            )

        for i, _ in enumerate(self.out_feature_identifiers):
            assert self._can_stack_features(self.out_features[i]), (
                f"features {self.out_feature_identifiers[i]} are of different sizes in batch"
            )
            self.out_features[i] = torch.stack(
                [torch.as_tensor(feat) for feat in self.out_features[i]]
            )
