# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Implementation of `PyGBridge`.

This module provides a bridge to PyG.
"""

import numpy as np
import torch
from plaid.containers.dataset import Dataset
from plaid.types import FeatureIdentifier
from torch_geometric.data import Data

from plaid_bridges.common import BaseBridge


class PyGBridge(BaseBridge):
    """PyGBridge."""

    def __init__(self):
        super().__init__(
            list
        )  # PyG Dataloader can be called on a list of PyG Data directly

    def transform(
        self, dataset: Dataset, features_ids: list[FeatureIdentifier]
    ) -> list[Data]:
        """Transform."""
        data_list = []
        for sample in dataset:
            fields = []
            for feat_id in features_ids:
                if feat_id["type"] == "field":
                    fields.append(sample.get_feature_from_identifier(feat_id))
            fields = np.array(fields).T
            data_list.append(Data(x=torch.Tensor(fields)))

        return data_list

    # def inverse_transform(
    #     self,
    #     features_ids: list[FeatureIdentifier],
    #     all_transformed_features: list[list[torch.Tensor]],
    # ) -> list[list[Feature]]:
    #     all_features = []
    #     for transformed_features in all_transformed_features:
    #         features = []
    #         for feat_id, transf_feature in zip(features_ids, transformed_features):
    #             assert isinstance(transf_feature, torch.Tensor)
    #             _type = feat_id["type"]
    #             if _type == "scalar":
    #                 feature = np.mean(transf_feature.numpy())
    #             elif _type == "field":
    #                 feature = transf_feature.numpy().flatten()
    #             else:
    #                 raise Exception(
    #                     f"feature type {_type} not compatible with `prediction_to_structured_grid`"
    #                 )  # pragma: no cover
    #             features.append(feature)
    #         all_features.append(features)

    #     return all_features
