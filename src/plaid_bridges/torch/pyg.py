# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""PyGBridge: Transforming PLAID datasets for PyTorch Geometric (PyG).

This module implements the `PyGBridge` class, which enables conversion of PLAID
datasets into PyTorch Geometric (PyG) Data objects. It supports both field and
scalar features, handling mesh connectivity and node positions for graph-based
deep learning workflows. The bridge also provides inverse transformation to
convert model predictions back to their original PLAID feature format.

Utility functions for mesh and field visualization using matplotlib are included.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from plaid.containers.dataset import Dataset
from plaid.types import Feature, FeatureIdentifier
from torch_geometric.data import Data

from plaid_bridges.common import BaseBridge


class PyGBridge(BaseBridge):
    """Bridge for transforming PLAID datasets into PyTorch Geometric Data objects.

    This bridge processes datasets containing field and scalar features, converting
    them into PyG Data objects suitable for graph neural network models. It handles
    mesh connectivity, node positions, and feature constraints based on base and
    zone names. The bridge also supports inverse transformation of predictions.
    """

    def __init__(
        self,
        base_name: Optional[str] = None,
        zone_name: Optional[str] = None,
    ):
        """Initialize the PyGBridge.

        Args:
            base_name: Optional name of the mesh base to filter features.
            zone_name: Optional name of the mesh zone to filter features.
        """
        self.base_name = base_name
        self.zone_name = zone_name
        super().__init__(
            list
        )  # PyG Dataloader can be called on a list of PyG Data directly

    def _check_base_zone_constaints(
        self, feat_id: FeatureIdentifier, default_base: str, default_zone: str
    ) -> None:
        """Check that feature base and zone names match bridge constraints.

        Args:
            feat_id: The feature identifier to check.
            default_base: The default base name for the sample.
            default_zone: The default zone name for the sample.

        Raises:
            AssertionError: If the feature's base or zone name does not match
                the bridge's constraints or the sample's defaults.
        """
        # Check base_name constraints
        if "base_name" in feat_id:
            if self.base_name is not None:
                assert feat_id["base_name"] == self.base_name, (
                    f"Feature base_name '{feat_id['base_name']}' does not match "
                    f"requested base_name '{self.base_name}'"
                )
            else:
                assert feat_id["base_name"] in [None, default_base], (
                    f"Feature base_name '{feat_id['base_name']}' is not compatible "
                    f"with default base '{default_base}'"
                )
        # Check zone_name constraints
        if "zone_name" in feat_id:
            if self.zone_name is not None:
                assert feat_id["zone_name"] == self.zone_name, (
                    f"Feature zone_name '{feat_id['zone_name']}' does not match "
                    f"requested zone_name '{self.zone_name}'"
                )
            else:
                assert feat_id["zone_name"] in [None, default_zone], (
                    f"Feature zone_name '{feat_id['zone_name']}' is not compatible "
                    f"with default zone '{default_zone}'"
                )

    def transform(
        self, dataset: Dataset, features_ids: list[FeatureIdentifier]
    ) -> list[Data]:
        """Transform a PLAID dataset into a list of PyG Data objects.

        For each sample in the dataset, this method extracts field and scalar features,
        assembles node positions, computes mesh connectivity (edges), and packages
        everything into a PyG Data object.

        Args:
            dataset: The input PLAID dataset to transform.
            features_ids: List of feature identifiers to extract and transform.

        Returns:
            A list of PyG Data objects, one per sample in the dataset.
        """
        data_list = []
        for sample in dataset:
            fields = []
            field_names = []
            scalars = []
            scalar_names = []

            default_base = sample.meshes.get_base_assignment(self.base_name)
            default_zone = sample.meshes.get_zone_assignment(
                self.zone_name, self.base_name
            )

            for feat_id in features_ids:
                feature = sample.get_feature_from_identifier(feat_id)
                assert feature is not None, f"retrieved field from {feat_id} is None"

                if feat_id["type"] == "field":
                    self._check_base_zone_constaints(
                        feat_id, default_base, default_zone
                    )
                    fields.append(feature)
                    field_names.append(feat_id["name"])
                if feat_id["type"] == "scalar":
                    scalars.append(feature)
                    scalar_names.append(feat_id["name"])

            fields = torch.tensor(np.array(fields).T)
            scalars = torch.tensor(np.array(scalars))

            nodes = torch.tensor(
                sample.meshes.get_nodes(self.zone_name, self.base_name)
            )

            elements = sample.meshes.get_elements(self.zone_name, self.base_name)
            edges = []
            for element_array in elements.values():
                element_array = np.array(element_array)

                # Efficiently create all edges for all elements using broadcasting
                from_nodes = element_array
                to_nodes = np.roll(element_array, -1, axis=1)
                element_edges = np.stack([from_nodes, to_nodes], axis=-1)
                element_edges = element_edges.reshape(-1, 2)
                edges.extend(element_edges.tolist())

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

            data_list.append(
                Data(
                    x=fields,
                    pos=nodes,
                    edge_index=edge_index,
                    scalars=scalars,
                    field_names=field_names,
                    scalar_names=scalar_names,
                )
            )

        return data_list

    def inverse_transform(
        self,
        features_ids: list[FeatureIdentifier],
        all_transformed_features: list[list[np.ndarray]],
    ) -> list[list[Feature]]:
        """Inverse transform PyG model outputs to original PLAID feature format.

        Converts predicted PyTorch tensors back to their original feature format.
        Scalar features are converted to floats; field features are converted to numpy arrays.

        Args:
            features_ids: List of feature identifiers that were transformed.
            all_transformed_features: List of lists of PyTorch tensors to convert back.

        Returns:
            A list of lists of features in their original PLAID format.

        Raises:
            Exception: If a feature type is not supported by this transformer.
        """
        all_features = []
        for transformed_features in all_transformed_features:
            features = []
            for feat_id, transf_feature in zip(features_ids, transformed_features):
                assert isinstance(transf_feature, torch.Tensor)
                _type = feat_id["type"]
                if _type == "scalar":
                    feature = float(transf_feature)
                elif _type == "field":
                    feature = transf_feature.numpy()
                else:
                    raise Exception(
                        f"feature type {_type} not compatible with `prediction_to_structured_grid`"
                    )  # pragma: no cover
                features.append(feature)
            all_features.append(features)

        return all_features


def plot_sample_mesh(pyg_sample: Data, block: bool = True):
    """Visualize the mesh connectivity of a PyG Data sample.

    Args:
        pyg_sample: The PyG Data object containing node positions and edge indices.
        block: Whether to block execution until the plot window is closed.
    """
    pos = pyg_sample.pos.numpy()
    edges = pyg_sample.edge_index.t().numpy()

    segments = pos[edges]
    lc = LineCollection(segments, colors="k", linewidths=0.2, alpha=0.5)

    _, ax = plt.subplots(figsize=(12, 12))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")

    plt.show(block=block)


def plot_sample_field(pyg_sample: Data, field_name: str, block: bool = True):
    """Visualize a field feature on the mesh nodes of a PyG Data sample.

    Args:
        pyg_sample: The PyG Data object containing node positions and field features.
        field_name: The name of the field feature to visualize.
        block: Whether to block execution until the plot window is closed.
    """
    pos = pyg_sample.pos.numpy()

    field_rank = pyg_sample.field_names.index(field_name)
    values = pyg_sample.x[:, field_rank]

    _, ax = plt.subplots(figsize=(12, 12))

    sc = ax.scatter(pos[:, 0], pos[:, 1], c=values, cmap="viridis", s=2, zorder=2)

    plt.colorbar(sc, ax=ax, label=f"field_{field_name}")
    ax.autoscale()
    ax.set_aspect("equal")

    plt.show(block=block)
