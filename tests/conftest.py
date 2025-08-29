"""This file defines shared pytest fixtures and test configurations."""

from typing import List, Tuple

import numpy as np
import pytest
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools import MeshCreationTools as MCT
from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.types import CGNSTree


@pytest.fixture()
def nodes():
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 1.5],
        ]
    )


@pytest.fixture()
def triangles():
    return np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [2, 4, 3],
        ]
    )


@pytest.fixture()
def mesh(nodes, triangles):
    mesh = MCT.CreateMeshOfTriangles(nodes, triangles)
    mesh.nodeFields["in_field"] = np.arange(5)
    mesh.nodeFields["out_field"] = 1 + np.arange(5)
    return mesh


@pytest.fixture()
def tree(mesh):
    return MeshToCGNS(mesh, exportOriginalIDs=False)


@pytest.fixture()
def sample():
    return Sample()


@pytest.fixture()
def sample_with_tree(tree: CGNSTree) -> Sample:
    """Generate a Sample objects with a tree."""
    sample = Sample()
    sample.add_tree(tree)
    sample.add_scalar("in_scalar", 1.0)
    sample.add_scalar("out_scalar", 2.0)
    sample.add_time_series("b", [0.0, 1.0], [3.0, 4.0])
    return sample


@pytest.fixture()
def dataset(sample_with_tree: Sample) -> Dataset:
    """Generate a Sample objects with a tree."""
    return Dataset.from_list_of_samples([sample_with_tree, sample_with_tree])


@pytest.fixture()
def scalar_features(dataset: Dataset) -> List:
    all_feat_ids = dataset.get_all_features_identifiers()
    scalar_features = [f for f in all_feat_ids if "scalar" in f.values()]
    scalar_features.sort(key=lambda f: f["name"])
    return scalar_features


@pytest.fixture()
def field_features(dataset: Dataset) -> List:
    all_feat_ids = dataset.get_all_features_identifiers()
    field_features = [f for f in all_feat_ids if "field" in f.values()]
    field_features.sort(key=lambda f: f["name"])
    return field_features


@pytest.fixture()
def in_out_features(scalar_features: List, field_features: List) -> Tuple:
    in_features_identifiers = [scalar_features[0], field_features[0]]
    out_features_identifiers = [field_features[1], scalar_features[1]]
    return (in_features_identifiers, out_features_identifiers)
