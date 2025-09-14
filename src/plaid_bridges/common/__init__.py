"""Common base classes and transformers for PLAID bridges.

This module provides the foundational classes and generic transformers that serve as
the building blocks for all PLAID bridges. It includes base classes for creating
ML-ready datasets and handling feature transformations, as well as implementations
for homogeneous data types.

The module exports:

- BaseBridge: Abstract base class for all bridge implementations

- ArrayDataset: Wrapper class for handling data in ML workflows

- HomogeneousBridge: Transformer for datasets with features of the same type
"""

from plaid_bridges.common.base import (
    BaseBridge,
)
from plaid_bridges.common.homogeneous import ArrayDataset, HomogeneousBridge

__all__ = ["BaseBridge", "ArrayDataset", "HomogeneousBridge"]
