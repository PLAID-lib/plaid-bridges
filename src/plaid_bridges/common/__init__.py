"""Common base classes and transformers for PLAID bridges.

This module provides the foundational classes and generic transformers that serve as
the building blocks for all PLAID bridges. It includes base classes for creating
ML-ready datasets and handling feature transformations, as well as implementations
for homogeneous data types.

The module exports:
- BaseBridge: Abstract base class for all bridge implementations
- MLDataset: Wrapper class for handling data in ML workflows
- HomogeneousBridge: Transformer for datasets with features of the same type
"""

from plaid_bridges.common.base_regression import (
    BaseBridge,
    MLDataset,
)
from plaid_bridges.common.homogeneous import HomogeneousBridge

__all__ = ["MLDataset", "BaseBridge", "HomogeneousBridge"]
