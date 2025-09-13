"""PyTorch-specific bridges for PLAID datasets.

This module provides bridge implementations specifically designed for PyTorch workflows.
It includes transformers that convert PLAID datasets into PyTorch tensors, enabling
seamless integration with PyTorch models and training pipelines.

The module exports:
- GridFieldsAndScalarsBridge: Transformer for handling grid-based field data and scalar features
"""

from plaid_bridges.torch.grid_fields_and_scalars import (
    GridFieldsAndScalarsBridge,
)
from plaid_bridges.torch.pyg import (
    PyGBridge,
)

__all__ = ["GridFieldsAndScalarsBridge", "PyGBridge"]
