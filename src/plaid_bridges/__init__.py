"""Module that implements PLAID bridges.

This package provides a collection of bridge classes that transform PLAID datasets
into formats suitable for machine learning frameworks. The bridges handle the
conversion of heterogeneous data types (scalars, fields, grids) into tensors
compatible with popular ML libraries like PyTorch and NumPy.

The module is organized into:

- common: Base classes and generic transformers for homogeneous data

- torch: PyTorch-specific bridges for deep learning applications

Each bridge implements forward transformation for model input and inverse
transformation for converting predictions back to the original data format.
"""

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "None"
