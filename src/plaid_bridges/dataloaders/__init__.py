"""Module that implements data loaders."""

from plaid_bridges.dataloaders.generic import (
    PlaidDataLoader,
    PlaidSampleDataLoader,
)

# from plaid_bridges.dataloaders.torch import (
#     TorchTensorDataLoader,
# )


__all__ = [
    "PlaidSampleDataLoader",
    "PlaidDataLoader",
    # "TorchTensorDataLoader",
]
