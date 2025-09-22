# Quickstart

Everything you need to start using plaid-bridges and contributing effectively.

## 1 Using the library

First create an environment using conda/mamba:

```bash
mamba create -n plaid-bridges python=3.11 hdf5 "vtk>=9.4" pycgns-core muscat-core=2.5 -c conda-forge
mamba activate plaid-bridges
```

### User mode

Relies on the published PyPi package:

```bash
pip install plaid-bridges
```

### Developper mode

Installing from the sources:

```bash
pip install -e .[dev]
```

Note: this will install the last stable versions of PLAID and plaid-ops.

## 2 Core concepts

plaid-bridges provides high-level adapters that connect PLAID datasets with popular machine learning ecosystems. These adapters:

- Enable seamless conversion from structured PLAID data to ML-ready inputs, and back to engineering-friendly outputs.
- Support both grid-like fields and simple scalars, covering common physics/engineering use cases.
- Preserve mesh and geometry context where relevant, so downstream models can leverage structure.
- Offer a consistent workflow to prepare data for training/inference and to post-process model predictions.
- Reduce boilerplate when moving between PLAID and ML frameworks, keeping pipelines clear and maintainable.

## 3 Going further

See the documentation for a concise getting started guide and end-to-end examples:
- [Examples & Tutorials](https://plaid-bridges.readthedocs.io/en/latest/source/notebooks.html)
- [API reference](https://plaid-bridges.readthedocs.io/en/latest/autoapi/plaid_bridges/index.html)