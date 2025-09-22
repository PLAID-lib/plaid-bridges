<div align="center">
<img src="https://plaid-lib.github.io/assets/images/plaid-bridges-logo.png" width="300">
</div>

| | |
| --- | --- |
| Testing | [![CI Status](https://github.com/PLAID-lib/plaid-bridges/actions/workflows/testing.yml/badge.svg)](https://github.com/PLAID-lib/plaid-bridges/actions/workflows/testing.yml) [![Documentation Status](https://readthedocs.org/projects/plaid-bridges/badge/?version=latest)](https://plaid-bridges.readthedocs.io/en/latest/?badge=latest) [![Coverage](https://codecov.io/gh/plaid-lib/plaid-bridges/branch/main/graph/badge.svg)](https://app.codecov.io/gh/plaid-lib/plaid-bridges/tree/main?search=&displayType=list) ![Last Commit](https://img.shields.io/github/last-commit/PLAID-lib/plaid-bridges/main) |
| Package | [![PyPI Latest Release](https://img.shields.io/pypi/v/plaid-bridges.svg)](https://pypi.org/project/plaid-bridges/) [![PyPI Downloads](https://static.pepy.tech/badge/plaid-bridges)](https://pepy.tech/projects/plaid-bridges) ![Platform](https://img.shields.io/badge/platform-any-blue) ![Python Version](https://img.shields.io/pypi/pyversions/plaid-bridges)  |
| Other | [![License - BSD 3-Clause](https://anaconda.org/conda-forge/plaid/badges/license.svg)](https://github.com/PLAID-lib/plaid-bridges/blob/main/LICENSE.txt) ![GitHub stars](https://img.shields.io/github/stars/PLAID-lib/plaid-bridges?style=social)|


# plaid-bridges

High-level adapters that connect PLAID datasets with popular machine learning ecosystems.


> [!WARNING]
> The code is still in its initial configuration stages; interfaces may change. Use with care.


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

Note: this will install the last stable version of [PLAID](https://github.com/PLAID-lib/plaid-bridges).

## 2 Core concepts

plaid-bridges provides high-level adapters that connect PLAID datasets with popular machine learning ecosystems. These adapters:

- Enable seamless conversion from structured PLAID data to ML-ready inputs, and back to engineering-friendly outputs.
- Support both grid-like fields and simple scalars, covering common physics/engineering use cases.
- Preserve mesh and geometry context where relevant, so downstream models can leverage structure.
- Offer a consistent workflow to prepare data for training/inference and to post-process model predictions.
- Reduce boilerplate when moving between PLAID and ML frameworks, keeping pipelines clear and maintainable.

## 3 Going further

See the documentation for a concise getting started guide and end-to-end examples:
- [Getting Started](https://plaid-bridges.readthedocs.io/en/latest/source/getting_started.html)
- [Examples & Tutorials](https://plaid-bridges.readthedocs.io/en/latest/source/notebooks.html)
- [API reference](https://plaid-bridges.readthedocs.io/en/latest/autoapi/plaid_bridges/index.html)
