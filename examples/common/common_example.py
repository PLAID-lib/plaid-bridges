# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: plaid-bridges
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Common examples

# %% [markdown]
# ## Homogeneous dataset

# %%
import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore", message=".*IProgress not found.*")

import copy

import numpy as np
from datasets.utils.logging import disable_progress_bar
from datasets import load_dataset
from plaid.bridges.huggingface_bridge import (
    huggingface_dataset_to_plaid,
)
from torch.utils.data import DataLoader

from plaid_bridges.common import HomogeneousBridge

disable_progress_bar()

# %%
hf_dataset = load_dataset("PLAID-datasets/VKI-LS59", split="all_samples[:2]")

dataset, _ = huggingface_dataset_to_plaid(
    hf_dataset, processes_number=2, verbose=False
)
print(dataset)

all_feat_ids = dataset[0].get_all_features_identifiers()

# %% [markdown]
# ### Field inputs and scalar outputs

# %%
scalar_features = [f for f in all_feat_ids if "scalar" in f.values()]
field_features = [
    f for f in all_feat_ids if "field" in f.values() and "Base_2_2" in f.values()
]

in_features_identifiers = field_features
out_features_identifiers = scalar_features

print(in_features_identifiers)
print(out_features_identifiers)

# %%
bridge = HomogeneousBridge()
homogen_dataset = bridge.convert(
    dataset, in_features_identifiers, out_features_identifiers
)

loader = DataLoader(
    homogen_dataset,
    batch_size=2,
    shuffle=False,
)

out_feat_id = out_features_identifiers[0]
before = copy.deepcopy(dataset[1].get_feature_from_identifier(out_feat_id))

predictions = []
for batch_x, batch_y in loader:
    for torch_sample in batch_y:
        predictions.append(torch_sample.detach().cpu().numpy())

dataset_pred = bridge.restore(dataset, predictions, out_features_identifiers)

after = copy.deepcopy(dataset_pred[1].get_feature_from_identifier(out_feat_id))

print("Error after transform then inverse transform (2nd sample):")
print(np.linalg.norm(after - before) / np.linalg.norm(before))

# %% [markdown]
# ### Scalar inputs and Field outputs

# %%
in_features_identifiers = scalar_features
out_features_identifiers = field_features

print(in_features_identifiers)
print(out_features_identifiers)

# %%
bridge = HomogeneousBridge()
homogen_dataset = bridge.convert(
    dataset, in_features_identifiers, out_features_identifiers
)

loader = DataLoader(
    homogen_dataset,
    batch_size=2,
    shuffle=False,
)

out_feat_id = out_features_identifiers[0]
before = copy.deepcopy(dataset[1].get_feature_from_identifier(out_feat_id))

predictions = []
for batch_x, batch_y in loader:
    for torch_sample in batch_y:
        predictions.append(torch_sample.detach().cpu().numpy())

dataset_pred = bridge.restore(dataset, predictions, out_features_identifiers)

after = copy.deepcopy(dataset_pred[1].get_feature_from_identifier(out_feat_id))

print("Error after transform then inverse transform (2nd sample):")
print(np.linalg.norm(after - before) / np.linalg.norm(before))
