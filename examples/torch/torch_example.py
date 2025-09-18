# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: plaid-bridged
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Torch bridges examples

# %%
import logging
logging.disable(logging.CRITICAL)

import copy

import numpy as np
from datasets import load_dataset
from plaid.bridges.huggingface_bridge import (
    huggingface_dataset_to_plaid,
    huggingface_description_to_problem_definition,
)
from plaid_ops.mesh.feature_engineering import update_dataset_with_sdf
from plaid_ops.mesh.transformations import (
    compute_bounding_box,
    project_on_regular_grid,
)
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from plaid_bridges.torch import GridFieldsAndScalarsBridge, PyGBridge
from plaid_bridges.torch.pyg import plot_sample_field, plot_sample_mesh

# %% [markdown]
# ## Projection on constant rectilinear grid, with scalars as constant fields

# %%
hf_dataset = load_dataset(
    "PLAID-datasets/2D_Multiscale_Hyperelasticity", split="all_samples"
)
pb_def = huggingface_description_to_problem_definition(hf_dataset.info.description)
ids_train = pb_def.get_split("DOE_train")[:2]

dataset_train, _ = huggingface_dataset_to_plaid(
    hf_dataset, ids=ids_train, processes_number=2   , verbose=False
)

print(dataset_train)

# %%
dims = (101, 101)
dataset_train = update_dataset_with_sdf(dataset_train, verbose=False)

bbox = compute_bounding_box(dataset_train)
projected_dataset_train = project_on_regular_grid(
    dataset_train, dimensions=dims, bbox=bbox, verbose=False
)

all_feat_ids = dataset_train[ids_train[0]].get_all_features_identifiers()
scalar_features = [f for f in all_feat_ids if "scalar" in f.values()]
field_features = [f for f in all_feat_ids if "field" in f.values()]

in_features_identifiers = [field_features[0], scalar_features[0]]
out_features_identifiers = [scalar_features[1], field_features[1]]

# %%
bridge = GridFieldsAndScalarsBridge(dimensions=dims)
torch_dataset = bridge.convert(
    projected_dataset_train, in_features_identifiers, out_features_identifiers
)

loader = DataLoader(
    torch_dataset,
    batch_size=2,
    shuffle=False,
)

out_feat_id = scalar_features[0]

before = copy.deepcopy(
    projected_dataset_train[ids_train[1]].get_feature_from_identifier(out_feat_id)
)

predictions = []
for batch_x, batch_y in loader:
    for torch_sample in batch_y:
        predictions.append(torch_sample.detach().cpu())

pred_projected_dataset_train = bridge.restore(
    projected_dataset_train, predictions, out_features_identifiers
)

after = copy.deepcopy(
    pred_projected_dataset_train[ids_train[1]].get_feature_from_identifier(out_feat_id)
)

print("Error after transform then inverse transform (2nd sample):")
print(np.linalg.norm(after - before) / np.linalg.norm(before))

# %% [markdown]
# ## Pytorch geometric

# %% [markdown]
# ### Heterogenous example: 2D_Multiscale_Hyperelasticity

# %%
bridge = PyGBridge()

pyg_dataset = bridge.convert(dataset_train, in_features_identifiers)

print(in_features_identifiers)

# %%
loader = PyGDataLoader(
    pyg_dataset,
    batch_size=2,
    shuffle=False,
)

before = copy.deepcopy(
    dataset_train[ids_train[1]].get_feature_from_identifier(in_features_identifiers[1])
)

predictions = []
for batch in loader:
    print("batch.x.shape =", batch.x.shape)
    for pyg_samples in batch.to_data_list():
        predictions.append(
            [pyg_samples.x.detach().cpu()[:, 0], pyg_samples.scalars.detach().cpu()[0]]
        )

pred_dataset_train = bridge.restore(dataset_train, predictions, in_features_identifiers)

after = copy.deepcopy(
    pred_dataset_train[ids_train[1]].get_feature_from_identifier(
        in_features_identifiers[1]
    )
)

print("Error after transform then inverse transform (2nd sample):")
print(np.linalg.norm(after - before) / np.linalg.norm(before))

# %%
plot_sample_mesh(pyg_dataset[0], block = False)

# %%
plot_sample_field(pyg_dataset[0], pyg_dataset[0].field_names[0], block = False)

# %% [markdown]
# ### Multi-base example: VKI-LS59

# %%
hf_dataset = load_dataset("PLAID-datasets/VKI-LS59", split="all_samples")
pb_def = huggingface_description_to_problem_definition(hf_dataset.info.description)
ids_train = pb_def.get_split("train")[:10]

dataset_train, _ = huggingface_dataset_to_plaid(
    hf_dataset, ids=ids_train, processes_number=5, verbose=False
)

print(dataset_train)

# %%
all_feat_ids = dataset_train[ids_train[0]].get_all_features_identifiers()
scalar_features = [f for f in all_feat_ids if "scalar" in f.values()]

# %% [markdown]
# #### Base "Base_1_2"

# %%
field_features = [
    f for f in all_feat_ids if "field" in f.values() if f["base_name"] == "Base_1_2"
]
features_identifiers = scalar_features + field_features

bridge = PyGBridge(base_name="Base_1_2")

pyg_dataset = bridge.convert(dataset_train, features_identifiers)

# %%
plot_sample_mesh(pyg_dataset[0], block = False)

# %%
plot_sample_field(pyg_dataset[0], "M_iso", block = False)

# %%
print("field_names =", pyg_dataset[0].field_names)
print("scalar_names =", pyg_dataset[0].scalar_names)
print("scalars =", pyg_dataset[0].scalars)

# %% [markdown]
# #### Base "Base_2_2"

# %%
field_features = [
    f for f in all_feat_ids if "field" in f.values() if f["base_name"] == "Base_2_2"
]
features_identifiers = scalar_features + field_features

bridge = PyGBridge(base_name="Base_2_2")

pyg_dataset = bridge.convert(dataset_train, features_identifiers)

# %%
plot_sample_mesh(pyg_dataset[0], block = False)

# %%
plot_sample_field(pyg_dataset[0], "nut", block = False)
