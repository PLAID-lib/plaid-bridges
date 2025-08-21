import numpy as np

from plaid_bridges.torch.base import HomogeneousDataset


class Test_Base_Torch:
    def test_HomogeneousDataset(self, dataset, scalar_features, field_features):
        HomogeneousDataset(dataset, scalar_features, field_features)
        homogen_dataset = HomogeneousDataset(dataset, field_features, scalar_features)
        homogen_dataset.inverse_transform_single_feature(
            scalar_features[0], np.arange(5)
        )
