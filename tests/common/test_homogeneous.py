import numpy as np

from plaid_bridges.common import ArrayDataset, HomogeneousBridge


class Test_Base_Torch:
    def test_ArrayDataset(self):
        ml_dataset = ArrayDataset((np.zeros((2, 3)), np.ones((3, 4))))
        ml_dataset[0]
        len(ml_dataset)

    def test_HomogeneousDataset(self, dataset, scalar_features_ids, field_features_ids):
        bridge = HomogeneousBridge()

        homogen_dataset = bridge.convert(dataset, scalar_features_ids)
        assert len(homogen_dataset[0]) == 1

        homogen_dataset = bridge.convert(
            dataset, scalar_features_ids, field_features_ids
        )

        bridge.restore(dataset, [np.zeros((2, 3)), np.ones((3, 4))], field_features_ids)

        print(homogen_dataset)
        len(homogen_dataset)
        assert len(homogen_dataset[0]) == 2
