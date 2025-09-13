import numpy as np
import pytest

from plaid_bridges.common import BaseBridge, MLDataset


class Test_Base_Common:
    def test_MLDataset(self):
        ml_dataset = MLDataset(np.zeros((2, 3)), np.ones((3, 4)))
        ml_dataset[0]
        len(ml_dataset)

    def test_BaseBridge(self, dataset, in_out_features_ids):
        bridge = BaseBridge()

        with pytest.raises(NotImplementedError):
            bridge.convert(
                dataset=dataset,
                features_ids_list=in_out_features_ids,
            )

        with pytest.raises(NotImplementedError):
            bridge.restore(
                dataset=dataset,
                all_transformed_features=[np.zeros((2, 3)), np.ones((3, 4))],
                features_ids=in_out_features_ids[0],
            )
