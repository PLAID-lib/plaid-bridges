import numpy as np
import pytest

from plaid_bridges.common import ArrayDataset, BaseBridge


class Test_Base_Common:
    def test_BaseBridge(self, dataset, in_out_features_ids):
        bridge = BaseBridge(ArrayDataset)

        with pytest.raises(NotImplementedError):
            bridge.convert(
                dataset,
                in_out_features_ids[0],
            )

        with pytest.raises(NotImplementedError):
            bridge.restore(
                dataset=dataset,
                all_transformed_features=[np.zeros((2, 3)), np.ones((3, 4))],
                features_ids=in_out_features_ids[0],
            )
