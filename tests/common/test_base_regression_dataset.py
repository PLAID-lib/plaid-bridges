import numpy as np
import pytest

from plaid_bridges.common import BaseRegressionDataset


class Test_Base_Common:
    def test_BaseRegressionDataset(self, dataset, in_out_features):
        reg_dataset = BaseRegressionDataset(
            dataset=dataset,
            in_features_identifiers=in_out_features[0],
            out_features_identifiers=in_out_features[1],
        )

        reg_dataset.show_details()

        assert np.allclose(reg_dataset[0][0][0], 1.0)
        assert np.allclose(reg_dataset[0][0][1], np.arange(5))
        assert np.allclose(reg_dataset[0][1][0], 1 + np.arange(5))
        assert np.allclose(reg_dataset[0][1][1], 2.0)

        assert np.allclose(reg_dataset[1][0][0], 1.0)
        assert np.allclose(reg_dataset[1][0][1], np.arange(5))
        assert np.allclose(reg_dataset[1][1][0], 1 + np.arange(5))
        assert np.allclose(reg_dataset[1][1][1], 2.0)

        assert len(reg_dataset) == 2

        with pytest.raises(NotImplementedError):
            reg_dataset.inverse_transform_single_feature(1, 2)

        reg_dataset = BaseRegressionDataset(
            dataset=dataset,
            in_features_identifiers=in_out_features[0],
            out_features_identifiers=in_out_features[1],
            train=True,
            online_transform=lambda x: x,
        )
        reg_dataset[0]

        reg_dataset = BaseRegressionDataset(
            dataset=dataset,
            in_features_identifiers=in_out_features[0],
            out_features_identifiers=in_out_features[1],
            train=False,
            online_transform=lambda x: x,
        )

        with pytest.raises(NotImplementedError):
            prediction = [reg_dataset[0][1], reg_dataset[1][1]]
            reg_dataset.inverse_transform(prediction)
