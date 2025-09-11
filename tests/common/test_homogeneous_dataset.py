import numpy as np

from plaid_bridges.common import BaseRegressionDataset, HomogeneousOfflineTransformer


class Test_Base_Torch:
    def test_HomogeneousDataset(self, dataset, scalar_features_ids, field_features_ids):
        offline_in_transformer = HomogeneousOfflineTransformer(
            features_identifiers=scalar_features_ids,
        )
        offline_out_transformer = HomogeneousOfflineTransformer(
            features_identifiers=field_features_ids,
        )

        homogen_dataset = BaseRegressionDataset(
            dataset=dataset,
            offline_in_transformer=offline_in_transformer,
        )
        homogen_dataset[0]

        homogen_dataset = BaseRegressionDataset(
            dataset=dataset,
            offline_in_transformer=offline_in_transformer,
            offline_out_transformer=offline_out_transformer,
        )

        offline_out_transformer.inverse_transform(dataset, [np.arange(3), np.arange(4)])

        print(homogen_dataset)
        len(homogen_dataset)

        assert len(homogen_dataset[0]) == 2
