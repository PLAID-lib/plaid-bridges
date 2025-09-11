import numpy as np

from plaid_bridges.common import BaseRegressionDataset, HomogeneousOfflineTransformer


class Test_Base_Torch:
    def test_HomogeneousDataset(self, dataset, scalar_features_ids, field_features_ids):
        offline_transformer = HomogeneousOfflineTransformer(
            in_features_identifiers=scalar_features_ids,
            out_features_identifiers=field_features_ids,
        )

        homogen_dataset = BaseRegressionDataset(
            dataset=dataset,
            offline_transformer=offline_transformer,
        )
        offline_transformer.inverse_transform(dataset, [np.arange(3), np.arange(4)])

        print(homogen_dataset)
        len(homogen_dataset)
        homogen_dataset[0]
