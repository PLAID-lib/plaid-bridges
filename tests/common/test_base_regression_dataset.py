import pytest

from plaid_bridges.common import BaseRegressionDataset, BaseTransformer


class Test_Base_Common:
    def test_BaseTransformer(self, in_out_features_ids, dataset):
        transformer = BaseTransformer(
            features_identifiers=in_out_features_ids[0],
        )
        print(transformer)
        transformer.show_details()

        with pytest.raises(NotImplementedError):
            transformer.transform(dataset)

        with pytest.raises(NotImplementedError):
            transformer.inverse_transform_single_feature(dataset, 1.0)

    def test_BaseRegressionDataset(self, dataset, in_out_features_ids):
        offline_in_transformer = BaseTransformer(
            features_identifiers=in_out_features_ids[0],
        )

        with pytest.raises(NotImplementedError):
            BaseRegressionDataset(
                dataset=dataset,
                offline_in_transformer=offline_in_transformer,
            )

        offline_out_transformer = BaseTransformer(
            features_identifiers=in_out_features_ids[1],
        )

        with pytest.raises(NotImplementedError):
            BaseRegressionDataset(
                dataset=dataset,
                offline_in_transformer=offline_in_transformer,
                offline_out_transformer=offline_out_transformer,
            )

        # len(reg_dataset)
        # print(reg_dataset)

        # assert np.allclose(reg_dataset[0][0][0], 1.0)
        # assert np.allclose(reg_dataset[0][0][1], np.arange(5))
        # assert np.allclose(reg_dataset[0][1][0], 1 + np.arange(5))
        # assert np.allclose(reg_dataset[0][1][1], 2.0)

        # assert np.allclose(reg_dataset[1][0][0], 1.0)
        # assert np.allclose(reg_dataset[1][0][1], np.arange(5))
        # assert np.allclose(reg_dataset[1][1][0], 1 + np.arange(5))
        # assert np.allclose(reg_dataset[1][1][1], 2.0)

        # assert len(reg_dataset) == 2

        # with pytest.raises(NotImplementedError):
        #     reg_dataset.inverse_transform_single_feature(1, 2)

        # reg_dataset = BaseRegressionDataset(
        #     dataset=dataset,
        #     in_features_identifiers=in_out_features[0],
        #     out_features_identifiers=in_out_features[1],
        #     train=True,
        #     online_transform=lambda x: x,
        # )
        # reg_dataset[0]

        # reg_dataset = BaseRegressionDataset(
        #     dataset=dataset,
        #     in_features_identifiers=in_out_features[0],
        #     out_features_identifiers=in_out_features[1],
        #     train=False,
        #     online_transform=lambda x: x,
        # )

        # with pytest.raises(NotImplementedError):
        #     prediction = [reg_dataset[0][1], reg_dataset[1][1]]
        #     reg_dataset.inverse_transform(prediction)
