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
