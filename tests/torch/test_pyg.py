from plaid_bridges.torch import PyGBridge


class Test_PyG:
    def test_PyG(self, dataset, in_out_features_ids):
        bridge = PyGBridge()
        pyg_dataset = bridge.convert(
            dataset, in_out_features_ids[0] + in_out_features_ids[1]
        )

        pyg_dataset[0]
