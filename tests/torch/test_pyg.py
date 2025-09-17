import pytest

from plaid_bridges.torch import PyGBridge
from plaid_bridges.torch.pyg import plot_sample_field, plot_sample_mesh


class Test_PyG:
    def test_PyG(self, dataset, in_out_features_ids):
        with pytest.raises(AssertionError):
            bridge = PyGBridge(base_name="Base_2_2_toto")
            pyg_dataset = bridge.convert(dataset, in_out_features_ids[0])

        with pytest.raises(AssertionError):
            bridge = PyGBridge(zone_name="Zone_toto")
            pyg_dataset = bridge.convert(dataset, in_out_features_ids[0])

        # -------------------
        bridge = PyGBridge()
        pyg_dataset = bridge.convert(
            dataset, in_out_features_ids[0] + in_out_features_ids[1]
        )

        pyg_dataset[0]

        prediction = [
            pyg_dataset[0].x.detach().cpu()[:, 0],
            pyg_dataset[1].x.detach().cpu()[:, 0],
        ]
        bridge.restore(dataset, prediction, in_out_features_ids[1])

    def test_PyG_plot(self, dataset, in_out_features_ids):
        bridge = PyGBridge()
        pyg_dataset = bridge.convert(dataset, in_out_features_ids[0])

        plot_sample_mesh(pyg_dataset[0], block=False)
        plot_sample_field(pyg_dataset[0], "in_field", block=False)
