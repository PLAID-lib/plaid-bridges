from plaid_ops.mesh.transformations import (
    compute_bounding_box,
    project_on_regular_grid,
)

from plaid_bridges.torch import GridFieldsAndScalarsBridge


class Test_Torch_Grid:
    def test_GridFieldsAndScalars(self, dataset, in_out_features_ids):
        bbox = compute_bounding_box(dataset)
        proj_dataset = project_on_regular_grid(
            dataset, dimensions=(5, 5), bbox=bbox, verbose=False
        )
        bridge = GridFieldsAndScalarsBridge(dimensions=(5, 5))
        torch_dataset = bridge.convert(proj_dataset, [in_out_features_ids[0]])
        torch_dataset = bridge.convert(proj_dataset, in_out_features_ids)

        prediction = [
            torch_dataset[0][1].detach().cpu(),
            torch_dataset[1][1].detach().cpu(),
        ]
        bridge.restore(proj_dataset, prediction, in_out_features_ids[1])
