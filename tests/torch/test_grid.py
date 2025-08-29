from plaid_ops.mesh.transformations import (
    compute_bounding_box,
    project_on_regular_grid,
)

from plaid_bridges.torch.grid import GridFieldsAndScalarsDataset


class Test_Torch_Grid:
    def test_BaseRegressionDataset(self, dataset, in_out_features):
        bbox = compute_bounding_box(dataset)
        proj_dataset = project_on_regular_grid(
            dataset, dimensions=(5, 5), bbox=bbox, verbose=True
        )
        torch_dataset = GridFieldsAndScalarsDataset(
            dataset=proj_dataset,
            dimensions=(5, 5),
            in_features_identifiers=in_out_features[0],
            out_features_identifiers=in_out_features[1],
        )
        prediction = [
            torch_dataset[0][1].detach().cpu().numpy(),
            torch_dataset[1][1].detach().cpu().numpy(),
        ]
        torch_dataset.inverse_transform(prediction)

    def test_BaseRegressionDataset_with_empty_out_field(self, dataset, in_out_features):
        bbox = compute_bounding_box(dataset)
        dataset[0].del_field("out_field")
        proj_dataset = project_on_regular_grid(
            dataset, dimensions=(5, 5), bbox=bbox, verbose=True
        )
        GridFieldsAndScalarsDataset(
            dataset=proj_dataset,
            dimensions=(5, 5),
            in_features_identifiers=in_out_features[0],
            out_features_identifiers=in_out_features[1],
        )
