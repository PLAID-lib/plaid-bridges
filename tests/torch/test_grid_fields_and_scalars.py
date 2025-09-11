from plaid_ops.mesh.transformations import (
    compute_bounding_box,
    project_on_regular_grid,
)

from plaid_bridges.common import BaseRegressionDataset
from plaid_bridges.torch import GridFieldsAndScalarsOfflineTransformer


class Test_Torch_Grid:
    def test_GridFieldsAndScalars(self, dataset, in_out_features_ids):
        bbox = compute_bounding_box(dataset)
        proj_dataset = project_on_regular_grid(
            dataset, dimensions=(5, 5), bbox=bbox, verbose=True
        )

        offline_transformer = GridFieldsAndScalarsOfflineTransformer(
            in_features_identifiers=in_out_features_ids[0],
            out_features_identifiers=in_out_features_ids[1],
            dimensions=(5, 5),
        )

        torch_dataset = BaseRegressionDataset(
            dataset=proj_dataset,
            offline_transformer=offline_transformer,
        )

        prediction = [
            torch_dataset[0][1].detach().cpu(),
            torch_dataset[1][1].detach().cpu(),
        ]
        offline_transformer.inverse_transform(proj_dataset, prediction)
