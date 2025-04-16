"""Implement the test of simple layers."""

import os
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substrafl.experiment import simulate_experiment
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode

from fedpydeseq2.substra_utils.utils import get_client
from tests.unit_tests.layers.layers_tester import SimpleLayersTester


def get_data(
    n_centers: int = 2, min_n_obs: int = 80, max_n_obs: int = 120, n_vars: int = 20
) -> tuple[list, list]:
    """Get data given hyperparameters.

    For each center, generate a random number of observations
    between min_n_obs and max_n_obs.

    Parameters
    ----------
    n_centers : int
        The number of centers.

    min_n_obs : int
        The minimum number of observations.

    max_n_obs : int
        The maximum number of observations.

    n_vars : int
        The number of variables. Corresponds to the number of genes in
        DGEA.

    Returns
    -------
    list_counts : list
        A list of count matrices.

    list_obs_names : list
        A list of lists of observation names.
    """

    list_counts = []
    list_obs_names = []
    n_obs_offset = 0
    for _ in range(n_centers):
        n_obs = np.random.randint(min_n_obs, max_n_obs)
        list_counts.append(np.random.randint(0, 1000, size=(n_obs, n_vars)))
        obs_names = [f"sample_{i}" for i in range(n_obs_offset, n_obs_offset + n_obs)]
        n_obs_offset += n_obs
        list_obs_names.append(obs_names)

    return list_counts, list_obs_names


@pytest.mark.parametrize(
    "num_row_values_equal_n_params, add_cooks, save_cooks, "
    "has_save_layers_to_disk_attribute, save_layers_to_disk, "
    "has_layers_to_save_on_disk_attribute, test_layers_to_save_on_disk_attribute, "
    "test_layers_to_load, test_layers_to_save",
    [
        parameters
        for parameters in product(
            [
                True,
                False,
            ],
            [
                True,
                False,
            ],
            [
                True,
                False,
            ],
            [
                True,
                False,
            ],
            [
                True,
                False,
            ],
            [
                True,
                False,
            ],
            [
                True,
                False,
            ],
            [
                True,
                False,
            ],
            [
                True,
                False,
            ],
        )
        if parameters[1] or not parameters[2]
    ],
)
def test_reconstruct_adatas_decorator(
    num_row_values_equal_n_params: bool,
    add_cooks: bool,
    save_cooks: bool,
    has_save_layers_to_disk_attribute: bool,
    save_layers_to_disk: bool,
    has_layers_to_save_on_disk_attribute: bool,
    test_layers_to_save_on_disk_attribute: bool,
    test_layers_to_save: bool,
    test_layers_to_load: bool,
    n_centers: int = 2,
    n_params: bool = 5,
    min_n_obs: int = 80,
    max_n_obs: int = 120,
    n_vars: int = 20,
):
    """Test the reconstruct_adatas decorator.

    Parameters
    ----------
    num_row_values_equal_n_params : bool
        Whether the number of values taken by the design is equal to
        the number of parameters.
        If False, the number of row values is equal to the number of parameters + 1.

    add_cooks : bool
        Whether to add the cooks layer. Defaults to False.



    has_save_layers_to_disk_attribute : bool
        Whether the strategy has the save_layers_to_disk attribute.

    save_layers_to_disk : bool
        Whether to save the layers to disk.

    has_layers_to_save_load_attribute : bool
        Whether the strategy has the layers_to_save and layers_to_load attribute.

    test_layers_to_load : bool
        Whether to test the layers to load.

    test_layers_to_save : bool
        Whether to test the layers to save.

    n_centers : int
        The number of centers.

    n_params : int
        The number of parameters.

    min_n_obs : int
        The minimum number of observations.

    max_n_obs : int
        The maximum number of observations.

    n_vars : int
        The number of variables. Corresponds to the number of genes in
        DGEA.
    """

    strategy = SimpleLayersTester(
        num_row_values_equal_n_params=num_row_values_equal_n_params,
        add_cooks=add_cooks,
        n_params=n_params,
        save_layers_to_disk=save_layers_to_disk,
        has_save_layers_to_disk_attribute=has_save_layers_to_disk_attribute,
        has_layers_to_save_on_disk_attribute=has_layers_to_save_on_disk_attribute,
        test_layers_to_save_on_disk_attribute=test_layers_to_save_on_disk_attribute,
        save_cooks=save_cooks,
        test_layers_to_load=test_layers_to_load,
        test_layers_to_save=test_layers_to_save,
    )
    list_counts, list_obs_names = get_data(n_centers, min_n_obs, max_n_obs, n_vars)
    n_centers = len(list_counts)
    n_clients = n_centers + 1
    backend = "subprocess"
    exp_path = Path(tempfile.mkdtemp())
    assets_directory = Path(__file__).parent / "opener"

    clients_ = [get_client(backend_type=backend) for _ in range(n_clients)]

    clients = {
        client.organization_info().organization_id: client for client in clients_
    }

    # Store organization IDs
    all_orgs_id = list(clients.keys())
    algo_org_id = all_orgs_id[0]  # Algo provider is defined as the first organization.
    data_providers_ids = all_orgs_id[
        1:
    ]  # Data providers orgs are the remaining organizations.

    dataset_keys = {}
    train_datasample_keys = {}
    list_df_counts = []

    for i, org_id in enumerate(data_providers_ids):
        client = clients[org_id]
        permissions_dataset = Permissions(public=True, authorized_ids=all_orgs_id)
        dataset = DatasetSpec(
            name="Test",
            type="csv",
            data_opener=assets_directory / "opener.py",
            description=assets_directory / "description.md",
            permissions=permissions_dataset,
            logs_permission=permissions_dataset,
        )
        print(
            f"Adding dataset to client "
            f"{str(client.organization_info().organization_id)}"
        )
        dataset_keys[org_id] = client.add_dataset(dataset)
        print("Dataset added. Key: ", dataset_keys[org_id])

        os.makedirs(exp_path / f"dataset_{org_id}", exist_ok=True)
        n_genes = list_counts[i].shape[1]
        columns = [f"gene_{i}" for i in range(n_genes)]

        # set seed

        df_counts = pd.DataFrame(
            list_counts[i],
            index=list_obs_names[i],
            columns=columns,
        )
        list_df_counts.append(df_counts)
        df_counts.to_csv(exp_path / f"dataset_{org_id}" / "counts_data.csv")

        data_sample = DataSampleSpec(
            data_manager_keys=[dataset_keys[org_id]],
            path=exp_path / f"dataset_{org_id}",
        )
        train_datasample_keys[org_id] = client.add_data_sample(data_sample)

    aggregation_node = AggregationNode(algo_org_id)

    train_data_nodes = []

    for org_id in data_providers_ids:
        # Create the Train Data Node (or training task) and save it in a list
        train_data_node = TrainDataNode(
            organization_id=org_id,
            data_manager_key=dataset_keys[org_id],
            data_sample_keys=[train_datasample_keys[org_id]],
        )
        train_data_nodes.append(train_data_node)

    simulate_experiment(
        client=clients[algo_org_id],
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=None,
        aggregation_node=aggregation_node,
        clean_models=True,
        num_rounds=strategy.num_round,
        experiment_folder=exp_path,
    )
