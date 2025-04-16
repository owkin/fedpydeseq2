"""Implements a function running a substrafl experiment with tcga dataset."""

import itertools
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydeseq2.utils import trimmed_mean
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substrafl.experiment import simulate_experiment
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode

from fedpydeseq2.substra_utils.utils import get_client
from tests.unit_tests.fed_algorithms.trimmed_mean_strategy.trimmed_mean_strategy import (  # noqa: E501
    TrimmedMeanStrategyForTesting,
)

LIST_MOCK_DATA_TYPE = [
    "random_2",
    "random_5",
    "random_10",
    "duplicate_5",
    "one_center_tie_case_1",
    "one_center_tie_case_2",
    "one_center_tie_case_3",
    "5_center_with_ties",
]


def get_data(data_type):
    if data_type.startswith("random_"):
        n_centers = int(data_type.split("_")[-1])
        n_genes = 100
        list_counts = []
        for _ in range(n_centers):
            n_samples = np.random.randint(80, 120)
            list_counts.append(
                np.random.randint(0, 1000, size=(n_samples, n_genes)).astype(float)
            )
    elif data_type.startswith("duplicate_"):
        n_centers = int(data_type.split("_")[-1])
        n_genes = 100
        list_counts = []
        counts = np.random.randint(0, 1000, size=(100, n_genes)).astype(float)
        for _ in range(n_centers):
            list_counts.append(counts.copy())
    elif data_type == "one_center_tie_case_1":
        n_genes = 100
        n_samples = 100
        counts = np.random.randint(0, 1000, size=(n_samples, n_genes)).astype(float)
        mask = np.random.choice([True, False], size=(n_samples, n_genes))
        counts[mask] = 0.0
        list_counts = [counts]
    elif data_type == "one_center_tie_case_2":
        n_genes = 100
        n_samples = 100
        counts = np.random.randint(0, 1000, size=(n_samples, n_genes)).astype(float)
        mask = np.random.choice([True, False], size=(n_samples, n_genes))
        counts[mask] = 1000.0
        list_counts = [counts]
    elif data_type == "one_center_tie_case_3":
        n_genes = 100
        n_samples = 100
        counts = np.random.randint(0, 1000, size=(n_samples, n_genes)).astype(float)
        mask = np.random.choice([-1, 0, 1], size=(n_samples, n_genes))
        counts[mask == -1] = 0.0
        counts[mask == 1] = 1000.0
        list_counts = [counts]
    elif data_type == "5_center_with_ties":
        n_genes = 100
        n_samples = 100
        list_counts = []
        for _ in range(5):
            counts = np.random.randint(0, 1000, size=(n_samples, n_genes)).astype(float)
            mask = np.random.choice([-1, 0, 1], size=(n_samples, n_genes))
            counts[mask == -1] = 0.0
            counts[mask == 1] = 1000.0
            list_counts.append(counts)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    return list_counts


@pytest.mark.parametrize(
    "data_type, trim_ratio, nb_iter, layer_used, refit",
    itertools.product(
        LIST_MOCK_DATA_TYPE,
        [0.1, 0.125, 0.2],
        [40, 50],
        ["layer_1", "layer_2"],
        [True, False],
    ),
)
def test_trimmed_mean_strategy(data_type, trim_ratio, nb_iter, layer_used, refit):
    strategy = TrimmedMeanStrategyForTesting(
        trim_ratio=trim_ratio, layer_used=layer_used, nb_iter=nb_iter, refit=refit
    )
    list_counts = get_data(data_type)
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
        n_samples = list_counts[i].shape[0]
        columns = [f"gene_{i}" for i in range(n_genes)]
        index = [f"sample_{i}" for i in range(n_samples)]
        # set seed

        df_counts = pd.DataFrame(
            list_counts[i],
            index=index,
            columns=columns,
        )
        df_metadata = pd.DataFrame(
            np.random.randint(0, 1000, size=(n_samples, n_genes)),
            index=index,
        )
        list_df_counts.append(df_counts)
        df_counts.to_csv(exp_path / f"dataset_{org_id}" / "counts_data.csv")
        df_metadata.to_csv(exp_path / f"dataset_{org_id}" / "metadata.csv")
        with open(exp_path / f"dataset_{org_id}" / "layer_used.txt", "w") as f:
            f.write(layer_used)

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

    _, intermediate_train_state, intermediate_state_agg = simulate_experiment(
        client=clients[algo_org_id],
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=None,
        aggregation_node=aggregation_node,
        clean_models=True,
        num_rounds=strategy.num_round,
        experiment_folder=exp_path,
    )

    # Gather results from the aggregation node

    agg_client_id_mask = [
        w == clients[algo_org_id].organization_info().organization_id
        for w in intermediate_state_agg.worker
    ]

    agg_round_id_mask = [
        r == max(intermediate_state_agg.round_idx)
        for r in intermediate_state_agg.round_idx
    ]

    agg_state_idx = np.where(
        [r and w for r, w in zip(agg_round_id_mask, agg_client_id_mask, strict=False)]
    )[0][0]

    fl_results = intermediate_state_agg.state[agg_state_idx].results

    total_df_counts = pd.concat(list_df_counts, axis=0)
    # in refit mode, we only keep the first 10 genes
    if refit:
        total_df_counts = total_df_counts.iloc[:, :10]

    pooled_trimmed_mean = trimmed_mean(total_df_counts, trim=trim_ratio, axis=0)

    assert np.allclose(
        fl_results[f"trimmed_mean_{layer_used}"], pooled_trimmed_mean, rtol=1e-6
    ), (
        "Trimmed mean is not the same : "
        + str(fl_results[f"trimmed_mean_{layer_used}"])
        + " vs "
        + str(pooled_trimmed_mean)
    )
