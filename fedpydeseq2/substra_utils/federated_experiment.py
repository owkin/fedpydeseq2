import pickle as pkl
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml  # type: ignore
from loguru import logger
from substra.sdk.models import ComputePlanStatus
from substra.sdk.schemas import BackendType
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from substrafl import ComputePlanBuilder
from substrafl.experiment import execute_experiment
from substrafl.experiment import simulate_experiment
from substrafl.model_loading import download_aggregate_shared_state
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode

from fedpydeseq2.substra_utils.utils import check_datasample_folder
from fedpydeseq2.substra_utils.utils import get_client
from fedpydeseq2.substra_utils.utils import get_dependencies


def run_federated_experiment(
    strategy: ComputePlanBuilder,
    n_centers: int = 2,
    backend: BackendType = "subprocess",
    register_data: bool = False,
    simulate: bool = True,
    centers_root_directory: Path | None = None,
    assets_directory: Path | None = None,
    compute_plan_name: str = "FedPyDESeq2Experiment",
    dataset_name: str = "TCGA",
    remote_timeout: int = 86400,  # 24 hours
    clean_models: bool = True,
    save_filepath: str | Path | None = None,
    credentials_path: str | Path | None = None,
    dataset_datasamples_keys_path: str | Path | None = None,
    cp_id_path: str | Path | None = None,
    fedpydeseq2_wheel_path: str | Path | None = None,
) -> dict:
    """Run a federated experiment with the given strategy.

    In remote mode, if the data is already registered,
    the assets_directory and centers_root_directory
    are not used (register_data=False).

    Otherwise, the assets_directory and centers_root_directory must be
    provided. The assets_directory is expected to contain the opener.py
    and description.md files, used to create the dataset for all centers.
    The centers_root_directory is expected to contain a subdirectory for each center,
    in the following form:

    ```
    <centers_root_directory>
    ├── center_0
    ├── center_1

    ```

    These directories contain the necessary data for each center and are passed
    to the DataSampleSpec object to register the data to substra.

    Parameters
    ----------
    strategy : ComputePlanBuilder
        The strategy to use for the federated experiment.

    n_centers : int
        The number of centers to use in the experiment.

    backend : BackendType
        The backend to use for the experiment. Can be one of "subprocess",
        "docker", or "remote".

    register_data : bool
        Whether to register the data. If True, the assets_directory and
        centers_root_directory must be provided.
        Can be False only in "remote" mode.

    simulate : bool
        Whether to simulate the experiment. If True, the experiment must be run
        in subprocess mode.

    centers_root_directory : Optional[Path]
        The path to the root directory containing the data for each center.
        This is only used if register_data is True.

    assets_directory : Optional[Path]
        The path to the assets directory. It must contain the opener.py file
        and the description.md file. This is only used if register_data is True.

    compute_plan_name : str
        The name of the compute plan.

    dataset_name : str
        The name of the dataset to use, to be passed to the DatasetSpec object and used
        to create the path of the yaml file storing the dataset and datasample keys.

    remote_timeout : int
        The timeout for the remote backend in seconds.

    clean_models : bool
        Whether to clean the models after the experiment.

    save_filepath : Optional[Union[str, Path]]
        The path to save the results. If None, the results are not saved.

    credentials_path : Optional[Union[str, Path]]
        The path to the credentials file. By default, will be set to
        Path(__file__).parent / "credentials/credentials.yaml"
        This file is used only in remote mode, and is expected to be a dictionary with
        the following structure:
        ```
        org1:
            url: "****"
            token: "****"
        org2:
            url: "****"
            token: "****"
        ...
        ```
        The first organization is assumed to be the algorithm provider.
        The other organizations are the data providers.


    dataset_datasamples_keys_path : Optional[Union[str, Path]]
        The path to the file containing the dataset and datasamples keys.
        If None, and if backend is "remote", will be set to
        Path(__file__).parent / "credentials/<dataset>-datasamples-keys.yaml"
        This file is used only in remote mode, and is expected to be a dictionary with
        the following structure:
        ```
        org_id:
            dataset_key: "****"
            datasample_key: "****"
        ...
        ```
        Where all data provider org ids are present, and there is only one
        datasample key per org id.
        This file is generated if register_data is True and backend is "remote".
        This file is loaded if register_data is False and backend is "remote".

    cp_id_path : str or Path or None
        The path to a file where we save the necessary information to
        retrieve the compute plan. This parameter
        is only used in remote mode.
        If None, this information is not saved.
        If a path is provided, the information is saved in a yaml file with the
        following structure:
        ```
        compute_plan_key: "****"
        credentials_path: "****"
        algo_org_name: "****"
        ```

    fedpydeseq2_wheel_path : Optional[Union[str, Path]]
        The path to the wheel file of the fedpydeseq2 package. If provided and the
        backend is remote, this wheel will be added to the dependencies.

    Returns
    -------
    dict
        Result of the strategy, which are assumed to be contained in the
        results attribute of the last round of the aggregation node.
    """
    # %%
    # Setup
    # *****
    # In the following code cell, we define the different
    #  organizations needed for our FL experiment.
    # Every computation will run in `subprocess` mode,
    # where everything runs locally in Python
    # subprocesses.
    # Others backend_types are:
    # "docker" mode where computations run locally in docker
    #  containers
    # "remote" mode where computations run remotely (you need to
    # have a deployed platform for that)
    logger.info("Setting up organizations...")
    n_clients = n_centers + 1
    if backend == "remote":
        clients_ = [
            get_client(
                backend_type=backend,
                org_name=f"org{i}",
                credentials_path=credentials_path,
            )
            for i in range(1, n_clients + 1)
        ]
    else:
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

    # %%
    # Dataset registration
    # ====================
    #
    # A :ref:`documentation/concepts:Dataset` is composed of an **opener**,
    # which is a Python script that can load
    # the data from the files in memory and a description markdown file.
    # The :ref:`documentation/concepts:Dataset` object itself does not contain
    #  the data. The proper asset that contains the
    # data is the **datasample asset**.
    #
    # A **datasample** contains a local path to the data. A datasample can be
    #  linked to a dataset in order to add data to a
    # dataset.
    #
    # Data privacy is a key concept for Federated Learning experiments.
    # That is why we set
    # :ref:`documentation/concepts:Permissions` for :ref:`documentation/concepts:Assets`
    #  to determine how each organization can access a specific asset.
    #
    # Note that metadata such as the assets' creation date and the asset owner are
    #  visible to all the organizations of a
    # network.

    # Define the path to the asset.
    if register_data:
        logger.info("Registering the datasets...")
    else:
        logger.info("Using pre-registered datasets...")

    dataset_keys = {}
    train_datasample_keys = {}

    if dataset_datasamples_keys_path is None:
        dataset_datasamples_keys_path = (
            Path(__file__).parent / f"credentials/{dataset_name}-datasamples-keys.yaml"
        )
    else:
        dataset_datasamples_keys_path = Path(dataset_datasamples_keys_path)

    if not register_data:
        # Check that we are in remote mode
        assert backend == "remote", (
            "register_data must be True if backend is not remote,"
            "as the datasets can be saved and reused only in remote mode."
            "If register_data is False, the dataset_datasamples_keys_path "
            "provides the necessary information to load the data which is "
            "already present on each remote organization."
        )
        # Load the dataset and datasample keys from the file
        with open(dataset_datasamples_keys_path) as file:
            dataset_datasamples_keys = yaml.load(file, Loader=yaml.FullLoader)
        for org_id in data_providers_ids:
            dataset_keys[org_id] = dataset_datasamples_keys[org_id]["dataset_key"]
            train_datasample_keys[org_id] = dataset_datasamples_keys[org_id][
                "datasample_key"
            ]
        logger.info("Datasets fetched.")
    else:
        for i, org_id in enumerate(data_providers_ids):
            client = clients[org_id]

            # In this case, check that the assets_directory is provided
            assert (
                assets_directory is not None
            ), "assets_directory must be provided if register_data is True"
            # In this case, check that the centers_root_directory is provided
            assert centers_root_directory is not None, (
                "centers_root_directory must be provided if" "register_data is True"
            )

            permissions_dataset = Permissions(public=True, authorized_ids=all_orgs_id)

            # DatasetSpec is the specification of a dataset. It makes sure every field
            # is well-defined, and that our dataset is ready to be registered.
            # The real dataset object is created in the add_dataset method.
            dataset = DatasetSpec(
                name=dataset_name,
                data_opener=assets_directory / "opener.py",
                description=assets_directory / "description.md",
                permissions=permissions_dataset,
                logs_permission=permissions_dataset,
            )
            logger.info(
                f"Adding dataset to client "
                f"{str(client.organization_info().organization_id)}"
            )
            dataset_keys[org_id] = client.add_dataset(dataset)
            logger.info(f"Dataset added. Key: {dataset_keys[org_id]}")
            assert dataset_keys[org_id], "Missing dataset key"
            data_sample = DataSampleSpec(
                data_manager_keys=[dataset_keys[org_id]],
                path=centers_root_directory / f"center_{i}",
            )
            if backend == "remote":
                check_datasample_folder(data_sample.path)
            train_datasample_keys[org_id] = client.add_data_sample(data_sample)

        # Create the dataset and datasample keys file if the backend is remote
        if backend == "remote":
            dataset_datasamples_dico = {
                org_id: {
                    "dataset_key": dataset_keys[org_id],
                    "datasample_key": train_datasample_keys[org_id],
                }
                for org_id in data_providers_ids
            }
            with open(dataset_datasamples_keys_path, "w") as file:
                yaml.dump(dataset_datasamples_dico, file)
        logger.info("Datasets registered.")

    logger.info(f"Dataset keys: {dataset_keys}")

    # %%
    # Where to train where to aggregate
    # =================================
    #
    # We specify on which data we want to train our model, using
    # the :ref:`substrafl_doc/api/nodes:TrainDataNode` object.
    # Here we train on the two datasets that we have registered earlier.
    #
    # The :ref:`substrafl_doc/api/nodes:AggregationNode` specifies the
    #  organization on which the aggregation operation
    # will be computed.

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

    # %%
    # Running the experiment
    # **********************
    #
    # We now have all the necessary objects to launch our experiment.
    # Please see a summary below of all the objects we created so far:
    #
    # - A :ref:`documentation/references/sdk:Client` to add or retrieve
    #  the assets of our experiment, using their keys to
    #   identify them.
    # - A `Federated Strategy <substrafl_doc/api/strategies:Strategies>`_,
    #  implementing the pipeline that will be run.
    # - `Train data nodes <substrafl_doc/api/nodes:TrainDataNode>`_ to
    # indicate on which data to train.
    # - An :ref:`substrafl_doc/api/nodes:AggregationNode`, to specify the
    #  organization on which the aggregation operation
    #   will be computed.
    # - An **experiment folder** to save a summary of the operation made.
    # - The :ref:`substrafl_doc/api/dependency:Dependency` to define the
    # libraries on which the experiment needs to run.

    # The Dependency object is instantiated in order to install the right
    #  libraries in the Python environment of each organization.

    algo_deps = get_dependencies(
        backend_type=backend, fedpydeseq2_wheel_path=fedpydeseq2_wheel_path
    )

    exp_path = tempfile.mkdtemp()

    if simulate:
        if backend != "subprocess":
            raise ValueError("Simulated experiment can only be run in subprocess mode.")
        _, intermediate_train_state, intermediate_state_agg = simulate_experiment(
            client=clients[algo_org_id],
            strategy=strategy,
            train_data_nodes=train_data_nodes,
            evaluation_strategy=None,
            aggregation_node=aggregation_node,
            clean_models=clean_models,
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
            [
                r and w
                for r, w in zip(agg_round_id_mask, agg_client_id_mask, strict=False)
            ]
        )[0][0]

        fl_results = intermediate_state_agg.state[agg_state_idx].results
    else:
        algo_client = clients[algo_org_id]

        compute_plan = execute_experiment(
            client=algo_client,
            strategy=strategy,
            train_data_nodes=train_data_nodes,
            evaluation_strategy=None,
            aggregation_node=aggregation_node,
            num_rounds=strategy.num_round,
            experiment_folder=exp_path,
            dependencies=algo_deps,
            clean_models=clean_models,
            name=compute_plan_name,
        )

        compute_plan_key = compute_plan.key

        # Extract the results. The method used here downloads the results from the
        # training nodes, as we cannot download
        # results from the aggregation node. Note that it implies an extra step
        # for the aggregation node to share the result with the training nodes.

        if cp_id_path is not None:
            cp_id_path = Path(cp_id_path)
            cp_id_path.parent.mkdir(parents=True, exist_ok=True)
            with cp_id_path.open("w") as f:
                yaml.dump(
                    {
                        "compute_plan_key": compute_plan_key,
                        "credentials_path": credentials_path,
                        "algo_org_name": "org1",
                    },
                    f,
                )

        if backend == "remote":
            sleep_time = 60
            t1 = time.time()
            finished = False
            while (time.time() - t1) < remote_timeout:
                status = algo_client.get_compute_plan(compute_plan_key).status
                logger.info(
                    f"Compute plan status is {status}, after {(time.time() - t1):.2f}s"
                )
                if status == ComputePlanStatus.done:
                    logger.info("Compute plan has finished successfully")
                    finished = True
                    break
                elif (
                    status == ComputePlanStatus.failed
                    or status == ComputePlanStatus.canceled
                ):
                    raise ValueError("Compute plan has failed")
                elif (
                    status == ComputePlanStatus.doing
                    or status == ComputePlanStatus.created
                ):
                    pass
                else:
                    logger.info(
                        f"Compute plan status is {status}, this shouldn't "
                        f"happen, sleeping {sleep_time} and retrying "
                        f"until timeout {remote_timeout}"
                    )
                time.sleep(sleep_time)
            if not finished:
                raise ValueError(
                    f"Compute plan did not finish after {remote_timeout} seconds"
                )

        fl_results = download_aggregate_shared_state(
            client=algo_client,
            compute_plan_key=compute_plan_key,
            round_idx=None,
        )
    if save_filepath is not None:
        pkl_save_filepath = Path(save_filepath) / "fl_result.pkl"
        with pkl_save_filepath.open("wb") as f:
            pkl.dump(fl_results, f)

    return fl_results
