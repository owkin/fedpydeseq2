from pathlib import Path

import yaml  # type: ignore
from substra.sdk.schemas import BackendType

from fedpydeseq2.core.deseq2_strategy import DESeq2Strategy
from fedpydeseq2.substra_utils.federated_experiment import run_federated_experiment


def run_fedpydeseq2_experiment(
    n_centers: int = 2,
    backend: BackendType = "subprocess",
    register_data: bool = False,
    simulate: bool = True,
    asset_directory: Path | None = None,
    centers_root_directory: Path | None = None,
    compute_plan_name: str = "FedPyDESeq2Experiment",
    dataset_name: str = "MyDatasetName",
    remote_timeout: int = 86400,  # 24 hours
    clean_models: bool = True,
    save_filepath: str | Path | None = None,
    credentials_path: str | Path | None = None,
    dataset_datasamples_keys_path: str | Path | None = None,
    cp_id_path: str | Path | None = None,
    parameter_file: str | Path | None = None,
    fedpydeseq2_wheel_path: str | Path | None = None,
    **kwargs,
) -> dict:
    """Run a federated experiment using the DESeq2 strategy.

    Parameters
    ----------
    n_centers : int
        Number of centers to use in the federated experiment.

    backend : BackendType
        Backend to use for the experiment. Should be one of "subprocess", "docker"
        or "remote".

    register_data : bool
        Whether to register the data on the substra platform. Can be True only
        when using the remote backend.

    simulate : bool
        Whether to simulate the experiment. If True, the experiment will be simulated
        and no data will be sent to the centers. This can be True only in subprocess
        backend.

    asset_directory : Path
        Path to the directory containing the assets (opener.py and description.md).

    centers_root_directory : Path, optional
        Path to the directory containing the centers data. Can be None only in remote
        mode when register_data is False.
        The centers data should be organized as follows:
        ```
        <centers_root_directory>
        ├── center_0
        │   ├── counts_data.csv
        │   └── metadata.csv
        ├── center_1
        │   ├── counts_data.csv
        │   └── metadata.csv
        └──

        ```
        where the metadata.csv file is indexed by sample barcodes and contains
        all columns needed to build the design matrix, and the counts_data.csv file
        represents a dataframe with gene names as columns and sample barcodes as rows,
        in the "barcode" column.

    compute_plan_name : str
        Name of the compute plan to use for the experiment.

    dataset_name : str
        Name of the dataset to fill in the Dataset schema.

    remote_timeout : int
        Timeout in seconds for the remote backend.

    clean_models : bool
        Whether to clean the models after the experiment.

    save_filepath : str or Path
        Path to save the results of the experiment.

    credentials_path : str or Path
        Path to the file containing the credentials to use for the remote backend.

    dataset_datasamples_keys_path : str or Path
        Path to the file containing the datasamples keys of the dataset.
        Only used for the remote backend.
        Is filled in if register_data is True, and read if register_data is False.

    cp_id_path : str or Path, optional
        Path to the file containing the compute plan id.
        This file is a yaml file with the following structure:
        ```
        algo_org_name: str
        credentials_path: str
        compute_plan_key: str
        ```

    parameter_file : str or Path, optional
        If not None, yaml file containing the parameters to pass to the DESeq2Strategy.
        If None, the default parameters are used.

    fedpydeseq2_wheel_path : str or Path, optional
        Path to the wheel file of the fedpydeseq2 package. If provided and the backend
        is remote, this wheel will be added to the dependencies.

    **kwargs
        Arguments to pass to the DESeq2Strategy. They will overwrite those specified
        in the parameter_file if the file is not None.

    Returns
    -------
    dict
        Result of the strategy, which are assumed to be contained in the
        results attribute of the last round of the aggregation node.


    """
    if parameter_file is not None:
        with open(parameter_file, "rb") as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)
    else:
        parameters = {}
    parameters.update(kwargs)
    strategy = DESeq2Strategy(**parameters)

    return run_federated_experiment(
        strategy=strategy,
        n_centers=n_centers,
        backend=backend,
        register_data=register_data,
        simulate=simulate,
        centers_root_directory=centers_root_directory,
        assets_directory=asset_directory,
        compute_plan_name=compute_plan_name,
        dataset_name=dataset_name,
        remote_timeout=remote_timeout,
        clean_models=clean_models,
        save_filepath=save_filepath,
        credentials_path=credentials_path,
        dataset_datasamples_keys_path=dataset_datasamples_keys_path,
        cp_id_path=cp_id_path,
        fedpydeseq2_wheel_path=fedpydeseq2_wheel_path,
    )
