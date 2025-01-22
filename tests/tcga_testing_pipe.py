"""Implements a function running a substrafl experiment with tcga dataset."""

from datetime import datetime
from pathlib import Path

import pandas as pd
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.create_reference_dds import setup_tcga_ground_truth_dds
from fedpydeseq2_datasets.process_and_split_data import setup_tcga_dataset
from fedpydeseq2_datasets.utils import get_experiment_id
from substra.sdk.schemas import BackendType
from substrafl import ComputePlanBuilder

from fedpydeseq2.fedpydeseq2.core.utils.logging import set_log_config_path
from fedpydeseq2.substra_utils.federated_experiment import run_federated_experiment

from .conftest import DEFAULT_LOGGING_CONFIGURATION_PATH


def run_tcga_testing_pipe(
    strategy: ComputePlanBuilder,
    raw_data_path: Path,
    processed_data_path: Path,
    assets_directory: Path,
    backend: BackendType = "subprocess",
    simulate: bool = True,
    dataset_name: TCGADatasetNames = "TCGA-LUAD",
    small_samples: bool = False,
    small_genes: bool = False,
    only_two_centers: bool = False,
    register_data: bool = False,
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    reference_dds_ref_level: tuple[str, ...] | None = ("stage", "Advanced"),
    refit_cooks: bool = False,
    remote_timeout: int = 86400,  # 24 hours
    clean_models: bool = True,
    logging_config_file_path: str | Path | None = DEFAULT_LOGGING_CONFIGURATION_PATH,
) -> dict:
    """Runa tcga experiment using the given substrafl strategy.

    The raw_data_path is expected to have the following structure:
    ```
    <raw_data_path>
    ├── tcga
    │   ├── <dataset_name>_clinical.tsv.gz
    │   └── <dataset_name>_raw_RNAseq.tsv.gz
    └──
    ```

    The processed_data_path will be created if it does not exist and
    will contain the following structure:

    ```
    <processed_data_path>
    ├── centers_data
    │   └── tcga
    │       └── <dataset_name>
    │           └── center_0
    │               ├── counts_data.csv
    │               ├── metadata.csv
    │               └── ground_truth_dds.pkl
    ├── pooled_data
    │   └── tcga
    │       └── <dataset_name>
    │           ├── counts_data.csv
    │           ├── metadata.csv
    │           └── ground_truth_dds.pkl
    └──

    ```

    Parameters
    ----------
    strategy : ComputePlanBuilder

    raw_data_path : Path
        The path to the raw tcga data. must contain a folder "tcga" with the structure
        described above.
    processed_data_path : Path
        The path to the processed data. The subfolders will be created if does not
        exist.
    assets_directory : Path
        The path to the assets directory. Is expected to contain the opener.py and
        description.md files.
    backend : str, optional
        'docker', or 'subprocess'. (Default='subprocess').
    simulate : bool, optional
        If True, the experiment is simulated. (Default=True).
    dataset_name : Literal["TCGA-LUAD"], optional
        The dataset to preprocess, by default "TCGA-LUAD".
    small_samples : bool, optional
        If True, only preprocess a small subset of the data, by default False.
        The number of samples is reduced to 10 per center.
    small_genes : bool, optional
        If True, only preprocess a small subset of the genes, by default False.
        The number of genes is reduced to 100.
    only_two_centers : bool, optional
        If True, merged the centers into two centers, by default False.
    register_data : bool, optional
        If True, register the data to substra. Otherwise, use pre-registered dataset
         and data samples. By default False.
    design_factors : Union[str, list[str]]
        Name of the columns of metadata to be used as design variables.
        For now, only "stage", "gender" and "CPE" are supported.
    continuous_factors : list[str] or None
        The factors which are continuous.
    reference_dds_ref_level : tuple[str, ...] or None
        The reference levels for the design factors. If None, the first level is used.
    refit_cooks : bool
        If True, refit the model after removing the Cook's distance outliers.
    remote_timeout : int
        The timeout for the remote experiment in seconds.
        This means that we wait for at most `remote_timeout` seconds for the experiment
        to finish. If the experiment is not finished after this time, we raise an error.
        The default is 86400 s (24h).
    clean_models : bool
        If True, clean the models after the experiment. (Default=True).
    logging_config_file_path : str or Path or None
        The path to the logging configuration file. If None, no logging is done.
        By default, the default logging configuration is used, see the
        DEFAULT_LOGGING_CONFIGURATION_PATH constant.



    Returns
    -------
    dict
        Result of the strategy, which are assumed to be contained in the
        results attribute of the last round of the aggregation node.
    """
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    compute_plan_name = f"FedPyDESeq2_{dataset_name}_{current_datetime}"

    print("Setting up TCGA dataset...")
    setup_tcga_dataset(
        raw_data_path,
        processed_data_path,
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
    )
    print("Setting up TCGA ground truth DESeq2 datasets...")
    setup_tcga_ground_truth_dds(
        processed_data_path,
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        reference_dds_ref_level=reference_dds_ref_level,
        default_refit_cooks=refit_cooks,
    )

    experiment_id = get_experiment_id(
        dataset_name,
        small_samples,
        small_genes,
        only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
    )
    metadata = pd.read_csv(
        processed_data_path / "pooled_data" / "tcga" / experiment_id / "metadata.csv"
    )
    n_centers = len(metadata.center_id.unique())

    # Set the logging configuration file
    set_log_config_path(logging_config_file_path)

    fl_results = run_federated_experiment(
        strategy=strategy,
        register_data=register_data,
        n_centers=n_centers,
        backend=backend,
        simulate=simulate,
        centers_root_directory=processed_data_path
        / "centers_data"
        / "tcga"
        / experiment_id,
        assets_directory=assets_directory,
        compute_plan_name=compute_plan_name,
        dataset_name=dataset_name,
        remote_timeout=remote_timeout,
        clean_models=clean_models,
        save_filepath=None,
    )

    return fl_results
