"""Unit test for the Fed Prox Quasi Newton algorithm."""

import pytest
from fedpydeseq2_datasets.constants import TCGADatasetNames

from tests.unit_tests.fed_algorithms.fed_prox_quasi_newton.pqn_test_pipe import (
    pipe_test_compute_lfc_with_pqn,
)

TESTING_PARAMTERS_LIST = [
    ("TCGA-LUAD", 0.0, 5),
    ("TCGA-PAAD", 0.0, 5),
    ("TCGA-LUAD", 0.5, 50),
    ("TCGA-PAAD", 0.5, 50),
]


@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_lfc_with_pqn(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    PQN_min_mu=0.0,
    tolerated_failed_genes=2,
):
    """Perform a unit test to see if compute_lfc is working as expected.

    Note that the catching of IRLS is very simple here, as there are not enough
    genes to observe significant differences in the log fold changes.

    The behaviour of the fed prox algorithm is tested on a self hosted runner.

    Moreover, we only test with the fisher scaling mode, as the other modes are
    tested in the other tests, and perform less well in our tested datasets.

    We do not clip mu as this seems to yield better results.

    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    PQN_min_mu: float
        The minimum mu in the prox newton method.

    tolerated_failed_genes: int
        The number of tolerated failed genes.
        Is set to 2 by default.
    """

    pipe_test_compute_lfc_with_pqn(
        data_path=raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=False,
        small_genes=True,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors="stage",
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
        PQN_min_mu=PQN_min_mu,
        tolerated_failed_genes=tolerated_failed_genes,
    )


@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize(
    "dataset_name, PQN_min_mu, tolerated_failed_genes",
    TESTING_PARAMTERS_LIST,
)
def test_lfc_with_pqn_on_self_hosted(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    dataset_name: TCGADatasetNames,
    PQN_min_mu: bool,
    tolerated_failed_genes: int,
):
    """Perform a unit test for compute_lfc using the fisher scaling mode.

    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    dataset_name: TCGADatasetNames
        The name of the dataset, for example "TCGA-LUAD".

    PQN_min_mu: float
        The minimum mu in the prox newton method.

    tolerated_failed_genes: int
        The number of tolerated failed genes.
    """

    pipe_test_compute_lfc_with_pqn(
        data_path=raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name=dataset_name,
        small_samples=False,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors="stage",
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
        PQN_min_mu=PQN_min_mu,
        tolerated_failed_genes=tolerated_failed_genes,
    )


@pytest.mark.local
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize(
    "dataset_name, PQN_min_mu, tolerated_failed_genes",
    TESTING_PARAMTERS_LIST,
)
def test_lfc_with_pqn_on_local(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    dataset_name: TCGADatasetNames,
    PQN_min_mu: float,
    tolerated_failed_genes: int,
):
    """Perform a unit test for compute_lfc.

    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    dataset_name: TCGADatasetNames
        The name of the dataset, for example "TCGA-LUAD".

    PQN_min_mu: float
        The minimum mu in the prox newton method.

    tolerated_failed_genes: int
        The number of tolerated failed genes.
    """

    pipe_test_compute_lfc_with_pqn(
        data_path=raw_data_path,
        processed_data_path=processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name=dataset_name,
        small_samples=False,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors="stage",
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
        PQN_min_mu=PQN_min_mu,
        tolerated_failed_genes=tolerated_failed_genes,
    )
