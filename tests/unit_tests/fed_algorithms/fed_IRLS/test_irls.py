"""Unit test for the Fed Prox Quasi Newton algorithm."""

import pytest
from fedpydeseq2_datasets.constants import TCGADatasetNames

from tests.unit_tests.fed_algorithms.fed_IRLS.irls_test_pipe import (
    pipe_test_compute_lfc_with_irls,
)

TESTING_PARAMETERS_LIST = [
    "TCGA-LUAD",
    "TCGA-PAAD",
]


@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_lfc_with_irls(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
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

    tmp_processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.
    """

    pipe_test_compute_lfc_with_irls(
        data_path=raw_data_path,
        processed_data_path=tmp_processed_data_path,
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
        rtol=1e-2,
        atol=1e-3,
    )


@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize(
    "dataset_name",
    TESTING_PARAMETERS_LIST,
)
def test_lfc_with_irls_on_self_hosted(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    dataset_name: TCGADatasetNames,
):
    """Perform a unit test for compute_lfc using the fisher scaling mode.

    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    tmp_processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    dataset_name: TCGADatasetNames
        The name of the dataset, for example "TCGA-LUAD".
    """

    pipe_test_compute_lfc_with_irls(
        data_path=raw_data_path,
        processed_data_path=tmp_processed_data_path,
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
        rtol=1e-2,
        atol=1e-3,
    )


@pytest.mark.local
@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
@pytest.mark.parametrize(
    "dataset_name",
    TESTING_PARAMETERS_LIST,
)
def test_lfc_with_irls_on_local(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    dataset_name: TCGADatasetNames,
):
    """Perform a unit test for compute_lfc.

    Parameters
    ----------
    raw_data_path: Path
        The path to the root data.

    local_processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    dataset_name: TCGADatasetNames
        The name of the dataset, for example "TCGA-LUAD".
    """

    pipe_test_compute_lfc_with_irls(
        data_path=raw_data_path,
        processed_data_path=local_processed_data_path,
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
        rtol=1e-2,
        atol=1e-3,
    )
