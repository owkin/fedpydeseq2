"""Module to implement a testing pipeline for PQN method.

It consists in computing the log fold changes using the PQN method
directly, and checking that the nll obtained using this method
is lower or better than the one obtained using the standard pipe.
"""

import pickle as pkl
from pathlib import Path

import numpy as np
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from substra import BackendType

from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.fed_algorithms.fed_IRLS.fed_IRLS_tester import FedIRLSTester


def pipe_test_compute_lfc_with_irls(
    data_path: Path,
    processed_data_path: Path,
    tcga_assets_directory: Path,
    dataset_name: TCGADatasetNames = "TCGA-LUAD",
    small_samples: bool = False,
    small_genes: bool = False,
    simulate: bool = True,
    backend: BackendType = "subprocess",
    only_two_centers: bool = False,
    design_factors: str | list[str] = "stage",
    ref_levels: dict[str, str] | None = {"stage": "Advanced"},  # noqa: B006
    reference_dds_ref_level: tuple[str, ...] | None = None,
    rtol: float = 1e-2,
    atol: float = 1e-3,
):
    r"""Perform a unit test for the log fold change computation with IRLS.

    As IRLS does not always converge, we check that for all genes
    that converged, the log fold changes are the same as the ones
    obtained with the pooled data.

    Parameters
    ----------
    data_path: Path
        The path to the root data.

    processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    dataset_name: TCGADatasetNames
        The name of the dataset, for example "TCGA-LUAD".

    small_samples: bool
        Whether to use a small number of samples.
        If True, the number of samples is reduced to 10 per center.

    small_genes: bool
        Whether to use a small number of genes.
        If True, the number of genes is reduced to 100.

    simulate: bool
        If true, use the simulation mode, otherwise use the subprocess mode.

    backend: BackendType
        The backend to use. Either "subprocess" or "docker".

    only_two_centers: bool
        If true, restrict the data to two centers.

    design_factors: str or list
        The design factors to use.

    ref_levels: dict or None
        The reference levels of the design factors.

    reference_dds_ref_level: tuple or None
        The reference level of the design factors.

    rtol: float
        The relative tolerance to use for the comparison.

    atol: float
        The absolute tolerance to use for the comparison.

    """

    # Setup the ground truth path.
    experiment_id = get_experiment_id(
        dataset_name,
        small_samples,
        small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=None,
    )

    reference_data_path = processed_data_path / "centers_data" / "tcga" / experiment_id
    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        FedIRLSTester(
            design_factors=design_factors,
            ref_levels=ref_levels,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
        ),
        raw_data_path=data_path,
        processed_data_path=processed_data_path,
        assets_directory=tcga_assets_directory,
        simulate=simulate,
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        backend=backend,
        only_two_centers=only_two_centers,
        register_data=True,
        design_factors=design_factors,
        reference_dds_ref_level=reference_dds_ref_level,
    )

    # pooled dds file name
    pooled_dds_file_name = get_ground_truth_dds_name(reference_dds_ref_level)

    pooled_dds_file_path = (
        processed_data_path
        / "pooled_data"
        / "tcga"
        / experiment_id
        / f"{pooled_dds_file_name}.pkl"
    )
    with open(pooled_dds_file_path, "rb") as file:
        pooled_dds = pkl.load(file)

    # FL gene name by convergence type
    fl_beta = fl_results["beta"]
    fl_irls_diverged_mask = fl_results["irls_diverged_mask"]
    fl_non_zero_gene_names = fl_results["non_zero_genes_names"]
    converged_gene_names = fl_non_zero_gene_names[~fl_irls_diverged_mask]

    fl_LFC_converged = fl_beta[~fl_irls_diverged_mask, :]

    # pooled LFC results
    pooled_LFC_converged = (
        pooled_dds.varm["LFC"].loc[converged_gene_names, :].to_numpy()
    )

    #### ---- Check for the IRLS_converged ---- ####

    LFC_error_tol = np.abs(pooled_LFC_converged) * rtol + atol
    LFC_abs_error = np.abs(fl_LFC_converged - pooled_LFC_converged)

    assert np.all(LFC_abs_error < LFC_error_tol)
