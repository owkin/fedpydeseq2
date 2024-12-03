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
from pydeseq2.dds import DeseqDataSet
from substra import BackendType

from fedpydeseq2.core.utils import vec_loss
from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.fed_algorithms.fed_prox_quasi_newton.fed_pqn_tester import (
    FedProxQuasiNewtonTester,
)


def pipe_test_compute_lfc_with_pqn(
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
    PQN_min_mu: float = 0.0,
    rtol: float = 0.02,
    atol: float = 1e-3,
    nll_rtol: float = 0.02,
    nll_atol: float = 1e-3,
    tolerated_failed_genes: int = 5,
):
    r"""Perform a unit test for the log fold change computation.

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

    PQN_min_mu: float
        The minimum value for mu in the PQN method.

    rtol: float
        The relative tolerance for the LFC.

    atol: float
        The absolute tolerance for the LFC.

    nll_rtol: float
        The relative tolerance for the nll.

    nll_atol: float
        The absolute tolerance for the nll.

    tolerated_failed_genes: int
        The number of tolerated failed genes.

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
        FedProxQuasiNewtonTester(
            design_factors=design_factors,
            ref_levels=ref_levels,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
            PQN_min_mu=PQN_min_mu,
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
    fl_PQN_diverged_mask = fl_results["PQN_diverged_mask"]
    fl_non_zero_gene_names = fl_results["non_zero_genes_names"]
    converged_gene_names = fl_non_zero_gene_names[~fl_PQN_diverged_mask]
    diverged_gene_names = fl_non_zero_gene_names[fl_PQN_diverged_mask]

    fl_LFC_converged = fl_beta[~fl_PQN_diverged_mask, :]
    fl_LFC_diverged = fl_beta[fl_PQN_diverged_mask, :]

    # pooled LFC results
    pooled_LFC_converged = (
        pooled_dds.varm["LFC"].loc[converged_gene_names, :].to_numpy()
    )
    pooled_LFC_diverged = pooled_dds.varm["LFC"].loc[diverged_gene_names, :].to_numpy()

    #### ---- Check for the PQN_converged ---- ####

    # For genes that have converged with the prox newton method,
    # we check that the LFC are the same
    # for the FL and the pooled results.
    # If it is not the case, we check the relative log likelihood is not
    # too different.
    # If that is not the case, we check that the relative optimization error wrt
    # the beta init is not too different.
    # We tolerate a few failed genes.

    beta_nll_relative_error_testing(
        fl_LFC_converged,
        pooled_LFC_converged,
        pooled_dds,
        converged_gene_names,
        tolerated_failed_genes=tolerated_failed_genes,
    )

    #### ---- Check for the all_diverged ---- ####

    # We perform the same checks for the genes that have not converged as well.

    beta_nll_relative_error_testing(
        fl_LFC_diverged,
        pooled_LFC_diverged,
        pooled_dds,
        diverged_gene_names,
        rtol=rtol,
        atol=atol,
        nll_rtol=nll_rtol,
        nll_atol=nll_atol,
        tolerated_failed_genes=tolerated_failed_genes,
    )


def beta_nll_relative_error_testing(
    fl_LFC: np.ndarray,
    pooled_LFC: np.ndarray,
    pooled_dds: DeseqDataSet,
    fl_genes: list[str],
    rtol: float = 0.02,
    atol: float = 1e-3,
    nll_rtol: float = 0.02,
    nll_atol: float = 1e-3,
    tolerated_failed_genes: int = 5,
):
    r"""Testing for genes.

    Parameters
    ----------

    fl_LFC: np.ndarray
        The LFC from the FL results.

    pooled_LFC: np.ndarray
        The LFC from the pooled results.

    pooled_dds: DeseqDataSet
        The pooled DeseqDataSet.

    fl_genes: list[str]
        The genes that are not IRLS converged.

    rtol: float
        The relative tolerance for the LFC.

    atol: float
        The absolute tolerance for the LFC.

    nll_rtol: float
        The relative tolerance for the nll.

    nll_atol: float
        The absolute tolerance for the nll.

    tolerated_failed_genes: int
        The number of tolerated failed genes.

    """
    accepted_error = np.abs(pooled_LFC) * rtol + atol
    absolute_error = np.abs(fl_LFC - pooled_LFC)

    # We check that the relative errors are not too high.
    to_check_mask = (absolute_error > accepted_error).any(axis=1)

    if np.sum(to_check_mask) > 0:
        # For the genes whose relative error is too high,
        # We will start by checking the relative error for the nll.
        to_check_genes = fl_genes[to_check_mask]
        to_check_genes_index = pooled_dds.var_names.get_indexer(to_check_genes)

        counts = pooled_dds[:, to_check_genes_index].X
        design = pooled_dds.obsm["design_matrix"].values
        dispersions = pooled_dds[:, to_check_genes_index].varm["dispersions"]

        # We compute the mu values for the FL and the pooled results.
        size_factors = pooled_dds.obsm["size_factors"]
        mu_fl = np.maximum(
            size_factors[:, None] * np.exp(design @ fl_LFC[to_check_mask].T),
            0.5,
        )
        mu_pooled = np.maximum(
            size_factors[:, None] * np.exp(design @ pooled_LFC[to_check_mask].T),
            0.5,
        )
        fl_nll = vec_loss(
            counts,
            design,
            mu_fl,
            dispersions,
        )
        pooled_nll = vec_loss(
            counts,
            design,
            mu_pooled,
            dispersions,
        )

        # Note: here I test the nll and not the regularized NLL which is
        # the real target of the optimization. However, this should not be
        # an issue since we add only a small regularization and there is
        # a bound on the beta values.
        accepted_nll_error = np.abs(pooled_nll) * nll_rtol + nll_atol
        nll_error = fl_nll - pooled_nll
        failed_test_mask = accepted_nll_error < nll_error

        # We identify the genes that do not pass the nll relative error criterion.

        if np.sum(failed_test_mask) > 0:
            # We tolerate a few failed genes.
            print(
                "Genes that do not pass the nll relative error criterion with "
                f"tolerance {nll_rtol}."
            )
            print(to_check_genes[failed_test_mask])
            print("Correponding error and accepted errors: ")
            print(nll_error[failed_test_mask])
            print(accepted_nll_error[failed_test_mask])

            assert np.sum(failed_test_mask) <= tolerated_failed_genes
