import pickle as pkl
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fedpydeseq2_datasets.constants import TCGADatasetNames
from fedpydeseq2_datasets.utils import get_experiment_id
from fedpydeseq2_datasets.utils import get_ground_truth_dds_name
from pydeseq2.dds import DeseqDataSet
from scipy.linalg import solve  # type: ignore
from substra import BackendType

from fedpydeseq2.core.utils import vec_loss
from tests.tcga_testing_pipe import run_tcga_testing_pipe
from tests.unit_tests.deseq2_core.deseq2_lfc_dispersions.compute_lfc.compute_lfc_tester import (  # noqa: E501
    ComputeLFCTester,
)
from tests.unit_tests.unit_test_helpers.levels import make_reference_and_fl_ref_levels


def pipe_test_compute_lfc(
    data_path: Path,
    processed_data_path: Path,
    tcga_assets_directory: Path,
    dataset_name: TCGADatasetNames = "TCGA-LUAD",
    small_samples: bool = False,
    small_genes: bool = False,
    simulate: bool = True,
    backend: BackendType = "subprocess",
    only_two_centers: bool = False,
    lfc_mode: Literal["lfc", "mu_init"] = "lfc",
    design_factors: str | list[str] = "stage",
    continuous_factors: list[str] | None = None,
    ref_levels: dict[str, str] | None = {"stage": "Advanced"},  # noqa: B006
    reference_dds_ref_level: tuple[str, ...] | None = None,
    PQN_min_mu: float = 0.0001,
    rtol_irls: float = 1e-5,
    atol_irls: float = 1e-8,
    rtol_pqn: float = 2e-2,
    atol_pqn: float = 1e-3,
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

    lfc_mode: str
        The mode of the ComputeLFC algorithm.

    design_factors: str or list
        The design factors to use.

    ref_levels: dict or None
        The reference levels of the design factors.

    continuous_factors : list or None
        The continuous factors to use.

    reference_dds_ref_level: tuple or None
        The reference level of the design factors.

    PQN_min_mu : float
        The minimum mu in the prox newton method.

    tolerated_failed_genes: int
        The number of tolerated failed genes.

    rtol_irls: float
        The relative tolerance for the comparison of the LFC and mu values for genes
        where IRLS converges.

    atol_irls: float
        The absolute tolerance for the comparison of the LFC and mu values for genes
        where IRLS converges.

    rtol_pqn: float
        The relative tolerance for the comparison of the LFC and mu values for genes
        where PQN is used.

    atol_pqn: float
        The absolute tolerance for the comparison of the LFC and mu values for genes
        where PQN is used.

    nll_rtol: float
        The relative tolerance for the comparison of the nll values, when the LFC and mu
        values are too different.

    nll_atol: float
        The absolute tolerance for the comparison of the nll values, when the LFC and mu
        values are too different.

    """
    complete_ref_levels, reference_dds_ref_level = make_reference_and_fl_ref_levels(
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        ref_levels=ref_levels,
        reference_dds_ref_level=reference_dds_ref_level,
    )

    # Setup the ground truth path.
    experiment_id = get_experiment_id(
        dataset_name,
        small_samples,
        small_genes,
        only_two_centers=only_two_centers,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
    )

    reference_data_path = processed_data_path / "centers_data" / "tcga" / experiment_id
    # Get FL results.
    fl_results = run_tcga_testing_pipe(
        ComputeLFCTester(
            design_factors=design_factors,
            lfc_mode=lfc_mode,
            ref_levels=complete_ref_levels,
            continuous_factors=continuous_factors,
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
        continuous_factors=continuous_factors,
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

    # Get the sample ids and permutations to reorder
    sample_ids = pooled_dds.obs_names
    sample_ids_fl = fl_results["sample_ids"]
    # Get the permutation that sorts the sample ids
    perm = np.argsort(sample_ids)
    perm_fl = np.argsort(sample_ids_fl)

    # Get the initialization beta
    fl_beta_init = fl_results[f"{lfc_mode}_beta_init"]

    # FL gene name by convergence type
    fl_irls_genes = fl_results[f"{lfc_mode}_irls_genes"]
    fl_PQN_genes = fl_results[f"{lfc_mode}_PQN_genes"]
    fl_all_diverged_genes = fl_results[f"{lfc_mode}_all_diverged_genes"]

    # Get mu and beta param names
    beta_param_name = fl_results["beta_param_name"]
    mu_param_name = fl_results["mu_param_name"]

    # fl mu results
    fl_mu_irls_converged = fl_results[f"{mu_param_name}_irls_converged"]
    fl_mu_PQN_converged = fl_results[f"{mu_param_name}_PQN_converged"]
    fl_mu_all_diverged = fl_results[f"{mu_param_name}_all_diverged"]

    # Reorder the mu results
    fl_mu_irls_converged = fl_mu_irls_converged[perm_fl]
    fl_mu_PQN_converged = fl_mu_PQN_converged[perm_fl]
    fl_mu_all_diverged = fl_mu_all_diverged[perm_fl]

    # fl LFC results
    fl_LFC_irls_converged = fl_results[f"{beta_param_name}_irls_converged"]
    fl_LFC_PQN_converged = fl_results[f"{beta_param_name}_PQN_converged"]
    fl_LFC_all_diverged = fl_results[f"{beta_param_name}_all_diverged"]

    # pooled beta init
    pooled_beta_init = get_beta_init_pooled(pooled_dds)

    # pooled mu results
    pooled_beta_param_name, pooled_mu_param_name = get_pooled_results(
        pooled_dds, lfc_mode
    )

    pooled_mu_irls_converged = pooled_dds.layers[pooled_mu_param_name][
        :, pooled_dds.var_names.get_indexer(fl_irls_genes)
    ]
    pooled_mu_PQN_converged = pooled_dds.layers[pooled_mu_param_name][
        :, pooled_dds.var_names.get_indexer(fl_PQN_genes)
    ]
    pooled_mu_all_diverged = pooled_dds.layers[pooled_mu_param_name][
        :, pooled_dds.var_names.get_indexer(fl_all_diverged_genes)
    ]

    # Reorder the mu results
    pooled_mu_irls_converged = pooled_mu_irls_converged[perm]
    pooled_mu_PQN_converged = pooled_mu_PQN_converged[perm]
    pooled_mu_all_diverged = pooled_mu_all_diverged[perm]

    # pooled LFC results
    pooled_LFC_irls_converged = (
        pooled_dds.varm[pooled_beta_param_name].loc[fl_irls_genes, :].to_numpy()
    )
    pooled_LFC_PQN_converged = (
        pooled_dds.varm[pooled_beta_param_name].loc[fl_PQN_genes, :].to_numpy()
    )
    pooled_LFC_all_diverged = (
        pooled_dds.varm[pooled_beta_param_name].loc[fl_all_diverged_genes, :].to_numpy()
    )

    #### ---- Check for the beta init ---- ####

    assert np.allclose(
        fl_beta_init,
        pooled_beta_init,
        equal_nan=True,
    )

    #### ---- Check for the irls_converged ---- ####

    try:
        assert np.allclose(
            fl_LFC_irls_converged,
            pooled_LFC_irls_converged,
            equal_nan=True,
            rtol=rtol_irls,
            atol=atol_irls,
        )

    except AssertionError:
        # This is likely due to the fact that beta values are small.
        # We will check the relative error for the nll.
        relative_LFC_mu_nll_test(
            fl_mu_irls_converged,
            pooled_mu_irls_converged,
            fl_LFC_irls_converged,
            pooled_LFC_irls_converged,
            pooled_dds,
            fl_irls_genes,
            rtol=rtol_irls,
            atol=atol_irls,
            nll_rtol=nll_rtol,
            nll_atol=nll_atol,
            tolerated_failed_genes=0,
        )

    #### ---- Check for the PQN_converged ---- ####

    # For genes that have converged with the prox newton method,
    # we check that the hat diagonals, mu LFC and LFC are the same
    # for the FL and the pooled results.
    # If it is not the case, we check the relative log likelihood is not
    # too different.
    # If that is not the case, we check that the relative optimization error wrt
    # the beta init is not too different.
    # We tolerate a few failed genes.

    relative_LFC_mu_nll_test(
        fl_mu_PQN_converged,
        pooled_mu_PQN_converged,
        fl_LFC_PQN_converged,
        pooled_LFC_PQN_converged,
        pooled_dds,
        fl_PQN_genes,
        rtol=rtol_pqn,
        atol=atol_pqn,
        nll_rtol=nll_rtol,
        nll_atol=nll_atol,
        tolerated_failed_genes=tolerated_failed_genes,
    )

    #### ---- Check for the all_diverged ---- ####

    # We perform the same checks for the genes that have not converged as well.

    relative_LFC_mu_nll_test(
        fl_mu_all_diverged,
        pooled_mu_all_diverged,
        fl_LFC_all_diverged,
        pooled_LFC_all_diverged,
        pooled_dds,
        fl_all_diverged_genes,
        rtol=rtol_pqn,
        atol=atol_pqn,
        nll_rtol=nll_rtol,
        nll_atol=nll_atol,
        tolerated_failed_genes=tolerated_failed_genes,
    )


def relative_LFC_mu_nll_test(
    fl_mu_LFC: np.ndarray,
    pooled_mu_LFC: np.ndarray,
    fl_LFC: np.ndarray,
    pooled_LFC: np.ndarray,
    pooled_dds: DeseqDataSet,
    fl_genes: list[str],
    rtol: float = 2e-2,
    atol: float = 1e-3,
    nll_rtol: float = 0.02,
    nll_atol: float = 1e-3,
    tolerated_failed_genes: int = 5,
    irsl_mode: Literal["lfc", "mu_init"] = "lfc",
):
    r"""Perform the relative error test for the LFC and mu values.

    This test checks that the relative error for the LFC and mu values is not too high.
    If it is too high, we check the relative error for the nll.

    Parameters
    ----------
    fl_mu_LFC: np.ndarray
        The mu LFC from the FL results.

    pooled_mu_LFC: np.ndarray
        The mu LFC from the pooled results.

    fl_LFC: np.ndarray
        The LFC from the FL results.

    pooled_LFC: np.ndarray
        The LFC from the pooled results.

    pooled_dds: DeseqDataSet
        The pooled DeseqDataSet.

    fl_genes: list[str]
        The genes that are not IRLS converged.

    rtol: float
        The relative tolerance for the comparison of the LFC and mu values.

    atol: float
        The absolute tolerance for the comparison of the LFC and mu values.

    nll_rtol: float
        The relative tolerance for the comparison of the nll values.

    nll_atol: float
        The absolute tolerance for the comparison of the nll values.

    tolerated_failed_genes: int
        The number of tolerated failed genes.

    irsl_mode: Literal["lfc", "mu_init"]
        The mode of the ComputeLFC algorithm.
    """
    mu_LFC_error_tol = np.abs(pooled_mu_LFC) * rtol + atol
    mu_LFC_abs_error = np.abs(fl_mu_LFC - pooled_mu_LFC)
    LFC_error_tol = np.abs(pooled_LFC) * rtol + atol
    LFC_abs_error = np.abs(fl_LFC - pooled_LFC)

    # We check that the relative errors are not too high.
    to_check_mask = (mu_LFC_abs_error > mu_LFC_error_tol).any(axis=0) | (
        LFC_abs_error > LFC_error_tol
    ).any(axis=1)

    if np.sum(to_check_mask) > 0:
        print(
            f"{to_check_mask.sum()} genes do not pass the relative error criterion."
            f" Genes that do not pass the relative error criterion with tolerance 0.02:"
        )
        print(fl_genes[to_check_mask])
        print("Corresponding top mu relative error")
        mu_LFC_rel_error = np.abs((fl_mu_LFC - pooled_mu_LFC) / pooled_mu_LFC)
        print(np.sort(mu_LFC_rel_error[:, to_check_mask])[-10:])
        print("Corresponding LFC relative error")
        LFC_rel_error = np.abs((fl_LFC - pooled_LFC) / pooled_LFC)
        print(LFC_rel_error[to_check_mask, :])
        # For the genes whose relative error is too high,
        # We will start by checking the relative error for the nll.
        to_check_genes = fl_genes[to_check_mask]
        to_check_genes_index = pooled_dds.var_names.get_indexer(to_check_genes)

        counts = pooled_dds[:, to_check_genes_index].X
        design = pooled_dds.obsm["design_matrix"].values
        if irsl_mode == "lfc":
            dispersions = pooled_dds[:, to_check_genes_index].varm["dispersions"]
        else:
            dispersions = pooled_dds[:, to_check_genes_index].varm["_MoM_dispersions"]

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
        nll_error_tol = np.abs(pooled_nll) * nll_rtol + nll_atol

        failed_test_mask = (fl_nll - pooled_nll) > nll_error_tol

        # We identify the genes that do not pass the nll relative error criterion.

        if np.sum(failed_test_mask) > 0:
            print(
                f"{np.sum(failed_test_mask)} "
                f"genes do not pass the nll relative error"
                f" criterion with relative tolerance {nll_rtol} and absolute"
                f"tolerance {nll_atol}."
            )
            print("These genes are")
            print(to_check_genes[failed_test_mask])
            print("Corresponding absolute error")
            print(fl_nll[failed_test_mask] - pooled_nll[failed_test_mask])
            print("Corresponding error tolerance")
            print(nll_error_tol[failed_test_mask])
            print("LFC pooled")
            print(pooled_LFC[to_check_mask][failed_test_mask])
            print("LFC FL")
            print(fl_LFC[to_check_mask][failed_test_mask])
            # We tolerate a few failed genes

            assert np.sum(failed_test_mask) <= tolerated_failed_genes


def get_pooled_results(
    pooled_dds: DeseqDataSet, lfc_mode: Literal["lfc", "mu_init"] = "lfc"
) -> tuple[str, str]:
    """Get the pooled results and set them in the pooled DeseqDataSet.

    Parameters
    ----------
    pooled_dds : DeseqDataSet
        The pooled DeseqDataSet.

    lfc_mode : Literal["lfc", "mu_init"]
        The mode of the ComputeLFC algorithm.
        If "lfc", the results are already computed.
        If "mu_init", the results are computed using the pooled DeseqDataSet, as they
        are not saved in the DeseqDataSet object.

    Returns
    -------
    tuple[str, str]
        The beta and mu parameter names , by which we can access them in the pooled
        dataset.

    """
    if lfc_mode == "lfc":
        return "LFC", "_mu_LFC"

    design_matrix = pooled_dds.obsm["design_matrix"].values

    mle_lfcs_, mu_, _, _ = pooled_dds.inference.irls(
        counts=pooled_dds.X[:, pooled_dds.non_zero_idx],
        size_factors=pooled_dds.obsm["size_factors"],
        design_matrix=design_matrix,
        disp=pooled_dds.varm["_MoM_dispersions"][pooled_dds.non_zero_idx],
        min_mu=pooled_dds.min_mu,
        beta_tol=pooled_dds.beta_tol,
    )

    pooled_dds.varm["_LFC_mu_hat"] = pd.DataFrame(
        np.nan,
        index=pooled_dds.var_names,
        columns=pooled_dds.obsm["design_matrix"].columns,
    )

    pooled_dds.varm["_LFC_mu_hat"].update(
        pd.DataFrame(
            mle_lfcs_,
            index=pooled_dds.non_zero_genes,
            columns=pooled_dds.obsm["design_matrix"].columns,
        )
    )

    pooled_dds.layers["_mu_hat"] = np.full(
        (pooled_dds.n_obs, pooled_dds.n_vars), np.nan
    )
    pooled_dds.layers["_mu_hat"][:, pooled_dds.varm["non_zero"]] = mu_

    return "_LFC_mu_hat", "_mu_hat"


def get_beta_init_pooled(pooled_dds: DeseqDataSet) -> np.ndarray:
    """Get the initial beta values for the pooled DeseqDataSet.

    These initial beta values are used to initialize the optimization
    of the log fold changes.

    Parameters
    ----------
    pooled_dds : DeseqDataSet
        The reference pooled DeseqDataSet from which we want to compute the initial
        value of beta when computing log fold changes.

    Returns
    -------
    np.ndarray
        The initial beta values.

    """
    design_matrix = pooled_dds.obsm["design_matrix"].values
    counts = pooled_dds.X[:, pooled_dds.non_zero_idx]
    size_factors = pooled_dds.obsm["size_factors"]

    num_vars = design_matrix.shape[1]
    X = design_matrix
    if np.linalg.matrix_rank(X) == num_vars:
        Q, R = np.linalg.qr(X)
        y = np.log(counts / size_factors[:, None] + 0.1)
        beta_init = solve(R, Q.T @ y)
    else:  # Initialise intercept with log base mean
        beta_init = np.zeros(num_vars)
        beta_init[0] = np.log(counts / size_factors[:, None]).mean()

    return beta_init.T
