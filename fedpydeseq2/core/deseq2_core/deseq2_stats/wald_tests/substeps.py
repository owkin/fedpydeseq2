from typing import Literal

import anndata as ad
import numpy as np
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.fed_algorithms.fed_irls.utils import (
    make_irls_update_summands_and_nll_batch,
)
from fedpydeseq2.core.utils import build_contrast_vector
from fedpydeseq2.core.utils import wald_test
from fedpydeseq2.core.utils.layers import prepare_cooks_agg
from fedpydeseq2.core.utils.layers import prepare_cooks_local
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocBuildContrastVectorHMatrix:
    """Mixin to get compute contrast vectors and local H matrices."""

    local_adata: ad.AnnData
    num_jobs: int
    joblib_verbosity: int
    joblib_backend: str
    irls_batch_size: int

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    @prepare_cooks_local
    def compute_contrast_vector_and_H_matrix(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        """Build the contrast vector and the local H matrices.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Not used.

        Returns
        -------
        dict
            Contains:
            - local_H_matrix: np.ndarray
                The local H matrix.
            - LFC: np.ndarray
                The log fold changes, in natural log scale.
            - contrast_vector: np.ndarray
                The contrast vector.
        """
        # Build contrast vector and index
        (
            self.local_adata.uns["contrast_vector"],
            self.local_adata.uns["contrast_idx"],
        ) = build_contrast_vector(
            self.local_adata.uns["contrast"],
            self.local_adata.varm["LFC"].columns,
        )

        # ---- Compute the summands for the covariance matrix ---- #

        with parallel_backend(self.joblib_backend):
            res = Parallel(n_jobs=self.num_jobs, verbose=self.joblib_verbosity)(
                delayed(make_irls_update_summands_and_nll_batch)(
                    self.local_adata.obsm["design_matrix"].values,
                    self.local_adata.obsm["size_factors"],
                    self.local_adata.varm["LFC"][i : i + self.irls_batch_size].values,
                    self.local_adata.varm["dispersions"][i : i + self.irls_batch_size],
                    self.local_adata.X[:, i : i + self.irls_batch_size],
                    0,
                )
                for i in range(0, self.local_adata.n_vars, self.irls_batch_size)
            )

        H = np.concatenate([r[0] for r in res])

        return {
            "local_H_matrix": H,
            "LFC": self.local_adata.varm["LFC"],
            "contrast_vector": self.local_adata.uns["contrast_vector"],
        }


class AggRunWaldTests:
    """Mixin to run Wald tests."""

    lfc_null: float
    alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"] | None
    num_jobs: int
    joblib_verbosity: int
    joblib_backend: str

    @remote
    @log_remote
    @prepare_cooks_agg
    def agg_run_wald_tests(self, shared_states: list) -> dict:
        """Run the Wald tests.

        Parameters
        ----------
        shared_states : list
            List of shared states containing:
            - local_H_matrix: np.ndarray
                The local H matrix.
            - LFC: np.ndarray
                The log fold changes, in natural log scale.
            - contrast_vector: np.ndarray
                The contrast vector.

        Returns
        -------
        dict
            Contains:
            - p_values: np.ndarray
                The (unadjusted) p-values (n_genes,).
            - wald_statistics: np.ndarray
                The Wald statistics (n_genes,).
            - wald_se: np.ndarray
                The standard errors of the Wald statistics (n_genes,).
        """
        # First step: aggregate the local H matrices

        H = sum([state["local_H_matrix"] for state in shared_states])

        # Second step: compute the Wald tests in parallel
        with parallel_backend(self.joblib_backend):
            wald_test_results = Parallel(
                n_jobs=self.num_jobs, verbose=self.joblib_verbosity
            )(
                delayed(wald_test)(
                    H[i],
                    shared_states[0]["LFC"].values[i],
                    None,
                    shared_states[0]["contrast_vector"],
                    np.log(2) * self.lfc_null,
                    self.alt_hypothesis,
                )
                for i in range(len(H))
            )

        # Finally, unpack the results
        p_values = np.array([r[0] for r in wald_test_results])
        wald_statistics = np.array([r[1] for r in wald_test_results])
        wald_se = np.array([r[2] for r in wald_test_results])

        return {
            "p_values": p_values,
            "wald_statistics": wald_statistics,
            "wald_se": wald_se,
        }
