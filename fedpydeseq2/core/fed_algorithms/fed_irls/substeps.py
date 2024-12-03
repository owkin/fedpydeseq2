"""Module to implement the substeps for the fitting of log fold changes.

This module contains all these substeps as mixin classes.
"""

from typing import Any

import numpy as np
from anndata import AnnData
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.fed_algorithms.fed_irls.utils import (
    make_irls_update_summands_and_nll_batch,
)
from fedpydeseq2.core.utils.compute_lfc_utils import get_lfc_utils_from_gene_mask_adata
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocMakeIRLSSummands:
    """Mixin to make the summands for the IRLS algorithm.

    Attributes
    ----------
    local_adata : AnnData
        The local AnnData object.
    num_jobs : int
        The number of cpus to use.
    joblib_verbosity : int
        The verbosity of the joblib backend.
    joblib_backend : str
        The backend to use for the joblib parallelization.
    irls_batch_size : int
        The batch size to use for the IRLS algorithm.
    min_mu : float
        The minimum value for the mu parameter.
    irls_num_iter : int
        The number of iterations for the IRLS algorithm.

    Methods
    -------
    make_local_irls_summands_and_nlls
        A remote_data method. Makes the summands for the IRLS algorithm.
        It also passes on the necessary global quantities.

    """

    local_adata: AnnData
    refit_adata: AnnData
    num_jobs: int
    joblib_verbosity: int
    joblib_backend: str
    irls_batch_size: int
    min_mu: float
    irls_num_iter: int

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def make_local_irls_summands_and_nlls(
        self,
        data_from_opener: AnnData,
        shared_state: dict[str, Any],
        refit_mode: bool = False,
    ):
        """Make the summands for the IRLS algorithm.

        This functions does two main operations:

        1) It computes the summands for the beta update.
        2) It computes the local quantities to compute the global_nll
        of the current beta


        Parameters
        ----------
        data_from_opener : AnnData
            Not used.

        shared_state : dict
            A dictionary containing the following
            keys:
            - beta: ndarray
                The current beta, of shape (n_non_zero_genes, n_params).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if fed avg should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - irls_mask: ndarray
                A boolean mask indicating if IRLS should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - global_nll: ndarray
                The global_nll of the current beta from the previous beta, of shape\
                (n_non_zero_genes,).
            - round_number_irls: int
                The current round number of the IRLS algorithm.

        refit_mode : bool
            Whether to run on `refit_adata`s instead of `local_adata`s.
            (default: False).

        Returns
        -------
        dict
            The state to share to the server.
            It contains the following fields:
            - beta: ndarray
                The current beta, of shape (n_non_zero_genes, n_params).
            - local_nll: ndarray
                The local nll of the current beta, of shape (n_irls_genes,).
            - local_hat_matrix: ndarray
                The local hat matrix, of shape (n_irls_genes, n_params, n_params).
                n_irsl_genes is the number of genes that are still active (non zero
                gene names on the irls_mask).
            - local_features: ndarray
                The local features, of shape (n_irls_genes, n_params).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if fed avg should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - irls_mask: ndarray
                A boolean mask indicating if IRLS should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - global_nll: ndarray
                The global_nll of the current beta of shape
                (n_non_zero_genes,).
                This parameter is simply passed to the next shared state
            - round_number_irls: int
                The current round number of the IRLS algorithm.
                This round number is not updated here.

        """
        if refit_mode:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        # Put all elements in the shared state in readable variables
        beta = shared_state["beta"]
        irls_mask = shared_state["irls_mask"]
        irls_diverged_mask = shared_state["irls_diverged_mask"]
        global_nll = shared_state["global_nll"]
        round_number_irls = shared_state["round_number_irls"]

        # Get the quantitie stored in the adata
        disp_param_name = adata.uns["_irls_disp_param_name"]

        # If this is the first round, save the beta init in a field of the local adata
        if round_number_irls == 0:
            adata.uns["_irls_beta_init"] = beta.copy()

        (
            irls_gene_names,
            design_matrix,
            size_factors,
            counts,
            dispersions,
            beta_genes,
        ) = get_lfc_utils_from_gene_mask_adata(
            adata, irls_mask, beta=beta, disp_param_name=disp_param_name
        )

        # ---- Compute the summands for the beta update and the local nll ---- #

        with parallel_backend(self.joblib_backend):
            res = Parallel(n_jobs=self.num_jobs, verbose=self.joblib_verbosity)(
                delayed(make_irls_update_summands_and_nll_batch)(
                    design_matrix,
                    size_factors,
                    beta_genes[i : i + self.irls_batch_size],
                    dispersions[i : i + self.irls_batch_size],
                    counts[:, i : i + self.irls_batch_size],
                    self.min_mu,
                )
                for i in range(0, len(beta_genes), self.irls_batch_size)
            )

        if len(res) == 0:
            H = np.zeros((0, beta.shape[1], beta.shape[1]))
            y = np.zeros((0, beta.shape[1]))
            local_nll = np.zeros(0)
        else:
            H = np.concatenate([r[0] for r in res])
            y = np.concatenate([r[1] for r in res])
            local_nll = np.concatenate([r[2] for r in res])

        # Create the shared state
        return {
            "beta": beta,
            "local_nll": local_nll,
            "local_hat_matrix": H,
            "local_features": y,
            "irls_gene_names": irls_gene_names,
            "irls_diverged_mask": irls_diverged_mask,
            "irls_mask": irls_mask,
            "global_nll": global_nll,
            "round_number_irls": round_number_irls,
        }


class AggMakeIRLSUpdate:
    """Mixin class to aggregate IRLS summands.

    Please refer to the method make_local_irls_summands_and_nlls for more.

    Attributes
    ----------
    num_jobs : int
        The number of cpus to use.
    joblib_verbosity : int
        The verbosity of the joblib backend.
    joblib_backend : str
        The backend to use for the joblib parallelization.
    irls_batch_size : int
        The batch size to use for the IRLS algorithm.
    max_beta : float
        The maximum value for the beta parameter.
    beta_tol : float
        The tolerance for the beta parameter.
    irls_num_iter : int
        The number of iterations for the IRLS algorithm.

    Methods
    -------
    make_global_irls_update
        A remote method. Aggregates the local quantities to create
        the global IRLS update. It also updates the masks indicating which genes
        have diverged or converged according to the deviance.

    """

    num_jobs: int
    joblib_verbosity: int
    joblib_backend: str
    irls_batch_size: int
    max_beta: float
    beta_tol: float
    irls_num_iter: int

    @remote
    @log_remote
    def make_global_irls_update(self, shared_states: list[dict]) -> dict[str, Any]:
        """Make the summands for the IRLS algorithm.

        The role of this function is twofold.

        1) It computes the global_nll and updates the masks according to the deviance,
        for the beta values that have been computed in the previous round.

        2) It aggregates the local hat matrix and features to solve the linear system
        and get the new beta values.

        Parameters
        ----------
        shared_states: list[dict]
            A list of dictionaries containing the following
            keys:
            - local_hat_matrix: ndarray
                The local hat matrix, of shape (n_irls_genes, n_params, n_params).
                n_irsl_genes is the number of genes that are still active (non zero
                gene names on the irls_mask).
            - local_features: ndarray
                The local features, of shape (n_irls_genes, n_params).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if fed avg should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - irls_mask: ndarray
                A boolean mask indicating if IRLS should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - global_nll: ndarray
                The global_nll of the current beta from the previous beta, of shape\
                (n_non_zero_genes,).
            - round_number_irls: int
                The current round number of the IRLS algorithm.

        Returns
        -------
        dict[str, Any]
            A dictionary containing all the necessary info to run IRLS.
            It contains the following fields:
            - beta: ndarray
                The log fold changes, of shape (n_non_zero_genes, n_params).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if fed avg should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - irls_mask: ndarray
                A boolean mask indicating if IRLS should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - global_nll: ndarray
                The global_nll of the current beta from the previous beta, of shape\
                (n_non_zero_genes,).
            - round_number_irls: int
                The current round number of the IRLS algorithm.


        """
        # Load main params from the first state
        beta = shared_states[0]["beta"]
        irls_mask = shared_states[0]["irls_mask"]
        irls_diverged_mask = shared_states[0]["irls_diverged_mask"]
        global_nll = shared_states[0]["global_nll"]
        round_number_irls = shared_states[0]["round_number_irls"]

        # ---- Step 0: Aggregate the local hat matrix, features and global_nll ---- #

        global_hat_matrix = sum([state["local_hat_matrix"] for state in shared_states])
        global_features = sum([state["local_features"] for state in shared_states])
        global_nll_on_irls_mask = sum([state["local_nll"] for state in shared_states])

        # ---- Step 1: update global_nll and masks ---- #

        # The first round needs to be handled separately
        if round_number_irls == 0:
            # In that case, the irls_masks consists in all True values
            # We only need set the initial global_nll
            global_nll = global_nll_on_irls_mask

        else:
            old_global_nll = global_nll.copy()
            old_irls_mask = irls_mask.copy()

            global_nll[irls_mask] = global_nll_on_irls_mask

            # Set the new masks with the dev ratio and beta values
            deviance_ratio = np.abs(2 * global_nll - 2 * old_global_nll) / (
                np.abs(2 * global_nll) + 0.1
            )
            irls_diverged_mask = irls_diverged_mask | (
                np.abs(beta) > self.max_beta
            ).any(axis=1)

            irls_mask = irls_mask & (deviance_ratio > self.beta_tol)
            irls_mask = irls_mask & ~irls_diverged_mask
            new_mask_in_old_mask = (irls_mask & old_irls_mask)[old_irls_mask]
            global_hat_matrix = global_hat_matrix[new_mask_in_old_mask]
            global_features = global_features[new_mask_in_old_mask]

        if round_number_irls == self.irls_num_iter:
            # In this case, we must prepare the switch to fed prox newton
            return {
                "beta": beta,
                "irls_diverged_mask": irls_diverged_mask,
                "irls_mask": irls_mask,
                "global_nll": global_nll,
                "round_number_irls": round_number_irls,
            }

        # ---- Step 2: Solve the system to compute beta ---- #

        ridge_factor = np.diag(np.repeat(1e-6, global_hat_matrix.shape[1]))
        with parallel_backend(self.joblib_backend):
            res = Parallel(n_jobs=self.num_jobs, verbose=self.joblib_verbosity)(
                delayed(np.linalg.solve)(
                    global_hat_matrix[i : i + self.irls_batch_size] + ridge_factor,
                    global_features[i : i + self.irls_batch_size],
                )
                for i in range(0, len(global_hat_matrix), self.irls_batch_size)
            )
        if len(res) > 0:
            beta_hat = np.concatenate(res)
        else:
            beta_hat = np.zeros((0, global_hat_matrix.shape[1]))

        # TODO :  it would be cleaner to pass an update, which is None at the first
        #  round. That way we do not update beta in a different step its evaluation.

        # Update the beta
        beta[irls_mask] = beta_hat

        round_number_irls = round_number_irls + 1

        return {
            "beta": beta,
            "irls_diverged_mask": irls_diverged_mask,
            "irls_mask": irls_mask,
            "global_nll": global_nll,
            "round_number_irls": round_number_irls,
        }
