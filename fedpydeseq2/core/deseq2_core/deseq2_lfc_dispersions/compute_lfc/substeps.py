"""Module to implement the substeps for the fitting of log fold changes.

This module contains all these substeps as mixin classes.
"""

from typing import Any
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.layers.utils import set_mu_layer
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocGetGramMatrixAndLogFeatures:
    """Mixin accessing the quantities to compute the initial beta of ComputeLFC.

    Attributes
    ----------
    local_adata : AnnData
        The local AnnData object.

    Methods
    -------
    get_gram_matrix_and_log_features
        A remote_data method. Creates the local quantities necessary
        to compute the initial beta.
        If the gram matrix is full rank, it shares the features vector
        and the gram matrix. If the gram matrix is not full rank, it shares
        the normed log means and the number of observations.
    """

    local_adata: AnnData
    refit_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_gram_matrix_and_log_features(
        self,
        data_from_opener: AnnData,
        shared_state: dict[str, Any],
        lfc_mode: Literal["lfc", "mu_init"],
        refit_mode: bool = False,
    ):
        """Create the local quantities necessary to compute the initial beta.

        To do so, we assume that the local_adata.uns contains the following fields:
        - n_params: int
            The number of parameters.
        - _global_gram_matrix: ndarray
            The global gram matrix.

        From the IRLS mode, we will set the following fields:
        - _irls_mu_param_name: str
            The name of the mu parameter, to save at the end of the IRLS run
            This is None if we do not want to save the mu parameter.
        - _irls_beta_param_name: str
            The name of the beta parameter, to save as a varm at the end of the
            fed irls run
            This is None if we do not want to save the beta parameter.
        - _irls_disp_param_name: str
            The name of the dispersion parameter.
        - _lfc_mode: str
            The mode of the IRLS algorithm. This is used to set the previous fields.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used.

        shared_state : dict
            Not used, all the necessary info is stored in the local adata.

        lfc_mode : Literal["lfc", "mu_init"]
            The mode of the IRLS algorithm ("lfc", or "mu_init").

        refit_mode : bool
            Whether to run the pipeline on `refit_adata` instead of `local_adata`.

        Returns
        -------
        dict
            The state to share to the server.
            It always contains the following fields:
            - gram_full_rank: bool
                Whether the gram matrix is full rank.
            - n_non_zero_genes: int
                The number of non zero genes.
            - n_params: int
                The number of parameters.
            - If the gram matrix is full rank, the state contains:
                - local_log_features: ndarray
                    The local log features.
                - global_gram_matrix: ndarray
                    The global gram matrix.
            - If the gram matrix is not full rank, the state contains:
                - normed_log_means: ndarray
                    The normed log means.
                - n_obs: int
                    The number of observations.
        """
        global_gram_matrix = self.local_adata.uns["_global_gram_matrix"]

        if refit_mode:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        # Elements to pass on to the next steps of the method
        if lfc_mode == "lfc":
            adata.uns["_irls_mu_param_name"] = "_mu_LFC"
            adata.uns["_irls_beta_param_name"] = "LFC"
            adata.uns["_irls_disp_param_name"] = "dispersions"
            adata.uns["_lfc_mode"] = "lfc"
        elif lfc_mode == "mu_init":
            adata.uns["_irls_mu_param_name"] = "_irls_mu_hat"
            adata.uns["_irls_beta_param_name"] = "_mu_hat_LFC"
            adata.uns["_irls_disp_param_name"] = "_MoM_dispersions"
            adata.uns["_lfc_mode"] = "mu_init"

        else:
            raise NotImplementedError(
                f"Only 'lfc' and 'mu_init' irls modes are supported, got {lfc_mode}."
            )

        # Get non zero genes
        non_zero_genes_names = adata.var_names[adata.varm["non_zero"]]

        # See if gram matrix is full rank
        gram_full_rank = (
            np.linalg.matrix_rank(global_gram_matrix) == adata.uns["n_params"]
        )
        # If the gram matrix is full rank, share the features vector and the gram
        # matrix

        shared_state = {
            "gram_full_rank": gram_full_rank,
            "n_non_zero_genes": len(non_zero_genes_names),
        }

        if gram_full_rank:
            # Make log features
            design = adata.obsm["design_matrix"].values
            log_counts = np.log(
                adata[:, non_zero_genes_names].layers["normed_counts"] + 0.1
            )
            log_features = (design.T @ log_counts).T
            shared_state.update(
                {
                    "local_log_features": log_features,
                    "global_gram_matrix": global_gram_matrix,
                }
            )
        else:
            # TODO: check that this is correctly recomputed in refit mode
            if "normed_log_means" not in adata.varm:
                with np.errstate(divide="ignore"):  # ignore division by zero warnings
                    log_counts = np.log(adata.layers["normed_counts"])
                    adata.varm["normed_log_means"] = log_counts.mean(0)
            normed_log_means = adata.varm["normed_log_means"]
            n_obs = adata.n_obs
            shared_state.update({"normed_log_means": normed_log_means, "n_obs": n_obs})
        return shared_state


class AggCreateBetaInit:
    """Mixin to create the beta init.

    Methods
    -------
    create_beta_init
        A remote method. Creates the beta init (initialization value for the
        ComputeLFC algorithm) and returns the initialization state for the
        IRLS algorithm containing this initialization value and
        other necessary quantities.
    """

    @remote
    @log_remote
    def create_beta_init(self, shared_states: list[dict]) -> dict[str, Any]:
        """Create the beta init.

        It does so either by solving the least squares regression system if
        the gram matrix is full rank, or by aggregating the log means if the
        gram matrix is not full rank.

        Parameters
        ----------
        shared_states: list[dict]
            A list of dictionaries containing the following
            keys:
            - gram_full_rank: bool
                Whether the gram matrix is full rank.
            - n_non_zero_genes: int
                The number of non zero genes.
            - n_params: int
                The number of parameters.
            If the gram matrix is full rank, the state contains:
            -  local_log_features: ndarray
                The local log features, only if the gram matrix is full rank.
            - global_gram_matrix: ndarray
                The global gram matrix, only if the gram matrix is full rank.
            If the gram matrix is not full rank, the state contains:
            - normed_log_means: ndarray
                The normed log means, only if the gram matrix is not full rank.
            - n_obs: int
                The number of observations, only if the gram matrix is not full rank.


        Returns
        -------
        dict[str, Any]
            A dictionary containing all the necessary info to run IRLS.
            It contains the following fields:
            - beta: ndarray
                The initial beta, of shape (n_non_zero_genes, n_params).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if fed avg should be used for a given gene
                (shape: (n_non_zero_genes,)). Is set to False initially, and will
                be set to True if the gene has diverged.
            - irls_mask: ndarray
                A boolean mask indicating if IRLS should be used for a given gene
                (shape: (n_non_zero_genes,)). Is set to True initially, and will be
                set to False if the gene has converged or diverged.
            - global_nll: ndarray
                The global_nll of the current beta from the previous beta, of shape
                (n_non_zero_genes,).
            - round_number_irls: int
                The current round number of the IRLS algorithm.
        """
        # Get the global quantities
        gram_full_rank = shared_states[0]["gram_full_rank"]
        n_non_zero_genes = shared_states[0]["n_non_zero_genes"]

        # Step 1: Get the beta init
        # Condition on whether or not the gram matrix is full rank
        if gram_full_rank:
            # Get global gram matrix
            global_gram_matrix = shared_states[0]["global_gram_matrix"]

            # Aggregate the feature vectors
            feature_vectors = sum(
                [state["local_log_features"] for state in shared_states]
            )

            # Solve the system
            beta_init = np.linalg.solve(global_gram_matrix, feature_vectors.T).T

        else:
            # Aggregate the log means
            tot_counts = sum([state["n_obs"] for state in shared_states])
            beta_init = (
                sum(
                    [
                        state["normed_log_means"] * state["n_obs"]
                        for state in shared_states
                    ]
                )
                / tot_counts
            )

        # Step 2: instantiate other necessary quantities
        irls_diverged_mask = np.full(n_non_zero_genes, False)
        irls_mask = np.full(n_non_zero_genes, True)
        global_nll = np.full(n_non_zero_genes, 1000.0)

        return {
            "beta": beta_init,
            "irls_diverged_mask": irls_diverged_mask,
            "irls_mask": irls_mask,
            "global_nll": global_nll,
            "round_number_irls": 0,
        }


class LocSaveLFC:
    """Mixin to create the local quantities to compute the final hat matrix.

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

    Methods
    -------
    make_local_final_hat_matrix_summands
        A remote_data method. Creates the local quantities to compute the
        final hat matrix, which must be computed on all genes. This step
        is expected to be applied after catching the IRLS method
        with the fed prox quasi newton method, and takes as an input a
        shared state from the last iteration of that method.
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
    def save_lfc_to_local(
        self,
        data_from_opener: AnnData,
        shared_state: dict[str, Any],
        refit_mode: bool = False,
    ):
        """Create the local quantities to compute the final hat matrix.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used.

        shared_state : dict
            The shared state.
            The shared state is a dictionary containing the following
            keys:
            - beta: ndarray
                The current beta, of shape (n_non_zero_genes, n_params).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if the irsl method has diverged.
                In that case, these genes are caught with the fed prox newton
                method.
                (shape: (n_non_zero_genes,)).
            - PQN_diverged_mask: ndarray
                A boolean mask indicating if the fed prox newton method has
                diverged. These genes are not caught by any method, and the
                returned beta value is the output of the PQN method, even
                though it has not converged.

        refit_mode : bool
            Whether to run the pipeline on `refit_adata` instead of `local_adata`.
            (default: False).
        """
        beta = shared_state["beta"]

        if refit_mode:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        # TODO keeping this in memory for now, see if need for removal at the end
        adata.uns["_irls_diverged_mask"] = shared_state["irls_diverged_mask"]
        adata.uns["_PQN_diverged_mask"] = shared_state["PQN_diverged_mask"]

        # Get the param names stored in the local adata
        mu_param_name = adata.uns["_irls_mu_param_name"]
        beta_param_name = adata.uns["_irls_beta_param_name"]
        # ---- Step 2: Store the mu, the diagonal of the hat matrix  ---- #
        # ----           and beta in the adata                       ---- #

        design_column_names = adata.obsm["design_matrix"].columns

        non_zero_genes_names = adata.var_names[adata.varm["non_zero"]]

        beta_dataframe = pd.DataFrame(
            np.NaN, index=adata.var_names, columns=design_column_names
        )
        beta_dataframe.loc[non_zero_genes_names, :] = beta

        adata.varm[beta_param_name] = beta_dataframe

        if mu_param_name is not None:
            set_mu_layer(
                local_adata=adata,
                lfc_param_name=beta_param_name,
                mu_param_name=mu_param_name,
                n_jobs=self.num_jobs,
                joblib_verbosity=self.joblib_verbosity,
                joblib_backend=self.joblib_backend,
                batch_size=self.irls_batch_size,
            )
