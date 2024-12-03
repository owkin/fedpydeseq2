"""A mixin class and a function to implement common test utils for IRLS and PQN."""

from typing import Any

import anndata as ad
import numpy as np
from numpy.linalg import solve
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data
from tests.unit_tests.unit_test_helpers.unit_tester import UnitTester


class FedIRLSPQNTester(UnitTester):
    """A mixin class to implement method for IRLS and PQN testing classes.


    Methods
    -------
    init_local_states
        A remote_data method, which initializes the local states by setting the local
        adata. It also returns the normalized counts and the design matrix, in order
        to create the intial beta (Note that this is only for testing purposes)


    set_beta_init
        A remote_data method, which sets the beta init in the local states, and passes
        on the shared state which is used as an initialization state for the
        Prox Quasi Newton algorithm.


    """

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def init_local_states(
        self, data_from_opener: ad.AnnData, shared_state: Any
    ) -> dict:
        """Initialize the local states.

        This methods sets the local_adata and the local gram matrix.
        from the reference_dds.


        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : Any
            Not used.

        Returns
        -------
        local_states : dict
            Local states containing the local Gram matrix.
        """

        # Using super().init_local_states_from_opener(data_from_opener, shared_state)
        # does not work, so we have to duplicate the code
        self.local_adata = self.local_reference_dds.copy()
        # This field is not saved in pydeseq2 but used in fedpyseq2
        self.local_adata.uns["n_params"] = self.local_adata.obsm["design_matrix"].shape[
            1
        ]
        non_zero_genes_names = self.local_adata.var_names[
            self.local_adata.varm["non_zero"]
        ]

        # Get the local gram matrix for all genes
        design_matrix = self.local_adata.obsm["design_matrix"].values

        # Get the counts on non zero genes
        normed_counts_non_zero = self.local_adata[:, non_zero_genes_names].layers[
            "normed_counts"
        ]

        self.local_adata.uns["_irls_disp_param_name"] = "dispersions"

        return {
            "local_normed_counts_non_zero": normed_counts_non_zero,
            "local_design_matrix": design_matrix,
        }

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def set_beta_init(self, data_from_opener: ad.AnnData, shared_state: dict) -> dict:
        """Set the beta init.

        Since the fed prox newton algorithm requires the beta init to be
        set in the uns, we do so in this step.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            The initial shared state of the fed pqn algorithm.

        Returns
        -------
        local_state : dict
            The same initial shared state.

        """

        self.local_adata.uns["_irls_beta_init"] = shared_state["beta"]
        return shared_state


def compute_initial_beta(shared_states: list[dict]) -> np.ndarray:
    """Compute the beta initialization from a list of shared states.

    Parameters
    ----------
    shared_states : list
        List of shared states.

    Returns
    -------
    np.ndarray
        The beta initialization.
    """

    # Concatenate the local design matrices
    X = np.concatenate([state["local_design_matrix"] for state in shared_states])

    # Concatenate the normed counts
    normed_counts_non_zero = np.concatenate(
        [state["local_normed_counts_non_zero"] for state in shared_states]
    )

    # Compute the beta initialization
    num_vars = X.shape[1]
    n_non_zero_genes = normed_counts_non_zero.shape[1]

    # if full rank, estimate initial betas for IRLS below
    if np.linalg.matrix_rank(X) == num_vars:
        Q, R = np.linalg.qr(X)
        y = np.log(normed_counts_non_zero + 0.1)
        beta_init = solve(R[None, :, :], (Q.T @ y).T[:, :])
    else:  # Initialise intercept with log base mean
        beta_init = np.zeros(n_non_zero_genes, num_vars)
        beta_init[:, 0] = np.log(normed_counts_non_zero).mean(axis=0)

    return beta_init
