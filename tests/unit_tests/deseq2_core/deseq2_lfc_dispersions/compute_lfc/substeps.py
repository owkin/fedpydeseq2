"""Substeps for the ComputeLFC testing class, to aggregate information."""


import numpy as np
from anndata import AnnData
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocGetLocalComputeLFCResults:
    """Get the local ComputeLFC results.

    Attributes
    ----------
    local_adata: AnnData
        The local AnnData.


    Methods
    -------
    get_local_irls_results
        Get the local ComputeLFC results.
    """

    local_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_local_irls_results(self, data_from_opener: AnnData, shared_state: dict):
        """Get the local ComputeLFC results.

        Parameters
        ----------
        data_from_opener: AnnData
            Not used.

        shared_state: dict
            Not used.

        Returns
        -------
        dict
            The state to share to the server.
            It contains the following fields:
            - beta: ndarray or None
                The current beta, of shape (n_non_zero_genes, n_params).
            - mu: ndarray or None
                The mu, of shape (n_obs, n_non_zero_genes).
            - hat_diag: ndarray or None
                The hat diagonal, of shape (n_obs, n_non_zero_genes).
            - gene_names: list[str]
                The names of the genes that are still active (non zero gene names
                on the irls_mask).
            - lfc_mode: str
                The mode of the ComputeLFC algorithm.
                For now, only "lfc" is supported.
            - mu_param_name: str or None
                The name of the mu parameter in the adata.
            - beta_param_name: str
                The name of the beta parameter in the adata.
        """
        mu_param_name = self.local_adata.uns["_irls_mu_param_name"]
        beta_param_name = self.local_adata.uns["_irls_beta_param_name"]
        lfc_mode = self.local_adata.uns["_lfc_mode"]

        irls_diverged_mask = self.local_adata.uns["_irls_diverged_mask"]
        PQN_diverged_mask = self.local_adata.uns["_PQN_diverged_mask"]

        non_zero_genes_names = self.local_adata.var_names[
            self.local_adata.varm["non_zero"]
        ]
        non_zero_genes_mask = self.local_adata.varm["non_zero"]

        # Get the initial beta value
        beta_init = self.local_adata.uns["_irls_beta_init"]

        # Get beta from the varm of the local adata
        beta_dataframe = self.local_adata.varm[beta_param_name]
        beta: np.ndarray | None = beta_dataframe.loc[non_zero_genes_names, :].to_numpy()
        assert beta is not None
        beta_irls_converged = beta[~irls_diverged_mask]
        beta_PQN_converged = beta[irls_diverged_mask & ~PQN_diverged_mask]
        beta_all_diverged = beta[irls_diverged_mask & PQN_diverged_mask]

        # Get mu from the layers of the local adata
        if mu_param_name is not None:
            mu: np.ndarray | None = self.local_adata.layers[mu_param_name][
                :, non_zero_genes_mask
            ]
            assert mu is not None
            mu_irls_converged = mu[:, ~irls_diverged_mask]
            mu_PQN_converged = mu[:, irls_diverged_mask & ~PQN_diverged_mask]
            mu_all_diverged = mu[:, irls_diverged_mask & PQN_diverged_mask]
        else:
            mu_irls_converged = None
            mu_PQN_converged = None
            mu_all_diverged = None

        irls_genes = non_zero_genes_names[~irls_diverged_mask]
        PQN_genes = non_zero_genes_names[irls_diverged_mask & ~PQN_diverged_mask]
        all_diverged_genes = non_zero_genes_names[
            irls_diverged_mask & PQN_diverged_mask
        ]

        # Get the sample ids of the local adata
        sample_ids = self.local_adata.obs_names

        shared_state = {
            "beta_param_name": beta_param_name,
            "mu_param_name": mu_param_name,
            "beta_irls_converged": beta_irls_converged,
            "beta_PQN_converged": beta_PQN_converged,
            "beta_all_diverged": beta_all_diverged,
            "mu_irls_converged": mu_irls_converged,
            "mu_PQN_converged": mu_PQN_converged,
            "mu_all_diverged": mu_all_diverged,
            "irls_genes": irls_genes,
            "PQN_genes": PQN_genes,
            "all_diverged_genes": all_diverged_genes,
            "lfc_mode": lfc_mode,
            "beta_init": beta_init,
            "sample_ids": sample_ids,
        }
        return shared_state


class AggConcatenateHandMu:
    """Mixin to concatenate the hat matrix and mu.

    Methods
    -------
    concatenate_irls_outputs
        Concatenate that hat and mu matrices, in order to save these outputs for
        evaluation.
    """

    @remote
    @log_remote
    def concatenate_irls_outputs(self, shared_states: dict):
        """Concatenate that hat and mu matrices.

        Parameters
        ----------
        shared_states : list[dict]
            The shared states.
            It is a list of dictionaries containing the following
            keys:
            - beta: ndarray or None
                The current beta, of shape (n_non_zero_genes, n_params).
            - mu: ndarray or None
                The mu, of shape (n_obs, n_non_zero_genes).
            - hat_diag: ndarray or None
                The hat diagonal, of shape (n_obs, n_non_zero_genes).
            - gene_names: list[str]
                The names of the genes that are still active (non zero gene names
                on the irls_mask).
            - lfc_mode: str
                The mode of the ComputeLFC algorithm.
                For now, only "lfc" is supported.
            - mu_param_name: str or None
                The name of the mu parameter in the adata.
            - beta_param_name: str or None
                The name of the beta parameter in the adata.


        Returns
        -------
        dict
            A dictionary of results containing the following fields:
            - beta_param_name: np.ndarray or None
                The current beta, of shape (n_non_zero_genes, n_params).
            - mu_param_name: np.ndarray or None
                The mu, of shape (n_obs, n_non_zero_genes).
            - {lfc_mode}_gene_names: list[str]
                The names of the genes that are still active (non zero gene names)
        """
        # Get the sample independent quantities
        beta_irls_converged = shared_states[0]["beta_irls_converged"]
        beta_PQN_converged = shared_states[0]["beta_PQN_converged"]
        beta_all_diverged = shared_states[0]["beta_all_diverged"]
        irls_genes = shared_states[0]["irls_genes"]
        PQN_genes = shared_states[0]["PQN_genes"]
        all_diverged_genes = shared_states[0]["all_diverged_genes"]
        beta_param_name = shared_states[0]["beta_param_name"]
        mu_param_name = shared_states[0]["mu_param_name"]
        lfc_mode = shared_states[0]["lfc_mode"]
        beta_init = shared_states[0]["beta_init"]

        # Concatenate mu
        if mu_param_name is not None:
            mu_irls_converged: np.ndarray | None = np.concatenate(
                [state["mu_irls_converged"] for state in shared_states], axis=0
            )
            mu_PQN_converged: np.ndarray | None = np.concatenate(
                [state["mu_PQN_converged"] for state in shared_states], axis=0
            )
            mu_all_diverged: np.ndarray | None = np.concatenate(
                [state["mu_all_diverged"] for state in shared_states], axis=0
            )
        else:
            mu_irls_converged = None
            mu_PQN_converged = None
            mu_all_diverged = None

        # Conctenate sample ids
        sample_ids = np.concatenate(
            [state["sample_ids"] for state in shared_states], axis=0
        )

        self.results = {
            "beta_param_name": beta_param_name,
            "mu_param_name": mu_param_name,
            f"{beta_param_name}_irls_converged": beta_irls_converged,
            f"{beta_param_name}_PQN_converged": beta_PQN_converged,
            f"{beta_param_name}_all_diverged": beta_all_diverged,
            f"{mu_param_name}_irls_converged": mu_irls_converged,
            f"{mu_param_name}_PQN_converged": mu_PQN_converged,
            f"{mu_param_name}_all_diverged": mu_all_diverged,
            f"{lfc_mode}_irls_genes": irls_genes,
            f"{lfc_mode}_PQN_genes": PQN_genes,
            f"{lfc_mode}_all_diverged_genes": all_diverged_genes,
            f"{lfc_mode}_beta_init": beta_init,
            "sample_ids": sample_ids,
        }
