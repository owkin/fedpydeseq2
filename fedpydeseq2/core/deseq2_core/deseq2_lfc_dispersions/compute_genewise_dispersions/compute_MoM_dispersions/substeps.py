"""Module to implement the substeps for the rough dispersions step.

This module contains all these substeps as mixin classes.
"""

import numpy as np
from anndata import AnnData
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.layers.build_layers import set_y_hat
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class AggRoughDispersion:
    """Mixin to aggregate local rough dispersions."""

    @remote
    @log_remote
    def aggregate_rough_dispersions(self, shared_states):
        """Aggregate local rough dispersions.

        Parameters
        ----------
        shared_states : list
            List of results (rough_dispersions, n_obs, n_params) from training nodes.

        Returns
        -------
        dict
            Global rough dispersions.
        """
        rough_dispersions = sum(
            [state["local_rough_dispersions"] for state in shared_states]
        )

        tot_obs = sum([state["local_n_obs"] for state in shared_states])
        n_params = shared_states[0]["local_n_params"]

        if tot_obs <= n_params:
            raise ValueError(
                "The number of samples is smaller or equal to the number of design "
                "variables, i.e., there are no replicates to estimate the "
                "dispersions. Please use a design with fewer variables."
            )

        return {
            "rough_dispersions": np.maximum(rough_dispersions / (tot_obs - n_params), 0)
        }


class AggCreateRoughDispersionsSystem:
    """Mixin to solve the linear system for rough dispersions."""

    @remote
    @log_remote
    def create_rough_dispersions_system(self, shared_states, refit_mode: bool = False):
        """Solve the linear system in for rough dispersions.

        Parameters
        ----------
        shared_states : list
            List of results (local_gram_matrix, local_features) from training nodes.

        refit_mode : bool
            Whether to run the pipeline in refit mode, after cooks outliers were
            replaced. If True, there is no need to compute the Gram matrix which was
            already computed in the compute_size_factors step (default: False).

        Returns
        -------
        dict
            The global feature vector and the global hat matrix if refit_mode is
            ``False``.
        """
        shared_state = {
            "global_feature_vector": sum(
                [state["local_features"] for state in shared_states]
            )
        }
        if not refit_mode:
            shared_state["global_gram_matrix"] = sum(
                [state["local_gram_matrix"] for state in shared_states]
            )

        return shared_state


class LocRoughDispersion:
    """Mixin to compute local rough dispersions."""

    local_adata: AnnData
    refit_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def local_rough_dispersions(
        self, data_from_opener, shared_state, refit_mode: bool = False
    ) -> dict:
        """Compute local rough dispersions, and save the global gram matrix.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Shared state containing
                - the gram matrix, if refit_mode is ``False``,
                - the global feature vector.

        refit_mode : bool
            Whether to run the pipeline in refit mode, after cooks outliers were
            replaced. If True, the pipeline will be run on `refit_adata`s instead of
            `local_adata`s (default: False).

        Returns
        -------
        dict
            Dictionary containing local rough dispersions, number of samples and
            number of parameters (i.e. number of columns in the design matrix).
        """
        if not refit_mode:
            global_gram_matrix = shared_state["global_gram_matrix"]
            self.local_adata.uns["_global_gram_matrix"] = global_gram_matrix
        else:
            global_gram_matrix = self.local_adata.uns["_global_gram_matrix"]

        beta_rough_dispersions = np.linalg.solve(
            global_gram_matrix, shared_state["global_feature_vector"]
        )

        if refit_mode:
            adata = self.refit_adata
        else:
            adata = self.local_adata
        adata.varm["_beta_rough_dispersions"] = beta_rough_dispersions.T
        # Save the rough dispersions beta so that we can reconstruct y_hat
        set_y_hat(adata)

        # Save global beta in the local data because so it can be used later in
        # fit_lin_mu. Do it before clipping.

        y_hat = np.maximum(adata.layers["_y_hat"], 1)
        unnormed_alpha_rde = (
            ((adata.layers["normed_counts"] - y_hat) ** 2 - y_hat) / (y_hat**2)
        ).sum(0)
        return {
            "local_rough_dispersions": unnormed_alpha_rde,
            "local_n_obs": adata.n_obs,
            "local_n_params": adata.uns["n_params"],
        }


class LocInvSizeMean:
    """Mixin to compute local means of inverse size factors."""

    local_adata: AnnData
    refit_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def local_inverse_size_mean(
        self, data_from_opener, shared_state=None, refit_mode: bool = False
    ) -> dict:
        """Compute local means of inverse size factors, counts, and squared counts.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Shared state containing rough dispersions from aggregator.

        refit_mode : bool
            Whether to run the pipeline in refit mode, after cooks outliers were
            replaced. If True, the pipeline will be run on `refit_adata`s instead of
            `local_adata`s (default: False).

        Returns
        -------
        dict
            dictionary containing all quantities required to compute MoM dispersions:
            local inverse size factor means, counts means, squared counts means,
            rough dispersions and number of samples.
        """
        if refit_mode:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        adata.varm["_rough_dispersions"] = shared_state["rough_dispersions"]

        return {
            "local_inverse_size_mean": (1 / adata.obsm["size_factors"]).mean(),
            "local_counts_mean": adata.layers["normed_counts"].mean(0),
            "local_squared_squared_mean": (adata.layers["normed_counts"] ** 2).mean(0),
            "local_n_obs": adata.n_obs,
            # Pass rough dispersions to the aggregation node, to compute MoM dispersions
            "rough_dispersions": shared_state["rough_dispersions"],
        }


class AggMomentsDispersion:
    """Mixin to compute MoM dispersions."""

    local_adata: AnnData
    max_disp: float
    min_disp: float

    @remote
    @log_remote
    def aggregate_moments_dispersions(self, shared_states):
        """Compute global moments dispersions.

        Parameters
        ----------
        shared_states : list
            List of results (local_inverse_size_mean, local_counts_mean,
            local_squared_squared_mean, local_n_obs, rough_dispersions)
            from training nodes.

        Returns
        -------
        dict
            Global moments dispersions, the mask of all zero genes, the total
            number of samples (used to set max_disp and lr), and
            the total normed counts mean (used in the independent filtering
            step).
        """
        tot_n_obs = sum([state["local_n_obs"] for state in shared_states])

        # Compute the mean of inverse size factors
        tot_inv_size_mean = (
            sum(
                [
                    state["local_n_obs"] * state["local_inverse_size_mean"]
                    for state in shared_states
                ]
            )
            / tot_n_obs
        )

        # Compute the mean and variance of normalized counts

        tot_counts_mean = (
            sum(
                [
                    state["local_n_obs"] * state["local_counts_mean"]
                    for state in shared_states
                ]
            )
            / tot_n_obs
        )
        non_zero = tot_counts_mean != 0

        tot_squared_mean = (
            sum(
                [
                    state["local_n_obs"] * state["local_squared_squared_mean"]
                    for state in shared_states
                ]
            )
            / tot_n_obs
        )

        counts_variance = (
            tot_n_obs / (tot_n_obs - 1) * (tot_squared_mean - tot_counts_mean**2)
        )

        moments_dispersions = np.zeros(
            counts_variance.shape, dtype=counts_variance.dtype
        )
        moments_dispersions[non_zero] = (
            counts_variance[non_zero] - tot_inv_size_mean * tot_counts_mean[non_zero]
        ) / tot_counts_mean[non_zero] ** 2

        # Get rough dispersions from the first center
        rough_dispersions = shared_states[0]["rough_dispersions"]

        # Compute the maximum dispersion
        max_disp = np.maximum(self.max_disp, tot_n_obs)

        # Return moment estimate
        alpha_hat = np.minimum(rough_dispersions, moments_dispersions)
        MoM_dispersions = np.clip(alpha_hat, self.min_disp, max_disp)

        # Set MoM dispersions of all zero genes to NaN

        MoM_dispersions[~non_zero] = np.nan
        return {
            "MoM_dispersions": MoM_dispersions,
            "non_zero": non_zero,
            "tot_num_samples": tot_n_obs,
            "tot_counts_mean": tot_counts_mean,
        }
