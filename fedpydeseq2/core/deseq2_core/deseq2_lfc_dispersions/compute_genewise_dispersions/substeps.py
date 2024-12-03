from anndata import AnnData
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.layers.build_layers import set_fit_lin_mu_hat
from fedpydeseq2.core.utils.layers.build_layers import set_mu_hat_layer
from fedpydeseq2.core.utils.logging import log_remote_data


class LocLinMu:
    """Mixin to fit linear mu estimates locally."""

    local_adata: AnnData
    refit_adata: AnnData
    min_mu: float
    max_disp: float

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def fit_lin_mu(
        self, data_from_opener, shared_state, min_mu=0.5, refit_mode: bool = False
    ):
        """Fit linear mu estimates and store them locally.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            Not used.

        shared_state : dict
            Contains values to be saved in local adata:
            - "MoM_dispersions": MoM dispersions,
            - "nom_zero": Mask of all zero genes,
            - "tot_num_samples": Total number of samples.

        min_mu : float
            Lower threshold for fitted means, for numerical stability.
            (default: ``0.5``).

        refit_mode : bool
            Whether to run the pipeline in refit mode. If True, the pipeline will be run
            on `refit_adata`s instead of `local_adata`s. (default: ``False``).

        """
        if refit_mode:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        # save MoM dispersions computed in the previous step
        adata.varm["_MoM_dispersions"] = shared_state["MoM_dispersions"]

        # save mask of all zero genes.
        # TODO: check that we should also do this in refit mode
        adata.varm["non_zero"] = shared_state["non_zero"]

        if not refit_mode:  # In refit mode, those are unchanged
            # save the total number of samples
            self.local_adata.uns["tot_num_samples"] = shared_state["tot_num_samples"]

            # use it to set max_disp
            self.local_adata.uns["max_disp"] = max(
                self.max_disp, self.local_adata.uns["tot_num_samples"]
            )

        # save the base_mean for independent filtering
        adata.varm["_normed_means"] = shared_state["tot_counts_mean"]

        # compute mu_hat
        set_fit_lin_mu_hat(adata, min_mu=min_mu)


class LocSetMuHat:
    """Mixin to set mu estimates locally."""

    local_adata: AnnData
    refit_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def set_mu_hat(
        self,
        data_from_opener,
        shared_state,
        refit_mode: bool = False,
    ) -> None:
        """Pick between linear and IRLS mu estimates.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            Not used.

        shared_state : dict
            Not used.

        refit_mode : bool
            Whether to run on `refit_adata`s instead of `local_adata`s.
            (default: ``False``).
        """
        if refit_mode:
            adata = self.refit_adata
        else:
            adata = self.local_adata
        # TODO make sure that the adata has the num_replicates and the n_params
        set_mu_hat_layer(adata)
        del adata.layers["_fit_lin_mu_hat"]
        del adata.layers["_irls_mu_hat"]
