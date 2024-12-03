import pandas as pd
from anndata import AnnData
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import prepare_cooks_agg
from fedpydeseq2.core.utils.layers import prepare_cooks_local
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.layers.build_layers import set_sqerror_layer
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocComputeSqerror:
    """Compute the squared error between the normalized counts and the trimmed mean."""

    local_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def local_compute_sqerror(
        self,
        data_from_opener,
        shared_state=dict,
    ) -> None:
        """
        Compute the squared error between the normalized counts and the trimmed mean.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict, optional
            Results to save in the local states.

        """
        cell_means = shared_state["trimmed_mean_normed_counts"]
        if isinstance(cell_means, pd.DataFrame):
            cell_means.index = self.local_adata.var_names
            self.local_adata.varm["cell_means"] = cell_means
        else:
            # In this case, the cell means are not computed per
            # level but overall
            self.local_adata.varm["cell_means"] = cell_means
        set_sqerror_layer(self.local_adata)


class LocGetNormedCounts:
    """Get the mean of the normalized counts."""

    local_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    @prepare_cooks_local
    def local_get_normed_count_means(
        self,
        data_from_opener,
        shared_state=dict,
    ) -> dict:
        """
        Send local normed counts means.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict, optional
            Dictionary with the following keys:
            - varEst: variance estimate for Cook's distance calculation

        Returns
        -------
        dict
            Because of the decorator, dictionary with the following keys:
            - mean_normed_counts: mean of the normalized counts
            - n_samples: number of samples
            - varEst: variance estimate

        """
        return {}


class AggComputeDispersionForCook:
    """Compute the dispersion for Cook's distance calculation."""

    @remote
    @log_remote
    @prepare_cooks_agg
    def agg_compute_dispersion_for_cook(
        self,
        shared_states: list[dict],
    ) -> dict:
        """
        Compute the dispersion for Cook's distance calculation.

        Parameters
        ----------
        shared_states : list[dict]
            list of shared states with the following keys:
            - mean_normed_counts: mean of the normalized counts
            - n_samples: number of samples
            - varEst: variance estimate

        Returns
        -------
        dict
            Because it is decorated, the dictionary will have the following key:
            - cooks_dispersions: dispersion values

        """
        return {}
