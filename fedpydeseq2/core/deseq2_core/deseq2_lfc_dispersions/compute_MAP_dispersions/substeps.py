import numpy as np
from anndata import AnnData
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data


class LocFilterMAPDispersions:
    """Mixin to filter MAP dispersions and obtain the final dispersion estimates."""

    local_adata: AnnData
    refit_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def filter_outlier_genes(
        self,
        data_from_opener,
        shared_state,
        refit_mode: bool = False,
    ) -> None:
        """Filter out outlier genes.

        Avoids shrinking the dispersions of genes that are too far from the trend curve.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            Not used.

        shared_state : dict
            Contains:
            - "MAP_dispersions": MAP dispersions,

        refit_mode : bool
            Whether to run the pipeline on `refit_adata`s instead of `local_adata`s.
            (default: False).
        """
        if refit_mode:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        adata.varm["MAP_dispersions"] = shared_state["MAP_dispersions"].copy()

        adata.varm["dispersions"] = adata.varm["MAP_dispersions"].copy()
        adata.varm["_outlier_genes"] = np.log(
            adata.varm["genewise_dispersions"]
        ) > np.log(adata.varm["fitted_dispersions"]) + 2 * np.sqrt(
            adata.uns["_squared_logres"]
        )
        adata.varm["dispersions"][adata.varm["_outlier_genes"]] = adata.varm[
            "genewise_dispersions"
        ][adata.varm["_outlier_genes"]]
