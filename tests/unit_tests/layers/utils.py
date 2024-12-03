"""Module to test create dummy AnnData with layers."""

import anndata as ad
import numpy as np
import pandas as pd

from fedpydeseq2.core.utils.layers import set_mu_layer
from fedpydeseq2.core.utils.layers.build_layers import set_fit_lin_mu_hat
from fedpydeseq2.core.utils.layers.build_layers import set_mu_hat_layer
from fedpydeseq2.core.utils.layers.build_layers import set_normed_counts
from fedpydeseq2.core.utils.layers.build_layers import set_sqerror_layer
from fedpydeseq2.core.utils.layers.build_layers import set_y_hat


def create_dummy_adata_with_layers(
    data_from_opener: ad.AnnData,
    num_row_values_equal_n_params=False,
    add_cooks=False,
    n_params=5,
) -> ad.AnnData:
    """Create a dummy AnnData object with the necessary layers for testing.

    Parameters
    ----------
    data_from_opener : ad.AnnData
        The data from the opener.

    num_row_values_equal_n_params : bool, optional
        Whether the number of values taken by the design is equal to
        the number of parameters.
        If False, the number of row values is equal to the number of parameters + 1.
        Defaults to False.

    add_cooks : bool, optional
        Whether to add the cooks layer. Defaults to False.

    n_params : int, optional
        Number of parameters. Defaults to 5.

    Returns
    -------
    ad.AnnData
        The dummy AnnData object.
    """
    n_obs, n_vars = data_from_opener.X.shape

    adata = ad.AnnData(
        X=data_from_opener.X,
        obs=data_from_opener.obs,
        var=data_from_opener.var,
    )

    # We need to have a "cells" obs field
    adata.obs["cells"] = np.random.choice(["A", "B", "C", "D", "E"], size=n_obs)

    # We need to create the following obsm fields
    # - design_matrix
    # - size_factors
    adata.obsm["design_matrix"] = pd.DataFrame(
        index=adata.obs_names,
        data=np.random.randint(low=0, high=2, size=(n_obs, n_params)),
    )
    adata.obsm["size_factors"] = np.random.rand(n_obs)

    # We need to have the following uns fields
    # - n_params
    adata.uns["n_params"] = n_params
    # - num_replicates
    adata.uns["num_replicates"] = pd.DataFrame(
        index=np.arange(n_params if num_row_values_equal_n_params else n_params + 1),
        data=np.random.rand(
            n_params if num_row_values_equal_n_params else n_params + 1
        ),
    )

    # We need to create the following varm fields
    # - _beta_rough_dispersions
    adata.varm["_beta_rough_dispersions"] = np.random.rand(n_vars, n_params)
    adata.varm["non_zero"] = np.random.rand(n_vars) > 0.2
    adata.varm["_mu_hat_LFC"] = pd.DataFrame(
        index=adata.var_names, data=np.random.rand(n_vars, n_params)
    )
    adata.varm["LFC"] = pd.DataFrame(
        index=adata.var_names, data=np.random.rand(n_vars, n_params)
    )
    adata.varm["cell_means"] = pd.DataFrame(
        index=adata.var_names,
        columns=["A", "B", "C", "D", "E"],
        data=np.random.rand(n_vars, 5),
    )

    set_normed_counts(adata)
    set_mu_layer(
        local_adata=adata,
        lfc_param_name="LFC",
        mu_param_name="_mu_LFC",
    )
    set_mu_layer(
        local_adata=adata,
        lfc_param_name="_mu_hat_LFC",
        mu_param_name="_irls_mu_hat",
    )

    set_sqerror_layer(adata)
    set_y_hat(adata)
    set_fit_lin_mu_hat(adata)
    set_mu_hat_layer(adata)

    if add_cooks:
        adata.layers["cooks"] = np.random.rand(n_obs, n_vars)

    return adata
