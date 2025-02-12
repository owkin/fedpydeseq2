import anndata as ad
import numpy as np

from fedpydeseq2.core.utils.layers.build_layers import can_get_fit_lin_mu_hat
from fedpydeseq2.core.utils.layers.build_layers import can_get_mu_hat
from fedpydeseq2.core.utils.layers.build_layers import can_get_normed_counts
from fedpydeseq2.core.utils.layers.build_layers import can_get_sqerror_layer
from fedpydeseq2.core.utils.layers.build_layers import can_get_y_hat
from fedpydeseq2.core.utils.layers.build_layers import can_set_cooks_layer
from fedpydeseq2.core.utils.layers.build_layers import can_set_hat_diagonals_layer
from fedpydeseq2.core.utils.layers.build_layers import can_set_mu_layer
from fedpydeseq2.core.utils.layers.build_layers import set_cooks_layer
from fedpydeseq2.core.utils.layers.build_layers import set_fit_lin_mu_hat
from fedpydeseq2.core.utils.layers.build_layers import set_hat_diagonals_layer
from fedpydeseq2.core.utils.layers.build_layers import set_mu_hat_layer
from fedpydeseq2.core.utils.layers.build_layers import set_mu_layer
from fedpydeseq2.core.utils.layers.build_layers import set_normed_counts
from fedpydeseq2.core.utils.layers.build_layers import set_sqerror_layer
from fedpydeseq2.core.utils.layers.build_layers import set_y_hat

AVAILABLE_LAYERS = [
    "normed_counts",
    "_mu_LFC",
    "_irls_mu_hat",
    "sqerror",
    "_y_hat",
    "_fit_lin_mu_hat",
    "_mu_hat",
    "_hat_diagonals",
    "cooks",
]


def get_available_layers(
    adata: ad.AnnData | None,
    shared_state: dict | None,
    refit: bool = False,
    all_layers_from_disk: bool = False,
) -> list[str]:
    """Get the available layers in the adata.

    Parameters
    ----------
    adata : Optional[ad.AnnData]
        The local adata.

    shared_state : dict
        The shared state containing the Cook's dispersion values.

    refit : bool
        Whether to refit the layers.

    all_layers_from_disk : bool
        Whether to get all layers from disk.

    Returns
    -------
    list[str]
        List of available layers.
    """
    if adata is None:
        return []
    if all_layers_from_disk:
        return list(adata.layers.keys())
    available_layers = []
    if can_get_normed_counts(adata, raise_error=False):
        available_layers.append("normed_counts")
    if can_get_y_hat(adata, raise_error=False):
        available_layers.append("_y_hat")
    if can_get_mu_hat(adata, raise_error=False):
        available_layers.append("_mu_hat")
    if can_get_fit_lin_mu_hat(adata, raise_error=False):
        available_layers.append("_fit_lin_mu_hat")
    if can_get_sqerror_layer(adata, raise_error=False):
        available_layers.append("sqerror")
    if not refit and can_set_cooks_layer(
        adata, shared_state=shared_state, raise_error=False
    ):
        available_layers.append("cooks")
    if not refit and can_set_hat_diagonals_layer(
        adata, shared_state=shared_state, raise_error=False
    ):
        available_layers.append("_hat_diagonals")
    if can_set_mu_layer(
        adata, lfc_param_name="LFC", mu_param_name="_mu_LFC", raise_error=False
    ):
        available_layers.append("_mu_LFC")
    if can_set_mu_layer(
        adata,
        lfc_param_name="_mu_hat_LFC",
        mu_param_name="_irls_mu_hat",
        raise_error=False,
    ):
        available_layers.append("_irls_mu_hat")

    return available_layers


def load_layers(
    adata: ad.AnnData,
    shared_state: dict | None,
    layers_to_load: list[str],
    n_jobs: int = 1,
    joblib_verbosity: int = 0,
    joblib_backend: str = "loky",
    batch_size: int = 100,
):
    """Load the simple layers from the data_from_opener and the adata object.

    This function loads the layers in the layers_to_load attribute in the
    adata object.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object to load the layers into.

    shared_state : dict, optional
        The shared state containing the Cook's dispersion values.

    layers_to_load : list[str]
        The list of layers to load.

    n_jobs : int
        The number of jobs to use for parallel processing.

    joblib_verbosity : int
        The verbosity level of joblib.

    joblib_backend : str
        The joblib backend to use.

    batch_size : int
        The batch size for parallel processing.
    """
    # Assert that all layers are either complex or simple
    assert np.all(
        layer in AVAILABLE_LAYERS for layer in layers_to_load
    ), f"All layers in layers_to_load must be in {AVAILABLE_LAYERS}"

    if "normed_counts" in layers_to_load:
        set_normed_counts(adata=adata)
    if "_mu_LFC" in layers_to_load:
        set_mu_layer(
            local_adata=adata,
            lfc_param_name="LFC",
            mu_param_name="_mu_LFC",
            n_jobs=n_jobs,
            joblib_verbosity=joblib_verbosity,
            joblib_backend=joblib_backend,
            batch_size=batch_size,
        )
    if "_irls_mu_hat" in layers_to_load:
        set_mu_layer(
            local_adata=adata,
            lfc_param_name="_mu_hat_LFC",
            mu_param_name="_irls_mu_hat",
            n_jobs=n_jobs,
            joblib_verbosity=joblib_verbosity,
            joblib_backend=joblib_backend,
            batch_size=batch_size,
        )
    if "sqerror" in layers_to_load:
        set_sqerror_layer(adata)
    if "_y_hat" in layers_to_load:
        set_y_hat(adata)
    if "_fit_lin_mu_hat" in layers_to_load:
        set_fit_lin_mu_hat(adata)
    if "_mu_hat" in layers_to_load:
        set_mu_hat_layer(adata)
    if "_hat_diagonals" in layers_to_load:
        set_hat_diagonals_layer(adata=adata, shared_state=shared_state)
    if "cooks" in layers_to_load:
        set_cooks_layer(adata=adata, shared_state=shared_state)


def remove_layers(
    adata: ad.AnnData,
    layers_to_save_on_disk: list[str],
    refit: bool = False,
):
    """Remove the simple layers from the adata object.

    This function removes the simple layers from the adata object. The layers_to_save
    parameter can be used to specify which layers to save in the local state.
    If layers_to_save is None, no layers are saved.

    This function also adds all present layers to the _available_layers field in the
    adata object. This field is used to keep track of the layers that are present in
    the adata object.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object to remove the layers from.

    refit : bool
        Whether the adata object is the refit_adata object.

    layers_to_save_on_disk : list[str]
        The list of layers to save. If None, no layers are saved.
    """
    adata.X = None
    if refit:
        adata.obsm = None

    layer_names = list(adata.layers.keys()).copy()
    for layer_name in layer_names:
        if layer_name in layers_to_save_on_disk:
            continue
        del adata.layers[layer_name]
