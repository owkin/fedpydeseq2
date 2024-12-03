from collections.abc import Callable
from functools import wraps
from typing import Any
from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

from fedpydeseq2.core.utils.compute_lfc_utils import get_lfc_utils_from_gene_mask_adata
from fedpydeseq2.core.utils.layers.joblib_utils import get_joblib_parameters


def prepare_cooks_local(method: Callable):
    """Decorate the local method just preceding a local method needing cooks.

    This method is only applied if the Cooks layer is not present or must not be
    saved between steps.

    This step is used to compute the local hat matrix and the mean normed counts.

    Before the method is called, the varEst must be accessed from the shared state,
    or from the local adata if it is not present in the shared state.

    The local hat matrix and the mean normed counts are computed, and the following
    keys are added to the shared state:
    - local_hat_matrix
    - mean_normed_counts
    - n_samples
    - varEst

    Parameters
    ----------
    method : Callable
        The remote_data method to decorate.

    Returns
    -------
    Callable:
        The decorated method.

    """

    @wraps(method)
    def method_inner(
        self,
        data_from_opener: ad.AnnData,
        shared_state: Any = None,
        **method_parameters,
    ):
        # ---- Step 0: If can skip, we skip ---- #
        if can_skip_local_cooks_preparation(self):
            shared_state = method(
                self, data_from_opener, shared_state, **method_parameters
            )
            shared_state["_skip_cooks"] = True
            return shared_state

        # ---- Step 1: Access varEst ---- #

        if "varEst" in self.local_adata.varm.keys():
            varEst = self.local_adata.varm["varEst"]
        else:
            assert "varEst" in shared_state
            varEst = shared_state["varEst"]
            self.local_adata.varm["varEst"] = varEst

        # ---- Step 2: Run the method ---- #
        shared_state = method(self, data_from_opener, shared_state, **method_parameters)

        # ---- Step 3: Compute the local hat matrix ---- #

        n_jobs, joblib_verbosity, joblib_backend, batch_size = get_joblib_parameters(
            self
        )
        # Compute hat matrix
        (
            gene_names,
            design_matrix,
            size_factors,
            counts,
            dispersions,
            beta,
        ) = get_lfc_utils_from_gene_mask_adata(
            self.local_adata,
            None,
            "dispersions",
            lfc_param_name="LFC",
        )

        with parallel_backend(joblib_backend):
            res = Parallel(n_jobs=n_jobs, verbose=joblib_verbosity)(
                delayed(make_hat_matrix_summands_batch)(
                    design_matrix,
                    size_factors,
                    beta[i : i + batch_size],
                    dispersions[i : i + batch_size],
                    self.min_mu,
                )
                for i in range(0, len(beta), batch_size)
            )

        if len(res) == 0:
            H = np.zeros((0, beta.shape[1], beta.shape[1]))
        else:
            H = np.concatenate(res)

        shared_state["local_hat_matrix"] = H

        # ---- Step 4: Compute the mean normed counts ---- #

        mean_normed_counts = self.local_adata.layers["normed_counts"].mean(axis=0)

        shared_state["mean_normed_counts"] = mean_normed_counts
        shared_state["n_samples"] = self.local_adata.n_obs
        shared_state["varEst"] = varEst
        shared_state["_skip_cooks"] = False

        return shared_state

    return method_inner


def prepare_cooks_agg(method: Callable):
    """Decorate the aggregation step to compute the Cook's distance.

    This decorator is supposed to be placed on the aggregation step just before
    a local step which needs the "cooks" layer. The decorator will check if the
    shared state contains the necessary keys for the Cook's distance computation.
    If this is not the case, then the Cook's distance must have been saved in the
    layers_to_save.
    It will compute the Cook's dispersion, the hat matrix inverse, and then call
    the method.

    It will add the following keys to the shared state:
    - cooks_dispersions
    - global_hat_matrix_inv

    Parameters
    ----------
    method : Callable
        The aggregation method to decorate.
        It must have the following signature:
        method(self, shared_states: Optional[list], **method_parameters).

    Returns
    -------
    Callable:
        The decorated method.

    """

    @wraps(method)
    def method_inner(
        self,
        shared_states: list | None,
        **method_parameters,
    ):
        # Check that the shared state contains the necessary keys
        # for the Cook's distance computation
        # If this is not the case, then the cooks distance must have
        # been saved in the layers_to_save

        try:
            assert isinstance(shared_states, list)
            assert "n_samples" in shared_states[0].keys()
            assert "varEst" in shared_states[0].keys()
            assert "mean_normed_counts" in shared_states[0].keys()
            assert "local_hat_matrix" in shared_states[0].keys()
        except AssertionError as assertion_error:
            only_from_disk = (
                not hasattr(self, "save_layers_to_disk") or self.save_layers_to_disk
            )
            if only_from_disk:
                return method(self, shared_states, **method_parameters)
            elif isinstance(shared_states, list) and shared_states[0]["_skip_cooks"]:
                return method(self, shared_states, **method_parameters)
            raise ValueError(
                "The shared state does not contain the necessary keys for"
                "the Cook's distance computation."
            ) from assertion_error

        assert isinstance(shared_states, list)

        # ---- Step 1: Compute Cooks dispersion ---- #

        n_sample_tot = sum(
            [shared_state["n_samples"] for shared_state in shared_states]
        )
        varEst = shared_states[0]["varEst"]
        mean_normed_counts = (
            np.array(
                [
                    (shared_state["mean_normed_counts"] * shared_state["n_samples"])
                    for shared_state in shared_states
                ]
            ).sum(axis=0)
            / n_sample_tot
        )
        mask_zero = mean_normed_counts == 0
        mask_varEst_zero = varEst == 0
        alpha = varEst - mean_normed_counts
        alpha[~mask_zero] = alpha[~mask_zero] / mean_normed_counts[~mask_zero] ** 2
        alpha[mask_varEst_zero & mask_zero] = np.nan
        alpha[mask_varEst_zero & (~mask_zero)] = (
            np.inf * alpha[mask_varEst_zero & (~mask_zero)]
        )

        # cannot use the typical min_disp = 1e-8 here or else all counts in the same
        # group as the outlier count will get an extreme Cook's distance
        minDisp = 0.04
        alpha = cast(pd.Series, np.maximum(alpha, minDisp))

        # --- Step 2: Compute the hat matrix inverse --- #

        global_hat_matrix = sum([state["local_hat_matrix"] for state in shared_states])
        n_jobs, joblib_verbosity, joblib_backend, batch_size = get_joblib_parameters(
            self
        )
        ridge_factor = np.diag(np.repeat(1e-6, global_hat_matrix.shape[1]))
        with parallel_backend(joblib_backend):
            res = Parallel(n_jobs=n_jobs, verbose=joblib_verbosity)(
                delayed(np.linalg.inv)(hat_matrices + ridge_factor)
                for hat_matrices in np.split(
                    global_hat_matrix,
                    range(
                        batch_size,
                        len(global_hat_matrix),
                        batch_size,
                    ),
                )
            )

        global_hat_matrix_inv = np.concatenate(res)

        # ---- Step 3: Run the method ---- #

        shared_state = method(self, shared_states, **method_parameters)

        # ---- Step 4: Save the Cook's dispersion and the hat matrix inverse ---- #

        shared_state["cooks_dispersions"] = alpha
        shared_state["global_hat_matrix_inv"] = global_hat_matrix_inv

        return shared_state

    return method_inner


def can_skip_local_cooks_preparation(self: Any) -> bool:
    """Check if the Cook's distance is in the layers to save.

    This function checks if the Cook's distance is in the layers to save.

    Parameters
    ----------
    self : Any
        The object.

    Returns
    -------
    bool:
        Whether the Cook's distance is in the layers to save.

    """
    only_from_disk = (
        not hasattr(self, "save_layers_to_disk") or self.save_layers_to_disk
    )
    if only_from_disk and "cooks" in self.local_adata.layers.keys():
        return True
    if hasattr(self, "layers_to_save_on_disk"):
        layers_to_save_on_disk = self.layers_to_save_on_disk
        if (
            layers_to_save_on_disk is not None
            and "local_adata" in layers_to_save_on_disk
            and layers_to_save_on_disk["local_adata"] is not None
            and "cooks" in layers_to_save_on_disk["local_adata"]
        ):
            return True
    return False


def make_hat_matrix_summands_batch(
    design_matrix: np.ndarray,
    size_factors: np.ndarray,
    beta: np.ndarray,
    dispersions: np.ndarray,
    min_mu: float,
) -> np.ndarray:
    """Make the local hat matrix.

    This is quite similar to the make_irls_summands_batch function, but it does not
    require the counts, and returns only the H matrix.

    This is used in the final step of the IRLS algorithm to compute the local hat
    matrix.

    Parameters
    ----------
    design_matrix : np.ndarray
        The design matrix, of shape (n_obs, n_params).
    size_factors : np.ndarray
        The size factors, of shape (n_obs).
    beta : np.ndarray
        The log fold change matrix, of shape (batch_size, n_params).
    dispersions : np.ndarray
        The dispersions, of shape (batch_size).
    min_mu : float
        Lower bound on estimated means, to ensure numerical stability.


    Returns
    -------
    H : np.ndarray
        The H matrix, of shape (batch_size, n_params, n_params).
    """
    mu = size_factors[:, None] * np.exp(design_matrix @ beta.T)

    mu = np.maximum(mu, min_mu)

    W = mu / (1.0 + mu * dispersions[None, :])

    H = (design_matrix.T[:, :, None] * W).transpose(2, 0, 1) @ design_matrix[None, :, :]

    return H
