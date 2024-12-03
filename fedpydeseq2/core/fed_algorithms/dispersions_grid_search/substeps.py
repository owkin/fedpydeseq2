"""Module to implement the substeps to fit dispersions with MLE.

This module contains all the substeps to fit dispersions using a grid search.
"""


import numpy as np
from anndata import AnnData
from joblib import Parallel  # type: ignore
from joblib import delayed
from joblib import parallel_backend
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.utils import global_grid_cr_loss
from fedpydeseq2.core.utils import local_grid_summands
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


class LocGridLoss:
    """Mixin to compute local MLE summands on a grid."""

    local_adata: AnnData
    refit_adata: AnnData
    grid_batch_size: int
    grid_length: int
    min_disp: float
    num_jobs: int
    joblib_backend: str

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def local_grid_loss(
        self,
        data_from_opener,
        shared_state,
        prior_reg: bool = False,
        refit_mode: bool = False,
    ) -> dict:
        """
        Compute local MLE losses and Cox-Reid summands on a grid.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            Not used.

        shared_state : dict, optional
            Shared states with the previous search intervals "lower_log_bounds" and
            "upper_log_bounds", except at initial step where it is None in the case
            of gene-wise dispersions, or contains the output of the trend fitting
            in the case of MAP dispersions.

        prior_reg : bool
            Whether to include prior regularization, for MAP estimation
            (default: False).

        refit_mode : bool
            Whether to run on `refit_adata`s instead of `local_adata`s (default: False).

        Returns
        -------
        dict
            Keys:
            - "nll": local negative log-likelihoods (n_genes x grid_length),
            - "CR_summand": local Cox-Reid adjustment summands
            (n_params x n_params x n_genes x grid_length),
            - "grid": grid of dispersions to evaluate (n_genes x grid_length),
            - "n_samples": number of samples in the local dataset,
            - "max_disp": global upper bound on dispersions.
            - "non_zero": mask of all zero genes.
            - "reg": quadratic regularization term for MAP estimation (only if
              `prior_reg=True`).
        """
        if refit_mode:
            adata = self.refit_adata
        else:
            adata = self.local_adata

        # If we are fitting MAP dispersions and this is the first iteration, we need
        # to save the results of the trend curve fitting.
        # In refit mode, we can use the results from the previous iteration.
        if not refit_mode:
            if prior_reg and ("fitted_dispersions" not in self.local_adata.varm):
                self.local_adata.varm["fitted_dispersions"] = shared_state[
                    "fitted_dispersions"
                ]
                self.local_adata.uns["trend_coeffs"] = shared_state["trend_coeffs"]
                self.local_adata.uns["prior_disp_var"] = shared_state["prior_disp_var"]
                self.local_adata.uns["_squared_logres"] = shared_state[
                    "_squared_logres"
                ]
                self.local_adata.uns["disp_function_type"] = shared_state[
                    "disp_function_type"
                ]
                self.local_adata.uns["mean_disp"] = shared_state["mean_disp"]

        # Compute log space grids
        if (shared_state is not None) and ("lower_log_bounds" in shared_state):
            # Get the bounds from the previous iteration. Each gene has its own bounds.
            min_log_alpha = shared_state["lower_log_bounds"]  # ndarray (n_genes)
            max_log_alpha = shared_state["upper_log_bounds"]  # ndarray (n_genes)
            grid = np.exp(np.linspace(min_log_alpha, max_log_alpha, self.grid_length)).T
            # of size n_genes x grid_length
        else:
            # At first iteration, all genes get the same grid
            min_log_alpha = np.log(self.min_disp)  # float
            max_log_alpha = np.log(adata.uns["max_disp"])  # float
            grid = np.exp(np.linspace(min_log_alpha, max_log_alpha, self.grid_length))
            # of size n_genes x grid_length
            grid = np.repeat(grid[None, :], adata.n_vars, axis=0)

        design = adata.obsm["design_matrix"].values
        n_params = design.shape[1]

        with parallel_backend(self.joblib_backend):
            res = Parallel(
                n_jobs=self.num_jobs,
            )(
                delayed(local_grid_summands)(
                    counts=adata.X[:, i : i + self.grid_batch_size],
                    design=design,
                    mu=adata.layers["_mu_hat"][:, i : i + self.grid_batch_size],
                    alpha_grid=grid[i : i + self.grid_batch_size, :],
                )
                for i in range(0, adata.n_vars, self.grid_batch_size)
            )
            if len(res) == 0:
                nll = np.zeros((0, self.grid_length))
                CR_summand = np.zeros(
                    (0, self.grid_length, n_params, n_params),
                )
            else:
                nll = np.vstack([x[0] for x in res])
                CR_summand = np.vstack([x[1] for x in res])

            result_shared_state = {
                "nll": nll,
                "CR_summand": CR_summand,
                "grid": grid,
                "max_disp": adata.uns["max_disp"],
                "non_zero": adata.varm["non_zero"],
            }

            if prior_reg:
                reg = (
                    np.log(grid) - np.log(adata.varm["fitted_dispersions"])[:, None]
                ) ** 2 / (2 * adata.uns["prior_disp_var"])

                result_shared_state["reg"] = reg

        return result_shared_state


class AggGridUpdate:
    """Mixin to compute global MLE grid updates."""

    min_disp: float
    grid_batch_size: int
    num_jobs: int
    joblib_backend: str

    @remote
    @log_remote
    def global_grid_update(
        self,
        shared_states,
        prior_reg: bool = False,
        dispersions_param_name: str = "genewise_dispersions",
    ) -> dict:
        """Aggregate local MLE summands on a grid and update global dispersion.

        Also sets new search intervals for recursion.

        Parameters
        ----------
        shared_states : list
            List of local states dictionaries, with:
            - "nll": local negative log-likelihoods (n_genes x grid_length),
            - "CR_summand": local Cox-Reid adjustment summands
            (n_params x n_params x n_genes x grid_length),
            - "grid": grid of dispersions that were evaluated (n_genes x grid_length),
            - "max_disp": global upper bound on dispersions.
            - "reg": prior regularization to add for MAP dispersions
              (only if prior_reg is True).

        prior_reg : bool
            Whether to include prior regularization, for MAP estimation
            (default: False).


        dispersions_param_name : str
            Name of the dispersion parameter to update. Dispersions will be saved under
            this name. (default: "genewise_dispersions").

        Returns
        -------
        dict
            Keys:
            - dispersions_param_name: updated dispersions (n_genes),
            - "lower_log_bounds": updated lower log bounds (n_genes),
            - "upper_log_bounds": updated upper log bounds (n_genes).
        """
        nll = sum([state["nll"] for state in shared_states])
        global_CR_summand = sum([state["CR_summand"] for state in shared_states])

        # Compute (batched) global losses
        with parallel_backend(self.joblib_backend):
            res = Parallel(
                n_jobs=self.num_jobs,
            )(
                delayed(global_grid_cr_loss)(
                    nll=nll[i : i + self.grid_batch_size],
                    cr_grid=global_CR_summand[i : i + self.grid_batch_size],
                )
                for i in range(0, len(nll), self.grid_batch_size)
            )

        if len(res) == 0:
            global_losses = np.zeros((0, nll.shape[1]))
        else:
            global_losses = np.concatenate(res, axis=0)

        if prior_reg:
            global_losses += shared_states[0]["reg"]

        # For each gene, find the argmin alpha, and the new search interval
        grids = shared_states[0]["grid"]
        # min_idx of shape n_genes
        min_idx = np.argmin(global_losses, axis=1)
        # delta of shape n_genes
        alpha = grids[np.arange(len(grids)), min_idx]

        # Compute the new bounds
        # Note: the grid should be in log space
        delta_grid = np.log(grids[:, 1]) - np.log(grids[:, 0])
        log_grid_lower_bounds = np.maximum(
            np.log(self.min_disp), np.log(alpha) - delta_grid
        )
        log_grid_upper_bounds = np.minimum(
            np.log(shared_states[0]["max_disp"]), np.log(alpha) + delta_grid
        )

        # Set the dispersions of all-zero genes to NaN
        alpha[~shared_states[0]["non_zero"]] = np.NaN

        return {
            dispersions_param_name: alpha,
            "lower_log_bounds": log_grid_lower_bounds,
            "upper_log_bounds": log_grid_upper_bounds,
        }
