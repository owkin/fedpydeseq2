"""Module containing the substeps for the computation of size factors."""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
from pydeseq2.default_inference import DefaultInference
from pydeseq2.utils import mean_absolute_deviation
from scipy.special import polygamma  # type: ignore
from scipy.stats import trim_mean  # type: ignore
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_dispersion_prior.utils import (  # noqa: E501
    disp_function,
)
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data


# TODO : This step could be removed now that genewise dispersions are computed in the
# pipeline. This would save an aggregation -> local node -> aggregation node
# communication.
class LocGetMeanDispersionAndMean:
    """Mixin to get the local mean and dispersion."""

    local_adata: ad.AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def get_local_mean_and_dispersion(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> dict:
        # pylint: disable=unused-argument
        """Return local gene means and dispersion.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Shared state returned by the last step of gene-wise dispersion computation.
            Contains a "genewise_dispersions" key with the gene-wise dispersions.

        Returns
        -------
        dict
            Local results to be shared via shared_state to the aggregation node. dict
            with the following keys:
            - mean_normed_counts: np.ndarray[float] of shape (n_genes,)
                The mean normed counts.
            - n_obs: int
                The number of observations.
            - non_zero: np.ndarray[bool] of shape (n_genes,)
                Mask of the genes with non zero counts.
            - genewise_dispersions: np.ndarray[float] of shape (n_genes,)
                The genewise dispersions.
            - num_vars: int
                The number of variables.
        """
        # Save gene-wise dispersions from the previous step.
        # Dispersions of all-zero genes should already be NaN.
        self.local_adata.varm["genewise_dispersions"] = shared_state[
            "genewise_dispersions"
        ]

        # TODO: these could be gathered earlier and sent directly to the aggregation
        # node.
        return {
            "mean_normed_counts": self.local_adata.layers["normed_counts"].mean(0),
            "n_obs": self.local_adata.n_obs,
            "non_zero": self.local_adata.varm["non_zero"],
            "genewise_dispersions": self.local_adata.varm["genewise_dispersions"],
            "n_params": self.local_adata.uns["n_params"],
        }


class AggFitDispersionTrendAndPrior:
    """Mixin class to implement the fit of the dispersion trend."""

    min_disp: float

    @remote
    @log_remote
    def agg_fit_dispersion_trend_and_prior_dispersion(self, shared_states):
        """Fit the dispersion trend, and compute the dispersion prior.

        Parameters
        ----------
        shared_states : dict
            Shared states from the local step with the following keys:
            - genewise_dispersions: np.ndarray of shape (n_genes,)
            - n_params: int
            - non_zero: np.ndarray of shape (n_genes,)
            - mean_normed_counts: np.ndarray of shape (n_genes,)
            - n_obs: int

        Returns
        -------
        dict
            dict with the following keys:
            - prior_disp_var: float
                The prior dispersion variance.
            - _squared_logres: float
                The squared log-residuals.
            - trend_coeffs: np.ndarray of shape (2,)
                The coefficients of the parametric dispersion trend.
            - fitted_dispersions: np.ndarray of shape (n_genes,)
                The fitted dispersions, computed from the dispersion trend.
            - disp_function_type: str
                The type of dispersion function (parametric or mean).
            - mean_disp: float, optional
                The mean dispersion (if "mean" fit type).
        """
        genewise_dispersions = shared_states[0]["genewise_dispersions"]
        n_params = shared_states[0]["n_params"]
        non_zero = shared_states[0]["non_zero"]
        n_total_obs = sum([state["n_obs"] for state in shared_states])
        mean_normed_counts = (
            sum(
                [
                    state["mean_normed_counts"] * state["n_obs"]
                    for state in shared_states
                ]
            )
            / n_total_obs
        )

        # Exclude all-zero counts
        targets = pd.Series(
            genewise_dispersions.copy(),
        )
        targets = targets[non_zero]
        covariates = pd.Series(1 / mean_normed_counts[non_zero], index=targets.index)

        for gene in targets.index:
            if (
                np.isinf(covariates.loc[gene]).any()
                or np.isnan(covariates.loc[gene]).any()
            ):
                targets.drop(labels=[gene], inplace=True)
                covariates.drop(labels=[gene], inplace=True)

        # Initialize coefficients
        old_coeffs = pd.Series([0.1, 0.1])
        coeffs = pd.Series([1.0, 1.0])
        mean_disp = None

        disp_function_type = "parametric"
        while (coeffs > 1e-10).all() and (
            np.log(np.abs(coeffs / old_coeffs)) ** 2
        ).sum() >= 1e-6:
            old_coeffs = coeffs
            (
                coeffs,
                predictions,
                converged,
            ) = DefaultInference().dispersion_trend_gamma_glm(covariates, targets)

            if not converged or (coeffs <= 1e-10).any():
                warnings.warn(
                    "The dispersion trend curve fitting did not converge. "
                    "Switching to a mean-based dispersion trend.",
                    UserWarning,
                    stacklevel=2,
                )
                mean_disp = trim_mean(
                    genewise_dispersions[genewise_dispersions > 10 * self.min_disp],
                    proportiontocut=0.001,
                )
                disp_function_type = "mean"

            pred_ratios = genewise_dispersions[covariates.index] / predictions

            targets.drop(
                targets[(pred_ratios < 1e-4) | (pred_ratios >= 15)].index,
                inplace=True,
            )
            covariates.drop(
                covariates[(pred_ratios < 1e-4) | (pred_ratios >= 15)].index,
                inplace=True,
            )

        fitted_dispersions = np.full_like(genewise_dispersions, np.NaN)

        fitted_dispersions[non_zero] = disp_function(
            mean_normed_counts[non_zero],
            disp_function_type=disp_function_type,
            coeffs=coeffs,
            mean_disp=mean_disp,
        )

        disp_residuals = np.log(genewise_dispersions[non_zero]) - np.log(
            fitted_dispersions[non_zero]
        )

        # Compute squared log-residuals and prior variance based on genes whose
        # dispersions are above 100 * min_disp. This is to reproduce DESeq2's behaviour.
        above_min_disp = genewise_dispersions[non_zero] >= (100 * self.min_disp)

        _squared_logres = mean_absolute_deviation(disp_residuals[above_min_disp]) ** 2

        prior_disp_var = np.maximum(
            _squared_logres - polygamma(1, (n_total_obs - n_params) / 2),
            0.25,
        )

        return {
            "prior_disp_var": prior_disp_var,
            "_squared_logres": _squared_logres,
            "trend_coeffs": coeffs,
            "fitted_dispersions": fitted_dispersions,
            "disp_function_type": disp_function_type,
            "mean_disp": mean_disp,
        }


class LocUpdateFittedDispersions:
    """Mixin to update the fitted dispersions after replacing outliers.

    To use in refit mode only
    """

    local_adata: ad.AnnData
    refit_adata: ad.AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def loc_update_fitted_dispersions(
        self,
        data_from_opener,
        shared_state: dict,
    ) -> None:
        """Update the fitted dispersions after replacing outliers.

        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            A dictionary with a "fitted_dispersions" key, containing the dispersions
            fitted before replacing the outliers.
        """
        # Start by updating gene-wise dispersions
        self.refit_adata.varm["genewise_dispersions"] = shared_state[
            "genewise_dispersions"
        ]

        # Update the fitted dispersions
        non_zero = self.refit_adata.varm["non_zero"]
        self.refit_adata.uns["disp_function_type"] = self.local_adata.uns[
            "disp_function_type"
        ]

        fitted_dispersions = np.full_like(
            self.refit_adata.varm["genewise_dispersions"], np.NaN
        )

        fitted_dispersions[non_zero] = disp_function(
            self.refit_adata.varm["_normed_means"][non_zero],
            disp_function_type=self.refit_adata.uns["disp_function_type"],
            coeffs=self.refit_adata.uns["trend_coeffs"],
            mean_disp=(
                self.refit_adata.uns["mean_disp"]
                if self.refit_adata.uns["disp_function_type"] == "parametric"
                else None
            ),
        )

        self.refit_adata.varm["fitted_dispersions"] = fitted_dispersions
