from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from pydeseq2.utils import lowess
from scipy.stats import false_discovery_control
from substrafl.remote import remote_data

from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote_data


class IndependentFiltering:
    """Mixin class implementing independent filtering.

    Attributes
    ----------
    local_adata : AnnData
        Local AnnData object.

    alpha : float
        Significance level.

    Methods
    -------
    run_independent_filtering
        Run independent filtering on the p-values trend
    """

    local_adata: AnnData
    alpha: float

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def run_independent_filtering(self, data_from_opener, shared_state: Any):
        """Run independent filtering on the p-values trend.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used.

        shared_state : dict
            Shared state containing the results of the wald tests, namely
            - "p_values" : p-values
            - "wald_statistics" : Wald statistics
            - "wald_se" : Wald standard errors
        """
        p_values = shared_state["p_values"]
        wald_statistics = shared_state["wald_statistics"]
        wald_se = shared_state["wald_se"]

        self.local_adata.varm["p_values"] = p_values
        self.local_adata.varm["wald_statistics"] = wald_statistics
        self.local_adata.varm["wald_se"] = wald_se

        base_mean = self.local_adata.varm["_normed_means"]

        lower_quantile = np.mean(base_mean == 0)

        if lower_quantile < 0.95:
            upper_quantile = 0.95
        else:
            upper_quantile = 1

        theta = np.linspace(lower_quantile, upper_quantile, 50)
        cutoffs = np.quantile(base_mean, theta)

        result = pd.DataFrame(
            np.nan, index=self.local_adata.var_names, columns=np.arange(len(theta))
        )

        for i, cutoff in enumerate(cutoffs):
            use = (base_mean >= cutoff) & (~np.isnan(p_values))
            U2 = p_values[use]
            if not len(U2) == 0:
                result.loc[use, i] = false_discovery_control(U2, method="bh")

        num_rej = (result < self.alpha).sum(0)
        lowess_res = lowess(theta, num_rej, frac=1 / 5)

        if num_rej.max() <= 10:
            j = 0
        else:
            residual = num_rej[num_rej > 0] - lowess_res[num_rej > 0]
            thresh = lowess_res.max() - np.sqrt(np.mean(residual**2))

            if np.any(num_rej > thresh):
                j = np.where(num_rej > thresh)[0][0]
            else:
                j = 0

        self.local_adata.varm["padj"] = result.loc[:, j]


class PValueAdjustment:
    """Mixin class implementing p-value adjustment.

    Attributes
    ----------
    local_adata : AnnData
        Local AnnData object.

    Methods
    -------
    run_p_value_adjustment
        Run p-value adjustment on the p-values trend using the Benjamini-Hochberg
        method.
    """

    local_adata: AnnData

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def run_p_value_adjustment(self, data_from_opener, shared_state: Any):
        """Run p-value adjustment on the p-values trend using the BH method.

        Parameters
        ----------
        data_from_opener : AnnData
            Not used.

        shared_state : dict
            Shared state containing the results of the Wald tests, namely
            - "p_values" : p-values, as a numpy array
            - "wald_statistics" : Wald statistics
            - "wald_se" : Wald standard errors
        """
        p_values = shared_state["p_values"]
        wald_statistics = shared_state["wald_statistics"]
        wald_se = shared_state["wald_se"]

        self.local_adata.varm["p_values"] = p_values
        self.local_adata.varm["wald_statistics"] = wald_statistics
        self.local_adata.varm["wald_se"] = wald_se

        padj = pd.Series(np.nan, index=self.local_adata.var_names)
        padj.loc[~np.isnan(p_values)] = false_discovery_control(
            p_values[~np.isnan(p_values)], method="bh"
        )

        self.local_adata.varm["padj"] = padj
