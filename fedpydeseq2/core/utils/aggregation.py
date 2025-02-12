"""Aggregation functions.

Copy-pasted from the CancerLINQ repo.
"""

from typing import Any

import numpy as np


# pylint: disable=deprecated-typing-alias
def aggregate_means(
    local_means: list[Any], n_local_samples: list[int], filter_nan: bool = False
):
    """Aggregate local means.

    Aggregate the local means into a global mean by using the local number of samples.

    Parameters
    ----------
    local_means : list[Any]
        list of local means. Could be array, float, Series.
    n_local_samples : list[int]
        list of number of samples used for each local mean.
    filter_nan : bool, optional
        Filter NaN values in the local means, by default False.

    Returns
    -------
    Any
        Aggregated mean. Same type of the local means
    """
    tot_samples = 0
    tot_mean = np.zeros_like(local_means[0])
    for mean, n_sample in zip(local_means, n_local_samples, strict=False):
        if filter_nan:
            mean = np.nan_to_num(mean, nan=0, copy=False)
        tot_mean += mean * n_sample
        tot_samples += n_sample

    return tot_mean / tot_samples
