"""Module containing the methods that are used multiple times in the pipeline.

These methods have been adapted from a pooled setting to a federated setting.
They are:
- the computation of the trimmed mean
- the federated IRLS computation with a negative binomial distribution
- the federated Proximal Quasi Newton computation with a negative binomial distribution
- the federated grid search computation with a negative binomial distribution for the
    alpha parameter.
"""

from fedpydeseq2.core.fed_algorithms.compute_trimmed_mean import ComputeTrimmedMean
from fedpydeseq2.core.fed_algorithms.dispersions_grid_search import (
    ComputeDispersionsGridSearch,
)
from fedpydeseq2.core.fed_algorithms.fed_irls import FedIRLS
from fedpydeseq2.core.fed_algorithms.fed_PQN import FedProxQuasiNewton
