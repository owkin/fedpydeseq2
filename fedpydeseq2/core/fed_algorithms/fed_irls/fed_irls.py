"""Module containing the ComputeLFC method."""

from substrafl.nodes import AggregationNode

from fedpydeseq2.core.fed_algorithms.fed_irls.substeps import AggMakeIRLSUpdate
from fedpydeseq2.core.fed_algorithms.fed_irls.substeps import LocMakeIRLSSummands
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import end_iteration
from fedpydeseq2.core.utils.logging.logging_decorators import end_loop
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method
from fedpydeseq2.core.utils.logging.logging_decorators import start_iteration
from fedpydeseq2.core.utils.logging.logging_decorators import start_loop


class FedIRLS(
    LocMakeIRLSSummands,
    AggMakeIRLSUpdate,
):
    r"""Mixin class to implement the LFC computation algorithm.

    The goal of this class is to implement the IRLS algorithm specifically applied
    to the negative binomial distribution, with fixed dispersion parameter (only
    the mean parameter, expressed as the exponential of the log fold changes times
    the design matrix, is estimated). This algorithm is caught with another method on
    the genes on which it fails.

    To the best of our knowledge, there is no explicit implementation of IRLS for the
    negative binomial in a federated setting. However, the steps of IRLS are akin
    to the ones of a Newton-Raphson algorithm, with the difference that the Hessian
    matrix is replaced by the Fisher information matrix.

    Let us recall the steps of the IRLS algorithm for one gene (this method then
    implements these iterations for all genes in parallell).
    We want to estimate the log fold changes :math:`\beta` from the counts :math:`y`
    and the design matrix :math:`X`. The negative binomial likelihood is given by:

    .. math::
        \mathcal{L}(\beta) = \sum_{i=1}^n \left( y_i \log(\mu_i) -
        (y_i + \alpha^{-1}) \log(\mu_i + \alpha^{-1}) \right) + \text{const}(y, \alpha)

    where :math:`\mu_i = \gamma_i\exp(X_i \cdot \beta)` and :math:`\alpha` is
    the dispersion parameter.

    Given an iterate :math:`\beta_k`, the IRLS algorithm computes the next iterate
    :math:`\beta_{k+1}` as follows.

    First, we compute the mean parameter :math:`\mu_k` from the current iterate, using
    the formula of the log fold changes:

    .. math::
        (\mu_{k})_i = \gamma_i \exp(X_i \cdot \beta_k)

    In practice, we trim the values of :math:`\mu_k` to a minimum value to ensure
    numerical stability.

    Then, we compute the weight matrix :math:`W_k` from the current iterate
    :math:`\beta_k`, which is a diagonal matrix with diagonal elements:

    .. math::
        (W_k)_{ii} = \frac{\mu_{k,i}}{1 + \mu_{k,i} \alpha}

    where :math:`\alpha` is the dispersion parameter.
    This weight matrix is used to compute both the estimated variance (or hat matrix)
    and the feature vector :math:`z_k`:

    .. math::
        z_k = \log\left(\frac{\mu_k}{\gamma}\right) + \frac{y - \mu_k}{\mu_k}

    The estimated variance is given by:

    .. math::
        H_k = X^T W_k X

    The update step is then given by:

    .. math::
        \beta_{k+1} = (H_k)^{-1} X^T W_k z_k

    This is akin to the Newton-Raphson algorithm, with the
    Hessian matrix replaced by the Fisher information, and the gradient replaced by the
    feature vector.

    Methods
    -------
    run_fed_irls
        Run the IRLS algorithm.

    """

    @log_organisation_method
    def run_fed_irls(
        self,
        train_data_nodes: list,
        aggregation_node: AggregationNode,
        local_states: dict,
        input_shared_state: dict,
        round_idx: int,
        clean_models: bool = True,
        refit_mode: bool = False,
    ):
        """Run the IRLS algorithm.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        input_shared_state: dict
            Shared state with the following keys:
            - beta: ndarray
                The current beta, of shape (n_non_zero_genes, n_params).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if fed avg should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - irls_mask: ndarray
                A boolean mask indicating if IRLS should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - global_nll: ndarray
                The global_nll of the current beta from the previous beta, of shape\
                (n_non_zero_genes,).
            - round_number_irls: int
                The current round number of the IRLS algorithm.

        round_idx: int
            The current round.

        clean_models: bool
            If True, the models are cleaned.

        refit_mode: bool
            Whether to run on `refit_adata`s instead of `local_adata`s.
            (default: False).

        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.

        global_irls_summands_nlls_shared_state: dict
            Shared states containing the final IRLS results.
            It contains nothing for now.
            - beta: ndarray
                The current beta, of shape (n_non_zero_genes, n_params).
            - irls_diverged_mask: ndarray
                A boolean mask indicating if fed avg should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - irls_mask: ndarray
                A boolean mask indicating if IRLS should be used for a given gene
                (shape: (n_non_zero_genes,)).
            - global_nll: ndarray
                The global_nll of the current beta from the previous beta, of shape\
                (n_non_zero_genes,).
            - round_number_irls: int
                The current round number of the IRLS algorithm.

        round_idx: int
            The updated round index.

        """
        #### ---- Main training loop ---- #####

        global_irls_summands_nlls_shared_state = input_shared_state

        start_loop()
        for iteration in range(self.irls_num_iter + 1):
            start_iteration(iteration)
            # ---- Compute local IRLS summands and nlls ---- #

            (
                local_states,
                local_irls_summands_nlls_shared_states,
                round_idx,
            ) = local_step(
                local_method=self.make_local_irls_summands_and_nlls,
                train_data_nodes=train_data_nodes,
                output_local_states=local_states,
                round_idx=round_idx,
                input_local_states=local_states,
                input_shared_state=global_irls_summands_nlls_shared_state,
                aggregation_id=aggregation_node.organization_id,
                description="Compute local IRLS summands and nlls.",
                clean_models=clean_models,
                method_params={"refit_mode": refit_mode},
            )

            # ---- Compute global IRLS update and nlls ---- #

            global_irls_summands_nlls_shared_state, round_idx = aggregation_step(
                aggregation_method=self.make_global_irls_update,
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
                input_shared_states=local_irls_summands_nlls_shared_states,
                round_idx=round_idx,
                description="Update the log fold changes and nlls in IRLS.",
                clean_models=clean_models,
            )
            end_iteration()
        end_loop()

        return local_states, global_irls_summands_nlls_shared_state, round_idx
