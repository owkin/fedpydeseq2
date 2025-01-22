"""Main module to compute genewise dispersions."""
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.compute_MoM_dispersions import (  # noqa: E501
    ComputeMoMDispersions,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.get_num_replicates import (  # noqa: E501
    GetNumReplicates,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.substeps import (  # noqa: E501
    LocLinMu,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions.substeps import (  # noqa: E501
    LocSetMuHat,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_lfc import ComputeLFC
from fedpydeseq2.core.fed_algorithms import ComputeDispersionsGridSearch
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class ComputeGenewiseDispersions(
    ComputeDispersionsGridSearch,
    ComputeMoMDispersions,
    LocLinMu,
    GetNumReplicates,
    ComputeLFC,
    LocSetMuHat,
):
    """Mixin class to implement the computation of both genewise and MAP dispersions.

    The switch between genewise and MAP dispersions is done by setting the `fit_mode`
    argument in the `fit_dispersions` to either "MLE" or "MAP".

    Methods
    -------
    fit_gene_wise_dispersions
        A method to fit gene-wise dispersions using a grid search.
        Performs four steps:
        1. Compute the first dispersions estimates using a
        method of moments (MoM) approach.
        2. Compute the number of replicates for each combination of factors.
        This step is necessary to compute the mean estimate in one case, and
        in downstream steps (cooks distance, etc).
        3. Compute an estimate of the mean from these dispersions.
        4. Fit the dispersions using a grid search.
    """

    @log_organisation_method
    def fit_genewise_dispersions(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        gram_features_shared_states,
        round_idx,
        clean_models,
        refit_mode: bool = False,
    ):
        """Fit the gene-wise dispersions.

        Performs four steps:
        1. Compute the first dispersions estimates using a
        method of moments (MoM) approach.
        2. Compute the number of replicates for each combination of factors.
        This step is necessary to compute the mean estimate in one case, and
        in downstream steps (cooks distance, etc).
        3. Compute an estimate of the mean from these dispersions.
        4. Fit the dispersions using a grid search.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        gram_features_shared_states: list
            The list of shared states outputed by the compute_size_factors step.
            They contain a "local_gram_matrix" and a "local_features" fields.

        round_idx: int
            The current round.

        clean_models: bool
            Whether to clean the models after the computation.

        refit_mode: bool
            Whether to run the pipeline in refit mode, after cooks outliers were
            replaced. If True, the pipeline will be run on `refit_adata`s instead of
            `local_adata`s. (default: False).

        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.

        shared_state: dict or list[dict]
            A dictionary containing:
            - "genewise_dispersions": The MLE dispersions, to be stored locally at
            - "lower_log_bounds": log lower bounds for the grid search (only used in
            internal loop),
            - "upper_log_bounds": log upper bounds for the grid search (only used in
            internal loop).

        round_idx: int
            The updated round index.
        """
        # ---- Compute MoM dispersions ---- #
        (
            local_states,
            mom_dispersions_shared_state,
            round_idx,
        ) = self.compute_MoM_dispersions(
            train_data_nodes,
            aggregation_node,
            local_states,
            gram_features_shared_states,
            round_idx,
            clean_models,
            refit_mode=refit_mode,
        )

        # ---- Compute the initial mu estimates ---- #

        # 1 - Compute the linear mu estimates.

        local_states, linear_shared_states, round_idx = local_step(
            local_method=self.fit_lin_mu,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=mom_dispersions_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Compute local linear mu estimates.",
            round_idx=round_idx,
            clean_models=clean_models,
            method_params={"refit_mode": refit_mode},
        )

        # 2 - Compute IRLS estimates.
        local_states, round_idx = self.compute_lfc(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
            lfc_mode="mu_init",
            refit_mode=refit_mode,
        )

        # 3 - Compare the number of replicates to the number of design matrix columns
        # and decide whether to use the IRLS estimates or the linear estimates.

        # Compute the number of replicates
        local_states, round_idx = self.get_num_replicates(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

        local_states, shared_states, round_idx = local_step(
            local_method=self.set_mu_hat,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Pick between linear and irls mu_hat.",
            round_idx=round_idx,
            clean_models=clean_models,
            method_params={"refit_mode": refit_mode},
        )

        # ---- Fit dispersions ---- #
        local_states, shared_state, round_idx = self.fit_dispersions(
            train_data_nodes,
            aggregation_node,
            local_states,
            shared_state=None,
            round_idx=round_idx,
            clean_models=clean_models,
            fit_mode="MLE",
            refit_mode=refit_mode,
        )

        return local_states, shared_state, round_idx
