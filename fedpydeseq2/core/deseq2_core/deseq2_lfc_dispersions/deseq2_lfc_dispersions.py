from loguru import logger

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_dispersion_prior import (  # noqa: E501
    ComputeDispersionPrior,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_genewise_dispersions import (  # noqa: E501
    ComputeGenewiseDispersions,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_lfc import ComputeLFC
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_MAP_dispersions import (  # noqa: E501
    ComputeMAPDispersions,
)
from fedpydeseq2.core.utils import local_step


class DESeq2LFCDispersions(
    ComputeGenewiseDispersions,
    ComputeDispersionPrior,
    ComputeMAPDispersions,
    ComputeLFC,
):
    """Mixin class to compute the log fold change and the dispersions with DESeq2.

    This class encapsulates the steps to compute the log fold change and the
    dispersions from a given count matrix and a design matrix.

    Methods
    -------
    run_deseq2_lfc_dispersions
        The method to compute the log fold change and the dispersions.
        It starts from the design matrix and the count matrix.
        It returns the shared states by the local nodes after the computation of Cook's
        distances.
        It is meant to be run two times in the main pipeline if Cook's refitting
        is applied/
    """

    def run_deseq2_lfc_dispersions(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        gram_features_shared_states,
        round_idx,
        clean_models,
        refit_mode=False,
    ):
        """
        Run the DESeq2 pipeline to compute the log fold change and the dispersions.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: list[dict]
            Local states. Required to propagate intermediate results.

        gram_features_shared_states: list[dict]
            Output of the "compute_size_factor step" if refit_mode is False.
            Output of the "replace_outliers" step if refit_mode is True.
            In both cases, contains a "local_features" key with the features vector
            to input to compute_genewise_dispersion.
            In the non refit mode case, it also contains a "local_gram_matrix" key
             with the local gram matrix.

        round_idx: int
            Index of the current round.

        clean_models: bool
            Whether to clean the models after the computation.

        refit_mode: bool
            Whether we are refittinh Cooks outliers or not.


        Returns
        -------
        local_states: dict
            Local states updated with the results of the DESeq2 pipeline.

        round_idx: int
            The updated round index.

        """
        #### Fit genewise dispersions ####

        # Note : for optimization purposes, we could avoid two successive local
        # steps here, at the cost of a more complex initialization of the
        # fit_dispersions method.
        logger.info("Fit genewise dispersions...")
        (
            local_states,
            genewise_dispersions_shared_state,
            round_idx,
        ) = self.fit_genewise_dispersions(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            gram_features_shared_states=gram_features_shared_states,
            local_states=local_states,
            round_idx=round_idx,
            clean_models=clean_models,
            refit_mode=refit_mode,
        )
        logger.info("Finished fitting genewise dispersions.")

        if not refit_mode:
            #### Fit dispersion trends ####
            logger.info("Compute dispersion prior...")
            (
                local_states,
                dispersion_trend_share_state,
                round_idx,
            ) = self.compute_dispersion_prior(
                train_data_nodes,
                aggregation_node,
                local_states,
                genewise_dispersions_shared_state,
                round_idx,
                clean_models,
            )
            logger.info("Finished computing dispersion prior.")
        else:
            # Just update the fitted dispersions
            (
                local_states,
                dispersion_trend_share_state,
                round_idx,
            ) = local_step(
                local_method=self.loc_update_fitted_dispersions,
                train_data_nodes=train_data_nodes,
                output_local_states=local_states,
                round_idx=round_idx,
                input_local_states=local_states,
                input_shared_state=genewise_dispersions_shared_state,
                aggregation_id=aggregation_node.organization_id,
                description="Update fitted dispersions",
                clean_models=clean_models,
            )

        #### Fit MAP dispersions ####
        logger.info("Fit MAP dispersions...")
        (
            local_states,
            round_idx,
        ) = self.fit_MAP_dispersions(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            shared_state=dispersion_trend_share_state if not refit_mode else None,
            round_idx=round_idx,
            clean_models=clean_models,
            refit_mode=refit_mode,
        )
        logger.info("Finished fitting MAP dispersions.")

        #### Compute log fold changes ####
        logger.info("Compute log fold changes...")
        local_states, round_idx = self.compute_lfc(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=True,
            lfc_mode="lfc",
            refit_mode=refit_mode,
        )
        logger.info("Finished computing log fold changes.")

        return local_states, round_idx
