"""Module containing the ComputeLFC method."""

from typing import Literal

from substrafl.nodes import AggregationNode

from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_lfc.substeps import (
    AggCreateBetaInit,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_lfc.substeps import (
    LocGetGramMatrixAndLogFeatures,
)
from fedpydeseq2.core.deseq2_core.deseq2_lfc_dispersions.compute_lfc.substeps import (
    LocSaveLFC,
)
from fedpydeseq2.core.fed_algorithms import FedIRLS
from fedpydeseq2.core.fed_algorithms import FedProxQuasiNewton
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.logging.logging_decorators import log_organisation_method


class ComputeLFC(
    LocGetGramMatrixAndLogFeatures,
    AggCreateBetaInit,
    LocSaveLFC,
    FedProxQuasiNewton,
    FedIRLS,
):
    r"""Mixin class to implement the LFC computation algorithm.

    The goal of this class is to implement the IRLS algorithm specifically applied
    to the negative binomial distribution, with fixed dispersion parameter, and
    in the case where it fails, to catch it with the FedProxQuasiNewton algorithm.

    This class also initializes the beta parameters and computes the final hat matrix.

    Methods
    -------
    compute_lfc
        The main method to compute the log fold changes by
        running the IRLS algorithm and catching it with the
        FedProxQuasiNewton algorithm.
    """

    @log_organisation_method
    def compute_lfc(
        self,
        train_data_nodes: list,
        aggregation_node: AggregationNode,
        local_states: dict,
        round_idx: int,
        clean_models: bool = True,
        lfc_mode: Literal["lfc", "mu_init"] = "lfc",
        refit_mode: bool = False,
    ):
        """Compute the log fold changes.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        round_idx: int
            The current round.

        clean_models: bool
            If True, the models are cleaned.

        lfc_mode: Literal["lfc", "mu_init"]
            The mode of the IRLS algorithm ("lfc" or "mu_init").

        refit_mode: bool
            Whether to run the pipeline in refit mode, after cooks outliers were
            replaced. If True, the pipeline will be run on `refit_adata`s instead of
            `local_adata`s. (default: False).


        Returns
        -------
        local_states: dict
            Local states. Required to propagate intermediate results.

        round_idx: int
            The updated round index.
        """
        #### ---- Initialization ---- ####

        # ---- Compute initial local beta estimates ---- #

        local_states, local_beta_init_shared_states, round_idx = local_step(
            local_method=self.get_gram_matrix_and_log_features,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Create local initialization beta.",
            clean_models=clean_models,
            method_params={
                "lfc_mode": lfc_mode,
                "refit_mode": refit_mode,
            },
        )

        # ---- Compute initial global beta estimates ---- #

        global_irls_summands_nlls_shared_state, round_idx = aggregation_step(
            aggregation_method=self.create_beta_init,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=local_beta_init_shared_states,
            description="Create initialization beta paramater.",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### ---- Run IRLS ---- #####
        (
            local_states,
            irls_result_shared_state,
            round_idx,
        ) = self.run_fed_irls(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            input_shared_state=global_irls_summands_nlls_shared_state,
            round_idx=round_idx,
            clean_models=clean_models,
            refit_mode=refit_mode,
        )

        #### ---- Catch with FedProxQuasiNewton ----####

        local_states, PQN_shared_state, round_idx = self.run_fed_PQN(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            PQN_shared_state=irls_result_shared_state,
            first_iteration_mode="irls_catch",
            round_idx=round_idx,
            clean_models=clean_models,
            refit_mode=refit_mode,
        )

        # ---- Compute final hat matrix summands ---- #

        (
            local_states,
            _,
            round_idx,
        ) = local_step(
            local_method=self.save_lfc_to_local,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            round_idx=round_idx,
            input_local_states=local_states,
            input_shared_state=PQN_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Compute local hat matrix summands and last nll.",
            clean_models=clean_models,
            method_params={"refit_mode": refit_mode},
        )

        return local_states, round_idx
