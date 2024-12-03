from typing import Literal

from fedpydeseq2.core.fed_algorithms.fed_PQN.substeps import (
    AggChooseStepComputeAscentDirection,
)
from fedpydeseq2.core.fed_algorithms.fed_PQN.substeps import (
    LocMakeFedPQNFisherGradientNLL,
)
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step


class FedProxQuasiNewton(
    LocMakeFedPQNFisherGradientNLL, AggChooseStepComputeAscentDirection
):
    """Mixin class to implement a Prox Newton method for box constraints.

    It implements the method presented here:
    https://www.cs.utexas.edu/~inderjit/public_papers/pqnj_sisc10.pdf
    More context can be found here
    https://optml.mit.edu/papers/sksChap.pdf

    Methods
    -------
    run_fed_PQN
        The method to run the Prox Quasi Newton algorithm.
        It relies on the methods inherited from the LocMakeFedPQNFisherGradientNLL and
        AggChooseStepComputeAscentDirection classes.

    """

    PQN_num_iters: int

    def run_fed_PQN(
        self,
        train_data_nodes,
        aggregation_node,
        local_states,
        PQN_shared_state,
        first_iteration_mode: Literal["irls_catch"] | None,
        round_idx,
        clean_models,
        refit_mode: bool = False,
    ):
        """Run the Prox Quasi Newton  algorithm.

        Parameters
        ----------
        train_data_nodes: list
            List of TrainDataNode.

        aggregation_node: AggregationNode
            The aggregation node.

        local_states: dict
            Local states. Required to propagate intermediate results.

        PQN_shared_state: dict
            The input shared state.
            The requirements for this shared state are defined in the
            LocMakeFedPQNFisherGradientNLL class and depend on the
            first_iteration_mode.

        first_iteration_mode: Optional[Literal["irls_catch"]]
            The first iteration mode.
            This defines the input requirements for the algorithm, and is passed
            to the make_local_fisher_gradient_nll method at the first iteration.

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

        irls_final_shared_states: dict
            Shared states containing the final IRLS results.
            It contains nothing for now.

        round_idx: int
            The updated round index.

        """
        #### ---- Main training loop ---- #####

        for pqn_iter in range(self.PQN_num_iters + 1):
            # ---- Compute local IRLS summands and nlls ---- #

            (
                local_states,
                local_fisher_gradient_nlls_shared_states,
                round_idx,
            ) = local_step(
                local_method=self.make_local_fisher_gradient_nll,
                method_params={
                    "first_iteration_mode": first_iteration_mode
                    if pqn_iter == 0
                    else None,
                    "refit_mode": refit_mode,
                },
                train_data_nodes=train_data_nodes,
                output_local_states=local_states,
                round_idx=round_idx,
                input_local_states=local_states,
                input_shared_state=PQN_shared_state,
                aggregation_id=aggregation_node.organization_id,
                description="Compute local Prox Newton summands and nlls.",
                clean_models=clean_models,
            )

            # ---- Compute global IRLS update and nlls ---- #

            PQN_shared_state, round_idx = aggregation_step(
                aggregation_method=self.choose_step_and_compute_ascent_direction,
                train_data_nodes=train_data_nodes,
                aggregation_node=aggregation_node,
                input_shared_states=local_fisher_gradient_nlls_shared_states,
                round_idx=round_idx,
                description="Update the log fold changes and nlls in IRLS.",
                clean_models=clean_models,
            )

        #### ---- End of training ---- ####

        return local_states, PQN_shared_state, round_idx
