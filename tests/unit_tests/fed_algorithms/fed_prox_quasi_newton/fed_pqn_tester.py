"""A class to implement a unit tester class for Fed Prox Quasi Newton."""
from pathlib import Path

import anndata as ad
import numpy as np
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.remote import remote
from substrafl.remote import remote_data

from fedpydeseq2.core.fed_algorithms import FedProxQuasiNewton
from fedpydeseq2.core.utils import aggregation_step
from fedpydeseq2.core.utils import local_step
from fedpydeseq2.core.utils.layers import reconstruct_adatas
from fedpydeseq2.core.utils.logging import log_remote
from fedpydeseq2.core.utils.logging import log_remote_data
from fedpydeseq2.core.utils.pass_on_results import AggPassOnResults
from tests.unit_tests.fed_algorithms.fed_IRLS_PQN_tester import FedIRLSPQNTester
from tests.unit_tests.fed_algorithms.fed_IRLS_PQN_tester import compute_initial_beta
from tests.unit_tests.unit_test_helpers.pass_on_first_shared_state import (
    AggPassOnFirstSharedState,
)


class FedProxQuasiNewtonTester(
    FedIRLSPQNTester,
    FedProxQuasiNewton,
    AggPassOnResults,
    AggPassOnFirstSharedState,
):
    """A class to implement a unit test for FedProxQuasiNewton.

    This unit step is performed on an optimization problem
    which is solved by ComputeLFC (and therefore by IRLS in a majority
    of cases) in the pipeline. Here, we directly use the FedProxQuasiNewton
    method as an optimization algorithm. It is therefore quite oriented
    towards the particular optimization problem arising in the DESeq2 pipeline.

    Parameters
    ----------
    design_factors : Union[str, list[str]]
        The design factors.

    ref_levels : Optional[dict[str, str]]
        The reference levels.

    continuous_factors : Optional[list[str]]
        The continuous factors.

    PQN_min_mu : float
        The minimum mu.

    max_beta : float
        The maximum beta.

    joblib_backend : str
        The IRLS backend.

    num_jobs : int
        The number of CPUs.

    joblib_verbosity : int
        The joblib verbosity.

    irls_batch_size : int
        The IRLS batch size, i.e. the number of genes used per parallelization
        batch.

    PQN_c1 : float
        The prox quasi newton c_1 constant used in the Armijo line search.

    PQN_ftol : float
        The relative tolerance used as a stopping criterion in the Prox Quasi Newton
        method.

    PQN_num_iters_ls : int
        The number of iterations used in the line search.

    PQN_num_iters : int
        The number of iterations used in the Prox Quasi Newton method.

    reference_data_path : Optional[Union[str, Path]]
        The path to the reference data.

    reference_dds_ref_level : Optional[tuple[str, ...]]
        The reference level of the reference DeseqDataSet.


    Methods
    -------
    init_local_states
        A remote_data method, which initializes the local states by setting the local
        adata. It also returns the normalized counts and the design matrix, in order
        to create the intial beta (Note that this is only for testing purposes)

    compute_start_state
        A remote method, which computes the start state by concatenating the local
        design matrices and the normed counts, and computing the initial beta value
        as in PyDESeq2 (Note that this is only for testing purposes, and is done in
        a federated way in the real pipeline).

    set_beta_init
        A remote_data method, which sets the beta init in the local states, and passes
        on the shared state which is used as an initialization state for the
        Prox Quasi Newton algorithm.

    pass_on_shared_state
        A remote method, which passes on the shared state to the centers.

    local_add_non_zero_genes
        A remote_data method, which adds the non zero genes (stored in the local adata)
        to the shared state, and returns this new shared state to the server.

    build_compute_plan
        A method to build the computation graph to run the Fed Prox Quasi Newton
        algorithm on the LFC computation problem in PyDESeq2.


    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        PQN_min_mu: float = 0.5,
        max_beta: float = 30,
        joblib_backend: str = "loky",
        num_jobs: int = 8,
        joblib_verbosity: int = 3,
        irls_batch_size: int = 100,
        PQN_c1: float = 1e-4,
        PQN_ftol: float = 1e-7,
        PQN_num_iters_ls: int = 20,
        PQN_num_iters: int = 100,
        reference_data_path: str | Path | None = None,
        reference_dds_ref_level: tuple[str, ...] | None = None,
    ):
        super().__init__(
            design_factors=design_factors,
            ref_levels=ref_levels,
            continuous_factors=continuous_factors,
            reference_data_path=reference_data_path,
            reference_dds_ref_level=reference_dds_ref_level,
            PQN_min_mu=PQN_min_mu,
            max_beta=max_beta,
            joblib_backend=joblib_backend,
            num_jobs=num_jobs,
            joblib_verbosity=joblib_verbosity,
            irls_batch_size=irls_batch_size,
            PQN_c1=PQN_c1,
            PQN_ftol=PQN_ftol,
            PQN_num_iters_ls=PQN_num_iters_ls,
            PQN_num_iters=PQN_num_iters,
        )

        #### Define hyper parameters ####

        self.PQN_min_mu = PQN_min_mu
        self.max_beta = max_beta

        # Parameters of the PQN algorithm
        self.PQN_c1 = PQN_c1
        self.PQN_ftol = PQN_ftol
        self.PQN_num_iters_ls = PQN_num_iters_ls
        self.PQN_num_iters = PQN_num_iters

        #### Define job parallelization parameters ####
        self.joblib_verbosity = joblib_verbosity
        self.num_jobs = num_jobs
        self.joblib_backend = joblib_backend
        self.irls_batch_size = irls_batch_size

        self.layers_to_save_on_disk = {
            "local_adata": [
                "_mu_hat",
            ],
            "refit_adata": [
                None,
            ],
        }

    @remote
    @log_remote
    def compute_start_state(self, shared_states: list[dict]) -> dict:
        """Compute the beta initialization, and share to the centers

        Parameters
        ----------
        shared_states : list
            List of shared states.

        Returns
        -------
        dict
            Dictionary containing the global gram matrix.
        """

        # Concatenate the local design matrices
        beta_init = compute_initial_beta(shared_states)
        n_non_zero_genes = beta_init.shape[0]

        # Share it with the centers
        return {
            "beta": beta_init,
            "ascent_direction_on_mask": None,
            "PQN_mask": np.ones((n_non_zero_genes,), dtype=bool),
            "irls_diverged_mask": np.zeros((n_non_zero_genes,), dtype=bool),
            "PQN_diverged_mask": np.zeros((n_non_zero_genes,), dtype=bool),
            "global_reg_nll": np.nan * np.ones((n_non_zero_genes,)),
            "round_number_PQN": 0,
            "newton_decrement_on_mask": None,
        }

    @remote_data
    @log_remote_data
    @reconstruct_adatas
    def local_add_non_zero_genes(
        self, data_from_opener: ad.AnnData, shared_state: dict
    ) -> dict:
        """Initialize the local states.

        This methods sets the local_adata and the local gram matrix.
        from the reference_dds.


        Parameters
        ----------
        data_from_opener : ad.AnnData
            AnnData returned by the opener. Not used.

        shared_state : dict
            Shared state which comes from the last iteration of
            the Fed Prox Quasi Newton algorithm.
            Contains a `beta` and a `PQN_diverged_mask` field.

        Returns
        -------
        local_state : dict
            The local state containing the non zero genes mask and genes
            as an addition to the input shared state.

        """
        non_zero_genes_names = self.local_adata.var_names[
            self.local_adata.varm["non_zero"]
        ]
        non_zero_genes_mask = self.local_adata.varm["non_zero"]
        return {
            "PQN_diverged_mask": shared_state["PQN_diverged_mask"],
            "beta": shared_state["beta"],
            "non_zero_genes_mask": non_zero_genes_mask,
            "non_zero_genes_names": non_zero_genes_names,
        }

    def build_compute_plan(
        self,
        train_data_nodes: list[TrainDataNode],
        aggregation_node: AggregationNode,
        evaluation_strategy=None,
        num_rounds=None,
        clean_models=True,
    ):
        """Build the computation graph to run a FedDESeq2 pipe.

        Parameters
        ----------
        train_data_nodes : list[TrainDataNode]
            List of the train nodes.
        aggregation_node : AggregationNode
            Aggregation node.
        evaluation_strategy : EvaluationStrategy
            Not used.
        num_rounds : int
            Number of rounds. Not used.
        clean_models : bool
            Whether to clean the models after the computation. (default: ``False``).
        """
        round_idx = 0
        local_states: dict[str, LocalStateRef] = {}

        #### Create a reference DeseqDataset object ####

        local_states, shared_states, round_idx = self.set_local_reference_dataset(
            train_data_nodes,
            aggregation_node,
            local_states,
            round_idx,
            clean_models=clean_models,
        )

        #### Load reference dataset as local_adata and set local states ####

        # This step also shares the counts and design matrix to compute
        # the beta initialization directly

        local_states, shared_states, round_idx = local_step(
            local_method=self.init_local_states,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=None,
            aggregation_id=aggregation_node.organization_id,
            description="Initialize local states",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### Compute the initialization shared state ####

        starting_shared_state, round_idx = aggregation_step(
            aggregation_method=self.compute_start_state,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Compute the global gram matrix.",
            clean_models=clean_models,
        )

        #### Set the beta init in the local states ####

        local_states, shared_states, round_idx = local_step(
            local_method=self.set_beta_init,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=starting_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Set the beta init",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        starting_shared_state, round_idx = aggregation_step(
            aggregation_method=self.pass_on_shared_state,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            round_idx=round_idx,
            description="Pass on the shared state",
            clean_models=clean_models,
        )

        #### Perform fed PQN ####

        local_states, pqn_shared_state, round_idx = self.run_fed_PQN(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            PQN_shared_state=starting_shared_state,
            first_iteration_mode=None,
            round_idx=round_idx,
            clean_models=clean_models,
        )

        #### Share the results ####

        # Local step to add non_zero genes to the shared state
        local_states, shared_states, round_idx = local_step(
            local_method=self.local_add_non_zero_genes,
            train_data_nodes=train_data_nodes,
            output_local_states=local_states,
            input_local_states=local_states,
            input_shared_state=pqn_shared_state,
            aggregation_id=aggregation_node.organization_id,
            description="Add non zero genes to the shared state",
            round_idx=round_idx,
            clean_models=clean_models,
        )

        aggregation_step(
            aggregation_method=self.pass_on_results,
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            input_shared_states=shared_states,
            description="Save the first shared state",
            round_idx=round_idx,
            clean_models=False,
        )
