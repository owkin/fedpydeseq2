import pickle as pkl
from pathlib import Path
from typing import Any
from typing import Literal

import anndata as ad
from substrafl import ComputePlanBuilder
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef

from fedpydeseq2.core.deseq2_core import DESeq2FullPipe
from fedpydeseq2.core.utils.logging import log_save_local_state


class DESeq2Strategy(ComputePlanBuilder, DESeq2FullPipe):
    """DESeq2 strategy.

    This strategy is an implementation of the DESeq2 algorithm.

    Parameters
    ----------
    design_factors : str or list
        Name of the columns of metadata to be used as design variables.
        If you are using categorical and continuous factors, you must put
        all of them here.

    ref_levels : dict, optional
        An optional dictionary of the form ``{"factor": "test_level"}``
        specifying for each factor the reference (control) level against which
        we're testing, e.g. ``{"condition", "A"}``. Factors that are left out
        will be assigned random reference levels. (default: ``None``).

    continuous_factors : list, optional
        An optional list of continuous (as opposed to categorical) factors. Any factor
        not in ``continuous_factors`` will be considered categorical
        (default: ``None``).

    contrast : list, optional
        A list of three strings, in the following format:
        ``['variable_of_interest', 'tested_level', 'ref_level']``.
        Names must correspond to the metadata data passed to the DeseqDataSet.
        E.g., ``['condition', 'B', 'A']`` will measure the LFC of 'condition B' compared
        to 'condition A'.
        For continuous variables, the last two strings should be left empty, e.g.
        ``['measurement', '', ''].``
        If None, the last variable from the design matrix is chosen
        as the variable of interest, and the reference level is picked alphabetically.
        (default: ``None``).

    lfc_null : float
        The (log2) log fold change under the null hypothesis. (default: ``0``).

    alt_hypothesis : str, optional
        The alternative hypothesis for computing wald p-values. By default, the normal
        Wald test assesses deviation of the estimated log fold change from the null
        hypothesis, as given by ``lfc_null``.
        One of ``["greaterAbs", "lessAbs", "greater", "less"]`` or ``None``.
        The alternative hypothesis corresponds to what the user wants to find rather
        than the null hypothesis. (default: ``None``).

    min_replicates : int
        Minimum number of replicates a condition should have
        to allow refitting its samples. (default: ``7``).

    min_disp : float
        Lower threshold for dispersion parameters. (default: ``1e-8``).

    max_disp : float
        Upper threshold for dispersion parameters.
        Note: The threshold that is actually enforced is max(max_disp, len(counts)).
        (default: ``10``).

    grid_batch_size : int
        The number of genes to put in each batch for local parallel processing.
        (default: ``100``).

    grid_depth : int
        The number of grid interval selections to perform (if using GridSearch).
        (default: ``3``).

    grid_length : int
        The number of grid points to use for the grid search (if using GridSearch).
        (default: ``100``).

    num_jobs : int
        The number of jobs to use for local parallel processing in MLE tasks.
        (default: ``8``).

    independent_filter : bool
        Whether to perform independent filtering to correct p-value trends.
        (default: ``True``).

    alpha : float
        P-value and adjusted p-value significance threshold (usually 0.05).
        (default: ``0.05``).

    min_mu : float
        The minimum value of the mean parameter mu. (default: ``0.5``).

    beta_tol : float
        The tolerance for the beta parameter. (default: ``1e-8``). This is used
        in the IRLS algorithm to stop the iterations when the relative change in
        the beta parameter is smaller than beta_tol.

    max_beta : float
        The maximum value for the beta parameter. (default: ``30``).

    irls_num_iter : int
        The number of iterations to perform in the IRLS algorithm. (default: ``20``).

    joblib_backend : str
        The backend to use for parallel processing. (default: ``loky``).

    joblib_verbosity : int
        The verbosity level of joblib. (default: ``3``).

    irls_batch_size : int
        The number of genes to put in each batch for local parallel processing in the
        IRLS algorithm. (default: ``100``).

    PQN_c1 : float
        The Armijo line search constant for the prox newton.

    PQN_ftol : float
        The functional stopping criterion for the prox newton method (relative error
        smaller than ftol).

    PQN_num_iters_ls : int
        The number of iterations performed in the line search at each prox newton step.

    PQN_num_iters : int
        The number of iterations in the prox newton catch of IRLS.

    PQN_min_mu : float
        The minimum value for mu in the prox newton method.

    refit_cooks : bool
        Whether to refit the model after computation of Cooks distance.
        (default: ``True``).

    cooks_filter : bool
        Whether to filter out genes with high Cooks distance in the pvalue computation.
        (default: ``True``).

    save_layers_to_disk : bool
        Whether to save the layers to disk. (default: ``False``).
        If True, the layers will be saved to disk.

    trimmed_mean_num_iter: int
        The number of iterations to use when computing the trimmed mean
        in a federated way, i.e. the number of dichotomy steps. The default is
        40.
    """

    def __init__(
        self,
        design_factors: str | list[str],
        ref_levels: dict[str, str] | None = None,
        continuous_factors: list[str] | None = None,
        contrast: list[str] | None = None,
        lfc_null: float = 0.0,
        alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"]
        | None = None,
        min_replicates: int = 7,
        min_disp: float = 1e-8,
        max_disp: float = 10.0,
        grid_batch_size: int = 250,
        grid_depth: int = 3,
        grid_length: int = 100,
        num_jobs=8,
        min_mu: float = 0.5,
        beta_tol: float = 1e-8,
        max_beta: float = 30,
        irls_num_iter: int = 20,
        joblib_backend: str = "loky",
        joblib_verbosity: int = 0,
        irls_batch_size: int = 100,
        independent_filter: bool = True,
        alpha: float = 0.05,
        PQN_c1: float = 1e-4,
        PQN_ftol: float = 1e-7,
        PQN_num_iters_ls: int = 20,
        PQN_num_iters: int = 100,
        PQN_min_mu: float = 0.0,
        refit_cooks: bool = True,
        cooks_filter: bool = True,
        save_layers_to_disk: bool = False,
        trimmed_mean_num_iter: int = 40,
        *args,
        **kwargs,
    ):
        # Add all arguments to super init so that they can be retrieved by nodes.
        super().__init__(
            design_factors=design_factors,
            ref_levels=ref_levels,
            continuous_factors=continuous_factors,
            contrast=contrast,
            lfc_null=lfc_null,
            alt_hypothesis=alt_hypothesis,
            min_replicates=min_replicates,
            min_disp=min_disp,
            max_disp=max_disp,
            grid_batch_size=grid_batch_size,
            grid_depth=grid_depth,
            grid_length=grid_length,
            num_jobs=num_jobs,
            min_mu=min_mu,
            beta_tol=beta_tol,
            max_beta=max_beta,
            irls_num_iter=irls_num_iter,
            joblib_backend=joblib_backend,
            joblib_verbosity=joblib_verbosity,
            irls_batch_size=irls_batch_size,
            independent_filter=independent_filter,
            alpha=alpha,
            PQN_c1=PQN_c1,
            PQN_ftol=PQN_ftol,
            PQN_num_iters_ls=PQN_num_iters_ls,
            PQN_num_iters=PQN_num_iters,
            PQN_min_mu=PQN_min_mu,
            refit_cooks=refit_cooks,
            cooks_filter=cooks_filter,
            trimmed_mean_num_iter=trimmed_mean_num_iter,
        )

        #### Define hyper parameters ####

        self.min_disp = min_disp
        self.max_disp = max_disp
        self.grid_batch_size = grid_batch_size
        self.grid_depth = grid_depth
        self.grid_length = grid_length
        self.min_mu = min_mu
        self.beta_tol = beta_tol
        self.max_beta = max_beta

        # Parameters of the IRLS algorithm
        self.irls_num_iter = irls_num_iter
        self.min_replicates = min_replicates
        self.PQN_c1 = PQN_c1
        self.PQN_ftol = PQN_ftol
        self.PQN_num_iters_ls = PQN_num_iters_ls
        self.PQN_num_iters = PQN_num_iters
        self.PQN_min_mu = PQN_min_mu

        # Parameters for the trimmed mean computation
        self.trimmed_mean_num_iter = trimmed_mean_num_iter

        #### Stat parameters
        self.independent_filter = independent_filter
        self.alpha = alpha

        #### Define job parallelization parameters ####

        self.num_jobs = num_jobs
        self.joblib_verbosity = joblib_verbosity
        self.joblib_backend = joblib_backend
        self.irls_batch_size = irls_batch_size

        #### Define quantities to set the design ####

        # Convert design_factors to list if a single string was provided.
        self.design_factors = (
            [design_factors] if isinstance(design_factors, str) else design_factors
        )

        self.ref_levels = ref_levels
        self.continuous_factors = continuous_factors

        if self.continuous_factors is not None:
            self.categorical_factors = [
                factor
                for factor in self.design_factors
                if factor not in self.continuous_factors
            ]
        else:
            self.categorical_factors = self.design_factors

        self.contrast = contrast

        #### Set test parameters ####
        self.lfc_null = lfc_null
        self.alt_hypothesis = alt_hypothesis

        #### If we want to refit cooks outliers
        self.refit_cooks = refit_cooks

        #### Define quantities to compute statistics
        self.cooks_filter = cooks_filter

        #### Set attributes to be registered / saved later on ####
        self.local_adata: ad.AnnData | None = None
        self.refit_adata: ad.AnnData | None = None
        self.results: dict | None = None

        #### Save layers to disk
        self.save_layers_to_disk = save_layers_to_disk

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
            Whether to clean the models after the computation. (default: ``True``).
        """
        round_idx = 0
        local_states: dict[str, LocalStateRef] = {}

        self.run_deseq_pipe(
            train_data_nodes=train_data_nodes,
            aggregation_node=aggregation_node,
            local_states=local_states,
            round_idx=round_idx,
            clean_models=clean_models,
        )

    @log_save_local_state
    def save_local_state(self, path: Path) -> None:
        """Save the local state of the strategy.

        Parameters
        ----------
        path : Path
            Path to the file where to save the state. Automatically handled by subtrafl.
        """
        state_to_save = {
            "local_adata": self.local_adata,
            "refit_adata": self.refit_adata,
            "results": self.results,
        }
        with open(path, "wb") as file:
            pkl.dump(state_to_save, file)

    def load_local_state(self, path: Path) -> Any:
        """Load the local state of the strategy.

        Parameters
        ----------
        path : Path
            Path to the file where to load the state from. Automatically handled by
            subtrafl.
        """
        with open(path, "rb") as file:
            state_to_load = pkl.load(file)

        self.local_adata = state_to_load["local_adata"]
        self.refit_adata = state_to_load["refit_adata"]
        self.results = state_to_load["results"]

        return self

    @property
    def num_round(self):
        """Return the number of round in the strategy.

        TODO do something clever with this.

        Returns
        -------
        int
            Number of round in the strategy.
        """
        return None
