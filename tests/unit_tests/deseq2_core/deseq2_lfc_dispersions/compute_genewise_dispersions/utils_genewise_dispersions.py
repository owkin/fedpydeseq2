import numpy as np

from fedpydeseq2.core.utils import vec_loss


def perform_dispersions_and_nll_relative_check(
    fl_dispersions,
    pooled_dds,
    dispersions_param_name: str = "genewise_dispersions",
    prior_reg: bool = False,
    rtol=0.02,
    atol=1e-3,
    nll_rtol=0.02,
    nll_atol=1e-3,
    tolerated_failed_genes=0,
):
    """Perform the relative error check on the dispersions and likelihoods.

    This function checks the relative error on the dispersions. If the relative error
    is above rtol, it checks the likelihoods. If the likelihoods are above rtol higher
    than the pooled likelihoods, we fail the test (with a certain tolerance
    for failure).

    Parameters
    ----------
    fl_dispersions: np.ndarray
        The dispersions computed by the federated algorithm.

    pooled_dds: DeseqDataSet
        The pooled DeseqDataSet.

    dispersions_param_name: str
        The name of the parameter in the varm that contains the dispersions.

    prior_reg: bool
        If True, the prior regularization is applied.

    alpha_hat: Optional[np.ndarray]
        The alpha_hat parameter for the prior regularization.

    prior_disp_var: float
        The prior_disp_var parameter for the prior regularization.

    rtol: float
        The relative tolerance for between the FL and pooled dispersions.

    atol: float
        The absolute tolerance for between the FL and pooled dispersions.

    nll_rtol: float
        The relative tolerance for between the FL and pooled likelihoods, in the

    nll_atol: float
        The absolute tolerance for between the FL and pooled likelihoods, in the

    tolerated_failed_genes: int
        The number of genes that are allowed to fail the relative nll criterion.
    """
    # If any of the relative errors is above 2%, check likelihoods
    pooled_dispersions = pooled_dds.varm[dispersions_param_name]
    accepted_error = np.abs(pooled_dispersions) * rtol + atol
    absolute_error = np.abs(fl_dispersions - pooled_dispersions)

    if np.any(absolute_error > accepted_error):
        to_check = absolute_error > accepted_error
        print(
            f"{to_check.sum()} genes do not pass the relative error criterion."
            f" Genes that do not pass the relative error criterion with rtol {rtol} and"
            f" atol {atol} are : "
        )
        print(pooled_dds.var_names[to_check])

        counts = pooled_dds[:, to_check].X
        design = pooled_dds.obsm["design_matrix"].values
        mu = pooled_dds[:, to_check].layers["_mu_hat"]

        if prior_reg:
            alpha_hat = pooled_dds[:, to_check].varm["fitted_dispersions"]
            prior_disp_var = pooled_dds.uns["prior_disp_var"]
        else:
            alpha_hat = None
            prior_disp_var = None

        # Compute the likelihoods
        fl_nll = vec_loss(
            counts,
            design,
            mu,
            fl_dispersions[to_check],
            prior_reg=prior_reg,
            alpha_hat=alpha_hat,
            prior_disp_var=prior_disp_var,
        )
        pooled_nll = vec_loss(
            counts,
            design,
            mu,
            pooled_dds[:, to_check].varm[dispersions_param_name],
            prior_reg=prior_reg,
            alpha_hat=alpha_hat,
            prior_disp_var=prior_disp_var,
        )

        # Check that FL likelihood is smaller than pooled likelihood
        nll_error = fl_nll - pooled_nll
        nll_accepted_error = np.abs(pooled_nll) * nll_rtol + nll_atol

        failed_nll_criterion = nll_error > nll_accepted_error

        if np.sum(failed_nll_criterion) > 0:
            print(
                f"{failed_nll_criterion.sum()} genes do not pass the nll criterion."
                f"The corresponding gene names are : "
            )
            print(pooled_dds.var_names[to_check][failed_nll_criterion])

            assert np.sum(failed_nll_criterion) <= tolerated_failed_genes
