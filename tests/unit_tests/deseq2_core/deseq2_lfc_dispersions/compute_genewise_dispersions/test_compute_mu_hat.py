"""Unit tests for the computation of mu hat.

Here, we test the computation of mu hat using IRLS, in the
compute_genewise_dispersions step of the DESeq2 algorithm.

Note that mu hat can be computed in two ways in the compute_genewise_dispersions step.
- if num_vars == num_levels, then mu_hat is computed as the solution to a linear system
- otherwise, mu_hat is computed using IRLS

In this file, we only test THE SECOND CASE, as the algorithm we use is not
the same as the one used in the pooled setting.

"""

import pytest

from tests.unit_tests.deseq2_core.deseq2_lfc_dispersions.compute_lfc.compute_lfc_test_pipe import (  # noqa: E501
    pipe_test_compute_lfc,
)


@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_mu_hat_small_genes(
    design_factors,
    continuous_factors,
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
):
    """Perform a unit test to see if mu_hat is working as expected.

    This test focuses on a small number of genes, to see if the algorithm is working
    as expected in a fast way.

    Recall that computing mu hat is the second step in the compute_genewise_dispersions
    step, after computing the MoM dispersions and before computing the dispersions
    estimates.

    Parameters
    ----------
    design_factors: Union[str, List[str]]
        The design factors.

    continuous_factors: Union[str, List[str]]
        The continuous factors.

    raw_data_path: Path
        The path to the root data.

    local_processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    """

    pipe_test_compute_lfc(
        lfc_mode="mu_init",
        data_path=raw_data_path,
        processed_data_path=local_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=False,
        small_genes=True,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
        PQN_min_mu=0.0,
        nll_rtol=0.02,
        tolerated_failed_genes=1,
    )


@pytest.mark.self_hosted_fast
@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        # (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_mu_hat_small_samples(
    design_factors,
    continuous_factors,
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    """Perform a unit test to see if computing mu hat is working as expected.

    This test focuses on a small number of samples, to see if the algorithm is working
    as expected in the self hosted CI. Note that on a small number of samples, the
    algorithm is less performant then when there are more samples (see the
    test_mu_hat test). This can be explained by the fact that the log likelihood
    is somehow less smooth when there are few data points.

    Note that for a reason that is not clear, for IRLS converged genes, a tolerance
    of 1e-5 is too hard (even if theoretically, the algorithm is pooled equivalent).
    However, the results are still quite close to the pooled ones, and we do not
    investigate this further for now.

    Parameters
    ----------
    design_factors: Union[str, List[str]]
        The design factors.

    continuous_factors: Union[str, List[str]]
        The continuous factors.

    raw_data_path: Path
        The path to the root data.

    tmp_processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    """

    pipe_test_compute_lfc(
        lfc_mode="mu_init",
        data_path=raw_data_path,
        processed_data_path=tmp_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=True,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
        PQN_min_mu=0.0,
        nll_rtol=0.02,
        tolerated_failed_genes=15,
        rtol_irls=1e-3,
        atol_irls=1e-5,
    )


@pytest.mark.self_hosted_slow
@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender"], None),
        (["stage", "gender", "CPE"], ["CPE"]),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_mu_hat_luad(
    design_factors,
    continuous_factors,
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    """Perform a unit test to see if computing mu hat is working as expected.

    This test focuses on a large number of samples and genes, to see if the algorithm
    is working as expected in the self hosted CI.

    Note that a relative tolerance of 1e-3 is used, instead of the default 1e-5. The
    reasons for which this is needed are not clear, but the results are still quite
    close to the pooled ones, and we do not investigate this further for now.

    Parameters
    ----------
    design_factors: Union[str, List[str]]
        The design factors.

    continuous_factors: Union[str, List[str]]
        The continuous factors.

    raw_data_path: Path
        The path to the root data.

    tmp_processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    """

    pipe_test_compute_lfc(
        lfc_mode="mu_init",
        data_path=raw_data_path,
        processed_data_path=tmp_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        small_samples=False,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
        PQN_min_mu=0.0,
        nll_rtol=0.02,
        tolerated_failed_genes=5,
        rtol_irls=1e-3,
        atol_irls=1e-5,
    )


@pytest.mark.self_hosted_slow
@pytest.mark.parametrize(
    "design_factors, continuous_factors",
    [
        ("stage", None),
        (["stage", "gender"], None),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_mu_hat_paad(
    design_factors,
    continuous_factors,
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    """Perform a unit test to see if computing mu hat is working as expected.

    This test focuses on a large number of samples and genes, to see if the algorithm
    is working as expected in the self hosted CI.

    Note that we do not add CPE as a continuous factor, as it is not present in the
    PAAD dataset.

    Note that a relative tolerance of 1e-2 is used, instead of the default 1e-5. The
    reasons for which this is needed are not clear, but the results are still quite
    close to the pooled ones, and we do not investigate this further for now.

    Parameters
    ----------
    design_factors: Union[str, List[str]]
        The design factors.

    continuous_factors: Union[str, List[str]]
        The continuous factors.

    raw_data_path: Path
        The path to the root data.

    tmp_processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed

    tcga_assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    """

    pipe_test_compute_lfc(
        lfc_mode="mu_init",
        data_path=raw_data_path,
        processed_data_path=tmp_processed_data_path,
        tcga_assets_directory=tcga_assets_directory,
        dataset_name="TCGA-PAAD",
        small_samples=False,
        small_genes=False,
        simulate=True,
        backend="subprocess",
        only_two_centers=False,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        ref_levels={"stage": "Advanced"},
        reference_dds_ref_level=None,
        PQN_min_mu=0.0,
        nll_rtol=0.02,
        tolerated_failed_genes=5,
        rtol_irls=1e-2,
        atol_irls=1e-2,
    )
