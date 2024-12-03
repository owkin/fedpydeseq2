"""Module testing the logmeans computed in the DESeq2Strategy."""
from itertools import product

import pytest

from .test_deseq2_pipe_utils import pipeline_to_test

COOKS_FILTER = [True, False]


@pytest.mark.local
@pytest.mark.parametrize(
    "only_two_centers, independent_filter, cooks_filter",
    list(product([True], [True], [True])),
)
@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_subprocess_mode_local_small_genes(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
    only_two_centers,
    independent_filter,
    cooks_filter,
):
    """Compare FL and pooled deseq2 pipelines.

    This test is here to be able to locally test the pipeline on non simulated data.

    The data is TCGA-LUAD.

    Parameters
    ----------
    raw_data_path : Path
        The path to the root data.
    local_processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.
    only_two_centers: bool
        If true, restrict the data to two centers.
    independent_filter: bool
        If true, use the independent filtering step. If not, use standard
    cooks_filter: bool
        If true, the Cook's filtering is applied at the end of the pipeline.

    """
    pipeline_to_test(
        raw_data_path=raw_data_path,
        processed_data_path=local_processed_data_path,
        assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        simulate=False,
        small_genes=True,
        cooks_filter=cooks_filter,
        only_two_centers=only_two_centers,
        independent_filter=independent_filter,
    )


@pytest.mark.local
@pytest.mark.parametrize(
    "independent_filter, cooks_filter",
    product([True, False], [True, False]),
)
@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_simu_mode_local(
    independent_filter: bool,
    cooks_filter: bool,
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
):
    """Compare FL and pooled deseq2 pipelines.

    This test is here to be able to locally test the pipeline on non simulated data.

    The data is TCGA-LUAD.

    Parameters
    ----------
    independent_filter: bool
        If true, use the independent filtering step. If not, use standard
        p-value adjustment.
    cooks_filter: bool
        If true, the Cook's filtering is applied at the end of the pipeline.
    raw_data_path : Path
        The path to the root data.
    local_processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    """
    pipeline_to_test(
        raw_data_path=raw_data_path,
        processed_data_path=local_processed_data_path,
        assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        simulate=True,
        small_genes=False,
        cooks_filter=cooks_filter,
        independent_filter=independent_filter,
    )


@pytest.mark.local
@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_subprocess_mode_local(
    raw_data_path,
    local_processed_data_path,
    tcga_assets_directory,
):
    """Compare FL and pooled deseq2 pipelines.

    This test is here to be able to locally test the pipeline on non simulated data.

    The data is TCGA-LUAD.

    Parameters
    ----------
    raw_data_path : Path
        The path to the root data.
    local_processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    """
    pipeline_to_test(
        raw_data_path=raw_data_path,
        processed_data_path=local_processed_data_path,
        assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        simulate=False,
        small_genes=False,
        small_samples=False,
        clean_models=False,
        only_two_centers=True,
    )


@pytest.mark.local
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_subprocess_mode_local_on_self_hosted_keep_models(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    """Compare FL and pooled deseq2 pipelines.

    This test is here to be able to locally test the pipeline on non simulated data.

    The data is TCGA-LUAD.

    Parameters
    ----------
    raw_data_path : Path
        The path to the root data.
    tmp_processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    """
    pipeline_to_test(
        raw_data_path=raw_data_path,
        processed_data_path=tmp_processed_data_path,
        assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        simulate=False,
        small_genes=False,
        small_samples=False,
        clean_models=False,
        only_two_centers=True,
    )


@pytest.mark.local
@pytest.mark.parametrize(
    "design_factors, continuous_factors, contrast",
    [
        (["stage", "gender"], None, ["stage", "Advanced", "Non-advanced"]),
        (["stage", "gender", "CPE"], ["CPE"], ["gender", "female", "male"]),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "tmp_processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_multifactor_on_subprocess_mode_local_on_self_hosted_keep_models(
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
    contrast,
):
    """Compare FL and pooled deseq2 pipelines.

    This test is here to be able to locally test the pipeline on non simulated data.
    The keep_models flag is set to True, in order to evaluate the memory usage.

    The data is TCGA-LUAD.

    Parameters
    ----------
    raw_data_path : Path
        The path to the root data.
    tmp_processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.
    design_factors: str or list[str]
        The design factors to use.
    continuous_factors: list[str] or None
        The continuous factors to use.
    contrast: list[str] or None
        The contrast to use.

    """
    pipeline_to_test(
        raw_data_path=raw_data_path,
        processed_data_path=tmp_processed_data_path,
        assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        simulate=False,
        small_genes=False,
        small_samples=False,
        clean_models=False,
        only_two_centers=True,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        contrast=contrast,
    )


@pytest.mark.local
@pytest.mark.parametrize(
    "independent_filter, cooks_filter",
    product([True, False], COOKS_FILTER),
)
@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_simulation_mode_local_small_genes(
    independent_filter: bool,
    cooks_filter: bool,
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    """Compare FL and pooled deseq2 pipelines.

    The data is TCGA-LUAD, restricted to a small number of genes.

    Parameters
    ----------
    independent_filter: bool
        If true, use the independent filtering step. If not, use standard
        p-value adjustment.
    cooks_filter: bool
        If true, the Cook's filtering is applied at the end of the pipeline.
    raw_data_path : Path
        The path to the root data.
    tmp_processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    """
    pipeline_to_test(
        raw_data_path,
        tmp_processed_data_path,
        tcga_assets_directory,
        small_genes=True,
        small_samples=False,
        cooks_filter=cooks_filter,
        simulate=True,
        only_two_centers=False,
        independent_filter=independent_filter,
    )


@pytest.mark.local
@pytest.mark.parametrize(
    "independent_filter, cooks_filter",
    product(
        [True, False],
        COOKS_FILTER,
    ),
)
@pytest.mark.usefixtures(
    "raw_data_path", "local_processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_simulation_mode_local_small_samples(
    independent_filter: bool,
    cooks_filter: bool,
    raw_data_path,
    tmp_processed_data_path,
    tcga_assets_directory,
):
    """Compare FL and pooled deseq2 pipelines.

    The data is TCGA-LUAD, restricted to a small number of genes.

    Parameters
    ----------
    independent_filter: bool
        If true, use the independent filtering step. If not, use standard
        p-value adjustment.
    cooks_filter: bool
        If true, the Cook's filtering is applied at the end of the pipeline.
    raw_data_path : Path
        The path to the root data.
    tmp_processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.

    """
    pipeline_to_test(
        raw_data_path,
        tmp_processed_data_path,
        tcga_assets_directory,
        small_genes=False,
        small_samples=True,
        simulate=True,
        only_two_centers=False,
        independent_filter=independent_filter,
    )
