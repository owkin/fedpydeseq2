"""Module testing the logmeans computed in the DESeq2Strategy."""

import os
from itertools import product

import pytest
from fedpydeseq2_datasets.constants import TCGADatasetNames

from .test_deseq2_pipe_utils import pipeline_to_test

COOKS_FILTER = [False, True]

DESIGN_FACTORS = ["stage", ["gender", "stage"]]


@pytest.mark.self_hosted_slow
@pytest.mark.parametrize(
    "independent_filter, cooks_filter, design_factors",
    [
        (independent_filter, cooks_filter, design_factors)
        for independent_filter, cooks_filter, design_factors in product(
            [True, False], COOKS_FILTER, DESIGN_FACTORS
        )
        if design_factors == "stage" or (not independent_filter and not cooks_filter)
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_simulation_mode_small_samples(
    independent_filter: bool,
    cooks_filter: bool,
    design_factors,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    """Compare FL and pooled deseq2 pipelines.

    The data is TCGA-LUAD, restricted to a small number of samples.

    Parameters
    ----------
    independent_filter: bool
        If true, use the independent filtering step. If not, use standard
        p-value adjustment.
    cooks_filter: bool
        If true, the Cook's filtering is applied at the end of the pipeline.
    design_factors: str or list
        The design factors to use.
    raw_data_path : Path
        The path to the root data.
    processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.
    """
    pipeline_to_test(
        raw_data_path,
        processed_data_path,
        tcga_assets_directory,
        small_samples=True,
        simulate=True,
        cooks_filter=cooks_filter,
        independent_filter=independent_filter,
        only_two_centers=False,
        design_factors=design_factors,
    )


@pytest.mark.parametrize(
    "only_two_centers, independent_filter, cooks_filter, design_factors",
    [
        (only_two_centers, independent_filter, cooks_filter, design_factors)
        for (
            only_two_centers,
            independent_filter,
            cooks_filter,
            design_factors,
        ) in product([True, False], [True, False], COOKS_FILTER, DESIGN_FACTORS)
        if (
            only_two_centers
            and independent_filter
            and cooks_filter
            and design_factors == "stage"
        )
        or (
            not only_two_centers
            and (design_factors == "stage" or (cooks_filter and independent_filter))
        )
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_simulation_mode_small_genes(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    only_two_centers,
    independent_filter,
    cooks_filter,
    design_factors,
):
    """Compare FL and pooled deseq2 pipelines.

    The data is TCGA-LUAD, restricted to a small number of genes.

    Parameters
    ----------
    raw_data_path : Path
        The path to the root data.
    processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.
    only_two_centers: bool
        If true, restrict the data to two centers.
    independent_filter: bool
        If true, use the independent filtering step. If not, use standard
        p-value adjustment.
    cooks_filter: bool
        If true, the Cook's filtering is applied at the end of the pipeline.
    design_factors : str or list
        The design factors to use.
    """
    pipeline_to_test(
        raw_data_path,
        processed_data_path,
        tcga_assets_directory,
        small_genes=True,
        simulate=True,
        only_two_centers=only_two_centers,
        independent_filter=independent_filter,
        cooks_filter=cooks_filter,
        refit_cooks=True,
        design_factors=design_factors,
    )


@pytest.mark.parametrize(
    "design_factors, continuous_factors, contrast",
    [
        (["stage", "gender"], None, ["stage", "Advanced", "Non-advanced"]),
        (["stage", "gender", "CPE"], ["CPE"], ["CPE", "", ""]),
        (["stage", "gender", "CPE"], ["CPE"], ["gender", "female", "male"]),
    ],
)
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_multifactor_on_simulation_mode_small_genes(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    design_factors,
    continuous_factors,
    contrast,
):
    """Compare FL and pooled deseq2 pipelines.

    The data is TCGA-LUAD, restricted to a small number of genes.

    Parameters
    ----------
    raw_data_path : Path
        The path to the root data.
    processed_data_path : Path
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
        raw_data_path,
        processed_data_path,
        tcga_assets_directory,
        small_genes=True,
        simulate=True,
        only_two_centers=True,
        independent_filter=True,
        cooks_filter=True,
        refit_cooks=True,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        contrast=contrast,
    )


@pytest.mark.parametrize(
    "cooks_filter, design_factors", list(product(COOKS_FILTER, DESIGN_FACTORS))
)
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_simulation_mode_small_genes_refit_cooks_false(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    cooks_filter,
    design_factors,
):
    """Compare FL and pooled deseq2 pipelines.

    The data is TCGA-LUAD, restricted to a small number of genes.

    Parameters
    ----------
    raw_data_path : Path
        The path to the root data.
    processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.
    cooks_filter: bool
        If true, the Cook's filtering is applied at the end of the pipeline.
    design_factors : str or list
        The design factors to use.
    """
    pipeline_to_test(
        raw_data_path,
        processed_data_path,
        tcga_assets_directory,
        small_genes=True,
        simulate=True,
        only_two_centers=True,
        independent_filter=True,
        cooks_filter=cooks_filter,
        refit_cooks=False,
        design_factors=design_factors,
    )


@pytest.mark.docker
@pytest.mark.self_hosted_slow
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_docker_mode_on_self_hosted_small_genes(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    # This is a workaround to avoid the docker daemon issue.
    os.environ["DOCKER_HOST"] = "unix:///run/user/1000/docker.sock"
    pipeline_to_test(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        assets_directory=tcga_assets_directory,
        dataset_name="TCGA-LUAD",
        simulate=False,
        backend="docker",
        small_genes=True,
        cooks_filter=True,
    )


@pytest.mark.self_hosted_slow
@pytest.mark.parametrize(
    "dataset_name",
    ["TCGA-LUAD", "TCGA-PAAD"],
)
@pytest.mark.usefixtures(
    "raw_data_path", "processed_data_path", "tcga_assets_directory"
)
def test_end_to_end_on_subprocess_mode_on_self_hosted_small_genes_small_samples(
    dataset_name: TCGADatasetNames,
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
):
    """Compare FL and pooled deseq2 pipelines.

    The data is TCGA-LUAD on a self hosted runner.

    Parameters
    ----------
    dataset_name: TCAGDatasetNames
        The name of the dataset, for example "TCGA-LUAD".
    raw_data_path : Path
        The path to the root data.
    processed_data_path : Path
        The path to the processed data. The subdirectories will
        be created if needed.
    tcga_assets_directory : Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.
    """
    # Get the ground truth.
    pipeline_to_test(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        assets_directory=tcga_assets_directory,
        dataset_name=dataset_name,
        simulate=False,
        independent_filter=True,
        cooks_filter=True,
        backend="subprocess",
        small_genes=True,
        small_samples=True,
        only_two_centers=True,
    )
