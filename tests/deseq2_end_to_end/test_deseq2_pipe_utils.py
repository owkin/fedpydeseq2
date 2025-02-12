"""Main utilities to test the deseq2 pipeline."""

from pathlib import Path
from typing import Literal

from fedpydeseq2 import DESeq2Strategy
from tests.conftest import DEFAULT_LOGGING_CONFIGURATION_PATH
from tests.tcga_testing_pipe import run_tcga_testing_pipe


def pipeline_to_test(
    raw_data_path: Path,
    processed_data_path: Path,
    assets_directory: Path,
    dataset_name="TCGA-LUAD",
    small_samples=False,
    small_genes=False,
    simulate=True,
    independent_filter: bool = True,
    cooks_filter: bool = True,
    refit_cooks: bool = True,
    backend="subprocess",
    only_two_centers=False,
    design_factors: str | list[str] = "stage",
    ref_levels: dict[str, str] | None = {"stage": "Advanced"},  # noqa: B006
    continuous_factors: list[str] | None = None,
    contrast: list[str] | None = None,
    alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"] | None = None,
    lfc_null: float = 0.0,
    reference_dds_ref_level: tuple[str, ...] | None = ("stage", "Advanced"),
    clean_models: bool = True,
    logging_config_file_path: str | Path | None = DEFAULT_LOGGING_CONFIGURATION_PATH,
):
    """Compare FL and pooled deseq2 pipelines.

    Parameters
    ----------
    raw_data_path: Path
        The path to the root data .
    processed_data_path: Path
        The path to the processed data. The subdirectories will
        be created if needed
    assets_directory: Path
        The path to the assets directory. It must contain the
        opener.py file and the description.md file.
    dataset_name: TCGADatasetNames
        The name of the dataset, for example "TCGA-LUAD".
    small_samples: bool
        Whether to use a small number of samples.
        If True, the number of samples is reduced to 10 per center.
    small_genes: bool
        Whether to use a small number of genes.
        If True, the number of genes is reduced to 100.
    simulate: bool
        If true, use the simulation mode, otherwise use the subprocess mode.
    independent_filter: bool
        If true, use the independent filtering step. If not, use standard
        p-value adjustment.
    cooks_filter: bool
        If true, the Cook's filtering is applied at the end of the pipeline.
    refit_cooks: bool
        If true, refit the Cook's filtering.
    backend: str
        The backend to use. Either "subprocess" or "docker".
    only_two_centers: bool
        If true, restrict the data to two centers.
    design_factors: str or list
        The design factors to use.
    ref_levels: dict or None
        The reference levels of the design factors.
    continuous_factors: list or None
        The continuous factors to use.
    contrast: list or None
        The contrast to use.
    alt_hypothesis: Literal["greaterAbs", "lessAbs", "greater", "less"] or None
        The alternative hypothesis to use.
    lfc_null: float
        The null log fold change.
    reference_dds_ref_level: tuple or None
        The reference level of the design factors.
    clean_models: bool
        Whether to clean the models after the computation.
    logging_config_file_path: str or Path or None
        The path to the logging configuration file.
        Default is the default logging configuration file, which
        logs the content of the shared state and the adatas, but
        not the size.
    """
    # Run the tcga experiment to check convergence
    run_tcga_testing_pipe(
        DESeq2Strategy(
            design_factors=design_factors,
            ref_levels=ref_levels,
            independent_filter=independent_filter,
            cooks_filter=cooks_filter,
            refit_cooks=refit_cooks,
            continuous_factors=continuous_factors,
            contrast=contrast,
            alt_hypothesis=alt_hypothesis,
            lfc_null=lfc_null,
        ),
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        assets_directory=assets_directory,
        simulate=simulate,
        dataset_name=dataset_name,
        small_samples=small_samples,
        small_genes=small_genes,
        backend=backend,
        only_two_centers=only_two_centers,
        register_data=True,
        design_factors=design_factors,
        continuous_factors=continuous_factors,
        reference_dds_ref_level=reference_dds_ref_level,
        clean_models=clean_models,
        logging_config_file_path=logging_config_file_path,
    )
