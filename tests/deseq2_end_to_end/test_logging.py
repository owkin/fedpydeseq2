"""Module testing the logmeans computed in the DESeq2Strategy."""
import json
import os

import pytest

from .test_deseq2_pipe_utils import pipeline_to_test

COOKS_FILTER = [False, True]

DESIGN_FACTORS = ["stage", ["gender", "stage"]]


@pytest.mark.usefixtures(
    "raw_data_path",
    "processed_data_path",
    "tcga_assets_directory",
    "workflow_logging_configuration_path",
)
def test_logging_end_to_end(
    raw_data_path,
    processed_data_path,
    tcga_assets_directory,
    workflow_logging_configuration_path,
):
    """Compare FL and pooled deseq2 pipelines.

    The data is TCGA-LUAD, restricted to a small number of genes.
    This is only tested to check that the pipeline runs correctly.

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
    workflow_logging_configuration_path : Path
        The path to the logging configuration file.
    """
    pipeline_to_test(
        raw_data_path,
        processed_data_path,
        tcga_assets_directory,
        logging_config_file_path=workflow_logging_configuration_path,
    )

    # Now open the config file and get the 'generate_workflow' 'workflow_file_path' key
    # and check that the file exists
    with open(workflow_logging_configuration_path) as f:
        logging_config = json.load(f)
        workflow_file_path = logging_config["generate_workflow"]["workflow_file_path"]
        assert os.path.exists(workflow_file_path)
        # Check that there are at leat 100 lines in the file
        with open(workflow_file_path) as wf:
            assert len(wf.readlines()) >= 100
