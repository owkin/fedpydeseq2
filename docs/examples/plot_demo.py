"""FedPyDESeq2 demo on the TCGA-LUAD dataset.

This example demonstrates how to run a FedPyDESeq2 experiment on the TCGA-LUAD dataset
from a single machine, using Substra's simulated mode.

We will show how to perform a simple differential expression analysis, comparing samples
with `"Advanced"` vs `"Non-advanced"` tumoral `stage`.
"""

# %%

from pathlib import Path

import pandas as pd
from fedpydeseq2_datasets.process_and_split_data import setup_tcga_dataset
from IPython.display import display

from fedpydeseq2.fedpydeseq2_pipeline import run_fedpydeseq2_experiment

# %%
# ## Dataset setup
#
# FedPyDESeq2 assumes the data to be organized in the following structure:
#
# ```
# processed_data_path/
# ├── centers_data/
# │   └── tcga/
# │       └── {experiment_id}/
# │           ├── center_0/
# │           │   ├── counts.csv
# │           │   └── metadata.csv
# │           ├── center_1/
# │           │   ├── counts.csv
# │           │   └── metadata.csv
# │           └── ...
# └── pooled_data/
#     └── tcga/
#         └── {experiment_id}/
#             ├── counts.csv
#             └── metadata.csv
# ```
#
# In this tutorial, we have already downloaded the data in the `data/raw` directory.
#
# The `setup_tcga_dataset` function from `fedpydeseq2_datasets` will automatically
# organize the data in the `data/processed` directory.
#
# It will split the TCGA-LUAD dataset into 7 centers according to the geographical
# origin of the samples, as described in the
# [FedPyDESeq2 paper](https://www.biorxiv.org/content/10.1101/2024.12.06.627138v1).
#
# See also the [`fedpydeseq2_datasets`](https://github.com/owkin/fedpydeseq2-datasets)
# repository for more details.


dataset_name = "TCGA-LUAD"
raw_data_path = Path("data/raw").resolve()
processed_data_path = Path("data/processed").resolve()
design_factors = "stage"


setup_tcga_dataset(
    raw_data_path,
    processed_data_path,
    dataset_name=dataset_name,
    small_samples=False,
    small_genes=False,
    only_two_centers=False,
    design_factors=design_factors,
    force=True,
)

experiment_id = "TCGA-LUAD-stage"

# %%
# ## Running the experiment
#
# We can now run the experiment.
#
# [Substra](https://github.com/substra), the FL framework on which FedPyDESeq2 is built,
# supports a simulated mode which may be run locally from a single machine, which we
# will use here.
#
# Let's run our FedPyDESeq2 experiment. This may be done using the
# `run_fedpydeseq2_experiment` wrapper function, which takes the following parameters:
#
# * `n_centers=7`: Our data is distributed across 7 different medical centers
#
# * `backend="subprocess"` and `simulate=True`: We'll run the analysis locally on our
#   machine to simulate a federated setup, rather than in a real distributed environment
#
# * `register_data=True`: We'll register our dataset with Substra before analysis.
#   In the case of a real federated setup, this would be set to `False` if data was
#   already registered by Substra.
#
# * `asset_directory`: This directory should contain an opener.py file, containing an
#   Opener class, and datasets.description.md file. Here, we copy them from
#   [`fedpydeseq2_datasets/assets/tcga`](https://github.com/owkin/fedpydeseq2-datasets/tree/main/fedpydeseq2_datasets/assets/tcga) # noqa: E501
#
# * `centers_root_directory`: Where the processed data for each center is stored
#
# * `compute_plan_name`: We'll call this analysis "Example-TCGA-LUAD-pipeline"
#   in Substra
#
# * `dataset_name`: We're working with the TCGA-LUAD lung cancer dataset
#
# * `dataset_datasamples_keys_path`: Path to a YAML file containing the keys for our
#   data samples. This is only used in the case of a real (unsimulated) federated setup.
#
# * `design_factors`: This should be a list of the design factors we wish to include in
#   our analysis. Here, we're studying how "stage" (the cancer stage) affects gene
#   expression
#
# * `ref_levels`: We're setting "Non-advanced" as our baseline cancer stage
#
# * `contrast`: This should be a list of three strings, of the form
#  `["factor", "alternative_level", "baseline_level"]`. To compare gene expression
#   between "Advanced" vs "Non-advanced" stages, we set
#  `contrast=["stage", "Advanced", "Non-advanced"]`.
#
# * `refit_cooks=True`: After finding outliers using Cook's distance, we'll refit the
#   model without them for more robust results

fl_results = run_fedpydeseq2_experiment(
    n_centers=7,
    backend="subprocess",
    simulate=True,
    register_data=True,
    asset_directory=Path("assets/tcga").resolve(),
    centers_root_directory=processed_data_path
    / "centers_data"
    / "tcga"
    / experiment_id,
    compute_plan_name="Example-TCGA-LUAD-pipeline",
    dataset_name="TCGA-LUAD",
    dataset_datasamples_keys_path=Path(
        f"credentials/{experiment_id}-datasamples-keys.yaml"
    ).resolve(),
    design_factors="stage",
    ref_levels={"stage": "Non-advanced"},
    contrast=["stage", "Advanced", "Non-advanced"],
    refit_cooks=True,
)

# %%
# ## Results
# The results are then stored in a `fl_results` dictionary, which does not contain any
# individual sample information.
fl_results.keys()

# %%
# We can then extract the results for our contrast of interest, and store them in a
# pandas DataFrame.

res_df = pd.DataFrame()
res_df["LFC"] = fl_results["LFC"]["stage_Advanced_vs_Non-advanced"]
res_df["pvalue"] = fl_results["p_values"]
res_df["padj"] = fl_results["padj"]

res_df = res_df.loc[fl_results["non_zero"], :]

# %%
display(res_df)

# %%
