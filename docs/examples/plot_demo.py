"""FedPyDESeq2 demo on the TCGA-LUAD dataset."""

# %%

from pathlib import Path

import pandas as pd
from fedpydeseq2_datasets.process_and_split_data import setup_tcga_dataset
from IPython.display import display

from fedpydeseq2.fedpydeseq2_pipeline import run_fedpydeseq2_experiment

# %%
# Paths to the raw and processed data

raw_data_path = Path("data/raw").resolve()
processed_data_path = Path("data/processed").resolve()
dataset_name = "TCGA-LUAD"

# %%

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


metadata = pd.read_csv(
    processed_data_path / "pooled_data" / "tcga" / experiment_id / "metadata.csv"
)

n_centers = len(metadata.center_id.unique())

centers_root_directory = processed_data_path / "centers_data" / "tcga" / experiment_id


dataset_datasamples_keys_path = Path(
    f"credentials/{experiment_id}-datasamples-keys.yaml"
).resolve()


ref_levels = {"stage": "Non-advanced"}
contrast = ["stage", "Advanced", "Non-advanced"]


fl_results = run_fedpydeseq2_experiment(
    n_centers=n_centers,
    backend="subprocess",
    simulate=True,
    register_data=True,
    asset_directory=Path("assets/tcga").resolve(),
    centers_root_directory=centers_root_directory,
    compute_plan_name="Example-TCGA-LUAD-pipeline",
    dataset_name="TCGA-LUAD",
    dataset_datasamples_keys_path=dataset_datasamples_keys_path,
    design_factors="stage",
    ref_levels=ref_levels,
    contrast=contrast,
    refit_cooks=True,
)

# %%
fl_results.keys()

# %%
res_df = pd.DataFrame()
res_df["LFC"] = fl_results["LFC"]["stage_Advanced_vs_Non-advanced"]
res_df["pvalue"] = fl_results["p_values"]
res_df["padj"] = fl_results["padj"]

res_df = res_df.loc[fl_results["non_zero"], :]

# %%
display(res_df)

# %%
