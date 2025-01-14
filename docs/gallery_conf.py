from pathlib import Path

import fedpydeseq2_datasets
from fedpydeseq2_datasets.download_data.download_data import download_data
from mkdocs_gallery.gen_gallery import DefaultResetArgv

config_path = Path("docs/examples/config/config_luad.yaml").resolve()
raw_data_path = Path("docs/examples/data/raw").resolve()
processed_data_path = Path("docs/examples/data/processed").resolve()

if (raw_data_path / "tcga" / "LUAD").exists() and any(
    (raw_data_path / "tcga" / "LUAD").iterdir()
):
    print(f"Data already exists in {raw_data_path}, skipping download")
else:
    print(f"Downloading data to {raw_data_path}")
    download_data(
        config_path=config_path,
        download_data_directory=Path(
            fedpydeseq2_datasets.download_data.download_data.__file__
        ).parent.resolve(),
        raw_data_output_path=raw_data_path,
        snakemake_env_name="snakemake_env",
        conda_activate_path=None,
    )


conf = {
    "reset_argv": DefaultResetArgv(),
}
