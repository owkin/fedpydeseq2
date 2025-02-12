# Package installation for developpers

## 0 - Clone the repository

Start by cloning this repository

```bash
git clone git@github.com:owkin/fedpydeseq2.git
```

## 1 - Create a conda environment with python 3.10+

```
conda create -n fedpydeseq2 python=3.11 # or a version compatible
conda activate fedpydeseq2
```

## 2 - Install `poetry`

Run

```
conda install pip
pip install poetry==1.8.2
```

and test the installation with `poetry --version`.



## 3 - Install the package and its dependencies using `poetry`

`cd` to the root of the repository and run

```
poetry install --with linting,testing
```

## 4 - Download the data to run the tests on

To download the data, `cd` to the root of the repository run this command.

```bash
fedpydeseq2-download-data --raw_data_output_path data/raw
```

This way, you create a `data/raw` subdirectory in the directory containing all the necessary data. If you want to modify
the location of this raw data, you can in the following way. Run this command instead:

```bash
fedpydeseq2-download-data --raw_data_output_path MY_RAW_PATH
```


And create a file in the `tests` directory named `paths.json` containing
- A `raw_data` field with the path to the raw data `MY_RAW_PATH`
- An optional `assets_tcga` field with the path to the directory containing the `opener.py` file and its description (by default present in the fedpydeseq2_datasets module, so no need to specify this unless you need to modify the opener).
- An optional `processed_data` field with the path to the directory where you want to save processed data. This is used
if you want to run tests locally without reprocessing the data during each test session. Otherwise, the processed data will be saved in a temporary file during each test session.
- An optional `default_logging_config` field with the path to the logging configuration used by default in tests. For more details on how logging works, please refer to this [dedicated section](logging.md)
- An optional `workflow_logging_config` field with the path to the logging configuration used in the logging tests.
