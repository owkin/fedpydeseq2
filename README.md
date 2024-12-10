# FedPyDESeq2

This repository contains the FedDESeq2 package, a Python package for Federated
Differential Expression Analysis based on [PyDESeq2](https://github.com/owkin/pydeseq2)
which is itself a python implementation of
[DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html).

## Setup

### Package installation through PyPI

You can install the package from PyPI using the following command:

```bash
pip install fedpydeseq2
```


### Package installation for developpers

#### 0 - Clone the repository

Start by cloning this repository

```bash
git clone git@github.com:owkin/fedpydeseq2.git
```

#### 1 - Create a conda environment with python 3.10+

```
conda create -n fedpydeseq2 python=3.11 # or a version compatible
conda activate fedpydeseq2
```

#### 2 - Install `poetry`

Run

```
conda install pip
pip install poetry==1.8.2
```

and test the installation with `poetry --version`.



#### 3 - Install the package and its dependencies using `poetry`

`cd` to the root of the repository and run

```
poetry install --with linting,testing
```

#### 4 - Download the data to run the tests on

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
- An optional `assets_tcga` field with the path to the directory containing the `opener.py` file and its description (by default present in the fedpydeseq2_datasets module, so no need to specify this unless you need to modify the opener);
- An optional `processed_data` field with the path to the directory where you want to save processed data. Note that this is used
only if you want to run tests locally without reprocessing the data during each test session (test marked with the `local` marker).




#### 5 - Install `pre-commit` hooks

Still in the root of the repository, run

`pre-commit install`

You are now ready to contribute.

## CI on a self-hosted runner
Tests are run using a self-hosted runner. To add a self-hosted runner, instantiate the machine
you want to use as a runner, go to the repository settings, then to the `Actions` tab, and click on
`Add runner`. Follow the instructions to install the runner on the machine you want
to use as a self-hosted runner.

Make sure to label the self-hosted runner with the label "fedpydeseq2-self-hosted" so that
the CI workflow can find it.

### Docker CI
The docker mode is only tested manually. To test it, first run `poetry build`
in order to create a wheel in the `dist` folder. Then launch in a tmux the
following:
```
pytest -m "docker" -s
```
The `-s` option enables to print all the logs/outputs continuously. Otherwise, these
outputs appear only once the test is done. As the test takes time, it's better to
print them continuously.

## Running on a real Substra environment

### running the CP
To run a compute plan on an environment with the substra front-end, you need first to generate token in each of the
substra nodes. Then you need to duplicate
[credentials-template.yaml](fedpydeseq2/substra_utils/credentials/credentials-template.yaml)
into a new file
[credentials.yaml](fedpydeseq2/substra_utils/credentials/credentials.yaml) and fill in the
tokens. You should not need to rebuild the wheel manually by running
`poetry build` as the script will try to do it for you, but watch out for
related error message when executing the file.
