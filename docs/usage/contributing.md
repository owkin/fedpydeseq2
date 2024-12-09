# Contributing

## Setup

### 1- Installing the repository with the development dependencies

As in the [installation page](./installation.md), start by creating a conda environment with
the right dependencies.

1. Create a conda environment
```
conda create -n fedpydeseq2 python=3.9 # or a higher python version
conda activate fedpydeseq2
```

2. Add poetry
```
conda install pip
pip install poetry==1.8.2
```

and test the installation with `poetry --version`.

3. Install the package with `testing,linting,docs` dependencies.

`cd` to the root of the repository and run

```
poetry install --with linting,testing,docs
```

### 2 - Install `pre-commit` hooks

Still in the root of the repository, run

`pre-commit install`

You are now ready to contribute.

## Details on the CI

### Modifications on the raw data

If there is a change in the raw data used, as we cannot directly download the data on abstra,
one must follow the following steps.

- Download the data locally (as explained [here](../datasets/data_download.md))
- scp the data to the correct abstra folder (`/home/owkin/project/data/raw/tcga` for ex)
- Remove the `processed` folder in the data folder on abstra

### CI on abstra using a self-hosted runner
Tests are run on abstra using a self-hosted runner. To add a self-hosted runner
on abstra, go to the repository settings, then to the `Actions` tab, and click on
`Add runner`. Follow the instructions to install the runner on the machine you want
to use as a self-hosted runner.

Make sure to label the self-hosted runner with the label "abstra-fedomics" so that
the CI workflow can find it.

### Docker CI
The docker mode is only tested manually on abstra. To test it, first run `poetry build`
in order to create a wheel in the `dist` folder. Then launch in a tmux in abstra the
following:
```
pytest -m "docker" -s
```
The `-s` option enables to print all the logs/outputs continuously. Otherwise, these
outputs appear only once the test is done. As the test takes time, it's better to
print them continuously.

### Remote execution CI

If for some reason we need to regenerate the remote execution workflow, here are the
necessary steps to do so.

Start by running the minimal fedpydeseq2 version. Make sure
you remove the `experiments/credentials/<experiment_id>-datasampes-keys.yaml` file.
In the branch of you choice, run the following
command (note that you have to have defined a `credentials.yaml` file in the `experiments/credentials`)

```bash
poetry build
poetry run python experiments/remote_tcga_pipe.py --small_cp --compute_plan_name MinimalFedPyDESeq2_TCGA-LUAD-stage --register_data
```

This will generate your `<experiment_id>-datasampes-keys.yaml` file.

Now that you have both you `credentials.yaml` and your `<experiment_id>-datasampes-keys.yaml` files,
you must encrypt them, as so (if you `cd experiments/credentials`):

```bash
gpg --symmetric --cipher-algo AES256 credentials.yaml
gpg --symmetric --cipher-algo AES256 TCGA-LUAD-stage--two-centers-datasamples-keys.yaml
```

For each encryption, you will be asked to enter a passphrase. Use the same for both files
Add this passphrase as the `CREDENTIALS_PWD` secret in the repository settings.

Then, move the encrypted files to the `experiments/credentials` folder in `.ci` folder.

You can test the workflow by running

```bash
poetry build
poetry run python experiments/remote_tcga_pipe.py --small_cp --compute_plan_name MinimalFedPyDESeq2_TCGA-LUAD-stage --compute_plan_name_suffix $GITHUB_SHA --cp_id_path .ci/ci_cp_id.yaml --n_centers 2
```

to make sure that the data has been properly registered.

The workflow should now be ready to run, and must be tested before merging the PR.
