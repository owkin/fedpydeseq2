# Contributing

## Setup

### 1- Installing the repository with the development dependencies

As in the [installation page](./installation.md), start by creating a conda environment with
the right dependencies.

1. Create a conda environment
```
conda create -n fedpydeseq2 python=3.10 # or a higher python version
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
