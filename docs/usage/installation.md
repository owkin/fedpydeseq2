# Installation

## 1 - Create a conda environment with python 3.10+

```
conda create -n fedpydeseq2 python=3.10 # or a higher python version
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
poetry install
```

If you want to install to contribute, please run

```
poetry install --with linting,testing
```

If you wish to manually create the documentation, please run

```
poetry install --with docs
```

If you want to run the paper experiments, please run

```
poetry install --with experiments
```
