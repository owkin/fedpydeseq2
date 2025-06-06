# FedPyDESeq2

This repository contains the FedDESeq2 package, a Python package for Federated
Differential Expression Analysis based on [PyDESeq2](https://github.com/owkin/pydeseq2)
which is itself a python implementation of
[DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html).

## Documentation

For detailed documentation, please visit:
- [Online Documentation](https://owkin.github.io/fedpydeseq2/) - Complete API reference and usage examples
- [Quick Start Guide](https://owkin.github.io/fedpydeseq2/generated/gallery/) - Get started with basic usage
- [API Reference](https://owkin.github.io/fedpydeseq2/api/index.html) - Detailed API documentation


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
- An optional `assets_tcga` field with the path to the directory containing the `opener.py` file and its description (by default present in the fedpydeseq2_datasets module, so no need to specify this unless you need to modify the opener).
- An optional `processed_data` field with the path to the directory where you want to save processed data. This is used
if you want to run tests locally without reprocessing the data during each test session. Otherwise, the processed data will be saved in a temporary file during each test session.
- An optional `default_logging_config` field with the path to the logging configuration used by default in tests. For more details on how logging works, please refer to the [README](logging/README.md) in the logging folder.
- An optional `workflow_logging_config` field with the path to the logging configuration used in the logging tests.





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

### Running the compute plan
To run a compute plan on an environment with the substra front-end, you need first to generate token in each of the
substra nodes. Then you need to duplicate
[credentials-template.yaml](fedpydeseq2/substra_utils/credentials/credentials-template.yaml)
into a new file
[credentials.yaml](fedpydeseq2/substra_utils/credentials/credentials.yaml) and fill in the
tokens. You should not need to rebuild the wheel manually by running
`poetry build` as the script will try to do it for you, but watch out for
related error message when executing the file.

## Citing this work

```
@article{muzellec2024fedpydeseq2,
  title={FedPyDESeq2: a federated framework for bulk RNA-seq differential expression analysis},
  author={Muzellec, Boris and Marteau-Ferey, Ulysse and Marchand, Tanguy},
  journal={bioRxiv},
  pages={2024--12},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## References

[1] Love, M. I., Huber, W., & Anders, S. (2014). "Moderated estimation of fold
        change and dispersion for RNA-seq data with DESeq2." Genome biology, 15(12), 1-21.
        <https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8>

[2] Muzellec, B., Teleńczuk, M., Cabeli, V., & Andreux, M. (2023). "PyDESeq2: a python package
        for bulk RNA-seq differential expression analysis." Bioinformatics, 39(9), btad547.
        <https://academic.oup.com/bioinformatics/article/39/9/btad547/7260507>

## License

FedPyDESeq2 is released under an [MIT license](https://github.com/owkin/FedPyDESeq2/blob/main/LICENSE](https://github.com/owkin/fedpydeseq2/blob/main/LICENSE)).
