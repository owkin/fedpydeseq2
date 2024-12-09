# Fed-PyDESeq2 documentation

This package is a federated python implementation of the
[DESeq2](https://bioconductor.org/packages/release/bioc/html/DESeq2.html) method [@love2014moderated]
for differential expression analysis (DEA) with bulk RNA-seq data, originally in R.
This federated implementation is based on [Substra](https://docs.substra.org/en/stable/), an open source federated
learning software.


Note that this package is actually based on (and benchmarked against) [PyDESeq2](https://github.com/owkin/PyDESeq2/tree/main) [@muzellec2022pydeseq2], which is a python re-implementation of DESeq2.


Currently, available features broadly correspond to the default settings of DESeq2 (v1.34.0) for single-factor and
multi-factor analysis (with categorical or continuous factors) using Wald tests, without the LFC shrinkage step.




## Citing this work

## License
