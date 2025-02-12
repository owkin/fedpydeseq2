"""Module to test the different components for computing genewise dispersions.

In the single factor case, it is possible to perform a test on the whole block.

However, for the multifactor case, we must decompose the test into four smaller
steps that are individually unit tested, because they are sensitive to the propagation
of intermediate results.

- The first step is to compute the MoM dispersions, which is theoretically independent
of the number of design factors.
- The second step is to compute the mu_hat estimate.
- The third step is to compute the number of replicates.
- The fourth step is to compute the dispersion from mu_hat and the number of replicates.
"""
