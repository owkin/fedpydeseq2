import anndata as ad
import numpy as np


def get_lfc_utils_from_gene_mask_adata(
    adata: ad.AnnData,
    gene_mask: np.ndarray | None,
    disp_param_name: str,
    beta: np.ndarray | None = None,
    lfc_param_name: str | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the necessary data for LFC computations from the local adata and genes.

    Parameters
    ----------
    adata : ad.AnnData
        The local AnnData object.

    gene_mask : np.ndarray or None
        The mask of genes to use for the IRLS algorithm.
        This mask identifies the genes in the non_zero_gene_names.
        If None, all non zero genes are used.

    disp_param_name : str
        The name of the dispersion parameter in the adata.varm.

    beta : Optional[np.ndarray]
        The log fold change values, of shape (n_non_zero_genes,).

    lfc_param_name: Optional[str]
        The name of the lfc parameter in the adata.varm.
        Is incompatible with beta.

    Returns
    -------
    gene_names : list[str]
        The names of the genes to use for the IRLS algorithm.
    design_matrix : np.ndarray
        The design matrix.
    size_factors : np.ndarray
        The size factors.
    counts : np.ndarray
        The count matrix from the local adata.
    dispersions : np.ndarray
        The dispersions from the local adata.
    beta_on_mask : np.ndarray
        The log fold change values on the mask.
    """
    # Check that one of beta or lfc_param_name is not None
    assert (beta is not None) ^ (
        lfc_param_name is not None
    ), "One of beta or lfc_param_name must be not None"

    # Get non zero genes
    non_zero_genes_names = adata.var_names[adata.varm["non_zero"]]

    # Get the irls genes
    if gene_mask is None:
        gene_names = non_zero_genes_names
    else:
        gene_names = non_zero_genes_names[gene_mask]

    # Get beta
    if lfc_param_name is not None:
        beta_on_mask = adata[:, gene_names].varm[lfc_param_name].to_numpy()
    elif gene_mask is not None:
        assert beta is not None  # for mypy
        beta_on_mask = beta[gene_mask]
    else:
        assert beta is not None  # for mypy
        beta_on_mask = beta.copy()

    design_matrix = adata.obsm["design_matrix"].values
    size_factors = adata.obsm["size_factors"]
    counts = adata[:, gene_names].X
    dispersions = adata[:, gene_names].varm[disp_param_name]

    return gene_names, design_matrix, size_factors, counts, dispersions, beta_on_mask
