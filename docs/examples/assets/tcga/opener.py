import pathlib

import anndata as ad
import pandas as pd
import substratools as tools
from pydeseq2.utils import load_example_data


class TCGAOpener(tools.Opener):
    """Opener class for TCGA RNA-seq datasets.

    Creates an AnnData object from a path containing a counts_data.csv and a
    metadata.csv.
    """

    def fake_data(self, n_samples=None):
        """Create a fake AnnData object for testing purposes.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate. If None, generate 100 samples.

        Returns
        -------
        AnnData
            An AnnData object with fake counts and metadata.
        """
        N_SAMPLES = n_samples if n_samples and n_samples <= 100 else 100
        fake_counts = load_example_data(modality="raw_counts").iloc[:N_SAMPLES]
        fake_metadata = load_example_data(modality="metadata").iloc[:N_SAMPLES]
        return ad.AnnData(X=fake_counts, obs=fake_metadata)

    def get_data(self, folders):
        """Open the TCGA dataset.

        Parameters
        ----------
        folders : list
            list of paths to the dataset folders, whose first element should contain a
            counts_data.csv and a metadata.csv file.

        Returns
        -------
        AnnData
            An AnnData object containing the counts and metadata loaded for the FL pipe.
        """
        # get .csv files
        data_path = pathlib.Path(folders[0]).resolve()
        counts_data = pd.read_csv(data_path / "counts_data.csv", index_col=0)
        metadata = pd.read_csv(data_path / "metadata.csv", index_col=0)
        center_id = metadata["center_id"].iloc[0]
        # We assume that the center id is not present in the counts data, if it is
        # present, we raise an error (it should have been removed in an earlier
        # step)
        if "center_id" in counts_data.columns:
            raise ValueError("center_id column found in counts data")
        metadata.drop(columns=["center_id"], inplace=True)
        # Build an Anndata object
        adata = ad.AnnData(X=counts_data, obs=metadata)
        # Add the center id to be accessible within the local states
        adata.uns["center_id"] = center_id
        return adata
