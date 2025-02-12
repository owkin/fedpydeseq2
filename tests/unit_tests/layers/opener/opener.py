import pathlib

import anndata as ad
import pandas as pd
import substratools as tools


class SimpleOpener(tools.Opener):
    """Opener class for testing purposes.

    Creates an AnnData object from a path containing a counts_data.csv.
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
        pass

    def get_data(self, folders):
        """Get the data.

        Parameters
        ----------
        folders : list
            List of paths to the dataset folders, whose first element should contain a
            counts_data.csv and a metadata.csv file.

        Returns
        -------
        AnnData
            An AnnData object containing the counts and metadata loaded for the FL pipe.
        """
        data_path = pathlib.Path(folders[0]).resolve()
        counts_data = pd.read_csv(data_path / "counts_data.csv", index_col=0)

        adata = ad.AnnData(X=counts_data)
        return adata
