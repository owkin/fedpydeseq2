"""Module to implement the passing of the first shared state.

# TODO remove after all savings have been factored out, if not needed anymore.
"""

from substrafl.remote import remote

from fedpydeseq2.core.utils.logging import log_remote


class AggPassOnResults:
    """Mixin to pass on the first shared state."""

    results: dict | None

    @remote
    @log_remote
    def pass_on_results(self, shared_states: list[dict]) -> dict:
        """Pass on the shared state.

        This method simply returns the first shared state.

        Parameters
        ----------
        shared_states : list
            List of shared states.

        Returns
        -------
        dict : The first shared state.
        """
        results = shared_states[0]
        # This is an ugly way to save the results for the simulation mode.
        # In simulation mode, we will look at the results attribute of the class
        # to get the results.
        # In the real mode, we will download the last shared state.
        self.results = results
        return results
