"""Module to implement the passing of the first shared state.

# TODO remove after all savings have been factored out, if not needed anymore.
"""
from substrafl.remote import remote

from fedpydeseq2.core.utils.logging import log_remote


class AggPassOnFirstSharedState:
    """Mixin to pass on the first shared state."""

    @remote
    @log_remote
    def pass_on_shared_state(self, shared_states: list[dict]) -> dict:
        """Pass on the shared state.

        This method simply returns the first shared state.

        Parameters
        ----------
        shared_states : list
            List of shared states.

        Returns
        -------
        shared_state : dict
            The shared state to be passed on.

        """
        return shared_states[0]
