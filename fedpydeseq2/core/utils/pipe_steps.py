from collections.abc import Callable

from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.nodes.references.shared_state import SharedStateRef


def local_step(
    local_method: Callable,
    train_data_nodes: list[TrainDataNode],
    output_local_states: dict[str, LocalStateRef],
    round_idx: int,
    input_local_states: dict[str, LocalStateRef] | None = None,
    input_shared_state: SharedStateRef | None = None,
    aggregation_id: str | None = None,
    description: str = "",
    clean_models: bool = True,
    method_params: dict | None = None,
) -> tuple[dict[str, LocalStateRef], list[SharedStateRef], int]:
    """Local step of the federated learning strategy.

    Used as a wrapper to execute a local method on the data of each organization.

    Parameters
    ----------
    local_method : Callable
        Method to be executed on the local data.
    train_data_nodes : TrainDataNode
        List of TrainDataNode.
    output_local_states : dict
        Dictionary of local states to be updated.
    round_idx : int
        Round index.
    input_local_states : dict, optional
        Dictionary of local states to be used as input.
    input_shared_state : SharedStateRef, optional
        Shared state to be used as input.
    aggregation_id : str, optional
        Aggregation node id.
    description : str
        Description of the algorithm.
    clean_models : bool
        Whether to clean the models after the computation.
    method_params : dict, optional
        Optional keyword arguments to be passed to the local method.

    Returns
    -------
    output_local_states : dict
        Local states containing the results of the local method,
        to keep within the training nodes.
    output_shared_states : list
        Shared states containing the results of the local method,
         to be sent to the aggregation node.
    round_idx : int
        Round index incremented by 1
    """
    output_shared_states = []
    method_params = method_params or {}

    for node in train_data_nodes:
        next_local_state, next_shared_state = node.update_states(
            local_method(
                node.data_sample_keys,
                shared_state=input_shared_state,
                _algo_name=description,
                **method_params,
            ),
            local_state=(
                input_local_states[node.organization_id] if input_local_states else None
            ),
            round_idx=round_idx,
            authorized_ids={node.organization_id},
            aggregation_id=aggregation_id,
            clean_models=clean_models,
        )

        output_local_states[node.organization_id] = next_local_state
        output_shared_states.append(next_shared_state)

    round_idx += 1
    return output_local_states, output_shared_states, round_idx


def aggregation_step(
    aggregation_method: Callable,
    train_data_nodes: list[TrainDataNode],
    aggregation_node: AggregationNode,
    input_shared_states: list[SharedStateRef],
    round_idx: int,
    description: str = "",
    clean_models: bool = True,
    method_params: dict | None = None,
) -> tuple[SharedStateRef, int]:
    """Perform an aggregation step of the federated learning strategy.

    Used as a wrapper to execute an aggregation method on the data of each organization.

    Parameters
    ----------
    aggregation_method : Callable
        Method to be executed on the shared states.
    train_data_nodes : list
        List of TrainDataNode.
    aggregation_node : AggregationNode
        Aggregation node.
    input_shared_states : list
        List of shared states to be aggregated.
    round_idx : int
        Round index.
    description:  str
        Description of the algorithm.
    clean_models : bool
        Whether to clean the models after the computation.
    method_params : dict, optional
        Optional keyword arguments to be passed to the aggregation method.

    Returns
    -------
    SharedStateRef
        A shared state containing the results of the aggregation.
    round_idx : int
        Round index incremented by 1
    """
    method_params = method_params or {}
    share_state = aggregation_node.update_states(
        aggregation_method(
            shared_states=input_shared_states,
            _algo_name=description,
            **method_params,
        ),
        round_idx=round_idx,
        authorized_ids={
            train_data_node.organization_id for train_data_node in train_data_nodes
        },
        clean_models=clean_models,
    )
    round_idx += 1
    return share_state, round_idx
