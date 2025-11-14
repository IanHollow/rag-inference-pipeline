"""
Enumerations and constants for the distributed ML pipeline.
"""

from enum import Enum


class NodeRole(str, Enum):
    """Role assignment for each node in the cluster."""

    GATEWAY = "gateway"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"


class ServiceEndpoint(str, Enum):
    """API endpoints exposed by services."""

    # Gateway endpoints
    QUERY = "/query"
    HEALTH = "/health"
    METRICS = "/metrics"

    # Retrieval service endpoints
    RETRIEVE = "/retrieve"

    # Generation service endpoints
    GENERATE = "/generate"


def derive_node_role(node_number: int) -> NodeRole:
    """
    Derive the role of a node based on its number.
    Args:
        node_number: The node number (0-based index)

    Returns:
        NodeRole: The role assigned to this node

    Raises:
        ValueError: If node_number is invalid
    """
    role_map = {
        0: NodeRole.GATEWAY,
        1: NodeRole.RETRIEVAL,
        2: NodeRole.GENERATION,
    }

    if node_number not in role_map:
        raise ValueError(f"Invalid node_number {node_number}.")

    return role_map[node_number]
