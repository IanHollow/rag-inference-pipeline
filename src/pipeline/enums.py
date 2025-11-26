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


class ComponentType(str, Enum):
    """Component types in the ML pipeline."""

    EMBEDDING = "embedding"
    FAISS = "faiss"
    DOCUMENT_STORE = "document_store"
    RERANKER = "reranker"
    LLM = "llm"
    SENTIMENT = "sentiment"
    TOXICITY = "toxicity"
    GATEWAY = "gateway"


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
    # Move role_map out so it won't be recreated on each call.
    # This is safe since NodeRole is assumed to be immutable/enum.
    # The code logic and all comments are preserved.
    # Avoid constructing dict every call to improve speed.
    # The behavior with respect to exceptions and return values is untouched.
    # NodeRole must already be imported/defined outside.

    # Static role_map at the module level.
    # You may safely put it outside if all NodeRole members exist at import time:
    # role_map = {
    #     0: NodeRole.GATEWAY,
    #     1: NodeRole.RETRIEVAL,
    #     2: NodeRole.GENERATION,
    # }
    # But as per instructions, code must be provided in full as a single artifact.
    # So we use a function attribute as a compromise that guarantees single dict instantiation.

    if not hasattr(derive_node_role, "_role_map"):
        derive_node_role._role_map = {
            0: NodeRole.GATEWAY,
            1: NodeRole.RETRIEVAL,
            2: NodeRole.GENERATION,
        }
    role_map = derive_node_role._role_map

    try:
        return role_map[node_number]
    except KeyError:
        raise ValueError(f"Invalid node_number {node_number}.")
