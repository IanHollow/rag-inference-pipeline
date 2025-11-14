"""
Tests for pipeline enumerations and node role derivation.
"""

import pytest

from pipeline.enums import NodeRole, ServiceEndpoint, derive_node_role


class TestNodeRole:
    """Test suite for NodeRole enumeration."""

    def test_node_role_values(self) -> None:
        """Test that NodeRole enum has expected values."""
        assert NodeRole.GATEWAY.value == "gateway"
        assert NodeRole.RETRIEVAL.value == "retrieval"
        assert NodeRole.GENERATION.value == "generation"

    def test_node_role_string_comparison(self) -> None:
        """Test that NodeRole values can be compared with strings."""
        assert NodeRole.GATEWAY.value == "gateway"
        assert NodeRole.RETRIEVAL.value == "retrieval"
        assert NodeRole.GENERATION.value == "generation"


class TestServiceEndpoint:
    """Test suite for ServiceEndpoint enumeration."""

    def test_common_endpoints(self) -> None:
        """Test that common endpoints are defined."""
        assert ServiceEndpoint.QUERY.value == "/query"
        assert ServiceEndpoint.HEALTH.value == "/health"
        assert ServiceEndpoint.METRICS.value == "/metrics"

    def test_retrieval_endpoints(self) -> None:
        """Test that retrieval service endpoints are defined."""
        assert ServiceEndpoint.RETRIEVE.value == "/retrieve"

    def test_generation_endpoints(self) -> None:
        """Test that generation service endpoints are defined."""
        assert ServiceEndpoint.GENERATE.value == "/generate"


class TestDeriveNodeRole:
    """Test suite for derive_node_role function."""

    def test_derive_role_node_0(self) -> None:
        """Test that node 0 is assigned Gateway role."""
        role = derive_node_role(0)
        assert role == NodeRole.GATEWAY

    def test_derive_role_node_1(self) -> None:
        """Test that node 1 is assigned Retrieval role."""
        role = derive_node_role(1)
        assert role == NodeRole.RETRIEVAL

    def test_derive_role_node_2(self) -> None:
        """Test that node 2 is assigned Generation role."""
        role = derive_node_role(2)
        assert role == NodeRole.GENERATION

    def test_derive_role_invalid_negative(self) -> None:
        """Test that negative node numbers raise ValueError."""
        with pytest.raises(ValueError, match="Invalid node_number"):
            derive_node_role(-1)

    def test_derive_role_invalid_too_large(self) -> None:
        """Test that node numbers >= 3 raise ValueError."""
        with pytest.raises(ValueError, match="Invalid node_number"):
            derive_node_role(3)

        with pytest.raises(ValueError, match="Invalid node_number"):
            derive_node_role(100)

    def test_derive_role_comprehensive(self) -> None:
        """Test all valid node numbers in a single test."""
        expected_mapping = {
            0: NodeRole.GATEWAY,
            1: NodeRole.RETRIEVAL,
            2: NodeRole.GENERATION,
        }

        for node_num, expected_role in expected_mapping.items():
            actual_role = derive_node_role(node_num)
            assert actual_role == expected_role, (
                f"Node {node_num} should be {expected_role}, got {actual_role}"
            )
