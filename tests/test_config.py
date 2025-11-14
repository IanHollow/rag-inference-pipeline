"""
Tests for pipeline configuration and environment variable parsing.

These tests verify that the PipelineSettings correctly derives node roles,
parses environment variables, and validates constraints per the project spec.
"""

import os
from unittest.mock import patch

from pydantic import ValidationError
import pytest

from pipeline.config import PipelineSettings, get_settings
from pipeline.enums import NodeRole


class TestPipelineSettings:
    """Test suite for PipelineSettings configuration class."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        settings = PipelineSettings()
        assert settings.total_nodes == 3
        assert settings.node_number == 0
        assert settings.node_0_ip == "localhost:8000"
        assert settings.node_1_ip == "localhost:8001"
        assert settings.node_2_ip == "localhost:8002"
        assert settings.faiss_index_path == "faiss_index.bin"
        assert settings.documents_dir == "documents/"
        assert settings.only_cpu is True

    def test_env_var_parsing(self) -> None:
        """Test that environment variables are correctly parsed."""
        env_vars = {
            "TOTAL_NODES": "3",
            "NODE_NUMBER": "1",
            "NODE_0_IP": "192.168.1.100:8000",
            "NODE_1_IP": "192.168.1.101:8001",
            "NODE_2_IP": "192.168.1.102:8002",
            "FAISS_INDEX_PATH": "/data/faiss.index",
            "DOCUMENTS_DIR": "/data/documents/",
            "ONLY_CPU": "false",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = PipelineSettings()
            assert settings.total_nodes == 3
            assert settings.node_number == 1
            assert settings.node_0_ip == "192.168.1.100:8000"
            assert settings.node_1_ip == "192.168.1.101:8001"
            assert settings.node_2_ip == "192.168.1.102:8002"
            assert settings.faiss_index_path == "/data/faiss.index"
            assert settings.documents_dir == "/data/documents/"
            assert settings.only_cpu is False

    def test_node_role_derivation_node_0(self) -> None:
        """Test that Node 0 is assigned Gateway role."""
        env_vars = {
            "TOTAL_NODES": "3",
            "NODE_NUMBER": "0",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = PipelineSettings()
            assert settings.role == NodeRole.GATEWAY

    def test_node_role_derivation_node_1(self) -> None:
        """Test that Node 1 is assigned Retrieval role."""
        env_vars = {
            "TOTAL_NODES": "3",
            "NODE_NUMBER": "1",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = PipelineSettings()
            assert settings.role == NodeRole.RETRIEVAL

    def test_node_role_derivation_node_2(self) -> None:
        """Test that Node 2 is assigned Generation role."""
        env_vars = {
            "TOTAL_NODES": "3",
            "NODE_NUMBER": "2",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = PipelineSettings()
            assert settings.role == NodeRole.GENERATION

    def test_invalid_total_nodes(self) -> None:
        """Test that validation fails if TOTAL_NODES is not 3."""
        env_vars = {
            "TOTAL_NODES": "2",
            "NODE_NUMBER": "0",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(ValidationError) as exc_info:
                PipelineSettings()

            error = exc_info.value.errors()[0]
            # Pydantic uses the alias (env var name) in the error location
            assert "TOTAL_NODES" in error["loc"]
            assert "must be 3" in error["msg"].lower()

    def test_invalid_node_number(self) -> None:
        """Test that validation fails if NODE_NUMBER is not 0, 1, or 2."""
        env_vars = {
            "TOTAL_NODES": "3",
            "NODE_NUMBER": "3",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            with pytest.raises(ValidationError) as exc_info:
                PipelineSettings()

            error = exc_info.value.errors()[0]
            # Pydantic uses the alias (env var name) in the error location
            assert "NODE_NUMBER" in error["loc"]

    def test_node_ips_property(self) -> None:
        """Test that node_ips property returns correct mapping."""
        env_vars = {
            "NODE_0_IP": "host0:8000",
            "NODE_1_IP": "host1:8001",
            "NODE_2_IP": "host2:8002",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = PipelineSettings()
            node_ips = settings.node_ips

            assert node_ips[0] == "host0:8000"
            assert node_ips[1] == "host1:8001"
            assert node_ips[2] == "host2:8002"

    def test_service_urls(self) -> None:
        """Test that service URL properties are correctly formatted."""
        env_vars = {
            "NODE_0_IP": "gateway.local:9000",
            "NODE_1_IP": "retrieval.local:9001",
            "NODE_2_IP": "generation.local:9002",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = PipelineSettings()

            assert settings.gateway_url == "http://gateway.local:9000"
            assert settings.retrieval_url == "http://retrieval.local:9001"
            assert settings.generation_url == "http://generation.local:9002"

    def test_listen_host_and_port_node_0(self) -> None:
        """Test that Node 0 derives correct listen host and port."""
        env_vars = {
            "TOTAL_NODES": "3",
            "NODE_NUMBER": "0",
            "NODE_0_IP": "10.0.0.1:7000",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = PipelineSettings()
            assert settings.listen_host == "10.0.0.1"
            assert settings.listen_port == 7000

    def test_listen_host_and_port_node_1(self) -> None:
        """Test that Node 1 derives correct listen host and port."""
        env_vars = {
            "TOTAL_NODES": "3",
            "NODE_NUMBER": "1",
            "NODE_1_IP": "10.0.0.2:7001",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = PipelineSettings()
            assert settings.listen_host == "10.0.0.2"
            assert settings.listen_port == 7001

    def test_listen_host_default_port(self) -> None:
        """Test that listen_port defaults to 8000 if not in IP string."""
        env_vars = {
            "TOTAL_NODES": "3",
            "NODE_NUMBER": "0",
            "NODE_0_IP": "localhost",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = PipelineSettings()
            assert settings.listen_host == "localhost"
            assert settings.listen_port == 8000

    def test_pipeline_constants(self) -> None:
        """Test that pipeline constants match project spec."""
        settings = PipelineSettings()

        # Per project spec requirements
        assert settings.faiss_dim == 768
        assert settings.max_tokens == 128
        assert settings.retrieval_k == 10
        assert settings.truncate_length == 512

    def test_batching_configuration(self) -> None:
        """Test that batching settings have correct defaults."""
        # Note: These don't have environment variable aliases, so we test defaults
        settings = PipelineSettings()

        assert settings.gateway_batch_size == 4
        assert settings.gateway_batch_timeout_ms == 100
        assert settings.retrieval_batch_size == 8
        assert settings.generation_batch_size == 4

    def test_log_level_validation_valid(self) -> None:
        """Test that valid log levels are accepted."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            # Use clear=True to avoid pollution from other tests or the environment
            with patch.dict(os.environ, {"LOG_LEVEL": level}, clear=True):
                settings = PipelineSettings()
                assert settings.log_level == level

    def test_log_level_validation_case_insensitive(self) -> None:
        """Test that log level is case-insensitive."""
        with patch.dict(os.environ, {"LOG_LEVEL": "debug"}, clear=True):
            settings = PipelineSettings()
            assert settings.log_level == "DEBUG"

    def test_log_level_validation_invalid(self) -> None:
        """Test that invalid log levels raise validation error."""
        # Invalid log levels should just be uppercased, not raise errors
        # Let's change this test to ensure validation happens correctly
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                PipelineSettings()

            error = exc_info.value.errors()[0]
            assert "LOG_LEVEL" in str(error["loc"]) or "log_level" in str(error["loc"])


class TestGetSettings:
    """Test suite for the get_settings() singleton function."""

    def test_singleton_behavior(self) -> None:
        """Test that get_settings() returns the same instance."""
        # Note: This test may be affected by other tests if they call get_settings()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_settings_cached(self) -> None:
        """Test that settings are loaded once and cached."""
        settings = get_settings()
        assert isinstance(settings, PipelineSettings)


class TestNodeRoleAssignment:
    """Test suite verifying the complete node role assignment flow."""

    def test_three_node_cluster_assignment(self) -> None:
        """Test that a 3-node cluster assigns roles correctly."""
        # Simulate 3 nodes with different NODE_NUMBER values
        test_cases = [
            (0, NodeRole.GATEWAY, "localhost:8000"),
            (1, NodeRole.RETRIEVAL, "localhost:8001"),
            (2, NodeRole.GENERATION, "localhost:8002"),
        ]

        for node_num, expected_role, expected_ip in test_cases:
            env_vars = {
                "TOTAL_NODES": "3",
                "NODE_NUMBER": str(node_num),
                f"NODE_{node_num}_IP": expected_ip,
            }

            with patch.dict(os.environ, env_vars, clear=False):
                settings = PipelineSettings()
                assert settings.node_number == node_num
                assert settings.role == expected_role
                assert settings.node_ips[node_num] == expected_ip
