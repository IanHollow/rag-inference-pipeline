"""Tests for scripts/sync_requirements.py"""

from unittest.mock import mock_open, patch

import pytest

from scripts.sync_requirements import sync_requirements


@pytest.fixture
def mock_pyproject_content() -> bytes:
    """Sample pyproject.toml content for testing."""
    return b"""
[project]
dependencies = [
    "torch>=2.9.1",
    "transformers>=4.57.1",
    "numpy>=2.2.6,<2.3.0",
]
"""


def test_sync_requirements_reads_pyproject(mock_pyproject_content: bytes) -> None:
    """Test that sync_requirements reads pyproject.toml correctly."""
    mock_requirements_file = mock_open()

    with (
        patch(
            "pathlib.Path.open",
            side_effect=[
                mock_open(read_data=mock_pyproject_content).return_value,
                mock_requirements_file.return_value,
            ],
        ),
        patch("builtins.print") as mock_print,
    ):
        sync_requirements()

        # Check that requirements.txt was written with correct content
        handle = mock_requirements_file()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert "torch>=2.9.1" in written_content
        assert "transformers>=4.57.1" in written_content
        assert "numpy>=2.2.6,<2.3.0" in written_content

        # Check that success message was printed
        mock_print.assert_called_once()
        assert "Synced 3 dependencies" in str(mock_print.call_args)


def test_sync_requirements_empty_dependencies() -> None:
    """Test handling of empty dependencies list."""
    mock_pyproject_content = b"[project]\ndependencies = []"
    mock_requirements_file = mock_open()

    with (
        patch(
            "pathlib.Path.open",
            side_effect=[
                mock_open(read_data=mock_pyproject_content).return_value,
                mock_requirements_file.return_value,
            ],
        ),
        patch("builtins.print") as mock_print,
    ):
        sync_requirements()

        # Should write empty file (just newline)
        handle = mock_requirements_file()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        assert written_content == "\n"

        # Should print zero dependencies
        assert "Synced 0 dependencies" in str(mock_print.call_args)
