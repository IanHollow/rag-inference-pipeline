"""Tests for scripts/create_test_docs.py."""

import os
from pathlib import Path
from unittest.mock import patch


def test_initialize_documents_skips_if_exists(tmp_path: Path) -> None:
    """Test that _initialize_documents does not recreate existing database."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    db_path = docs_dir / "documents.db"
    db_path.touch()

    with (
        patch.dict(os.environ, {"DOCUMENTS_DIR": str(docs_dir)}),
        patch("builtins.print") as mock_print,
    ):
        from scripts.create_test_docs import _initialize_documents

        _initialize_documents()
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert not any("Creating document database" in call for call in print_calls)
