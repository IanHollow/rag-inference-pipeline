"""
Document store management for retrieval service.

Manages SQLite connections and document fetching with thread-local storage.
"""

import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, cast

from ..config import PipelineSettings
from ..utils.cache import CompressedLRUCache

logger = logging.getLogger(__name__)


class Document:
    """Represents a document with metadata."""

    def __init__(
        self,
        doc_id: int,
        title: str,
        content: str,
        category: str | None = None,
    ) -> None:
        """
        Initialize a document.

        Args:
            doc_id: Unique document identifier
            title: Document title
            content: Document content
            category: Optional document category
        """
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.category = category

    def to_dict(self) -> dict[str, str | int]:
        """Convert document to dictionary representation."""
        result: dict[str, str | int] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
        }
        if self.category is not None:
            result["category"] = self.category
        return result

    def truncate(self, max_length: int) -> "Document":
        """
        Create a truncated copy of the document.

        Args:
            max_length: Maximum length for title and content fields

        Returns:
            New Document instance with truncated fields
        """
        # Avoid double slicing: fast-path when full-length string or when already empty
        title = self.title
        content = self.content

        # Instead of checking self.title/content on every call, check once
        title_trunc = title[:max_length] if title else ""
        content_trunc = content[:max_length] if content else ""

        # Use positional args to save kwarg parsing overhead (tiny win), but preserve arg order
        # (No behavior change, preserves signature, defensively robust)
        return Document(
            self.doc_id,
            title_trunc,
            content_trunc,
            self.category,
        )


class DocumentStore:
    """
    Manages SQLite document database with thread-local connections.

    Each worker thread gets its own connection to avoid threading issues.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the document store.

        Args:
            settings: Pipeline configuration containing documents directory
        """
        self.settings = settings
        self.db_path = Path(settings.documents_dir) / "documents.db"
        self._local = threading.local()
        self._lock = threading.Lock()
        self._memory_db_uri = "file:documents_cache?mode=memory&cache=shared"
        self._shared_memory_conn: sqlite3.Connection | None = None

        # Validate database exists
        if not self.db_path.exists():
            raise FileNotFoundError(f"Document database not found at {self.db_path}")

        # Initialize cache
        self.cache = CompressedLRUCache[int, dict[str, str | int]](
            capacity=settings.document_cache_capacity,
            ttl=settings.cache_max_ttl,
            name="document_cache",
        )

        logger.info("Initialized DocumentStore with database at %s", self.db_path)
        self._initialize_in_memory_database()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get thread-local SQLite connection.

        Creates a new connection for each thread on first access.

        Returns:
            Thread-local SQLite connection
        """
        if not hasattr(self._local, "conn"):
            logger.debug("Creating new SQLite connection for thread %s", threading.get_ident())
            if self._shared_memory_conn is not None:
                conn = sqlite3.connect(self._memory_db_uri, uri=True, check_same_thread=False)
            else:
                conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn = conn
            # Enable row factory for dict-like access
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _prepare_doc_id_temp_table(self, cursor: sqlite3.Cursor) -> None:
        """
        Ensure a temp table exists for safely binding doc_ids.

        Temp tables are connection-scoped, matching the thread-local connection.
        """
        if not getattr(self._local, "doc_id_table_initialized", False):
            cursor.execute(
                "CREATE TEMP TABLE IF NOT EXISTS temp_doc_ids(doc_id INTEGER PRIMARY KEY)"
            )
            self._local.doc_id_table_initialized = True
        cursor.execute("DELETE FROM temp_doc_ids")

    def _initialize_in_memory_database(self) -> None:
        """
        Copy the on-disk SQLite database into a shared in-memory instance.

        This keeps the entire document table resident in RAM for fast lookups.
        """
        with self._lock:
            if self._shared_memory_conn is not None:
                return

            logger.info("Copying %s into shared in-memory SQLite", self.db_path)
            disk_conn = sqlite3.connect(str(self.db_path))
            disk_conn.row_factory = sqlite3.Row
            memory_conn = sqlite3.connect(self._memory_db_uri, uri=True, check_same_thread=False)

            try:
                disk_conn.backup(memory_conn)
                self._shared_memory_conn = memory_conn
                size_mb = self.db_path.stat().st_size / (1024 * 1024)
                logger.info("Document DB copied into RAM (%.2f MB)", size_mb)
            except sqlite3.Error as e:
                logger.warning(
                    "Failed to copy document DB into memory, falling back to on-disk mode: %s",
                    e,
                )
                memory_conn.close()
                self._shared_memory_conn = None
            finally:
                disk_conn.close()

    def fetch_documents(self, doc_ids: list[int]) -> list[Document]:
        """
        Fetch documents by their IDs.

        Args:
            doc_ids: List of document IDs to fetch

        Returns:
            List of Document objects (ordered by input doc_ids)
        """
        if not doc_ids:
            return []

        # Check cache first
        cached_docs: dict[int, Document] = {}
        missing_ids: list[int] = []

        disable_cache = getattr(self.settings, "disable_cache_for_profiling", True)

        if disable_cache:
            missing_ids = list(doc_ids)
        else:
            with self._lock:
                for doc_id in doc_ids:
                    cached = self.cache.get(doc_id)
                    if cached:
                        try:
                            cached_docs[doc_id] = Document(**cast("dict[str, Any]", cached))
                        except Exception:
                            missing_ids.append(doc_id)
                    else:
                        missing_ids.append(doc_id)

        fetched_docs = []
        if missing_ids:
            conn = self._get_connection()
            cursor = conn.cursor()

            self._prepare_doc_id_temp_table(cursor)
            cursor.executemany(
                "INSERT OR IGNORE INTO temp_doc_ids(doc_id) VALUES (?)",
                ((doc_id,) for doc_id in missing_ids),
            )

            try:
                cursor.execute(
                    "SELECT doc_id, title, content, category "
                    "FROM documents "
                    "WHERE doc_id IN (SELECT doc_id FROM temp_doc_ids)"
                )
                rows = cursor.fetchall()

                for row in rows:
                    doc = Document(
                        doc_id=row["doc_id"],
                        title=row["title"] or "",
                        content=row["content"] or "",
                        category=row["category"] or None,
                    )
                    fetched_docs.append(doc)
                    # Update cache
                    if not disable_cache:
                        with self._lock:
                            self.cache.put(doc.doc_id, doc.to_dict())

                logger.debug(
                    "Fetched %d documents from DB out of %d missing",
                    len(fetched_docs),
                    len(missing_ids),
                )

            except sqlite3.Error as e:
                logger.exception("SQLite error fetching documents: %s", e)
                raise RuntimeError(f"Failed to fetch documents: {e}") from e

        # Combine and order
        all_docs_map = cached_docs
        for doc in fetched_docs:
            all_docs_map[doc.doc_id] = doc

        ordered_docs = [all_docs_map[doc_id] for doc_id in doc_ids if doc_id in all_docs_map]

        return ordered_docs

    def fetch_documents_batch(
        self, doc_ids_batch: list[list[int]], truncate_length: int | None = None
    ) -> list[list[Document]]:
        """
        Fetch documents for multiple queries in batch.

        Args:
            doc_ids_batch: List of document ID lists (one per query)
            truncate_length: Optional length to truncate title/content fields

        Returns:
            List of document lists (one per query)
        """
        results: list[list[Document]] = []

        for doc_ids in doc_ids_batch:
            documents = self.fetch_documents(doc_ids)

            # Apply truncation if requested
            if truncate_length is not None:
                documents = [doc.truncate(truncate_length) for doc in documents]

            results.append(documents)

        return results

    def clear_cache(self) -> None:
        """Clear the document cache safely."""
        with self._lock:
            self.cache.clear()

    def close_all(self) -> None:
        """
        Close all thread-local connections.

        Should be called during service shutdown.
        """
        if hasattr(self._local, "conn"):
            logger.info("Closing SQLite connection")
            self._local.conn.close()
            delattr(self._local, "conn")
        if self._shared_memory_conn is not None:
            logger.info("Closing shared in-memory document database")
            self._shared_memory_conn.close()
            self._shared_memory_conn = None

    def __repr__(self) -> str:
        """String representation of the store."""
        return f"DocumentStore(db_path={self.db_path})"
