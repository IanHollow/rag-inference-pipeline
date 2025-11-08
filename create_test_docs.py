from pathlib import Path
import sqlite3

import faiss
import numpy as np

DOCUMENTS_DIR = "documents/"


def _initialize_documents() -> None:
    """Create dummy documents database if it doesn't exist"""

    NUM_DOCUMENTS = 1000000
    Path(DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)

    db_path = Path(DOCUMENTS_DIR) / "documents.db"

    if not db_path.exists():
        print("Creating document database...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table with index on doc_id for fast lookups
        cursor.execute(
            """
                CREATE TABLE documents (
                    doc_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL
                )
            """
        )

        # Insert documents in batches for better performance
        batch_size = 10000
        documents = []

        for i in range(NUM_DOCUMENTS):
            documents.append(
                (
                    i,
                    f"Document {i}",
                    f"This is the content of document {i}. It contains information about customer support issue {i % 100}.",
                    ["technical", "billing", "shipping", "general"][i % 4],
                )
            )

            # Insert in batches
            if len(documents) >= batch_size:
                cursor.executemany(
                    "INSERT INTO documents (doc_id, title, content, category) VALUES (?, ?, ?, ?)",
                    documents,
                )
                conn.commit()
                documents = []
                print(f"Created {i + 1}/{NUM_DOCUMENTS} documents...")

        # Insert remaining documents
        if documents:
            cursor.executemany(
                "INSERT INTO documents (doc_id, title, content, category) VALUES (?, ?, ?, ?)",
                documents,
            )
            conn.commit()

        conn.close()
        print(f"Document database created at {db_path}")
        print(f"Database size: {db_path.stat().st_size / 1e6:.2f} MB")


def _create_faiss_index() -> None:
    """Create a large FAISS index"""
    dim = 768
    num_docs = 1000000
    index_path = Path(DOCUMENTS_DIR) / "faiss_index.bin"

    nlist = 4096
    # Create index
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    rng = np.random.default_rng()
    index.train(rng.standard_normal((10000, dim)).astype("float32"))
    # Add vectors in batches to manage memory
    batch_size = 10000
    for i in range(0, num_docs, batch_size):
        # Generate random embeddings (in real scenario, these would be document embeddings)
        batch_embeddings = rng.standard_normal((min(batch_size, num_docs - i), dim)).astype(
            "float32"
        )
        index.add(batch_embeddings)

        if i % 100000 == 0:
            print(f"Added {i}/{num_docs} vectors to index...")

    # Save index
    index.nprobe = 64
    faiss.write_index(index, index_path)
    print("FAISS index created and saved")


if __name__ == "__main__":
    _initialize_documents()
    _create_faiss_index()
