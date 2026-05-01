from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, RAGSettings
from backend.services.retriever import ICDRetriever


def main() -> None:
    settings = RAGSettings()
    if FAISS_INDEX_PATH.exists():
        FAISS_INDEX_PATH.unlink()
    if FAISS_METADATA_PATH.exists():
        FAISS_METADATA_PATH.unlink()
    retriever = ICDRetriever(
        embedding_model=settings.embedding_model,
        embedding_batch_size=settings.embedding_batch_size,
    )
    print(f"Built FAISS index at {FAISS_INDEX_PATH}")
    print(f"Saved metadata at {FAISS_METADATA_PATH}")
    print(f"Indexed ICD entries: {len(retriever.entries)}")


if __name__ == "__main__":
    main()
