from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "faiss-cpu is not installed. Run `pip install -r requirements.txt` before using the API."
        ) from exc
    return faiss


@dataclass(slots=True)
class VectorSearchResult:
    index: int
    score: float
    metadata: dict[str, Any]


class FaissVectorStore:
    def __init__(self, *, index, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
        self.index = index
        self.records = records
        self.metadata = metadata

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        records: list[dict[str, Any]],
        *,
        metadata: dict[str, Any],
    ) -> "FaissVectorStore":
        if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
            raise ValueError("Embeddings and metadata records must align row-for-row")
        faiss = _load_faiss()
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(np.asarray(embeddings, dtype=np.float32))
        return cls(index=index, records=records, metadata=metadata)

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "FaissVectorStore":
        faiss = _load_faiss()
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Vector store artifacts do not exist yet")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(
            index=faiss.read_index(str(index_path)),
            records=list(payload.get("records", [])),
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, index_path: Path, metadata_path: Path) -> None:
        faiss = _load_faiss()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(
            json.dumps({"metadata": self.metadata, "records": self.records}, indent=2),
            encoding="utf-8",
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[VectorSearchResult]:
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        top_k = min(max(top_k, 1), len(self.records))
        if top_k == 0:
            return []
        scores, indices = self.index.search(np.asarray(query_embedding, dtype=np.float32), top_k)
        results: list[VectorSearchResult] = []
        for position, score in zip(indices[0], scores[0], strict=True):
            if position < 0:
                continue
            results.append(
                VectorSearchResult(
                    index=int(position),
                    score=float(score),
                    metadata=self.records[int(position)],
                )
            )
        return results
