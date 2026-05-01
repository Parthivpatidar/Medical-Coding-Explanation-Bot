from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
import re

from backend.core.config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    ICD_KB_PATH,
    ensure_directories,
)
from backend.core.text import clean_text
from backend.services.data_loader import load_icd_catalog
from backend.services.embedding import SentenceTransformerEmbedder
from backend.services.vector_store import FaissVectorStore, VectorSearchResult


@dataclass(slots=True)
class RetrievedICDEntry:
    code: str
    title: str
    keywords: list[str]
    definition: str
    explanation_hint: str
    chunk_text: str
    similarity_score: float

    def to_context_dict(self) -> dict[str, object]:
        return asdict(self)


LEXICAL_STOPWORDS = {
    "and",
    "assessment",
    "clinical",
    "confirmed",
    "diagnosis",
    "documented",
    "for",
    "from",
    "history",
    "note",
    "patient",
    "presenting",
    "this",
    "visit",
}
GENERIC_DIAGNOSIS_MODIFIERS = {
    "acute",
    "chronic",
    "due",
    "infection",
    "lobe",
    "left",
    "lower",
    "organism",
    "right",
    "specified",
    "type",
    "unspecified",
    "upper",
    "with",
    "without",
}
DIAGNOSIS_HEADER_PATTERN = (
    r"assessment(?:\s*/\s*diagnosis)?|diagnosis|diagnoses|impression|final diagnosis|"
    r"discharge diagnosis|dx"
)
SECTION_HEADER_PATTERN = re.compile(
    rf"^({DIAGNOSIS_HEADER_PATTERN})\s*:?\s*(.*)$",
    re.IGNORECASE,
)
INLINE_SECTION_HEADER_PATTERN = re.compile(
    rf"(?:^|[\n.;]\s*)({DIAGNOSIS_HEADER_PATTERN})\s*:\s*([^\n.;]+)",
    re.IGNORECASE,
)
FOCUS_VALUE_TERMINATOR_PATTERN = re.compile(
    r"\b(?:chief complaint|history of present illness|history|hpi|physical examination|"
    r"examination|plan|treatment|medications?|follow up|follow-up|vitals?|vital signs|"
    r"review of systems|ros|patient|age|date|temp|rr|hr|bp|oxygen saturation|chest x-ray|"
    r"x-ray)\s*:",
    re.IGNORECASE,
)
GENERIC_HEADER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z /()-]{1,60}:$")
RELEVANCE_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


def rerank_vector_hits(
    query: str,
    hits: list[VectorSearchResult],
    *,
    top_k: int,
    focus_phrases: list[str] | None = None,
) -> list[VectorSearchResult]:
    normalized_focus_phrases = focus_phrases or extract_focus_phrases(query)
    ranked = sorted(
        hits,
        key=lambda hit: (
            hit.score
            + lexical_relevance_score(query, hit.metadata)
            + diagnosis_focus_score(query, normalized_focus_phrases, hit.metadata),
            hit.score,
        ),
        reverse=True,
    )
    return ranked[:top_k]


def lexical_relevance_score(query: str, record: dict[str, object]) -> float:
    normalized_query = _normalized_phrase(query)
    query_tokens = _relevance_tokens(normalized_query)
    record_tokens = _record_tokens(record)
    if not query_tokens or not record_tokens:
        return 0.0

    overlap = len(query_tokens & record_tokens)
    if overlap == 0:
        return 0.0

    coverage = overlap / len(record_tokens)
    recall = overlap / len(query_tokens)
    exact_title_bonus = 0.2 if _normalized_phrase(str(record.get("title", ""))) in normalized_query else 0.0
    return (0.45 * coverage) + (0.15 * recall) + exact_title_bonus


def diagnosis_focus_score(query: str, focus_phrases: list[str], record: dict[str, object]) -> float:
    if not focus_phrases:
        return 0.0

    query_tokens = _relevance_tokens(query)
    record_tokens = _record_tokens(record)
    if not query_tokens or not record_tokens:
        return 0.0

    normalized_title = _normalized_phrase(str(record.get("title", "")))
    raw_keywords = record.get("keywords", [])
    normalized_keywords = {
        _normalized_phrase(str(keyword))
        for keyword in raw_keywords
        if str(keyword).strip()
    } if isinstance(raw_keywords, list) else set()

    best_score = 0.0
    for phrase in focus_phrases:
        phrase_tokens = _relevance_tokens(phrase)
        if not phrase_tokens:
            continue
        overlap = len(phrase_tokens & record_tokens)
        if overlap == 0:
            continue

        recall = overlap / len(phrase_tokens)
        precision = overlap / len(record_tokens)
        normalized_phrase = _normalized_phrase(phrase)
        exact_bonus = 0.45 if (
            normalized_phrase == normalized_title
            or normalized_phrase in normalized_keywords
        ) else 0.0
        unsupported_modifiers = record_tokens - query_tokens - GENERIC_DIAGNOSIS_MODIFIERS - phrase_tokens
        modifier_penalty = min(len(unsupported_modifiers) * 0.25, 1.0)
        candidate_score = (0.95 * recall) + (0.25 * precision) + exact_bonus - modifier_penalty
        if candidate_score > best_score:
            best_score = candidate_score

    return max(best_score, 0.0)


def extract_focus_phrases(text: str, max_phrases: int = 4) -> list[str]:
    lines = [line.strip() for line in str(text or "").splitlines()]
    phrases: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        match = SECTION_HEADER_PATTERN.match(line)
        if not match:
            index += 1
            continue

        inline_value = _trim_focus_value(match.group(2))
        if inline_value:
            normalized_inline = _normalize_focus_phrase(inline_value)
            if normalized_inline:
                phrases.append(normalized_inline)
        else:
            next_index = index + 1
            collected_lines: list[str] = []
            while next_index < len(lines):
                candidate = lines[next_index].strip()
                if not candidate:
                    if collected_lines:
                        break
                    next_index += 1
                    continue
                if SECTION_HEADER_PATTERN.match(candidate) or GENERIC_HEADER_PATTERN.match(candidate):
                    break
                collected_lines.append(candidate)
                next_index += 1
            normalized_block = _normalize_focus_phrase(" ".join(collected_lines))
            if normalized_block:
                phrases.append(normalized_block)
            index = next_index
            continue

        index += 1

    phrases.extend(_extract_inline_focus_phrases(text))

    deduplicated: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        deduplicated.append(phrase)
        if len(deduplicated) >= max_phrases:
            break
    return deduplicated


def build_focus_queries(text: str) -> list[str]:
    queries: list[str] = []
    for phrase in extract_focus_phrases(text):
        queries.append(phrase)
        queries.append(f"diagnosis {phrase}")
        queries.append(f"assessment {phrase}")
    deduplicated: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized_query = _normalized_phrase(query)
        if not normalized_query or normalized_query in seen:
            continue
        seen.add(normalized_query)
        deduplicated.append(query)
    return deduplicated


def _record_tokens(record: dict[str, object]) -> set[str]:
    tokens = _relevance_tokens(str(record.get("title", "")))
    raw_keywords = record.get("keywords", [])
    if isinstance(raw_keywords, list):
        for keyword in raw_keywords:
            tokens.update(_relevance_tokens(str(keyword)))
    return tokens


def _relevance_tokens(text: str) -> set[str]:
    normalized = _normalized_phrase(text)
    return {
        token
        for token in RELEVANCE_TOKEN_PATTERN.findall(normalized)
        if len(token) > 2 and token not in LEXICAL_STOPWORDS
    }


def _normalized_phrase(text: str) -> str:
    return clean_text(text).replace("/", " ").strip()


def _normalize_focus_phrase(text: str) -> str:
    normalized = _normalized_phrase(_trim_focus_value(text))
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" .:-")


def _extract_inline_focus_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    for match in INLINE_SECTION_HEADER_PATTERN.finditer(str(text or "")):
        normalized = _normalize_focus_phrase(match.group(2))
        if normalized:
            phrases.append(normalized)
    return phrases


def _trim_focus_value(text: str) -> str:
    value = str(text or "").strip(" .:-")
    terminator = FOCUS_VALUE_TERMINATOR_PATTERN.search(value)
    if terminator:
        value = value[: terminator.start()]
    return value.strip(" .:-")


class ICDRetriever:
    def __init__(
        self,
        *,
        catalog_path: Path = ICD_KB_PATH,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = FAISS_METADATA_PATH,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_batch_size: int = 32,
    ) -> None:
        ensure_directories()
        self.catalog_path = catalog_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedder = SentenceTransformerEmbedder(
            model_name=embedding_model,
            batch_size=embedding_batch_size,
        )
        self.entries = load_icd_catalog(self.catalog_path)
        self.catalog_by_code = {entry.code: entry for entry in self.entries}
        self._query_cache: OrderedDict[tuple[str, int, bool], list[RetrievedICDEntry]] = OrderedDict()
        self.vector_store = self._load_or_build_vector_store()

    def retrieve(
        self,
        clinical_text: str,
        top_k: int = 5,
        *,
        expand_queries: bool = True,
    ) -> list[RetrievedICDEntry]:
        query = clean_text(clinical_text)
        if not query:
            return []
        cache_key = (query, int(top_k), bool(expand_queries))
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
            return list(cached)

        multiplier = 10 if expand_queries else 6
        minimum_pool = 20 if expand_queries else 10
        candidate_pool_size = min(max(top_k * multiplier, minimum_pool), len(self.entries))
        aggregated_hits: dict[str, VectorSearchResult] = {}
        search_queries = [query, *build_focus_queries(clinical_text)] if expand_queries else [query]

        for current_query in search_queries:
            current_hits = self.vector_store.search(
                self.embedder.encode_query(clean_text(current_query)),
                top_k=candidate_pool_size,
            )
            for hit in current_hits:
                code = str(hit.metadata["code"])
                existing = aggregated_hits.get(code)
                if existing is None or hit.score > existing.score:
                    aggregated_hits[code] = hit

        hits = rerank_vector_hits(
            query,
            list(aggregated_hits.values()),
            top_k=top_k,
            focus_phrases=extract_focus_phrases(clinical_text),
        )
        retrieved: list[RetrievedICDEntry] = []
        for hit in hits:
            record = hit.metadata
            retrieved.append(
                RetrievedICDEntry(
                    code=str(record["code"]),
                    title=str(record["title"]),
                    keywords=list(record.get("keywords", [])),
                    definition=str(record.get("definition", "")),
                    explanation_hint=str(record.get("explanation_hint", "")),
                    chunk_text=str(record["chunk_text"]),
                    similarity_score=round(float(hit.score), 4),
                )
            )
        self._query_cache[cache_key] = list(retrieved)
        if len(self._query_cache) > 128:
            self._query_cache.popitem(last=False)
        return list(retrieved)

    def rebuild(self) -> None:
        self.entries = load_icd_catalog(self.catalog_path)
        self.catalog_by_code = {entry.code: entry for entry in self.entries}
        self.vector_store = self._build_vector_store()

    def _load_or_build_vector_store(self) -> FaissVectorStore:
        if self.index_path.exists() and self.metadata_path.exists():
            vector_store = FaissVectorStore.load(self.index_path, self.metadata_path)
            if not self._is_stale(vector_store.metadata):
                return vector_store
        return self._build_vector_store()

    def _build_vector_store(self) -> FaissVectorStore:
        texts = [clean_text(entry.chunk_text) for entry in self.entries]
        embeddings = self.embedder.encode(texts)
        records = [entry.to_dict() for entry in self.entries]
        metadata = self._catalog_signature()
        vector_store = FaissVectorStore.build(embeddings, records, metadata=metadata)
        vector_store.save(self.index_path, self.metadata_path)
        return vector_store

    def _catalog_signature(self) -> dict[str, object]:
        stat = self.catalog_path.stat()
        return {
            "catalog_path": str(self.catalog_path.resolve()),
            "catalog_mtime_ns": stat.st_mtime_ns,
            "catalog_size": len(self.entries),
            "embedding_model": self.embedder.model_name,
        }

    def _is_stale(self, metadata: dict[str, object]) -> bool:
        expected = self._catalog_signature()
        return any(metadata.get(key) != value for key, value in expected.items())
