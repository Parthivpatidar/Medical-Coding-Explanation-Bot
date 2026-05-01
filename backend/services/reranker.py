from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, TypeVar

from backend.services.feature_alignment import (
    FeatureAlignmentResult,
    align_candidate_to_note,
)
from backend.services.semantic_feature_extractor import extract_semantic_features


T = TypeVar("T")

DEFAULT_SIMILARITY_WEIGHT = 0.72
DEFAULT_FEATURE_WEIGHT = 0.28
SPECIFICITY_TIE_BONUS = 0.04


@dataclass(slots=True)
class RankedCandidate:
    candidate: Any
    similarity_score: float
    feature_alignment_score: float
    semantic_detail_overlap: float
    final_score: float
    alignment: FeatureAlignmentResult


def score_semantic_candidates(
    clinical_text: str,
    candidates: Iterable[T],
    *,
    similarity_weight: float = DEFAULT_SIMILARITY_WEIGHT,
    feature_weight: float = DEFAULT_FEATURE_WEIGHT,
) -> list[RankedCandidate]:
    note_features = extract_semantic_features(clinical_text)
    normalized_similarity_weight, normalized_feature_weight = _normalize_weights(
        similarity_weight,
        feature_weight,
    )

    scored: list[RankedCandidate] = []
    for candidate in candidates:
        similarity_score = _coerce_score(getattr(candidate, "similarity_score", 0.0))
        alignment = align_candidate_to_note(note_features, candidate)
        final_score = (
            normalized_similarity_weight * similarity_score
            + normalized_feature_weight * alignment.score
            + SPECIFICITY_TIE_BONUS * alignment.detail_overlap_score
        )
        scored.append(
            RankedCandidate(
                candidate=candidate,
                similarity_score=similarity_score,
                feature_alignment_score=alignment.score,
                semantic_detail_overlap=alignment.detail_overlap_score,
                final_score=round(final_score, 4),
                alignment=alignment,
            )
        )
    return scored


def rerank_retrieved_candidates(
    clinical_text: str,
    candidates: list[T],
    *,
    similarity_weight: float = DEFAULT_SIMILARITY_WEIGHT,
    feature_weight: float = DEFAULT_FEATURE_WEIGHT,
) -> list[T]:
    ranked = score_semantic_candidates(
        clinical_text,
        candidates,
        similarity_weight=similarity_weight,
        feature_weight=feature_weight,
    )
    ranked.sort(key=_rank_sort_key, reverse=True)
    return [item.candidate for item in ranked]


def select_semantically_supported_candidates(
    ranked_candidates: list[RankedCandidate],
    *,
    max_candidates: int = 3,
    min_alignment_score: float = 0.12,
    min_final_score_ratio: float = 0.84,
) -> list[RankedCandidate]:
    if not ranked_candidates:
        return []

    ranked = sorted(ranked_candidates, key=_rank_sort_key, reverse=True)
    selected: list[RankedCandidate] = [ranked[0]]
    top_score = max(ranked[0].final_score, 0.0001)

    for candidate in ranked[1:]:
        if len(selected) >= max_candidates:
            break
        if candidate.feature_alignment_score < min_alignment_score:
            continue
        if candidate.final_score < top_score * min_final_score_ratio:
            continue
        if not _adds_distinct_semantic_evidence(candidate, selected):
            continue
        selected.append(candidate)
    return selected


def _rank_sort_key(candidate: RankedCandidate) -> tuple[float, float, float, float]:
    return (
        candidate.final_score,
        candidate.semantic_detail_overlap,
        candidate.feature_alignment_score,
        candidate.similarity_score,
    )


def _adds_distinct_semantic_evidence(
    candidate: RankedCandidate,
    selected: list[RankedCandidate],
) -> bool:
    candidate_matches = _matched_feature_set(candidate)
    if not candidate_matches:
        return False

    selected_matches: set[str] = set()
    for item in selected:
        selected_matches.update(_matched_feature_set(item))

    new_matches = candidate_matches - selected_matches
    return len(new_matches) >= 1


def _matched_feature_set(candidate: RankedCandidate) -> set[str]:
    matches: set[str] = set()
    for category, values in candidate.alignment.matched_features.items():
        for value in values:
            matches.add(f"{category}:{value}")
    return matches


def _normalize_weights(similarity_weight: float, feature_weight: float) -> tuple[float, float]:
    similarity_weight = max(float(similarity_weight), 0.0)
    feature_weight = max(float(feature_weight), 0.0)
    total = similarity_weight + feature_weight
    if total <= 0:
        return DEFAULT_SIMILARITY_WEIGHT, DEFAULT_FEATURE_WEIGHT
    return similarity_weight / total, feature_weight / total


def _coerce_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(score, 0.0), 1.0)
