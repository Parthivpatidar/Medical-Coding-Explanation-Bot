from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from backend.services.semantic_feature_extractor import (
    FEATURE_CATEGORIES,
    extract_semantic_features,
)


CATEGORY_WEIGHTS = {
    "diagnosis_terms": 0.18,
    "anatomy": 0.28,
    "temporal": 0.12,
    "severity": 0.15,
    "diagnostics": 0.16,
    "qualifiers": 0.10,
    "symptoms": 0.08,
    "symptom_clusters": 0.03,
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class FeatureAlignmentResult:
    score: float
    detail_overlap_score: float
    category_scores: dict[str, float]
    matched_features: dict[str, list[str]]
    note_features: dict[str, list[str]]
    candidate_features: dict[str, list[str]]


def extract_candidate_features(candidate: Any) -> dict[str, list[str]]:
    features = extract_semantic_features(_candidate_text(candidate))
    title = str(getattr(candidate, "title", "") or "").strip()
    if title:
        diagnosis_terms = set(features.get("diagnosis_terms", []))
        diagnosis_terms.add(_normalize_phrase(title))
        for fragment in re.split(r"\b(?:with|and|secondary to|complicated by|due to)\b", title.lower()):
            normalized_fragment = _normalize_phrase(fragment)
            if len(normalized_fragment) >= 4:
                diagnosis_terms.add(normalized_fragment)
        features["diagnosis_terms"] = sorted(item for item in diagnosis_terms if item)
    return features


def compute_feature_alignment(
    note_features: dict[str, list[str]],
    candidate_features: dict[str, list[str]],
) -> FeatureAlignmentResult:
    category_scores: dict[str, float] = {}
    matched_features: dict[str, list[str]] = {}
    weighted_score = 0.0
    total_available_weight = 0.0
    weighted_detail_overlap = 0.0

    for category in FEATURE_CATEGORIES:
        note_values = set(note_features.get(category, []))
        candidate_values = set(candidate_features.get(category, []))
        if not note_values:
            category_scores[category] = 0.0
            matched_features[category] = []
            continue

        weight = CATEGORY_WEIGHTS.get(category, 0.0)
        total_available_weight += weight
        score, matches = _category_alignment(note_values, candidate_values)
        category_scores[category] = score
        matched_features[category] = sorted(matches)
        weighted_score += weight * score
        weighted_detail_overlap += weight * _detail_overlap(note_values, matches)

    denominator = total_available_weight or 1.0
    return FeatureAlignmentResult(
        score=round(weighted_score / denominator, 4),
        detail_overlap_score=round(weighted_detail_overlap / denominator, 4),
        category_scores=category_scores,
        matched_features=matched_features,
        note_features=note_features,
        candidate_features=candidate_features,
    )


def align_candidate_to_note(
    note_features: dict[str, list[str]],
    candidate: Any,
) -> FeatureAlignmentResult:
    return compute_feature_alignment(
        note_features=note_features,
        candidate_features=extract_candidate_features(candidate),
    )


def _category_alignment(note_values: set[str], candidate_values: set[str]) -> tuple[float, set[str]]:
    if not note_values or not candidate_values:
        return 0.0, set()

    matched: set[str] = set()
    cumulative_score = 0.0
    for note_value in note_values:
        best_score = max(
            (_feature_similarity(note_value, candidate_value) for candidate_value in candidate_values),
            default=0.0,
        )
        if best_score >= 0.55:
            matched.add(note_value)
        cumulative_score += best_score

    return min(cumulative_score / len(note_values), 1.0), matched


def _detail_overlap(note_values: set[str], matched_values: set[str]) -> float:
    if not note_values:
        return 0.0
    return min(len(matched_values) / len(note_values), 1.0)


def _feature_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0

    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0

    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0

    recall = len(overlap) / len(left_tokens)
    precision = len(overlap) / len(right_tokens)
    return (0.65 * recall) + (0.35 * precision)


def _tokens(value: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(value.lower()))


def _candidate_text(candidate: Any) -> str:
    parts = [
        str(getattr(candidate, "title", "") or ""),
        str(getattr(candidate, "definition", "") or ""),
        str(getattr(candidate, "explanation_hint", "") or ""),
        str(getattr(candidate, "chunk_text", "") or ""),
    ]
    keywords = getattr(candidate, "keywords", [])
    if isinstance(keywords, list):
        parts.append(" ".join(str(keyword) for keyword in keywords))
    return " ".join(part for part in parts if part.strip())


def _normalize_phrase(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.lower()))
