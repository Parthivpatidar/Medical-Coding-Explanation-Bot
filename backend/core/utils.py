from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable


LOGGER_NAME = "medichar.rag"
JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
SUPPORTED_CONFIDENCE = {"high": "High", "medium": "Medium", "low": "Low"}


def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


def safe_json_loads(payload: str) -> dict[str, Any]:
    if not payload.strip():
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = JSON_BLOCK_PATTERN.search(payload)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def coerce_confidence(label: str | None, similarity_score: float | None = None) -> str:
    if label:
        normalized = SUPPORTED_CONFIDENCE.get(label.strip().lower())
        if normalized:
            return normalized
    if similarity_score is None:
        return "Low"
    if similarity_score >= 0.45:
        return "High"
    if similarity_score >= 0.25:
        return "Medium"
    return "Low"


def normalize_icd_code_key(code: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code or "").strip().upper())


def deduplicate_codes(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for item in items:
        code = str(item.get("code", "")).strip().upper()
        if not code or code in seen_codes:
            continue
        cleaned = dict(item)
        cleaned["code"] = code
        deduped.append(cleaned)
        seen_codes.add(code)
    return deduped


def coerce_text_list(raw_value: Any, *, max_items: int = 5) -> list[str]:
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        candidates = raw_value
    elif isinstance(raw_value, str):
        candidates = re.split(r"[\n;|]", raw_value)
    else:
        return []

    cleaned_items: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = str(item).strip().lstrip("-*").strip()
        normalized = text.lower()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        cleaned_items.append(text)
        if len(cleaned_items) >= max_items:
            break
    return cleaned_items


def parse_cors_origins(raw_origins: str) -> list[str]:
    value = raw_origins.strip()
    if not value or value == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]
