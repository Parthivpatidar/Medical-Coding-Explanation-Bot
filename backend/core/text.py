from __future__ import annotations

import re
from collections import OrderedDict


ABBREVIATION_PATTERNS = OrderedDict(
    {
        r"\bpt\b": "patient",
        r"\bdm\b": "diabetes mellitus",
        r"\bdm2\b": "type 2 diabetes mellitus",
        r"\bhtn\b": "hypertension",
        r"\bcp\b": "chest pain",
        r"\bsob\b": "shortness of breath",
        r"\bw\s*/\s*o\b": "without",
        r"\bw\s*/\s*": "with ",
        r"\bc\/o\b": "complains of",
        r"\bhx\b": "history",
        r"\bdx\b": "diagnosis",
        r"\bfx\b": "fracture",
        r"\bneurop\b": "neuropathy",
    }
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def expand_abbreviations(text: str) -> str:
    expanded = text
    for pattern, replacement in ABBREVIATION_PATTERNS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def normalize_clinical_text(text: str) -> str:
    normalized = text.replace("\n", " ")
    normalized = expand_abbreviations(normalized)
    normalized = re.sub(r"[^A-Za-z0-9\s/.\-]", " ", normalized)
    normalized = normalized.replace("/", " / ")
    return normalize_whitespace(normalized)


def clean_text(text: str) -> str:
    return normalize_clinical_text(text).lower()


def split_keywords(raw_keywords: str | None) -> list[str]:
    if not raw_keywords:
        return []
    return [
        normalize_whitespace(item)
        for item in str(raw_keywords).split("|")
        if normalize_whitespace(item)
    ]


def parse_icd_codes(*raw_values: str | None) -> list[str]:
    seen: set[str] = set()
    codes: list[str] = []
    for raw in raw_values:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        for part in re.split(r"[|;,]", text):
            code = part.strip().upper()
            if not code or code in seen:
                continue
            seen.add(code)
            codes.append(code)
    return codes


def extract_note_evidence(text: str, limit: int = 2) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|;\s+", normalize_clinical_text(text))
    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
    return cleaned[:limit]
