from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.core.config import ICD_KB_PATH
from backend.core.text import parse_icd_codes, split_keywords


@dataclass(slots=True)
class ICDEntry:
    code: str
    title: str
    keywords: list[str]
    definition: str
    explanation_hint: str
    chunk_text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationExample:
    note_id: str
    note_text: str
    icd_codes: list[str]


def build_chunk_text(
    *,
    code: str,
    title: str,
    keywords: list[str],
    definition: str,
    explanation_hint: str,
) -> str:
    return (
        f"Code: {code}, "
        f"Title: {title}, "
        f"Keywords: {', '.join(keywords) or 'N/A'}, "
        f"Definition: {definition or 'N/A'}, "
        f"Hint: {explanation_hint or 'N/A'}"
    )


def load_icd_catalog(path: Path | None = None) -> list[ICDEntry]:
    catalog_path = path or ICD_KB_PATH
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"ICD catalog not found at {catalog_path}. "
            "Set MEDICHAR_ICD_KB_PATH or add the MIMIC demo dataset before starting the API."
        )

    with catalog_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if {"code", "title", "keywords", "definition", "explanation_hint"}.issubset(fieldnames):
            return _load_enriched_catalog(reader, catalog_path)
        if {"icd_code", "icd_version", "long_title"}.issubset(fieldnames):
            return _load_mimic_icd_catalog(reader, catalog_path)
        raise ValueError(
            "ICD KB must be either the enriched catalog schema "
            "`code,title,keywords,definition,explanation_hint` or the MIMIC schema "
            "`icd_code,icd_version,long_title`."
        )


def _load_enriched_catalog(reader: csv.DictReader, catalog_path: Path) -> list[ICDEntry]:
    entries: list[ICDEntry] = []
    for row in reader:
        code = str(row.get("code", "")).strip().upper()
        title = str(row.get("title", "")).strip()
        if not code or not title:
            continue
        keywords = split_keywords(row.get("keywords"))
        definition = str(row.get("definition", "")).strip()
        explanation_hint = str(row.get("explanation_hint", "")).strip()
        entries.append(_build_entry(code, title, keywords, definition, explanation_hint))
    if not entries:
        raise ValueError(f"No valid ICD rows were found in {catalog_path}")
    return entries


def _load_mimic_icd_catalog(reader: csv.DictReader, catalog_path: Path) -> list[ICDEntry]:
    entries: list[ICDEntry] = []
    for row in reader:
        if str(row.get("icd_version", "")).strip() != "10":
            continue
        code = str(row.get("icd_code", "")).strip().upper()
        title = str(row.get("long_title", "")).strip()
        if not code or not title:
            continue
        keywords = _keywords_from_title(title)
        definition = f"MIMIC-IV ICD-10-CM diagnosis title: {title}."
        explanation_hint = "Use only when the clinical note supports this ICD-10-CM diagnosis."
        entries.append(_build_entry(code, title, keywords, definition, explanation_hint))
    if not entries:
        raise ValueError(f"No ICD-10 rows were found in {catalog_path}")
    return entries


def _build_entry(
    code: str,
    title: str,
    keywords: list[str],
    definition: str,
    explanation_hint: str,
) -> ICDEntry:
    return ICDEntry(
        code=code,
        title=title,
        keywords=keywords,
        definition=definition,
        explanation_hint=explanation_hint,
        chunk_text=build_chunk_text(
            code=code,
            title=title,
            keywords=keywords,
            definition=definition,
            explanation_hint=explanation_hint,
        ),
    )


def _keywords_from_title(title: str) -> list[str]:
    normalized = title.replace(",", " ").replace("(", " ").replace(")", " ")
    words = [word.strip().lower() for word in normalized.split() if len(word.strip()) > 2]
    keywords: list[str] = [title]
    seen = {title.lower()}
    for word in words:
        if word not in seen:
            keywords.append(word)
            seen.add(word)
    return keywords[:12]


def load_evaluation_examples(path: Path) -> list[EvaluationExample]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[EvaluationExample] = []
        for row in reader:
            rows.append(
                EvaluationExample(
                    note_id=_extract_value(row, "note_id", "row_id", "hadm_id", default="").strip(),
                    note_text=_extract_value(
                        row,
                        "note_text",
                        "text",
                        "clinical_text",
                        "note",
                        "discharge_summary",
                        default="",
                    ).strip(),
                    icd_codes=parse_icd_codes(
                        row.get("icd_code"),
                        row.get("icd_codes"),
                        row.get("label"),
                        row.get("labels"),
                    ),
                )
            )
    return rows


def _extract_value(row: dict[str, object], *keys: str, default: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default
