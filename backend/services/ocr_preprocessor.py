from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backend.core.config import RAGSettings
from backend.core.text import ABBREVIATION_PATTERNS, normalize_whitespace
from backend.core.utils import coerce_text_list, coerce_confidence
from backend.services.data_loader import ICDEntry
from backend.services.llm import BaseLLMClient

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover
    fuzz = None
    process = None


LOGGER = logging.getLogger("medichar.rag.ocr")
WORD_TOKEN_PATTERN = re.compile(r"^[a-z0-9%./+-]+$")
TOKEN_PATTERN = re.compile(r"\n|[a-z0-9%./+-]+|[:;,()\-]")
CONTROL_PATTERN = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u200b\u200c\u200d\ufeff]")
REPEATED_PUNCT_PATTERN = re.compile(r"([,.;:!?])\1+")
INTRAWORD_SYMBOL_PATTERN = re.compile(r"(?<=[a-z0-9])[{}\[\]|~`^*_+=<>\\](?=[a-z0-9])")
STRAY_SYMBOL_PATTERN = re.compile(r"[{}\[\]|~`^*_<>\\]")
SHORT_KEEPERS = {
    "cc",
    "dx",
    "hpi",
    "hx",
    "pmh",
    "ros",
    "pe",
    "sob",
    "cxr",
    "rr",
    "hr",
    "bp",
    "o2",
    "iv",
    "po",
    "im",
    "sat",
    "yo",
    "y/o",
    "ril",
    "rll",
    "rul",
    "lll",
    "lul",
    "in",
    "of",
    "to",
    "no",
    "on",
    "at",
}
DEFAULT_CLINICAL_NOTE_TERMS = (
    "admitted",
    "assessment",
    "breathing",
    "ceftriaxone",
    "chief complaint",
    "chills",
    "complains of",
    "condition",
    "confirmed",
    "consolidation",
    "cough",
    "crackles",
    "creatinine",
    "date",
    "diagnosis",
    "discharge condition",
    "dysuria",
    "edema",
    "emergency",
    "exam",
    "fatigue",
    "fever",
    "findings",
    "follow up",
    "history of present illness",
    "hospital course",
    "hypotension",
    "hypoxia",
    "impression",
    "improved",
    "infiltrate",
    "intravenous",
    "issues",
    "kidney injury",
    "leukocytosis",
    "lungs",
    "monitor",
    "oxygen",
    "patient",
    "physical examination",
    "pneumonia",
    "presents",
    "productive cough",
    "progressive",
    "recheck",
    "reports",
    "respiratory",
    "room air",
    "saturation",
    "secondary to",
    "sepsis",
    "shortness of breath",
    "stable",
    "supplemental oxygen",
    "tachycardic",
    "tachypneic",
    "temperature",
    "therapy",
    "tolerating",
    "urinalysis",
    "urinary tract infection",
    "vasopressor",
    "vital signs",
    "weakness",
    "xray",
)
OCR_SUBSTITUTIONS = (
    ("0", "o"),
    ("1", "l"),
    ("5", "s"),
    ("6", "g"),
    ("8", "b"),
    ("rn", "m"),
    ("vv", "w"),
)
REPLACEMENT_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00b4": "'",
        "\u02bc": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\t": " ",
    }
)


@dataclass(slots=True)
class MedicalVocabulary:
    terms: tuple[str, ...]
    term_set: frozenset[str]
    terms_by_length: dict[int, tuple[str, ...]]


@dataclass(slots=True)
class TokenCorrection:
    original: str
    corrected: str
    confidence: float
    reason: str


@dataclass(slots=True)
class OCRCorrectionResult:
    corrected_text: str
    corrections: list[TokenCorrection]
    uncertain_fragments: list[str]
    noise_score: float


@dataclass(slots=True)
class OCRReconstructionResult:
    reconstructed_note: str
    confidence: str
    uncertain_fragments: list[str]
    used_fallback: bool
    error: str | None = None


@dataclass(slots=True)
class OCRProcessingResult:
    raw_text: str
    cleaned_text: str
    fuzzy_corrected_text: str
    final_text_for_rag: str
    corrections: list[TokenCorrection]
    uncertain_fragments: list[str]
    reconstruction_confidence: str
    low_confidence: bool


class OCRPreprocessor:
    def __init__(
        self,
        *,
        settings: RAGSettings,
        llm_client: BaseLLMClient,
        catalog_entries: Iterable[ICDEntry],
    ) -> None:
        self.settings = settings
        self.llm_client = llm_client
        self.vocabulary = load_medical_vocabulary(
            self.settings.ocr_medical_vocab_path,
            catalog_entries=catalog_entries,
        )

    def process_ocr_text(self, text: str) -> OCRProcessingResult:
        return self.process_ocr_text_with_overrides(text)

    def process_ocr_text_with_overrides(
        self,
        text: str,
        *,
        mode: str | None = None,
        reconstruct_noise_threshold: float | None = None,
    ) -> OCRProcessingResult:
        return process_ocr_text(
            text,
            vocabulary=self.vocabulary,
            llm_client=self.llm_client,
            mode=mode if mode is not None else self.settings.ocr_preprocessing_mode,
            reconstruct_noise_threshold=(
                reconstruct_noise_threshold
                if reconstruct_noise_threshold is not None
                else self.settings.ocr_reconstruct_noise_threshold
            ),
            fuzzy_threshold=self.settings.ocr_fuzzy_threshold,
            short_token_threshold=self.settings.ocr_short_token_threshold,
            enabled=self.settings.ocr_preprocessing_enabled,
        )


def basic_clean_text(text: str) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = CONTROL_PATTERN.sub(" ", normalized)
    normalized = normalized.translate(REPLACEMENT_TABLE)
    normalized = INTRAWORD_SYMBOL_PATTERN.sub("", normalized)
    normalized = STRAY_SYMBOL_PATTERN.sub(" ", normalized)
    normalized = REPEATED_PUNCT_PATTERN.sub(r"\1", normalized)
    normalized = re.sub(r"[ \t]*\n[ \t]*", "\n", normalized)
    normalized = re.sub(r"\s*:\s*", ": ", normalized)
    normalized = re.sub(r"\s*;\s*", "; ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r"\s*([!?])\s*", r"\1 ", normalized)
    normalized = re.sub(r"(?<!\d)[ \t]*\.[ \t]*(?!\d)", ". ", normalized)
    normalized = re.sub(r"[ ]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = _join_short_letter_runs(normalized)
    normalized = _normalize_lines(normalized)
    return normalized.lower()


def tokenize_ocr_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_medical_vocabulary(
    path: str | Path,
    *,
    catalog_entries: Iterable[ICDEntry],
) -> MedicalVocabulary:
    terms: set[str] = set(SHORT_KEEPERS)
    vocab_path = Path(path)

    if vocab_path.exists():
        for line in vocab_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            _add_terms_from_text(terms, stripped)

    for replacement in ABBREVIATION_PATTERNS.values():
        _add_terms_from_text(terms, replacement)
    for phrase in DEFAULT_CLINICAL_NOTE_TERMS:
        _add_terms_from_text(terms, phrase)

    for entry in catalog_entries:
        _add_terms_from_text(terms, entry.title)
        for keyword in entry.keywords:
            _add_terms_from_text(terms, keyword)

    cleaned_terms = sorted(term for term in terms if _is_vocab_term(term))
    terms_by_length: dict[int, list[str]] = {}
    for term in cleaned_terms:
        terms_by_length.setdefault(len(term), []).append(term)

    return MedicalVocabulary(
        terms=tuple(cleaned_terms),
        term_set=frozenset(cleaned_terms),
        terms_by_length={key: tuple(value) for key, value in terms_by_length.items()},
    )


def fuzzy_correct_text(
    text: str,
    vocab: MedicalVocabulary,
    *,
    fuzzy_threshold: float = 90.0,
    short_token_threshold: float = 94.0,
) -> OCRCorrectionResult:
    tokens = tokenize_ocr_text(text)
    corrected_tokens: list[str] = []
    corrections: list[TokenCorrection] = []
    uncertain_fragments: list[str] = []
    word_token_count = 0
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if token == "\n" or not _is_word_token(token):
            corrected_tokens.append(token)
            index += 1
            continue

        word_token_count += 1
        merge_result = _maybe_merge_with_next(tokens, index, vocab)
        if merge_result is not None:
            merge_token, corrected_value, score = merge_result
            corrected_tokens.append(corrected_value)
            corrections.append(
                TokenCorrection(
                    original=merge_token,
                    corrected=corrected_value,
                    confidence=score,
                    reason="merge_split_token",
                )
            )
            index += 2
            continue

        split_result = _maybe_split_token(token, vocab)
        if split_result is not None:
            split_tokens, score = split_result
            corrected_tokens.extend(split_tokens)
            corrections.append(
                TokenCorrection(
                    original=token,
                    corrected=" ".join(split_tokens),
                    confidence=score,
                    reason="split_merged_token",
                )
            )
            index += 1
            continue

        corrected_token, confidence, reason = _best_token_correction(
            token,
            vocab,
            fuzzy_threshold=fuzzy_threshold,
            short_token_threshold=short_token_threshold,
        )
        corrected_tokens.append(corrected_token)
        if corrected_token != token:
            corrections.append(
                TokenCorrection(
                    original=token,
                    corrected=corrected_token,
                    confidence=confidence,
                    reason=reason,
                )
            )
        elif _looks_uncertain_token(token, vocab):
            uncertain_fragments.append(token)
        index += 1

    corrected_text = _detokenize_tokens(corrected_tokens)
    noise_score = len(uncertain_fragments) + (len(corrections) / max(word_token_count, 1))
    return OCRCorrectionResult(
        corrected_text=corrected_text,
        corrections=corrections,
        uncertain_fragments=_deduplicate_fragments(uncertain_fragments),
        noise_score=noise_score,
    )


def reconstruct_note_with_llm(
    text: str,
    *,
    llm_client: BaseLLMClient | None,
    uncertain_fragments: list[str] | None = None,
) -> OCRReconstructionResult:
    fallback_fragments = _deduplicate_fragments(uncertain_fragments or [])
    if llm_client is None:
        return OCRReconstructionResult(
            reconstructed_note=_inject_uncertainty_markers(text, fallback_fragments),
            confidence="Low",
            uncertain_fragments=fallback_fragments,
            used_fallback=True,
            error="No LLM client available for OCR reconstruction.",
        )

    try:
        payload = llm_client.reconstruct_note(noisy_text=text)
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("OCR reconstruction failed; falling back to fuzzy-corrected text: %s", exc)
        return OCRReconstructionResult(
            reconstructed_note=_inject_uncertainty_markers(text, fallback_fragments),
            confidence="Low",
            uncertain_fragments=fallback_fragments,
            used_fallback=True,
            error=str(exc),
        )

    reconstructed_note = str(payload.get("reconstructed_note") or "").strip()
    if not reconstructed_note:
        return OCRReconstructionResult(
            reconstructed_note=_inject_uncertainty_markers(text, fallback_fragments),
            confidence="Low",
            uncertain_fragments=fallback_fragments,
            used_fallback=True,
            error="LLM reconstruction returned an empty note.",
        )

    confidence = coerce_confidence(str(payload.get("confidence", "")))
    llm_uncertain = coerce_text_list(payload.get("uncertain_fragments"), max_items=8)
    merged_uncertain = _deduplicate_fragments(llm_uncertain)
    normalized_note = _normalize_reconstructed_note(reconstructed_note)
    if confidence == "Low" and merged_uncertain:
        normalized_note = _inject_uncertainty_markers(normalized_note, merged_uncertain)

    return OCRReconstructionResult(
        reconstructed_note=normalized_note,
        confidence=confidence,
        uncertain_fragments=merged_uncertain,
        used_fallback=False,
    )


def process_ocr_text(
    text: str,
    *,
    vocabulary: MedicalVocabulary,
    llm_client: BaseLLMClient | None,
    mode: str = "always",
    reconstruct_noise_threshold: float = 2.0,
    fuzzy_threshold: float = 90.0,
    short_token_threshold: float = 94.0,
    enabled: bool = True,
) -> OCRProcessingResult:
    raw_text = text or ""
    LOGGER.info("Raw OCR text: %s", raw_text)

    if not enabled or mode.strip().lower() == "off":
        return OCRProcessingResult(
            raw_text=raw_text,
            cleaned_text=raw_text,
            fuzzy_corrected_text=raw_text,
            final_text_for_rag=raw_text,
            corrections=[],
            uncertain_fragments=[],
            reconstruction_confidence="Low",
            low_confidence=False,
        )

    cleaned_text = basic_clean_text(raw_text)
    LOGGER.info("Basic cleaned OCR text: %s", cleaned_text)

    correction_result = fuzzy_correct_text(
        cleaned_text,
        vocabulary,
        fuzzy_threshold=fuzzy_threshold,
        short_token_threshold=short_token_threshold,
    )
    LOGGER.info("Fuzzy-corrected OCR text: %s", correction_result.corrected_text)
    LOGGER.info("Applied OCR corrections: %s", correction_result.corrections)

    normalized_mode = mode.strip().lower()
    if normalized_mode == "auto" and correction_result.noise_score < reconstruct_noise_threshold:
        final_text = correction_result.corrected_text
        if correction_result.uncertain_fragments:
            final_text = _inject_uncertainty_markers(final_text, correction_result.uncertain_fragments)
        LOGGER.info("Final reconstructed OCR note: %s", final_text)
        return OCRProcessingResult(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            fuzzy_corrected_text=correction_result.corrected_text,
            final_text_for_rag=final_text,
            corrections=correction_result.corrections,
            uncertain_fragments=correction_result.uncertain_fragments,
            reconstruction_confidence="Low",
            low_confidence=bool(correction_result.uncertain_fragments),
        )

    reconstruction = reconstruct_note_with_llm(
        correction_result.corrected_text,
        llm_client=llm_client,
        uncertain_fragments=correction_result.uncertain_fragments,
    )
    LOGGER.info("Final reconstructed OCR note: %s", reconstruction.reconstructed_note)

    return OCRProcessingResult(
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        fuzzy_corrected_text=correction_result.corrected_text,
        final_text_for_rag=reconstruction.reconstructed_note,
        corrections=correction_result.corrections,
        uncertain_fragments=reconstruction.uncertain_fragments,
        reconstruction_confidence=reconstruction.confidence,
        low_confidence=reconstruction.confidence == "Low" or bool(reconstruction.uncertain_fragments),
    )


def _add_terms_from_text(terms: set[str], text: str) -> None:
    cleaned = basic_clean_text(text)
    if not cleaned:
        return
    for token in tokenize_ocr_text(cleaned):
        if _is_vocab_term(token):
            terms.add(token)


def _is_vocab_term(token: str) -> bool:
    if not token or token == "\n":
        return False
    if token in SHORT_KEEPERS:
        return True
    if len(token) < 2:
        return False
    if not _is_word_token(token):
        return False
    return any(character.isalpha() for character in token)


def _is_word_token(token: str) -> bool:
    return bool(WORD_TOKEN_PATTERN.fullmatch(token))


def _normalize_lines(text: str) -> str:
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return normalize_whitespace(text)
    return "\n".join(lines)


def _join_short_letter_runs(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        parts = match.group(0).split()
        if 2 <= len(parts) <= 4 and all(len(part) == 1 for part in parts):
            return "".join(parts)
        return match.group(0)

    return re.sub(r"\b(?:[a-zA-Z]\s+){1,3}[a-zA-Z]\b", replace, text)


def _detokenize_tokens(tokens: list[str]) -> str:
    joined = " ".join(tokens)
    joined = re.sub(r"\s+([,.;:%)])", r"\1", joined)
    joined = re.sub(r"([(])\s+", r"\1", joined)
    joined = re.sub(r"\s*\n\s*", "\n", joined)
    joined = re.sub(r"\s{2,}", " ", joined)
    return joined.strip()


def _maybe_merge_with_next(
    tokens: list[str],
    index: int,
    vocab: MedicalVocabulary,
) -> tuple[str, str, float] | None:
    if index + 1 >= len(tokens):
        return None
    current = tokens[index]
    nxt = tokens[index + 1]
    if not (_is_word_token(current) and _is_word_token(nxt)):
        return None
    if current in vocab.term_set:
        return None

    merged = f"{current}{nxt}"
    if merged in vocab.term_set:
        return (f"{current} {nxt}", merged, 100.0)

    best_candidate, score = _best_vocab_match(merged, vocab)
    if best_candidate and score >= 97.0:
        return (f"{current} {nxt}", best_candidate, score)
    return None


def _maybe_split_token(token: str, vocab: MedicalVocabulary) -> tuple[list[str], float] | None:
    if token in vocab.term_set or len(token) < 6:
        return None

    best_split: tuple[list[str], float] | None = None
    for split_index in range(2, len(token) - 1):
        left = token[:split_index]
        right = token[split_index:]
        if left in vocab.term_set and right in vocab.term_set:
            score = 100.0 - abs(len(left) - len(right))
            candidate = ([left, right], score)
            if best_split is None or candidate[1] > best_split[1]:
                best_split = candidate
    return best_split


def _best_token_correction(
    token: str,
    vocab: MedicalVocabulary,
    *,
    fuzzy_threshold: float,
    short_token_threshold: float,
) -> tuple[str, float, str]:
    if token in vocab.term_set or _should_skip_token(token):
        return token, 100.0, "unchanged"

    for candidate in _generate_token_variants(token):
        if candidate in vocab.term_set:
            return candidate, 100.0, "ocr_variant_exact"

    best_candidate, score = _best_vocab_match(token, vocab)
    threshold = short_token_threshold if len(token) <= 4 else fuzzy_threshold
    if best_candidate and score >= threshold:
        return best_candidate, score, "fuzzy_match"
    return token, score, "unchanged"


def _generate_token_variants(token: str) -> list[str]:
    variants = {token}
    for source, target in OCR_SUBSTITUTIONS:
        if source in token:
            variants.add(token.replace(source, target))
    return sorted(variants)


def _best_vocab_match(token: str, vocab: MedicalVocabulary) -> tuple[str | None, float]:
    candidate_terms = _candidate_terms(token, vocab)
    if not candidate_terms:
        return None, 0.0

    best_term: str | None = None
    best_score = 0.0
    for variant in _generate_token_variants(token):
        match_term, score = _extract_best_match(variant, candidate_terms)
        if score > best_score or (score == best_score and match_term and best_term and match_term < best_term):
            best_term = match_term
            best_score = score
    return best_term, best_score


def _candidate_terms(token: str, vocab: MedicalVocabulary) -> list[str]:
    lengths = range(max(2, len(token) - 2), len(token) + 3)
    candidates: list[str] = []
    for length in lengths:
        candidates.extend(vocab.terms_by_length.get(length, ()))
    return candidates or list(vocab.terms)


def _extract_best_match(token: str, terms: list[str]) -> tuple[str | None, float]:
    if process is not None and fuzz is not None:
        match = process.extractOne(token, terms, scorer=fuzz.ratio)
        if match is None:
            return None, 0.0
        return str(match[0]), float(match[1])

    best_term: str | None = None
    best_score = 0.0
    for term in terms:
        score = difflib.SequenceMatcher(None, token, term).ratio() * 100
        if score > best_score or (score == best_score and best_term is not None and term < best_term):
            best_term = term
            best_score = score
    return best_term, best_score


def _should_skip_token(token: str) -> bool:
    if token in SHORT_KEEPERS:
        return True
    if token.replace(".", "", 1).isdigit():
        return True
    if re.fullmatch(r"\d+[/-]\d+[/-]\d+", token):
        return True
    if token.endswith("%") and token[:-1].replace(".", "", 1).isdigit():
        return True
    return False


def _looks_uncertain_token(token: str, vocab: MedicalVocabulary) -> bool:
    if token in vocab.term_set or _should_skip_token(token):
        return False
    if any(character.isdigit() for character in token) and any(character.isalpha() for character in token):
        return True
    if re.search(r"[^a-z0-9%./+-]", token):
        return True
    if len(token) >= 4 and not re.search(r"[aeiou]", token):
        return True
    if len(token) >= 6:
        _, score = _best_vocab_match(token, vocab)
        if 0 < score < 80:
            return True
    return False


def _deduplicate_fragments(fragments: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []
    for fragment in fragments:
        cleaned = normalize_whitespace(str(fragment).strip())
        normalized = cleaned.lower()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(cleaned)
    return deduplicated[:8]


def _normalize_reconstructed_note(note: str) -> str:
    normalized = unicodedata.normalize("NFKC", note).strip()
    normalized = _strip_wrapping_quotes(normalized)
    normalized = _remove_trailing_uncertain_fragment_section(normalized)
    normalized = re.sub(r"\r\n?", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    for quote in ('"', "'"):
        if len(stripped) >= 2 and stripped.startswith(quote) and stripped.endswith(quote):
            return stripped[1:-1].strip()
    return stripped


def _remove_trailing_uncertain_fragment_section(text: str) -> str:
    return re.sub(
        r"\n+\s*Uncertain\s+Fragments\s*:\s*(?:\n\s*-\s*\[unclear:[^\n]*\]\s*)+\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def _inject_uncertainty_markers(text: str, uncertain_fragments: list[str]) -> str:
    if not uncertain_fragments:
        return text

    updated = text
    unresolved: list[str] = []
    for fragment in uncertain_fragments:
        if f"[unclear: {fragment.lower()}]" in updated.lower():
            continue
        pattern = re.compile(re.escape(fragment), re.IGNORECASE)
        match = pattern.search(updated)
        if match:
            updated = pattern.sub(f"[unclear: {fragment}]", updated, count=1)
        else:
            unresolved.append(fragment)

    if unresolved:
        markers = "\n".join(f"- [unclear: {fragment}]" for fragment in unresolved[:5])
        updated = f"{updated}\n\nUncertain Fragments:\n{markers}"
    return updated.strip()
