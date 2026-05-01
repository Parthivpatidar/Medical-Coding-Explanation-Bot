from __future__ import annotations

import logging
import re
from typing import Any

from backend.core.config import RAGSettings
from backend.core.text import extract_note_evidence
from backend.core.utils import (
    coerce_confidence,
    coerce_text_list,
    deduplicate_codes,
    normalize_icd_code_key,
)
from backend.services.llm import HeuristicLLMClient, build_llm_client
from backend.services.local_vision import LocalTrOCRVisionClient
from backend.services.ocr_preprocessor import OCRPreprocessor
from backend.services.retriever import (
    ICDRetriever,
    RetrievedICDEntry,
    diagnosis_focus_score,
    extract_focus_phrases,
    lexical_relevance_score,
)
from backend.services.reranker import rerank_retrieved_candidates, score_semantic_candidates
from backend.services.vision import WindowsOCRVisionClient, build_vision_client
from backend.services.vision_types import VisionExtractionResult


LOGGER = logging.getLogger("medichar.rag.api")


FOCUS_SPLIT_PATTERN = re.compile(
    r"\b(?:with|and|secondary to|complicated by|due to|leading to|in the setting of|in setting of)\b",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")
OCR_UNCERTAINTY_PATTERN = re.compile(r"\[unclear:[^\]]+\]", re.IGNORECASE)
TITLE_STOPWORDS = {
    "and",
    "condition",
    "diagnosis",
    "disease",
    "disorder",
    "due",
    "for",
    "in",
    "of",
    "on",
    "other",
    "the",
    "to",
    "type",
}
GENERIC_TITLE_TERMS = {
    "acute",
    "chronic",
    "complication",
    "essential",
    "exacerbation",
    "lower",
    "organism",
    "primary",
    "right",
    "secondary",
    "specified",
    "unspecified",
    "upper",
    "with",
    "without",
}
NOISE_MARKERS = ("unclear", "\ufffd", "\x07", "{", "}", "|", "~", "`")
STRUCTURED_NOTE_HEADER_PATTERN = re.compile(
    r"\b(?:chief complaint|history of present illness|hospital course|assessment|diagnosis|"
    r"physical examination|examination|plan|findings|discharge condition)\b\s*:?",
    re.IGNORECASE,
)
OCR_GARBLED_TOKEN_PATTERN = re.compile(
    r"\b(?:[a-z]*\d+[a-z]+|[a-z]+\d+[a-z]*|[a-z]{1,3}[?~`|{}\\/\[\]]+[a-z0-9]*)\b",
    re.IGNORECASE,
)
TRAUMA_CONTEXT_PATTERN = re.compile(
    r"\b(?:trauma|fall|fracture|wound|laceration|burn|accident|assault|"
    r"motor vehicle|mvc|poison(?:ing)?|overdose|toxic effect)\b",
    re.IGNORECASE,
)
PREGNANCY_CONTEXT_PATTERN = re.compile(
    r"\b(?:pregnan(?:cy|t)|postpartum|antepartum|delivery|labor|obstetric|gestation)\b",
    re.IGNORECASE,
)
NEWBORN_CONTEXT_PATTERN = re.compile(
    r"\b(?:newborn|neonat(?:al|e)|perinatal|infant of)\b",
    re.IGNORECASE,
)
GENERIC_COMPLICATION_TERMS = {
    "abscess",
    "angiopathy",
    "coma",
    "gangrene",
    "hemorrhage",
    "ketoacidosis",
    "necrosis",
    "perforation",
    "poisoning",
    "septic shock",
    "toxic",
    "ulcer",
}
DIAGNOSIS_EVIDENCE_TERMS = {
    "assessment",
    "clinical picture consistent with",
    "diagnosis",
    "diagnoses",
    "final diagnosis",
    "impression",
    "secondary to",
    "with",
}


class MedicalCodingService:
    def __init__(self, settings: RAGSettings | None = None) -> None:
        self.settings = settings or RAGSettings()
        self.retriever = ICDRetriever(
            embedding_model=self.settings.embedding_model,
            embedding_batch_size=self.settings.embedding_batch_size,
        )
        self.llm = build_llm_client(
            provider=self.settings.llm_provider,
            model=self.settings.llm_model,
            searchapi_base_url=self.settings.searchapi_base_url,
            searchapi_engine=self.settings.searchapi_engine,
            searchapi_location=self.settings.searchapi_location,
            searchapi_hl=self.settings.searchapi_hl,
            searchapi_gl=self.settings.searchapi_gl,
            temperature=self.settings.llm_temperature,
            timeout_seconds=self.settings.request_timeout_seconds,
        )
        self.fallback_llm = HeuristicLLMClient()
        self.vision_client = build_vision_client(
            provider=self.settings.vision_provider,
            openai_api_key=self.settings.openai_api_key,
            openai_responses_url=self.settings.openai_responses_url,
            model=self.settings.vision_model,
            detail=self.settings.vision_detail,
            timeout_seconds=self.settings.vision_timeout_seconds,
            device=self.settings.vision_device,
        )
        self.handwritten_vision_client = LocalTrOCRVisionClient(
            model_name=self.settings.vision_model,
            device=self.settings.vision_device,
        )
        self.printed_vision_client = WindowsOCRVisionClient(ocr_mode="printed")
        self.ocr_preprocessor = OCRPreprocessor(
            settings=self.settings,
            llm_client=self.llm,
            catalog_entries=self.retriever.entries,
        )

    def predict(
        self,
        *,
        clinical_text: str,
        top_k: int | None = None,
        return_context: bool = False,
        return_similarity: bool = False,
    ) -> dict[str, Any]:
        return self._predict_text(
            clinical_text=clinical_text,
            top_k=top_k,
            return_context=return_context,
            return_similarity=return_similarity,
        )

    def predict_from_image(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        supplemental_text: str = "",
        ocr_mode: str = "printed",
        top_k: int | None = None,
        return_context: bool = False,
        return_similarity: bool = False,
    ) -> dict[str, Any]:
        extraction = self._extract_and_process_note_image(
            image_bytes=image_bytes,
            filename=filename,
            supplemental_text=supplemental_text,
            ocr_mode=ocr_mode,
        )
        return self._predict_text(
            clinical_text=extraction.note_text,
            top_k=top_k,
            return_context=return_context,
            return_similarity=return_similarity,
        )

    def extract_note_from_image(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        supplemental_text: str = "",
        ocr_mode: str = "printed",
    ) -> dict[str, Any]:
        extraction = self._extract_and_process_note_image(
            image_bytes=image_bytes,
            filename=filename,
            supplemental_text=supplemental_text,
            ocr_mode=ocr_mode,
        )
        return {
            "text": extraction.note_text,
            "confidence": extraction.confidence,
            "uncertain_fragments": extraction.uncertain_fragments,
            "raw_text": extraction.raw_text or extraction.note_text,
            "cleaned_text": extraction.cleaned_text or extraction.note_text,
            "processed_text": extraction.processed_text or extraction.note_text,
        }

    def _extract_and_process_note_image(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        supplemental_text: str = "",
        ocr_mode: str = "printed",
    ) -> VisionExtractionResult:
        vision_client = self._vision_client_for_ocr_mode(ocr_mode)
        extraction = vision_client.extract_note_from_image(
            image_bytes=image_bytes,
            filename=filename,
            supplemental_text=supplemental_text,
        )
        try:
            preprocessing_mode = "auto" if str(ocr_mode or "").strip().lower() == "handwritten" else None
            reconstruction_threshold = 3.25 if preprocessing_mode == "auto" else None
            ocr_result = self.ocr_preprocessor.process_ocr_text_with_overrides(
                extraction.note_text,
                mode=preprocessing_mode,
                reconstruct_noise_threshold=reconstruction_threshold,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("OCR post-processing failed, returning raw extracted note: %s", exc)
            return extraction

        final_text = self._select_final_ocr_text(
            raw_text=extraction.note_text,
            ocr_result=ocr_result,
            ocr_mode=ocr_mode,
            extraction_confidence=extraction.confidence,
        )
        uncertain_fragments = list(
            dict.fromkeys([*extraction.uncertain_fragments, *ocr_result.uncertain_fragments])
        )
        confidence = ocr_result.reconstruction_confidence
        if confidence == "Low" and extraction.confidence in {"High", "Medium"} and not ocr_result.low_confidence:
            confidence = extraction.confidence
        return VisionExtractionResult(
            note_text=final_text,
            confidence=coerce_confidence(confidence),
            uncertain_fragments=uncertain_fragments[:8],
            provider=extraction.provider,
            model=extraction.model,
            raw_text=extraction.raw_text or extraction.note_text,
            cleaned_text=ocr_result.cleaned_text or extraction.cleaned_text or extraction.note_text,
            processed_text=(
                ocr_result.final_text_for_rag
                or ocr_result.fuzzy_corrected_text
                or ocr_result.cleaned_text
                or extraction.processed_text
                or extraction.note_text
            ),
        )

    def _select_final_ocr_text(
        self,
        *,
        raw_text: str,
        ocr_result: Any,
        ocr_mode: str,
        extraction_confidence: str,
    ) -> str:
        processed_text = (
            ocr_result.final_text_for_rag
            or ocr_result.fuzzy_corrected_text
            or ocr_result.cleaned_text
            or raw_text
        )
        normalized_mode = str(ocr_mode or "").strip().lower()
        if normalized_mode != "handwritten":
            return processed_text

        raw_score = self._score_handwritten_ocr_text(raw_text)
        processed_score = self._score_handwritten_ocr_text(processed_text)
        if extraction_confidence in {"High", "Medium"} and raw_score >= processed_score:
            return raw_text
        if OCR_UNCERTAINTY_PATTERN.search(processed_text) and not OCR_UNCERTAINTY_PATTERN.search(raw_text):
            return raw_text
        return processed_text

    def _score_handwritten_ocr_text(self, text: str) -> float:
        value = str(text or "")
        if not value.strip():
            return -10.0
        lowered = value.lower()
        score = 0.0
        score += min(len(re.findall(r"\n", value)), 12) * 0.2
        score += len(re.findall(r"\b(?:cc|hpi|exam|dx|plan|o2|iv|cxr|sob|rll)\b", lowered)) * 0.8
        score += len(re.findall(r"\b\d+(?:\.\d+)?(?:%|/24|/yo)?\b", lowered)) * 0.25
        score -= len(OCR_UNCERTAINTY_PATTERN.findall(value)) * 2.2
        score -= len(re.findall(r"[{}\[\]|~`^*_<>\\]", value)) * 0.25
        score -= len(re.findall(r"\b[a-z]*\d+[a-z]+|[a-z]+\d+[a-z]*\b", lowered)) * 0.9
        return score

    def _vision_client_for_ocr_mode(self, ocr_mode: str) -> Any:
        normalized_mode = str(ocr_mode or "printed").strip().lower()
        if normalized_mode == "handwritten":
            return self.handwritten_vision_client
        if normalized_mode == "printed":
            return self.printed_vision_client
        raise RuntimeError("Invalid OCR mode. Use 'handwritten' or 'printed'.")

    def _predict_text(
        self,
        *,
        clinical_text: str,
        top_k: int | None = None,
        return_context: bool = False,
        return_similarity: bool = False,
    ) -> dict[str, Any]:
        effective_top_k = top_k or self.settings.retrieval_top_k
        normalized_text = clinical_text
        diagnosis_context = self._build_diagnosis_context(clinical_text)
        internal_candidate_pool = min(
            max(effective_top_k * 6, 18),
            max(len(getattr(self.retriever, "entries", [])), effective_top_k),
            30,
        )
        retrieved = self._retrieve_diagnosis_centered_candidates(
            clinical_text=clinical_text,
            diagnosis_context=diagnosis_context,
            candidate_pool_size=internal_candidate_pool,
        )
        if self._should_apply_ocr_preprocessing_to_text(clinical_text):
            try:
                ocr_result = self.ocr_preprocessor.process_ocr_text(clinical_text)
                processed_text = (
                    ocr_result.final_text_for_rag
                    or ocr_result.fuzzy_corrected_text
                    or ocr_result.cleaned_text
                    or clinical_text
                )
                if processed_text.strip() and processed_text.strip() != clinical_text.strip():
                    processed_context = self._build_diagnosis_context(processed_text)
                    processed_retrieved = self._retrieve_diagnosis_centered_candidates(
                        clinical_text=processed_text,
                        diagnosis_context=processed_context,
                        candidate_pool_size=internal_candidate_pool,
                    )
                    if self._score_retrieved_candidates(
                        clinical_text=clinical_text,
                        retrieved=processed_retrieved,
                    ) >= self._score_retrieved_candidates(
                        clinical_text=clinical_text,
                        retrieved=retrieved,
                    ):
                        normalized_text = processed_text
                        retrieved = processed_retrieved
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("OCR preprocessing failed, falling back to raw note text: %s", exc)
        retrieved = self._augment_retrieved_with_focus_catalog_matches(
            clinical_text=normalized_text,
            retrieved=retrieved,
            top_k=effective_top_k,
        )
        if not retrieved or max(item.similarity_score for item in retrieved) < self.settings.retrieval_min_score:
            return {"codes": [], "message": "Insufficient context"}
        retrieved = self._rerank_retrieved_candidates(
            clinical_text=normalized_text,
            retrieved=retrieved,
        )
        retrieved = self._filter_unsupported_specific_candidates(
            clinical_text=normalized_text,
            retrieved=retrieved,
            min_candidates=min(effective_top_k, 3),
        )
        retrieved = self._apply_generic_safety_filters(
            clinical_text=normalized_text,
            retrieved=retrieved,
            min_candidates=min(effective_top_k, 3),
        )

        payload = self._generate_payload(clinical_text=normalized_text, retrieved=retrieved)
        codes = self._post_process_predictions(
            payload=payload,
            retrieved=retrieved,
            return_similarity=return_similarity,
        )
        codes, priority_payload = self._prioritize_predictions_with_llm(
            clinical_text=normalized_text,
            codes=codes,
        )
        codes = self._validate_final_predictions(
            clinical_text=normalized_text,
            codes=codes,
            retrieved=retrieved,
        )
        if not codes:
            return {"codes": [], "message": "Insufficient context"}

        response: dict[str, Any] = {"codes": codes}
        if priority_payload:
            response.update(priority_payload)
        summary = self._build_summary(
            payload=payload,
            clinical_text=normalized_text,
            codes=codes,
        )
        if summary:
            response["summary"] = summary
        condition_overview = self._build_condition_overview(payload=payload, codes=codes)
        if condition_overview:
            response["condition_overview"] = condition_overview
        precautions = self._build_precautions(payload=payload, codes=codes)
        if precautions:
            response["precautions"] = precautions
        diet_advice = self._build_diet_advice(payload=payload, codes=codes)
        if diet_advice:
            response["diet_advice"] = diet_advice
        if return_context:
            response["retrieved_context"] = [
                {
                    "code": item.code,
                    "title": item.title,
                    "similarity_score": item.similarity_score,
                    "chunk_text": item.chunk_text,
                }
                for item in retrieved
            ]
        return response

    def _should_apply_ocr_preprocessing_to_text(self, clinical_text: str) -> bool:
        text = str(clinical_text or "")
        if not text.strip():
            return False
        lowered = text.lower()
        if not getattr(self.settings, "ocr_preprocessing_enabled", True):
            return False
        if any(marker in lowered for marker in NOISE_MARKERS):
            return True
        structured_header_count = len(STRUCTURED_NOTE_HEADER_PATTERN.findall(text))
        garbled_token_count = len(OCR_GARBLED_TOKEN_PATTERN.findall(text))
        suspicious_symbol_count = sum(character in "{}[]|~`^*_+=<>\\" for character in text)
        alpha_word_count = len(re.findall(r"[A-Za-z]{3,}", text))
        if alpha_word_count == 0:
            return False
        if structured_header_count >= 2 and garbled_token_count == 0 and suspicious_symbol_count <= 2:
            return False
        symbol_ratio = suspicious_symbol_count / max(alpha_word_count, 1)
        garbled_ratio = garbled_token_count / max(alpha_word_count, 1)
        return symbol_ratio > 0.12 or garbled_ratio > 0.08

    def _augment_retrieved_with_focus_catalog_matches(
        self,
        *,
        clinical_text: str,
        retrieved: list[RetrievedICDEntry],
        top_k: int,
    ) -> list[RetrievedICDEntry]:
        focus_fragments = self._focus_fragments(clinical_text)
        if not focus_fragments:
            return retrieved

        existing_codes = {item.code for item in retrieved}
        additions: list[tuple[float, RetrievedICDEntry]] = []
        for entry in self.retriever.entries:
            if entry.code in existing_codes:
                continue
            score = max(
                self._focus_candidate_support_score(fragment, entry.title, entry.keywords, clinical_text)
                for fragment in focus_fragments
            )
            if score < 0.78:
                continue
            additions.append(
                (
                    score,
                    RetrievedICDEntry(
                        code=entry.code,
                        title=entry.title,
                        keywords=entry.keywords,
                        definition=entry.definition,
                        explanation_hint=entry.explanation_hint,
                        chunk_text=entry.chunk_text,
                        similarity_score=round(max(0.42, min(score, 0.92)), 4),
                    ),
                )
            )

        additions.sort(key=lambda pair: (pair[0], pair[1].similarity_score), reverse=True)
        augmented = [item for _, item in additions[: max(1, top_k // 2)]]
        augmented.extend(retrieved)
        return self._deduplicate_retrieved(augmented)[: max(top_k, len(retrieved))]

    def _filter_unsupported_specific_candidates(
        self,
        *,
        clinical_text: str,
        retrieved: list[RetrievedICDEntry],
        min_candidates: int,
    ) -> list[RetrievedICDEntry]:
        if not retrieved:
            return retrieved
        query_tokens = _support_tokens(clinical_text)
        filtered: list[RetrievedICDEntry] = []
        deferred: list[tuple[int, int, RetrievedICDEntry]] = []
        for item in retrieved:
            title_tokens = _support_tokens(item.title)
            unsupported = title_tokens - query_tokens - GENERIC_TITLE_TERMS
            if len(unsupported) >= 2 and len(title_tokens & query_tokens) <= 2:
                deferred.append((len(unsupported), -len(title_tokens & query_tokens), item))
                continue
            filtered.append(item)

        if len(filtered) >= min_candidates:
            return filtered
        deferred.sort(key=lambda item: (item[0], item[1]))
        for _, _, item in deferred:
            if item not in filtered:
                filtered.append(item)
            if len(filtered) >= min_candidates:
                break
        return filtered

    def _deduplicate_retrieved(self, retrieved: list[RetrievedICDEntry]) -> list[RetrievedICDEntry]:
        deduplicated: list[RetrievedICDEntry] = []
        seen: set[str] = set()
        for item in retrieved:
            if item.code in seen:
                continue
            seen.add(item.code)
            deduplicated.append(item)
        return deduplicated

    def _focus_fragments(self, clinical_text: str) -> list[str]:
        fragments: list[str] = []
        for phrase in extract_focus_phrases(clinical_text, max_phrases=8):
            fragments.append(phrase)
            fragments.extend(FOCUS_SPLIT_PATTERN.split(phrase))

        deduplicated: list[str] = []
        seen: set[str] = set()
        for fragment in fragments:
            cleaned = " ".join(_support_tokens(fragment))
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduplicated.append(fragment)
        return deduplicated

    def _build_diagnosis_context(self, clinical_text: str) -> dict[str, Any]:
        focus_fragments = self._focus_fragments(clinical_text)
        note_text = str(clinical_text or "")
        lowered = note_text.lower()
        evidence_lines = extract_note_evidence(clinical_text, limit=6)
        explicit_diagnosis = any(term in lowered for term in DIAGNOSIS_EVIDENCE_TERMS)
        return {
            "diagnosis_fragments": focus_fragments,
            "primary_candidates": focus_fragments[:1],
            "secondary_candidates": focus_fragments[1:],
            "evidence": evidence_lines,
            "has_explicit_diagnosis": explicit_diagnosis,
            "has_trauma_context": bool(TRAUMA_CONTEXT_PATTERN.search(note_text)),
            "has_pregnancy_context": bool(PREGNANCY_CONTEXT_PATTERN.search(note_text)),
            "has_newborn_context": bool(NEWBORN_CONTEXT_PATTERN.search(note_text)),
            "note_tokens": _support_tokens(clinical_text),
        }

    def _retrieve_diagnosis_centered_candidates(
        self,
        *,
        clinical_text: str,
        diagnosis_context: dict[str, Any],
        candidate_pool_size: int,
    ) -> list[RetrievedICDEntry]:
        fragments = list(diagnosis_context.get("diagnosis_fragments", []))[:4]
        queries = [clinical_text, *fragments]
        deduplicated_queries: list[str] = []
        seen_queries: set[str] = set()
        for query in queries:
            normalized_query = " ".join(_support_tokens(query))
            if not normalized_query or normalized_query in seen_queries:
                continue
            seen_queries.add(normalized_query)
            deduplicated_queries.append(query)

        aggregated: dict[str, tuple[float, RetrievedICDEntry]] = {}
        note_query = deduplicated_queries[0] if deduplicated_queries else clinical_text
        fragment_queries = deduplicated_queries[1:]
        note_top_k = max(min(candidate_pool_size, 12), 5)
        fragment_top_k = max(min(max(candidate_pool_size // 2, 3), 6), 3)

        for item in self.retriever.retrieve(note_query, top_k=note_top_k, expand_queries=True):
            fragment_support = max(
                (
                    self._focus_candidate_support_score(fragment, item.title, item.keywords, clinical_text)
                    for fragment in fragments
                ),
                default=0.0,
            )
            adjusted_similarity = min(
                0.99,
                item.similarity_score + min(fragment_support * 0.18, 0.16),
            )
            enriched_item = RetrievedICDEntry(
                code=item.code,
                title=item.title,
                keywords=item.keywords,
                definition=item.definition,
                explanation_hint=item.explanation_hint,
                chunk_text=item.chunk_text,
                similarity_score=round(adjusted_similarity, 4),
            )
            aggregated[item.code] = (fragment_support, enriched_item)

        for query in fragment_queries:
            for item in self.retriever.retrieve(query, top_k=fragment_top_k, expand_queries=False):
                fragment_support = max(
                    (
                        self._focus_candidate_support_score(fragment, item.title, item.keywords, clinical_text)
                        for fragment in fragments
                    ),
                    default=0.0,
                )
                adjusted_similarity = min(
                    0.99,
                    item.similarity_score + min(fragment_support * 0.18, 0.16),
                )
                enriched_item = RetrievedICDEntry(
                    code=item.code,
                    title=item.title,
                    keywords=item.keywords,
                    definition=item.definition,
                    explanation_hint=item.explanation_hint,
                    chunk_text=item.chunk_text,
                    similarity_score=round(adjusted_similarity, 4),
                )
                existing = aggregated.get(item.code)
                if existing is None or (fragment_support, adjusted_similarity) > (
                    existing[0],
                    existing[1].similarity_score,
                ):
                    aggregated[item.code] = (fragment_support, enriched_item)

        ranked = sorted(
            aggregated.values(),
            key=lambda pair: (pair[0], pair[1].similarity_score),
            reverse=True,
        )
        return [item for _, item in ranked[:candidate_pool_size]]

    def _focus_candidate_support_score(
        self,
        focus_fragment: str,
        title: str,
        keywords: list[str],
        clinical_text: str,
    ) -> float:
        focus_tokens = _support_tokens(focus_fragment)
        if not focus_tokens:
            return 0.0
        query_tokens = _support_tokens(clinical_text)
        title_tokens = _support_tokens(title)
        keyword_tokens: set[str] = set()
        for keyword in keywords:
            keyword_tokens.update(_support_tokens(keyword))
        candidate_tokens = title_tokens | keyword_tokens
        overlap = focus_tokens & candidate_tokens
        if not overlap:
            return 0.0

        focus_recall = len(overlap) / len(focus_tokens)
        candidate_precision = len(overlap) / max(len(title_tokens), 1)
        unsupported_title_tokens = title_tokens - query_tokens - GENERIC_TITLE_TERMS - focus_tokens
        unsupported_penalty = min(len(unsupported_title_tokens) * 0.18, 0.75)
        exact_focus_bonus = 0.2 if focus_tokens <= title_tokens else 0.0
        return (0.85 * focus_recall) + (0.25 * candidate_precision) + exact_focus_bonus - unsupported_penalty

    def _apply_generic_safety_filters(
        self,
        *,
        clinical_text: str,
        retrieved: list[RetrievedICDEntry],
        min_candidates: int,
    ) -> list[RetrievedICDEntry]:
        if not retrieved:
            return []

        supported: list[RetrievedICDEntry] = []
        deferred: list[RetrievedICDEntry] = []
        for item in retrieved:
            if self._candidate_is_blocked_by_generic_safety(clinical_text=clinical_text, candidate=item):
                deferred.append(item)
                continue
            supported.append(item)

        if len(supported) >= min_candidates or not supported:
            return supported or deferred[:min_candidates]
        return supported + deferred[: max(min_candidates - len(supported), 0)]

    def _candidate_is_blocked_by_generic_safety(
        self,
        *,
        clinical_text: str,
        candidate: RetrievedICDEntry,
    ) -> bool:
        normalized_code = normalize_icd_code_key(candidate.code)
        title_lower = str(candidate.title or "").lower()
        note_text = str(clinical_text or "")
        note_lower = note_text.lower()
        has_trauma = bool(TRAUMA_CONTEXT_PATTERN.search(note_text))
        has_pregnancy = bool(PREGNANCY_CONTEXT_PATTERN.search(note_text))
        has_newborn = bool(NEWBORN_CONTEXT_PATTERN.search(note_text))

        if normalized_code.startswith(("S", "T")) and not has_trauma:
            return True
        if normalized_code.startswith("O") and not has_pregnancy:
            return True
        if normalized_code.startswith("P") and not has_newborn:
            return True
        if "injury" in title_lower and not has_trauma:
            return True
        if "pregnan" in title_lower and not has_pregnancy:
            return True
        if "newborn" in title_lower and not has_newborn:
            return True

        complication_terms = [
            term for term in GENERIC_COMPLICATION_TERMS
            if term in title_lower and term not in {"poisoning", "toxic"}
        ]
        if complication_terms and not any(term in note_lower for term in complication_terms):
            overlapping_core_terms = _support_tokens(candidate.title) & _support_tokens(clinical_text)
            if len(overlapping_core_terms) <= 2:
                return True

        if any(term in title_lower for term in ("poisoning", "toxic effect")) and not has_trauma:
            return True
        return False

    def _prioritize_predictions_with_llm(
        self,
        *,
        clinical_text: str,
        codes: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not codes:
            return codes, {}
        if len(codes) == 1:
            return codes, self._build_priority_metadata_from_codes(
                codes,
                reasoning="Single supported ICD candidate retained without an additional prioritization call.",
            )
        if not hasattr(self.llm, "prioritize_codes"):
            return codes, self._build_priority_metadata_from_codes(codes, reasoning="")

        try:
            priority = self.llm.prioritize_codes(
                clinical_note=clinical_text,
                candidate_codes=codes,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("LLM prioritization failed, retaining generated code order: %s", exc)
            return codes, self._build_priority_metadata_from_codes(codes, reasoning="")
        if not isinstance(priority, dict):
            return codes, self._build_priority_metadata_from_codes(codes, reasoning="")

        prioritized_codes, metadata = self._apply_priority_payload(
            codes=codes,
            priority=priority,
        )
        return prioritized_codes, metadata

    def _apply_priority_payload(
        self,
        *,
        codes: list[dict[str, Any]],
        priority: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        codes_by_key = {
            normalize_icd_code_key(item["code"]): item
            for item in codes
            if normalize_icd_code_key(item.get("code", ""))
        }
        original_keys = [normalize_icd_code_key(item["code"]) for item in codes]

        primary_key = normalize_icd_code_key(str(priority.get("primary_icd") or ""))
        raw_secondary_codes = priority.get("secondary_icd", [])
        if not isinstance(raw_secondary_codes, list):
            raw_secondary_codes = []
        secondary_keys = [
            normalize_icd_code_key(str(code))
            for code in raw_secondary_codes
            if normalize_icd_code_key(str(code))
        ]
        raw_dropped_codes = priority.get("dropped_codes", [])
        if not isinstance(raw_dropped_codes, list):
            raw_dropped_codes = []
        dropped_keys = {
            normalize_icd_code_key(str(code))
            for code in raw_dropped_codes
            if normalize_icd_code_key(str(code))
        }

        ordered_keys: list[str] = []
        if primary_key in codes_by_key:
            ordered_keys.append(primary_key)
        for key in secondary_keys:
            if key in codes_by_key and key not in ordered_keys and key not in dropped_keys:
                ordered_keys.append(key)

        if not ordered_keys:
            ordered_keys = [key for key in original_keys if key and key not in dropped_keys]
        if not ordered_keys:
            ordered_keys = original_keys

        prioritized_codes = [codes_by_key[key] for key in ordered_keys if key in codes_by_key]
        primary_code = prioritized_codes[0]["code"] if prioritized_codes else ""
        secondary_codes = [item["code"] for item in prioritized_codes[1:]]
        retained_keys = {normalize_icd_code_key(code) for code in [primary_code, *secondary_codes]}
        dropped_codes = [
            codes_by_key[key]["code"]
            for key in original_keys
            if key in codes_by_key and key not in retained_keys
        ]
        reasoning = str(priority.get("reasoning") or "").strip()

        metadata = {
            "primary_icd": primary_code,
            "secondary_icd": secondary_codes,
            "dropped_codes": dropped_codes,
            "prioritization_reasoning": reasoning,
        }
        return prioritized_codes, metadata

    def _build_priority_metadata_from_codes(
        self,
        codes: list[dict[str, Any]],
        *,
        reasoning: str,
    ) -> dict[str, Any]:
        if not codes:
            return {}
        return {
            "primary_icd": codes[0]["code"],
            "secondary_icd": [item["code"] for item in codes[1:]],
            "dropped_codes": [],
            "prioritization_reasoning": reasoning,
        }

    def _validate_final_predictions(
        self,
        *,
        clinical_text: str,
        codes: list[dict[str, Any]],
        retrieved: list[RetrievedICDEntry],
    ) -> list[dict[str, Any]]:
        if not codes:
            return []

        retrieved_by_key = {
            normalize_icd_code_key(item.code): item
            for item in retrieved
            if normalize_icd_code_key(item.code)
        }
        validated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in codes:
            normalized_code = normalize_icd_code_key(item.get("code", ""))
            if not normalized_code or normalized_code in seen:
                continue
            retrieved_item = retrieved_by_key.get(normalized_code)
            if retrieved_item is None:
                continue
            if self._candidate_is_blocked_by_generic_safety(
                clinical_text=clinical_text,
                candidate=retrieved_item,
            ):
                continue
            validated.append(item)
            seen.add(normalized_code)
        return validated

    def _score_retrieved_candidates(
        self,
        *,
        clinical_text: str,
        retrieved: list[RetrievedICDEntry],
    ) -> float:
        if not retrieved:
            return 0.0

        focus_phrases = extract_focus_phrases(clinical_text)
        score = 0.0
        for rank, item in enumerate(retrieved[:3], start=1):
            weight = 1.0 if rank == 1 else 0.35 if rank == 2 else 0.2
            record = {
                "code": item.code,
                "title": item.title,
                "keywords": item.keywords,
                "definition": item.definition,
                "explanation_hint": item.explanation_hint,
                "chunk_text": item.chunk_text,
            }
            score += weight * (
                item.similarity_score
                + lexical_relevance_score(clinical_text, record)
                + diagnosis_focus_score(clinical_text, focus_phrases, record)
            )
        for rank, item in enumerate(
            score_semantic_candidates(
                clinical_text,
                retrieved[:3],
                similarity_weight=getattr(self.settings, "semantic_similarity_weight", 0.72),
                feature_weight=getattr(self.settings, "semantic_feature_weight", 0.28),
            ),
            start=1,
        ):
            weight = 0.25 if rank == 1 else 0.12 if rank == 2 else 0.07
            score += weight * item.feature_alignment_score
        return score

    def _rerank_retrieved_candidates(
        self,
        *,
        clinical_text: str,
        retrieved: list[RetrievedICDEntry],
    ) -> list[RetrievedICDEntry]:
        if not getattr(self.settings, "semantic_reranking_enabled", True):
            return retrieved
        return rerank_retrieved_candidates(
            clinical_text,
            retrieved,
            similarity_weight=getattr(self.settings, "semantic_similarity_weight", 0.72),
            feature_weight=getattr(self.settings, "semantic_feature_weight", 0.28),
        )

    def _align_predictions_with_focus_diagnosis(
        self,
        *,
        clinical_text: str,
        codes: list[dict[str, Any]],
        retrieved: list[RetrievedICDEntry],
        return_similarity: bool,
    ) -> list[dict[str, Any]]:
        focus_fallback_codes = self._build_focus_fallback_predictions(
            clinical_text=clinical_text,
            retrieved=retrieved,
            return_similarity=return_similarity,
        )
        if not focus_fallback_codes:
            return codes
        if not codes:
            return focus_fallback_codes

        retrieved_by_code = {item.code: item for item in retrieved}
        focus_phrases = extract_focus_phrases(clinical_text)
        top_predicted = codes[0]
        predicted_entry = retrieved_by_code.get(top_predicted["code"])
        predicted_focus_score = 0.0
        if predicted_entry is not None:
            predicted_focus_score = diagnosis_focus_score(
                clinical_text,
                focus_phrases,
                {
                    "code": predicted_entry.code,
                    "title": predicted_entry.title,
                    "keywords": predicted_entry.keywords,
                    "definition": predicted_entry.definition,
                    "explanation_hint": predicted_entry.explanation_hint,
                    "chunk_text": predicted_entry.chunk_text,
                },
            )

        preferred_focus = focus_fallback_codes[0]
        if preferred_focus["code"] == top_predicted["code"]:
            return codes
        if predicted_focus_score >= 0.6:
            return codes

        aligned: list[dict[str, Any]] = [preferred_focus]
        seen_codes = {preferred_focus["code"]}
        for item in codes:
            if item["code"] in seen_codes:
                continue
            aligned.append(item)
            seen_codes.add(item["code"])
        return aligned

    def _build_focus_fallback_predictions(
        self,
        *,
        clinical_text: str,
        retrieved: list[RetrievedICDEntry],
        return_similarity: bool,
    ) -> list[dict[str, Any]]:
        focus_phrases = extract_focus_phrases(clinical_text)
        if not focus_phrases:
            return []

        scored_entries: list[tuple[float, RetrievedICDEntry]] = []
        for item in retrieved:
            focus_score = diagnosis_focus_score(
                clinical_text,
                focus_phrases,
                {
                    "code": item.code,
                    "title": item.title,
                    "keywords": item.keywords,
                    "definition": item.definition,
                    "explanation_hint": item.explanation_hint,
                    "chunk_text": item.chunk_text,
                },
            )
            if focus_score < 0.8:
                continue
            scored_entries.append((focus_score, item))

        if not scored_entries:
            return []

        scored_entries.sort(
            key=lambda pair: (pair[0], pair[1].similarity_score),
            reverse=True,
        )

        predictions: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for focus_score, item in scored_entries[:2]:
            title_key = item.title.strip().lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            explanation = (
                "The clinical note contains an explicit diagnosis-focused section that aligns with this "
                f"retrieved ICD context. Definition: {item.definition or 'No definition available.'}"
            )
            prediction: dict[str, Any] = {
                "code": item.code,
                "title": item.title,
                "explanation": explanation,
                "confidence": coerce_confidence("High", item.similarity_score + (focus_score * 0.1)),
            }
            if return_similarity:
                prediction["similarity_score"] = item.similarity_score
            predictions.append(prediction)
        return predictions

    def health(self) -> dict[str, Any]:
        provider_name = self.llm.__class__.__name__.replace("LLMClient", "")
        model_name = getattr(self.llm, "model", getattr(self.llm, "model_name", "heuristic"))
        return {
            "status": "ok",
            "catalog_size": len(self.retriever.entries),
            "llm_provider": provider_name,
            "llm_model": str(model_name),
        }

    def lookup_code(self, raw_code: str, max_suggestions: int = 5) -> dict[str, Any]:
        normalized_code = raw_code.strip().upper()
        catalog_code, entry = self._lookup_catalog_code(normalized_code)
        if entry is not None:
            return {
                "raw_code": raw_code,
                "normalized_code": catalog_code,
                "found": True,
                "title": entry.title,
                "definition": entry.definition,
                "explanation_hint": entry.explanation_hint,
                "keywords": entry.keywords,
                "suggestions": [],
                "catalog_source": self.retriever.catalog_path.name,
            }

        suggestions: list[dict[str, str]] = []
        for code, entry in self._lookup_suggestions(normalized_code, max_suggestions=max_suggestions):
            suggestions.append({"code": code, "title": entry.title})

        return {
            "raw_code": raw_code,
            "normalized_code": normalized_code,
            "found": False,
            "title": "",
            "definition": "",
            "explanation_hint": "",
            "keywords": [],
            "suggestions": suggestions,
            "catalog_source": self.retriever.catalog_path.name,
        }

    def lookup_codes(self, raw_codes: list[str], max_suggestions: int = 5) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for raw_code in raw_codes:
            text = str(raw_code or "").strip()
            if not text:
                continue
            results.append(self.lookup_code(text, max_suggestions=max_suggestions))
        return results

    def _generate_payload(
        self,
        *,
        clinical_text: str,
        retrieved: list[RetrievedICDEntry],
    ) -> dict[str, Any]:
        try:
            return self.llm.generate_codes(
                clinical_note=clinical_text,
                retrieved_entries=retrieved,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Primary LLM generation failed, falling back to heuristic mode: %s", exc)
            return self.fallback_llm.generate_codes(
                clinical_note=clinical_text,
                retrieved_entries=retrieved,
            )

    def _post_process_predictions(
        self,
        *,
        payload: dict[str, Any],
        retrieved: list[RetrievedICDEntry],
        return_similarity: bool,
    ) -> list[dict[str, Any]]:
        raw_codes = payload.get("codes", [])
        if not isinstance(raw_codes, list):
            return []

        allowed_codes = {item.code: item for item in retrieved}
        allowed_codes_by_key = {
            normalize_icd_code_key(item.code): item
            for item in retrieved
            if normalize_icd_code_key(item.code)
        }
        predictions: list[dict[str, Any]] = []
        seen_prediction_codes: set[str] = set()
        for item in deduplicate_codes(raw_codes):
            code = item["code"]
            retrieved_entry = allowed_codes.get(code) or allowed_codes_by_key.get(
                normalize_icd_code_key(code)
            )
            if retrieved_entry is None:
                continue
            if retrieved_entry.code in seen_prediction_codes:
                continue
            seen_prediction_codes.add(retrieved_entry.code)
            title = str(item.get("title") or retrieved_entry.title).strip()
            explanation = str(item.get("explanation") or "").strip()
            if not explanation:
                explanation = (
                    f"Retrieved context supports {retrieved_entry.title}. "
                    f"Definition: {retrieved_entry.definition or 'No definition available.'}"
                )
            prediction: dict[str, Any] = {
                "code": retrieved_entry.code,
                "title": title,
                "explanation": explanation,
                "confidence": coerce_confidence(
                    str(item.get("confidence", "")),
                    retrieved_entry.similarity_score,
                ),
            }
            if return_similarity:
                prediction["similarity_score"] = retrieved_entry.similarity_score
            predictions.append(prediction)
        return predictions

    def _build_summary(
        self,
        *,
        payload: dict[str, Any],
        clinical_text: str,
        codes: list[dict[str, Any]],
    ) -> str | None:
        raw_summary = str(payload.get("summary") or "").strip()
        if raw_summary:
            return raw_summary
        if not codes:
            return None

        top_code = codes[0]
        summary_parts = [
            f"The note most closely matches {top_code['title']} ({top_code['code']}).",
        ]

        evidence = extract_note_evidence(clinical_text, limit=1)
        if evidence:
            summary_parts.append(f"Key note evidence: {evidence[0]}")

        if len(codes) > 1:
            alternatives = ", ".join(
                f"{item['code']} ({item['title']})" for item in codes[1:3]
            )
            if alternatives:
                summary_parts.append(f"Other supported matches to review: {alternatives}.")

        return " ".join(summary_parts)

    def _build_condition_overview(
        self,
        *,
        payload: dict[str, Any],
        codes: list[dict[str, Any]],
    ) -> str | None:
        raw_value = str(payload.get("condition_overview") or "").strip()
        if raw_value:
            return raw_value
        if not codes:
            return None
        top_code = codes[0]
        return (
            f"{top_code['title']} is the condition most strongly supported by the note and retrieved catalog context. "
            "Use the detailed code explanations below for the coding rationale."
        )

    def _build_precautions(
        self,
        *,
        payload: dict[str, Any],
        codes: list[dict[str, Any]],
    ) -> list[str]:
        raw_items = coerce_text_list(payload.get("precautions"))
        if raw_items:
            return raw_items
        if not codes:
            return []
        return [
            "Follow prescribed treatment and keep symptoms under review.",
            "Seek medical help quickly if severe pain, breathing trouble, fainting, confusion, or rapid worsening occurs.",
            "Arrange follow-up review if the condition does not improve as expected.",
        ]

    def _build_diet_advice(
        self,
        *,
        payload: dict[str, Any],
        codes: list[dict[str, Any]],
    ) -> list[str]:
        raw_items = coerce_text_list(payload.get("diet_advice"))
        if raw_items:
            return raw_items
        if not codes:
            return []
        return [
            "Choose balanced meals with vegetables, fruit, protein, and fewer heavily processed foods.",
            "Stay hydrated unless the care team has advised a fluid restriction.",
            "Support recovery with regular sleep, movement as tolerated, and avoidance of smoking or excess alcohol.",
        ]

    def _lookup_suggestions(self, normalized_code: str, max_suggestions: int) -> list[tuple[str, Any]]:
        results: list[tuple[str, Any]] = []
        search_key = normalize_icd_code_key(normalized_code)
        for code, entry in sorted(self.retriever.catalog_by_code.items()):
            catalog_key = normalize_icd_code_key(code)
            if search_key and catalog_key.startswith(search_key):
                results.append((code, entry))
            elif search_key and search_key in catalog_key:
                results.append((code, entry))
            if len(results) >= max_suggestions:
                break
        return results

    def _lookup_catalog_code(self, normalized_code: str) -> tuple[str, Any | None]:
        direct_entry = self.retriever.catalog_by_code.get(normalized_code)
        if direct_entry is not None:
            return normalized_code, direct_entry

        search_key = normalize_icd_code_key(normalized_code)
        if not search_key:
            return normalized_code, None

        for code, entry in self.retriever.catalog_by_code.items():
            if normalize_icd_code_key(code) == search_key:
                return code, entry
        return normalized_code, None


def _support_tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(str(text or "").lower())
        if len(token) > 2 and token not in TITLE_STOPWORDS
    }
