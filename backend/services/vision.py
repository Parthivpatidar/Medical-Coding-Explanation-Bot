from __future__ import annotations

import base64
import logging
import mimetypes
import os
from typing import Any, Callable

from backend.core.utils import coerce_confidence, coerce_text_list, safe_json_loads
from backend.services.local_vision import LocalTrOCRVisionClient
from backend.services.vision_prompt import (
    VISION_EXTRACTION_SYSTEM_PROMPT,
    build_vision_extraction_instruction,
    build_vision_text_format,
)
from backend.services.vision_types import VisionExtractionResult
from backend.services.windows_ocr import (
    confidence_from_ocr_score,
    extract_text_from_image_bytes as extract_text_with_windows_ocr,
    score_ocr_candidate,
)


LOGGER = logging.getLogger("medichar.rag.vision")


class BaseVisionClient:
    def extract_note_from_image(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        supplemental_text: str = "",
    ) -> VisionExtractionResult:
        raise NotImplementedError


class DisabledVisionClient(BaseVisionClient):
    def extract_note_from_image(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        supplemental_text: str = "",
    ) -> VisionExtractionResult:
        del image_bytes, filename, supplemental_text
        raise RuntimeError(
            "Vision note extraction is disabled. Configure MEDICHAR_VISION_PROVIDER=local with local OCR "
            "dependencies, or MEDICHAR_VISION_PROVIDER=openai with OPENAI_API_KEY."
        )


class WindowsOCRVisionClient(BaseVisionClient):
    def __init__(
        self,
        *,
        ocr_mode: str = "auto",
        extractor: Callable[..., tuple[str | None, str | None]] | None = None,
    ) -> None:
        self.ocr_mode = ocr_mode
        self.extractor = extractor or extract_text_with_windows_ocr

    def extract_note_from_image(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        supplemental_text: str = "",
    ) -> VisionExtractionResult:
        del supplemental_text
        if not image_bytes:
            raise RuntimeError("Uploaded image is empty.")

        extracted_text, error = self.extractor(
            image_bytes,
            filename=filename,
            ocr_mode=self.ocr_mode,
        )
        if not extracted_text:
            raise RuntimeError(error or "Windows OCR did not return readable text.")

        score = score_ocr_candidate(extracted_text, self.ocr_mode)
        return VisionExtractionResult(
            note_text=_normalize_note_text(extracted_text),
            confidence=confidence_from_ocr_score(score),
            uncertain_fragments=[],
            provider="WindowsOCR",
            model="WinRT-OCR",
            raw_text=_normalize_note_text(extracted_text),
            cleaned_text=_normalize_note_text(extracted_text),
            processed_text=_normalize_note_text(extracted_text),
        )


class HybridLocalVisionClient(BaseVisionClient):
    def __init__(
        self,
        *,
        fast_client: BaseVisionClient,
        fallback_client: BaseVisionClient,
        fast_accept_score: float = 520.0,
        fast_skip_fallback_score: float = 300.0,
        score_fn: Callable[[VisionExtractionResult], float] | None = None,
    ) -> None:
        self.fast_client = fast_client
        self.fallback_client = fallback_client
        self.fast_accept_score = fast_accept_score
        self.fast_skip_fallback_score = fast_skip_fallback_score
        self.score_fn = score_fn or (lambda result: score_ocr_candidate(result.note_text, "auto"))

    def extract_note_from_image(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        supplemental_text: str = "",
    ) -> VisionExtractionResult:
        fast_result: VisionExtractionResult | None = None
        fast_error: str | None = None

        try:
            fast_result = self.fast_client.extract_note_from_image(
                image_bytes=image_bytes,
                filename=filename,
                supplemental_text=supplemental_text,
            )
            fast_score = self.score_fn(fast_result)
            LOGGER.info(
                "Fast OCR candidate %s/%s scored %.1f",
                fast_result.provider,
                fast_result.model,
                fast_score,
            )
            if fast_score >= self.fast_accept_score or (
                fast_result.confidence != "Low" and fast_score >= self.fast_skip_fallback_score
            ):
                return fast_result
        except RuntimeError as exc:
            fast_error = str(exc)
            fast_score = 0.0

        try:
            fallback_result = self.fallback_client.extract_note_from_image(
                image_bytes=image_bytes,
                filename=filename,
                supplemental_text=supplemental_text,
            )
        except RuntimeError as exc:
            if fast_result is not None:
                LOGGER.warning(
                    "Falling back to fast OCR result after handwriting OCR failed: %s",
                    exc,
                )
                return fast_result
            raise RuntimeError(fast_error or str(exc)) from exc

        fallback_score = self.score_fn(fallback_result)
        LOGGER.info(
            "Fallback OCR candidate %s/%s scored %.1f",
            fallback_result.provider,
            fallback_result.model,
            fallback_score,
        )
        if fast_result is None:
            return fallback_result
        if (
            fast_result.confidence != "Low"
            and fallback_result.confidence == "Low"
            and fast_score >= 300.0
            and fast_score >= (fallback_score * 0.6)
        ):
            return fast_result
        if fallback_score > fast_score:
            return fallback_result
        return fast_result


class OpenAIVisionClient(BaseVisionClient):
    def __init__(
        self,
        *,
        api_key: str,
        responses_url: str,
        model: str,
        detail: str,
        timeout_seconds: float,
        requests_module: Any | None = None,
    ) -> None:
        if requests_module is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "requests is not installed. Run `pip install -r requirements.txt` before using "
                    "the OpenAI vision provider."
                ) from exc
            requests_module = requests

        self.requests = requests_module
        self.api_key = api_key
        self.responses_url = responses_url
        self.model = model
        self.detail = detail
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI vision provider.")

    def extract_note_from_image(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        supplemental_text: str = "",
    ) -> VisionExtractionResult:
        if not image_bytes:
            raise RuntimeError("Uploaded image is empty.")

        data_url = _build_image_data_url(image_bytes=image_bytes, filename=filename)
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": VISION_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_vision_extraction_instruction(
                                supplemental_text=supplemental_text
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": data_url,
                            "detail": self.detail,
                        },
                    ],
                },
            ],
            "text": build_vision_text_format(),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.requests.post(
            self.responses_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        response_payload = response.json()

        raw_text = str(response_payload.get("output_text") or "").strip()
        if not raw_text:
            raw_text = _extract_output_text(response_payload.get("output"))

        parsed = safe_json_loads(raw_text)
        if parsed:
            note_text = _normalize_note_text(str(parsed.get("note_text") or ""))
            confidence = coerce_confidence(str(parsed.get("confidence") or ""))
            uncertain_fragments = coerce_text_list(parsed.get("uncertain_fragments"), max_items=8)
        else:
            note_text = _normalize_note_text(raw_text)
            confidence = "Low"
            uncertain_fragments = []

        if not note_text:
            raise RuntimeError("Vision extraction returned an empty note.")

        LOGGER.info("Vision-extracted clinical note: %s", note_text)
        return VisionExtractionResult(
            note_text=note_text,
            confidence=confidence,
            uncertain_fragments=uncertain_fragments,
            provider="OpenAI",
            model=self.model,
            raw_text=note_text,
            cleaned_text=note_text,
            processed_text=note_text,
        )


def build_vision_client(
    *,
    provider: str,
    openai_api_key: str,
    openai_responses_url: str,
    model: str,
    detail: str,
    timeout_seconds: float,
    device: str = "auto",
) -> BaseVisionClient:
    normalized_provider = provider.strip().lower()
    if normalized_provider == "auto":
        normalized_provider = "local"

    if normalized_provider == "local":
        if os.name == "nt":
            return HybridLocalVisionClient(
                fast_client=WindowsOCRVisionClient(),
                fallback_client=LocalTrOCRVisionClient(
                    model_name=model,
                    device=device,
                ),
            )
        return LocalTrOCRVisionClient(
            model_name=model,
            device=device,
        )
    if normalized_provider == "openai":
        openai_model = model
        if "/" in openai_model or "trocr" in openai_model.lower():
            openai_model = "gpt-4.1"
        return OpenAIVisionClient(
            api_key=openai_api_key,
            responses_url=openai_responses_url,
            model=openai_model,
            detail=detail,
            timeout_seconds=timeout_seconds,
        )
    return DisabledVisionClient()


def _build_image_data_url(*, image_bytes: bytes, filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename or "")
    normalized_mime = mime_type or "image/png"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{normalized_mime};base64,{encoded}"


def _extract_output_text(outputs: Any) -> str:
    if not isinstance(outputs, list):
        return ""

    parts: list[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            text = str(block.get("text") or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _normalize_note_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines())
    while "\n\n\n" in normalized:
        normalized = normalized.replace("\n\n\n", "\n\n")
    return normalized.strip()
