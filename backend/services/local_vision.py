from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from backend.services.vision_types import VisionExtractionResult


LOGGER = logging.getLogger("medichar.rag.vision.local")
SUSPICIOUS_TOKEN_PATTERN = re.compile(r"[^a-zA-Z0-9:/%.,()\-+ ]")
HANDWRITTEN_SECTION_LABEL_PATTERN = re.compile(r"^(?:cc|hpi|exam|dx|plan)\b[:.]?", re.IGNORECASE)


@dataclass(slots=True)
class LineRegion:
    image: Image.Image
    top: int
    bottom: int
    left: int
    right: int


class LocalTrOCRVisionClient:
    def __init__(
        self,
        *,
        model_name: str,
        device: str = "auto",
        batch_size: int = 4,
        processor: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device_name = device
        self.batch_size = max(1, int(batch_size))
        self._processor = processor
        self._model = model
        self._torch = torch_module

    def extract_note_from_image(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        supplemental_text: str = "",
    ) -> VisionExtractionResult:
        del filename, supplemental_text
        if not image_bytes:
            raise RuntimeError("Uploaded image is empty.")

        regions = detect_line_regions(image_bytes)
        if not regions:
            raise RuntimeError("No readable text lines were detected in the uploaded image.")

        recognized = self._recognize_regions(regions)
        note_text = assemble_note_text(recognized)
        if not note_text:
            raise RuntimeError("Local handwritten OCR could not read the uploaded note.")

        uncertain_fragments = [item["text"] for item in recognized if item["uncertain"] and item["text"]]
        confidence = summarize_confidence(recognized)
        LOGGER.info("Locally extracted clinical note: %s", note_text)
        return VisionExtractionResult(
            note_text=note_text,
            confidence=confidence,
            uncertain_fragments=uncertain_fragments[:8],
            provider="LocalTrOCR",
            model=self.model_name,
            raw_text=note_text,
            cleaned_text=note_text,
            processed_text=note_text,
        )

    def _ensure_runtime(self) -> tuple[Any, Any, Any, str]:
        torch_module = self._torch
        if torch_module is None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyTorch is required for local handwritten OCR. Install a CPU or CUDA build of torch."
                ) from exc
            torch_module = torch
            self._torch = torch_module

        if self._processor is None or self._model is None:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "transformers is required for local handwritten OCR. Install it with the project dependencies."
                ) from exc

            LOGGER.info("Loading local handwritten OCR model: %s", self.model_name)
            self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        resolved_device = resolve_torch_device(self.device_name, torch_module)
        self._model.to(resolved_device)
        if resolved_device == "cuda":
            try:
                self._model.half()
            except Exception:  # pragma: no cover
                LOGGER.debug("Local OCR model did not accept half precision; continuing with default precision.")
        self._model.eval()
        return torch_module, self._processor, self._model, resolved_device

    def _recognize_regions(self, regions: list[LineRegion]) -> list[dict[str, Any]]:
        torch_module, processor, model, device = self._ensure_runtime()
        prepared_images = [prepare_region_image(region.image) for region in regions]

        recognized: list[dict[str, Any]] = []
        with torch_module.inference_mode():
            for start in range(0, len(prepared_images), self.batch_size):
                batch_images = prepared_images[start : start + self.batch_size]
                encoded = processor(images=batch_images, return_tensors="pt")
                pixel_values = encoded.pixel_values.to(device)
                generation = model.generate(
                    pixel_values,
                    max_new_tokens=72,
                    num_beams=3,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                texts = processor.batch_decode(generation.sequences, skip_special_tokens=True)
                raw_scores = getattr(generation, "sequences_scores", None)
                if raw_scores is not None:
                    scores = [float(value) for value in raw_scores.detach().cpu().tolist()]
                else:
                    scores = [None] * len(texts)

                for offset, text in enumerate(texts):
                    region = regions[start + offset]
                    normalized_text = normalize_local_ocr_text(text)
                    score = scores[offset]
                    recognized.append(
                        {
                            "text": normalized_text,
                            "score": score,
                            "uncertain": is_uncertain_line(
                                normalized_text,
                                score,
                                region_width=region.right - region.left,
                                region_height=region.bottom - region.top,
                            ),
                            "top": region.top,
                            "bottom": region.bottom,
                            "left": region.left,
                            "right": region.right,
                        }
                    )
        return recognized


def detect_line_regions(image_bytes: bytes) -> list[LineRegion]:
    if cv2 is None:
        raise RuntimeError(
            "opencv-python is required for local image extraction. Install it with the project dependencies."
        )
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return []

    image = downscale_large_image(image)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
    normalized = cv2.normalize(grayscale, None, 0, 255, cv2.NORM_MINMAX)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    text_mask = suppress_notebook_lines(binary)
    text_mask = cv2.morphologyEx(
        text_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    text_mask = cv2.dilate(
        text_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1,
    )

    row_projection = np.count_nonzero(text_mask, axis=1)
    row_threshold = max(8, int(image.shape[1] * 0.01))
    raw_bands = extract_projection_bands(row_projection, row_threshold)
    if not raw_bands:
        return []

    merged_bands = merge_close_bands(raw_bands, max_gap=max(6, image.shape[0] // 120))
    refined_bands = refine_projection_bands(
        merged_bands,
        row_projection,
        base_threshold=row_threshold,
        image_height=image.shape[0],
    )
    pil_grayscale = Image.fromarray(grayscale)
    regions: list[LineRegion] = []

    for top, bottom in refined_bands:
        band = text_mask[top:bottom, :]
        if band.size == 0:
            continue
        band_density = float(np.count_nonzero(band)) / float(band.size)
        if band_density < 0.1:
            continue
        cols = np.where(np.count_nonzero(band, axis=0) > 0)[0]
        if cols.size == 0:
            continue
        left = max(int(cols[0]) - 28, 0)
        right = min(int(cols[-1]) + 28, image.shape[1] - 1)
        height = bottom - top
        if height < max(12, image.shape[0] // 120):
            continue
        padded_top = max(top - 10, 0)
        padded_bottom = min(bottom + 10, image.shape[0])
        crop = pil_grayscale.crop((left, padded_top, right + 1, padded_bottom))
        regions.append(
            LineRegion(
                image=crop,
                top=padded_top,
                bottom=padded_bottom,
                left=left,
                right=right,
            )
        )

    return regions


def downscale_large_image(image: np.ndarray, max_width: int = 2200) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / float(width)
    resized_height = max(int(height * scale), 1)
    return cv2.resize(image, (max_width, resized_height), interpolation=cv2.INTER_AREA)


def suppress_notebook_lines(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 8), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, height // 4)))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    cleaned = cv2.subtract(binary, horizontal_lines)
    cleaned = cv2.subtract(cleaned, vertical_lines)
    return cleaned


def extract_projection_bands(values: np.ndarray, threshold: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist()):
        if value >= threshold and start is None:
            start = index
        elif value < threshold and start is not None:
            bands.append((start, index))
            start = None
    if start is not None:
        bands.append((start, len(values)))
    return bands


def merge_close_bands(bands: list[tuple[int, int]], *, max_gap: int) -> list[tuple[int, int]]:
    if not bands:
        return []

    merged: list[tuple[int, int]] = [bands[0]]
    for top, bottom in bands[1:]:
        prev_top, prev_bottom = merged[-1]
        if top - prev_bottom <= max_gap:
            merged[-1] = (prev_top, bottom)
        else:
            merged.append((top, bottom))
    return merged


def refine_projection_bands(
    bands: list[tuple[int, int]],
    row_projection: np.ndarray,
    *,
    base_threshold: int,
    image_height: int,
    local_peak_ratio: float = 0.12,
) -> list[tuple[int, int]]:
    min_band_height = max(8, image_height // 140)
    refined: list[tuple[int, int]] = []

    for top, bottom in bands:
        band_projection = row_projection[top:bottom]
        if band_projection.size == 0:
            continue

        local_peak = int(band_projection.max())
        local_threshold = max(base_threshold, int(local_peak * local_peak_ratio))
        local_bands = extract_projection_bands(band_projection, local_threshold)

        if not local_bands:
            if bottom - top >= min_band_height:
                refined.append((top, bottom))
            continue

        kept_local_bands = [
            (top + local_top, top + local_bottom)
            for local_top, local_bottom in local_bands
            if local_bottom - local_top >= min_band_height
        ]
        if kept_local_bands:
            refined.extend(kept_local_bands)
        elif bottom - top >= min_band_height:
            refined.append((top, bottom))

    return refined


def prepare_region_image(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    autocontrast = ImageOps.autocontrast(grayscale)
    sharpened = autocontrast.filter(ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=2))
    enhanced = ImageOps.autocontrast(sharpened)
    width, height = enhanced.size
    scale = max(2, int(math.ceil(80 / max(height, 1))))
    resampling = getattr(Image, "Resampling", Image)
    resized = enhanced.resize((width * scale, height * scale), resample=resampling.LANCZOS)
    # TrOCR expects channel-aware images; keep OCR contrast work in grayscale, then
    # convert back to RGB so Hugging Face does not reject a 2D image tensor.
    return ImageOps.expand(resized, border=12, fill=255).convert("RGB")


def normalize_local_ocr_text(text: str) -> str:
    normalized = str(text or "").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:%)])", r"\1", normalized)
    normalized = re.sub(r"([(])\s+", r"\1", normalized)
    if HANDWRITTEN_SECTION_LABEL_PATTERN.match(normalized):
        normalized = re.sub(r"^(cc|hpi|exam|dx|plan)\b", lambda m: m.group(1).upper(), normalized, flags=re.IGNORECASE)
    return normalized.strip()


def is_uncertain_line(
    text: str,
    score: float | None,
    *,
    region_width: int,
    region_height: int,
) -> bool:
    if not text:
        return True
    if score is not None and score < -1.25:
        return True
    alpha_count = sum(character.isalpha() for character in text)
    if alpha_count <= 1 and region_width > 120:
        return True
    if len(text) <= 2 and region_height > 18:
        return True
    if SUSPICIOUS_TOKEN_PATTERN.search(text):
        return True
    if sum(character in "?[]" for character in text) >= 2:
        return True
    return False


def summarize_confidence(recognized_lines: list[dict[str, Any]]) -> str:
    if not recognized_lines:
        return "Low"
    uncertain_count = sum(1 for item in recognized_lines if item["uncertain"])
    scores = [item["score"] for item in recognized_lines if item["score"] is not None]
    average_score = sum(scores) / len(scores) if scores else None

    if uncertain_count == 0 and (average_score is None or average_score >= -0.65):
        return "High"
    if uncertain_count <= max(1, len(recognized_lines) // 4) and (
        average_score is None or average_score >= -1.1
    ):
        return "Medium"
    return "Low"


def assemble_note_text(recognized_lines: list[dict[str, Any]]) -> str:
    lines = [item for item in recognized_lines if item["text"]]
    if not lines:
        return ""

    lines.sort(key=lambda item: item["top"])
    lines = merge_handwritten_line_blocks(lines)
    gaps = [
        max(current["top"] - previous["bottom"], 0)
        for previous, current in zip(lines, lines[1:])
    ]
    median_gap = float(np.median(gaps)) if gaps else 0.0
    paragraph_gap = max(16.0, median_gap * 1.6)

    assembled: list[str] = []
    previous_bottom: int | None = None
    for item in lines:
        if previous_bottom is not None and item["top"] - previous_bottom > paragraph_gap:
            assembled.append("")
        assembled.append(item["text"])
        previous_bottom = item["bottom"]
    return "\n".join(assembled).strip()


def merge_handwritten_line_blocks(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not lines:
        return []

    merged: list[dict[str, Any]] = [dict(lines[0])]
    for current in lines[1:]:
        previous = merged[-1]
        vertical_gap = max(int(current["top"]) - int(previous["bottom"]), 0)
        left_delta = int(current.get("left", 0)) - int(previous.get("left", 0))
        previous_text = str(previous.get("text") or "").strip()
        current_text = str(current.get("text") or "").strip()
        previous_is_label = bool(HANDWRITTEN_SECTION_LABEL_PATTERN.match(previous_text))
        current_is_continuation = left_delta >= 18 or current_text[:1].islower()
        should_merge = (
            vertical_gap <= 10
            and (
                previous_is_label
                or current_is_continuation
                or previous_text.endswith(":")
            )
        )
        if should_merge:
            previous["text"] = f"{previous_text}\n{current_text}".strip()
            previous["bottom"] = max(int(previous["bottom"]), int(current["bottom"]))
            previous["right"] = max(int(previous.get("right", 0)), int(current.get("right", 0)))
            previous["uncertain"] = bool(previous.get("uncertain")) or bool(current.get("uncertain"))
            previous_score = previous.get("score")
            current_score = current.get("score")
            if previous_score is None:
                previous["score"] = current_score
            elif current_score is not None:
                previous["score"] = min(float(previous_score), float(current_score))
            continue
        merged.append(dict(current))
    return merged


def resolve_torch_device(device_name: str, torch_module: Any) -> str:
    normalized = device_name.strip().lower()
    if normalized == "auto":
        return "cuda" if getattr(torch_module.cuda, "is_available", lambda: False)() else "cpu"
    if normalized == "cuda" and not getattr(torch_module.cuda, "is_available", lambda: False)():
        LOGGER.warning("CUDA was requested for local OCR, but no CUDA device is available; falling back to CPU.")
        return "cpu"
    return normalized
