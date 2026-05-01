from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps


WINDOWS_OCR_SCRIPT = """
$ErrorActionPreference = 'Stop'
$imagePath = '__IMAGE_PATH__'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStreamWithContentType, Windows.Storage.Streams, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime]

function Await($WinRtTask, $ResultType) {
  $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
  })[0]
  $netTask = $asTaskGeneric.MakeGenericMethod($ResultType).Invoke($null, @($WinRtTask))
  $netTask.Wait(-1) | Out-Null
  $netTask.Result
}

$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($imagePath)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenReadAsync()) ([Windows.Storage.Streams.IRandomAccessStreamWithContentType])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
Write-Output $result.Text
""".strip()

STRUCTURED_NOTE_HINTS = (
    "patient",
    "age",
    "date",
    "chief",
    "complaint",
    "history",
    "present",
    "illness",
    "physical",
    "exam",
    "assessment",
    "plan",
    "cough",
    "fever",
    "sob",
    "breath",
    "fatigue",
    "chills",
    "crackles",
    "oxygen",
    "saturation",
    "o2",
    "cxr",
    "pneumonia",
    "ceftriaxone",
    "therapy",
    "follow",
    "recheck",
    "rr",
    "temp",
    "iv",
    "rll",
    "resp",
    "hpi",
    "cc",
    "dx",
)


def extract_text_from_image_bytes(
    image_bytes: bytes,
    *,
    filename: str = "note.png",
    ocr_mode: str = "auto",
) -> tuple[str | None, str | None]:
    if os.name != "nt":
        return None, "Windows OCR is available only on Windows."

    normalized_mode = ocr_mode.strip().lower()
    candidates: list[str] = []
    errors: list[str] = []

    try:
        for variant_bytes in build_ocr_image_variants(image_bytes, normalized_mode):
            extracted_text, error = run_windows_ocr(
                variant_bytes,
                suffix=Path(filename or "note.png").suffix or ".png",
            )
            if extracted_text:
                candidates.append(extracted_text)
            elif error:
                errors.append(error)
    except OSError as exc:
        return None, f"Windows OCR could not start: {exc}"

    if candidates:
        best_text = max(candidates, key=lambda text: score_ocr_candidate(text, normalized_mode))
        return normalize_ocr_text(best_text), None

    if errors:
        return None, errors[0]
    return None, "No readable text was detected in the uploaded image."


def run_windows_ocr(image_bytes: bytes, *, suffix: str = ".png") -> tuple[str | None, str | None]:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".png") as handle:
            handle.write(image_bytes)
            temp_path = Path(handle.name)

        script = WINDOWS_OCR_SCRIPT.replace("__IMAGE_PATH__", str(temp_path).replace("'", "''"))
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        extracted_text = normalize_ocr_text(completed.stdout)
        if completed.returncode != 0:
            error_text = completed.stderr.strip() or completed.stdout.strip() or "Windows OCR failed."
            return None, error_text
        if not extracted_text:
            return None, "No readable text was detected in the uploaded image."
        return extracted_text, None
    except subprocess.TimeoutExpired:
        return None, "Windows OCR took too long. Try a clearer or smaller image."
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def build_ocr_image_variants(image_bytes: bytes, ocr_mode: str) -> list[bytes]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            resampling = getattr(Image, "Resampling", Image)
            base = ImageOps.exif_transpose(image).convert("RGB")
            grayscale = ImageOps.grayscale(base)
            scale_factor = 3 if ocr_mode == "handwritten" else 2
            max_width = 2800 if ocr_mode == "handwritten" else 2200
            width = max(min(grayscale.width * scale_factor, max_width), grayscale.width)
            height = max(int(grayscale.height * (width / max(grayscale.width, 1))), grayscale.height)
            resized = grayscale.resize((width, height), resample=resampling.LANCZOS)

            variants: list[Image.Image] = []
            normalized = ImageOps.autocontrast(resized)
            variants.append(normalized)

            sharpened = normalized.filter(ImageFilter.SHARPEN)
            variants.append(sharpened)

            if ocr_mode in {"auto", "handwritten"}:
                smoothed = normalized.filter(ImageFilter.MedianFilter(size=3))
                hand_enhanced = ImageOps.autocontrast(smoothed.filter(ImageFilter.SHARPEN))
                variants.append(hand_enhanced)
                threshold = hand_enhanced.point(lambda pixel: 255 if pixel > 150 else 0)
                variants.append(threshold)
                soft_threshold = hand_enhanced.point(lambda pixel: 255 if pixel > 185 else 0)
                variants.append(soft_threshold)
            else:
                threshold = sharpened.point(lambda pixel: 255 if pixel > 170 else 0)
                variants.append(threshold)

            serialized: list[bytes] = []
            seen: set[bytes] = set()
            for variant in variants:
                buffer = io.BytesIO()
                variant.save(buffer, format="PNG")
                payload = buffer.getvalue()
                if payload in seen:
                    continue
                seen.add(payload)
                serialized.append(payload)
            return serialized
    except Exception:
        return [image_bytes]


def normalize_ocr_text(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    return "\n".join(lines).strip()


def score_ocr_candidate(text: str, ocr_mode: str) -> float:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return 0.0

    words = re.findall(r"[A-Za-z0-9]{2,}", normalized)
    alpha_count = sum(character.isalpha() for character in normalized)
    digit_count = sum(character.isdigit() for character in normalized)
    whitespace_count = sum(character.isspace() for character in normalized)
    odd_symbol_count = sum(character in "{}[]|~`#•" or ord(character) > 126 for character in normalized)
    long_word_bonus = sum(1 for word in words if 3 <= len(word) <= 14)
    no_space_penalty = 12 if len(words) <= 1 and len(normalized) > 12 else 0
    structured_hint_bonus = sum(1 for hint in STRUCTURED_NOTE_HINTS if hint in normalized.lower())

    score = (
        (len(words) * 6)
        + (alpha_count * 0.18)
        + (digit_count * 0.08)
        + (whitespace_count * 1.2)
        + (long_word_bonus * 1.5)
        + (structured_hint_bonus * 5.0)
        - (odd_symbol_count * 6)
        - no_space_penalty
    )

    if ocr_mode == "handwritten":
        score += min(len(words), 8) * 1.5
    return score


def confidence_from_ocr_score(score: float) -> str:
    if score >= 520:
        return "High"
    if score >= 280:
        return "Medium"
    return "Low"
