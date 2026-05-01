from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "backend" / "data" / "ocr_benchmark_cases.json"


@dataclass(slots=True)
class OCRBenchmarkCase:
    case_id: str
    difficulty: str
    ocr_mode: str
    expected_text: str
    image_path: str


def load_cases(path: Path) -> list[OCRBenchmarkCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[OCRBenchmarkCase] = []
    for item in payload.get("cases", []):
        cases.append(
            OCRBenchmarkCase(
                case_id=str(item["case_id"]),
                difficulty=str(item["difficulty"]),
                ocr_mode=str(item["ocr_mode"]),
                expected_text=str(item["expected_text"]),
                image_path=str(item["image_path"]),
            )
        )
    return cases


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def word_overlap_score(expected_text: str, predicted_text: str) -> float:
    expected_words = set(normalize_text(expected_text).split())
    predicted_words = set(normalize_text(predicted_text).split())
    if not expected_words and not predicted_words:
        return 1.0
    if not expected_words or not predicted_words:
        return 0.0
    intersection = len(expected_words & predicted_words)
    union = len(expected_words | predicted_words)
    return round(intersection / max(union, 1), 4)


def line_recall_score(expected_text: str, predicted_text: str) -> float:
    expected_lines = [normalize_text(line) for line in expected_text.splitlines() if normalize_text(line)]
    predicted_normalized = normalize_text(predicted_text)
    if not expected_lines:
        return 1.0
    matched = sum(1 for line in expected_lines if line in predicted_normalized)
    return round(matched / len(expected_lines), 4)


def evaluate_case(service: Any, case: OCRBenchmarkCase) -> dict[str, Any]:
    image_bytes = Path(case.image_path).read_bytes()
    extraction = service.extract_note_from_image(
        image_bytes=image_bytes,
        filename=Path(case.image_path).name,
        ocr_mode=case.ocr_mode,
    )
    predicted_text = str(extraction.get("text") or "")
    return {
        "case_id": case.case_id,
        "difficulty": case.difficulty,
        "ocr_mode": case.ocr_mode,
        "confidence": extraction.get("confidence"),
        "word_overlap": word_overlap_score(case.expected_text, predicted_text),
        "line_recall": line_recall_score(case.expected_text, predicted_text),
        "uncertain_count": len(extraction.get("uncertain_fragments") or []),
        "predicted_text": predicted_text,
    }


def main() -> None:
    from backend.services.medical_coding import MedicalCodingService

    benchmark_path = DEFAULT_BENCHMARK_PATH
    if not benchmark_path.exists():
        raise SystemExit(
            f"Benchmark file not found: {benchmark_path}\n"
            "Create it from the provided template and point image_path entries at local test images."
        )

    service = MedicalCodingService()
    cases = load_cases(benchmark_path)
    results = [evaluate_case(service, case) for case in cases]

    averages = {
        "word_overlap": round(sum(item["word_overlap"] for item in results) / max(len(results), 1), 4),
        "line_recall": round(sum(item["line_recall"] for item in results) / max(len(results), 1), 4),
        "uncertain_count": round(sum(item["uncertain_count"] for item in results) / max(len(results), 1), 2),
    }
    print(json.dumps({"summary": averages, "results": results}, indent=2))


if __name__ == "__main__":
    main()
