from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VisionExtractionResult:
    note_text: str
    confidence: str
    uncertain_fragments: list[str]
    provider: str
    model: str
    raw_text: str = ""
    cleaned_text: str = ""
    processed_text: str = ""
