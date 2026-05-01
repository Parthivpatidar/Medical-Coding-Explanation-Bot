from __future__ import annotations


VISION_EXTRACTION_SYSTEM_PROMPT = (
    "You are a medical note transcription assistant. Read the uploaded clinical note image and "
    "convert it into a clean, structured clinical note. Preserve only information that is visible "
    "or strongly supported by the image. Do not invent diagnoses, symptoms, medications, plans, "
    "or durations. If a word is unclear, keep it marked as [unclear: ...]. Preserve section "
    "headings and line structure when they are visible."
)


VISION_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "note_text": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["High", "Medium", "Low"],
        },
        "uncertain_fragments": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["note_text", "confidence", "uncertain_fragments"],
    "additionalProperties": False,
}


def build_vision_extraction_instruction(*, supplemental_text: str = "") -> str:
    hint = supplemental_text.strip()
    hint_block = ""
    if hint:
        hint_block = (
            "\nOptional user hint:\n"
            f"{hint}\n\n"
            "Use this only if it agrees with the image and helps resolve an obvious ambiguity. "
            "Ignore it if it conflicts with the image."
        )

    return (
        "Transcribe and lightly reconstruct the uploaded clinical note image into readable clinical "
        "text with line breaks. Expand only obvious shorthand. Keep uncertainty markers inline when "
        "needed. Return JSON only."
        f"{hint_block}"
    )


def build_vision_text_format() -> dict[str, object]:
    return {
        "format": {
            "type": "json_schema",
            "name": "medical_note_extraction",
            "schema": VISION_EXTRACTION_SCHEMA,
            "strict": True,
        }
    }
