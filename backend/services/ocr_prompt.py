from __future__ import annotations


OCR_RECONSTRUCTION_SYSTEM_PROMPT = (
    "You are a medical text reconstruction assistant. Your job is to convert noisy OCR output "
    "from a handwritten or messy clinical note into a clean, structured clinical note. Correct "
    "only obvious OCR errors using context. Preserve the original meaning. Do not invent new "
    "symptoms, diagnoses, medications, or test findings. If a word or phrase is unclear, keep it "
    "marked as [unclear: ...]. Output only the corrected clinical note in a standard clinical format."
)


def build_ocr_reconstruction_query(*, noisy_text: str) -> str:
    return f"""{OCR_RECONSTRUCTION_SYSTEM_PROMPT}

Noisy OCR Text:
{noisy_text}

Instructions:
1. Reconstruct the intended clinical note conservatively.
2. Use only supported sections when evidence exists:
   - Chief Complaint
   - History of Present Illness
   - Examination
   - Findings
   - Assessment / Diagnosis
   - Plan
3. Do not output ICD codes.
4. Keep uncertainty inline as [unclear: ...] when any fragment remains ambiguous.
5. Prefer standard clinical shorthand expansion only when the source strongly supports it.
6. Preserve medications, vitals, imaging findings, and anatomy only when they are supported by the OCR text.

Return only valid JSON in this exact format:
{{
  "reconstructed_note": "...",
  "confidence": "High",
  "uncertain_fragments": ["...", "..."]
}}
"""
