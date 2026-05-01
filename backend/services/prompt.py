from __future__ import annotations

from typing import Any

from backend.core.text import extract_note_evidence
from backend.services.retriever import RetrievedICDEntry
from backend.services.semantic_feature_extractor import extract_semantic_features


SYSTEM_PROMPT = (
    "You are a careful ICD-10 coding expert. Use the full uploaded clinical note plus the provided "
    "retrieved ICD context. Avoid hallucination. Prefer specific supported codes over unspecified "
    "codes when the note contains matching detail. Explanations must be case-aware and grounded in "
    "the patient-specific evidence from the note, not generic textbook language."
)


def format_retrieved_context(entries: list[RetrievedICDEntry]) -> str:
    blocks: list[str] = []
    for index, entry in enumerate(entries, start=1):
        blocks.append(
            "\n".join(
                [
                    f"{index}. Code: {entry.code}",
                    f"Title: {entry.title}",
                    f"Keywords: {', '.join(entry.keywords) or 'N/A'}",
                    f"Definition: {entry.definition or 'N/A'}",
                    f"Hint: {entry.explanation_hint or 'N/A'}",
                    f"Similarity: {entry.similarity_score:.4f}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prediction_messages(
    *,
    clinical_note: str,
    retrieved_entries: list[RetrievedICDEntry],
) -> list[dict[str, str]]:
    context = format_retrieved_context(retrieved_entries)
    note_evidence = format_note_evidence(clinical_note)
    semantic_context = format_semantic_context(clinical_note)
    user_prompt = f"""Context:
{context}

Full Uploaded Clinical Note:
{clinical_note}

Key Note Evidence:
{note_evidence}

Structured Semantic Signals:
{semantic_context}

Task:
1. Review the entire uploaded note, not just the chief complaint
2. Extract ALL confirmed diagnoses explicitly stated or strongly supported in the note
3. Capture both primary and secondary conditions when they are independently supported
4. If multiple conditions appear in phrases such as "with", "and", "secondary to", or complication statements, assess whether each condition needs its own ICD-10 code
5. Do not stop after identifying a single diagnosis when additional supported diagnoses are present
6. Write a brief plain-language summary so a user can understand the case and why the selected codes fit
7. Add a simple condition overview explaining the likely disease/condition in user-friendly language
8. Add 3 to 5 practical precautions or self-care points relevant to the likely condition
9. Add 3 to 5 diet or healthy routine suggestions relevant to the likely condition
10. For each code explanation, cite concrete note evidence such as diagnosis wording, symptoms, vitals, imaging, lab findings, or treatment context
11. Avoid generic code explanations that could fit any patient
12. Avoid incorrect or unsupported codes
13. Provide confidence score (High/Medium/Low)
14. Return only codes that are supported by the retrieved context and the uploaded note
15. Keep guidance general and educational, not a replacement for a clinician
16. If the context is insufficient, return {{"codes": []}}
17. If the clinical note has an explicit Assessment, Diagnosis, Dx, Final Diagnosis, or Impression, prioritize those diagnosis statements over isolated symptom mentions
18. Do not replace a documented diagnosis with a symptom-only code unless the symptom is clearly assessed as a separate diagnosis

Output strictly in JSON format:
{{
  "summary": "...",
  "condition_overview": "...",
  "precautions": ["...", "..."],
  "diet_advice": ["...", "..."],
  "codes": [
    {{
      "code": "...",
      "title": "...",
      "explanation": "...",
      "confidence": "High"
    }}
  ]
}}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_prediction_query(
    *,
    clinical_note: str,
    retrieved_entries: list[RetrievedICDEntry],
) -> str:
    context = format_retrieved_context(retrieved_entries)
    note_evidence = format_note_evidence(clinical_note)
    semantic_context = format_semantic_context(clinical_note)
    return f"""{SYSTEM_PROMPT}

Use only the following ICD-10 context.

Context:
{context}

Full Uploaded Clinical Note:
{clinical_note}

Key Note Evidence:
{note_evidence}

Structured Semantic Signals:
{semantic_context}

Instructions:
- Review the full note before coding
- Extract all supported diagnoses, including primary and secondary conditions
- If the note contains multiple supported diagnoses joined by words like "with", "and", "secondary to", or complication language, consider separate ICD-10 codes for each supported condition
- Do not stop after identifying one diagnosis when additional confirmed conditions are present
- Prioritize explicit Assessment / Diagnosis / Dx / Final Diagnosis / Impression statements over symptom mentions
- Do not substitute symptom-only codes for a documented diagnosis unless the symptom is separately assessed
- Do not choose injury/trauma codes unless trauma, injury, poisoning, overdose, or accident context is present
- Do not choose pregnancy or perinatal codes unless the note explicitly supports that context
- Make each explanation case-aware by referring to the patient-specific evidence in the uploaded note
- Avoid generic textbook explanations and avoid unsupported codes

Return only valid JSON in this format:
{{
  "summary": "...",
  "condition_overview": "...",
  "precautions": ["...", "..."],
  "diet_advice": ["...", "..."],
  "codes": [
    {{
      "code": "...",
      "title": "...",
      "explanation": "...",
      "confidence": "High"
    }}
  ]
}}

If the context is insufficient, return:
{{"codes": []}}"""


def format_priority_candidates(candidates: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        parts = [
            f"{index}. Code: {candidate.get('code', '')}",
            f"Title: {candidate.get('title', '')}",
            f"Confidence: {candidate.get('confidence', 'Unknown')}",
            f"Explanation: {candidate.get('explanation', '')}",
        ]
        if candidate.get("similarity_score") is not None:
            parts.append(f"Similarity: {candidate['similarity_score']}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def build_priority_query(
    *,
    clinical_note: str,
    candidate_codes: list[dict[str, Any]],
) -> str:
    candidates = format_priority_candidates(candidate_codes)
    note_evidence = format_note_evidence(clinical_note)
    semantic_context = format_semantic_context(clinical_note)
    return f"""You are a clinical ICD-10 coding prioritization expert.

Your task is to assign priority to already-retrieved ICD-10 candidate codes for the uploaded clinical case.

Use only the clinical case and candidate ICD-10 codes below. Do not invent unsupported codes.

Clinical Case:
{clinical_note}

Key Note Evidence:
{note_evidence}

Structured Semantic Signals:
{semantic_context}

Candidate ICD-10 Codes:
{candidates}

Rules:
1. The primary ICD code must represent the main reason for admission or encounter.
2. Acute conditions take priority over chronic conditions when they cause admission.
3. If "acute on chronic" is present, make the acute condition primary and keep the chronic condition secondary when clinically relevant.
4. Include all clinically relevant secondary diagnoses from the provided candidates.
5. Prefer confirmed or strongly supported diagnoses over symptom-only codes when a diagnosis explains the symptom.
6. Drop candidate codes that are weakly related, unsupported by the case, or less appropriate than a clearer candidate.
7. Do not select injury/trauma candidates unless the case contains trauma, poisoning, overdose, or accident context.
8. Do not select pregnancy or perinatal candidates unless that context is explicitly present.
9. Select only from the provided candidate codes.

Return strict JSON only:
{{
  "primary_icd": "code",
  "secondary_icd": ["code1", "code2"],
  "dropped_codes": ["codeX"],
  "reasoning": "short clinical justification"
}}"""


def format_note_evidence(clinical_note: str) -> str:
    evidence = extract_note_evidence(clinical_note, limit=4)
    if not evidence:
        return "- No clear evidence lines extracted."
    return "\n".join(f"- {item}" for item in evidence)


def format_semantic_context(clinical_note: str) -> str:
    features = extract_semantic_features(clinical_note)
    blocks: list[str] = []
    for category, values in features.items():
        if not values:
            continue
        blocks.append(f"- {category}: {', '.join(values[:8])}")
    if not blocks:
        return "- No structured semantic signals extracted."
    return "\n".join(blocks)
