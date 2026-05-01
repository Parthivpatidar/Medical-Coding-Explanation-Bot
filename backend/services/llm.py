from __future__ import annotations

import os
from typing import Any

from backend.core.text import clean_text, extract_note_evidence
from backend.core.utils import coerce_confidence, safe_json_loads
from backend.services.ocr_prompt import build_ocr_reconstruction_query
from backend.services.prompt import build_prediction_query, build_priority_query
from backend.services.reranker import score_semantic_candidates, select_semantically_supported_candidates
from backend.services.retriever import RetrievedICDEntry


class BaseLLMClient:
    def generate_codes(
        self,
        *,
        clinical_note: str,
        retrieved_entries: list[RetrievedICDEntry],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def reconstruct_note(
        self,
        *,
        noisy_text: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def prioritize_codes(
        self,
        *,
        clinical_note: str,
        candidate_codes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise NotImplementedError


class SearchApiLLMClient(BaseLLMClient):
    def __init__(
        self,
        *,
        base_url: str,
        engine: str,
        timeout_seconds: float,
        location: str,
        hl: str,
        gl: str,
        api_key: str | None = None,
    ) -> None:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "requests is not installed. Run `pip install -r requirements.txt` before using "
                "the SearchAPI provider."
            ) from exc

        self.requests = requests
        self.api_key = api_key or os.getenv("SEARCHAPI_API_KEY")
        self.base_url = base_url
        self.engine = engine
        self.model_name = engine
        self.timeout_seconds = timeout_seconds
        self.location = location
        self.hl = hl
        self.gl = gl

        if not self.api_key:
            raise RuntimeError("SEARCHAPI_API_KEY is required when MEDICHAR_LLM_PROVIDER=searchapi")

    def generate_codes(
        self,
        *,
        clinical_note: str,
        retrieved_entries: list[RetrievedICDEntry],
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {
            "engine": self.engine,
            "q": build_prediction_query(
                clinical_note=clinical_note,
                retrieved_entries=retrieved_entries,
            ),
            "hl": self.hl,
            "gl": self.gl,
        }
        if self.location:
            params["location"] = self.location

        response = self.requests.get(
            self.base_url,
            headers=headers,
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        content = payload.get("markdown") or _text_blocks_to_string(payload.get("text_blocks", []))
        return safe_json_loads(content)

    def reconstruct_note(
        self,
        *,
        noisy_text: str,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {
            "engine": self.engine,
            "q": build_ocr_reconstruction_query(noisy_text=noisy_text),
            "hl": self.hl,
            "gl": self.gl,
        }
        if self.location:
            params["location"] = self.location

        response = self.requests.get(
            self.base_url,
            headers=headers,
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        content = payload.get("markdown") or _text_blocks_to_string(payload.get("text_blocks", []))
        return safe_json_loads(content)

    def prioritize_codes(
        self,
        *,
        clinical_note: str,
        candidate_codes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {
            "engine": self.engine,
            "q": build_priority_query(
                clinical_note=clinical_note,
                candidate_codes=candidate_codes,
            ),
            "hl": self.hl,
            "gl": self.gl,
        }
        if self.location:
            params["location"] = self.location

        response = self.requests.get(
            self.base_url,
            headers=headers,
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        content = payload.get("markdown") or _text_blocks_to_string(payload.get("text_blocks", []))
        return safe_json_loads(content)


class HeuristicLLMClient(BaseLLMClient):
    def generate_codes(
        self,
        *,
        clinical_note: str,
        retrieved_entries: list[RetrievedICDEntry],
    ) -> dict[str, Any]:
        note = clean_text(clinical_note)
        evidence = extract_note_evidence(clinical_note, limit=3)
        ranked_candidates = score_semantic_candidates(clinical_note, retrieved_entries[:5])
        selected_ranked = select_semantically_supported_candidates(
            ranked_candidates,
            max_candidates=3,
            min_alignment_score=0.05,
            min_final_score_ratio=0.72,
        )
        selected_entries = [item.candidate for item in selected_ranked] or retrieved_entries[:3]
        predictions: list[dict[str, str]] = []
        for entry in selected_entries:
            overlap = [keyword for keyword in entry.keywords if clean_text(keyword) in note]
            explanation_parts = [
                f"Retrieved ICD context matched {entry.title}.",
                f"Definition support: {entry.definition or 'No definition available.'}",
            ]
            if overlap:
                explanation_parts.append(f"Keyword overlap: {', '.join(overlap[:4])}.")
            if evidence:
                explanation_parts.append(f"Case evidence: {' | '.join(evidence[:2])}")
            if entry.explanation_hint:
                explanation_parts.append(f"Hint: {entry.explanation_hint}")
            predictions.append(
                {
                    "code": entry.code,
                    "title": entry.title,
                    "explanation": " ".join(explanation_parts),
                    "confidence": coerce_confidence(None, entry.similarity_score),
                }
            )
        summary = ""
        condition_overview = ""
        precautions: list[str] = []
        diet_advice: list[str] = []
        if predictions:
            top_entry = selected_entries[0]
            summary_parts = [
                f"The note most closely matches {top_entry.title} ({top_entry.code}).",
            ]
            if evidence:
                summary_parts.append(f"Key note evidence: {evidence[0]}")
            if len(predictions) > 1:
                summary_parts.append(
                    "Additional separately supported ICD-10 matches were also retained for review."
                )
            summary = " ".join(summary_parts)
            condition_overview = _build_condition_overview(top_entry)
            precautions = _build_precautions(top_entry)
            diet_advice = _build_diet_advice(top_entry)
        return {
            "summary": summary,
            "condition_overview": condition_overview,
            "precautions": precautions,
            "diet_advice": diet_advice,
            "codes": predictions,
        }

    def reconstruct_note(
        self,
        *,
        noisy_text: str,
    ) -> dict[str, Any]:
        evidence = extract_note_evidence(noisy_text, limit=2)
        return {
            "reconstructed_note": noisy_text,
            "confidence": "Low",
            "uncertain_fragments": evidence,
        }

    def prioritize_codes(
        self,
        *,
        clinical_note: str,
        candidate_codes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del clinical_note
        if not candidate_codes:
            return {
                "primary_icd": "",
                "secondary_icd": [],
                "dropped_codes": [],
                "reasoning": "No candidate ICD codes were available for prioritization.",
            }
        primary = str(candidate_codes[0].get("code") or "").strip()
        secondary = [
            str(item.get("code") or "").strip()
            for item in candidate_codes[1:]
            if str(item.get("code") or "").strip()
        ]
        return {
            "primary_icd": primary,
            "secondary_icd": secondary,
            "dropped_codes": [],
            "reasoning": "Candidate order was retained because the heuristic provider cannot perform external clinical prioritization.",
        }


def build_llm_client(
    *,
    provider: str,
    model: str,
    searchapi_base_url: str,
    searchapi_engine: str,
    searchapi_location: str,
    searchapi_hl: str,
    searchapi_gl: str,
    temperature: float,
    timeout_seconds: float,
) -> BaseLLMClient:
    del model
    del temperature

    normalized_provider = provider.strip().lower()
    if normalized_provider == "auto":
        if os.getenv("SEARCHAPI_API_KEY"):
            normalized_provider = "searchapi"
        else:
            normalized_provider = "heuristic"

    if normalized_provider == "searchapi":
        return SearchApiLLMClient(
            base_url=searchapi_base_url,
            engine=searchapi_engine,
            timeout_seconds=timeout_seconds,
            location=searchapi_location,
            hl=searchapi_hl,
            gl=searchapi_gl,
        )
    return HeuristicLLMClient()


def _text_blocks_to_string(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if "answer" in block and block["answer"]:
            parts.append(str(block["answer"]))
        elif "code" in block and block["code"]:
            parts.append(str(block["code"]))
    return "\n".join(parts)


def _build_condition_overview(entry: RetrievedICDEntry) -> str:
    title = entry.title
    definition = entry.definition or f"{title} is the main condition suggested by the note."
    title_lower = title.lower()

    if "hypertension" in title_lower:
        return (
            f"{title} means blood pressure is persistently elevated. "
            "It often needs regular monitoring, medicines when prescribed, and lifestyle measures "
            "to reduce long-term heart, brain, and kidney complications."
        )
    if "diabetes" in title_lower:
        return (
            f"{title} means the body is having trouble controlling blood sugar. "
            "Good glucose control, regular follow-up, and food choices that reduce sugar spikes help "
            "lower the risk of eye, kidney, nerve, and heart complications."
        )
    if "pneumonia" in title_lower:
        return (
            f"{title} is a lung infection that can cause cough, fever, weakness, and breathing difficulty. "
            "Recovery usually depends on timely treatment, rest, fluids, and watching closely for worsening symptoms."
        )
    if "urinary tract infection" in title_lower or "cystitis" in title_lower:
        return (
            f"{title} usually reflects an infection affecting the urinary system. "
            "Symptoms may include burning urination, urgency, lower abdominal discomfort, or fever, and prompt treatment "
            "helps prevent the infection from spreading."
        )
    return (
        f"{title} is the condition most strongly supported by the retrieved ICD context. "
        f"{definition}"
    )


def _build_precautions(entry: RetrievedICDEntry) -> list[str]:
    title_lower = entry.title.lower()

    if "hypertension" in title_lower:
        return [
            "Take blood pressure medicines exactly as prescribed and do not stop them suddenly.",
            "Check blood pressure regularly and keep a record for follow-up visits.",
            "Seek urgent care for severe headache, chest pain, shortness of breath, weakness, or vision change.",
        ]
    if "diabetes" in title_lower:
        return [
            "Monitor blood sugar as advised and take diabetes medicines consistently.",
            "Watch for symptoms of very high or very low sugar such as dizziness, sweating, confusion, or unusual thirst.",
            "Protect the feet, stay hydrated, and arrange regular eye and kidney follow-up.",
        ]
    if "pneumonia" in title_lower:
        return [
            "Rest well, complete the prescribed treatment course, and monitor breathing symptoms closely.",
            "Seek urgent care for worsening breathlessness, chest pain, bluish lips, confusion, or persistent high fever.",
            "Avoid smoking and reduce exposure to dust or smoke while recovering.",
        ]
    if "urinary tract infection" in title_lower or "cystitis" in title_lower:
        return [
            "Take prescribed treatment fully and do not ignore worsening pain, fever, or vomiting.",
            "Drink fluids unless a clinician has told you to restrict them.",
            "Seek medical review quickly if symptoms spread to the back, fever appears, or the patient is pregnant or frail.",
        ]
    return [
        "Follow the treatment plan and review symptoms with a clinician if they worsen or do not improve.",
        "Monitor for new red-flag symptoms such as severe pain, breathing difficulty, fainting, or confusion.",
        "Keep follow-up appointments so treatment can be adjusted based on progress.",
    ]


def _build_diet_advice(entry: RetrievedICDEntry) -> list[str]:
    title_lower = entry.title.lower()

    if "hypertension" in title_lower:
        return [
            "Reduce salt-heavy foods such as chips, packaged snacks, and processed meals.",
            "Favor vegetables, fruits, pulses, whole grains, and balanced home-cooked meals.",
            "Limit alcohol and sugary drinks, and maintain a healthy body weight.",
        ]
    if "diabetes" in title_lower:
        return [
            "Choose meals with fiber, vegetables, protein, and controlled portions of rice, bread, or other carbs.",
            "Limit sweets, sweet drinks, and frequent refined snacks that can spike sugar quickly.",
            "Spread meals evenly through the day and pair diet changes with regular physical activity if approved.",
        ]
    if "pneumonia" in title_lower:
        return [
            "Drink enough fluids and use light, easy-to-eat meals while appetite is low.",
            "Prioritize protein-rich foods, soups, fruits, and soft meals that support recovery.",
            "Avoid smoking, excessive alcohol, and foods that worsen dehydration.",
        ]
    if "urinary tract infection" in title_lower or "cystitis" in title_lower:
        return [
            "Drink enough water through the day unless fluids are medically restricted.",
            "Choose balanced meals and avoid excess sugary drinks if they worsen symptoms or glucose control.",
            "Limit irritants such as excessive caffeine or alcohol if they trigger bladder discomfort.",
        ]
    return [
        "Choose balanced meals with vegetables, fruit, adequate protein, and whole grains where possible.",
        "Keep hydration adequate unless the care team has advised fluid restriction.",
        "Avoid excess alcohol, smoking, and heavily processed foods while recovering.",
    ]
