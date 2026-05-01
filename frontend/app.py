from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_DIR = PROJECT_ROOT / "backend" / "data" / "mimic-iv-clinical-database-demo-2.2"
MIMIC_HOSP_DIR = MIMIC_DIR / "hosp"
API_BASE_URL = os.getenv("MEDICHAR_API_BASE_URL", "http://localhost:8000").rstrip("/")
IMAGE_EXTRACTION_TIMEOUT_SECONDS = float(os.getenv("MEDICHAR_IMAGE_EXTRACTION_TIMEOUT", "300"))
PREDICT_TIMEOUT_SECONDS = float(os.getenv("MEDICHAR_PREDICT_TIMEOUT", "180"))
PREFERRED_EXAMPLE_CODES: tuple[str, ...] = ("A419", "I10", "J189", "N390")
CURATED_NOTE_TEMPLATES: dict[str, str] = {
    "A419": (
        "Admission Type: Emergency\n"
        "History of Present Illness:\n"
        "58-year-old patient presented with fever, chills, hypotension, tachycardia, and generalized weakness. "
        "Blood cultures were obtained and broad-spectrum antibiotics were started, but the note does not specify the organism.\n"
        "Hospital Course:\n"
        "The patient required IV fluids, close monitoring, and ongoing infectious workup during admission.\n"
        "Assessment:\n"
        "Sepsis, unspecified organism.\n"
        "Plan:\n"
        "Continue IV antibiotics, monitor cultures, trend vitals, and reassess response to therapy."
    ),
    "I10": (
        "Admission Type: Outpatient\n"
        "History of Present Illness:\n"
        "54-year-old male presented for follow-up after repeatedly elevated blood pressure readings over the last month. "
        "He denied chest pain, dyspnea, focal weakness, headache, or renal symptoms.\n"
        "Assessment:\n"
        "Essential (primary) hypertension.\n"
        "Plan:\n"
        "Continue antihypertensive therapy, reinforce low-salt diet, monitor blood pressure at home, and follow up in clinic."
    ),
    "J189": (
        "Admission Type: Emergency\n"
        "History of Present Illness:\n"
        "67-year-old male presented with 4 days of productive cough, fever, fatigue, and shortness of breath. "
        "Oxygen saturation was reduced on room air, and chest imaging supported pneumonia without a specified organism.\n"
        "Assessment:\n"
        "Pneumonia, unspecified organism.\n"
        "Plan:\n"
        "Start IV ceftriaxone, provide oxygen therapy, monitor respiratory status, and follow up after clinical improvement."
    ),
    "N390": (
        "Admission Type: Emergency\n"
        "History of Present Illness:\n"
        "48-year-old female presented with dysuria, urinary frequency, suprapubic discomfort, and low-grade fever for 2 days. "
        "Urinalysis was consistent with infection, and there was no pyelonephritis or sepsis documented in this note.\n"
        "Assessment:\n"
        "Urinary tract infection, site not specified.\n"
        "Plan:\n"
        "Start antibiotics, encourage fluids, review urine culture results, and arrange follow-up if symptoms persist."
    ),
}


st.set_page_config(
    page_title="MediChar",
    page_icon="M",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(215, 236, 229, 0.9), transparent 30%),
                radial-gradient(circle at top right, rgba(255, 231, 209, 0.9), transparent 28%),
                linear-gradient(180deg, #f6f2e9 0%, #fbf8f2 52%, #f1efe8 100%);
        }
        .hero-panel {
            padding: 1.3rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(13, 74, 62, 0.95), rgba(38, 102, 79, 0.92));
            color: #f7fbf7;
            box-shadow: 0 18px 40px rgba(29, 59, 52, 0.18);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        .hero-copy {
            font-size: 1rem;
            line-height: 1.55;
            max-width: 60rem;
            color: rgba(247, 251, 247, 0.9);
        }
        .hero-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.95rem;
        }
        .hero-badge {
            display: inline-block;
            padding: 0.3rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.16);
            color: #f2fbf7;
            font-size: 0.84rem;
            letter-spacing: 0.01em;
        }
        .metric-card {
            background: rgba(255, 252, 246, 0.94);
            border: 1px solid rgba(49, 87, 77, 0.12);
            border-radius: 18px;
            padding: 1rem 1.05rem 0.9rem 1.05rem;
            box-shadow: 0 8px 22px rgba(67, 78, 69, 0.06);
            min-height: 110px;
        }
        .spotlight-card {
            background: linear-gradient(180deg, rgba(255, 252, 246, 0.98), rgba(251, 246, 238, 0.92));
            border: 1px solid rgba(49, 87, 77, 0.12);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 8px 22px rgba(67, 78, 69, 0.06);
            min-height: 138px;
        }
        .spotlight-kicker {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #8b6c4f;
            margin-bottom: 0.45rem;
        }
        .spotlight-title {
            font-size: 1.12rem;
            font-weight: 800;
            color: #1d372f;
            margin-bottom: 0.35rem;
        }
        .spotlight-copy {
            color: #536862;
            font-size: 0.93rem;
            line-height: 1.45;
        }
        .spotlight-foot {
            margin-top: 0.7rem;
            color: #295246;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #5a6e64;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #1f3831;
            margin-bottom: 0.15rem;
        }
        .metric-foot {
            font-size: 0.9rem;
            color: #5f6d67;
            line-height: 1.35;
        }
        .section-panel {
            background: rgba(255, 252, 246, 0.95);
            border: 1px solid rgba(54, 80, 69, 0.12);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 26px rgba(56, 70, 61, 0.05);
        }
        .top-result {
            background: linear-gradient(135deg, rgba(244, 251, 247, 0.96), rgba(255, 249, 240, 0.98));
            border: 1px solid rgba(42, 107, 86, 0.15);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
        }
        .top-code {
            font-size: 1.4rem;
            font-weight: 800;
            color: #18453b;
        }
        .top-sub {
            color: #516862;
            font-size: 0.95rem;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.5rem;
        }
        .chip {
            display: inline-block;
            background: rgba(24, 69, 59, 0.08);
            border: 1px solid rgba(24, 69, 59, 0.12);
            color: #21473d;
            border-radius: 999px;
            padding: 0.22rem 0.62rem;
            font-size: 0.84rem;
        }
        .info-board {
            background:
                radial-gradient(circle at top right, rgba(210, 233, 223, 0.78), transparent 36%),
                linear-gradient(180deg, rgba(255, 252, 246, 0.98), rgba(250, 246, 238, 0.95));
            border: 1px solid rgba(54, 80, 69, 0.12);
            border-radius: 22px;
            padding: 1.15rem;
            box-shadow: 0 10px 26px rgba(56, 70, 61, 0.05);
        }
        .panel-kicker {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #8f7256;
            margin-bottom: 0.35rem;
        }
        .panel-title {
            font-size: 1.95rem;
            line-height: 1.05;
            font-weight: 800;
            color: #1c342d;
            margin-bottom: 0.45rem;
        }
        .panel-copy {
            color: #536862;
            font-size: 0.98rem;
            line-height: 1.55;
            margin-bottom: 0.95rem;
        }
        .step-card {
            border-radius: 16px;
            padding: 0.85rem 0.9rem;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(43, 81, 69, 0.1);
            margin-bottom: 0.7rem;
        }
        .step-no {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #8f7256;
            margin-bottom: 0.2rem;
        }
        .step-title {
            font-size: 1rem;
            font-weight: 800;
            color: #1f3831;
            margin-bottom: 0.18rem;
        }
        .step-copy {
            color: #566a64;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .value-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.8rem 0 0.95rem 0;
        }
        .value-pill {
            display: inline-block;
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            background: rgba(24, 69, 59, 0.08);
            border: 1px solid rgba(24, 69, 59, 0.1);
            color: #22483e;
            font-size: 0.83rem;
        }
        .catalog-pulse {
            margin-top: 0.95rem;
            padding-top: 0.85rem;
            border-top: 1px solid rgba(43, 81, 69, 0.1);
            color: #566a64;
            font-size: 0.92rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset_stats() -> dict[str, int]:
    stats: dict[str, int] = {"icd10_rows": 0, "diagnosis_rows": 0, "patient_rows": 0}
    dictionary_path = MIMIC_HOSP_DIR / "d_icd_diagnoses.csv"
    diagnoses_path = MIMIC_HOSP_DIR / "diagnoses_icd.csv"
    patients_path = MIMIC_HOSP_DIR / "patients.csv"

    if dictionary_path.exists():
        with dictionary_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            stats["icd10_rows"] = sum(
                1 for row in reader if str(row.get("icd_version", "")).strip() == "10"
            )
    if diagnoses_path.exists():
        with diagnoses_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            stats["diagnosis_rows"] = sum(1 for _ in reader)
    if patients_path.exists():
        with patients_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            stats["patient_rows"] = sum(1 for _ in reader)
    return stats


def build_dataset_note(code: str, title: str) -> str:
    curated = CURATED_NOTE_TEMPLATES.get(code)
    if curated:
        return curated
    return (
        "Admission Type: Outpatient\n"
        "History of Present Illness:\n"
        f"The patient presented for evaluation of {title.lower()} documented in the chart.\n"
        "Assessment:\n"
        f"{title}.\n"
        "Plan:\n"
        "Review symptoms, continue appropriate treatment, and arrange follow-up."
    )


@st.cache_data(show_spinner=False)
def load_dataset_examples() -> list[dict[str, str]]:
    dictionary_path = MIMIC_HOSP_DIR / "d_icd_diagnoses.csv"
    diagnoses_path = MIMIC_HOSP_DIR / "diagnoses_icd.csv"

    if not dictionary_path.exists() or not diagnoses_path.exists():
        return []

    titles_by_code: dict[str, str] = {}
    with dictionary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("icd_version", "")).strip() != "10":
                continue
            code = str(row.get("icd_code", "")).strip().upper()
            title = str(row.get("long_title", "")).strip()
            if code and title and code not in titles_by_code:
                titles_by_code[code] = title

    preferred_examples: dict[str, dict[str, str]] = {}
    fallback_examples: list[dict[str, str]] = []
    seen_codes: set[str] = set()

    with diagnoses_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("icd_version", "")).strip() != "10":
                continue
            code = str(row.get("icd_code", "")).strip().upper()
            if not code or code in seen_codes or code not in titles_by_code:
                continue

            title = titles_by_code[code]
            example = {
                "code": code,
                "title": title,
                "hadm_id": str(row.get("hadm_id", "")).strip(),
                "label": f"{code} - {title}",
                "note_text": build_dataset_note(code, title),
            }

            if code in PREFERRED_EXAMPLE_CODES and code not in preferred_examples:
                preferred_examples[code] = example
            else:
                fallback_examples.append(example)
            seen_codes.add(code)

            if len(preferred_examples) == len(PREFERRED_EXAMPLE_CODES) and len(fallback_examples) >= 4:
                break

    ordered_examples: list[dict[str, str]] = []
    for code in PREFERRED_EXAMPLE_CODES:
        if code in preferred_examples:
            ordered_examples.append(preferred_examples[code])

    for example in fallback_examples:
        if len(ordered_examples) >= 4:
            break
        if example["code"] not in {item["code"] for item in ordered_examples}:
            ordered_examples.append(example)

    return ordered_examples[:4]


@st.cache_data(ttl=30, show_spinner=False)
def load_api_health(api_base_url: str) -> dict[str, str]:
    try:
        response = requests.get(f"{api_base_url}/health", timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {
            "status": "offline",
            "catalog_size": 0,
            "llm_provider": "Unavailable",
            "llm_model": str(exc),
        }


def predict_note(api_base_url: str, text: str) -> dict:
    response = requests.post(
        f"{api_base_url}/predict",
        json={
            "text": text,
            "return_context": True,
            "return_similarity": True,
        },
        timeout=PREDICT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def extract_note_from_image_api(
    api_base_url: str,
    uploaded_image,
    *,
    supplemental_text: str = "",
    ocr_mode: str = "printed",
) -> dict:
    image_name = uploaded_image.name or "note.png"
    image_type = uploaded_image.type or "application/octet-stream"
    response = requests.post(
        f"{api_base_url}/extract-note-image",
        data={"supplemental_text": supplemental_text, "ocr_mode": ocr_mode},
        files={"image": (image_name, uploaded_image.getvalue(), image_type)},
        timeout=IMAGE_EXTRACTION_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def lookup_code(api_base_url: str, code: str) -> dict:
    response = requests.get(
        f"{api_base_url}/lookup/{quote(code)}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def lookup_codes(api_base_url: str, codes: list[str]) -> dict:
    response = requests.post(
        f"{api_base_url}/lookup-batch",
        json={"codes": codes},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def render_metric_card(label: str, value: str, foot: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_spotlight_card(kicker: str, title: str, copy: str, foot: str) -> None:
    st.markdown(
        f"""
        <div class="spotlight-card">
            <div class="spotlight-kicker">{kicker}</div>
            <div class="spotlight-title">{title}</div>
            <div class="spotlight-copy">{copy}</div>
            <div class="spotlight-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggestion(suggestion: dict, rank: int) -> None:
    with st.container(border=True):
        confidence = suggestion.get("confidence", "Unknown")
        similarity = suggestion.get("similarity_score")
        st.markdown(
            f"### {rank}. `{suggestion['code']}` - {suggestion['title']}\n"
            f"**Confidence:** {confidence}"
            + (f" | **Similarity:** {similarity:.3f}" if isinstance(similarity, (int, float)) else "")
        )
        st.write(suggestion.get("explanation", "No explanation returned."))


def render_guidance_list(items: list[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")


def replace_note_with_extracted_text() -> None:
    st.session_state.note_mode_input = st.session_state.get("note_mode_ocr_text", "")


def append_extracted_text() -> None:
    base_text = st.session_state.get("note_mode_input", "").strip()
    ocr_text = st.session_state.get("note_mode_ocr_text", "").strip()
    if ocr_text:
        st.session_state.note_mode_input = f"{base_text}\n\n{ocr_text}" if base_text else ocr_text


def clear_extracted_text() -> None:
    st.session_state.note_mode_ocr_text = ""
    st.session_state.note_mode_ocr_raw_text = ""
    st.session_state.note_mode_ocr_cleaned_text = ""
    st.session_state.note_mode_ocr_processed_text = ""


def parse_lookup_codes(raw_text: str) -> list[str]:
    seen: set[str] = set()
    codes: list[str] = []
    for piece in re.split(r"[\s,;|]+", raw_text or ""):
        code = piece.strip()
        normalized = code.upper()
        if not code or normalized in seen:
            continue
        seen.add(normalized)
        codes.append(code)
    return codes


def render_ocr_editor() -> None:
    st.markdown("### Image Note Review")
    raw_text = st.session_state.get("note_mode_ocr_raw_text", "").strip()
    cleaned_text = st.session_state.get("note_mode_ocr_cleaned_text", "").strip()
    processed_text = st.session_state.get("note_mode_ocr_processed_text", "").strip()
    if raw_text or cleaned_text or processed_text:
        review_tabs = st.tabs(["Editable Final Text", "Raw OCR", "Cleaned OCR"])
        with review_tabs[0]:
            st.text_area(
                "Review and correct extracted note text",
                key="note_mode_ocr_text",
                height=150,
                placeholder="Extracted note text will appear here for quick cleanup...",
            )
        with review_tabs[1]:
            st.text_area(
                "Raw OCR output",
                value=raw_text or processed_text,
                height=150,
                disabled=True,
            )
        with review_tabs[2]:
            st.text_area(
                "Backend-cleaned OCR output",
                value=cleaned_text or processed_text,
                height=150,
                disabled=True,
            )
    else:
        st.text_area(
            "Review and correct extracted note text",
            key="note_mode_ocr_text",
            height=150,
            placeholder="Extracted note text will appear here for quick cleanup...",
        )

    editor_cols = st.columns(3)
    with editor_cols[0]:
        st.button(
            "Replace Note With Extracted Text",
            width="stretch",
            on_click=replace_note_with_extracted_text,
        )
    with editor_cols[1]:
        st.button(
            "Append Extracted Text",
            width="stretch",
            on_click=append_extracted_text,
        )
    with editor_cols[2]:
        st.button(
            "Clear Extracted Text",
            width="stretch",
            on_click=clear_extracted_text,
        )

    st.caption("Fix any extraction mistakes here, then replace or append it into the main note before analysis.")


def render_response(response: dict) -> None:
    codes = response.get("codes", [])
    if not codes:
        st.warning(response.get("message", "No strong ICD suggestion was found."))
        return

    top = codes[0]
    primary_icd = response.get("primary_icd") or top["code"]
    secondary_icd = response.get("secondary_icd") or []
    st.markdown(
        f"""
        <div class="top-result">
            <div class="top-code">{top['code']} - {top['title']}</div>
            <div class="top-sub">LLM-prioritized primary ICD-10 suggestion grounded against the local MIMIC-backed catalog.</div>
            <div class="chip-row">
                <span class="chip">Primary: {primary_icd}</span>
                <span class="chip">Secondary: {len(secondary_icd)}</span>
                <span class="chip">Confidence: {top.get('confidence', 'Unknown')}</span>
                <span class="chip">Returned codes: {len(codes)}</span>
                <span class="chip">Retrieved context: {len(response.get('retrieved_context', []))}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if response.get("summary"):
        st.markdown("### Case Summary")
        st.write(response["summary"])

    if response.get("prioritization_reasoning"):
        st.markdown("### Coding Priority")
        st.write(response["prioritization_reasoning"])
        if secondary_icd:
            st.caption("Secondary ICD codes retained: " + ", ".join(secondary_icd))
        if response.get("dropped_codes"):
            st.caption("Dropped candidate codes: " + ", ".join(response["dropped_codes"]))

    if response.get("condition_overview"):
        st.markdown("### Condition Overview")
        st.write(response["condition_overview"])

    precautions = response.get("precautions") or []
    diet_advice = response.get("diet_advice") or []
    if precautions or diet_advice:
        care_col, diet_col = st.columns(2, gap="large")
        with care_col:
            if precautions:
                st.markdown("### Precautions")
                render_guidance_list(precautions)
        with diet_col:
            if diet_advice:
                st.markdown("### Health & Diet")
                render_guidance_list(diet_advice)
        st.caption("These care suggestions are general guidance and should not replace a clinician's advice.")

    prediction_tab, grounding_tab = st.tabs(["Predictions", "Retrieved Context"])

    with prediction_tab:
        for index, suggestion in enumerate(codes, start=1):
            render_suggestion(suggestion, index)

    with grounding_tab:
        if response.get("retrieved_context"):
            for item in response["retrieved_context"]:
                st.markdown(
                    f"- **{item['code']} - {item['title']}** | similarity={item['similarity_score']:.3f}"
                )
                st.caption(item["chunk_text"])
        else:
            st.caption("No retrieved context was returned.")


def render_lookup_response(response: dict) -> None:
    if response.get("found"):
        st.markdown(
            f"""
            <div class="top-result">
                <div class="top-code">{response['normalized_code']} - {response['title']}</div>
                <div class="top-sub">Catalog-backed ICD-10 lookup result from the API.</div>
                <div class="chip-row">
                    <span class="chip">Catalog source: {response['catalog_source']}</span>
                    <span class="chip">Keywords: {len(response.get('keywords', []))}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Official title**")
            st.write(response["title"])
            st.markdown("**Grounded ICD definition**")
            st.write(response.get("definition") or "No definition available.")
        with col2:
            st.markdown("**Coding hint**")
            st.write(response.get("explanation_hint") or "No coding hint available.")
            st.markdown("**Keywords**")
            if response.get("keywords"):
                st.markdown(", ".join(f"`{keyword}`" for keyword in response["keywords"]))
            else:
                st.caption("No keywords available for this code.")
        return

    st.warning(f"No exact ICD code match was found for `{response.get('normalized_code') or response.get('raw_code')}`.")
    if response.get("suggestions"):
        st.markdown("**Did you mean one of these codes?**")
        for suggestion in response["suggestions"]:
            st.markdown(f"- `{suggestion['code']}` - {suggestion['title']}")
    else:
        st.info("No close ICD code suggestions were found in the current catalog.")


def render_lookup_results(payload: dict) -> None:
    results = payload.get("results")
    if isinstance(results, list):
        if not results:
            st.warning("No ICD codes were provided for lookup.")
            return
        for index, response in enumerate(results, start=1):
            st.markdown(f"### {index}. `{response.get('normalized_code') or response.get('raw_code')}`")
            render_lookup_response(response)
        return
    render_lookup_response(payload)


def main() -> None:
    inject_styles()
    stats = load_dataset_stats()
    dataset_examples = load_dataset_examples()
    health = load_api_health(API_BASE_URL)
    catalog_size = int(health.get("catalog_size") or 0) or stats["icd10_rows"]

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">MediChar ICD-10 Workbench</div>
            <div class="hero-copy">
                Turn clinical notes, diagnosis summaries, and note images into grounded ICD-10 suggestions
                with case-aware explanations, ranked matches, and fast code lookup in one place.
            </div>
            <div class="hero-badge-row">
                <span class="hero-badge">Paste a note</span>
                <span class="hero-badge">Upload handwriting or screenshots</span>
                <span class="hero-badge">Review ranked code matches</span>
                <span class="hero-badge">Look up any ICD code directly</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_spotlight_card(
            "Start Fast",
            "Paste any clinical note",
            "Drop in an assessment, discharge summary, final diagnosis, or symptom-heavy note and get ranked ICD-10 suggestions.",
            "Best for quick note-to-code workflows.",
        )
    with metric_cols[1]:
        render_spotlight_card(
            "Image Input",
            "Upload note photos",
            "Send printed or handwritten note images through the OCR pipeline, then clean the extracted text before analysis.",
            "Useful for handwritten charts and screenshots.",
        )
    with metric_cols[2]:
        render_spotlight_card(
            "Grounded Review",
            "See why codes were chosen",
            "Read plain-language summaries, case-aware explanations, precautions, and the retrieved ICD context behind each result.",
            "Helps you review the reasoning, not just the answer.",
        )
    with metric_cols[3]:
        render_spotlight_card(
            "Direct Lookup",
            "Search any ICD code",
            "Jump from code to disease title, definitions, hints, and close suggestions when you already know the code family.",
            "Great for verification and quick reference.",
        )

    if health.get("status") != "ok":
        st.error(
            "Prediction and lookup are unavailable until the local service is running. Start it with `uvicorn backend.main:app --reload`."
        )

    mode = st.radio(
        "Workflow mode",
        ["Disease to Code", "Code to Disease"],
        horizontal=True,
    )

    left_col, right_col = st.columns([1.25, 0.75], gap="large")

    with left_col:
        st.markdown(f"## {mode}")
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)

        if "note_mode_input" not in st.session_state:
            st.session_state.note_mode_input = ""
        if "note_mode_ocr_text" not in st.session_state:
            st.session_state.note_mode_ocr_text = ""
        if "note_mode_ocr_raw_text" not in st.session_state:
            st.session_state.note_mode_ocr_raw_text = ""
        if "note_mode_ocr_cleaned_text" not in st.session_state:
            st.session_state.note_mode_ocr_cleaned_text = ""
        if "note_mode_ocr_processed_text" not in st.session_state:
            st.session_state.note_mode_ocr_processed_text = ""
        if "code_mode_input" not in st.session_state:
            st.session_state.code_mode_input = ""

        if mode == "Disease to Code":
            example_options = {item["label"]: item for item in dataset_examples}
            selected_example = st.selectbox(
                "Quick dataset example",
                ["Custom note"] + list(example_options.keys()),
                index=0,
                key="note_mode_example",
            )

            uploaded_note_image = st.file_uploader(
                "Upload description image",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
                key="note_mode_image",
                help="Upload a photo or screenshot containing the medical description or note, including handwritten text.",
            )
            ocr_mode_label = st.radio(
                "Image OCR mode",
                ["Handwritten", "Printed"],
                horizontal=True,
                help="Handwritten uses local TrOCR directly. Printed uses Windows OCR for typed documents.",
            )
            ocr_mode = ocr_mode_label.lower()

            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("Load Dataset Example", width="stretch"):
                    if selected_example != "Custom note":
                        st.session_state.note_mode_input = example_options[selected_example]["note_text"]
            with action_cols[1]:
                extract_clicked = st.button(
                    "Extract Text From Image",
                    width="stretch",
                    disabled=uploaded_note_image is None,
                )

            if uploaded_note_image is not None:
                st.image(uploaded_note_image, caption="Uploaded note image", width="stretch")

            if extract_clicked and uploaded_note_image is not None:
                try:
                    with st.spinner("Reading the uploaded note with the local OCR pipeline..."):
                        extraction = extract_note_from_image_api(
                            API_BASE_URL,
                            uploaded_note_image,
                            supplemental_text=st.session_state.note_mode_input.strip(),
                            ocr_mode=ocr_mode,
                        )
                    st.session_state.note_mode_ocr_text = str(extraction.get("text") or "").strip()
                    st.session_state.note_mode_ocr_raw_text = str(extraction.get("raw_text") or "").strip()
                    st.session_state.note_mode_ocr_cleaned_text = str(extraction.get("cleaned_text") or "").strip()
                    st.session_state.note_mode_ocr_processed_text = str(extraction.get("processed_text") or "").strip()
                    confidence = extraction.get("confidence")
                    uncertain = extraction.get("uncertain_fragments") or []
                    if confidence or uncertain:
                        detail_bits: list[str] = []
                        if confidence:
                            detail_bits.append(f"Confidence: {confidence}")
                        if uncertain:
                            detail_bits.append(
                                "Unclear: " + ", ".join(str(item) for item in uncertain[:3])
                            )
                        st.caption(" | ".join(detail_bits))
                except requests.Timeout:
                    st.error(
                        "Image extraction timed out. The backend OCR may still be loading a model for the first request. "
                        "Please wait a moment and try again."
                    )
                except requests.RequestException as exc:
                    st.error(f"Image extraction failed: {exc}")

            if st.session_state.note_mode_ocr_text:
                render_ocr_editor()

            st.text_area(
                "Disease / clinical note",
                key="note_mode_input",
                height=220,
                placeholder="Paste the clinical note, assessment, symptom summary, or diagnosed disease here...",
            )

            if st.session_state.note_mode_ocr_text:
                st.caption("Extracted note text is available above. You can analyze that corrected text directly or copy it into the main note box.")

            analyze = st.button("Analyze Note", type="primary", width="stretch")
            if analyze:
                clinical_text = st.session_state.note_mode_input.strip()
                if clinical_text:
                    try:
                        with st.spinner("Requesting ICD predictions from the backend..."):
                            st.session_state.note_mode_response = predict_note(API_BASE_URL, clinical_text)
                    except requests.RequestException as exc:
                        st.error(f"Prediction request failed: {exc}")
                else:
                    st.warning("Add note text before analysis. If you uploaded an image, extract it first, then replace or append the reviewed text into the main note box.")
        else:
            code_options = {item["label"]: item["code"] for item in dataset_examples}
            selected_code = st.selectbox(
                "Quick dataset code",
                ["Custom code"] + list(code_options.keys()),
                index=0,
                key="code_mode_example",
            )

            if st.button("Load Dataset Code", width="content"):
                if selected_code != "Custom code":
                    st.session_state.code_mode_input = code_options[selected_code]

            st.text_area(
                "ICD code(s)",
                key="code_mode_input",
                height=110,
                placeholder="Enter one or more ICD codes like I10, E119, or J189. You can separate multiple codes with commas, spaces, or new lines.",
            )

            lookup = st.button("Lookup Code", type="primary", width="stretch")
            if lookup and st.session_state.code_mode_input.strip():
                try:
                    with st.spinner("Looking up the ICD code through the backend API..."):
                        parsed_codes = parse_lookup_codes(st.session_state.code_mode_input)
                        if len(parsed_codes) <= 1:
                            st.session_state.code_mode_response = lookup_code(API_BASE_URL, parsed_codes[0])
                        else:
                            st.session_state.code_mode_response = lookup_codes(API_BASE_URL, parsed_codes)
                except requests.RequestException as exc:
                    st.error(f"Lookup request failed: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown(
            """
            <div class="info-board">
                <div class="panel-kicker">Start Here</div>
                <div class="panel-title">Use MediChar like a coding co-pilot</div>
                <div class="panel-copy">
                    This workspace is built to help you move from messy clinical text to grounded ICD-10 suggestions
                    without losing visibility into how the result was formed.
                </div>
            """,
            unsafe_allow_html=True,
        )
        if mode == "Disease to Code":
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-no">Step 1</div>
                    <div class="step-title">Choose your input style</div>
                    <div class="step-copy">Paste a fresh note, load a quick demo example, or upload an image if the source is handwritten or scanned.</div>
                </div>
                <div class="step-card">
                    <div class="step-no">Step 2</div>
                    <div class="step-title">Review the extracted text</div>
                    <div class="step-copy">If you use an image, edit the OCR output first so the model sees the clearest possible clinical context.</div>
                </div>
                <div class="step-card">
                    <div class="step-no">Step 3</div>
                    <div class="step-title">Analyze and compare</div>
                    <div class="step-copy">Run the note to get ranked ICD-10 candidates, grounded explanations, retrieved context, and practical case guidance.</div>
                </div>
                <div class="value-pill-row">
                    <span class="value-pill">Case summary</span>
                    <span class="value-pill">Condition overview</span>
                    <span class="value-pill">Precautions</span>
                    <span class="value-pill">Health and diet advice</span>
                    <span class="value-pill">Ranked ICD suggestions</span>
                    <span class="value-pill">Retrieved context trail</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="step-card">
                    <div class="step-no">Lookup</div>
                    <div class="step-title">Enter a full or partial ICD code</div>
                    <div class="step-copy">Use this mode when you already know the code family and want the title, definition, coding hint, or nearby matches fast.</div>
                </div>
                <div class="value-pill-row">
                    <span class="value-pill">Official title</span>
                    <span class="value-pill">Grounded definition</span>
                    <span class="value-pill">Coding hint</span>
                    <span class="value-pill">Nearby suggestions</span>
                </div>
                <div class="catalog-pulse">
                    Results are grounded in the local ICD dictionary loaded by the API, so this mode works well for quick verification before you return to note analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if mode == "Disease to Code" and "note_mode_response" in st.session_state:
        st.markdown("## Results")
        render_response(st.session_state.note_mode_response)
    elif mode == "Code to Disease" and "code_mode_response" in st.session_state:
        st.markdown("## Results")
        render_lookup_results(st.session_state.code_mode_response)
    else:
        st.info("Run a prediction or code lookup once the backend API is up.")


if __name__ == "__main__":
    main()
