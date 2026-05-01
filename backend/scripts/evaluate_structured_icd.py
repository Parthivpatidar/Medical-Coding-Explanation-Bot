from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import RAGSettings
from backend.services.data_loader import load_icd_catalog
from backend.services.medical_coding import MedicalCodingService


PRIMARY_PREDICTOR_AVAILABLE = True


Difficulty = str


@dataclass(frozen=True, slots=True)
class StructuredTestCase:
    admission_type: str
    age: int
    gender: str
    diagnoses: list[str]
    clinical_events: list[str]
    lab_flags: list[str]
    procedures: list[str]
    expected_icd: list[str]
    primary_icd: str
    difficulty: Difficulty

    def to_dict(self) -> dict[str, Any]:
        return {
            "admission_type": self.admission_type,
            "age": self.age,
            "gender": self.gender,
            "diagnoses": list(self.diagnoses),
            "clinical_events": list(self.clinical_events),
            "lab_flags": list(self.lab_flags),
            "procedures": list(self.procedures),
            "expected_icd": list(self.expected_icd),
            "primary_icd": self.primary_icd,
            "difficulty": self.difficulty,
        }


def generate_structured_test_cases() -> list[dict[str, Any]]:
    cases = [
        StructuredTestCase(
            admission_type="Emergency",
            age=67,
            gender="M",
            diagnoses=["Essential hypertension"],
            clinical_events=["Elevated blood pressure on arrival", "Antihypertensive therapy continued"],
            lab_flags=["No acute end-organ injury"],
            procedures=["Blood pressure monitoring"],
            expected_icd=["I10"],
            primary_icd="I10",
            difficulty="easy",
        ),
        StructuredTestCase(
            admission_type="Urgent",
            age=59,
            gender="F",
            diagnoses=["Type 2 diabetes mellitus without complications"],
            clinical_events=["Hyperglycemia managed medically", "Diet counseling provided"],
            lab_flags=["Elevated glucose"],
            procedures=["Glucose monitoring"],
            expected_icd=["E119"],
            primary_icd="E119",
            difficulty="easy",
        ),
        StructuredTestCase(
            admission_type="Emergency",
            age=72,
            gender="M",
            diagnoses=["Pneumonia"],
            clinical_events=["Productive cough", "Fever", "Shortness of breath"],
            lab_flags=["Chest x-ray infiltrate", "Low oxygen saturation"],
            procedures=["Chest x-ray", "Oxygen therapy", "IV antibiotics"],
            expected_icd=["J189"],
            primary_icd="J189",
            difficulty="easy",
        ),
        StructuredTestCase(
            admission_type="Emergency",
            age=76,
            gender="F",
            diagnoses=["Acute kidney injury", "Dehydration"],
            clinical_events=["Poor oral intake", "Hypotension improved with fluids"],
            lab_flags=["Elevated creatinine", "Elevated BUN"],
            procedures=["IV fluid resuscitation"],
            expected_icd=["N17", "E860"],
            primary_icd="N17",
            difficulty="medium",
        ),
        StructuredTestCase(
            admission_type="Emergency",
            age=81,
            gender="M",
            diagnoses=["Urinary tract infection", "Sepsis"],
            clinical_events=["Fever", "Altered mental status", "Tachycardia"],
            lab_flags=["Positive urine culture", "Elevated lactate", "Leukocytosis"],
            procedures=["IV antibiotics", "Fluid bolus"],
            expected_icd=["A419", "N390"],
            primary_icd="A419",
            difficulty="medium",
        ),
        StructuredTestCase(
            admission_type="Inpatient",
            age=69,
            gender="F",
            diagnoses=["Heart failure", "Chronic kidney disease", "Essential hypertension"],
            clinical_events=["Volume overload", "Lower extremity edema", "Dyspnea on exertion"],
            lab_flags=["Elevated BNP", "Reduced eGFR"],
            procedures=["Diuretic therapy", "Telemetry monitoring"],
            expected_icd=["I509", "N189", "I10"],
            primary_icd="I509",
            difficulty="medium",
        ),
        StructuredTestCase(
            admission_type="Urgent",
            age=63,
            gender="M",
            diagnoses=["Type 2 diabetes mellitus", "Chronic kidney disease"],
            clinical_events=["Medication review", "Nephrology follow-up recommended"],
            lab_flags=["Persistent albuminuria", "Reduced eGFR"],
            procedures=["Metabolic panel"],
            expected_icd=["E119", "N189"],
            primary_icd="E119",
            difficulty="medium",
        ),
        StructuredTestCase(
            admission_type="Emergency",
            age=78,
            gender="F",
            diagnoses=["Sepsis", "Pneumonia", "Acute kidney injury", "Chronic kidney disease"],
            clinical_events=["Hypotension", "Fever", "Respiratory distress", "Acute-on-chronic renal dysfunction"],
            lab_flags=["Chest x-ray infiltrate", "Elevated lactate", "Elevated creatinine", "Reduced baseline eGFR"],
            procedures=["IV antibiotics", "Oxygen therapy", "Fluid resuscitation"],
            expected_icd=["A419", "J189", "N17", "N189"],
            primary_icd="A419",
            difficulty="hard",
        ),
        StructuredTestCase(
            admission_type="Emergency",
            age=73,
            gender="M",
            diagnoses=["Acute on chronic heart failure", "Chronic kidney disease", "Essential hypertension", "Type 2 diabetes mellitus"],
            clinical_events=["Pulmonary edema", "Volume overload", "Acute dyspnea"],
            lab_flags=["Elevated BNP", "Reduced eGFR", "Hyperglycemia"],
            procedures=["IV diuretics", "Oxygen therapy", "Cardiac monitoring"],
            expected_icd=["I509", "N189", "I10", "E119"],
            primary_icd="I509",
            difficulty="hard",
        ),
        StructuredTestCase(
            admission_type="Emergency",
            age=66,
            gender="F",
            diagnoses=["Sepsis", "Urinary tract infection", "Acute kidney injury", "Dehydration", "Type 2 diabetes mellitus"],
            clinical_events=["Confusion", "Tachycardia", "Poor oral intake", "Weakness"],
            lab_flags=["Positive urine culture", "Elevated creatinine", "Elevated lactate", "Hyperglycemia"],
            procedures=["IV antibiotics", "IV fluids", "Glucose monitoring"],
            expected_icd=["A419", "N390", "N17", "E860", "E119"],
            primary_icd="A419",
            difficulty="hard",
        ),
    ]
    return [case.to_dict() for case in cases]


def build_structured_case_text(case: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Structured Admission Summary",
            f"Admission Type: {case['admission_type']}",
            f"Age: {case['age']}",
            f"Gender: {case['gender']}",
            f"Admission Diagnosis Priority: {case['diagnoses'][0] if case['diagnoses'] else 'Unknown'}",
            f"Diagnoses: {', '.join(case.get('diagnoses', [])) or 'None'}",
            f"Clinical Events: {', '.join(case.get('clinical_events', [])) or 'None'}",
            f"Lab Flags: {', '.join(case.get('lab_flags', [])) or 'None'}",
            f"Procedures: {', '.join(case.get('procedures', [])) or 'None'}",
            (
                "Coding Rule: prioritize the admission diagnosis first, then independently capture "
                "secondary conditions, complications, and acute-on-chronic problems when supported."
            ),
        ]
    )


@lru_cache(maxsize=1)
def get_prediction_service() -> MedicalCodingService:
    return MedicalCodingService(settings=RAGSettings())


def predict_icd(case: dict[str, Any]) -> list[str]:
    global PRIMARY_PREDICTOR_AVAILABLE
    if not PRIMARY_PREDICTOR_AVAILABLE:
        return predict_icd_offline(case)
    try:
        service = get_prediction_service()
        response = service.predict(
            clinical_text=build_structured_case_text(case),
            top_k=8,
            return_context=False,
            return_similarity=False,
        )
        codes = response.get("codes", [])
        parsed = [str(item.get("code", "")).strip().upper() for item in codes if str(item.get("code", "")).strip()]
        if parsed:
            return parsed
    except Exception as exc:
        PRIMARY_PREDICTOR_AVAILABLE = False
        print(f"[structured-eval] Primary predictor unavailable, using offline fallback: {exc}")
    return predict_icd_offline(case)


@lru_cache(maxsize=1)
def get_catalog_entries() -> list[Any]:
    return load_icd_catalog()


def predict_icd_offline(case: dict[str, Any], max_codes: int = 5) -> list[str]:
    catalog_entries = get_catalog_entries()
    diagnosis_terms = [str(item).strip().lower() for item in case.get("diagnoses", []) if str(item).strip()]
    lab_terms = [str(item).strip().lower() for item in case.get("lab_flags", []) if str(item).strip()]
    event_terms = [str(item).strip().lower() for item in case.get("clinical_events", []) if str(item).strip()]
    combined_terms = diagnosis_terms + lab_terms + event_terms

    scored: list[tuple[float, str]] = []
    for entry in catalog_entries:
        entry_text = " ".join(
            [
                str(entry.title or "").lower(),
                str(entry.definition or "").lower(),
                " ".join(str(keyword).lower() for keyword in entry.keywords),
            ]
        )
        score = 0.0
        for diagnosis in diagnosis_terms:
            if diagnosis and diagnosis in entry_text:
                score += 3.5
            else:
                score += _token_overlap_score(diagnosis, entry_text) * 2.4
        for term in combined_terms:
            if not term:
                continue
            score += _token_overlap_score(term, entry_text) * 0.7
        if score > 0:
            scored.append((score, _normalize_code(entry.code)))

    scored.sort(reverse=True)
    deduped: list[str] = []
    seen: set[str] = set()
    for _, code in scored:
        if not code or code in seen:
            continue
        deduped.append(code)
        seen.add(code)
        if len(deduped) >= max_codes:
            break
    return deduped


def evaluate_prediction(case: dict[str, Any], predicted_icd: list[str]) -> dict[str, Any]:
    expected = {_normalize_code(code) for code in case["expected_icd"]}
    predicted = [_normalize_code(code) for code in predicted_icd if _normalize_code(code)]
    predicted_set = set(predicted)
    intersection = expected & predicted_set
    union = expected | predicted_set

    primary_match = bool(predicted) and predicted[0] == _normalize_code(case["primary_icd"])
    exact_match = expected == predicted_set
    partial_match_score = round(len(intersection) / (len(union) or 1), 4)
    missed_codes = sorted(expected - predicted_set)
    wrong_codes = sorted(predicted_set - expected)
    error_types = infer_error_types(
        primary_match=primary_match,
        missed_codes=missed_codes,
        wrong_codes=wrong_codes,
    )

    return {
        "primary_match": primary_match,
        "exact_match": exact_match,
        "partial_match_score": partial_match_score,
        "missed_codes": missed_codes,
        "wrong_codes": wrong_codes,
        "error_types": error_types,
        "predicted_icd": predicted,
    }


def infer_error_types(
    *,
    primary_match: bool,
    missed_codes: list[str],
    wrong_codes: list[str],
) -> list[str]:
    error_types: list[str] = []
    if not primary_match:
        error_types.append("Wrong primary")
    if missed_codes:
        error_types.append("Missing critical code")
    if wrong_codes:
        error_types.append("Extra irrelevant code")
    return error_types


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    predictor: Callable[[dict[str, Any]], list[str]] = predict_icd,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_rows: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        predicted_icd = predictor(case)
        evaluation = evaluate_prediction(case, predicted_icd)
        case_rows.append(
            {
                "case_id": f"CASE-{index:02d}",
                "difficulty": case["difficulty"],
                "admission_type": case["admission_type"],
                "primary_icd": _normalize_code(case["primary_icd"]),
                "expected_icd": ", ".join(sorted(_normalize_code(code) for code in case["expected_icd"])),
                "predicted_icd": ", ".join(evaluation["predicted_icd"]) or "NONE",
                "primary_match": evaluation["primary_match"],
                "exact_match": evaluation["exact_match"],
                "partial_match_score": evaluation["partial_match_score"],
                "missed_codes": ", ".join(evaluation["missed_codes"]) or "-",
                "wrong_codes": ", ".join(evaluation["wrong_codes"]) or "-",
                "error_types": "; ".join(evaluation["error_types"]) or "None",
            }
        )

    cases_df = pd.DataFrame(case_rows)
    summary_df = build_summary_table(cases_df)
    failures_df = build_failures_table(cases_df)
    return cases_df, summary_df, failures_df


def build_summary_table(cases_df: pd.DataFrame) -> pd.DataFrame:
    overall_row = {
        "segment": "overall",
        "cases": int(len(cases_df)),
        "primary_accuracy_pct": round(cases_df["primary_match"].mean() * 100, 2),
        "exact_match_pct": round(cases_df["exact_match"].mean() * 100, 2),
        "avg_partial_match": round(cases_df["partial_match_score"].mean(), 4),
    }

    difficulty_rows = []
    for difficulty, group in cases_df.groupby("difficulty", sort=False):
        difficulty_rows.append(
            {
                "segment": difficulty,
                "cases": int(len(group)),
                "primary_accuracy_pct": round(group["primary_match"].mean() * 100, 2),
                "exact_match_pct": round(group["exact_match"].mean() * 100, 2),
                "avg_partial_match": round(group["partial_match_score"].mean(), 4),
            }
        )

    return pd.DataFrame([overall_row, *difficulty_rows])


def build_failures_table(cases_df: pd.DataFrame) -> pd.DataFrame:
    failures = cases_df.loc[~cases_df["exact_match"] | ~cases_df["primary_match"]].copy()
    if failures.empty:
        return failures
    return failures[
        [
            "case_id",
            "difficulty",
            "expected_icd",
            "predicted_icd",
            "missed_codes",
            "wrong_codes",
            "error_types",
        ]
    ]


def analyze_confusion_patterns(cases_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    confusion_counter: Counter[str] = Counter()
    failure_counter: Counter[str] = Counter()

    for _, row in cases_df.iterrows():
        if row["error_types"] == "None":
            continue
        for error_type in str(row["error_types"]).split("; "):
            if error_type:
                failure_counter[error_type] += 1

        expected_codes = [item.strip() for item in str(row["expected_icd"]).split(",") if item.strip()]
        predicted_codes = [item.strip() for item in str(row["predicted_icd"]).split(",") if item.strip() and item.strip() != "NONE"]

        if expected_codes and predicted_codes:
            confusion_counter[f"{expected_codes[0]} -> {predicted_codes[0]}"] += 1
        elif expected_codes and not predicted_codes:
            confusion_counter[f"{expected_codes[0]} -> NONE"] += 1

    confusion_rows = [
        {"pattern": pattern, "count": count}
        for pattern, count in confusion_counter.most_common(5)
    ]
    most_common_failure = failure_counter.most_common(1)[0][0] if failure_counter else "No dominant failure category"
    return pd.DataFrame(confusion_rows), most_common_failure


def print_evaluation_report(
    cases_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    most_common_failure: str,
) -> None:
    print("\n=== STRUCTURED ICD-10 EVALUATION ===")
    print("\nPer-case evaluation")
    print(cases_df.to_string(index=False))

    print("\nMetrics summary")
    print(summary_df.to_string(index=False))

    if not failures_df.empty:
        print("\nError analysis")
        for _, row in failures_df.iterrows():
            print(
                "\n".join(
                    [
                        f"Case ID: {row['case_id']}",
                        f"Difficulty: {row['difficulty']}",
                        f"Expected: {row['expected_icd']}",
                        f"Predicted: {row['predicted_icd']}",
                        f"Missed Codes: {row['missed_codes']}",
                        f"Wrong Codes: {row['wrong_codes']}",
                        f"Error Type: {row['error_types']}",
                    ]
                )
            )
            print("-" * 72)
    else:
        print("\nError analysis")
        print("No failed cases.")

    print("\nConfusion patterns")
    if confusion_df.empty:
        print("No confusion patterns detected.")
    else:
        print(confusion_df.to_string(index=False))

    print(f"\nMost common failure category: {most_common_failure}")


def _normalize_code(code: str) -> str:
    return "".join(character for character in str(code or "").upper() if character.isalnum())


def _token_overlap_score(term: str, entry_text: str) -> float:
    term_tokens = {token for token in term.replace("/", " ").split() if len(token) > 2}
    if not term_tokens:
        return 0.0
    entry_tokens = {token for token in entry_text.replace("/", " ").split() if len(token) > 2}
    overlap = term_tokens & entry_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(term_tokens)


def main() -> None:
    cases = generate_structured_test_cases()
    cases_df, summary_df, failures_df = evaluate_cases(cases)
    confusion_df, most_common_failure = analyze_confusion_patterns(cases_df)
    print_evaluation_report(cases_df, summary_df, failures_df, confusion_df, most_common_failure)


if __name__ == "__main__":
    main()
