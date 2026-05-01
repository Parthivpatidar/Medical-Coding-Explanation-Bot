from __future__ import annotations

import re
from collections import OrderedDict

from backend.services.retriever import extract_focus_phrases


FEATURE_CATEGORIES = (
    "diagnosis_terms",
    "anatomy",
    "temporal",
    "severity",
    "diagnostics",
    "qualifiers",
    "symptoms",
    "symptom_clusters",
)

TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9+-]*|\d+(?:\.\d+)?%?")
SPACE_PATTERN = re.compile(r"\s+")

ANATOMY_DESCRIPTORS = {
    "anterior",
    "apical",
    "basal",
    "bilateral",
    "central",
    "distal",
    "inferior",
    "lateral",
    "left",
    "lower",
    "medial",
    "middle",
    "posterior",
    "proximal",
    "right",
    "superior",
    "upper",
    "unilateral",
}
ANATOMY_TERMS = {
    "abdomen",
    "artery",
    "back",
    "bladder",
    "bone",
    "brain",
    "bronchus",
    "chest",
    "colon",
    "extremity",
    "finger",
    "foot",
    "hand",
    "heart",
    "hip",
    "joint",
    "kidney",
    "knee",
    "liver",
    "lobe",
    "lung",
    "muscle",
    "neck",
    "nerve",
    "pelvis",
    "segment",
    "shoulder",
    "skin",
    "spine",
    "stomach",
    "throat",
    "vein",
    "wall",
}
ANATOMY_CANONICAL = {
    "arterial": "artery",
    "bronchial": "bronchus",
    "cardiac": "heart",
    "cerebral": "brain",
    "hepatic": "liver",
    "lobar": "lobe",
    "lungs": "lung",
    "myocardial": "heart",
    "pulmonary": "lung",
    "renal": "kidney",
    "ventricular": "heart",
}

TEMPORAL_TERMS = (
    "acute",
    "chronic",
    "subacute",
    "recurrent",
    "recurring",
    "persistent",
    "sudden onset",
    "new onset",
    "progressive",
    "worsening",
    "exacerbation",
    "flare",
    "relapse",
)

SEVERITY_TERMS = (
    "mild",
    "moderate",
    "severe",
    "critical",
    "unstable",
    "hypoxia",
    "hypoxic",
    "hypoxemia",
    "oxygen saturation",
    "respiratory distress",
    "altered mental status",
    "shock",
    "sepsis",
    "intubation",
    "ventilator",
    "mechanical ventilation",
    "oxygen therapy",
    "supplemental oxygen",
    "iv",
    "intravenous",
    "icu",
    "vasopressor",
)

DIAGNOSTIC_TERMS = (
    "x-ray",
    "xray",
    "cxr",
    "ct",
    "mri",
    "ultrasound",
    "ecg",
    "ekg",
    "lab",
    "culture",
    "biopsy",
    "imaging",
    "troponin",
    "wbc",
    "creatinine",
    "glucose",
    "st elevation",
    "st depression",
    "q wave",
    "infiltrate",
    "opacity",
    "consolidation",
    "effusion",
    "fracture",
    "mass",
    "lesion",
    "abscess",
    "obstruction",
)

QUALIFIER_TERMS = (
    "confirmed",
    "diagnosed",
    "assessment",
    "impression",
    "suspected",
    "possible",
    "probable",
    "rule out",
    "ruled out",
    "no evidence of",
    "primary",
    "secondary",
    "complication",
    "complicated",
    "associated",
    "due to",
    "with",
    "without",
    "unspecified",
)

SYMPTOM_CLUSTERS = OrderedDict(
    {
        "respiratory": {
            "cough",
            "shortness of breath",
            "dyspnea",
            "wheezing",
            "sputum",
            "productive cough",
            "chest tightness",
        },
        "cardiac": {
            "chest pain",
            "palpitations",
            "syncope",
            "edema",
            "orthopnea",
        },
        "neurologic": {
            "headache",
            "weakness",
            "numbness",
            "confusion",
            "seizure",
            "dizziness",
        },
        "gastrointestinal": {
            "abdominal pain",
            "nausea",
            "vomiting",
            "diarrhea",
            "constipation",
        },
        "genitourinary": {
            "dysuria",
            "frequency",
            "urgency",
            "flank pain",
            "hematuria",
        },
        "musculoskeletal": {
            "back pain",
            "joint pain",
            "swelling",
            "stiffness",
            "tenderness",
        },
        "constitutional": {
            "fever",
            "chills",
            "fatigue",
            "weakness",
            "weight loss",
        },
    }
)

NEGATION_PATTERN = re.compile(
    r"(?:^|\b)(?:no|denies|denied|without|negative for|rule out|ruled out)\b"
    r"(?:\W+\w+){0,5}\W*$",
    re.IGNORECASE,
)


def extract_semantic_features(text: str) -> dict[str, list[str]]:
    normalized = _normalize_text(text)
    features: dict[str, set[str]] = {category: set() for category in FEATURE_CATEGORIES}

    _extract_diagnosis_terms(text, features["diagnosis_terms"])
    _extract_anatomy(normalized, features)
    _extract_terms(normalized, TEMPORAL_TERMS, features["temporal"])
    _extract_duration(normalized, features["temporal"])
    _extract_terms(normalized, SEVERITY_TERMS, features["severity"])
    _extract_vitals(normalized, features["severity"])
    _extract_terms(normalized, DIAGNOSTIC_TERMS, features["diagnostics"])
    _extract_lab_evidence(normalized, features["diagnostics"])
    _extract_terms(normalized, QUALIFIER_TERMS, features["qualifiers"], allow_negated=True)
    _extract_structural_qualifiers(normalized, features["qualifiers"])
    _extract_symptoms(normalized, features)

    return {
        category: sorted(values)
        for category, values in features.items()
    }


def flattened_features(features: dict[str, list[str]]) -> set[str]:
    flattened: set[str] = set()
    for category, values in features.items():
        for value in values:
            flattened.add(f"{category}:{value}")
    return flattened


def _normalize_text(text: str) -> str:
    normalized = str(text or "").lower()
    normalized = normalized.replace("o2", "oxygen")
    normalized = normalized.replace("spo2", "oxygen saturation")
    normalized = normalized.replace("cxr", "chest x-ray")
    normalized = normalized.replace("ekg", "ecg")
    normalized = re.sub(r"[\[\]{}()]", " ", normalized)
    normalized = SPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def _tokens(text: str) -> list[str]:
    return [_canonical_token(token) for token in TOKEN_PATTERN.findall(text)]


def _canonical_token(token: str) -> str:
    token = ANATOMY_CANONICAL.get(token, token)
    if token.endswith("ies") and len(token) > 4:
        token = f"{token[:-3]}y"
    elif token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        token = token[:-1]
    return ANATOMY_CANONICAL.get(token, token)


def _extract_anatomy(text: str, features: dict[str, set[str]]) -> None:
    tokens = _tokens(text)
    for index, token in enumerate(tokens):
        if token in ANATOMY_DESCRIPTORS:
            features["anatomy"].add(token)
            continue
        if token not in ANATOMY_TERMS:
            continue

        features["anatomy"].add(token)
        window = tokens[max(0, index - 3): min(len(tokens), index + 4)]
        descriptors = [item for item in window if item in ANATOMY_DESCRIPTORS]
        if descriptors:
            phrase_parts = [*dict.fromkeys(descriptors), token]
            features["anatomy"].add(" ".join(phrase_parts))


def _extract_diagnosis_terms(text: str, target: set[str]) -> None:
    for phrase in extract_focus_phrases(text, max_phrases=6):
        canonical = _canonical_phrase(phrase)
        if canonical:
            target.add(canonical)
        for fragment in re.split(r"\b(?:with|and|secondary to|complicated by|due to)\b", phrase):
            normalized_fragment = _canonical_phrase(fragment)
            if len(normalized_fragment) >= 4:
                target.add(normalized_fragment)


def _extract_terms(
    text: str,
    terms: tuple[str, ...],
    target: set[str],
    *,
    allow_negated: bool = False,
) -> None:
    for term in terms:
        for match in _iter_phrase_matches(text, term):
            if not allow_negated and _is_negated(text, match.start()):
                continue
            target.add(_canonical_phrase(term))


def _extract_duration(text: str, target: set[str]) -> None:
    for match in re.finditer(r"\b(\d+)\s*[- ]\s*(day|week|month|year)s?\b", text):
        if _is_negated(text, match.start()):
            continue
        target.add(match.group(2))
        target.add("duration")


def _extract_vitals(text: str, target: set[str]) -> None:
    for value in _extract_labeled_numbers(text, ("temp", "temperature", "t")):
        target.add("temperature")
        if value >= 100.4:
            target.add("fever")
        elif value <= 95.0:
            target.add("hypothermia")

    for value in _extract_labeled_numbers(text, ("rr", "respiratory rate")):
        target.add("respiratory rate")
        if value >= 22:
            target.add("tachypnea")

    for value in _extract_labeled_numbers(text, ("hr", "heart rate")):
        target.add("heart rate")
        if value >= 100:
            target.add("tachycardia")

    for match in re.finditer(r"\b(?:oxygen saturation|sat)\s*[:=]?\s*(\d{2,3})(?:\s*%)?", text):
        value = float(match.group(1))
        target.add("oxygen saturation")
        if value < 90:
            target.add("severe hypoxemia")
        elif value < 92:
            target.add("hypoxemia")

    for match in re.finditer(r"\bbp\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})", text):
        systolic = float(match.group(1))
        diastolic = float(match.group(2))
        target.add("blood pressure")
        if systolic < 90:
            target.add("hypotension")
        elif systolic >= 140 or diastolic >= 90:
            target.add("hypertension")


def _extract_labeled_numbers(text: str, labels: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    label_pattern = "|".join(re.escape(label) for label in labels)
    for match in re.finditer(rf"\b(?:{label_pattern})\s*[:=]?\s*(\d+(?:\.\d+)?)", text):
        values.append(float(match.group(1)))
    return values


def _extract_lab_evidence(text: str, target: set[str]) -> None:
    for match in re.finditer(r"\b(elevated|high|low|positive|negative)\s+([a-z][a-z0-9 -]{2,30})", text):
        value = f"{match.group(1)} {_canonical_phrase(match.group(2))}"
        target.add(value.strip())


def _extract_structural_qualifiers(text: str, target: set[str]) -> None:
    if re.search(r"\b(?:assessment|diagnosis|diagnosed|impression|dx)\s*:", text):
        target.add("confirmed")
    if re.search(r"\b(?:no evidence of|negative for|ruled out|rule out)\b", text):
        target.add("ruled out")


def _extract_symptoms(text: str, features: dict[str, set[str]]) -> None:
    for cluster_name, symptoms in SYMPTOM_CLUSTERS.items():
        present: list[str] = []
        for symptom in symptoms:
            for match in _iter_phrase_matches(text, symptom):
                if _is_negated(text, match.start()):
                    continue
                canonical = _canonical_phrase(symptom)
                features["symptoms"].add(canonical)
                present.append(canonical)
                break
        unique_present = sorted(set(present))
        if len(unique_present) >= 2:
            features["symptom_clusters"].add(f"{cluster_name}:{'+'.join(unique_present)}")


def _iter_phrase_matches(text: str, phrase: str) -> list[re.Match[str]]:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    pattern = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")
    return list(pattern.finditer(text))


def _is_negated(text: str, start: int) -> bool:
    before = text[max(0, start - 80): start]
    match = NEGATION_PATTERN.search(before)
    if not match:
        return False
    negated_span = before[match.start():]
    if re.search(r"\b(?:but|however|has|reports|reported|complains|complained)\b", negated_span):
        return False
    return True


def _canonical_phrase(value: str) -> str:
    tokens = [_canonical_token(token) for token in TOKEN_PATTERN.findall(value)]
    return " ".join(tokens)
