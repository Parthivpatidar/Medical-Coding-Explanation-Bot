from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = PROCESSED_DIR / "vector_store"
MIMIC_DEMO_DIR = DATA_DIR / "mimic-iv-clinical-database-demo-2.2"
MIMIC_HOSP_DIR = MIMIC_DEMO_DIR / "hosp"
MIMIC_ICD_DIAGNOSES_PATH = MIMIC_HOSP_DIR / "d_icd_diagnoses.csv"
ICD_KB_PATH = Path(os.getenv("MEDICHAR_ICD_KB_PATH", str(MIMIC_ICD_DIAGNOSES_PATH)))
MEDICAL_VOCAB_PATH = Path(
    os.getenv("MEDICHAR_MEDICAL_VOCAB_PATH", str(DATA_DIR / "medical_vocabulary.txt"))
)
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "icd10_index.faiss"
FAISS_METADATA_PATH = VECTOR_STORE_DIR / "icd10_index_metadata.json"
OPENAI_RESPONSES_URL = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")

if load_dotenv is not None:
    load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class RAGSettings:
    embedding_model: str = os.getenv("MEDICHAR_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    llm_provider: str = os.getenv("MEDICHAR_LLM_PROVIDER", "auto")
    llm_model: str = os.getenv("MEDICHAR_LLM_MODEL", "google_ai_mode")
    searchapi_base_url: str = os.getenv(
        "SEARCHAPI_BASE_URL",
        os.getenv("SEARCHAPI_API_URL", "https://www.searchapi.io/api/v1/search"),
    )
    searchapi_engine: str = os.getenv("SEARCHAPI_ENGINE", "google_ai_mode")
    searchapi_location: str = os.getenv("SEARCHAPI_LOCATION", "")
    searchapi_hl: str = os.getenv("SEARCHAPI_HL", "en")
    searchapi_gl: str = os.getenv("SEARCHAPI_GL", "us")
    retrieval_top_k: int = int(os.getenv("MEDICHAR_RETRIEVAL_TOP_K", "5"))
    retrieval_min_score: float = float(os.getenv("MEDICHAR_RETRIEVAL_MIN_SCORE", "0.2"))
    embedding_batch_size: int = int(os.getenv("MEDICHAR_EMBEDDING_BATCH_SIZE", "32"))
    llm_temperature: float = float(os.getenv("MEDICHAR_LLM_TEMPERATURE", "0.2"))
    request_timeout_seconds: float = float(os.getenv("MEDICHAR_REQUEST_TIMEOUT", "60"))
    cors_origins: str = os.getenv("MEDICHAR_CORS_ORIGINS", "*")
    vision_provider: str = os.getenv("MEDICHAR_VISION_PROVIDER", "local")
    vision_model: str = os.getenv("MEDICHAR_VISION_MODEL", "microsoft/trocr-base-handwritten")
    vision_device: str = os.getenv("MEDICHAR_VISION_DEVICE", "auto")
    vision_detail: str = os.getenv("MEDICHAR_VISION_DETAIL", "high")
    vision_timeout_seconds: float = float(
        os.getenv("MEDICHAR_VISION_TIMEOUT", os.getenv("MEDICHAR_REQUEST_TIMEOUT", "60"))
    )
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_responses_url: str = os.getenv("OPENAI_RESPONSES_URL", OPENAI_RESPONSES_URL)
    ocr_preprocessing_enabled: bool = os.getenv("MEDICHAR_OCR_PREPROCESSING_ENABLED", "true").lower() == "true"
    ocr_preprocessing_mode: str = os.getenv("MEDICHAR_OCR_PREPROCESSING_MODE", "always")
    ocr_reconstruct_noise_threshold: float = float(os.getenv("MEDICHAR_OCR_RECONSTRUCT_NOISE_THRESHOLD", "2.0"))
    ocr_fuzzy_threshold: float = float(os.getenv("MEDICHAR_OCR_FUZZY_THRESHOLD", "90"))
    ocr_short_token_threshold: float = float(os.getenv("MEDICHAR_OCR_SHORT_TOKEN_THRESHOLD", "94"))
    ocr_medical_vocab_path: str = os.getenv("MEDICHAR_MEDICAL_VOCAB_PATH", str(MEDICAL_VOCAB_PATH))
    semantic_reranking_enabled: bool = os.getenv("MEDICHAR_SEMANTIC_RERANKING_ENABLED", "true").lower() == "true"
    semantic_similarity_weight: float = float(os.getenv("MEDICHAR_SEMANTIC_SIMILARITY_WEIGHT", "0.72"))
    semantic_feature_weight: float = float(os.getenv("MEDICHAR_SEMANTIC_FEATURE_WEIGHT", "0.28"))


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
