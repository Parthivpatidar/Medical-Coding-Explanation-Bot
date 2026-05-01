from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: Annotated[str, Field(min_length=1, description="Clinical note text to code")]
    top_k: Annotated[int, Field(5, ge=1, le=10)] = 5
    return_context: bool = False
    return_similarity: bool = False


class PredictedCode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    title: str
    explanation: str
    confidence: Literal["High", "Medium", "Low"]
    similarity_score: float | None = None


class RetrievedContextItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    title: str
    similarity_score: float
    chunk_text: str


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codes: list[PredictedCode]
    primary_icd: str | None = None
    secondary_icd: list[str] | None = None
    dropped_codes: list[str] | None = None
    prioritization_reasoning: str | None = None
    summary: str | None = None
    condition_overview: str | None = None
    precautions: list[str] | None = None
    diet_advice: list[str] | None = None
    message: str | None = None
    retrieved_context: list[RetrievedContextItem] | None = None


class ImageNoteExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    confidence: Literal["High", "Medium", "Low"]
    uncertain_fragments: list[str] = Field(default_factory=list)
    raw_text: str | None = None
    cleaned_text: str | None = None
    processed_text: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    catalog_size: int
    llm_provider: str
    llm_model: str
