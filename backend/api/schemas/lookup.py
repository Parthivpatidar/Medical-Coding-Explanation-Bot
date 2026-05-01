from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LookupSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    title: str


class LookupResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_code: str
    normalized_code: str
    found: bool
    title: str
    definition: str
    explanation_hint: str
    keywords: list[str]
    suggestions: list[LookupSuggestion]
    catalog_source: str


class LookupBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codes: list[str] = Field(min_length=1)


class LookupBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[LookupResponse]
