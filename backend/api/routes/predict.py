from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.api.dependencies import get_service
from backend.api.schemas.predict import (
    HealthResponse,
    ImageNoteExtractionResponse,
    PredictRequest,
    PredictResponse,
)
from backend.services.medical_coding import MedicalCodingService


router = APIRouter()


@router.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
def predict(
    payload: PredictRequest,
    service: MedicalCodingService = Depends(get_service),
) -> PredictResponse:
    return PredictResponse.model_validate(
        service.predict(
            clinical_text=payload.text,
            top_k=payload.top_k,
            return_context=payload.return_context,
            return_similarity=payload.return_similarity,
        )
    )


@router.post("/extract-note-image", response_model=ImageNoteExtractionResponse)
async def extract_note_image(
    image: UploadFile = File(...),
    supplemental_text: Annotated[str, Form()] = "",
    ocr_mode: Annotated[str, Form()] = "printed",
    service: MedicalCodingService = Depends(get_service),
) -> ImageNoteExtractionResponse:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        payload = service.extract_note_from_image(
            image_bytes=image_bytes,
            filename=image.filename or "note.png",
            supplemental_text=supplemental_text,
            ocr_mode=ocr_mode,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ImageNoteExtractionResponse.model_validate(payload)


@router.get("/health", response_model=HealthResponse)
def health(service: MedicalCodingService = Depends(get_service)) -> HealthResponse:
    return HealthResponse.model_validate(service.health())
