from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_service
from backend.api.schemas.lookup import LookupBatchRequest, LookupBatchResponse, LookupResponse
from backend.services.medical_coding import MedicalCodingService


router = APIRouter()


@router.get("/lookup/{code}", response_model=LookupResponse)
def lookup_code(
    code: str,
    service: MedicalCodingService = Depends(get_service),
) -> LookupResponse:
    return LookupResponse.model_validate(service.lookup_code(code))


@router.post("/lookup-batch", response_model=LookupBatchResponse)
def lookup_codes(
    payload: LookupBatchRequest,
    service: MedicalCodingService = Depends(get_service),
) -> LookupBatchResponse:
    return LookupBatchResponse.model_validate(
        {"results": service.lookup_codes(payload.codes)}
    )
