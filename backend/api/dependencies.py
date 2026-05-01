from __future__ import annotations

from fastapi import HTTPException, Request

from backend.services.medical_coding import MedicalCodingService


def get_service(request: Request) -> MedicalCodingService:
    service = getattr(request.app.state, "medical_coding_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Medical coding service is not ready")
    return service
