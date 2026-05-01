from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.lookup import router as lookup_router
from backend.api.routes.predict import router as predict_router
from backend.core.config import RAGSettings, ensure_directories
from backend.core.utils import configure_logging, parse_cors_origins
from backend.services.medical_coding import MedicalCodingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = RAGSettings()
    logger = configure_logging()
    ensure_directories()
    logger.info("Initializing MediChar RAG backend")
    app.state.medical_coding_service = MedicalCodingService(settings=settings)
    yield


def create_app() -> FastAPI:
    settings = RAGSettings()
    app = FastAPI(
        title="MediChar ICD-10 Backend",
        version="1.0.0",
        description="FastAPI backend for ICD-10 prediction using retrieval-augmented generation.",
        lifespan=lifespan,
    )

    origins = parse_cors_origins(settings.cors_origins)
    allow_credentials = origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(predict_router)
    app.include_router(lookup_router)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
