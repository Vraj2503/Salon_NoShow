"""
FastAPI Inference API for Salon No-Show Prediction.
Provides endpoints for single and batch predictions, health checks, and model info.
"""

import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictor import NoShowPredictor
from api.schemas import (
    BookingRequest, BatchBookingRequest,
    PredictionResponse, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("salon-noshow-api")

# Global predictor instance
predictor = NoShowPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Loading prediction model...")
    try:
        predictor.load_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Salon No-Show Prediction API",
    description="AI-powered no-show risk prediction for salon bookings",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration_ms:.1f}ms)")
    return response


# --- Endpoints ---

@app.get("/", tags=["System"])
async def root():
    """Root endpoint — API welcome message."""
    return {
        "message": "Salon No-Show Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "health": "GET /health",
            "model_info": "GET /model/info",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor._loaded,
        version="1.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get current model metadata."""
    info = predictor.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(booking: BookingRequest):
    """Predict no-show probability for a single booking."""
    booking_dict = booking.model_dump()
    result = predictor.predict(booking_dict)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchBookingRequest):
    """Predict no-show probability for a batch of bookings."""
    bookings = [b.model_dump() for b in request.bookings]
    results = predictor.predict_batch(bookings)
    predictions = [PredictionResponse(**r) for r in results]
    return BatchPredictionResponse(predictions=predictions, total=len(predictions))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
