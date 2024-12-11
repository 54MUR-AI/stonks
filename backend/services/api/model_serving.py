"""Model serving API for production deployment."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional, Union
import torch
import numpy as np
import logging
import json
from pathlib import Path
import asyncio
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import Response

from backend.services.ml.monitoring.metrics import ModelMetrics
from backend.services.ml.monitoring.drift import DriftDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stonks Model Serving API")

# Initialize Prometheus metrics
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["model_id", "version"]
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency in seconds",
    ["model_id", "version"]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

class PredictionRequest(BaseModel):
    """Model prediction request."""
    model_id: str
    features: Dict[str, Union[float, List[float]]]
    version: Optional[str] = "latest"

class PredictionResponse(BaseModel):
    """Model prediction response."""
    model_id: str
    version: str
    prediction: Union[float, List[float]]
    confidence: float
    timestamp: str
    processing_time: float

class ModelMetadata(BaseModel):
    """Model metadata."""
    model_id: str
    versions: List[str]
    current_version: str
    metrics: Dict[str, float]
    last_updated: str
    status: str

# In-memory model cache
model_cache = {}
model_metrics = {}

async def load_model(model_id: str, version: str) -> bool:
    """Load model into memory.
    
    Args:
        model_id: Model identifier
        version: Model version
        
    Returns:
        Success status
    """
    try:
        model_path = Path(f"models/{model_id}/{version}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_id} version {version} not found")
            
        # Load model (implement actual loading logic)
        model = torch.load(model_path / "model.pt")
        model_cache[f"{model_id}_{version}"] = {
            "model": model,
            "loaded_at": datetime.now(),
            "metrics": ModelMetrics()
        }
        return True
        
    except Exception as e:
        logger.error(f"Error loading model {model_id} version {version}: {str(e)}")
        return False

async def unload_model(model_id: str, version: str):
    """Unload model from memory.
    
    Args:
        model_id: Model identifier
        version: Model version
    """
    key = f"{model_id}_{version}"
    if key in model_cache:
        del model_cache[key]
        logger.info(f"Unloaded model {model_id} version {version}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Make prediction using specified model.
    
    Args:
        request: Prediction request
        
    Returns:
        Prediction response
    """
    start_time = datetime.now()
    
    # Load model if not in cache
    model_key = f"{request.model_id}_{request.version}"
    if model_key not in model_cache:
        success = await load_model(request.model_id, request.version)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_id} version {request.version} not found"
            )
    
    try:
        # Convert features to tensor
        features = torch.tensor(
            [list(request.features.values())],
            dtype=torch.float32
        )
        
        # Make prediction
        model = model_cache[model_key]["model"]
        with torch.no_grad():
            prediction = model(features)
            
        # Calculate confidence (implement actual confidence calculation)
        confidence = 0.95  # Placeholder
        
        # Update metrics
        model_cache[model_key]["metrics"].update(
            features.numpy(),
            prediction.numpy()
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update Prometheus metrics
        PREDICTION_COUNTER.labels(model_id=request.model_id, version=request.version).inc()
        PREDICTION_LATENCY.labels(model_id=request.model_id, version=request.version).observe(processing_time)
        
        return PredictionResponse(
            model_id=request.model_id,
            version=request.version,
            prediction=prediction.tolist()[0],
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/models/{model_id}/metadata", response_model=ModelMetadata)
async def get_model_metadata(model_id: str) -> ModelMetadata:
    """Get model metadata.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model metadata
    """
    try:
        model_dir = Path(f"models/{model_id}")
        if not model_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
            
        # Get available versions
        versions = [v.name for v in model_dir.iterdir() if v.is_dir()]
        
        # Get current version metrics
        current_version = max(versions)
        model_key = f"{model_id}_{current_version}"
        
        if model_key in model_cache:
            metrics = model_cache[model_key]["metrics"].get_metrics()
        else:
            metrics = {}
            
        return ModelMetadata(
            model_id=model_id,
            versions=versions,
            current_version=current_version,
            metrics=metrics,
            last_updated=datetime.now().isoformat(),
            status="active"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model metadata: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model metadata: {str(e)}"
        )

@app.post("/models/{model_id}/versions/{version}/load")
async def load_model_version(
    model_id: str,
    version: str,
    background_tasks: BackgroundTasks
):
    """Load specific model version.
    
    Args:
        model_id: Model identifier
        version: Model version
        background_tasks: Background tasks
    """
    success = await load_model(model_id, version)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} version {version} not found"
        )
    return {"status": "success"}

@app.post("/models/{model_id}/versions/{version}/unload")
async def unload_model_version(
    model_id: str,
    version: str,
    background_tasks: BackgroundTasks
):
    """Unload specific model version.
    
    Args:
        model_id: Model identifier
        version: Model version
        background_tasks: Background tasks
    """
    background_tasks.add_task(unload_model, model_id, version)
    return {"status": "success"}

@app.get("/models/{model_id}/versions/{version}/metrics")
async def get_model_metrics(model_id: str, version: str):
    """Get model metrics.
    
    Args:
        model_id: Model identifier
        version: Model version
        
    Returns:
        Model metrics
    """
    model_key = f"{model_id}_{version}"
    if model_key not in model_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} version {version} not loaded"
        )
        
    metrics = model_cache[model_key]["metrics"].get_metrics()
    return metrics

@app.post("/models/{model_id}/versions/{version}/ab_test")
async def start_ab_test(
    model_id: str,
    version: str,
    traffic_percentage: float
):
    """Start A/B test for model version.
    
    Args:
        model_id: Model identifier
        version: Model version
        traffic_percentage: Percentage of traffic to route to new version
    """
    # Implement A/B testing logic
    return {"status": "success"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
