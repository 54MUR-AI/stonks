"""Model deployment and serving utilities."""

import os
import json
import joblib
import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

from .models import BaseModel

@dataclass
class ModelMetadata:
    """Metadata for deployed model."""
    model_id: str
    name: str
    version: str
    model_type: str
    model_class: str
    feature_names: List[str]
    target_type: str
    created_at: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    description: Optional[str] = None

class ModelRegistry:
    """Registry for managing deployed models."""
    
    def __init__(self, registry_path: str):
        """Initialize model registry.
        
        Args:
            registry_path: Path to model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.registry_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(
            filename=self.registry_path / "registry.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Unique model ID
        """
        unique_str = f"{name}-{version}-{datetime.datetime.now().isoformat()}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def save_model(
        self,
        model: BaseModel,
        name: str,
        version: str,
        metrics: Dict[str, float],
        description: Optional[str] = None
    ) -> str:
        """Save model to registry.
        
        Args:
            model: Model to save
            name: Model name
            version: Model version
            metrics: Model performance metrics
            description: Optional model description
            
        Returns:
            Model ID
        """
        # Generate model ID
        model_id = self._generate_model_id(name, version)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model.model_type,
            model_class=model.model.__class__.__name__,
            feature_names=model.feature_names,
            target_type=str(model.target_type),
            created_at=datetime.datetime.now().isoformat(),
            metrics=metrics,
            parameters=model.model.get_params(),
            description=description
        )
        
        # Save model artifacts
        model_dir = self.models_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        joblib.dump(model.model, model_dir / "model.joblib")
        joblib.dump(model.scaler, model_dir / "scaler.joblib")
        
        # Save metadata
        with open(self.metadata_path / f"{model_id}.json", 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
            
        logging.info(f"Saved model {name} version {version} with ID {model_id}")
        return model_id
    
    def load_model(self, model_id: str) -> Tuple[BaseModel, ModelMetadata]:
        """Load model from registry.
        
        Args:
            model_id: Model ID
            
        Returns:
            Tuple of (model, metadata)
        """
        # Load metadata
        try:
            with open(self.metadata_path / f"{model_id}.json", 'r') as f:
                metadata = ModelMetadata(**json.load(f))
        except FileNotFoundError:
            raise ValueError(f"Model {model_id} not found in registry")
            
        # Load model artifacts
        model_dir = self.models_path / model_id
        model = joblib.load(model_dir / "model.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib")
        
        # Create BaseModel instance
        base_model = BaseModel(
            name=metadata.name,
            model_type=metadata.model_type,
            model=model,
            scaler=scaler,
            feature_names=metadata.feature_names
        )
        
        logging.info(f"Loaded model {metadata.name} version {metadata.version}")
        return base_model, metadata
    
    def list_models(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[ModelMetadata]:
        """List models in registry.
        
        Args:
            name: Filter by model name
            version: Filter by version
            model_type: Filter by model type
            
        Returns:
            List of model metadata
        """
        models = []
        for metadata_file in self.metadata_path.glob("*.json"):
            with open(metadata_file, 'r') as f:
                metadata = ModelMetadata(**json.load(f))
                
            if name and metadata.name != name:
                continue
            if version and metadata.version != version:
                continue
            if model_type and metadata.model_type != model_type:
                continue
                
            models.append(metadata)
            
        return models
    
    def delete_model(self, model_id: str):
        """Delete model from registry.
        
        Args:
            model_id: Model ID to delete
        """
        try:
            # Remove metadata
            metadata_file = self.metadata_path / f"{model_id}.json"
            metadata_file.unlink()
            
            # Remove model artifacts
            model_dir = self.models_path / model_id
            for file in model_dir.glob("*"):
                file.unlink()
            model_dir.rmdir()
            
            logging.info(f"Deleted model {model_id}")
        except FileNotFoundError:
            raise ValueError(f"Model {model_id} not found in registry")

class ModelServer:
    """Server for deployed models."""
    
    def __init__(
        self,
        registry: ModelRegistry,
        max_workers: int = 4,
        batch_size: int = 1000
    ):
        """Initialize model server.
        
        Args:
            registry: Model registry
            max_workers: Maximum number of worker threads
            batch_size: Batch size for predictions
        """
        self.registry = registry
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = batch_size
        self.loaded_models: Dict[str, BaseModel] = {}
        
    def load_model(self, model_id: str):
        """Load model into memory.
        
        Args:
            model_id: Model ID to load
        """
        if model_id not in self.loaded_models:
            model, _ = self.registry.load_model(model_id)
            self.loaded_models[model_id] = model
            
    def unload_model(self, model_id: str):
        """Unload model from memory.
        
        Args:
            model_id: Model ID to unload
        """
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            
    def predict(
        self,
        model_id: str,
        features: pd.DataFrame,
        return_proba: bool = False
    ) -> np.ndarray:
        """Generate predictions using model.
        
        Args:
            model_id: Model ID to use
            features: Feature DataFrame
            return_proba: Whether to return probabilities
            
        Returns:
            Array of predictions
        """
        if model_id not in self.loaded_models:
            self.load_model(model_id)
            
        model = self.loaded_models[model_id]
        
        # Validate features
        if not all(f in features.columns for f in model.feature_names):
            missing = set(model.feature_names) - set(features.columns)
            raise ValueError(f"Missing features: {missing}")
            
        # Reorder columns to match model
        features = features[model.feature_names]
        
        # Process in batches
        predictions = []
        for i in range(0, len(features), self.batch_size):
            batch = features.iloc[i:i + self.batch_size]
            if return_proba and hasattr(model.model, 'predict_proba'):
                pred = model.predict_proba(batch)
            else:
                pred = model.predict(batch)
            predictions.append(pred)
            
        return np.concatenate(predictions)
    
    def predict_batch(
        self,
        model_id: str,
        features_list: List[pd.DataFrame],
        return_proba: bool = False
    ) -> List[np.ndarray]:
        """Generate predictions for multiple feature sets.
        
        Args:
            model_id: Model ID to use
            features_list: List of feature DataFrames
            return_proba: Whether to return probabilities
            
        Returns:
            List of prediction arrays
        """
        futures = []
        for features in features_list:
            future = self.executor.submit(
                self.predict,
                model_id,
                features,
                return_proba
            )
            futures.append(future)
            
        return [future.result() for future in futures]
