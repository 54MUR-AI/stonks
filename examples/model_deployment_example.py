"""Example of model deployment and serving."""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from backend.services.market_data.providers.polygon import PolygonDataProvider
from backend.services.ml.feature_engineering import FeatureEngineer
from backend.services.ml.model_implementations import create_model
from backend.services.ml.deployment import ModelRegistry, ModelServer

async def main():
    """Run model deployment example."""
    # Initialize components
    polygon = PolygonDataProvider()
    engineer = FeatureEngineer()
    registry = ModelRegistry("models_registry")
    server = ModelServer(registry)
    
    # Fetch historical data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print("\nFetching data...")
    all_data = {}
    all_features = {}
    for symbol in symbols:
        print(f"Processing {symbol}...")
        bars = await polygon.get_bars(
            symbol,
            start_date,
            end_date,
            "1d"
        )
        
        df = pd.DataFrame(bars)
        df.set_index('timestamp', inplace=True)
        all_data[symbol] = df
        
        # Prepare features
        feature_set = engineer.prepare_features(
            df,
            feature_sets=['momentum', 'trend', 'volatility', 'volume'],
            windows=[5, 10, 20],
            target_horizon=5,
            target_threshold=0.01
        )
        all_features[symbol] = feature_set
    
    # Train and save models
    print("\nTraining and saving models...")
    model_configs = [
        {
            'name': 'trend_classifier',
            'type': 'classification',
            'class': 'random_forest',
            'params': {'n_estimators': 100, 'random_state': 42}
        },
        {
            'name': 'trend_classifier',
            'type': 'classification',
            'class': 'xgboost',
            'params': {'n_estimators': 100, 'random_state': 42}
        }
    ]
    
    model_ids = []
    for config in model_configs:
        # Train model on AAPL data
        model = create_model(
            name=config['name'],
            model_type=config['type'],
            model_class=config['class'],
            model_params=config['params']
        )
        
        feature_set = all_features['AAPL']
        model.train(feature_set.features, feature_set.target)
        
        # Calculate metrics
        predictions = model.predict(feature_set.features)
        metrics = {
            'accuracy': accuracy_score(feature_set.target, predictions),
            'precision': precision_score(feature_set.target, predictions),
            'recall': recall_score(feature_set.target, predictions),
            'f1': f1_score(feature_set.target, predictions)
        }
        
        # Save model
        model_id = registry.save_model(
            model=model,
            name=config['name'],
            version=f"{config['class']}_v1",
            metrics=metrics,
            description=f"Trend classification model using {config['class']}"
        )
        model_ids.append(model_id)
        
    # List saved models
    print("\nSaved Models:")
    for metadata in registry.list_models():
        print(f"\nModel: {metadata.name} (ID: {metadata.model_id})")
        print(f"Version: {metadata.version}")
        print(f"Type: {metadata.model_type}")
        print("Metrics:")
        for metric, value in metadata.metrics.items():
            print(f"- {metric}: {value:.4f}")
    
    # Generate predictions for all symbols
    print("\nGenerating predictions...")
    for model_id in model_ids:
        model_metadata = registry.list_models()[0]
        print(f"\nUsing model: {model_metadata.name} {model_metadata.version}")
        
        # Get predictions for each symbol
        all_predictions = {}
        for symbol, feature_set in all_features.items():
            predictions = server.predict(
                model_id,
                feature_set.features,
                return_proba=True
            )
            all_predictions[symbol] = predictions[:, 1]  # Probability of positive class
            
        # Plot predictions
        plt.figure(figsize=(12, 6))
        for symbol, probs in all_predictions.items():
            plt.plot(feature_set.features.index[-30:], probs[-30:], label=symbol)
        
        plt.title(f'Recent Trend Predictions ({model_metadata.version})')
        plt.xlabel('Date')
        plt.ylabel('Probability of Uptrend')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f'predictions_{model_metadata.version}.png')
        plt.close()
        print(f"Prediction plot saved to predictions_{model_metadata.version}.png")
        
    # Batch prediction example
    print("\nTesting batch prediction...")
    features_list = [feature_set.features for feature_set in all_features.values()]
    batch_predictions = server.predict_batch(
        model_ids[0],  # Use first model
        features_list,
        return_proba=True
    )
    
    print(f"Generated {len(batch_predictions)} batch predictions")
    
    # Clean up
    print("\nCleaning up...")
    for model_id in model_ids:
        registry.delete_model(model_id)
    print("Deleted test models")

if __name__ == "__main__":
    asyncio.run(main())
