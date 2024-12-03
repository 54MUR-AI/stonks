"""Example of model interpretability tools."""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from backend.services.market_data.providers.polygon import PolygonDataProvider
from backend.services.ml.feature_engineering import FeatureEngineer
from backend.services.ml.model_implementations import create_model
from backend.services.ml.interpretability import ModelInterpreter

async def main():
    """Run model interpretability example."""
    # Initialize components
    polygon = PolygonDataProvider()
    engineer = FeatureEngineer()
    interpreter = ModelInterpreter()
    
    # Fetch historical data
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"\nFetching data for {symbol}...")
    bars = await polygon.get_bars(
        symbol,
        start_date,
        end_date,
        "1d"
    )
    
    df = pd.DataFrame(bars)
    df.set_index('timestamp', inplace=True)
    
    # Prepare features
    print("\nGenerating features...")
    feature_set = engineer.prepare_features(
        df,
        feature_sets=['momentum', 'trend', 'volatility', 'volume'],
        windows=[5, 10, 20],
        target_horizon=5,
        target_threshold=0.01  # Classification target
    )
    
    # Create and train model
    model = create_model(
        name='xgb_classifier',
        model_type='classification',
        model_class='xgboost',
        model_params={'n_estimators': 100, 'random_state': 42}
    )
    
    print("\nTraining model...")
    model.train(feature_set.features, feature_set.target)
    
    # Calculate SHAP values
    print("\nCalculating SHAP values...")
    shap_values, feature_names = interpreter.calculate_shap_values(
        model,
        feature_set.features,
        sample_size=1000  # Use sample for faster computation
    )
    
    # Plot SHAP summary
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    interpreter.plot_shap_summary(shap_values, feature_names)
    plt.savefig('shap_summary.png')
    plt.close()
    print("SHAP summary plot saved to shap_summary.png")
    
    # Analyze feature interactions
    print("\nAnalyzing feature interactions...")
    interactions = interpreter.analyze_feature_interactions(
        shap_values,
        feature_names,
        top_n=10
    )
    print("\nTop 10 Feature Interactions:")
    print(interactions)
    
    # Plot partial dependence for top features
    print("\nGenerating partial dependence plots...")
    top_features = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names
    ).nlargest(3).index
    
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        interpreter.plot_partial_dependence(
            model,
            feature_set.features,
            feature
        )
        plt.savefig(f'pdp_{feature}.png')
        plt.close()
        print(f"Partial dependence plot saved to pdp_{feature}.png")
    
    # Analyze specific predictions
    print("\nAnalyzing specific predictions...")
    # Get most recent prediction
    recent_idx = -1
    analysis = interpreter.analyze_prediction(
        model,
        feature_set.features,
        recent_idx
    )
    
    print("\nRecent Prediction Analysis:")
    print(f"Predicted Class: {analysis['predicted_class']}")
    print(f"Prediction Probability: {analysis['prediction']}")
    print("\nTop Contributing Features:")
    for feature, value in analysis['top_contributing_features'].items():
        print(f"{feature}: {value:.4f}")
    
    # Plot prediction waterfall
    print("\nGenerating prediction waterfall plot...")
    plt.figure(figsize=(12, 8))
    interpreter.plot_prediction_waterfall(
        shap_values,
        feature_set.features,
        recent_idx
    )
    plt.savefig('prediction_waterfall.png')
    plt.close()
    print("Prediction waterfall plot saved to prediction_waterfall.png")
    
    # Plot SHAP dependence for top feature
    top_feature = list(analysis['top_contributing_features'].keys())[0]
    print(f"\nGenerating SHAP dependence plot for {top_feature}...")
    plt.figure(figsize=(10, 6))
    interpreter.plot_shap_dependence(
        shap_values,
        feature_set.features,
        top_feature
    )
    plt.savefig(f'shap_dependence_{top_feature}.png')
    plt.close()
    print(f"SHAP dependence plot saved to shap_dependence_{top_feature}.png")

if __name__ == "__main__":
    asyncio.run(main())
