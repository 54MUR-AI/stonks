"""Example of model evaluation and cross-validation."""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from backend.services.market_data.providers.polygon import PolygonDataProvider
from backend.services.ml.feature_engineering import FeatureEngineer
from backend.services.ml.model_implementations import create_model
from backend.services.ml.evaluation import ModelEvaluator

async def main():
    """Run model evaluation example."""
    # Initialize components
    polygon = PolygonDataProvider()
    engineer = FeatureEngineer()
    evaluator = ModelEvaluator()
    
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
    
    # Create and evaluate models
    models = {
        'random_forest': create_model(
            name='rf_classifier',
            model_type='classification',
            model_class='random_forest',
            model_params={'n_estimators': 100, 'random_state': 42}
        ),
        'xgboost': create_model(
            name='xgb_classifier',
            model_type='classification',
            model_class='xgboost',
            model_params={'n_estimators': 100, 'random_state': 42}
        )
    }
    
    # Perform cross-validation for each model
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        cv_result = evaluator.cross_validate(
            model,
            feature_set.features,
            feature_set.target,
            n_splits=5,
            gap=5
        )
        results[name] = cv_result
        
        # Print metrics summary
        metrics_df = pd.DataFrame(cv_result.metrics)
        print(f"\n{name} CV Metrics Summary:")
        print(metrics_df.mean().round(4))
        print("\nMetrics Standard Deviation:")
        print(metrics_df.std().round(4))
        
        # Plot evaluation results
        plt.figure(figsize=(15, 10))
        
        # CV metrics distribution
        plt.subplot(2, 2, 1)
        evaluator.plot_cv_metrics(cv_result)
        
        # Feature importance
        plt.subplot(2, 2, 2)
        evaluator.plot_feature_importance(cv_result, top_n=10)
        
        # Train final model for ROC and confusion matrix
        model.train(feature_set.features, feature_set.target)
        y_pred = model.predict(feature_set.features)
        y_prob = model.predict_proba(feature_set.features)[:, 1]
        
        # ROC curve
        plt.subplot(2, 2, 3)
        evaluator.plot_roc_curve(feature_set.target, y_prob)
        
        # Confusion matrix
        plt.subplot(2, 2, 4)
        evaluator.plot_confusion_matrix(feature_set.target, y_pred)
        
        plt.tight_layout()
        plt.savefig(f'{name}_evaluation.png')
        plt.close()
        
        print(f"\nEvaluation plots saved to {name}_evaluation.png")
        
        # Save top features
        importance = cv_result.feature_importance.mean(axis=1).sort_values(ascending=False)
        print(f"\nTop 10 Important Features for {name}:")
        print(importance.head(10))

if __name__ == "__main__":
    asyncio.run(main())
