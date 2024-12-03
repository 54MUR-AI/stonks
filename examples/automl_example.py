"""Example of AutoML optimization."""

import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from backend.services.market_data.providers.polygon import PolygonDataProvider
from backend.services.ml.feature_engineering import FeatureEngineer
from backend.services.ml.automl import AutoML

async def main():
    """Run AutoML example."""
    # Initialize components
    polygon = PolygonDataProvider()
    engineer = FeatureEngineer()
    
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
    
    # Prepare initial features
    print("\nGenerating initial features...")
    feature_set = engineer.prepare_features(
        df,
        feature_sets=['momentum', 'trend', 'volatility', 'volume'],
        windows=[5, 10, 20],
        target_horizon=5,
        target_threshold=0.01  # Classification target
    )
    
    # Initialize AutoML
    print("\nInitializing AutoML...")
    automl = AutoML(
        task='classification',
        metric='f1',
        n_trials=50,  # Reduced for example
        cv_splits=5,
        feature_selection=True
    )
    
    # Run optimization
    print("\nStarting AutoML optimization...")
    result = automl.optimize(
        feature_set.features,
        feature_set.target,
        feature_sets=['technical', 'statistical']  # Generate additional features
    )
    
    # Print results
    print("\nAutoML Results:")
    print(f"Best model: {result.best_model.name}")
    print("\nBest parameters:")
    for param, value in result.best_params.items():
        print(f"- {param}: {value}")
        
    print("\nCross-validation scores:")
    for metric, scores in result.cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"- {metric}: {mean_score:.4f} ± {std_score:.4f}")
        
    # Plot optimization history
    plt.figure(figsize=(12, 6))
    history = result.optimization_history
    
    plt.subplot(1, 2, 1)
    sns.lineplot(data=history, x='trial', y='score')
    plt.title('Optimization History')
    plt.xlabel('Trial')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    model_scores = history.groupby('model_class')['score'].mean()
    sns.barplot(x=model_scores.index, y=model_scores.values)
    plt.title('Average Score by Model Type')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('automl_optimization.png')
    plt.close()
    print("\nOptimization plots saved to automl_optimization.png")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    importance = pd.Series(result.feature_importance)
    importance.sort_values(ascending=True).tail(20).plot(kind='barh')
    plt.title('Top 20 Features (Best Model)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('automl_feature_importance.png')
    plt.close()
    print("Feature importance plot saved to automl_feature_importance.png")
    
    # Save best model predictions
    predictions = result.best_model.predict(feature_set.features)
    probabilities = result.best_model.predict_proba(feature_set.features)
    
    predictions_df = pd.DataFrame({
        'actual': feature_set.target,
        'predicted': predictions,
        'probability': probabilities[:, 1]
    }, index=feature_set.features.index)
    
    predictions_df.to_csv('automl_predictions.csv')
    print("\nPredictions saved to automl_predictions.csv")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(predictions_df['actual'], predictions_df['predicted'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Best Model)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('automl_confusion_matrix.png')
    plt.close()
    print("Confusion matrix plot saved to automl_confusion_matrix.png")

if __name__ == "__main__":
    asyncio.run(main())
