"""Example of feature selection methods."""

import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from backend.services.market_data.providers.polygon import PolygonDataProvider
from backend.services.ml.feature_engineering import FeatureEngineer
from backend.services.ml.feature_selection import FeatureSelector

async def main():
    """Run feature selection example."""
    # Initialize components
    polygon = PolygonDataProvider()
    engineer = FeatureEngineer()
    selector = FeatureSelector()
    
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
        target_threshold=0.01
    )
    
    # Run different feature selection methods
    k = 20  # Number of features to select
    task = 'classification'
    
    print("\nRunning feature selection methods...")
    
    # 1. Mutual Information
    mi_result = selector.select_mutual_information(
        feature_set.features,
        feature_set.target,
        k=k,
        task=task
    )
    
    # 2. F-Score
    f_result = selector.select_f_score(
        feature_set.features,
        feature_set.target,
        k=k,
        task=task
    )
    
    # 3. RFE
    rfe_result = selector.select_rfe(
        feature_set.features,
        feature_set.target,
        k=k,
        task=task
    )
    
    # 4. Lasso
    lasso_result = selector.select_lasso(
        feature_set.features,
        feature_set.target,
        task=task
    )
    
    # 5. Boruta
    boruta_result = selector.select_boruta(
        feature_set.features,
        feature_set.target,
        task=task
    )
    
    # 6. Ensemble
    ensemble_result = selector.ensemble_selection(
        feature_set.features,
        feature_set.target,
        k=k,
        task=task
    )
    
    # Print results
    print("\nFeature Selection Results:")
    print(f"\nMutual Information selected {len(mi_result.selected_features)} features")
    print(f"F-Score selected {len(f_result.selected_features)} features")
    print(f"RFE selected {len(rfe_result.selected_features)} features")
    print(f"Lasso selected {len(lasso_result.selected_features)} features")
    print(f"Boruta selected {len(boruta_result.selected_features)} features")
    print(f"Ensemble selected {len(ensemble_result.selected_features)} features")
    
    # Plot feature importance comparison
    results = {
        'Mutual Information': mi_result,
        'F-Score': f_result,
        'RFE': rfe_result,
        'Lasso': lasso_result,
        'Boruta': boruta_result,
        'Ensemble': ensemble_result
    }
    
    # Create comparison matrix
    comparison = pd.DataFrame(index=feature_set.features.columns)
    for method, result in results.items():
        comparison[method] = [1 if f in result.selected_features else 0 for f in comparison.index]
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(comparison, cmap='YlOrRd', cbar_kws={'label': 'Selected'})
    plt.title('Feature Selection Methods Comparison')
    plt.tight_layout()
    plt.savefig('feature_selection_comparison.png')
    plt.close()
    print("\nFeature selection comparison plot saved to feature_selection_comparison.png")
    
    # Plot ensemble feature scores
    plt.figure(figsize=(12, 6))
    scores = pd.Series(ensemble_result.feature_scores)
    scores.sort_values(ascending=True).tail(20).plot(kind='barh')
    plt.title('Top 20 Features (Ensemble Selection)')
    plt.xlabel('Selection Score')
    plt.tight_layout()
    plt.savefig('ensemble_feature_scores.png')
    plt.close()
    print("Ensemble feature scores plot saved to ensemble_feature_scores.png")
    
    # Print common features across methods
    common_features = set(ensemble_result.selected_features)
    for result in results.values():
        common_features &= set(result.selected_features)
    
    print("\nFeatures selected by all methods:")
    for feature in common_features:
        print(f"- {feature}")

if __name__ == "__main__":
    asyncio.run(main())
