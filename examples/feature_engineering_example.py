"""Example of using the feature engineering pipeline."""

import asyncio
import pandas as pd
from datetime import datetime, timedelta

from backend.services.market_data.providers.polygon import PolygonDataProvider
from backend.services.ml.feature_engineering import FeatureEngineer

async def main():
    """Run feature engineering example."""
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
    
    print(f"Data shape: {df.shape}")
    print("\nSample of price data:")
    print(df.head())
    
    # Create feature sets
    print("\nGenerating features...")
    feature_set = engineer.prepare_features(
        df,
        feature_sets=['momentum', 'trend', 'volatility', 'volume'],
        windows=[5, 10, 20],
        target_horizon=5,
        target_threshold=0.01
    )
    
    print("\nFeature set metadata:")
    for key, value in feature_set.metadata.items():
        print(f"{key}: {value}")
    
    print("\nSample of features:")
    print(feature_set.features.head())
    
    print("\nFeature statistics:")
    print(feature_set.features.describe())
    
    # Analyze feature correlations
    print("\nTop feature correlations with target:")
    correlations = pd.concat([feature_set.features, feature_set.target], axis=1)
    correlations.columns = list(feature_set.features.columns) + ['target']
    target_corr = correlations.corr()['target'].sort_values(ascending=False)
    print(target_corr.head(10))
    print("\nBottom feature correlations with target:")
    print(target_corr.tail(10))
    
    # Save features to CSV for further analysis
    output_file = f"{symbol}_features.csv"
    print(f"\nSaving features to {output_file}...")
    
    output_data = pd.concat([
        feature_set.features,
        pd.Series(feature_set.target, name='target')
    ], axis=1)
    
    output_data.to_csv(output_file)
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
