"""Example of using advanced technical indicators and signal generation."""

import asyncio
import pandas as pd
from datetime import datetime, timedelta

from backend.services.market_data.providers.polygon import PolygonDataProvider
from backend.services.analysis.advanced_indicators import AdvancedIndicators
from backend.services.analysis.signals import SignalGenerator

async def main():
    """Run advanced technical analysis example."""
    # Initialize components
    polygon = PolygonDataProvider()
    indicators = AdvancedIndicators()
    signal_gen = SignalGenerator()
    
    # Fetch historical data for analysis
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    bars = await polygon.get_bars(
        symbol,
        start_date,
        end_date,
        "1d"
    )
    
    df = pd.DataFrame(bars)
    df.set_index('timestamp', inplace=True)
    
    # Calculate indicators
    ichimoku = indicators.ichimoku(df['high'], df['low'], df['close'])
    supertrend = indicators.supertrend(df['high'], df['low'], df['close'])
    elder_ray = indicators.elder_ray(df['high'], df['low'], df['close'])
    
    # Generate signals
    signals = []
    
    # Analyze Ichimoku signals
    ichimoku_signals = signal_gen.analyze_ichimoku(ichimoku, df['close'])
    signals.extend(ichimoku_signals)
    
    # Analyze SuperTrend signals
    supertrend_signals = signal_gen.analyze_supertrend(supertrend, df['close'])
    signals.extend(supertrend_signals)
    
    # Analyze Elder Ray signals
    elder_ray_signals = signal_gen.analyze_elder_ray(elder_ray)
    signals.extend(elder_ray_signals)
    
    # Combine signals with custom weights
    weights = {
        "Ichimoku": 0.4,
        "SuperTrend": 0.4,
        "Elder Ray": 0.2
    }
    
    combined_signals = signal_gen.combine_signals(signals, weights)
    
    # Print analysis results
    print(f"\nAnalysis Results for {symbol}")
    print("-" * 50)
    
    print("\nIchimoku Cloud Analysis:")
    print(f"Number of signals: {len(ichimoku_signals)}")
    for signal in ichimoku_signals[-5:]:
        print(f"  {signal.timestamp}: {signal.type.value} (strength: {signal.strength:.2f})")
    
    print("\nSuperTrend Analysis:")
    print(f"Number of signals: {len(supertrend_signals)}")
    for signal in supertrend_signals[-5:]:
        print(f"  {signal.timestamp}: {signal.type.value} (strength: {signal.strength:.2f})")
    
    print("\nElder Ray Analysis:")
    print(f"Number of signals: {len(elder_ray_signals)}")
    for signal in elder_ray_signals[-5:]:
        print(f"  {signal.timestamp}: {signal.type.value} (strength: {signal.strength:.2f})")
    
    print("\nCombined Signals:")
    print(f"Number of signals: {len(combined_signals)}")
    for signal in combined_signals[-5:]:
        print(f"  {signal.timestamp}: {signal.type.value} (strength: {signal.strength:.2f})")
        print("    Components:")
        for component in signal.metadata['components']:
            print(f"      {component['indicator']}: {component['type']} ({component['strength']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
