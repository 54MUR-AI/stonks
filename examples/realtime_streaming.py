"""Example of real-time market data streaming using Polygon.io WebSocket."""

import asyncio
import os
from datetime import datetime
import pandas as pd
from typing import Dict, Any

from backend.services.market_data.providers.polygon import PolygonProvider
from backend.services.market_data.providers.config import PolygonConfig

class MarketDataStreamingExample:
    """Example class demonstrating real-time market data streaming."""
    
    def __init__(self, api_key: str):
        self.provider = PolygonProvider(PolygonConfig(api_key=api_key))
        self.trades_df = pd.DataFrame()
        self.quotes_df = pd.DataFrame()
        self.bars_df = pd.DataFrame()
        
    async def initialize(self):
        """Initialize the provider and set up handlers."""
        await self.provider.initialize()
        
        # Register data handlers
        self.provider.on_trade(self.handle_trade)
        self.provider.on_quote(self.handle_quote)
        self.provider.on_aggregate(self.handle_aggregate)
        
    async def handle_trade(self, data: Dict[str, Any]):
        """Handle incoming trade data."""
        trade = {
            'timestamp': datetime.fromtimestamp(data['t'] / 1000),
            'symbol': data['sym'],
            'price': data['p'],
            'size': data['s'],
            'exchange': data['x']
        }
        self.trades_df = pd.concat([
            self.trades_df,
            pd.DataFrame([trade])
        ]).reset_index(drop=True)
        
        print(f"Trade: {trade['symbol']} - Price: ${trade['price']:.2f}, Size: {trade['size']}")
        
    async def handle_quote(self, data: Dict[str, Any]):
        """Handle incoming quote data."""
        quote = {
            'timestamp': datetime.fromtimestamp(data['t'] / 1000),
            'symbol': data['sym'],
            'bid_price': data['bp'],
            'bid_size': data['bs'],
            'ask_price': data['ap'],
            'ask_size': data['as'],
            'exchange': data['x']
        }
        self.quotes_df = pd.concat([
            self.quotes_df,
            pd.DataFrame([quote])
        ]).reset_index(drop=True)
        
        print(f"Quote: {quote['symbol']} - Bid: ${quote['bid_price']:.2f}, Ask: ${quote['ask_price']:.2f}")
        
    async def handle_aggregate(self, data: Dict[str, Any]):
        """Handle incoming aggregate (minute bar) data."""
        bar = {
            'timestamp': datetime.fromtimestamp(data['t'] / 1000),
            'symbol': data['sym'],
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        }
        self.bars_df = pd.concat([
            self.bars_df,
            pd.DataFrame([bar])
        ]).reset_index(drop=True)
        
        print(f"Bar: {bar['symbol']} - O: ${bar['open']:.2f}, H: ${bar['high']:.2f}, L: ${bar['low']:.2f}, C: ${bar['close']:.2f}")
        
    async def start_streaming(self, symbols: list[str]):
        """Start streaming market data for specified symbols."""
        print(f"Starting market data stream for symbols: {', '.join(symbols)}")
        
        # Create WebSocket connection
        self.ws = await self.provider.create_websocket()
        
        # Subscribe to different data types
        await self.provider.subscribe_trades(symbols)
        await self.provider.subscribe_quotes(symbols)
        await self.provider.subscribe_aggregates(symbols)
        
        print("Successfully subscribed to market data streams")
        
        try:
            while True:
                await asyncio.sleep(1)  # Keep the connection alive
        except KeyboardInterrupt:
            print("\nStopping market data stream...")
            await self.provider.unsubscribe_all()
            
    def save_data(self, output_dir: str):
        """Save collected data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.trades_df.empty:
            self.trades_df.to_csv(
                os.path.join(output_dir, 'trades.csv'),
                index=False
            )
        if not self.quotes_df.empty:
            self.quotes_df.to_csv(
                os.path.join(output_dir, 'quotes.csv'),
                index=False
            )
        if not self.bars_df.empty:
            self.bars_df.to_csv(
                os.path.join(output_dir, 'bars.csv'),
                index=False
            )
        
        print(f"Data saved to {output_dir}")

async def main():
    # Replace with your Polygon.io API key
    API_KEY = os.getenv("POLYGON_API_KEY")
    if not API_KEY:
        raise ValueError("Please set POLYGON_API_KEY environment variable")
    
    # Initialize the example
    example = MarketDataStreamingExample(API_KEY)
    await example.initialize()
    
    # List of symbols to stream
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    try:
        # Start streaming
        await example.start_streaming(symbols)
    except KeyboardInterrupt:
        print("\nSaving collected data...")
        example.save_data("market_data")
    finally:
        # Cleanup
        if hasattr(example, 'ws'):
            await example.ws.close()

if __name__ == "__main__":
    asyncio.run(main())
