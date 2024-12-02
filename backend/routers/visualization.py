from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import asyncio

from ..services.realtime_analytics import realtime_analytics_service
from ..analytics.market_analytics import (
    analyze_market_microstructure,
    analyze_liquidity_profile,
    analyze_trading_patterns
)

router = APIRouter()

@router.websocket("/ws/market-analytics/{symbol}")
async def websocket_analytics(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market analytics"""
    await websocket.accept()
    connection_id = str(id(websocket))
    
    try:
        # Subscribe to updates
        await realtime_analytics_service.subscribe(symbol, connection_id)
        
        # Send initial data
        initial_data = await realtime_analytics_service.get_analytics_update(symbol)
        await websocket.send_json({
            'type': 'initial_data',
            'symbol': symbol,
            'data': initial_data
        })
        
        # Listen for updates and client messages
        async for message in realtime_analytics_service.broadcast_analytics():
            if message[0] == connection_id:
                await websocket.send_text(message[1])
                
    except WebSocketDisconnect:
        await realtime_analytics_service.unsubscribe(symbol, connection_id)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await realtime_analytics_service.unsubscribe(symbol, connection_id)

@router.get("/visualization/price-chart/{symbol}")
async def get_price_chart_data(
    symbol: str,
    timeframe: str = Query("1d", regex="^(1d|5d|1mo|3mo|6mo|1y)$"),
    interval: str = Query("1m", regex="^(1m|5m|15m|30m|1h|1d)$")
):
    """Get price chart data with technical indicators"""
    try:
        data = await realtime_analytics_service.get_visualization_data(symbol, 'price')
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'interval': interval,
            'data': data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get price chart data: {str(e)}"
        )

@router.get("/visualization/volume-profile/{symbol}")
async def get_volume_profile(
    symbol: str,
    timeframe: str = Query("1d", regex="^(1d|5d|1mo)$")
):
    """Get volume profile visualization data"""
    try:
        data = await realtime_analytics_service.get_visualization_data(symbol, 'volume')
        patterns = analyze_trading_patterns(symbol)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'volume_data': data,
            'volume_profile': patterns['volume_profile'],
            'optimal_trading_hours': patterns['optimal_trading_hours']
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get volume profile: {str(e)}"
        )

@router.get("/visualization/liquidity-heatmap/{symbol}")
async def get_liquidity_heatmap(
    symbol: str,
    timeframe: str = Query("1d", regex="^(1d|5d|1mo)$")
):
    """Get liquidity heatmap visualization data"""
    try:
        liquidity = analyze_liquidity_profile(symbol)
        patterns = analyze_trading_patterns(symbol)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'liquidity_profile': liquidity,
            'trading_patterns': patterns,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get liquidity heatmap: {str(e)}"
        )

@router.get("/visualization/market-impact/{symbol}")
async def get_market_impact_visualization(
    symbol: str,
    quantity: int = Query(..., gt=0),
    price: Optional[float] = None
):
    """Get market impact visualization data"""
    try:
        # Get current market data
        analytics = await realtime_analytics_service.get_analytics_update(symbol)
        current_price = price or analytics['real_time_metrics']['price']
        
        # Calculate impact for different participation rates
        participation_rates = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        impact_curve = []
        
        for rate in participation_rates:
            impact = await estimate_market_impact(
                symbol=symbol,
                quantity=int(quantity * rate),
                price=current_price
            )
            impact_curve.append({
                'participation_rate': rate,
                'price_impact': impact.price_impact,
                'market_impact_cost': impact.market_impact_cost
            })
            
        return {
            'symbol': symbol,
            'quantity': quantity,
            'current_price': current_price,
            'impact_curve': impact_curve,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get market impact visualization: {str(e)}"
        )
