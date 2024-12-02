import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor

def calculate_market_impact(
    price: float,
    quantity: int,
    adv: float,  # Average Daily Volume
    volatility: float,
    spread: float
) -> float:
    """
    Calculate estimated market impact of a trade
    Uses square-root model with volatility adjustment
    """
    participation_rate = (quantity / adv) if adv > 0 else 0
    volatility_factor = volatility * np.sqrt(252)  # Annualized volatility
    
    # Square root impact model with spread and volatility adjustments
    impact = (0.1 * spread + 0.1 * volatility_factor) * np.sqrt(participation_rate)
    return impact * price * quantity

def calculate_transaction_costs(
    price: float,
    quantity: int,
    market_impact: float,
    commission_rate: float = 0.001,
    spread: float = 0
) -> Dict[str, float]:
    """
    Calculate detailed transaction costs
    """
    notional_value = price * quantity
    commission = notional_value * commission_rate
    spread_cost = (spread / 2) * notional_value
    
    return {
        "commission": commission,
        "spread_cost": spread_cost,
        "market_impact": market_impact,
        "total_cost": commission + spread_cost + market_impact,
        "total_cost_bps": (commission + spread_cost + market_impact) / notional_value * 10000
    }

def analyze_market_microstructure(
    symbol: str,
    lookback_days: int = 30
) -> Dict[str, float]:
    """
    Analyze market microstructure metrics
    """
    try:
        # Fetch detailed tick data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{lookback_days}d", interval="1m")
        
        if data.empty:
            return {}
        
        # Calculate metrics
        returns = data['Close'].pct_change().dropna()
        volume = data['Volume']
        
        metrics = {
            "avg_trade_size": volume.mean(),
            "volatility": returns.std() * np.sqrt(252 * 390),  # Annualized from minute data
            "volume_profile": volume.groupby(volume.index.hour).mean().to_dict(),
            "serial_correlation": returns.autocorr(),
            "volume_volatility_corr": returns.abs().corr(volume),
            "bid_ask_bounce": np.mean(np.abs(returns)) * np.sqrt(390),  # Estimated from returns
            "effective_spread": np.mean(np.abs(returns)) * 2,
            "price_impact": calculate_price_impact(returns, volume)
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error analyzing market microstructure for {symbol}: {str(e)}")
        return {}

def calculate_price_impact(returns: pd.Series, volume: pd.Series) -> float:
    """
    Calculate permanent price impact coefficient
    Uses Kyle's lambda methodology
    """
    signed_volume = volume * np.sign(returns)
    model = stats.linregress(signed_volume, returns)
    return abs(model.slope)

def analyze_liquidity_profile(
    symbol: str,
    lookback_days: int = 30
) -> Dict[str, any]:
    """
    Analyze detailed liquidity profile
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{lookback_days}d", interval="1h")
        
        if data.empty:
            return {}
        
        volume = data['Volume']
        close = data['Close']
        
        # Calculate liquidity metrics
        turnover = (volume * close).mean()
        adv = volume.mean()
        
        # Time-weighted metrics
        volume_profile = volume.groupby(volume.index.hour).mean()
        peak_volume_hour = volume_profile.idxmax()
        
        # Volatility-adjusted volume
        returns = close.pct_change()
        volatility = returns.std() * np.sqrt(252)
        vol_adj_volume = volume.mean() / volatility
        
        # Liquidity cost score
        spread_proxy = 2 * np.sqrt(np.pi/2) * returns.std()
        lcs = spread_proxy * np.sqrt(1e6 / adv)
        
        return {
            "avg_daily_volume": adv,
            "turnover": turnover,
            "volume_profile": volume_profile.to_dict(),
            "peak_volume_hour": peak_volume_hour,
            "volatility_adjusted_volume": vol_adj_volume,
            "liquidity_cost_score": lcs,
            "spread_proxy": spread_proxy
        }
        
    except Exception as e:
        print(f"Error analyzing liquidity profile for {symbol}: {str(e)}")
        return {}

def analyze_trading_patterns(
    symbol: str,
    lookback_days: int = 30
) -> Dict[str, any]:
    """
    Analyze trading patterns for optimal execution
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{lookback_days}d", interval="5m")
        
        if data.empty:
            return {}
        
        volume = data['Volume']
        returns = data['Close'].pct_change()
        
        # Intraday patterns
        volume_profile = volume.groupby(volume.index.hour).mean()
        volatility_profile = returns.groupby(returns.index.hour).std()
        
        # Volume-volatility relationship
        vol_vol_corr = returns.abs().corr(volume)
        
        # Momentum and mean reversion
        autocorr_5m = returns.autocorr()
        autocorr_1h = returns.resample('1H').mean().autocorr()
        
        # Price impact decay
        impact_decay = calculate_impact_decay(returns, volume)
        
        return {
            "volume_profile": volume_profile.to_dict(),
            "volatility_profile": volatility_profile.to_dict(),
            "volume_volatility_correlation": vol_vol_corr,
            "short_term_autocorr": autocorr_5m,
            "hourly_autocorr": autocorr_1h,
            "impact_decay": impact_decay,
            "optimal_trading_hours": find_optimal_trading_hours(volume_profile, volatility_profile)
        }
        
    except Exception as e:
        print(f"Error analyzing trading patterns for {symbol}: {str(e)}")
        return {}

def calculate_impact_decay(returns: pd.Series, volume: pd.Series) -> Dict[str, float]:
    """
    Calculate price impact decay parameters
    """
    lags = [1, 2, 3, 4, 5, 10, 15, 20]
    decay = {}
    
    signed_volume = volume * np.sign(returns)
    for lag in lags:
        lagged_impact = stats.linregress(signed_volume, returns.shift(-lag))
        decay[f"lag_{lag}"] = abs(lagged_impact.slope)
    
    return decay

def find_optimal_trading_hours(
    volume_profile: pd.Series,
    volatility_profile: pd.Series
) -> List[Dict[str, any]]:
    """
    Find optimal trading hours based on volume and volatility profiles
    """
    # Calculate trading quality score
    quality_score = volume_profile / (volatility_profile + 1e-10)
    quality_score = quality_score / quality_score.max()
    
    # Find best hours
    best_hours = []
    for hour, score in quality_score.items():
        best_hours.append({
            "hour": hour,
            "score": score,
            "volume_percentile": volume_profile[hour] / volume_profile.max(),
            "volatility_percentile": volatility_profile[hour] / volatility_profile.max()
        })
    
    return sorted(best_hours, key=lambda x: x['score'], reverse=True)

def get_market_depth(symbol: str) -> Dict[str, any]:
    """
    Get market depth analysis (simplified due to data limitations)
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return {
            "bid_price": info.get("bid", 0),
            "ask_price": info.get("ask", 0),
            "bid_size": info.get("bidSize", 0),
            "ask_size": info.get("askSize", 0),
            "last_price": info.get("regularMarketPrice", 0),
            "last_size": info.get("regularMarketVolume", 0),
            "spread": (info.get("ask", 0) - info.get("bid", 0)) / info.get("regularMarketPrice", 1),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        print(f"Error getting market depth for {symbol}: {str(e)}")
        return {}
