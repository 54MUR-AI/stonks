import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from ..schemas.rebalancing import (
    TargetAllocation, RebalanceConstraints, TradeAction,
    RebalanceAnalysis, RebalanceStrategy
)

def calculate_current_weights(positions: List[Dict]) -> Dict[str, float]:
    """Calculate current portfolio weights"""
    total_value = sum(pos['current_value'] for pos in positions)
    if total_value == 0:
        return {}
    
    return {
        pos['symbol']: (pos['current_value'] / total_value) * 100
        for pos in positions
    }

def calculate_trade_size(
    current_value: float,
    target_value: float,
    current_price: float,
    min_trade: float,
    max_trade: Optional[float] = None
) -> int:
    """Calculate optimal trade size in shares"""
    trade_value = target_value - current_value
    if abs(trade_value) < min_trade:
        return 0
        
    if max_trade and abs(trade_value) > max_trade:
        trade_value = max_trade if trade_value > 0 else -max_trade
        
    shares = int(trade_value / current_price)
    return shares

def estimate_tax_impact(
    position: Dict,
    trade_value: float,
    tax_rate: float = 0.20
) -> float:
    """Estimate tax impact of a trade"""
    if trade_value >= 0:  # Buying has no tax impact
        return 0.0
        
    cost_basis = position['avg_price'] * position['quantity']
    current_value = position['current_value']
    if current_value <= cost_basis:  # Tax loss harvesting opportunity
        return (cost_basis - current_value) * tax_rate * -1  # Tax benefit
    else:
        return (current_value - cost_basis) * tax_rate  # Tax cost

def calculate_risk_score(weights: Dict[str, float], historical_data: pd.DataFrame) -> float:
    """Calculate portfolio risk score based on historical volatility"""
    returns = historical_data.pct_change().dropna()
    portfolio_returns = sum(weight * returns[symbol] for symbol, weight in weights.items())
    volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Convert volatility to 1-10 risk score
    risk_score = min(10, max(1, volatility * 10))
    return risk_score

def calculate_tracking_error(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    historical_data: pd.DataFrame
) -> float:
    """Calculate tracking error between current and target portfolio"""
    returns = historical_data.pct_change().dropna()
    
    current_returns = sum(weight * returns[symbol] for symbol, weight in current_weights.items())
    target_returns = sum(weight * returns[symbol] for symbol, weight in target_weights.items())
    
    tracking_diff = current_returns - target_returns
    tracking_error = tracking_diff.std() * np.sqrt(252)
    return tracking_error

def generate_rebalancing_plan(
    positions: List[Dict],
    target_allocations: List[TargetAllocation],
    constraints: RebalanceConstraints,
    historical_data: pd.DataFrame
) -> RebalanceAnalysis:
    """Generate optimal rebalancing plan"""
    # Calculate current portfolio state
    total_value = sum(pos['current_value'] for pos in positions)
    current_weights = calculate_current_weights(positions)
    
    # Create position lookup
    position_map = {pos['symbol']: pos for pos in positions}
    
    # Calculate target values
    target_map = {t.symbol: t.target_weight for t in target_allocations}
    trades = []
    
    # Calculate initial risk metrics
    risk_score_before = calculate_risk_score(current_weights, historical_data)
    
    # Initialize tracking
    cash_position = sum(pos['current_value'] for pos in positions if pos['symbol'] == 'CASH')
    estimated_commission = 0
    total_tax_impact = 0
    
    # Generate trades
    for target in target_allocations:
        symbol = target.symbol
        target_value = (target.target_weight / 100) * total_value
        current_value = position_map.get(symbol, {'current_value': 0})['current_value']
        
        # Skip if within tolerance
        current_weight = current_weights.get(symbol, 0)
        if abs(current_weight - target.target_weight) <= target.tolerance:
            continue
            
        # Calculate trade
        price = historical_data[symbol].iloc[-1]
        shares = calculate_trade_size(
            current_value,
            target_value,
            price,
            constraints.min_trade_amount,
            constraints.max_trade_amount
        )
        
        if shares == 0:
            continue
            
        # Calculate trade impact
        trade_value = shares * price
        action = "buy" if shares > 0 else "sell"
        
        # Estimate tax impact
        if constraints.tax_loss_harvest and action == "sell":
            tax_impact = estimate_tax_impact(position_map[symbol], trade_value)
            total_tax_impact += tax_impact
            
        # Add trading commission
        estimated_commission += 1  # Placeholder for commission calculation
        
        trades.append(TradeAction(
            symbol=symbol,
            action=action,
            shares=abs(shares),
            estimated_value=abs(trade_value),
            current_weight=current_weight,
            target_weight=target.target_weight,
            price_estimate=price
        ))
    
    # Calculate post-rebalance weights
    post_weights = current_weights.copy()
    for trade in trades:
        symbol = trade.symbol
        trade_value = trade.estimated_value * (1 if trade.action == "buy" else -1)
        new_value = position_map.get(symbol, {'current_value': 0})['current_value'] + trade_value
        post_weights[symbol] = (new_value / total_value) * 100
    
    # Calculate final metrics
    risk_score_after = calculate_risk_score(post_weights, historical_data)
    tracking_error = calculate_tracking_error(current_weights, target_map, historical_data)
    
    return RebalanceAnalysis(
        current_total=total_value,
        target_total=total_value,
        trades=trades,
        estimated_commission=estimated_commission,
        tax_impact=total_tax_impact if constraints.tax_loss_harvest else None,
        risk_score_before=risk_score_before,
        risk_score_after=risk_score_after,
        tracking_error_impact=tracking_error,
        cash_position=cash_position
    )

def get_predefined_strategies() -> List[RebalanceStrategy]:
    """Get list of predefined portfolio strategies"""
    return [
        RebalanceStrategy(
            name="Conservative",
            description="Focus on capital preservation with steady income",
            target_allocations=[
                TargetAllocation(symbol="AGG", target_weight=60, tolerance=2.0),
                TargetAllocation(symbol="SPY", target_weight=30, tolerance=2.0),
                TargetAllocation(symbol="CASH", target_weight=10, tolerance=1.0)
            ],
            risk_score=3.0,
            expected_return=5.0,
            expected_volatility=6.0,
            sharpe_ratio=0.6
        ),
        RebalanceStrategy(
            name="Moderate Growth",
            description="Balanced approach with focus on long-term growth",
            target_allocations=[
                TargetAllocation(symbol="SPY", target_weight=60, tolerance=2.0),
                TargetAllocation(symbol="AGG", target_weight=30, tolerance=2.0),
                TargetAllocation(symbol="CASH", target_weight=10, tolerance=1.0)
            ],
            risk_score=5.0,
            expected_return=7.0,
            expected_volatility=10.0,
            sharpe_ratio=0.5
        ),
        RebalanceStrategy(
            name="Aggressive Growth",
            description="Maximum growth potential with higher volatility",
            target_allocations=[
                TargetAllocation(symbol="SPY", target_weight=70, tolerance=2.0),
                TargetAllocation(symbol="QQQ", target_weight=20, tolerance=2.0),
                TargetAllocation(symbol="CASH", target_weight=10, tolerance=1.0)
            ],
            risk_score=7.0,
            expected_return=9.0,
            expected_volatility=15.0,
            sharpe_ratio=0.4
        )
    ]
