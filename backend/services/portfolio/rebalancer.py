from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
from ..market_data.provider import MarketDataProvider
from ..risk.metrics import RiskMetrics

class PortfolioHolding(BaseModel):
    symbol: str
    quantity: float
    target_weight: float
    current_weight: Optional[float] = None
    price: Optional[float] = None
    market_value: Optional[float] = None

class RebalanceAction(BaseModel):
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: float
    estimated_value: float
    reason: str

class RebalanceResult(BaseModel):
    actions: List[RebalanceAction]
    total_value: float
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    tracking_error: float
    estimated_turnover: float
    risk_impact: Dict[str, float]

class PortfolioRebalancer:
    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        risk_metrics: RiskMetrics,
        rebalance_threshold: float = 0.02,  # 2% threshold
        min_trade_value: float = 100.0,  # Minimum trade size
        max_turnover: float = 0.20,  # 20% maximum turnover
    ):
        self.market_data = market_data_provider
        self.risk_metrics = risk_metrics
        self.rebalance_threshold = rebalance_threshold
        self.min_trade_value = min_trade_value
        self.max_turnover = max_turnover

    async def analyze_portfolio(
        self,
        holdings: List[PortfolioHolding]
    ) -> Tuple[Dict[str, float], float]:
        """
        Analyze current portfolio state and calculate metrics
        Returns: (current_weights, total_value)
        """
        # Get current prices
        symbols = [h.symbol for h in holdings]
        prices = await self.market_data.get_latest_prices(symbols)
        
        total_value = 0.0
        current_weights = {}
        
        # Calculate current portfolio value and weights
        for holding in holdings:
            price = prices.get(holding.symbol)
            if price is None:
                raise ValueError(f"No price data for {holding.symbol}")
                
            market_value = price * holding.quantity
            total_value += market_value
            holding.price = price
            holding.market_value = market_value
            
        # Calculate weights
        for holding in holdings:
            current_weights[holding.symbol] = holding.market_value / total_value
            holding.current_weight = current_weights[holding.symbol]
            
        return current_weights, total_value

    async def calculate_rebalance_actions(
        self,
        holdings: List[PortfolioHolding],
        cash: float = 0.0
    ) -> RebalanceResult:
        """
        Calculate required rebalancing actions to align portfolio with target weights
        """
        # Get current portfolio state
        current_weights, total_value = await self.analyze_portfolio(holdings)
        total_value += cash  # Include available cash
        
        # Initialize results
        actions: List[RebalanceAction] = []
        target_weights = {h.symbol: h.target_weight for h in holdings}
        
        # Calculate required trades
        for holding in holdings:
            weight_diff = holding.target_weight - holding.current_weight
            
            # Check if rebalancing is needed
            if abs(weight_diff) <= self.rebalance_threshold:
                continue
                
            # Calculate trade size
            target_value = total_value * holding.target_weight
            current_value = holding.market_value
            trade_value = target_value - current_value
            
            # Skip small trades
            if abs(trade_value) < self.min_trade_value:
                continue
                
            # Calculate quantity
            quantity = abs(trade_value / holding.price)
            quantity = round(quantity, 6)  # Round to 6 decimal places
            
            if trade_value > 0:
                action = "BUY"
                reason = "Underweight position"
            else:
                action = "SELL"
                reason = "Overweight position"
                
            actions.append(
                RebalanceAction(
                    symbol=holding.symbol,
                    action=action,
                    quantity=quantity,
                    estimated_value=abs(trade_value),
                    reason=reason
                )
            )
            
        # Calculate metrics
        turnover = sum(a.estimated_value for a in actions) / total_value
        
        # Check turnover constraint
        if turnover > self.max_turnover:
            # Scale back trades proportionally
            scale_factor = self.max_turnover / turnover
            for action in actions:
                action.quantity *= scale_factor
                action.estimated_value *= scale_factor
            turnover = self.max_turnover
            
        # Calculate risk impact
        current_positions = {h.symbol: h.quantity for h in holdings}
        new_positions = current_positions.copy()
        
        for action in actions:
            if action.action == "BUY":
                new_positions[action.symbol] += action.quantity
            else:
                new_positions[action.symbol] -= action.quantity
                
        risk_impact = await self.risk_metrics.compare_portfolios(
            current_positions,
            new_positions
        )
        
        # Calculate tracking error
        tracking_error = await self.risk_metrics.calculate_tracking_error(
            new_positions,
            target_weights
        )
        
        return RebalanceResult(
            actions=actions,
            total_value=total_value,
            current_weights=current_weights,
            target_weights=target_weights,
            tracking_error=tracking_error,
            estimated_turnover=turnover,
            risk_impact=risk_impact
        )

    async def optimize_rebalance(
        self,
        holdings: List[PortfolioHolding],
        cash: float = 0.0,
        constraints: Optional[Dict] = None
    ) -> RebalanceResult:
        """
        Optimize rebalancing actions considering additional constraints
        """
        if constraints is None:
            constraints = {}
            
        # Get initial rebalance suggestion
        result = await self.calculate_rebalance_actions(holdings, cash)
        
        # Apply optimization if needed based on constraints
        if constraints.get("minimize_tracking_error"):
            # TODO: Implement tracking error minimization
            pass
            
        if constraints.get("minimize_turnover"):
            # TODO: Implement turnover minimization
            pass
            
        if constraints.get("sector_constraints"):
            # TODO: Implement sector constraint handling
            pass
            
        return result

    async def validate_rebalance(
        self,
        holdings: List[PortfolioHolding],
        actions: List[RebalanceAction]
    ) -> bool:
        """
        Validate that rebalancing actions are valid and executable
        """
        # Get current prices and validate availability
        symbols = [h.symbol for h in holdings]
        prices = await self.market_data.get_latest_prices(symbols)
        
        if not all(symbol in prices for symbol in symbols):
            return False
            
        # Validate position sizes
        for holding in holdings:
            for action in actions:
                if action.symbol == holding.symbol:
                    if action.action == "SELL" and action.quantity > holding.quantity:
                        return False
                        
        return True
