from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict
from datetime import datetime, timedelta

from ..database import get_db
from ..models import Portfolio, Position, Trade
from ..schemas.trading import (
    TradeExecution, ExecutionResult, MarketHours,
    MarketCondition, OrderStatus
)
from ..analytics.trading import TradeExecutor
from ..analytics.market_data import get_market_summary
from ..analytics.market_analytics import (
    analyze_market_microstructure,
    analyze_liquidity_profile,
    analyze_trading_patterns,
    get_market_depth,
    calculate_market_impact,
    calculate_transaction_costs
)
from ..schemas.market_analytics import (
    MarketAnalytics,
    MarketImpact,
    TransactionCosts,
    MarketMicrostructure,
    LiquidityProfile,
    TradingPatterns,
    MarketDepth,
    MarketSignal, RiskMetrics
)
from ..services.market_signals import market_signals_service
from ..services.risk_metrics import risk_metrics_service
from ..services.portfolio_optimization import portfolio_optimizer
from ..services.risk_alerts import risk_alert_service

# Optional ML-heavy services (require TensorFlow/PyTorch)
try:
    from ..services.stress_testing import stress_testing_service, ScenarioType, StressScenario
    from ..services.risk_prediction import risk_predictor
    from ..services.portfolio_rebalancing import portfolio_rebalancer
    ML_SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"ML services not available (missing dependencies): {e}")
    stress_testing_service = None
    risk_predictor = None
    portfolio_rebalancer = None
    ScenarioType = None
    StressScenario = None
    ML_SERVICES_AVAILABLE = False

router = APIRouter()
trade_executor = TradeExecutor()

@router.post("/portfolios/{portfolio_id}/execute-trades", response_model=ExecutionResult)
async def execute_trades(
    portfolio_id: int,
    trade_execution: TradeExecution,
    db: Session = Depends(get_db)
):
    """
    Execute trades for a portfolio using specified execution strategy
    """
    # Validate portfolio
    portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Execute trades
    try:
        result = await trade_executor.execute_trades(
            trade_execution.orders,
            trade_execution.execution_params,
            trade_execution.dry_run
        )
        
        # Record trades in database if not dry run
        if not trade_execution.dry_run and result.success:
            for order_result in result.orders:
                if order_result["status"] == OrderStatus.FILLED:
                    trade = Trade(
                        portfolio_id=portfolio_id,
                        symbol=order_result["symbol"],
                        quantity=order_result["filled_quantity"],
                        price=order_result["average_price"],
                        timestamp=datetime.now(),
                        type="market",  # Simplified for now
                        status="completed"
                    )
                    db.add(trade)
            
            try:
                db.commit()
            except Exception as e:
                db.rollback()
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to record trades: {str(e)}"
                )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Trade execution failed: {str(e)}"
        )

@router.get("/market/hours", response_model=Dict[str, MarketHours])
async def get_market_hours():
    """
    Get trading hours for various markets
    """
    # Simplified for now - just returning NYSE hours
    now = datetime.now()
    market_open = datetime.now().replace(hour=9, minute=30, second=0)
    market_close = datetime.now().replace(hour=16, minute=0, second=0)
    
    return {
        "NYSE": MarketHours(
            market="NYSE",
            is_open=market_open.time() <= now.time() <= market_close.time(),
            next_open=market_open if now.time() < market_open.time() else market_open.replace(day=market_open.day + 1),
            next_close=market_close if now.time() < market_close.time() else market_close.replace(day=market_close.day + 1),
            trading_hours=[{"open": "09:30", "close": "16:00"}],
            holidays=[]  # TODO: Add actual market holidays
        )
    }

@router.get("/market/conditions/{symbol}", response_model=MarketCondition)
async def get_market_conditions(symbol: str):
    """
    Get current market conditions for a symbol
    """
    try:
        market_data = get_market_summary([symbol])
        if symbol not in market_data:
            raise HTTPException(
                status_code=404,
                detail=f"No market data available for {symbol}"
            )
            
        data = market_data[symbol]
        return MarketCondition(
            symbol=symbol,
            last_price=data.get("price", 0),
            bid_price=data.get("bid_price", 0),
            ask_price=data.get("ask_price", 0),
            volume=data.get("volume", 0),
            vwap=data.get("vwap", 0),
            volatility=data.get("volatility", 0),
            spread=data.get("spread", 0),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get market conditions: {str(e)}"
        )

@router.get("/market/analytics/{symbol}", response_model=MarketAnalytics)
async def get_market_analytics(symbol: str):
    """
    Get comprehensive market analytics for a symbol
    """
    try:
        microstructure = analyze_market_microstructure(symbol)
        liquidity = analyze_liquidity_profile(symbol)
        patterns = analyze_trading_patterns(symbol)
        depth = get_market_depth(symbol)
        
        return MarketAnalytics(
            symbol=symbol,
            microstructure=MarketMicrostructure(**microstructure),
            liquidity_profile=LiquidityProfile(**liquidity),
            trading_patterns=TradingPatterns(**patterns),
            market_depth=MarketDepth(**depth),
            last_updated=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get market analytics: {str(e)}"
        )

@router.post("/market/impact", response_model=MarketImpact)
async def estimate_market_impact(
    symbol: str,
    quantity: int,
    price: float = Query(None),
    participation_rate: float = Query(0.1, gt=0, le=1)
):
    """
    Estimate market impact for a potential trade
    """
    try:
        # Get market data
        liquidity = analyze_liquidity_profile(symbol)
        microstructure = analyze_market_microstructure(symbol)
        depth = get_market_depth(symbol)
        
        if not price:
            price = depth["last_price"]
            
        # Calculate impact
        adv = liquidity["avg_daily_volume"]
        volatility = microstructure["volatility"]
        spread = depth["spread"]
        
        impact = calculate_market_impact(
            price=price,
            quantity=quantity,
            adv=adv,
            volatility=volatility,
            spread=spread
        )
        
        return MarketImpact(
            price_impact=impact / price * 10000,  # Convert to bps
            volume_participation=quantity / adv * 100,
            market_impact_cost=impact,
            recovery_time=15  # Simplified estimate
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to estimate market impact: {str(e)}"
        )

@router.post("/market/transaction-costs", response_model=TransactionCosts)
async def estimate_transaction_costs(
    symbol: str,
    quantity: int,
    price: float = Query(None),
    commission_rate: float = Query(0.001, gt=0)
):
    """
    Estimate total transaction costs for a trade
    """
    try:
        # Get market data
        depth = get_market_depth(symbol)
        
        if not price:
            price = depth["last_price"]
            
        # Get market impact
        impact_result = await estimate_market_impact(
            symbol=symbol,
            quantity=quantity,
            price=price
        )
        
        # Calculate costs
        costs = calculate_transaction_costs(
            price=price,
            quantity=quantity,
            market_impact=impact_result.market_impact_cost,
            commission_rate=commission_rate,
            spread=depth["spread"]
        )
        
        return TransactionCosts(**costs)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to estimate transaction costs: {str(e)}"
        )

@router.get("/market/signals/{symbol}", response_model=List[MarketSignal])
async def get_market_signals(
    symbol: str,
    lookback_days: int = Query(30, gt=0, le=365)
):
    """
    Get comprehensive market signals for a symbol
    """
    try:
        # Fetch historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{lookback_days}d", interval="1d")
        
        if hist.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {symbol}"
            )
            
        # Generate signals
        signals = []
        
        # Price momentum signals
        momentum_signals = market_signals_service.analyze_price_momentum(
            hist['Close'],
            hist['Volume']
        )
        signals.extend(momentum_signals)
        
        # Volume signals
        volume_signals = market_signals_service.detect_volume_spikes(
            hist['Volume'],
            hist['Close']
        )
        signals.extend(volume_signals)
        
        # Volatility regime signals
        volatility_signals = market_signals_service.analyze_volatility_regime(
            hist['Close']
        )
        signals.extend(volatility_signals)
        
        # Technical signals
        technical_signals = market_signals_service.generate_technical_signals(
            hist['Close'],
            hist['Volume']
        )
        signals.extend(technical_signals)
        
        # Market regime signals
        spy = yf.Ticker("SPY")
        market_hist = spy.history(period=f"{lookback_days}d", interval="1d")
        
        if not market_hist.empty:
            regime_signals = market_signals_service.analyze_market_regime(
                hist['Close'],
                market_hist['Close']
            )
            signals.extend(regime_signals)
            
        return signals
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get market signals: {str(e)}"
        )

@router.get("/portfolio/{portfolio_id}/risk-metrics", response_model=RiskMetrics)
async def get_portfolio_risk_metrics(
    portfolio_id: int,
    lookback_days: int = Query(252, gt=0, le=1000),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive risk metrics for a portfolio
    """
    try:
        # Get portfolio
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
            
        # Get positions
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        if not positions:
            raise HTTPException(status_code=404, detail="No positions found")
            
        # Fetch historical data
        symbols = [pos.symbol for pos in positions]
        weights = np.array([pos.quantity * pos.average_price for pos in positions])
        weights = weights / weights.sum()
        
        returns_data = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{lookback_days}d", interval="1d")
            if not hist.empty:
                returns_data.append(hist['Close'].pct_change())
                
        if not returns_data:
            raise HTTPException(
                status_code=404,
                detail="No historical data available"
            )
            
        # Calculate portfolio returns
        returns_df = pd.concat(returns_data, axis=1)
        returns_df.columns = symbols
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Get market returns
        spy = yf.Ticker("SPY")
        market_hist = spy.history(period=f"{lookback_days}d", interval="1d")
        market_returns = market_hist['Close'].pct_change()
        
        # Calculate risk metrics
        risk_metrics = risk_metrics_service.calculate_portfolio_risk_metrics(
            portfolio_returns,
            market_returns,
            weights
        )
        
        return risk_metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get risk metrics: {str(e)}"
        )

@router.post("/portfolio/{portfolio_id}/stress-test")
async def stress_test_portfolio(
    portfolio_id: int,
    scenarios: Dict[str, Dict[str, float]],
    db: Session = Depends(get_db)
):
    """
    Run stress tests on portfolio with custom scenarios
    """
    if not ML_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML services not available")
    try:
        # Get portfolio
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
            
        # Get positions
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        if not positions:
            raise HTTPException(status_code=404, detail="No positions found")
            
        # Fetch historical data
        symbols = [pos.symbol for pos in positions]
        weights = np.array([pos.quantity * pos.average_price for pos in positions])
        weights = weights / weights.sum()
        
        returns_data = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y", interval="1d")
            if not hist.empty:
                returns_data.append(hist['Close'].pct_change())
                
        if not returns_data:
            raise HTTPException(
                status_code=404,
                detail="No historical data available"
            )
            
        # Prepare returns dataframe
        returns_df = pd.concat(returns_data, axis=1)
        returns_df.columns = symbols
        
        # Perform stress testing
        stress_results = risk_metrics_service.stress_test_portfolio(
            returns_df,
            weights,
            scenarios
        )
        
        return stress_results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform stress test: {str(e)}"
        )

@router.get("/stress-test/scenarios", response_model=Dict[str, List[Dict]])
async def get_stress_test_scenarios(
    portfolio_id: int,
    scenario_type: Optional[str] = Query(None, description="Type of scenarios: historical, monte_carlo, sensitivity"),
    db: Session = Depends(get_db)
):
    """Get available stress test scenarios"""
    if not ML_SERVICES_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="ML services not available - missing TensorFlow/PyTorch dependencies"
        )
    
    try:
        if scenario_type == "historical":
            scenarios = stress_testing_service.get_available_historical_scenarios()
            return {
                "scenarios": [
                    {
                        "name": name,
                        "type": "historical",
                        "description": stress_testing_service.get_historical_scenario(name).description
                    }
                    for name in scenarios
                ]
            }
            
        # For Monte Carlo, we need portfolio data to generate scenarios
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
            
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        if not positions:
            raise HTTPException(status_code=404, detail="No positions found")
            
        symbols = [pos.symbol for pos in positions]
        
        if scenario_type == "monte_carlo":
            scenarios = stress_testing_service.generate_monte_carlo_scenarios(symbols)
            return {
                "scenarios": [
                    {
                        "name": scenario.name,
                        "type": "monte_carlo",
                        "description": scenario.description,
                        "probability": scenario.probability
                    }
                    for scenario in scenarios
                ]
            }
            
        elif scenario_type == "sensitivity":
            # Use current portfolio weights as base case
            total_value = sum(pos.quantity * pos.average_price for pos in positions)
            base_shocks = {
                pos.symbol: -0.1  # 10% base shock
                for pos in positions
            }
            
            scenarios = stress_testing_service.run_sensitivity_analysis(
                symbols,
                base_shocks
            )
            
            return {
                "scenarios": [
                    {
                        "name": scenario.name,
                        "type": "sensitivity",
                        "description": scenario.description
                    }
                    for scenario in scenarios
                ]
            }
            
        # Return all available scenarios if no type specified
        historical = stress_testing_service.get_available_historical_scenarios()
        monte_carlo = stress_testing_service.generate_monte_carlo_scenarios(symbols, num_scenarios=5)
        sensitivity = stress_testing_service.run_sensitivity_analysis(symbols, {s: -0.1 for s in symbols}, steps=2)
        
        return {
            "scenarios": [
                *[{
                    "name": name,
                    "type": "historical",
                    "description": stress_testing_service.get_historical_scenario(name).description
                } for name in historical],
                *[{
                    "name": scenario.name,
                    "type": "monte_carlo",
                    "description": scenario.description,
                    "probability": scenario.probability
                } for scenario in monte_carlo],
                *[{
                    "name": scenario.name,
                    "type": "sensitivity",
                    "description": scenario.description
                } for scenario in sensitivity]
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get scenarios: {str(e)}"
        )

@router.post("/portfolio/{portfolio_id}/stress-test/run")
async def run_stress_test(
    portfolio_id: int,
    scenario_name: str = Query(..., description="Name of the scenario to run"),
    scenario_type: str = Query(..., description="Type: historical, hypothetical, monte_carlo, sensitivity"),
    custom_scenario: Optional[Dict] = None,
    db: Session = Depends(get_db)
):
    """Run a specific stress test scenario"""
    if not ML_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML services not available")
    try:
        # Get portfolio data
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
            
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        if not positions:
            raise HTTPException(status_code=404, detail="No positions found")
            
        # Prepare portfolio data
        portfolio_values = {
            pos.symbol: pos.quantity * pos.average_price
            for pos in positions
        }
        
        # Get scenario
        if scenario_type == "historical":
            scenario = stress_testing_service.get_historical_scenario(scenario_name)
            if not scenario:
                raise HTTPException(
                    status_code=404,
                    detail=f"Historical scenario {scenario_name} not found"
                )
                
        elif scenario_type == "hypothetical" and custom_scenario:
            scenario = stress_testing_service.create_hypothetical_scenario(
                name=scenario_name,
                description=custom_scenario.get("description", "Custom scenario"),
                shocks=custom_scenario["shocks"],
                correlation_adjustments=custom_scenario.get("correlation_adjustments"),
                volatility_adjustments=custom_scenario.get("volatility_adjustments")
            )
            
        elif scenario_type == "monte_carlo":
            symbols = list(portfolio_values.keys())
            scenarios = stress_testing_service.generate_monte_carlo_scenarios(symbols)
            scenario = next(
                (s for s in scenarios if s.name == scenario_name),
                None
            )
            if not scenario:
                raise HTTPException(
                    status_code=404,
                    detail=f"Monte Carlo scenario {scenario_name} not found"
                )
                
        elif scenario_type == "sensitivity":
            symbols = list(portfolio_values.keys())
            base_shocks = {symbol: -0.1 for symbol in symbols}
            scenarios = stress_testing_service.run_sensitivity_analysis(
                symbols,
                base_shocks
            )
            scenario = next(
                (s for s in scenarios if s.name == scenario_name),
                None
            )
            if not scenario:
                raise HTTPException(
                    status_code=404,
                    detail=f"Sensitivity scenario {scenario_name} not found"
                )
                
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid scenario type or missing custom scenario data"
            )
            
        # Run stress test
        result = stress_testing_service.run_stress_test(portfolio_values, scenario)
        
        return {
            "scenario_name": result.scenario_name,
            "portfolio_impact": result.portfolio_impact,
            "asset_impacts": result.asset_impacts,
            "risk_metrics": result.risk_metrics,
            "correlation_changes": result.correlation_changes,
            "volatility_changes": result.volatility_changes
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run stress test: {str(e)}"
        )

@router.post("/portfolio/{portfolio_id}/optimize")
async def optimize_portfolio(
    portfolio_id: int,
    optimization_type: str = Query(
        "mean_variance",
        regex="^(mean_variance|risk_parity|black_litterman|hierarchical)$"
    ),
    target_return: Optional[float] = None,
    target_risk: Optional[float] = None,
    constraints: Optional[Dict] = None,
    db: Session = Depends(get_db)
):
    """
    Optimize portfolio allocation using various strategies
    """
    try:
        # Get portfolio
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
            
        # Get positions
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        if not positions:
            raise HTTPException(status_code=404, detail="No positions found")
            
        # Fetch historical data
        symbols = [pos.symbol for pos in positions]
        returns_data = []
        market_caps = []
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y", interval="1d")
            if not hist.empty:
                returns_data.append(hist['Close'].pct_change())
                market_caps.append(ticker.info.get('marketCap', 1e9))  # Default to 1B if not available
                
        if not returns_data:
            raise HTTPException(
                status_code=404,
                detail="No historical data available"
            )
            
        # Prepare returns dataframe
        returns_df = pd.concat(returns_data, axis=1)
        returns_df.columns = symbols
        
        # Perform optimization
        if optimization_type == "mean_variance":
            result = portfolio_optimizer.mean_variance_optimization(
                returns_df,
                target_return=target_return,
                target_risk=target_risk,
                constraints=constraints
            )
        elif optimization_type == "risk_parity":
            result = portfolio_optimizer.risk_parity_optimization(returns_df)
        elif optimization_type == "black_litterman":
            result = portfolio_optimizer.black_litterman_optimization(
                returns_df,
                np.array(market_caps),
                views=[],  # Add views if available
                confidence=[1.0]  # Add confidence levels if available
            )
        else:  # hierarchical
            result = portfolio_optimizer.hierarchical_risk_parity(returns_df)
            
        return {
            "portfolio_id": portfolio_id,
            "optimization_type": optimization_type,
            "weights": dict(zip(symbols, result.weights)),
            "expected_return": result.expected_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "metadata": result.metadata
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize portfolio: {str(e)}"
        )

@router.websocket("/ws/risk-alerts/{portfolio_id}")
async def websocket_risk_alerts(
    websocket: WebSocket,
    portfolio_id: int,
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time risk alerts"""
    await websocket.accept()
    connection_id = str(id(websocket))
    
    try:
        # Subscribe to alerts
        await risk_alert_service.subscribe(portfolio_id, connection_id)
        
        # Get portfolio data
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            await websocket.close(code=4004, reason="Portfolio not found")
            return
            
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        if not positions:
            await websocket.close(code=4004, reason="No positions found")
            return
            
        # Prepare position data
        position_values = {
            pos.symbol: pos.quantity * pos.average_price
            for pos in positions
        }
        
        # Fetch historical data
        symbols = list(position_values.keys())
        returns_data = []
        
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y", interval="1d")
            if not hist.empty:
                returns_data.append(hist['Close'].pct_change())
                
        if returns_data:
            returns_df = pd.concat(returns_data, axis=1)
            returns_df.columns = symbols
            
            # Monitor portfolio and broadcast alerts
            while True:
                alerts = await risk_alert_service.monitor_portfolio(
                    portfolio_id,
                    returns_df,
                    position_values
                )
                
                if alerts:
                    await websocket.send_json({
                        "type": "risk_alerts",
                        "alerts": [
                            {
                                "type": alert.type.value,
                                "timestamp": alert.timestamp.isoformat(),
                                "severity": alert.severity,
                                "message": alert.message,
                                "metadata": alert.metadata
                            }
                            for alert in alerts
                        ]
                    })
                    
                await asyncio.sleep(60)  # Check every minute
                
    except WebSocketDisconnect:
        await risk_alert_service.unsubscribe(portfolio_id, connection_id)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await risk_alert_service.unsubscribe(portfolio_id, connection_id)

@router.get("/portfolio/{portfolio_id}/risk-alerts/history")
async def get_risk_alert_history(
    portfolio_id: int,
    start_date: datetime = Query(default=None),
    end_date: datetime = Query(default=None),
    severity: Optional[str] = Query(None, regex="^(low|medium|high|critical)$"),
    alert_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get historical risk alerts for a portfolio"""
    try:
        if portfolio_id not in risk_alert_service.alert_history:
            return {"alerts": []}
            
        alerts = risk_alert_service.alert_history[portfolio_id]
        
        # Apply filters
        if start_date:
            alerts = [a for a in alerts if a.timestamp >= start_date]
        if end_date:
            alerts = [a for a in alerts if a.timestamp <= end_date]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.type.value == alert_type]
            
        return {
            "alerts": [
                {
                    "type": alert.type.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "message": alert.message,
                    "metadata": alert.metadata
                }
                for alert in alerts
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get risk alert history: {str(e)}"
        )

@router.post("/portfolio/{portfolio_id}/risk/predict")
async def predict_portfolio_risk(
    portfolio_id: int,
    days_forward: int = Query(5, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """Predict portfolio risk using ML models"""
    try:
        # Get portfolio data
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
            
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        if not positions:
            raise HTTPException(status_code=404, detail="No positions found")
            
        # Get risk predictions for each position
        risk_predictions = {}
        model_predictions = {}
        confidence_intervals = {}
        
        for position in positions:
            prediction = risk_predictor.predict_risk(position.symbol, days_forward)
            risk_predictions[position.symbol] = prediction['predicted_volatility']
            model_predictions[position.symbol] = prediction['model_predictions']
            confidence_intervals[position.symbol] = prediction['confidence_interval']
            
        return {
            "portfolio_id": portfolio_id,
            "predictions": {
                "risk_predictions": risk_predictions,
                "model_predictions": model_predictions,
                "confidence_intervals": confidence_intervals
            },
            "metadata": {
                "days_forward": days_forward,
                "prediction_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to predict portfolio risk: {str(e)}"
        )

@router.post("/portfolio/{portfolio_id}/risk/train")
async def train_risk_models(
    portfolio_id: int,
    training_days: int = Query(365, ge=180, le=1800),
    db: Session = Depends(get_db)
):
    """Train risk prediction models for portfolio assets"""
    try:
        # Get portfolio data
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
            
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        if not positions:
            raise HTTPException(status_code=404, detail="No positions found")
            
        # Train models for each position
        training_results = {}
        start_date = datetime.now() - timedelta(days=training_days)
        
        for position in positions:
            metrics = risk_predictor.train_models(position.symbol, start_date)
            training_results[position.symbol] = metrics
            
        return {
            "portfolio_id": portfolio_id,
            "training_results": training_results,
            "metadata": {
                "training_days": training_days,
                "training_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to train risk models: {str(e)}"
        )

@router.post("/portfolio/{portfolio_id}/rebalance/recommend")
async def get_rebalancing_recommendations(
    portfolio_id: int,
    objective: str = Query(
        "sharpe",
        regex="^(sharpe|min_variance|max_diversification)$"
    ),
    db: Session = Depends(get_db)
):
    """Get portfolio rebalancing recommendations"""
    try:
        # Get portfolio data
        portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
            
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        if not positions:
            raise HTTPException(status_code=404, detail="No positions found")
            
        # Prepare portfolio data
        portfolio_data = {}
        for position in positions:
            portfolio_data[position.symbol] = {
                "quantity": position.quantity,
                "current_price": position.current_price,
                "average_price": position.average_price
            }
            
        # Get recommendations
        recommendations = portfolio_rebalancer.generate_rebalancing_recommendations(
            portfolio_data,
            portfolio.cash,
            objective
        )
        
        # Format response
        return {
            "portfolio_id": portfolio_id,
            "recommendations": [
                {
                    "symbol": rec.symbol,
                    "current_weight": rec.current_weight,
                    "target_weight": rec.target_weight,
                    "action": rec.action,
                    "quantity_change": rec.quantity_change,
                    "expected_impact": rec.expected_impact
                }
                for rec in recommendations
            ],
            "metadata": {
                "objective": objective,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate rebalancing recommendations: {str(e)}"
        )
