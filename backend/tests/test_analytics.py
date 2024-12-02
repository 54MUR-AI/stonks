import pytest
from models import Portfolio, Position
from analytics import calculate_portfolio_metrics, calculate_correlation_matrix
from portfolio_rebalancer import PortfolioRebalancer, optimize_portfolio_weights
from datetime import datetime, timedelta

@pytest.fixture
def sample_portfolio(db, test_user):
    portfolio = Portfolio(
        name="Sample Portfolio",
        owner_id=test_user.id
    )
    db.add(portfolio)
    db.commit()
    
    positions = [
        Position(
            portfolio_id=portfolio.id,
            symbol="AAPL",
            quantity=10,
            average_price=150.00
        ),
        Position(
            portfolio_id=portfolio.id,
            symbol="MSFT",
            quantity=8,
            average_price=300.00
        ),
        Position(
            portfolio_id=portfolio.id,
            symbol="GOOGL",
            quantity=5,
            average_price=2500.00
        )
    ]
    db.add_all(positions)
    db.commit()
    
    for pos in positions:
        db.refresh(pos)
    
    db.refresh(portfolio)
    return portfolio, positions

def test_portfolio_metrics(sample_portfolio):
    portfolio, positions = sample_portfolio
    metrics = calculate_portfolio_metrics(
        positions,
        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    assert metrics is not None
    assert "total_value" in metrics
    assert "unrealized_pl" in metrics
    assert "daily_return" in metrics
    assert "volatility" in metrics
    assert "sharpe_ratio" in metrics
    assert "beta" in metrics
    assert "positions" in metrics
    assert len(metrics["positions"]) == 3

def test_correlation_matrix(sample_portfolio):
    portfolio, positions = sample_portfolio
    correlation = calculate_correlation_matrix(
        positions,
        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    assert correlation is not None
    assert "AAPL" in correlation
    assert "MSFT" in correlation
    assert "GOOGL" in correlation
    
    # Check correlation properties
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        assert correlation[symbol][symbol] == 1.0  # Self-correlation
        for other in ["AAPL", "MSFT", "GOOGL"]:
            assert -1.0 <= correlation[symbol][other] <= 1.0

def test_portfolio_rebalancer(sample_portfolio):
    portfolio, positions = sample_portfolio
    target_weights = {
        "AAPL": 0.4,
        "MSFT": 0.35,
        "GOOGL": 0.25
    }
    
    rebalancer = PortfolioRebalancer(positions, target_weights)
    summary = rebalancer.get_rebalancing_summary()
    
    assert summary is not None
    assert "total_portfolio_value" in summary
    assert "total_rebalancing_value" in summary
    assert "rebalancing_trades" in summary
    assert "current_weights" in summary
    assert "target_weights" in summary
    
    # Check weights sum to 1
    current_sum = sum(summary["current_weights"].values())
    target_sum = sum(summary["target_weights"].values())
    assert abs(current_sum - 1.0) < 0.01
    assert abs(target_sum - 1.0) < 0.01

def test_portfolio_optimization(sample_portfolio):
    portfolio, positions = sample_portfolio
    optimal_weights = optimize_portfolio_weights(
        positions,
        risk_tolerance=0.5,
        min_weight=0.1,
        max_weight=0.5
    )
    
    assert optimal_weights is not None
    assert len(optimal_weights) == 3
    
    # Check constraints
    for weight in optimal_weights.values():
        assert 0.1 <= weight <= 0.5
    
    # Check sum of weights
    total_weight = sum(optimal_weights.values())
    assert abs(total_weight - 1.0) < 0.01
