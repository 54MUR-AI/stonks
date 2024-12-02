import pytest
from models import Portfolio, Position
from decimal import Decimal

@pytest.fixture
def test_portfolio(db, test_user):
    portfolio = Portfolio(
        name="Test Portfolio",
        owner_id=test_user.id
    )
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio

@pytest.fixture
def test_positions(db, test_portfolio):
    positions = [
        Position(
            portfolio_id=test_portfolio.id,
            symbol="AAPL",
            quantity=10,
            average_price=150.00
        ),
        Position(
            portfolio_id=test_portfolio.id,
            symbol="GOOGL",
            quantity=5,
            average_price=2500.00
        )
    ]
    db.add_all(positions)
    db.commit()
    for pos in positions:
        db.refresh(pos)
    return positions

def test_create_portfolio(authorized_client):
    response = authorized_client.post(
        "/portfolios/",
        json={"name": "New Portfolio"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "New Portfolio"

def test_get_portfolios(authorized_client, test_portfolio):
    response = authorized_client.get("/portfolios/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == test_portfolio.name

def test_get_portfolio_analytics(authorized_client, test_portfolio, test_positions):
    response = authorized_client.get(f"/portfolios/{test_portfolio.id}/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_value" in data
    assert "positions" in data
    assert len(data["positions"]) == 2

def test_portfolio_rebalance(authorized_client, test_portfolio, test_positions):
    response = authorized_client.post(
        f"/portfolios/{test_portfolio.id}/rebalance",
        json={
            "weights": {
                "AAPL": 0.6,
                "GOOGL": 0.4
            },
            "tolerance": 0.05
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "rebalancing_trades" in data
    assert "current_weights" in data
    assert "target_weights" in data

def test_portfolio_optimize(authorized_client, test_portfolio, test_positions):
    response = authorized_client.get(
        f"/portfolios/{test_portfolio.id}/optimize",
        params={"risk_tolerance": 0.5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "optimal_weights" in data
    weights_sum = sum(data["optimal_weights"].values())
    assert abs(weights_sum - 1.0) < 0.01  # Sum should be approximately 1

def test_portfolio_correlation(authorized_client, test_portfolio, test_positions):
    response = authorized_client.get(f"/portfolios/{test_portfolio.id}/correlation")
    assert response.status_code == 200
    data = response.json()
    assert "AAPL" in data
    assert "GOOGL" in data
