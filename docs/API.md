# Stonks API Documentation

## Authentication

### JWT Authentication
All API endpoints (except `/auth/token`) require a valid JWT token in the Authorization header:
```
Authorization: Bearer <token>
```

### Endpoints

#### POST /auth/token
Login to obtain access token.
```json
{
    "username": "string",
    "password": "string"
}
```

## Market Data

### GET /market/quote/{symbol}
Get real-time quote for a symbol.

### GET /market/history/{symbol}
Get historical data for a symbol.
Parameters:
- interval: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- start: ISO date
- end: ISO date

### WebSocket /ws/market
Real-time market data stream.
Authentication required via token query parameter.

## Portfolio Management

### GET /portfolios
List user portfolios.

### POST /portfolios
Create new portfolio.
```json
{
    "name": "string",
    "description": "string",
    "is_public": boolean
}
```

### GET /portfolios/{portfolio_id}
Get portfolio details.

### PUT /portfolios/{portfolio_id}
Update portfolio.

### DELETE /portfolios/{portfolio_id}
Delete portfolio.

### POST /portfolios/{portfolio_id}/positions
Add position to portfolio.
```json
{
    "symbol": "string",
    "quantity": number,
    "entry_price": number
}
```

### GET /portfolios/{portfolio_id}/analytics
Get portfolio analytics.
Returns:
- Performance metrics
- Risk metrics
- Correlation matrix
- Maximum drawdown

### POST /portfolios/{portfolio_id}/rebalance
Get portfolio rebalancing recommendations.
```json
{
    "target_weights": {
        "symbol": weight
    },
    "tolerance": number
}
```

### GET /portfolios/{portfolio_id}/optimize
Get optimal portfolio weights.
Parameters:
- risk_tolerance: number
- min_weight: number
- max_weight: number

### POST /portfolios/{portfolio_id}/optimize
Optimize portfolio allocation using various strategies.
```json
{
    "optimization_type": "mean_variance" | "risk_parity" | "black_litterman" | "hierarchical",
    "target_return": number,  // Optional
    "target_risk": number,    // Optional
    "constraints": {          // Optional
        "min_weights": {
            "symbol": number
        },
        "max_weights": {
            "symbol": number
        },
        "sector_constraints": {
            "sector": {
                "min": number,
                "max": number
            }
        }
    }
}

Response:
{
    "portfolio_id": number,
    "optimization_type": string,
    "weights": {
        "symbol": number
    },
    "expected_return": number,
    "volatility": number,
    "sharpe_ratio": number,
    "metadata": {
        // Strategy-specific metrics
    }
}

### WebSocket /ws/risk-alerts/{portfolio_id}
Real-time risk alerts for portfolio monitoring.
```json
// Server -> Client Message
{
    "type": "risk_alerts",
    "alerts": [
        {
            "type": "volatility" | "drawdown" | "var_breach" | "correlation" | "liquidity" | "concentration" | "regime_change" | "market_stress",
            "timestamp": "ISO datetime",
            "severity": "low" | "medium" | "high" | "critical",
            "message": "string",
            "metadata": {
                // Alert-specific data
            }
        }
    ]
}
```

### GET /portfolios/{portfolio_id}/risk-alerts/history
Get historical risk alerts with filtering.
Parameters:
- start_date: ISO datetime
- end_date: ISO datetime
- severity: "low" | "medium" | "high" | "critical"
- alert_type: string

Response:
```json
{
    "alerts": [
        {
            "type": string,
            "timestamp": "ISO datetime",
            "severity": string,
            "message": string,
            "metadata": {
                // Alert-specific data
            }
        }
    ]
}
```

### POST /portfolios/{portfolio_id}/share
Share portfolio with users.
```json
{
    "user_ids": [number],
    "permission": "view" | "comment"
}
```

### POST /users/{user_id}/follow
Follow a user.

### POST /portfolios/{portfolio_id}/comments
Add comment to portfolio.
```json
{
    "content": "string"
}
```

## Stress Testing

### GET /portfolio/{portfolio_id}/stress-test/scenarios
Get available stress test scenarios.
Parameters:
- scenario_type: Optional[str] - Filter scenarios by type (historical, hypothetical, monte_carlo, sensitivity, regime_change)

Response:
```json
{
    "scenarios": [
        {
            "name": "string",
            "type": "historical" | "hypothetical" | "monte_carlo" | "sensitivity" | "regime_change",
            "description": "string",
            "probability": "number"  // Only for Monte Carlo scenarios
        }
    ]
}
```

### POST /portfolio/{portfolio_id}/stress-test/run
Run stress test on portfolio.
Parameters:
- scenario_name: string - Name of the scenario to run
- scenario_type: string - Type of scenario (historical, hypothetical, monte_carlo, sensitivity, regime_change)
- custom_scenario: Optional object - Custom scenario definition for hypothetical scenarios

Request body for custom scenario:
```json
{
    "description": "string",
    "shocks": {
        "symbol": number  // Asset price shock magnitude
    },
    "correlation_adjustments": {  // Optional
        "symbol_pair": number
    },
    "volatility_adjustments": {   // Optional
        "symbol": number
    }
}
```

Response:
```json
{
    "scenario_name": "string",
    "portfolio_impact": number,    // Percentage change in portfolio value
    "asset_impacts": {
        "symbol": number          // Individual asset impacts
    },
    "risk_metrics": {
        "stressed_var_95": number,
        "stressed_sharpe": number,
        "portfolio_volatility": number
    },
    "correlation_changes": {      // Optional
        "symbol_pair": number
    },
    "volatility_changes": {       // Optional
        "symbol": number
    }
}
```

## Social Features

### POST /portfolios/{portfolio_id}/share
Share portfolio with users.
```json
{
    "user_ids": [number],
    "permission": "view" | "comment"
}
```

### POST /users/{user_id}/follow
Follow a user.

### POST /portfolios/{portfolio_id}/comments
Add comment to portfolio.
```json
{
    "content": "string"
}
```

## Activity & Notifications

### GET /activities/feed
Get activity feed.
Parameters:
- skip: number
- limit: number

### GET /activities/user/{user_id}
Get user activities.
Parameters:
- skip: number
- limit: number

### GET /notifications
Get user notifications.
Parameters:
- unread_only: boolean
- skip: number
- limit: number

### POST /notifications/{notification_id}/read
Mark notification as read.

### POST /notifications/read-all
Mark all notifications as read.

### PUT /users/notification-preferences
Update notification preferences.
```json
{
    "email": boolean,
    "web": boolean,
    "price_alerts": boolean,
    "portfolio_updates": boolean,
    "social_notifications": boolean
}
```

## Alerts

### POST /alerts
Create new price alert.
```json
{
    "symbol": "string",
    "price": number,
    "condition": "above" | "below"
}
```

### GET /alerts
List user alerts.

### DELETE /alerts/{alert_id}
Delete alert.

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
Invalid input data.
```json
{
    "detail": "Error message"
}
```

### 401 Unauthorized
Invalid or missing authentication.
```json
{
    "detail": "Could not validate credentials"
}
```

### 403 Forbidden
Insufficient permissions.
```json
{
    "detail": "Not authorized to access this resource"
}
```

### 404 Not Found
Resource not found.
```json
{
    "detail": "Resource not found"
}
```

### 500 Internal Server Error
Server error.
```json
{
    "detail": "Internal server error"
}
```

## Rate Limiting

API requests are limited to:
- 100 requests per minute for authenticated users
- 20 requests per minute for unauthenticated users

## WebSocket Events

### Market Data
```json
{
    "type": "market_update",
    "data": {
        "symbol": "string",
        "price": number,
        "volume": number,
        "timestamp": string
    }
}
```

### Portfolio Updates
```json
{
    "type": "portfolio_update",
    "data": {
        "portfolio_id": number,
        "type": "position_update" | "performance_update",
        "data": object
    }
}
```

### Notifications
```json
{
    "type": "notification",
    "data": {
        "id": number,
        "type": string,
        "message": string,
        "data": object
    }
}
```
