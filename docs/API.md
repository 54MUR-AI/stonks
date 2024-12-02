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
