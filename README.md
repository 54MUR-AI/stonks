# Stonks - Advanced Financial Market Analysis Platform

<p align="center">
  <img src="logo.png" alt="STONKS" width="400"/>
</p>

STONKS (Smart Trading Options for Novices and Knowledgeable Speculators) is a comprehensive web application for real-time market analysis, portfolio management, and AI-powered investment insights.

## Key Features

### Market Analysis
- Real-time market data with resilient provider switching:
  - Automatic failover to backup providers
  - Health-based provider selection
  - Performance tracking and monitoring
  - Exponential backoff with jitter
- Real-time market data visualization for stocks, crypto, and other securities
- Interactive charts with advanced technical indicators:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Bollinger Bands (BB)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
- Multi-timeframe analysis (1D, 1W, 1M, 3M, 6M, 1Y, ALL)
- AI-powered market insights and predictions
- Deep learning-based price predictions with confidence intervals
- Market regime detection and analysis
- Support and resistance level calculation
- Technical signal generation
- Fundamental analysis integration
- Market sentiment analysis
- Benchmark comparisons with major indices (SPY, QQQ, DIA, IWM)

### Portfolio Management
- Portfolio tracking and performance metrics
- Advanced Portfolio Optimization:
  - Mean-Variance Optimization
  - Risk Parity Optimization
  - Black-Litterman Optimization
  - Hierarchical Risk Parity
- Automated portfolio rebalancing
- Automated trading system:
  - Configurable trading parameters
  - Real-time portfolio monitoring
  - Stop loss and take profit management
  - Anomaly-based trading signals
  - Risk-aware execution
  - Liquidity analysis
- Comprehensive Risk Management:
  - Real-time risk alerts via WebSocket
  - Volatility monitoring
  - Drawdown tracking
  - VaR breach detection
  - Correlation analysis
  - Concentration risk monitoring
  - Market regime analysis
- Portfolio stress testing
- Maximum drawdown analysis
- Real-time portfolio value updates via WebSocket
- Position-level volatility tracking
- Customizable technical indicator overlays
- Benchmark performance comparison

### Social Features
- Portfolio sharing with granular permissions
- User following system
- Portfolio comments and discussions
- Public/private portfolio visibility

### Notifications & Alerts
- Real-time risk alerts with severity levels (low, medium, high, critical)
- Portfolio performance notifications
- Market regime change alerts
- Risk threshold breach notifications
- Daily portfolio summaries via email
- Activity feed with social interactions
- Customizable notification preferences

### Alert Analytics
- Pattern recognition
- Root cause probability analysis
- Machine learning-based anomaly detection
- Predictive alerting
- Analytics dashboard

### Incident Management
- Multi-level escalation (L1-Emergency)
- Automated response actions
- Configurable escalation policies
- Time-based auto-escalation
- Multi-channel notifications

### Notification System
- Multi-channel support (Email, Slack, SMS)
- Configurable notification rules
- Provider-specific settings
- Severity-based routing
- Notification tracking

## Market Data Providers 

The system supports multiple market data providers with intelligent failover, health monitoring, and comprehensive management capabilities:

### Features
- Advanced Provider Management:
  * Real-time health monitoring dashboard
  * Automated provider failover based on health metrics
  * Predictive health analysis with anomaly detection
  * Comprehensive provider management console
  * Priority-based provider selection
- Resilient Error Handling:
  * Circuit breaker pattern implementation
  * Intelligent retry logic with backoff
  * Rate limiting and backpressure handling
  * Comprehensive error tracking
- Performance Optimization:
  * Real-time metrics collection and analysis
  * Latency monitoring and optimization
  * Cache management with TTL support
  * Memory-aware operation
- Health Monitoring:
  * Real-time health status tracking
  * Metric history and trend analysis
  * Anomaly detection and alerting
  * Performance forecasting
  * Event timeline tracking

### Supported Providers
- Alpha Vantage (Production)
- Yahoo Finance (Production)
- Mock Provider (Testing/Development)

### Management Console
The provider management console offers:
- Real-time provider status monitoring
- Add/Edit/Delete provider functionality
- Start/Stop controls
- Priority management
- Health status visualization
- Configuration management
- Performance analytics

### Configuration
Market data providers can be configured using the `MarketDataConfig` class:

```python
config = MarketDataConfig(
    credentials=MarketDataCredentials(api_key="your_key"),
    base_url="provider_url",
    websocket_url="ws_url",
    request_timeout=1,  # seconds
    max_retries=3
)
```

### Usage Example
```python
# Initialize provider
provider = MockProvider(config)
await provider.connect()

# Subscribe to symbols
await provider.subscribe(["AAPL", "MSFT"])

# Get historical data
df = await provider.get_historical_data(
    symbol="AAPL",
    start_date=datetime.now() - timedelta(days=7),
    interval="1h"
)

# Get real-time quotes
quote = await provider.get_quote("AAPL")

# Get symbol statistics
stats = await provider.get_symbol_stats("AAPL")
```

## Tech Stack

### Backend
- Language: Python 3.11
- Framework: FastAPI
- Database: SQLAlchemy with SQLite
- Authentication: JWT-based with python-jose
- WebSocket support for real-time data
- Email integration with fastapi-mail
- Machine Learning:
  - TensorFlow/Keras for deep learning
  - scikit-learn for ensemble models
  - statsmodels for statistical analysis
  - arch for volatility modeling
- Trading System:
  - Asynchronous execution engine
  - Risk-aware order management
  - Real-time portfolio monitoring
  - Automated rebalancing

### Frontend
- Framework: React
- State Management: Redux Toolkit
- UI Components: Material-UI
- Charts: TradingView Lightweight Charts
- Technical Analysis: technicalindicators library

### Data Sources & APIs
- Yahoo Finance (yfinance)
- Market data providers
- Financial analysis APIs

### DevOps & CI/CD
- GitHub Actions for automated testing and deployment
- Docker containerization
- Code quality tools (flake8, black, isort)
- Security scanning (bandit, safety)

## Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stonks.git
cd stonks
```

2. Install Python dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Install Node.js dependencies:
```bash
cd frontend
npm install
```

4. Set up environment variables:
Create a `.env` file in the `backend` directory:
```env
DATABASE_URL=sqlite:///./stonks.db
SECRET_KEY=your_secret_key
ACCESS_TOKEN_EXPIRE_MINUTES=30
MAIL_USERNAME=your_email
MAIL_PASSWORD=your_password
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
```

5. Initialize the database:
```bash
cd backend
python init_db.py
```

6. Run the development servers:

Backend:
```bash
uvicorn main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm start
```

## Environment Configuration

The application uses environment variables for configuration. To get started:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Update the `.env` file with your configuration:
   - `STONKS_HOST`: Server host (default: 127.0.0.1)
   - `STONKS_PORT`: Server port (default: 8000)
   - `STONKS_ENV`: Environment (development/production)
   - `SECRET_KEY`: JWT secret key (change this in production!)
   - `DATABASE_URL`: Database connection string
   - API keys for external services

3. Security Notes:
   - Never commit `.env` file to version control
   - In production, use a secure secret key
   - When deploying, ensure proper network security is configured
   - Default configuration binds to localhost for security

## Project Structure

```
stonks/
├── backend/
│   ├── models/            # Database models
│   ├── routers/          # API route handlers
│   ├── services/         # Business logic
│   ├── tests/           # Test suite
│   └── utils/           # Helper functions
├── frontend/
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── redux/      # State management
│   │   ├── services/   # API clients
│   │   └── utils/      # Helper functions
│   └── public/         # Static assets
└── docs/              # Documentation
```

## API Documentation

The API documentation is available at `/docs` or `/redoc` when running the backend server.

## API Endpoints

### Portfolio Metrics
```
GET /api/portfolios/{portfolio_id}/metrics

Query Parameters:
- time_range: Time period for analysis (1D|1W|1M|3M|6M|1Y|ALL)
- indicators: Technical indicators to include (SMA|EMA|BB|RSI|MACD)
- benchmark: Benchmark index for comparison (SPY|QQQ|DIA|IWM)

Response includes:
- Current portfolio value and changes
- Historical value data
- Risk metrics (volatility, Sharpe ratio)
- Position-level metrics
- Technical indicators
- Benchmark comparison data
```

### Portfolio Management & Risk
```
POST /api/portfolio/{portfolio_id}/optimize
- Optimize portfolio allocation using various strategies
- Supports mean-variance, risk parity, Black-Litterman, and hierarchical optimization
- Customizable constraints and targets

WebSocket /ws/risk-alerts/{portfolio_id}
- Real-time risk alerts for portfolio monitoring
- Supports multiple alert types and severity levels

GET /api/portfolio/{portfolio_id}/risk-alerts/history
- Historical risk alerts with filtering capabilities
- Filter by date range, severity, and alert type
```

## Testing

Run backend tests:
```bash
cd backend
pytest
```

Run frontend tests:
```bash
cd frontend
npm test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security

- JWT-based authentication
- Password hashing with bcrypt
- WebSocket authentication
- Environment variable management
- Input validation
- Permission-based access control

## License

[MIT License](LICENSE)

## Acknowledgments

- TradingView for the Lightweight Charts library
- FastAPI for the excellent framework
- Material-UI for the component library

## Next Steps

### Frontend Development
- Implement portfolio visualization dashboard
- Add interactive charts for portfolio analysis
- Create user profile pages

### Backend Enhancements
- Add real-time market data streaming
- Implement portfolio rebalancing suggestions
- Add machine learning-based market insights

### Performance Optimization
- Implement caching for market data
- Optimize database queries
- Add background task processing

### Security Enhancements
- Add 2FA support
- Implement rate limiting
- Enhanced input validation

### Documentation
- Complete API documentation
- Add user guides
- Create developer documentation

## Advanced Features

### Portfolio Optimization
The platform offers multiple portfolio optimization strategies:
- Mean-Variance Optimization: Maximize returns for given risk level
- Risk Parity: Equal risk contribution from all assets
- Black-Litterman: Incorporate market views into optimization
- Hierarchical Risk Parity: Tree-based portfolio optimization

### Risk Management System
Real-time risk monitoring and alerts:
- Volatility tracking and threshold alerts
- Drawdown monitoring with configurable limits
- Value at Risk (VaR) breach detection
- Correlation analysis and diversification alerts
- Position concentration monitoring
- Market regime change detection
- Stress testing scenarios

## Testing Improvements

The testing suite has been significantly improved with:

- Enhanced test coverage for market data services
- Robust async operation handling
- Comprehensive error management
- Multiple provider support

Current test coverage:
- Market Data Adapter: 74%
- Alpha Vantage Provider: 36%
- Base Provider Interface: 81%
- Mock Provider: 66%

## Project Status

The project is currently in the development phase, with a focus on:

- Implementing real-time market data streaming
- Enhancing portfolio optimization and risk management features
- Improving performance and security

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Alpha Vantage](https://www.alphavantage.co/) for market data
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://reactjs.org/) for the frontend framework

## Support

For support:
1. Check the [documentation](docs/)
2. Create an issue
3. Contact the development team

## Screenshots

![Portfolio Performance](portfolio_performance.png)
![Efficient Frontier](efficient_frontier.png)
![Trade Analysis](trade_analysis.png)
![Drawdown Analysis](drawdown.png)
