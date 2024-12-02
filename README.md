# Stonks - Advanced Financial Market Analysis Platform

<p align="center">
  <img src="logo.png" alt="STONKS" width="400"/>
</p>

A comprehensive web application for real-time market analysis, portfolio management, and AI-powered investment insights.

## Key Features

### Market Analysis
- Real-time market data visualization for stocks, crypto, and other securities
- Interactive charts with advanced technical indicators:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Bollinger Bands (BB)
  - Relative Strength Index (RSI)
  - MACD (Moving Average Convergence Divergence)
- Multi-timeframe analysis (1D, 1W, 1M, 3M, 6M, 1Y, ALL)
- AI-powered market insights and predictions
- Benchmark comparisons with major indices (SPY, QQQ, DIA, IWM)

### Portfolio Management
- Portfolio tracking and performance metrics
- Modern Portfolio Theory optimization
- Automated portfolio rebalancing
- Risk analysis and correlation matrices
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
- Real-time price alerts
- Portfolio performance notifications
- Daily portfolio summaries via email
- Activity feed with social interactions
- Customizable notification preferences

## Tech Stack

### Backend
- Language: Python 3.11
- Framework: FastAPI
- Database: SQLAlchemy with SQLite
- Authentication: JWT-based with python-jose
- WebSocket support for real-time data
- Email integration with fastapi-mail

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
