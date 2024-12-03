# Stonks Development Guide

## Development Environment Setup

### Prerequisites
- Python 3.11
- Node.js 18+
- Git
- Docker (optional)
- VS Code (recommended)

### VS Code Extensions
- Python
- Pylance
- ESLint
- Prettier
- Docker
- GitLens
- REST Client

### Environment Setup

1. Clone and setup Python environment:
```bash
git clone https://github.com/yourusername/stonks.git
cd stonks
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Setup pre-commit hooks:
```bash
pre-commit install
```

3. Setup frontend development:
```bash
cd frontend
npm install
```

## Development Workflow

### Backend Development

#### Project Structure
```
backend/
├── models/          # SQLAlchemy models
├── routers/         # FastAPI route handlers
├── services/        # Business logic
│   ├── portfolio_optimization/  # Portfolio optimization strategies
│   │   ├── mean_variance.py
│   │   ├── risk_parity.py
│   │   ├── black_litterman.py
│   │   └── hierarchical.py
│   ├── risk_alerts/           # Risk monitoring system
│   │   ├── monitors/
│   │   ├── alerts/
│   │   └── websocket/
│   └── market_data/          # Market data services
├── analytics/       # Financial analysis modules
│   ├── indicators/  # Technical indicators
│   └── benchmarks/  # Market benchmark data
├── tests/          # Test suite
└── utils/          # Helper functions
```

#### Code Style
- Follow PEP 8
- Use type hints
- Document functions and classes
- Keep functions focused and small
- Use meaningful variable names

#### Testing
- Write tests for new features
- Maintain test coverage
- Use fixtures for common setup
- Mock external services

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=backend

# Run specific test file
pytest tests/test_portfolio.py
```

#### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

#### API Development Guidelines

1. Portfolio Optimization
   - Implement multiple optimization strategies
   - Support customizable constraints
   - Handle large datasets efficiently
   - Cache optimization results
   - Validate input parameters
   - Return detailed metadata

2. Risk Alert System
   - Monitor multiple risk metrics
   - Support real-time alerts
   - Implement severity levels
   - Handle WebSocket connections
   - Store alert history
   - Filter and query alerts

3. Portfolio Metrics Endpoint
   - Use query parameters for customization
   - Support multiple time ranges
   - Include technical indicators
   - Add benchmark comparisons
   - Return normalized data for charts

4. WebSocket Connections
   - Implement auto-reconnect
   - Handle connection errors
   - Manage subscription state
   - Monitor connection health

5. Technical Indicators
   - Follow standard calculation methods
   - Support customizable parameters
   - Return formatted chart data
   - Handle edge cases and errors

6. Benchmark Comparisons
   - Support major market indices
   - Normalize data for comparison
   - Cache frequently used data
   - Handle API rate limits

#### Portfolio Optimization Development

1. Mean-Variance Optimization
   - Implement efficient frontier calculation
   - Support risk and return constraints
   - Handle missing data
   - Optimize for large portfolios
   - Return detailed statistics

2. Risk Parity
   - Implement risk contribution calculation
   - Support risk budgeting
   - Handle correlation matrices
   - Optimize iteratively
   - Monitor convergence

3. Black-Litterman
   - Support market equilibrium
   - Handle investor views
   - Calculate posterior distribution
   - Implement confidence levels
   - Return detailed analytics

4. Hierarchical Risk Parity
   - Implement clustering
   - Handle correlation matrices
   - Support quasi-diagonalization
   - Optimize recursively
   - Return tree structure

#### Risk Alert System Development

1. Risk Monitors
   - Implement multiple monitoring strategies
   - Support real-time calculations
   - Handle large datasets
   - Cache intermediate results
   - Support custom thresholds

2. Alert Generation
   - Implement severity calculation
   - Support multiple alert types
   - Handle alert aggregation
   - Manage alert lifecycle
   - Store alert history

3. WebSocket Management
   - Handle multiple connections
   - Implement subscription system
   - Manage connection state
   - Handle reconnection
   - Monitor connection health

4. Alert Storage
   - Implement efficient storage
   - Support filtering and querying
   - Handle data retention
   - Manage alert lifecycle
   - Support analytics

#### Error Handling
- Use appropriate HTTP status codes
- Provide detailed error messages
- Handle WebSocket disconnections
- Validate input parameters
- Log errors for debugging

### Frontend Development

#### Project Structure
```
frontend/
├── src/
│   ├── components/    # React components
│   ├── hooks/        # Custom hooks
│   ├── pages/        # Page components
│   ├── redux/        # State management
│   ├── services/     # API clients
│   └── utils/        # Helper functions
└── public/          # Static assets
```

#### Code Style
- Use TypeScript
- Follow ESLint rules
- Use Prettier for formatting
- Use functional components
- Implement proper error boundaries
- Use proper prop types

#### State Management
- Use Redux Toolkit for global state
- Use React Query for server state
- Keep components pure
- Minimize prop drilling

#### Testing
```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run specific test file
npm test -- src/components/Portfolio.test.tsx
```

#### Chart Components
- Use TradingView Lightweight Charts
- Support multiple chart types
- Handle real-time updates
- Implement technical indicators
- Add benchmark overlays

#### WebSocket Integration
- Implement connection manager
- Handle reconnection logic
- Manage subscription state
- Update UI components

### API Development

#### Adding New Endpoints
1. Create route in appropriate router file
2. Add request/response models in schemas
3. Implement business logic in services
4. Add tests
5. Update API documentation

#### WebSocket Development
1. Implement handler in websocket.py
2. Add message schemas
3. Implement client-side handlers
4. Add reconnection logic
5. Test with multiple clients

### Database Development

#### Adding New Models
1. Create model in models/
2. Add relationships
3. Create migration
4. Add factory in tests
5. Update documentation

#### Performance Optimization
- Use appropriate indexes
- Optimize queries
- Use eager loading
- Implement caching
- Monitor query performance

### Security Considerations

#### Authentication
- Use JWT tokens
- Implement refresh tokens
- Secure password storage
- Rate limiting
- Input validation

#### Data Protection
- Sanitize user input
- Prevent SQL injection
- Use HTTPS
- Implement CORS
- Protect sensitive data

### Deployment

#### Local Development
```bash
# Backend
uvicorn main:app --reload --port 8000

# Frontend
npm start
```

#### Docker Development
```bash
# Build images
docker-compose build

# Start services
docker-compose up

# Stop services
docker-compose down
```

#### Production Deployment
1. Build production assets
2. Run security checks
3. Run tests
4. Build Docker images
5. Deploy to production
6. Monitor logs
7. Check metrics

### Monitoring & Debugging

#### Logging
- Use structured logging
- Include request ID
- Log appropriate levels
- Monitor error rates
- Set up log aggregation

#### Metrics
- Monitor API performance
- Track error rates
- Monitor database performance
- Track WebSocket connections
- Monitor system resources

### Contributing

#### Pull Request Process
1. Create feature branch
2. Write tests
3. Update documentation
4. Submit PR
5. Address review comments
6. Merge after approval

#### Code Review Guidelines
- Check code style
- Verify tests
- Review security implications
- Check performance impact
- Verify documentation

## Best Practices

### General
- Write clean, maintainable code
- Follow SOLID principles
- Use appropriate design patterns
- Keep dependencies updated
- Document technical decisions

### Backend
- Use async/await
- Implement proper error handling
- Use dependency injection
- Cache when appropriate
- Optimize database queries

### Frontend
- Use TypeScript
- Implement error boundaries
- Optimize renders
- Use proper state management
- Follow accessibility guidelines

### Testing
- Write unit tests
- Write integration tests
- Use test fixtures
- Mock external services
- Maintain test coverage

### Security
- Follow security best practices
- Keep dependencies updated
- Implement proper authentication
- Validate user input
- Protect sensitive data

## Troubleshooting

### Common Issues
1. Database connection issues
   - Check connection string
   - Verify database is running
   - Check permissions

2. WebSocket connection issues
   - Check connection URL
   - Verify token
   - Check firewall settings

3. Build issues
   - Clear node_modules
   - Update dependencies
   - Check Python version
   - Verify environment variables

### Getting Help
- Check documentation
- Search issue tracker
- Ask in team chat
- Create detailed bug report
- Include relevant logs

## Testing Strategy

### Market Data Services

The market data services implement a robust testing strategy with comprehensive coverage:

#### Market Data Adapter
- **Coverage**: 74% (as of latest update)
- **Test Suite**: `tests/test_market_data_adapter.py`
- **Key Test Areas**:
  - Lifecycle management (start/stop)
  - Symbol subscription handling
  - Rate limiting verification
  - Multiple provider support
  - Error handling and validation
  - Edge case scenarios

#### Alpha Vantage Provider
- **Coverage**: 36% (focused on core functionality)
- **Test Suite**: Integration with adapter tests
- **Features**:
  - Historical data retrieval
  - Real-time market data streaming
  - Rate limit compliance
  - Error handling

#### Future Testing Improvements
- [ ] Add streaming functionality tests
- [ ] Implement rate limit exceeded scenarios
- [ ] Add network timeout handling tests
- [ ] Test complex API interaction patterns
- [ ] Add performance benchmarking tests

### Best Practices

1. **Async Testing**
   - Use `pytest-asyncio` for async tests
   - Implement proper cleanup in `asyncTearDown`
   - Handle task cancellation with `asyncio.shield`
   - Set appropriate timeouts for async operations

2. **Mocking**
   - Use `AsyncMock` for async methods
   - Mock external API calls and sessions
   - Implement realistic error scenarios
   - Verify rate limiting behavior

3. **Error Handling**
   - Test both success and failure paths
   - Verify error propagation
   - Test timeout scenarios
   - Validate error messages

4. **Coverage**
   - Run with `pytest --cov`
   - Focus on critical code paths
   - Document uncovered scenarios
   - Track coverage trends

## API Documentation

See [API.md](API.md) for detailed API documentation.

## Performance Considerations

- Monitor rate limits for external APIs
- Use connection pooling where appropriate
- Implement caching strategies
- Profile critical code paths

## Security

- Store credentials in environment variables
- Use secure session management
- Implement proper error handling
- Avoid exposing sensitive data in logs

## Logging

- Use structured logging
- Include relevant context
- Implement appropriate log levels
- Monitor error rates

## Future Improvements

1. **Testing**
   - Expand integration test coverage
   - Add performance benchmarks
   - Implement stress testing
   - Add security testing

2. **Documentation**
   - Add architecture diagrams
   - Include sequence diagrams
   - Document error codes
   - Add troubleshooting guides

3. **Monitoring**
   - Add performance metrics
   - Implement health checks
   - Track error rates
   - Monitor API usage

4. **Development Tools**
   - Add pre-commit hooks
   - Implement automated code review
   - Add dependency scanning
   - Improve CI/CD pipeline
