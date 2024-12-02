# STONKS Platform Demo Guide

## Overview
This demo showcases the key features of the STONKS platform, including portfolio management, ML-powered predictions, automated trading, and comprehensive risk management.

## Prerequisites
- Python 3.11
- Required packages installed (`pip install -r requirements.txt`)
- Node.js and npm for frontend
- Demo data generated

## Setup Instructions

1. Generate Demo Data
```bash
python scripts/demo_data_generator.py
```

2. Start Backend Server
```bash
cd backend
uvicorn main:app --reload
```

3. Start Frontend (in new terminal)
```bash
cd frontend
npm start
```

4. Run Demo Script
```bash
python scripts/run_demo.py
```

## Demo Flow

### 1. Portfolio Overview
- Displays demo portfolio with optimized allocations
- Shows current positions and values
- Demonstrates real-time updates

### 2. ML Predictions
- Presents price predictions with confidence intervals
- Shows market regime analysis
- Displays technical signals and sentiment analysis
- Demonstrates multi-horizon forecasts

### 3. Automated Trading
- Shows trading system configuration
- Demonstrates real-time monitoring
- Displays risk-aware execution
- Shows position management (stop loss/take profit)

### 4. Risk Management
- Presents comprehensive risk metrics
- Shows real-time risk monitoring
- Displays detected market anomalies
- Demonstrates stress testing results

## Key Features Demonstrated

### Portfolio Management
- Real-time portfolio tracking
- Position-level analytics
- Performance metrics
- Automated rebalancing

### ML/AI Capabilities
- Deep learning price predictions
- Market regime detection
- Anomaly detection
- Sentiment analysis

### Trading Automation
- Configurable trading parameters
- Risk-aware execution
- Real-time monitoring
- Portfolio rebalancing

### Risk Management
- Advanced risk metrics
- Real-time alerts
- Stress testing
- Portfolio monitoring

## Demo Data
The demo uses a curated set of data for 10 major tech stocks:
- AAPL, MSFT, GOOGL, AMZN, NVDA
- META, TSLA, JPM, V, PYPL

Data includes:
- 1 year of historical prices
- Optimized portfolio allocations
- ML predictions and confidence intervals
- Risk metrics and anomaly detections

## Notes
- The demo uses simulated real-time data for demonstration
- Trading execution is simulated for demo purposes
- Risk metrics are calculated using historical data
- ML models are pre-trained for demo speed

## Troubleshooting

### Common Issues
1. Data Generation Fails
   - Ensure internet connection for Yahoo Finance data
   - Check Python package versions
   - Verify file permissions

2. Demo Script Errors
   - Ensure backend server is running
   - Check all required services are initialized
   - Verify demo data files exist

3. Frontend Issues
   - Check npm dependencies are installed
   - Verify WebSocket connection
   - Check browser console for errors

### Support
For technical support or questions during the demo:
- Check the logs in `backend/logs`
- Refer to API documentation at `/docs`
- Contact the development team

## Next Steps
After the demo, you can:
1. Explore the codebase
2. Customize trading parameters
3. Add new ML models
4. Extend risk metrics
5. Enhance visualization
