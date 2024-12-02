# Factor Analysis Documentation

## Overview
The factor analysis module provides sophisticated tools for decomposing portfolio returns into their underlying statistical factors, analyzing factor contributions to risk and return, and visualizing factor relationships.

## Components

### 1. Portfolio Factor Analyzer (`PortfolioFactorAnalyzer`)
Core class that implements statistical factor analysis using Principal Component Analysis (PCA).

#### Key Methods:
- `extract_statistical_factors(returns)`: Extracts statistical risk factors from asset returns
- `decompose_returns(returns, weights)`: Decomposes portfolio returns into factor contributions
- `get_factor_summary(returns, weights)`: Generates comprehensive factor analysis metrics
- `analyze_factor_contribution(returns, weights)`: Analyzes factor contribution to portfolio risk

### 2. Visualization Service (`VisualizationService`)
Provides visualization tools for factor analysis results.

#### Key Visualizations:

##### Factor Analysis Dashboard
```python
viz.plot_factor_analysis(factor_summary, save_path='dashboard.png')
```
Generates a comprehensive dashboard with five key panels:
1. **Cumulative Factor Returns**: Shows how each factor's returns compound over time
2. **Factor Risk Contribution**: Displays each factor's contribution to total portfolio risk
3. **Factor-Asset Correlations**: Heatmap showing relationships between factors and assets
4. **Factor Performance Metrics**: Key statistics including returns, volatility, and Sharpe ratios
5. **Cumulative Variance Explained**: Shows how much portfolio variance each factor explains

##### Factor Contribution Over Time
```python
viz.plot_factor_contribution_over_time(factor_returns, save_path='contribution.png')
```
- Stacked area plot showing how different factors contribute to total portfolio returns
- Helps identify which factors drive returns in different market regimes

##### Rolling Factor Risk Metrics
```python
viz.plot_rolling_factor_risk(factor_returns, window=63, save_path='risk.png')
```
- Rolling volatility and Sharpe ratios for each factor
- Helps assess factor stability and risk-adjusted performance over time

## Usage Example

```python
from backend.services.factor_analysis import PortfolioFactorAnalyzer
from backend.services.visualization import VisualizationService

# Initialize services
analyzer = PortfolioFactorAnalyzer(n_factors=4)
viz = VisualizationService()

# Perform analysis
factor_loadings, components = analyzer.extract_statistical_factors(returns)
factor_returns = analyzer.decompose_returns(returns, weights)
factor_summary = analyzer.get_factor_summary(returns, weights)

# Generate visualizations
viz.plot_factor_analysis(factor_summary, save_path='analysis.png')
viz.plot_factor_contribution_over_time(factor_returns, save_path='contribution.png')
viz.plot_rolling_factor_risk(factor_returns, window=63, save_path='risk.png')
```

## Interpretation Guide

### Factor Returns
- **Positive Returns**: Factor contributes positively to portfolio performance
- **Negative Returns**: Factor detracts from portfolio performance
- **High Volatility**: Factor has significant impact on portfolio risk

### Risk Contribution
- Larger contributions indicate factors that drive portfolio risk
- Diversification across factors with low correlations reduces overall risk

### Factor-Asset Correlations
- **Strong Positive**: Asset moves closely with factor
- **Strong Negative**: Asset moves opposite to factor
- **Near Zero**: Asset independent of factor

### Performance Metrics
- **Sharpe Ratio**: Risk-adjusted return measure
- **Annualized Vol**: Factor volatility
- **Risk Contribution**: Proportion of portfolio risk

## Best Practices

1. **Number of Factors**
   - Start with 3-5 factors for most portfolios
   - Add more factors if significant variance remains unexplained

2. **Analysis Period**
   - Use at least 2 years of data for stable factor extraction
   - Consider multiple market regimes

3. **Interpretation**
   - Focus on persistent factors with economic interpretation
   - Monitor factor stability over time
   - Consider factor interactions and correlations

4. **Risk Management**
   - Use factor analysis to identify concentration risks
   - Balance exposure across uncorrelated factors
   - Monitor changes in factor relationships

## Next Steps

1. **Machine Learning Integration**
   - Implement factor prediction models
   - Add anomaly detection for factor behavior

2. **Advanced Risk Models**
   - Incorporate conditional factor models
   - Add regime-switching capabilities

3. **Real-time Analytics**
   - Add streaming factor analysis
   - Implement factor-based alerts
