from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from ..market_data.provider import MarketDataProvider

class RiskMetrics:
    def __init__(
        self,
        market_data_provider: MarketDataProvider,
        lookback_period: int = 252,  # 1 year of trading days
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
    ):
        self.market_data = market_data_provider
        self.lookback_period = lookback_period
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self._cache = {}

    async def _get_returns_data(
        self,
        symbols: List[str],
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get historical returns data for the given symbols"""
        cache_key = (tuple(symbols), end_date)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=int(self.lookback_period * 1.5))

        # Get historical prices
        prices = await self.market_data.get_historical_prices(
            symbols,
            start_date,
            end_date,
            interval='1d'
        )

        # Calculate returns
        returns = prices.pct_change().dropna()
        returns = returns.tail(self.lookback_period)

        self._cache[cache_key] = returns
        return returns

    async def calculate_volatility(
        self,
        symbols: List[str],
        weights: Optional[Dict[str, float]] = None,
        annualize: bool = True
    ) -> float:
        """Calculate portfolio volatility"""
        returns = await self._get_returns_data(symbols)
        
        if weights is None:
            weights = {symbol: 1/len(symbols) for symbol in symbols}
            
        portfolio_weights = np.array([weights[symbol] for symbol in symbols])
        cov_matrix = returns.cov()
        
        portfolio_variance = portfolio_weights.T @ cov_matrix @ portfolio_weights
        volatility = np.sqrt(portfolio_variance)
        
        if annualize:
            volatility *= np.sqrt(252)  # Annualize daily volatility
            
        return float(volatility)

    async def calculate_var(
        self,
        symbols: List[str],
        weights: Optional[Dict[str, float]] = None,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk using different methods:
        - historical: historical simulation
        - parametric: parametric VaR assuming normal distribution
        - modified: modified VaR using Cornish-Fisher expansion
        """
        returns = await self._get_returns_data(symbols)
        
        if weights is None:
            weights = {symbol: 1/len(symbols) for symbol in symbols}
            
        portfolio_weights = np.array([weights[symbol] for symbol in symbols])
        portfolio_returns = returns @ portfolio_weights

        if method == 'historical':
            var = float(portfolio_returns.quantile(1 - self.confidence_level))
        elif method == 'parametric':
            z_score = stats.norm.ppf(1 - self.confidence_level)
            var = float(portfolio_returns.mean() + z_score * portfolio_returns.std())
        elif method == 'modified':
            z_score = stats.norm.ppf(1 - self.confidence_level)
            skew = stats.skew(portfolio_returns)
            kurt = stats.kurtosis(portfolio_returns)
            
            # Cornish-Fisher expansion
            cf_var = z_score + \
                (z_score**2 - 1) * skew / 6 + \
                (z_score**3 - 3*z_score) * (kurt - 3) / 24 - \
                (2*z_score**3 - 5*z_score) * skew**2 / 36
                
            var = float(portfolio_returns.mean() + cf_var * portfolio_returns.std())
        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return -var  # Return positive VaR number

    async def calculate_es(
        self,
        symbols: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        returns = await self._get_returns_data(symbols)
        
        if weights is None:
            weights = {symbol: 1/len(symbols) for symbol in symbols}
            
        portfolio_weights = np.array([weights[symbol] for symbol in symbols])
        portfolio_returns = returns @ portfolio_weights
        
        var_cutoff = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        es = portfolio_returns[portfolio_returns <= var_cutoff].mean()
        
        return float(-es)  # Return positive ES number

    async def calculate_beta(
        self,
        symbols: List[str],
        weights: Optional[Dict[str, float]] = None,
        benchmark: str = 'SPY'
    ) -> float:
        """Calculate portfolio beta relative to a benchmark"""
        all_symbols = symbols + [benchmark]
        returns = await self._get_returns_data(all_symbols)
        
        if weights is None:
            weights = {symbol: 1/len(symbols) for symbol in symbols}
            
        portfolio_weights = np.array([weights[symbol] for symbol in symbols])
        portfolio_returns = returns[symbols] @ portfolio_weights
        benchmark_returns = returns[benchmark]
        
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        beta = covariance / benchmark_variance
        return float(beta)

    async def calculate_sharpe_ratio(
        self,
        symbols: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate the Sharpe ratio"""
        returns = await self._get_returns_data(symbols)
        
        if weights is None:
            weights = {symbol: 1/len(symbols) for symbol in symbols}
            
        portfolio_weights = np.array([weights[symbol] for symbol in symbols])
        portfolio_returns = returns @ portfolio_weights
        
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        return float(sharpe)

    async def calculate_sortino_ratio(
        self,
        symbols: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate the Sortino ratio"""
        returns = await self._get_returns_data(symbols)
        
        if weights is None:
            weights = {symbol: 1/len(symbols) for symbol in symbols}
            
        portfolio_weights = np.array([weights[symbol] for symbol in symbols])
        portfolio_returns = returns @ portfolio_weights
        
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))
        
        sortino = np.sqrt(252) * excess_returns.mean() / downside_std
        return float(sortino)

    async def calculate_tracking_error(
        self,
        positions: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> float:
        """Calculate tracking error between current and target portfolio"""
        symbols = list(set(positions.keys()) | set(target_weights.keys()))
        returns = await self._get_returns_data(symbols)
        
        # Calculate current weights
        total_value = sum(abs(pos) for pos in positions.values())
        current_weights = {
            symbol: positions.get(symbol, 0) / total_value
            for symbol in symbols
        }
        
        # Ensure target weights are normalized
        target_sum = sum(abs(w) for w in target_weights.values())
        normalized_targets = {
            symbol: target_weights.get(symbol, 0) / target_sum
            for symbol in symbols
        }
        
        # Calculate return differences
        current_returns = returns @ np.array([current_weights[s] for s in symbols])
        target_returns = returns @ np.array([normalized_targets[s] for s in symbols])
        return_diff = current_returns - target_returns
        
        tracking_error = np.std(return_diff) * np.sqrt(252)
        return float(tracking_error)

    async def calculate_risk_contribution(
        self,
        symbols: List[str],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate risk contribution of each asset"""
        returns = await self._get_returns_data(symbols)
        
        portfolio_weights = np.array([weights[symbol] for symbol in symbols])
        cov_matrix = returns.cov()
        
        portfolio_volatility = np.sqrt(
            portfolio_weights.T @ cov_matrix @ portfolio_weights
        )
        
        # Calculate marginal risk contribution
        mrc = cov_matrix @ portfolio_weights
        
        # Calculate component risk contribution
        crc = portfolio_weights * mrc / portfolio_volatility
        
        return {symbol: float(rc) for symbol, rc in zip(symbols, crc)}

    async def calculate_risk_metrics(
        self,
        positions: Dict[str, float],
        include_stress_tests: bool = True
    ) -> Dict:
        """Calculate comprehensive risk metrics for a portfolio"""
        symbols = list(positions.keys())
        total_value = sum(abs(pos) for pos in positions.values())
        weights = {
            symbol: positions[symbol] / total_value
            for symbol in symbols
        }
        
        # Basic risk metrics
        volatility = await self.calculate_volatility(symbols, weights)
        var_hist = await self.calculate_var(symbols, weights, method='historical')
        var_param = await self.calculate_var(symbols, weights, method='parametric')
        var_mod = await self.calculate_var(symbols, weights, method='modified')
        es = await self.calculate_es(symbols, weights)
        beta = await self.calculate_beta(symbols, weights)
        sharpe = await self.calculate_sharpe_ratio(symbols, weights)
        sortino = await self.calculate_sortino_ratio(symbols, weights)
        risk_contrib = await self.calculate_risk_contribution(symbols, weights)
        
        metrics = {
            'volatility': volatility,
            'value_at_risk': {
                'historical': var_hist,
                'parametric': var_param,
                'modified': var_mod,
            },
            'expected_shortfall': es,
            'beta': beta,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'risk_contribution': risk_contrib,
        }
        
        if include_stress_tests:
            # Add stress test scenarios
            metrics['stress_tests'] = await self._run_stress_tests(
                symbols,
                weights
            )
            
        return metrics

    async def _run_stress_tests(
        self,
        symbols: List[str],
        weights: Dict[str, float]
    ) -> Dict:
        """Run various stress test scenarios"""
        returns = await self._get_returns_data(symbols)
        portfolio_weights = np.array([weights[symbol] for symbol in symbols])
        
        scenarios = {
            'market_crash': {
                'description': '2008-style market crash',
                'shock': -0.40,  # -40% market shock
            },
            'interest_rate_spike': {
                'description': 'Sharp interest rate increase',
                'shock': -0.15,  # -15% shock
            },
            'volatility_spike': {
                'description': 'Volatility regime change',
                'shock': -0.25,  # -25% shock
            },
        }
        
        results = {}
        for name, scenario in scenarios.items():
            # Calculate stressed returns
            stressed_returns = returns * (1 + scenario['shock'])
            portfolio_returns = stressed_returns @ portfolio_weights
            
            results[name] = {
                'description': scenario['description'],
                'impact': float(portfolio_returns.mean() * 252),  # Annualized impact
                'max_drawdown': float(
                    (portfolio_returns.cumsum() - portfolio_returns.cumsum().expanding().max()).min()
                ),
                'recovery_days': int(
                    len(portfolio_returns[portfolio_returns.cumsum() < 0].index)
                ),
            }
            
        return results

async def get_risk_metrics() -> RiskMetrics:
    """Factory function to create RiskMetrics instance"""
    from ..market_data.provider import get_market_data_provider
    
    market_data = await get_market_data_provider()
    return RiskMetrics(market_data)
