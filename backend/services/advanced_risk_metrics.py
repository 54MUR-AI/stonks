import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA
import statsmodels.api as sm
from arch import arch_model
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    var_parametric: float
    var_historical: float
    cvar_parametric: float
    cvar_historical: float
    beta: float
    alpha: float
    treynor_ratio: float
    information_ratio: float
    sortino_ratio: float
    omega_ratio: float
    kurtosis: float
    skewness: float
    tail_dependence: float
    max_drawdown: float
    calmar_ratio: float
    factor_exposures: Dict[str, float]
    risk_contribution: Dict[str, float]
    systematic_risk: float
    idiosyncratic_risk: float
    liquidity_score: float

class AdvancedRiskAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02  # Assumed risk-free rate
        self.market_index = "^GSPC"  # S&P 500 as market benchmark
        self.lookback_period = "2y"
        self.factor_tickers = {
            "Market": "^GSPC",    # S&P 500
            "Size": "^RUT",       # Russell 2000
            "Value": "IWD",       # Russell 1000 Value
            "Momentum": "MTUM",   # iShares Momentum ETF
            "Quality": "QUAL",    # iShares Quality ETF
            "LowVol": "USMV"      # iShares Low Vol ETF
        }
        
    def _get_returns_data(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch and prepare returns data"""
        data = pd.DataFrame()
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                data[symbol] = hist['Close'].pct_change()
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                
        return data.dropna()
        
    def _calculate_var_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float, float]:
        """Calculate VaR and CVaR using both parametric and historical methods"""
        # Parametric VaR
        mu = returns.mean()
        sigma = returns.std()
        var_parametric = stats.norm.ppf(1 - confidence_level, mu, sigma)
        
        # Historical VaR
        var_historical = returns.quantile(1 - confidence_level)
        
        # Parametric CVaR
        z_score = stats.norm.ppf(1 - confidence_level)
        cvar_parametric = mu - sigma * stats.norm.pdf(z_score) / (1 - confidence_level)
        
        # Historical CVaR
        cvar_historical = returns[returns <= var_historical].mean()
        
        return var_parametric, var_historical, cvar_parametric, cvar_historical
        
    def _calculate_factor_exposures(
        self,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate factor exposures using multi-factor regression"""
        # Get factor returns
        factor_returns = pd.DataFrame()
        for factor, ticker in self.factor_tickers.items():
            factor_data = yf.Ticker(ticker).history(period=self.lookback_period)
            factor_returns[factor] = factor_data['Close'].pct_change()
            
        # Prepare data for regression
        factor_returns = factor_returns.dropna()
        common_index = returns.index.intersection(factor_returns.index)
        Y = returns[common_index]
        X = factor_returns.loc[common_index]
        X = sm.add_constant(X)
        
        # Run regression
        model = sm.OLS(Y, X).fit()
        exposures = model.params.to_dict()
        exposures.pop('const', None)
        
        return exposures
        
    def _calculate_risk_contribution(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """Calculate risk contribution of each asset"""
        cov_matrix = returns.cov().values
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        return dict(zip(returns.columns, risk_contrib))
        
    def _calculate_tail_dependence(
        self,
        returns: pd.DataFrame,
        threshold: float = 0.05
    ) -> float:
        """Calculate tail dependence coefficient"""
        # Convert returns to ranks
        ranks = returns.rank() / (len(returns) + 1)
        
        # Calculate tail events
        tail_events = (ranks <= threshold).astype(int)
        
        # Calculate joint tail probability
        joint_tail_prob = tail_events.mean().mean()
        
        return joint_tail_prob / threshold
        
    def _calculate_liquidity_score(
        self,
        symbols: List[str]
    ) -> float:
        """Calculate portfolio liquidity score"""
        volumes = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo")
            avg_volume = hist['Volume'].mean()
            avg_price = hist['Close'].mean()
            dollar_volume = avg_volume * avg_price
            volumes.append(dollar_volume)
            
        # Normalize and average
        volumes = np.array(volumes)
        liquidity_score = np.mean(np.log(volumes) / np.log(volumes).max())
        
        return liquidity_score
        
    def calculate_advanced_metrics(
        self,
        portfolio_data: Dict[str, Dict],
        benchmark_symbol: str = "^GSPC"
    ) -> RiskMetrics:
        """Calculate comprehensive set of risk metrics"""
        symbols = list(portfolio_data.keys())
        weights = np.array([
            data['quantity'] * data['current_price']
            for data in portfolio_data.values()
        ])
        weights = weights / weights.sum()
        
        # Get returns data
        returns_data = self._get_returns_data(symbols + [benchmark_symbol])
        portfolio_returns = returns_data[symbols].dot(weights)
        benchmark_returns = returns_data[benchmark_symbol]
        
        # Calculate VaR and CVaR
        var_p, var_h, cvar_p, cvar_h = self._calculate_var_cvar(portfolio_returns)
        
        # Calculate traditional metrics
        excess_returns = portfolio_returns - self.risk_free_rate
        benchmark_excess = benchmark_returns - self.risk_free_rate
        
        beta = np.cov(portfolio_returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
        alpha = portfolio_returns.mean() - (self.risk_free_rate + beta * (benchmark_returns.mean() - self.risk_free_rate))
        
        portfolio_std = portfolio_returns.std()
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std()
        
        treynor_ratio = excess_returns.mean() / beta
        information_ratio = (portfolio_returns - benchmark_returns).mean() / (portfolio_returns - benchmark_returns).std()
        sortino_ratio = excess_returns.mean() / downside_std
        omega_ratio = len(portfolio_returns[portfolio_returns > 0]) / len(downside_returns)
        
        # Calculate drawdown metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / running_max - 1
        max_drawdown = drawdowns.min()
        calmar_ratio = portfolio_returns.mean() * 252 / abs(max_drawdown)
        
        # Calculate factor exposures and risk decomposition
        factor_exposures = self._calculate_factor_exposures(portfolio_returns)
        risk_contribution = self._calculate_risk_contribution(returns_data[symbols], weights)
        
        # Calculate systematic and idiosyncratic risk
        market_model = sm.OLS(
            portfolio_returns,
            sm.add_constant(benchmark_returns)
        ).fit()
        systematic_risk = (market_model.params[1] * benchmark_returns.std()) ** 2
        idiosyncratic_risk = market_model.resid.var()
        
        # Calculate tail risk measures
        tail_dependence = self._calculate_tail_dependence(returns_data[symbols])
        
        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(symbols)
        
        return RiskMetrics(
            var_parametric=var_p,
            var_historical=var_h,
            cvar_parametric=cvar_p,
            cvar_historical=cvar_h,
            beta=beta,
            alpha=alpha,
            treynor_ratio=treynor_ratio,
            information_ratio=information_ratio,
            sortino_ratio=sortino_ratio,
            omega_ratio=omega_ratio,
            kurtosis=stats.kurtosis(portfolio_returns),
            skewness=stats.skew(portfolio_returns),
            tail_dependence=tail_dependence,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            factor_exposures=factor_exposures,
            risk_contribution=risk_contribution,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            liquidity_score=liquidity_score
        )

risk_analyzer = AdvancedRiskAnalyzer()
