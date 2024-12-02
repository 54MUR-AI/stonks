import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import yfinance as yf
from datetime import datetime, timedelta
import statsmodels.api as sm
from arch import arch_model

class ScenarioType(str, Enum):
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"
    SENSITIVITY = "sensitivity"
    REGIME_CHANGE = "regime_change"

@dataclass
class StressScenario:
    name: str
    type: ScenarioType
    description: str
    shocks: Dict[str, float]
    correlation_adjustments: Optional[Dict[str, float]] = None
    volatility_adjustments: Optional[Dict[str, float]] = None
    probability: Optional[float] = None

@dataclass
class StressTestResult:
    scenario_name: str
    portfolio_impact: float
    asset_impacts: Dict[str, float]
    risk_metrics: Dict[str, float]
    correlation_changes: Optional[Dict[str, float]] = None
    volatility_changes: Optional[Dict[str, float]] = None

class StressTestingService:
    def __init__(self):
        self.historical_scenarios = {
            "2008_financial_crisis": StressScenario(
                name="2008 Financial Crisis",
                type=ScenarioType.HISTORICAL,
                description="Replication of 2008 financial crisis market conditions",
                shocks={
                    "SPY": -0.56,  # S&P 500
                    "QQQ": -0.54,  # NASDAQ
                    "IWM": -0.59,  # Russell 2000
                    "EFA": -0.61,  # International Developed
                    "EEM": -0.65,  # Emerging Markets
                    "AGG": 0.05,   # US Aggregate Bonds
                    "GLD": 0.02,   # Gold
                    "USO": -0.77,  # Oil
                },
                correlation_adjustments={
                    "SPY_AGG": -0.6,  # Increased flight to safety
                    "SPY_GLD": -0.4,  # Gold as safe haven
                }
            ),
            "covid_crash_2020": StressScenario(
                name="COVID-19 Market Crash",
                type=ScenarioType.HISTORICAL,
                description="Market conditions during March 2020 COVID crash",
                shocks={
                    "SPY": -0.34,
                    "QQQ": -0.28,
                    "IWM": -0.41,
                    "EFA": -0.35,
                    "EEM": -0.33,
                    "AGG": -0.06,
                    "GLD": -0.12,
                    "USO": -0.66,
                },
                correlation_adjustments={
                    "SPY_AGG": 0.3,  # Temporary correlation breakdown
                    "SPY_GLD": 0.2,  # Initial gold selloff
                }
            ),
            "taper_tantrum_2013": StressScenario(
                name="2013 Taper Tantrum",
                type=ScenarioType.HISTORICAL,
                description="Market reaction to Fed's announcement of QE tapering",
                shocks={
                    "SPY": -0.06,
                    "AGG": -0.08,
                    "TLT": -0.15,  # Long-term Treasury
                    "EMB": -0.12,  # Emerging Market Bonds
                    "EEM": -0.16,
                },
                volatility_adjustments={
                    "AGG": 1.5,
                    "TLT": 1.8,
                }
            ),
            "tech_bubble_2000": StressScenario(
                name="2000 Tech Bubble Burst",
                type=ScenarioType.HISTORICAL,
                description="Dot-com bubble burst market conditions",
                shocks={
                    "SPY": -0.49,
                    "QQQ": -0.83,
                    "IWM": -0.59,
                    "XLK": -0.85,  # Technology sector
                },
                correlation_adjustments={
                    "SPY_AGG": -0.3,
                    "QQQ_AGG": -0.4,
                }
            ),
            "black_monday_1987": StressScenario(
                name="Black Monday 1987",
                type=ScenarioType.HISTORICAL,
                description="Extreme market crash of October 19, 1987",
                shocks={
                    "SPY": -0.22,
                    "QQQ": -0.24,
                    "IWM": -0.25,
                    "EFA": -0.23,
                },
                volatility_adjustments={
                    "SPY": 3.0,
                    "QQQ": 3.2,
                }
            ),
        }

    def get_historical_scenario(self, name: str) -> Optional[StressScenario]:
        return self.historical_scenarios.get(name)

    def get_available_historical_scenarios(self) -> List[str]:
        return list(self.historical_scenarios.keys())

    def detect_market_regime(self, returns: pd.DataFrame, n_regimes: int = 2) -> Tuple[np.ndarray, GaussianMixture]:
        """Detect market regimes using Gaussian Mixture Models"""
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regimes = gmm.fit_predict(returns)
        return regimes, gmm

    def calculate_regime_parameters(self, returns: pd.DataFrame, regimes: np.ndarray) -> Dict:
        """Calculate parameters for each regime"""
        n_regimes = len(np.unique(regimes))
        regime_params = {}
        
        for i in range(n_regimes):
            regime_data = returns[regimes == i]
            regime_params[i] = {
                'mean': regime_data.mean(),
                'vol': regime_data.std(),
                'corr': regime_data.corr(),
                'frequency': len(regime_data) / len(returns)
            }
        
        return regime_params

    def generate_monte_carlo_scenarios(
        self,
        symbols: List[str],
        n_scenarios: int = 1000,
        horizon_days: int = 30,
        confidence_level: float = 0.95,
        use_garch: bool = True
    ) -> List[StressScenario]:
        """Generate Monte Carlo scenarios with GARCH and regime switching"""
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        data = pd.DataFrame()
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                data[symbol] = hist['Close'].pct_change()
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        data = data.dropna()
        
        # Detect market regimes
        regimes, gmm = self.detect_market_regime(data)
        regime_params = self.calculate_regime_parameters(data, regimes)
        
        scenarios = []
        
        # Generate scenarios for each regime
        for regime_idx, regime_param in regime_params.items():
            regime_mean = regime_param['mean']
            regime_vol = regime_param['vol']
            regime_corr = regime_param['corr']
            
            # Fit GARCH models if enabled
            if use_garch:
                garch_forecasts = {}
                for symbol in symbols:
                    try:
                        model = arch_model(data[symbol], vol='Garch', p=1, q=1)
                        result = model.fit(disp='off')
                        forecast = result.forecast(horizon=horizon_days)
                        garch_forecasts[symbol] = forecast.variance.values[-1][0]
                    except:
                        garch_forecasts[symbol] = regime_vol[symbol] ** 2
                
                # Adjust volatilities based on GARCH forecasts
                regime_vol = pd.Series({
                    symbol: np.sqrt(garch_forecasts[symbol])
                    for symbol in symbols
                })
            
            # Generate correlated random returns
            n_assets = len(symbols)
            L = np.linalg.cholesky(regime_corr)
            
            for i in range(n_scenarios // len(regime_params)):
                Z = np.random.standard_normal((n_assets, horizon_days))
                corr_returns = np.dot(L, Z)
                
                # Apply regime parameters
                returns = (
                    regime_mean.values.reshape(-1, 1) +
                    regime_vol.values.reshape(-1, 1) * corr_returns
                )
                
                # Calculate cumulative returns
                cum_returns = np.prod(1 + returns, axis=1) - 1
                
                # Create scenario
                shocks = {
                    symbol: return_val
                    for symbol, return_val in zip(symbols, cum_returns)
                }
                
                probability = stats.norm.cdf(
                    np.mean(cum_returns),
                    loc=np.mean(regime_mean) * horizon_days,
                    scale=np.sqrt(horizon_days) * np.mean(regime_vol)
                )
                
                scenario = StressScenario(
                    name=f"Monte Carlo (Regime {regime_idx}) - Scenario {i}",
                    type=ScenarioType.MONTE_CARLO,
                    description=f"Simulated scenario under market regime {regime_idx}",
                    shocks=shocks,
                    correlation_adjustments=None,
                    volatility_adjustments=None,
                    probability=probability
                )
                
                scenarios.append(scenario)
        
        # Sort scenarios by probability and select most extreme ones
        scenarios.sort(key=lambda x: abs(sum(x.shocks.values())))
        n_extreme = int(n_scenarios * (1 - confidence_level))
        
        return scenarios[:n_extreme]

    def run_sensitivity_analysis(
        self,
        symbols: List[str],
        base_shocks: Dict[str, float],
        steps: int = 5,
        correlation_range: float = 0.3
    ) -> List[StressScenario]:
        """Run sensitivity analysis with correlation impacts"""
        scenarios = []
        
        # Generate shock variations
        for step in range(steps):
            shock_multiplier = 0.5 + step * 0.5  # 0.5x to 2.5x
            
            # Vary correlations
            for corr_step in range(3):  # Low, Base, High correlation
                corr_adjustment = -correlation_range + corr_step * correlation_range
                
                shocks = {
                    symbol: shock * shock_multiplier
                    for symbol, shock in base_shocks.items()
                }
                
                # Create correlation adjustments for pairs of assets
                correlation_adjustments = {}
                symbols_list = list(symbols)
                for i in range(len(symbols_list)):
                    for j in range(i + 1, len(symbols_list)):
                        pair = f"{symbols_list[i]}_{symbols_list[j]}"
                        correlation_adjustments[pair] = corr_adjustment
                
                scenario = StressScenario(
                    name=f"Sensitivity {shock_multiplier}x (Corr: {corr_adjustment:+.1f})",
                    type=ScenarioType.SENSITIVITY,
                    description=f"Sensitivity test with {shock_multiplier}x base shocks and {corr_adjustment:+.1f} correlation adjustment",
                    shocks=shocks,
                    correlation_adjustments=correlation_adjustments
                )
                
                scenarios.append(scenario)
        
        return scenarios

    def create_hypothetical_scenario(
        self,
        name: str,
        description: str,
        shocks: Dict[str, float],
        correlation_adjustments: Optional[Dict[str, float]] = None,
        volatility_adjustments: Optional[Dict[str, float]] = None
    ) -> StressScenario:
        """Create a custom hypothetical scenario"""
        return StressScenario(
            name=name,
            type=ScenarioType.HYPOTHETICAL,
            description=description,
            shocks=shocks,
            correlation_adjustments=correlation_adjustments,
            volatility_adjustments=volatility_adjustments
        )

    def run_stress_test(
        self,
        portfolio_values: Dict[str, float],
        scenario: StressScenario
    ) -> StressTestResult:
        """Run stress test on a portfolio"""
        # Calculate direct impact from shocks
        asset_impacts = {}
        total_impact = 0
        total_portfolio_value = sum(portfolio_values.values())
        
        for symbol, value in portfolio_values.items():
            if symbol in scenario.shocks:
                impact = scenario.shocks[symbol]
                asset_impacts[symbol] = impact
                total_impact += (value / total_portfolio_value) * impact
        
        # Calculate risk metrics under stress
        stressed_values = {
            symbol: value * (1 + scenario.shocks.get(symbol, 0))
            for symbol, value in portfolio_values.items()
        }
        
        # Calculate stressed VaR (using historical simulation)
        stressed_var_95 = -np.percentile(list(scenario.shocks.values()), 5)
        
        # Calculate stressed Sharpe ratio (assuming risk-free rate of 2%)
        rf_rate = 0.02
        stressed_returns = np.array(list(scenario.shocks.values()))
        stressed_vol = np.std(stressed_returns)
        stressed_sharpe = (np.mean(stressed_returns) - rf_rate) / stressed_vol if stressed_vol > 0 else 0
        
        risk_metrics = {
            'stressed_var_95': stressed_var_95,
            'stressed_sharpe': stressed_sharpe,
            'portfolio_volatility': stressed_vol
        }
        
        return StressTestResult(
            scenario_name=scenario.name,
            portfolio_impact=total_impact,
            asset_impacts=asset_impacts,
            risk_metrics=risk_metrics,
            correlation_changes=scenario.correlation_adjustments,
            volatility_changes=scenario.volatility_adjustments
        )

stress_testing_service = StressTestingService()
