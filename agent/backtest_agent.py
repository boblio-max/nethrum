from typing import Dict, Any, Optional, List, Tuple
from .base_agent import BaseAgent
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from enum import Enum
import logging

class RiskMetric(Enum):
    """Supported risk-adjusted return metrics"""
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    OMEGA = "omega"

@dataclass
class BacktestResult:
    """Structured backtest results with comprehensive metrics"""
    symbol: str
    alpha: float
    score: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    total_returns: float
    volatility: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'alpha': self.alpha,
            'score': self.score,
            'metrics': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'max_drawdown': self.max_drawdown,
                'calmar_ratio': self.calmar_ratio,
                'win_rate': self.win_rate,
                'total_returns': self.total_returns,
                'volatility': self.volatility,
                'skewness': self.skewness,
                'kurtosis': self.kurtosis,
                'var_95': self.var_95,
                'cvar_95': self.cvar_95
            }
        }

class BacktestingAgent(BaseAgent):
    """
    Advanced backtesting agent with comprehensive risk analytics.
    
    Features:
    - Multiple risk-adjusted return metrics (Sharpe, Sortino, Calmar, Omega)
    - Drawdown analysis and tail risk measures (VaR, CVaR)
    - Distribution statistics (skewness, kurtosis)
    - Vectorized computations for performance
    - Robust error handling and validation
    - Detailed logging and diagnostics
    """
    
    def __init__(self, infra, risk_free_rate: float = 0.0, annual_trading_days: int = 252):
        super().__init__(name='BacktestingAgent', agent_type='backtest')
        self.infra = infra
        self.risk_free_rate = risk_free_rate
        self.annual_trading_days = annual_trading_days
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Register tasks
        self.register_task('run_backtest', self.task_run_backtest)
        self.register_task('compute_metrics', self.task_compute_metrics)
        self.register_task('analyze_drawdowns', self.task_analyze_drawdowns)
        
    def _validate_prices(self, prices: NDArray[np.float64]) -> bool:
        """Validate price data integrity"""
        if len(prices) < 2:
            self.logger.warning("Insufficient price data: length < 2")
            return False
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            self.logger.error("Invalid price data: contains NaN or Inf")
            return False
        if np.any(prices <= 0):
            self.logger.error("Invalid price data: contains non-positive values")
            return False
        return True
    
    def _compute_returns(self, prices: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute log returns for better statistical properties"""
        return np.diff(np.log(prices))
    
    def _compute_sharpe_ratio(self, returns: NDArray[np.float64]) -> float:
        """Compute annualized Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - self.risk_free_rate / self.annual_trading_days
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        if std_return < 1e-9:
            return 0.0
        return float(mean_return / std_return * np.sqrt(self.annual_trading_days))
    
    def _compute_sortino_ratio(self, returns: NDArray[np.float64]) -> float:
        """Compute Sortino ratio (downside deviation only)"""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - self.risk_free_rate / self.annual_trading_days
        mean_return = np.mean(excess_returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        if downside_std < 1e-9:
            return 0.0
        return float(mean_return / downside_std * np.sqrt(self.annual_trading_days))
    
    def _compute_max_drawdown(self, prices: NDArray[np.float64]) -> float:
        """Compute maximum drawdown from peak"""
        cumulative = np.cumprod(1 + np.diff(prices) / prices[:-1])
        cumulative = np.insert(cumulative, 0, 1.0)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(np.min(drawdowns))
    
    def _compute_calmar_ratio(self, returns: NDArray[np.float64], max_dd: float) -> float:
        """Compute Calmar ratio (annualized return / max drawdown)"""
        if abs(max_dd) < 1e-9:
            return 0.0
        annual_return = np.mean(returns) * self.annual_trading_days
        return float(annual_return / abs(max_dd))
    
    def _compute_win_rate(self, returns: NDArray[np.float64]) -> float:
        """Compute percentage of positive returns"""
        if len(returns) == 0:
            return 0.0
        return float(np.sum(returns > 0) / len(returns))
    
    def _compute_var_cvar(self, returns: NDArray[np.float64], confidence: float = 0.95) -> Tuple[float, float]:
        """Compute Value at Risk and Conditional VaR (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0, 0.0
        var = float(np.percentile(returns, (1 - confidence) * 100))
        cvar = float(np.mean(returns[returns <= var])) if np.any(returns <= var) else var
        return var, cvar
    
    def task_compute_metrics(self, prices: NDArray[np.float64]) -> Dict[str, float]:
        """Compute comprehensive performance metrics"""
        if not self._validate_prices(prices):
            return self._empty_metrics()
        
        returns = self._compute_returns(prices)
        max_dd = self._compute_max_drawdown(prices)
        var_95, cvar_95 = self._compute_var_cvar(returns)
        
        metrics = {
            'sharpe_ratio': self._compute_sharpe_ratio(returns),
            'sortino_ratio': self._compute_sortino_ratio(returns),
            'max_drawdown': max_dd,
            'calmar_ratio': self._compute_calmar_ratio(returns, max_dd),
            'win_rate': self._compute_win_rate(returns),
            'total_returns': float(np.sum(returns)),
            'volatility': float(np.std(returns, ddof=1) * np.sqrt(self.annual_trading_days)),
            'skewness': float(self._safe_skewness(returns)),
            'kurtosis': float(self._safe_kurtosis(returns)),
            'var_95': var_95,
            'cvar_95': cvar_95
        }
        
        return metrics
    
    def _safe_skewness(self, returns: NDArray[np.float64]) -> float:
        """Compute skewness with error handling"""
        if len(returns) < 3:
            return 0.0
        try:
            from scipy import stats
            return stats.skew(returns, bias=False)
        except (ImportError, Exception):
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            if std < 1e-9:
                return 0.0
            return np.mean(((returns - mean) / std) ** 3)
    
    def _safe_kurtosis(self, returns: NDArray[np.float64]) -> float:
        """Compute excess kurtosis with error handling"""
        if len(returns) < 4:
            return 0.0
        try:
            from scipy import stats
            return stats.kurtosis(returns, bias=False)
        except (ImportError, Exception):
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            if std < 1e-9:
                return 0.0
            return np.mean(((returns - mean) / std) ** 4) - 3
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return zero metrics for invalid data"""
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'total_returns': 0.0,
            'volatility': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0
        }
    
    def task_analyze_drawdowns(self, prices: NDArray[np.float64], top_n: int = 5) -> List[Dict[str, Any]]:
        """Analyze top N drawdown periods"""
        if not self._validate_prices(prices):
            return []
        
        cumulative = np.cumprod(1 + np.diff(prices) / prices[:-1])
        cumulative = np.insert(cumulative, 0, 1.0)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        # Find drawdown periods
        dd_periods = []
        in_dd = False
        start_idx = 0
        
        for i in range(len(drawdowns)):
            if drawdowns[i] < -1e-6 and not in_dd:
                start_idx = i
                in_dd = True
            elif drawdowns[i] >= -1e-6 and in_dd:
                min_idx = start_idx + np.argmin(drawdowns[start_idx:i])
                dd_periods.append({
                    'start': start_idx,
                    'trough': min_idx,
                    'end': i,
                    'depth': float(drawdowns[min_idx]),
                    'duration': i - start_idx
                })
                in_dd = False
        
        # Sort by depth and return top N
        dd_periods.sort(key=lambda x: x['depth'])
        return dd_periods[:top_n]
    
    def task_run_backtest(
        self, 
        symbol: str, 
        alpha: float, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest with advanced metrics
        
        Args:
            symbol: Trading symbol
            alpha: Alpha coefficient for scoring
            params: Additional parameters (lookback, metric_type, etc.)
        
        Returns:
            Dictionary containing BacktestResult
        """
        params = params or {}
        
        try:
            # Load data
            data_res = self.infra.execute_task(
                'load_data', 
                symbol=symbol, 
                lookback=params.get('lookback', 256)
            )
            
            raw_data = data_res.get('data') or data_res.get('result', {}).get('data', [])
            prices = np.array(raw_data, dtype=np.float64)
            
            if not self._validate_prices(prices):
                self.logger.warning(f"Invalid price data for {symbol}")
                result = self._create_empty_result(symbol, alpha)
                self.last_result = result.to_dict()
                return result.to_dict()
            
            # Compute all metrics
            metrics = self.task_compute_metrics(prices)
            
            # Determine primary metric for scoring
            metric_type = params.get('metric_type', RiskMetric.SHARPE.value)
            primary_metric = metrics.get(f'{metric_type}_ratio', metrics['sharpe_ratio'])
            
            # Calculate final score
            score = round(alpha * primary_metric, 6)
            
            # Create structured result
            result = BacktestResult(
                symbol=symbol,
                alpha=alpha,
                score=score,
                **{k: round(v, 6) for k, v in metrics.items()}
            )
            
            self.last_result = result.to_dict()
            self.logger.info(f"Backtest completed for {symbol}: score={score:.6f}")
            
            return result.to_dict()
            
        except Exception as e:
            self.logger.error(f"Backtest failed for {symbol}: {str(e)}")
            result = self._create_empty_result(symbol, alpha)
            self.last_result = result.to_dict()
            return result.to_dict()
    
    def _create_empty_result(self, symbol: str, alpha: float) -> BacktestResult:
        """Create empty result for error cases"""
        return BacktestResult(
            symbol=symbol,
            alpha=alpha,
            score=0.0,
            **self._empty_metrics()
        )
    
    def run_single(self, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run single symbol backtest"""
        return self.task_run_backtest(
            symbol, 
            params.get('alpha', 0.0), 
            params
        )
    
    def run_pipeline(self, symbol: str, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest as part of pipeline with shared state"""
        alpha = shared_state.get('alpha', 0.0)
        backtest_params = shared_state.get('backtest_params', {})
        
        # Use market_data from shared_state if available
        market_data = shared_state.get('market_data', [])
        if market_data:
            prices = np.array(market_data, dtype=np.float64)
            if self._validate_prices(prices):
                metrics = self.task_compute_metrics(prices)
                metric_type = backtest_params.get('metric_type', RiskMetric.SHARPE.value)
                primary_metric = metrics.get(f'{metric_type}_ratio', metrics['sharpe_ratio'])
                score = round(alpha * primary_metric, 6)
                result = BacktestResult(
                    symbol=symbol,
                    alpha=alpha,
                    score=score,
                    **{k: round(v, 6) for k, v in metrics.items()}
                ).to_dict()
            else:
                result = self._create_empty_result(symbol, alpha).to_dict()
        else:
            result = self.task_run_backtest(symbol, alpha, backtest_params)
        
        # Update shared state with results
        shared_state['backtest_score'] = result['score']
        shared_state['sharpe_ratio'] = result['metrics']['sharpe_ratio']
        shared_state['sortino_ratio'] = result['metrics']['sortino_ratio']
        shared_state['max_drawdown'] = result['metrics']['max_drawdown']
        shared_state['all_metrics'] = result['metrics']
        
        return {'backtest': result}
    
    def compare_strategies(
        self, 
        symbol: str, 
        alphas: List[float], 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Compare multiple alpha values for a single symbol"""
        results = []
        for alpha in alphas:
            result = self.task_run_backtest(symbol, alpha, params)
            results.append(result)
        return sorted(results, key=lambda x: x['score'], reverse=True)