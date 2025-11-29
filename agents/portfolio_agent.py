from typing import Dict, Any, Optional, List, Tuple, Set
from .base_agent import BaseAgent, TaskPriority, ExecutionMode, ExecutionResult
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import numpy as np
import logging

class AllocationMethod(Enum):
    """Portfolio allocation strategies"""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP = "market_cap"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL = "hierarchical"
    CUSTOM = "custom"

class RebalanceFrequency(Enum):
    """Rebalancing frequency options"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    THRESHOLD = "threshold"
    NEVER = "never"

@dataclass
class Position:
    """Comprehensive position representation"""
    symbol: str
    size: float  # Dollar value
    shares: float = 0.0
    weight: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    cost_basis: float = 0.0
    sector: str = ""
    asset_class: str = ""
    
    # Risk metrics
    var_contribution: float = 0.0
    beta: float = 1.0
    volatility: float = 0.0
    
    # Timestamps
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price if self.current_price > 0 else self.size
    
    @property
    def total_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl
    
    @property
    def pnl_percent(self) -> float:
        return (self.total_pnl / self.cost_basis * 100) if self.cost_basis > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'size': round(self.size, 2),
            'shares': round(self.shares, 6),
            'weight': round(self.weight, 4),
            'entry_price': round(self.entry_price, 4),
            'current_price': round(self.current_price, 4),
            'market_value': round(self.market_value, 2),
            'unrealized_pnl': round(self.unrealized_pnl, 2),
            'realized_pnl': round(self.realized_pnl, 2),
            'total_pnl': round(self.total_pnl, 2),
            'pnl_percent': round(self.pnl_percent, 2),
            'cost_basis': round(self.cost_basis, 2),
            'sector': self.sector,
            'asset_class': self.asset_class,
            'beta': round(self.beta, 4),
            'volatility': round(self.volatility, 4),
            'var_contribution': round(self.var_contribution, 4),
            'opened_at': self.opened_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio performance metrics"""
    total_value: float = 0.0
    total_cash: float = 0.0
    total_equity: float = 0.0
    invested_capital: float = 0.0
    
    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    
    # Risk metrics
    portfolio_volatility: float = 0.0
    portfolio_beta: float = 1.0
    portfolio_var_95: float = 0.0
    portfolio_cvar_95: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Diversification
    num_positions: int = 0
    concentration_hhi: float = 0.0
    effective_positions: float = 0.0
    
    # Exposures
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    
    # Sector/Asset class breakdown
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    asset_class_exposure: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': {
                'total_value': round(self.total_value, 2),
                'total_cash': round(self.total_cash, 2),
                'total_equity': round(self.total_equity, 2),
                'invested_capital': round(self.invested_capital, 2),
                'num_positions': self.num_positions
            },
            'performance': {
                'unrealized_pnl': round(self.unrealized_pnl, 2),
                'realized_pnl': round(self.realized_pnl, 2),
                'total_pnl': round(self.total_pnl, 2),
                'total_return_pct': round(self.total_return_pct, 2)
            },
            'risk': {
                'portfolio_volatility': round(self.portfolio_volatility, 4),
                'portfolio_beta': round(self.portfolio_beta, 4),
                'var_95': round(self.portfolio_var_95, 2),
                'cvar_95': round(self.portfolio_cvar_95, 2),
                'sharpe_ratio': round(self.sharpe_ratio, 4),
                'sortino_ratio': round(self.sortino_ratio, 4),
                'max_drawdown': round(self.max_drawdown, 4)
            },
            'diversification': {
                'concentration_hhi': round(self.concentration_hhi, 4),
                'effective_positions': round(self.effective_positions, 2)
            },
            'exposure': {
                'long': round(self.long_exposure, 2),
                'short': round(self.short_exposure, 2),
                'net': round(self.net_exposure, 2),
                'gross': round(self.gross_exposure, 2)
            },
            'sector_exposure': {k: round(v, 2) for k, v in self.sector_exposure.items()},
            'asset_class_exposure': {k: round(v, 2) for k, v in self.asset_class_exposure.items()}
        }

@dataclass
class RebalanceRecommendation:
    """Rebalancing trade recommendations"""
    symbol: str
    current_weight: float
    target_weight: float
    weight_diff: float
    current_size: float
    target_size: float
    trade_size: float
    trade_direction: str  # 'buy', 'sell', 'hold'
    priority: int = 0
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'current_weight': round(self.current_weight, 4),
            'target_weight': round(self.target_weight, 4),
            'weight_diff': round(self.weight_diff, 4),
            'current_size': round(self.current_size, 2),
            'target_size': round(self.target_size, 2),
            'trade_size': round(self.trade_size, 2),
            'trade_direction': self.trade_direction,
            'priority': self.priority,
            'reason': self.reason
        }

class PortfolioOptimizer:
    """Advanced portfolio optimization algorithms"""
    
    @staticmethod
    def equal_weight(symbols: List[str], **kwargs) -> Dict[str, float]:
        """Equal weight allocation"""
        n = len(symbols)
        return {symbol: 1.0 / n for symbol in symbols}
    
    @staticmethod
    def risk_parity(
        symbols: List[str],
        volatilities: Dict[str, float],
        correlations: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Risk parity allocation - equal risk contribution"""
        n = len(symbols)
        
        if correlations is None:
            # Assume uncorrelated
            vols = np.array([volatilities.get(s, 0.2) for s in symbols])
            inv_vols = 1.0 / vols
            weights = inv_vols / inv_vols.sum()
        else:
            # Use full covariance matrix
            vols = np.array([volatilities.get(s, 0.2) for s in symbols])
            cov_matrix = np.outer(vols, vols) * correlations
            
            # Iterative risk parity
            weights = np.ones(n) / n
            for _ in range(100):
                portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
                marginal_contrib = cov_matrix @ weights / portfolio_vol
                risk_contrib = weights * marginal_contrib
                target_risk = portfolio_vol / n
                weights = weights * target_risk / risk_contrib
                weights = weights / weights.sum()
        
        return {symbol: float(w) for symbol, w in zip(symbols, weights)}
    
    @staticmethod
    def min_variance(
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """Minimum variance portfolio"""
        n = len(symbols)
        
        # Solve: min w'Σw subject to w'1=1
        ones = np.ones(n)
        inv_cov = np.linalg.inv(covariance)
        weights = inv_cov @ ones / (ones @ inv_cov @ ones)
        weights = np.maximum(weights, 0)  # No shorts
        weights = weights / weights.sum()
        
        return {symbol: float(w) for symbol, w in zip(symbols, weights)}
    
    @staticmethod
    def max_sharpe(
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance: np.ndarray,
        risk_free_rate: float = 0.0,
        **kwargs
    ) -> Dict[str, float]:
        """Maximum Sharpe ratio portfolio"""
        n = len(symbols)
        returns = np.array([expected_returns.get(s, 0.0) for s in symbols])
        excess_returns = returns - risk_free_rate
        
        # Solve: max (w'r - rf) / sqrt(w'Σw)
        inv_cov = np.linalg.inv(covariance)
        weights = inv_cov @ excess_returns
        weights = np.maximum(weights, 0)  # No shorts
        weights = weights / weights.sum()
        
        return {symbol: float(w) for symbol, w in zip(symbols, weights)}
    
    @staticmethod
    def mean_variance(
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance: np.ndarray,
        risk_aversion: float = 1.0,
        **kwargs
    ) -> Dict[str, float]:
        """Mean-variance optimization with risk aversion parameter"""
        n = len(symbols)
        returns = np.array([expected_returns.get(s, 0.0) for s in symbols])
        
        # Solve: max w'r - λ/2 * w'Σw subject to w'1=1
        inv_cov = np.linalg.inv(covariance)
        ones = np.ones(n)
        
        # Analytical solution
        A = returns @ inv_cov @ ones
        B = ones @ inv_cov @ ones
        C = returns @ inv_cov @ returns
        
        weights = (inv_cov @ returns) / (risk_aversion * B)
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return {symbol: float(w) for symbol, w in zip(symbols, weights)}

class RiskManager:
    """Portfolio risk management and constraints"""
    
    def __init__(
        self,
        max_position_size: float = 0.2,
        max_sector_concentration: float = 0.4,
        max_single_loss: float = 0.05,
        var_limit: float = 0.02
    ):
        self.max_position_size = max_position_size
        self.max_sector_concentration = max_sector_concentration
        self.max_single_loss = max_single_loss
        self.var_limit = var_limit
    
    def check_constraints(
        self,
        positions: Dict[str, Position],
        total_value: float
    ) -> List[str]:
        """Check all risk constraints and return violations"""
        violations = []
        
        # Position size limits
        for symbol, pos in positions.items():
            weight = pos.market_value / total_value if total_value > 0 else 0
            if weight > self.max_position_size:
                violations.append(
                    f"{symbol} exceeds max position size: {weight:.2%} > {self.max_position_size:.2%}"
                )
        
        # Sector concentration
        sector_exposure = defaultdict(float)
        for pos in positions.values():
            if pos.sector:
                sector_exposure[pos.sector] += pos.market_value
        
        for sector, exposure in sector_exposure.items():
            weight = exposure / total_value if total_value > 0 else 0
            if weight > self.max_sector_concentration:
                violations.append(
                    f"Sector {sector} exceeds concentration limit: {weight:.2%} > {self.max_sector_concentration:.2%}"
                )
        
        return violations
    
    def apply_constraints(
        self,
        weights: Dict[str, float],
        positions: Dict[str, Position]
    ) -> Dict[str, float]:
        """Apply constraints to proposed weights"""
        adjusted = weights.copy()
        
        # Cap individual positions
        for symbol in adjusted:
            if adjusted[symbol] > self.max_position_size:
                adjusted[symbol] = self.max_position_size
        
        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted

class PortfolioAgent(BaseAgent):
    """
    Advanced portfolio management agent with optimization and risk management.
    
    Features:
    - Multiple allocation strategies (equal weight, risk parity, mean-variance, etc.)
    - Position tracking with P&L and risk metrics
    - Portfolio-level risk analytics (VaR, CVaR, Sharpe, Sortino)
    - Rebalancing recommendations with drift detection
    - Sector and asset class exposure tracking
    - Concentration and diversification metrics
    - Risk constraints and limit monitoring
    - Performance attribution
    - Transaction cost modeling
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
        rebalance_threshold: float = 0.05,
        max_position_size: float = 0.2,
        enable_risk_checks: bool = True,
        max_workers: int = 4
    ):
        super().__init__(
            name='PortfolioAgent',
            agent_type='portfolio',
            max_workers=max_workers
        )
        
        # Configuration
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.allocation_method = allocation_method
        self.rebalance_threshold = rebalance_threshold
        
        # State
        self.positions: Dict[str, Position] = {}
        self.historical_values: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        self.rebalance_history: List[Dict[str, Any]] = []
        
        # Components
        self.optimizer = PortfolioOptimizer()
        self.risk_manager = RiskManager(
            max_position_size=max_position_size
        ) if enable_risk_checks else None
        
        # Register tasks
        self.register_task(
            'build_position',
            self.task_build_position,
            priority=TaskPriority.HIGH,
            description="Build or update portfolio position"
        )
        self.register_task(
            'optimize_portfolio',
            self.task_optimize_portfolio,
            priority=TaskPriority.NORMAL,
            description="Optimize portfolio allocation"
        )
        self.register_task(
            'rebalance',
            self.task_rebalance,
            priority=TaskPriority.NORMAL,
            description="Generate rebalancing recommendations"
        )
        self.register_task(
            'calculate_metrics',
            self.task_calculate_metrics,
            priority=TaskPriority.NORMAL,
            description="Calculate portfolio metrics"
        )
        self.register_task(
            'update_prices',
            self.task_update_prices,
            priority=TaskPriority.HIGH,
            description="Update position prices and P&L"
        )
        self.register_task(
            'check_risk_limits',
            self.task_check_risk_limits,
            priority=TaskPriority.CRITICAL,
            description="Check risk constraints"
        )
        self.register_task(
            'get_exposure',
            self.task_get_exposure,
            priority=TaskPriority.LOW,
            description="Get portfolio exposures"
        )
        
        self.logger.info(
            f"PortfolioAgent initialized | capital=${initial_capital:,.2f} | "
            f"method={allocation_method.value}"
        )
    
    def task_build_position(
        self,
        symbol: str,
        allocation: float,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build or update portfolio position"""
        params = params or {}
        
        notional = params.get('notional', self.get_total_value())
        size = allocation * notional
        price = params.get('price', 100.0)
        shares = size / price if price > 0 else 0
        
        # Create or update position
        if symbol in self.positions:
            pos = self.positions[symbol]
            old_size = pos.size
            pos.size = size
            pos.shares = shares
            pos.current_price = price
            pos.updated_at = datetime.now()
            
            # Update cost basis for additional purchases
            if size > old_size:
                additional_cost = (size - old_size)
                pos.cost_basis += additional_cost
                self.current_cash -= additional_cost
        else:
            pos = Position(
                symbol=symbol,
                size=size,
                shares=shares,
                entry_price=price,
                current_price=price,
                cost_basis=size,
                sector=params.get('sector', ''),
                asset_class=params.get('asset_class', 'equity'),
                beta=params.get('beta', 1.0),
                volatility=params.get('volatility', 0.2)
            )
            self.positions[symbol] = pos
            self.current_cash -= size
        
        # Update weights
        self._update_weights()
        
        # Check risk limits
        if self.risk_manager:
            violations = self.risk_manager.check_constraints(
                self.positions,
                self.get_total_value()
            )
            if violations:
                self.logger.warning(f"Risk violations: {violations}")
        
        result = {
            'symbol': symbol,
            'size': round(size, 2),
            'shares': round(shares, 6),
            'allocation': round(allocation, 4),
            'notional': notional,
            'position': pos.to_dict()
        }
        
        self.last_result = result
        self.logger.info(f"Position built: {symbol} | size=${size:,.2f} | weight={allocation:.2%}")
        return result
    
    def task_optimize_portfolio(
        self,
        symbols: List[str],
        method: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation using specified method"""
        
        method_enum = AllocationMethod[method.upper()] if method else self.allocation_method
        
        # Get required data
        expected_returns = kwargs.get('expected_returns', {s: 0.1 for s in symbols})
        volatilities = kwargs.get('volatilities', {s: 0.2 for s in symbols})
        
        # Generate covariance matrix if needed
        if method_enum in [AllocationMethod.MIN_VARIANCE, AllocationMethod.MAX_SHARPE, AllocationMethod.MEAN_VARIANCE]:
            if 'covariance' not in kwargs:
                # Simple correlation assumption
                n = len(symbols)
                corr = np.full((n, n), 0.3)
                np.fill_diagonal(corr, 1.0)
                vols = np.array([volatilities.get(s, 0.2) for s in symbols])
                kwargs['covariance'] = np.outer(vols, vols) * corr
        
        # Optimize
        if method_enum == AllocationMethod.EQUAL_WEIGHT:
            weights = self.optimizer.equal_weight(symbols, **kwargs)
        elif method_enum == AllocationMethod.RISK_PARITY:
            weights = self.optimizer.risk_parity(symbols, volatilities, **kwargs)
        elif method_enum == AllocationMethod.MIN_VARIANCE:
            weights = self.optimizer.min_variance(symbols, expected_returns, **kwargs)
        elif method_enum == AllocationMethod.MAX_SHARPE:
            weights = self.optimizer.max_sharpe(symbols, expected_returns, **kwargs)
        elif method_enum == AllocationMethod.MEAN_VARIANCE:
            weights = self.optimizer.mean_variance(symbols, expected_returns, **kwargs)
        else:
            weights = self.optimizer.equal_weight(symbols)
        
        # Apply constraints if risk manager enabled
        if self.risk_manager:
            weights = self.risk_manager.apply_constraints(weights, self.positions)
        
        result = {
            'method': method_enum.value,
            'weights': {k: round(v, 6) for k, v in weights.items()},
            'num_positions': len(weights)
        }
        
        self.logger.info(f"Portfolio optimized: {method_enum.value} | {len(weights)} positions")
        return result
    
    def task_rebalance(
        self,
        target_weights: Optional[Dict[str, float]] = None,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate rebalancing recommendations"""
        
        total_value = self.get_total_value()
        
        # Get target weights
        if target_weights is None:
            if symbols is None:
                symbols = list(self.positions.keys())
            opt_result = self.task_optimize_portfolio(symbols, **kwargs)
            target_weights = opt_result['weights']
        
        # Calculate current weights
        current_weights = {
            symbol: pos.market_value / total_value if total_value > 0 else 0
            for symbol, pos in self.positions.items()
        }
        
        # Generate recommendations
        recommendations = []
        for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            current_size = self.positions[symbol].market_value if symbol in self.positions else 0.0
            target_size = target_weight * total_value
            trade_size = target_size - current_size
            
            # Determine action
            if abs(weight_diff) < self.rebalance_threshold:
                direction = 'hold'
                priority = 0
            elif trade_size > 0:
                direction = 'buy'
                priority = int(abs(weight_diff) * 100)
            else:
                direction = 'sell'
                priority = int(abs(weight_diff) * 100)
            
            reason = f"Drift: {weight_diff:+.2%}"
            
            rec = RebalanceRecommendation(
                symbol=symbol,
                current_weight=current_weight,
                target_weight=target_weight,
                weight_diff=weight_diff,
                current_size=current_size,
                target_size=target_size,
                trade_size=trade_size,
                trade_direction=direction,
                priority=priority,
                reason=reason
            )
            recommendations.append(rec)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        # Filter non-hold recommendations
        trades_needed = [r for r in recommendations if r.trade_direction != 'hold']
        
        result = {
            'total_value': round(total_value, 2),
            'recommendations': [r.to_dict() for r in recommendations],
            'trades_needed': len(trades_needed),
            'total_turnover': sum(abs(r.trade_size) for r in trades_needed)
        }
        
        self.rebalance_history.append({
            'timestamp': datetime.now(),
            'recommendations': result['recommendations']
        })
        
        self.logger.info(f"Rebalance analysis: {len(trades_needed)} trades recommended")
        return result
    
    def task_calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        
        total_value = self.get_total_value()
        total_equity = sum(pos.market_value for pos in self.positions.values())
        
        metrics = PortfolioMetrics()
        metrics.total_value = total_value
        metrics.total_cash = self.current_cash
        metrics.total_equity = total_equity
        metrics.invested_capital = self.initial_capital
        metrics.num_positions = len(self.positions)
        
        # P&L
        metrics.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        metrics.realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        metrics.total_pnl = metrics.unrealized_pnl + metrics.realized_pnl
        metrics.total_return_pct = (total_value / self.initial_capital - 1) * 100
        
        # Exposures
        for pos in self.positions.values():
            if pos.size > 0:
                metrics.long_exposure += pos.market_value
            else:
                metrics.short_exposure += abs(pos.market_value)
        
        metrics.net_exposure = metrics.long_exposure - metrics.short_exposure
        metrics.gross_exposure = metrics.long_exposure + metrics.short_exposure
        
        # Diversification
        weights = np.array([pos.weight for pos in self.positions.values()])
        metrics.concentration_hhi = float(np.sum(weights ** 2))
        metrics.effective_positions = 1.0 / metrics.concentration_hhi if metrics.concentration_hhi > 0 else 0.0
        
        # Sector/Asset class
        for pos in self.positions.values():
            if pos.sector:
                metrics.sector_exposure[pos.sector] = metrics.sector_exposure.get(pos.sector, 0) + pos.market_value
            if pos.asset_class:
                metrics.asset_class_exposure[pos.asset_class] = metrics.asset_class_exposure.get(pos.asset_class, 0) + pos.market_value
        
        # Risk metrics (simplified - would need historical returns for full calculation)
        if len(self.historical_values) > 1:
            values = [v for _, v in self.historical_values]
            returns = np.diff(values) / values[:-1]
            metrics.portfolio_volatility = float(np.std(returns) * np.sqrt(252))
            metrics.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
            
            # Max drawdown
            cummax = np.maximum.accumulate(values)
            drawdowns = (values - cummax) / cummax
            metrics.max_drawdown = float(np.min(drawdowns))
        
        return metrics.to_dict()
    
    def task_update_prices(
        self,
        prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """Update position prices and recalculate P&L"""
        
        updated = []
        for symbol, price in prices.items():
            if symbol in self.positions:
                pos = self.positions[symbol]
                old_price = pos.current_price
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.entry_price) * pos.shares
                pos.updated_at = datetime.now()
                updated.append(symbol)
        
        # Update weights
        self._update_weights()
        
        # Record portfolio value
        total_value = self.get_total_value()
        self.historical_values.append((datetime.now(), total_value))
        
        return {
            'updated_positions': updated,
            'total_value': round(total_value, 2),
            'timestamp': datetime.now().isoformat()
        }
    
    def task_check_risk_limits(self) -> Dict[str, Any]:
        """Check all risk constraints"""
        
        if not self.risk_manager:
            return {'risk_checks_enabled': False}
        
        total_value = self.get_total_value()
        violations = self.risk_manager.check_constraints(self.positions, total_value)
        
        return {
            'risk_checks_enabled': True,
            'violations': violations,
            'num_violations': len(violations),
            'compliant': len(violations) == 0
        }
    
    def task_get_exposure(
        self,
        by: str = 'sector'
    ) -> Dict[str, Any]:
        """Get portfolio exposures by sector or asset class"""
        
        exposure = defaultdict(float)
        total_value = self.get_total_value()
        
        for pos in self.positions.values():
            if by == 'sector' and pos.sector:
                exposure[pos.sector] += pos.market_value
            elif by == 'asset_class' and pos.asset_class:
                exposure[pos.asset_class] += pos.market_value
        
        # Calculate percentages
        exposure_pct = {
            k: round(v / total_value * 100, 2) if total_value > 0 else 0.0
            for k, v in exposure.items()
        }

        return {
            'exposure_pct': exposure_pct,
            'total_value': round(total_value, 2)
        }

    # ---- Helper methods ----
    def _update_weights(self):
        """Recalculate position weights based on current market values."""
        total = self.get_total_value()
        for pos in self.positions.values():
            pos.weight = pos.market_value / total if total > 0 else 0.0

    def get_total_value(self) -> float:
        """Compute current total portfolio value (cash + positions)."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return float(self.current_cash + positions_value)

    # ---- Required agent interface methods ----
    def run_single(self, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle single-symbol operations. Minimal compatibility implementation."""
        action = params.get('action', 'metrics')
        if action == 'build':
            allocation = params.get('allocation', 0.0)
            return self.task_build_position(symbol, allocation, params)
        if action == 'update_prices':
            price = params.get('price')
            if price is None:
                return {'status': 'error', 'message': 'price required for update_prices'}
            return self.task_update_prices({symbol: price})
        # default: return metrics
        return self.task_calculate_metrics()

    def run_pipeline(self, symbol: str, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with pipeline: apply allocation if provided and return portfolio snapshot."""
        alloc = shared_state.get('allocation')
        if alloc is not None:
            # Build/update a position sized by allocation and publish position for execution
            build_res = self.task_build_position(symbol, alloc, shared_state.get('portfolio_params', {}))
            # Publish a dollar-sized `position` for the execution agent and share shares
            shared_state['position'] = build_res.get('size', 0.0)
            shared_state['position_shares'] = build_res.get('shares', 0.0)
            # Provide execution parameters (market price if available)
            exec_params = shared_state.get('execution_params', {})
            # Prefer explicit last trade price if provided by infra agent
            if 'market_price' in shared_state and shared_state.get('market_price') is not None:
                exec_params.setdefault('market_price', float(shared_state.get('market_price')))
            else:
                market_data = shared_state.get('market_data') or shared_state.get('final_state', {}).get('market_data')
                if market_data and isinstance(market_data, (list, tuple)) and len(market_data) > 0:
                    exec_params.setdefault('market_price', float(market_data[-1]))
            exec_params.setdefault('order_type', exec_params.get('order_type', 'market'))
            shared_state['execution_params'] = exec_params

        metrics = self.task_calculate_metrics()
        shared_state['portfolio_metrics'] = metrics
        return {'portfolio': metrics}
