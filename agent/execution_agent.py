from typing import Dict, Any, Optional, List, Tuple
from .base_agent import BaseAgent, ExecutionResult, TaskPriority, ExecutionMode, AgentState
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
import random
import numpy as np
from collections import defaultdict
import logging

class OrderType(Enum):
    """Order execution types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"

class OrderStatus(Enum):
    """Order lifecycle status"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderSide(Enum):
    """Order side/direction"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Comprehensive order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    size: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Execution details
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    # Cost tracking
    commission: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    
    # Metadata
    parent_order_id: Optional[str] = None
    strategy: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    @property
    def fill_percentage(self) -> float:
        return (self.filled_size / self.size * 100) if self.size > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'order_type': self.order_type.value,
            'status': self.status.value,
            'filled_size': round(self.filled_size, 6),
            'remaining_size': round(self.remaining_size, 6),
            'avg_fill_price': round(self.avg_fill_price, 4),
            'fill_percentage': round(self.fill_percentage, 2),
            'commission': round(self.commission, 4),
            'slippage_bps': round(self.slippage_bps, 2),
            'market_impact_bps': round(self.market_impact_bps, 2),
            'created_at': self.created_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'strategy': self.strategy,
            'tags': self.tags
        }

@dataclass
class ExecutionReport:
    """Comprehensive execution analytics"""
    total_orders: int = 0
    total_volume: float = 0.0
    total_notional: float = 0.0
    total_commission: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_fill_time: float = 0.0
    fill_rate: float = 0.0
    
    # By status
    filled_orders: int = 0
    partial_orders: int = 0
    cancelled_orders: int = 0
    rejected_orders: int = 0
    
    # By side
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    
    # Performance
    best_execution_bps: float = 0.0
    worst_execution_bps: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': {
                'total_orders': self.total_orders,
                'total_volume': round(self.total_volume, 2),
                'total_notional': round(self.total_notional, 2),
                'total_commission': round(self.total_commission, 4),
                'avg_slippage_bps': round(self.avg_slippage_bps, 2),
                'fill_rate': round(self.fill_rate, 2)
            },
            'by_status': {
                'filled': self.filled_orders,
                'partial': self.partial_orders,
                'cancelled': self.cancelled_orders,
                'rejected': self.rejected_orders
            },
            'by_side': {
                'buy_volume': round(self.buy_volume, 2),
                'sell_volume': round(self.sell_volume, 2),
                'net_position': round(self.buy_volume - self.sell_volume, 2)
            },
            'performance': {
                'best_execution_bps': round(self.best_execution_bps, 2),
                'worst_execution_bps': round(self.worst_execution_bps, 2),
                'avg_fill_time': round(self.avg_fill_time, 2)
            }
        }

class MarketSimulator:
    """Advanced market microstructure simulation"""
    
    def __init__(
        self,
        base_volatility: float = 0.02,
        liquidity_depth: float = 1000000.0,
        bid_ask_spread_bps: float = 5.0
    ):
        self.base_volatility = base_volatility
        self.liquidity_depth = liquidity_depth
        self.bid_ask_spread_bps = bid_ask_spread_bps
        self.price_cache: Dict[str, Tuple[float, float]] = {}  # symbol -> (price, timestamp)
        
    def get_market_price(self, symbol: str, base_price: Optional[float] = None) -> Tuple[float, float]:
        """Get bid/ask prices with realistic microstructure"""
        now = time.time()
        
        # Check cache (prices valid for 1 second)
        if symbol in self.price_cache:
            cached_price, cached_time = self.price_cache[symbol]
            if now - cached_time < 1.0:
                mid = cached_price
            else:
                mid = self._update_price(cached_price)
        else:
            mid = base_price if base_price else 100.0 + random.uniform(-10, 10)
        
        # Add bid-ask spread
        half_spread = mid * (self.bid_ask_spread_bps / 10000.0) / 2
        bid = mid - half_spread
        ask = mid + half_spread
        
        self.price_cache[symbol] = (mid, now)
        return bid, ask
    
    def _update_price(self, current_price: float) -> float:
        """Simulate price movement with GBM"""
        dt = 1.0 / 252 / 390  # ~1 minute in trading days
        drift = 0.0  # Assume zero drift for simplicity
        shock = random.gauss(0, 1)
        return current_price * np.exp((drift - 0.5 * self.base_volatility**2) * dt + 
                                     self.base_volatility * np.sqrt(dt) * shock)
    
    def calculate_slippage(self, size: float, side: OrderSide, order_type: OrderType) -> float:
        """Calculate realistic slippage based on order size and type"""
        abs_size = abs(size)
        
        # Base slippage from liquidity
        liquidity_impact = (abs_size / self.liquidity_depth) ** 0.7
        
        # Order type adjustments
        if order_type == OrderType.MARKET:
            urgency_multiplier = 1.0
        elif order_type == OrderType.LIMIT:
            urgency_multiplier = 0.3
        elif order_type in [OrderType.TWAP, OrderType.VWAP]:
            urgency_multiplier = 0.5
        else:
            urgency_multiplier = 0.7
        
        # Random component for realism
        random_factor = random.uniform(0.8, 1.2)
        
        slippage_bps = liquidity_impact * urgency_multiplier * random_factor * 50  # Scale to bps
        return min(slippage_bps, 100.0)  # Cap at 100 bps
    
    def calculate_market_impact(self, size: float, order_type: OrderType) -> float:
        """Calculate permanent market impact"""
        abs_size = abs(size)
        
        if order_type in [OrderType.TWAP, OrderType.VWAP, OrderType.ICEBERG]:
            # Reduced impact for smart order routing
            impact = (abs_size / self.liquidity_depth) ** 0.5 * 10
        else:
            impact = (abs_size / self.liquidity_depth) ** 0.6 * 15
        
        return min(impact, 50.0)  # Cap at 50 bps

class ExecutionAgent(BaseAgent):
    """
    Advanced execution agent with realistic market simulation.
    
    Features:
    - Multiple order types (market, limit, TWAP, VWAP, iceberg)
    - Realistic slippage and market impact models
    - Order lifecycle management
    - Comprehensive execution analytics
    - Risk limits and pre-trade checks
    - Smart order routing simulation
    - Parent-child order relationships
    """
    
    def __init__(
        self,
        commission_bps: float = 5.0,
        max_order_size: float = 1000000.0,
        max_position_size: float = 10000000.0,
        enable_risk_checks: bool = True,
        max_workers: int = 4
    ):
        super().__init__(
            name='ExecutionAgent',
            agent_type='execution',
            max_workers=max_workers
        )
        
        # Configuration
        self.commission_bps = commission_bps
        self.max_order_size = max_order_size
        self.max_position_size = max_position_size
        self.enable_risk_checks = enable_risk_checks
        
        # State tracking
        self.orders: Dict[str, Order] = {}
        self.order_log: List[Order] = []
        self.positions: Dict[str, float] = defaultdict(float)
        self._order_counter = 0
        
        # Market simulation
        self.market_sim = MarketSimulator()
        
        # Register tasks
        self.register_task(
            'simulate_execution',
            self.task_simulate_execution,
            priority=TaskPriority.HIGH,
            description="Execute order with market simulation"
        )
        self.register_task(
            'execute_market_order',
            self.task_execute_market_order,
            priority=TaskPriority.CRITICAL,
            description="Execute market order immediately"
        )
        self.register_task(
            'execute_limit_order',
            self.task_execute_limit_order,
            priority=TaskPriority.NORMAL,
            description="Execute limit order with price checking"
        )
        self.register_task(
            'execute_twap',
            self.task_execute_twap,
            priority=TaskPriority.NORMAL,
            mode=ExecutionMode.ASYNC,
            description="Execute TWAP order over time"
        )
        self.register_task(
            'cancel_order',
            self.task_cancel_order,
            priority=TaskPriority.HIGH,
            description="Cancel pending order"
        )
        self.register_task(
            'get_execution_report',
            self.task_get_execution_report,
            description="Generate execution analytics"
        )
        
        self.logger.info(f"ExecutionAgent initialized with commission={commission_bps}bps")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self._order_counter += 1
        return f"ORD_{int(time.time())}_{self._order_counter:06d}"
    
    def _pre_trade_checks(self, symbol: str, size: float) -> Tuple[bool, Optional[str]]:
        """Run pre-trade risk checks"""
        if not self.enable_risk_checks:
            return True, None
        
        # Check order size
        if abs(size) > self.max_order_size:
            return False, f"Order size {abs(size)} exceeds limit {self.max_order_size}"
        
        # Check resulting position
        current_position = self.positions.get(symbol, 0.0)
        new_position = current_position + size
        if abs(new_position) > self.max_position_size:
            return False, f"Position {abs(new_position)} would exceed limit {self.max_position_size}"
        
        return True, None
    
    def _calculate_commission(self, notional: float) -> float:
        """Calculate commission based on notional value"""
        return notional * (self.commission_bps / 10000.0)
    
    def _update_position(self, symbol: str, size: float):
        """Update position tracking"""
        self.positions[symbol] += size
    
    def task_simulate_execution(
        self,
        symbol: str,
        size: float,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Legacy interface for backward compatibility"""
        params = params or {}
        
        side = OrderSide.BUY if size > 0 else OrderSide.SELL
        order_type = OrderType[params.get('order_type', 'MARKET').upper()]
        
        result = self._execute_order(
            symbol=symbol,
            size=abs(size),
            side=side,
            order_type=order_type,
            market_price=params.get('market_price'),
            strategy=params.get('strategy', '')
        )
        
        # Return legacy format
        return {
            'symbol': symbol,
            'size': size,
            'exec_price': result['avg_fill_price'],
            'slippage_pct': result['slippage_bps'] / 10000.0,
            'ts': time.time()
        }
    
    def task_execute_market_order(
        self,
        symbol: str,
        size: float,
        side: str,
        strategy: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute market order immediately"""
        side_enum = OrderSide[side.upper()]
        return self._execute_order(
            symbol=symbol,
            size=size,
            side=side_enum,
            order_type=OrderType.MARKET,
            strategy=strategy
        )
    
    def task_execute_limit_order(
        self,
        symbol: str,
        size: float,
        side: str,
        limit_price: float,
        strategy: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute limit order with price checking"""
        side_enum = OrderSide[side.upper()]
        return self._execute_order(
            symbol=symbol,
            size=size,
            side=side_enum,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            strategy=strategy
        )
    
    def task_execute_twap(
        self,
        symbol: str,
        size: float,
        side: str,
        duration_seconds: float = 300.0,
        num_slices: int = 10,
        strategy: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """Execute TWAP order by splitting into time slices"""
        side_enum = OrderSide[side.upper()]
        
        # Create parent order
        parent_id = self._generate_order_id()
        slice_size = size / num_slices
        
        total_filled = 0.0
        total_cost = 0.0
        child_orders = []
        
        for i in range(num_slices):
            # Simulate time delay
            if i > 0:
                time.sleep(duration_seconds / num_slices / 100)  # Scaled for testing
            
            result = self._execute_order(
                symbol=symbol,
                size=slice_size,
                side=side_enum,
                order_type=OrderType.TWAP,
                parent_order_id=parent_id,
                strategy=strategy
            )
            
            total_filled += result['filled_size']
            total_cost += result['avg_fill_price'] * result['filled_size']
            child_orders.append(result['order_id'])
        
        avg_price = total_cost / total_filled if total_filled > 0 else 0.0
        
        return {
            'order_id': parent_id,
            'symbol': symbol,
            'order_type': 'twap',
            'total_size': size,
            'filled_size': total_filled,
            'avg_fill_price': round(avg_price, 4),
            'num_slices': num_slices,
            'child_orders': child_orders
        }
    
    def _execute_order(
        self,
        symbol: str,
        size: float,
        side: OrderSide,
        order_type: OrderType,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        market_price: Optional[float] = None,
        parent_order_id: Optional[str] = None,
        strategy: str = ""
    ) -> Dict[str, Any]:
        """Core order execution logic"""
        
        # Pre-trade checks
        signed_size = size if side == OrderSide.BUY else -size
        passed, error = self._pre_trade_checks(symbol, signed_size)
        if not passed:
            self.logger.warning(f"Pre-trade check failed: {error}")
            order = Order(
                order_id=self._generate_order_id(),
                symbol=symbol,
                side=side,
                size=size,
                order_type=order_type,
                status=OrderStatus.REJECTED,
                strategy=strategy,
                tags={'rejection_reason': error}
            )
            self.orders[order.order_id] = order
            return order.to_dict()
        
        # Get market prices
        bid, ask = self.market_sim.get_market_price(symbol, market_price)
        
        # Determine execution price based on order type
        if order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("Limit price required for limit orders")
            
            # Check if limit price is marketable
            if side == OrderSide.BUY and limit_price < ask:
                execution_price = ask
                status = OrderStatus.PENDING
            elif side == OrderSide.SELL and limit_price > bid:
                execution_price = bid
                status = OrderStatus.PENDING
            else:
                execution_price = limit_price
                status = OrderStatus.FILLED
        else:
            # Market order or TWAP/VWAP
            execution_price = ask if side == OrderSide.BUY else bid
            status = OrderStatus.FILLED
        
        # Calculate costs
        slippage_bps = self.market_sim.calculate_slippage(size, side, order_type)
        market_impact_bps = self.market_sim.calculate_market_impact(size, order_type)
        
        # Apply slippage
        if side == OrderSide.BUY:
            final_price = execution_price * (1 + slippage_bps / 10000.0)
        else:
            final_price = execution_price * (1 - slippage_bps / 10000.0)
        
        notional = final_price * size
        commission = self._calculate_commission(notional)
        
        # Create order
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            size=size,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            status=status,
            filled_size=size if status == OrderStatus.FILLED else 0.0,
            avg_fill_price=final_price,
            commission=commission,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            parent_order_id=parent_order_id,
            strategy=strategy,
            filled_at=datetime.now() if status == OrderStatus.FILLED else None
        )
        
        # Update state
        self.orders[order.order_id] = order
        self.order_log.append(order)
        
        if status == OrderStatus.FILLED:
            self._update_position(symbol, signed_size)
        
        self.logger.info(
            f"Order executed: {order.order_id} | {symbol} {side.value} {size} @ {final_price:.4f} "
            f"| slippage={slippage_bps:.2f}bps"
        )
        
        return order.to_dict()
    
    def task_cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order"""
        if order_id not in self.orders:
            return {'status': 'error', 'message': f'Order {order_id} not found'}
        
        order = self.orders[order_id]
        
        if order.is_complete:
            return {'status': 'error', 'message': f'Order {order_id} already complete'}
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        
        self.logger.info(f"Order cancelled: {order_id}")
        return {'status': 'ok', 'order': order.to_dict()}
    
    def task_get_execution_report(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive execution analytics"""
        
        # Filter orders
        orders = self.order_log
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if strategy:
            orders = [o for o in orders if o.strategy == strategy]
        if start_time:
            orders = [o for o in orders if o.created_at >= start_time]
        if end_time:
            orders = [o for o in orders if o.created_at <= end_time]
        
        if not orders:
            return ExecutionReport().to_dict()
        
        # Calculate metrics
        report = ExecutionReport()
        report.total_orders = len(orders)
        
        slippages = []
        fill_times = []
        
        for order in orders:
            report.total_volume += order.filled_size
            report.total_notional += order.avg_fill_price * order.filled_size
            report.total_commission += order.commission
            
            if order.status == OrderStatus.FILLED:
                report.filled_orders += 1
                slippages.append(order.slippage_bps)
                
                if order.filled_at:
                    fill_time = (order.filled_at - order.created_at).total_seconds()
                    fill_times.append(fill_time)
            elif order.status == OrderStatus.PARTIAL:
                report.partial_orders += 1
            elif order.status == OrderStatus.CANCELLED:
                report.cancelled_orders += 1
            elif order.status == OrderStatus.REJECTED:
                report.rejected_orders += 1
            
            if order.side == OrderSide.BUY:
                report.buy_volume += order.filled_size
            else:
                report.sell_volume += order.filled_size
        
        report.fill_rate = (report.filled_orders / report.total_orders * 100) if report.total_orders > 0 else 0.0
        report.avg_slippage_bps = np.mean(slippages) if slippages else 0.0
        report.best_execution_bps = min(slippages) if slippages else 0.0
        report.worst_execution_bps = max(slippages) if slippages else 0.0
        report.avg_fill_time = np.mean(fill_times) if fill_times else 0.0
        
        return report.to_dict()
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions across all symbols"""
        return dict(self.positions)
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific order details"""
        order = self.orders.get(order_id)
        return order.to_dict() if order else None
    
    def get_orders_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all orders for a specific symbol"""
        return [o.to_dict() for o in self.order_log if o.symbol == symbol]
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all pending orders"""
        return [o.to_dict() for o in self.orders.values() if not o.is_complete]
    
    def run_single(self, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single order operation"""
        order_type = params.get('order_type', 'market').lower()
        
        if order_type == 'market':
            return self.task_execute_market_order(
                symbol=symbol,
                size=params.get('size', 0.0),
                side=params.get('side', 'buy'),
                strategy=params.get('strategy', '')
            )
        elif order_type == 'limit':
            return self.task_execute_limit_order(
                symbol=symbol,
                size=params.get('size', 0.0),
                side=params.get('side', 'buy'),
                limit_price=params['limit_price'],
                strategy=params.get('strategy', '')
            )
        elif order_type == 'twap':
            return self.task_execute_twap(
                symbol=symbol,
                size=params.get('size', 0.0),
                side=params.get('side', 'buy'),
                duration_seconds=params.get('duration', 300.0),
                num_slices=params.get('slices', 10),
                strategy=params.get('strategy', '')
            )
        else:
            return self.task_simulate_execution(symbol, params.get('size', 0.0), params)
    
    def run_pipeline(self, symbol: str, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order as part of pipeline"""
        size = shared_state.get('position', 0.0)
        side = OrderSide.BUY if size > 0 else OrderSide.SELL
        
        execution_params = shared_state.get('execution_params', {})
        order_type = OrderType[execution_params.get('order_type', 'MARKET').upper()]
        
        result = self._execute_order(
            symbol=symbol,
            size=abs(size),
            side=side,
            order_type=order_type,
            strategy=shared_state.get('strategy', ''),
            market_price=execution_params.get('market_price')
        )
        
        # Update shared state
        shared_state['execution'] = result
        shared_state['execution_price'] = result['avg_fill_price']
        shared_state['execution_cost'] = result['commission']
        shared_state['total_slippage_bps'] = result['slippage_bps']
        
        return {'execution': result}
    
    def reset_simulation(self):
        """Reset simulation state (useful for backtesting)"""
        self.orders.clear()
        self.order_log.clear()
        self.positions.clear()
        self._order_counter = 0
        self.market_sim.price_cache.clear()
        self.logger.info("Execution simulation reset")