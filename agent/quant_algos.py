from typing import List
import numpy as np

class TenQuantAlgos:
    """
    Collection of ten algo skeletons requested.
    Real models can replace the placeholders; the interface returns
    10 signals which are normalized to -1..1 by max-abs scaling.
    """
    def __init__(self):
        self.algo_names = [
            'mean_reversion',
            'momentum',
            'cross_sectional_factor',
            'pairs_trading',
            'ml_return_forecast',
            'order_flow_ml',
            'volatility_arbitrage',
            'risk_parity_alloc',
            'bayesian_portfolio',
            'event_driven'
        ]

    def run_all(self, data: List[float]) -> List[float]:
        if not data:
            return [0.0] * 10
        prices = np.array(data, dtype=float)
        signals = [
            self.mean_reversion(prices),
            self.momentum(prices),
            self.cross_sectional(prices),
            self.pairs_trading_signal(prices),
            self.ml_return_forecast(prices),
            self.order_flow_prediction(prices),
            self.volatility_arbitrage_signal(prices),
            self.risk_parity_signal(prices),
            self.bayesian_optim_signal(prices),
            self.event_driven_signal(prices)
        ]
        max_abs = max(abs(x) for x in signals) or 1.0
        return [float(x) / max_abs for x in signals]

    def mean_reversion(self, prices):
        if len(prices) < 10:
            return 0.0
        window = min(20, len(prices))
        sma = float(np.mean(prices[-window:]))
        std = float(np.std(prices[-window:])) or 1.0
        z = (prices[-1] - sma) / std
        return -z

    def momentum(self, prices):
        if len(prices) < 5:
            return 0.0
        return float((prices[-1] - prices[-5]) / (prices[-5] + 1e-9))

    def cross_sectional(self, prices):
        # Placeholder: single-series proxy of cross-sectional factor
        if len(prices) < 2:
            return 0.0
        ret = (prices[-1] - prices[-2]) / (prices[-2] + 1e-9)
        return float(ret)

    def pairs_trading_signal(self, prices):
        # Needs second series: placeholder
        return 0.0

    def ml_return_forecast(self, prices):
        if len(prices) < 10:
            return 0.0
        y = np.diff(prices[-10:])
        X = np.arange(len(y))
        coef = np.polyfit(X, y, 1)[0]
        return float(coef)

    def order_flow_prediction(self, prices):
        # Placeholder: orderflow not available
        return 0.0

    def volatility_arbitrage_signal(self, prices):
        if len(prices) < 10:
            return 0.0
        returns = np.diff(prices) / (prices[:-1] + 1e-9)
        vol = float(np.std(returns))
        return float(vol)

    def risk_parity_signal(self, prices):
        # Placeholder: needs multi-asset allocations
        return 0.0

    def bayesian_optim_signal(self, prices):
        # Placeholder
        return 0.0

    def event_driven_signal(self, prices):
        # Placeholder (requires event feed)
        return 0.0

