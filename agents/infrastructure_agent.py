import time
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional


try:
    import yfinance as yf
    import pandas as pd
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False


# =======================================================================
# Data Models
# =======================================================================

@dataclass
class DataMetadata:
    symbol: str
    source: str
    interval: str
    lookback: int
    points: int
    exchange: Optional[str] = None

    def as_dict(self):
        return {
            "symbol": self.symbol,
            "source": self.source,
            "interval": self.interval,
            "lookback": self.lookback,
            "points": self.points,
            "exchange": self.exchange
        }


# =======================================================================
# Synthetic generators
# =======================================================================

def _rand_walk(n: int, p0: float = 100.0, vol: float = 0.02):
    p = [p0]
    for _ in range(1, n):
        step = (random.random() - 0.5) * vol * p[-1]
        p.append(max(0.0001, p[-1] + step))
    return [round(x, 4) for x in p]


def _mean_revert(n: int, p0: float = 100.0, mean: float = 100.0, k: float = 0.05, sigma: float = 0.02):
    p = [p0]
    for _ in range(1, n):
        drift = k * (mean - p[-1])
        shock = random.gauss(0, sigma) * p[-1]
        nxt = max(0.0001, p[-1] + drift + shock)
        p.append(round(nxt, 4))
    return p


# =======================================================================
# Cache
# =======================================================================

_CACHE: Dict[str, Any] = {}
_CACHE_TTL = 300  # seconds


def _cache_get(key: str):
    obj = _CACHE.get(key)
    if not obj:
        return None
    data, ts = obj
    if time.time() - ts > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return data


def _cache_set(key: str, value: Any):
    _CACHE[key] = (value, time.time())


# =======================================================================
# Agent
# =======================================================================

class InfrastructureAgent:
    """
    Responsible for data access: market data, synthetic fallback,
    and minimal caching. Future extensions: fundamental data, alt feeds, storage mgmt.
    """

    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger("InfrastructureAgent")
        self.logger.setLevel(log_level)

    def task_load_data(self,
                       symbol: str,
                       lookback: int = 256,
                       interval: str = '1d',
                       synthetic: bool = False,
                       synthetic_mode: str = "walk",
                       force_live: bool = False,
                       **kwargs) -> Dict[str, Any]:
        """
        Return price series dict format:
            {
                'symbol': str,
                'data': list[float],
                'last_price': float,
                'exchange': Optional[str],
                'metadata': dict
            }
        """
        key = f":{symbol}:{lookback}:{interval}:{synthetic_mode}:v2"
        if not force_live:
            cached = _cache_get(key)
            if cached:
                return cached

        # =============================
        # Try live: yfinance
        # =============================
        if YF_AVAILABLE and not synthetic:
            try:
                ticker = yf.Ticker(symbol)
                period = f"{max(1, (lookback // 252) + 1)}y"

                df = ticker.history(period=period, interval=interval, actions=False)
                if df is None or df.empty:
                    raise RuntimeError("yfinance returned no data")

                if "Close" in df.columns:
                    prices = df["Close"].dropna().astype(float).tolist()
                else:
                    prices = df.iloc[:, 0].dropna().astype(float).tolist()

                prices = prices[-lookback:] if len(prices) > lookback else prices
                last = float(prices[-1])

                try:
                    ex = ticker.fast_info.exchange
                except Exception:
                    ex = None

                meta = DataMetadata(symbol, "yfinance", interval, lookback, len(prices), ex)

                result = {
                    "symbol": symbol,
                    "data": prices,
                    "last_price": last,
                    "exchange": ex,
                    "metadata": meta.as_dict()
                }
                _cache_set(key, result)
                return result
            except Exception as e:
                self.logger.warning(f"Live fetch failed for {symbol}: {e}")

        # =============================
        # Fallback: synthetic
        # =============================
        if synthetic_mode == "mean":
            prices = _mean_revert(lookback)
        else:
            prices = _rand_walk(lookback)

        last = float(prices[-1])
        meta = DataMetadata(symbol, "synthetic", interval, lookback, len(prices))

        result = {
            "symbol": symbol,
            "data": prices,
            "last_price": last,
            "exchange": None,
            "metadata": meta.as_dict(),
            "data_source_warning": "synthetic mode active"
        }
        _cache_set(key, result)
        return result

    def run_pipeline(self, symbol: str, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data load as part of pipeline and update shared_state."""
        infra_params = shared_state.get('infra_params', {})
        result = self.task_load_data(
            symbol=symbol,
            lookback=infra_params.get('lookback', 256),
            interval=infra_params.get('interval', '1d'),
            synthetic=infra_params.get('synthetic', False),
            force_live=infra_params.get('force_live', False)
        )
        
        # Update shared state for downstream agents
        shared_state['market_data'] = result.get('data')
        shared_state['data_metadata'] = result.get('metadata', {})
        shared_state['market_price'] = result.get('last_price')
        shared_state['exchange'] = result.get('exchange')
        
        if result.get('data_source_warning'):
            shared_state['data_source_warning'] = result.get('data_source_warning')
        
        return {'infra': result}

    def run_single(self, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single data load operation."""
        return self.task_load_data(
            symbol=symbol,
            lookback=params.get('lookback', 256),
            interval=params.get('interval', '1d'),
            synthetic=params.get('synthetic', False),
            force_live=params.get('force_live', False)
        )

    def execute_task(self, task_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a named task."""
        if task_name == 'load_data':
            return self.task_load_data(**kwargs)
        return {'error': f'Unknown task: {task_name}'}

    def get_status(self) -> str:
        return f"InfrastructureAgent: yfinance={'available' if YF_AVAILABLE else 'unavailable'}"


# =======================================================================
# Quick test
# =======================================================================

if __name__ == "__main__":
    agent = InfrastructureAgent()
    print(agent.task_load_data("AAPL", lookback=30))
    print(agent.task_load_data("MSFT", lookback=30, synthetic=False))