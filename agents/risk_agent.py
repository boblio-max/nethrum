from typing import Dict, Any
from .base_agent import BaseAgent

class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__(name='RiskAgent', agent_type='risk')
        self.register_task('assess', self.task_assess)

    def task_assess(self, alpha: float, backtest_score: float, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        factor = params.get('sizing_factor', 1.0)
        raw = alpha * factor * (1 + backtest_score * 0.01)
        alloc = max(min(raw, 1.0), -1.0)
        var = max(0.0, abs(alloc) * params.get('vol_est', 0.05))
        out = {'alloc': round(alloc, 6), 'var': round(var, 6)}
        self.last_result = out
        return out

    def run_single(self, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.task_assess(params.get('alpha', 0.0), params.get('backtest_score', 0.0), params)

    def run_pipeline(self, symbol: str, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        alpha = shared_state.get('alpha', 0.0)
        score = shared_state.get('backtest_score', 0.0)
        res = self.task_assess(alpha, score, shared_state.get('risk_params', {}))
        shared_state['allocation'] = res['alloc']
        shared_state['var'] = res['var']
        return {'risk': res}

