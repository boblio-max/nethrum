from typing import Dict, Any
from .base_agent import BaseAgent
from .quant_algos import TenQuantAlgos

class ResearchAgent(BaseAgent):
    def __init__(self, infra):
        super().__init__(name='ResearchAgent', agent_type='research')
        self.infra = infra
        self.quant = TenQuantAlgos()
        self.register_task('find_alpha', self.task_find_alpha)
        self.register_task('summarize_signals', self.task_summarize_signals)

    def task_find_alpha(self, symbol: str, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        data_res = self.infra.execute_task('load_data', symbol=symbol, lookback=params.get('lookback', 256))
        data = data_res.get('data') or data_res.get('result', {}).get('data', [])
        signals = self.quant.run_all(data)
        alpha = sum(signals) / len(signals) if signals else 0.0
        out = {'symbol': symbol, 'alpha': round(alpha, 6), 'signals': signals}
        self.last_result = out
        return out

    def task_summarize_signals(self, symbol: str, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        res = self.task_find_alpha(symbol, params)
        signals = res['signals']
        pos = sum(1 for s in signals if s > 0)
        neg = sum(1 for s in signals if s < 0)
        return {'symbol': symbol, 'positive': pos, 'negative': neg, 'alpha': res['alpha']}

    def run_single(self, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.task_find_alpha(symbol, params)

    def run_pipeline(self, symbol: str, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        # Use market_data from shared_state if available (from infra agent)
        data = shared_state.get('market_data', [])
        if data:
            signals = self.quant.run_all(data)
            alpha = sum(signals) / len(signals) if signals else 0.0
            res = {'symbol': symbol, 'alpha': round(alpha, 6), 'signals': signals}
        else:
            res = self.run_single(symbol, shared_state.get('research_params', {}))
        shared_state['alpha'] = res['alpha']
        shared_state['signals'] = res['signals']
        return {'research': res}

