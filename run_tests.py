"""Simple test harness to exercise core modules without external services.

Runs a few smoke checks to ensure agents interoperate and return serializable dicts.
"""
from history_manager import HistoryManager
from agents import InfrastructureAgent, ResearchAgent, BacktestingAgent, RiskAgent, PortfolioAgent, ExecutionAgent
from pipeline import Pipeline

def assert_serializable(obj):
    # quick check: try to convert to str for nested types; failure raises
    try:
        s = str(obj)
    except Exception as e:
        raise AssertionError(f"Not serializable: {e}")


def _json_default(o):
    try:
        import numpy as _np
    except Exception:
        _np = None
    if _np is not None and isinstance(o, _np.generic):
        return o.item()
    if _np is not None and isinstance(o, _np.ndarray):
        return o.tolist()
    if hasattr(o, 'to_dict'):
        return o.to_dict()
    return str(o)


def main():
    history = HistoryManager(path='test_history.json')

    infra = InfrastructureAgent(enable_cache=False)
    research = ResearchAgent(infra)
    backtest = BacktestingAgent(infra)
    risk = RiskAgent()
    portfolio = PortfolioAgent()
    execution = ExecutionAgent()

    agents = {
        'infra': infra,
        'research': research,
        'quant': research.quant,
        'backtest': backtest,
        'risk': risk,
        'portfolio': portfolio,
        'execution': execution
    }

    ordered = ['infra', 'research', 'backtest', 'risk', 'portfolio', 'execution']
    pipeline = Pipeline(ordered, agents, history_manager=history)

    print('Running pipeline smoke test for symbol: TEST')
    res = pipeline.run('TEST', {})

    # basic assertions
    assert isinstance(res, dict), 'Pipeline result must be a dict'
    assert 'final_state' in res, 'Pipeline must include final_state'

    # ensure stages returned serializable dicts
    for k, v in res.items():
        assert_serializable(k)
        assert_serializable(v)

    print('Smoke test passed.')
    # Also print JSON-serialized result for inspection
    import json
    print(json.dumps(res, default=_json_default, indent=2))

if __name__ == '__main__':
    main()
