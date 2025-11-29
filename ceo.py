from typing import Dict, Any, List

class CEO:
    """
    Simple approval + dispatch wrapper. Hook-in history manager to record dispatched jobs.
    """
    def __init__(self, agents: Dict[str, object], history_manager=None):
        self.agents = agents
        self.name = 'LLM_CEO'
        self.history = history_manager

    def approve_and_dispatch(self, targets: List[str], command: str, symbol: str, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        results = {}
        for t in targets:
            agent = self.agents.get(t)
            if not agent:
                results[t] = {'status': 'error', 'message': 'agent not found'}
                continue
            if command in getattr(agent, 'tasks', {}):
                try:
                    results[t] = agent.execute_task(command, **{'symbol': symbol, **params})
                except Exception as e:
                    results[t] = {'status': 'error', 'message': str(e)}
            else:
                try:
                    results[t] = {'status': 'ok', 'result': agent.run_single(symbol, params)}
                except Exception as e:
                    results[t] = {'status': 'error', 'message': str(e)}
        if self.history is not None:
            try:
                self.history.record_run(symbol, 'dispatch', results)
            except:
                pass
        return results
