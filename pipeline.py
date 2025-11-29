from typing import Dict, Any, List
import json
try:
    import numpy as _np
except Exception:
    _np = None
class Pipeline:
    def __init__(self, ordered_agent_names: List[str], agents: Dict[str, object], history_manager=None):
        self.ordered = ordered_agent_names
        self.agents = agents
        self.history = history_manager

    def run(self, symbol: str, params: Dict[str, Any]={}) -> Dict[str, Any]:
        # shared_state is passed between agents; include pipeline-level params
        shared_state: Dict[str, Any] = {}
        try:
            # make infra_params available for InfrastructureAgent.run_pipeline
            shared_state['infra_params'] = params or {}
        except Exception:
            shared_state['infra_params'] = {}
        results = {}
        for name in self.ordered:
            agent = self.agents.get(name)
            if not agent:
                results[name] = {'status': 'error', 'message': 'agent missing'}
                continue
            try:
                r = agent.run_pipeline(symbol, shared_state)

                # Use a local serializer to ensure all nested numpy types and
                # other non-serializable objects are converted to native types
                def _make_serializable_local(obj):
                    if obj is None or isinstance(obj, (str, bool, int, float)):
                        return obj
                    if _np is not None and isinstance(obj, _np.generic):
                        return obj.item()
                    if _np is not None and isinstance(obj, _np.ndarray):
                        try:
                            return obj.tolist()
                        except Exception:
                            pass
                    if isinstance(obj, dict):
                        return {str(k): _make_serializable_local(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple, set)):
                        return [_make_serializable_local(x) for x in obj]
                    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                        try:
                            return _make_serializable_local(obj.to_dict())
                        except Exception:
                            pass
                    try:
                        json.dumps(obj)
                        return obj
                    except Exception:
                        try:
                            return str(obj)
                        except Exception:
                            return None

                r = _make_serializable_local(r)

                def _generate_paragraph_for_agent(agent_name: str, data: Any) -> str:
                    """Create a short plain-English paragraph summarizing common fields.

                    This is a lightweight, deterministic summarizer that looks for
                    common keys produced by agents (alpha, signals, score, metrics,
                    allocation, portfolio summary, execution) and creates a single
                    sentence paragraph describing the outcome.
                    """
                    try:
                        parts = []
                        # If data is wrapped (e.g., {'infra': {...}}) drill down
                        if isinstance(data, dict) and len(data) == 1:
                            # pick the single nested value
                            k = next(iter(data.keys()))
                            inner = data.get(k) or {}
                        else:
                            inner = data or {}

                        # backtest-style
                        score = inner.get('score') or inner.get('backtest_score')
                        if score is not None:
                            parts.append(f"Backtest score {float(score):.4f}")

                        metrics = inner.get('metrics') or inner.get('all_metrics') or {}
                        if metrics and isinstance(metrics, dict):
                            sharpe = metrics.get('sharpe_ratio') or metrics.get('sharpe')
                            if sharpe is not None:
                                parts.append(f"Sharpe {float(sharpe):.3f}")

                        # research-style
                        research = inner.get('research') or inner
                        alpha = research.get('alpha') if isinstance(research, dict) else None
                        if alpha is not None:
                            parts.append(f"alpha {float(alpha):.4f}")

                        # signals
                        signals = inner.get('signals')
                        if isinstance(signals, (list, tuple)):
                            pos = sum(1 for s in signals if float(s) > 0)
                            neg = sum(1 for s in signals if float(s) < 0)
                            parts.append(f"signals +{pos}/-{neg} (n={len(signals)})")

                        # risk / allocation
                        alloc = inner.get('alloc') or inner.get('allocation')
                        if alloc is not None:
                            parts.append(f"allocation {float(alloc):.4f}")

                        # portfolio summary
                        portfolio = inner.get('portfolio') or inner
                        if isinstance(portfolio, dict):
                            summary = portfolio.get('summary') or portfolio
                            tv = summary.get('total_value') if isinstance(summary, dict) else None
                            if tv is not None:
                                parts.append(f"portfolio value ${float(tv):,.0f}")

                        # execution
                        execn = inner.get('execution') or inner
                        if isinstance(execn, dict):
                            oid = execn.get('order_id')
                            price = execn.get('avg_fill_price') or execn.get('execution_price')
                            if oid or price is not None:
                                seg = []
                                if oid:
                                    seg.append(f"order {oid}")
                                if price is not None:
                                    seg.append(f"price {float(price):.4f}")
                                parts.append(', '.join(seg))

                        if not parts:
                            return f"{agent_name}: no notable metrics produced."

                        return f"{agent_name}: " + "; ".join(parts) + '.'
                    except Exception:
                        return f"{agent_name}: (could not summarize output)"

                paragraph = _generate_paragraph_for_agent(name, r)
                results[name] = {'status':'ok','result':r, 'paragraph': paragraph}
            except Exception as e:
                results[name] = {'status':'error','message':str(e)}
        results['final_state'] = shared_state
        if self.history is not None:
            try:
                self.history.record_run(symbol, 'pipeline', results)
            except:
                pass
        return results
