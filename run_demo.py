from history_manager import HistoryManager
from agents.infrastructure_agent import InfrastructureAgent
from agents.research_agent import ResearchAgent
from agents.backtest_agent import BacktestingAgent
from agents.risk_agent import RiskAgent
from agents.portfolio_agent import PortfolioAgent
from agents.execution_agent import ExecutionAgent
from pipeline import Pipeline
import json
try:
    import numpy as _np
except Exception:
    _np = None

def _json_default(o):
    if _np is not None and isinstance(o, _np.generic):
        return o.item()
    if _np is not None and isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)


def main1(Ticker):
    history = HistoryManager()

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

    
    print(f'Running demo pipeline for symbol: {Ticker}')
    # Request real-time last trade prices from infra agent
    params = {'infra_params': {'price_mode': 'last'}}
    res = pipeline.run(f'{Ticker}', params)
    # Pretty-print as JSON using a default serializer for numpy types
    print(json.dumps(res, default=_json_default, indent=2))

    # Also print a concise human-readable summary
    def summarize_pipeline_result(result: dict):
        def _as_float(v):
            try:
                # handle numpy types
                if _np is not None and hasattr(_np, 'generic') and isinstance(v, _np.generic):
                    return float(v.item())
            except Exception:
                pass
            try:
                return float(v)
            except Exception:
                return v

        print('\n--- SUMMARY ---')

        # Market data
        market = result.get('final_state', {}).get('market_data') or result.get('infra', {}).get('result', {}).get('infra', {}).get('data')
        if market:
            try:
                data_len = len(market)
                last_price = _as_float(market[-1])
                mean_price = _as_float(sum(market) / max(1, data_len))
                print(f"Market data: {data_len} points | last={last_price:.4f} | mean={mean_price:.4f}")
            except Exception:
                print('Market data: (could not summarize)')

        # Research
        research = result.get('research', {}).get('result', {}).get('research')
        if research:
            alpha = _as_float(research.get('alpha'))
            signals = research.get('signals', [])
            pos = sum(1 for s in signals if (float(s) if s is not None else 0) > 0)
            neg = sum(1 for s in signals if (float(s) if s is not None else 0) < 0)
            print(f"Research: alpha={alpha:.6f} | signals +{pos}/-{neg} (n={len(signals)})")

        # Backtest
        bt = result.get('backtest', {}).get('result', {}).get('backtest')
        if bt:
            score = _as_float(bt.get('score'))
            metrics = bt.get('metrics', {})
            sharpe = _as_float(metrics.get('sharpe_ratio') or metrics.get('sharpe'))
            dd = _as_float(metrics.get('max_drawdown') or metrics.get('max_dd') or metrics.get('max_drawdown'))
            ret = _as_float(metrics.get('total_returns') or metrics.get('returns') or 0)
            print(f"Backtest: score={score:.6f} | sharpe={sharpe:.3f} | max_dd={dd:.3f} | total_returns={ret:.3f}")

        # Risk
        risk = result.get('risk', {}).get('result', {}).get('risk')
        if risk:
            alloc = _as_float(risk.get('alloc') or risk.get('allocation') or 0)
            var = _as_float(risk.get('var') or 0)
            print(f"Risk: allocation={alloc:.4f} | var={var:.6f}")

        # Portfolio
        port = result.get('portfolio', {}).get('result', {}).get('portfolio')
        if port:
            summary = port.get('summary', {})
            tv = _as_float(summary.get('total_value') or 0)
            cash = _as_float(summary.get('total_cash') or 0)
            equity = _as_float(summary.get('total_equity') or 0)
            npos = int(summary.get('num_positions') or 0)
            print(f"Portfolio: total_value={tv:.2f} | cash={cash:.2f} | equity={equity:.2f} | positions={npos}")

        # Execution
        execn = result.get('execution', {}).get('result', {}).get('execution')
        if execn:
            oid = execn.get('order_id')
            price = _as_float(execn.get('avg_fill_price') or 0)
            filled = _as_float(execn.get('filled_size') or 0)
            status = execn.get('status')
            print(f"Execution: order={oid} | price={price:.4f} | filled={filled} | status={status}")

        print('--- end summary ---\n')

    summarize_pipeline_result(res)


    def generate_action_paragraph(result: dict) -> str:
        def _as_float(v):
            try:
                if _np is not None and hasattr(_np, 'generic') and isinstance(v, _np.generic):
                    return float(v.item())
            except Exception:
                pass
            try:
                return float(v)
            except Exception:
                return 0.0

        market = result.get('final_state', {}).get('market_data') or result.get('infra', {}).get('result', {}).get('infra', {}).get('data')
        last_price = None
        if market:
            try:
                last_price = _as_float(market[-1])
            except Exception:
                last_price = None

        research = result.get('research', {}).get('result', {}).get('research', {}) or {}
        alpha = _as_float(research.get('alpha') or 0)
        signals = research.get('signals') or []
        pos = sum(1 for s in signals if (float(s) if s is not None else 0) > 0)
        neg = sum(1 for s in signals if (float(s) if s is not None else 0) < 0)

        bt = result.get('backtest', {}).get('result', {}).get('backtest', {}) or {}
        score = _as_float(bt.get('score') or 0)
        metrics = bt.get('metrics') or {}
        sharpe = _as_float(metrics.get('sharpe_ratio') or metrics.get('sharpe') or 0)
        total_returns = _as_float(metrics.get('total_returns') or 0)

        risk = result.get('risk', {}).get('result', {}).get('risk') or {}
        alloc = _as_float(risk.get('alloc') or risk.get('allocation') or 0)
        var = _as_float(risk.get('var') or 0)

        port = result.get('portfolio', {}).get('result', {}).get('portfolio') or {}
        summary = port.get('summary') or {}
        equity = _as_float(summary.get('total_equity') or 0)

        # Simple decision rules (informational only)
        recommendation = 'HOLD'
        action_detail = 'No immediate trade suggested.'
        confidence = max(0.0, min(1.0, (score * 10.0 + max(0.0, sharpe)) / 6.0))

        if score > 0.05 and sharpe > 0.5 and alpha > 0:
            recommendation = 'BUY'
            suggested_pct = min(0.1, max(0.01, alloc * 0.5 or 0.02))
            action_detail = f"Consider building a small position (~{suggested_pct*100:.1f}% of portfolio)."
        elif score < 0 or sharpe < 0.15:
            if alloc > 0:
                recommendation = 'REDUCE/SELL'
                suggested_pct = min(1.0, max(0.05, alloc * 0.5))
                action_detail = f"Consider reducing exposure by ~{suggested_pct*100:.1f}% of current allocation."
            else:
                recommendation = 'HOLD'
                action_detail = 'No current allocation to reduce.'
        else:
            # nuanced case: look at signals
            if pos > neg and alpha > 0:
                recommendation = 'BUY (opportunistic)'
                action_detail = 'Signals slightly positive; consider incremental buy if risk budget allows.'
            elif neg > pos and alpha <= 0:
                recommendation = 'SELL (opportunistic)'
                action_detail = 'Signals negative; consider trimming position.'

        parts = []

        # NASDAQ-specific guidance
        exchange = None
        try:
            exchange = result.get('final_state', {}).get('exchange') or result.get('infra', {}).get('result', {}).get('infra', {}).get('metadata', {}).get('exchange')
        except Exception:
            exchange = None

        if exchange and 'nas' in str(exchange).lower():
            # prepend NASDAQ plan
            parts.append("NASDAQ Plan: NASDAQ-listed stocks trade 9:30-16:00 ET; prefer limit/TWAP for large orders to reduce market impact; monitor pre/post-market for earnings-driven moves.")
        parts.append('Automated pipeline summary:')
        if last_price is not None:
            parts.append(f"Last price ${last_price:.4f}.")
        parts.append(f"Backtest score {score:.4f}, Sharpe {sharpe:.3f}, alpha {alpha:.4f}.")
        parts.append(f"Portfolio equity ${equity:,.2f}, current allocation {alloc:.4f}.")
        parts.append(f"Signals: +{pos}/-{neg}; risk var={var:.6f}.")
        parts.append(f"Recommendation: {recommendation}. {action_detail}")
        parts.append(f"Confidence: {confidence*100:.0f}% (algorithmic, not financial advice).")
        #parts.append('This summary is auto-generated and is not financial advice.')

        return ' '.join(parts)

    paragraph = generate_action_paragraph(res)
    print('\n--- Summary ---')
    print(paragraph)
    print('--- end action paragraph ---\n')

if __name__ == '__main__':
    main1(input("Enter the ticker you want to trace: "))
