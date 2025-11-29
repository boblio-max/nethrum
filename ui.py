import os
from pdb import run
from history_manager import HistoryManager
from agents.infrastructure_agent import InfrastructureAgent
from agents.research_agent import ResearchAgent
from agents.backtest_agent import BacktestingAgent
from agents.risk_agent import RiskAgent
from agents.portfolio_agent import PortfolioAgent
from agents.execution_agent import ExecutionAgent
from ceo import CEO
from secretary import Secretary
from pipeline import Pipeline
from g4f.client import Client
import run_demo
import json
try:
    import numpy as _np
except Exception:
    _np = None
import math

# plotting
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False
class ConsoleDashboard:
    def __init__(self, llm_key: str = None):
        # If `llm_key` is provided programmatically, do not prompt the user.
        
        if llm_key is None:
            # allow skipping interactive prompt in automated runs
            if os.environ.get('SKIP_LLM_PROMPT') == '1':
                llm_key1 = os.environ.get('LLM_KEY', '')
            else:
                llm_key = ""
                #llm_key1 = input("Enter LLM key (or leave blank to skip): ")
                
        else:
            llm_key1 = llm_key
        self.clear = lambda: os.system('cls' if os.name == 'nt' else 'clear')
        self.history = HistoryManager()
        # instantiate agents
        self.infra = InfrastructureAgent()
        self.research = ResearchAgent(self.infra)
        self.backtest = BacktestingAgent(self.infra)
        self.risk = RiskAgent()
        self.portfolio = PortfolioAgent()
        self.execution = ExecutionAgent()
        # registry (quant is a helper module inside research)
        self.agents = {
            'infra': self.infra,
            'research': self.research,
            'quant': self.research.quant,
            'backtest': self.backtest,
            'risk': self.risk,
            'portfolio': self.portfolio,
            'execution': self.execution
        }
        self.ordered = ['infra', 'research', 'backtest', 'risk', 'portfolio', 'execution']
        self.pipeline = Pipeline(self.ordered, self.agents, history_manager=self.history)
        self.ceo = CEO(self.agents, history_manager=self.history)
        self.secretary = Secretary(llm_provider='openai' if os.environ.get('OPENAI_API_KEY') else "None")
        # UI defaults
        self.default_price_mode = os.environ.get('DEFAULT_PRICE_MODE', 'last')

    def banner(self):
        print('='*80)
        print(' ' * 20 + 'NETHRUM - CONSOLE DASHBOARD (v4)')
        print('='*80)

    def help(self):
        print('\nCommands:')
        print('  secretary: <text>              -> brainstorm/parse commands (LLM if configured)')
        print('  secretary.llm: <prompt>        -> call LLM directly for brainstorming (requires OPENAI_API_KEY)')
        print('  run pipeline on <TICKER>       -> runs full pipeline')
        print('  <agent>: <task or message>     -> send task to single agent, e.g., research: find_alpha AAPL')
        print('  history                         -> view recent runs')
        print('  status                          -> show status of all agents')
        print('  tasks                           -> list available tasks for each agent')
        print('  clear / exit                    -> obvious')
        print()

    def list_tasks(self):
        for k,a in self.agents.items():
            print(f"[{k}] tasks: {list(getattr(a,'tasks',{}).keys())}")

    def show_status(self):
        for k,a in self.agents.items():
            print(f"--- {k} ---")
            try:
                print(a.get_status())
            except Exception as e:
                print('error getting status', e)

    def view_history(self):
        runs = self.history.get_runs(20)
        for r in runs:
            print(r)

    def _json_default(self, o):
        if _np is not None and isinstance(o, _np.generic):
            return o.item()
        if _np is not None and isinstance(o, _np.ndarray):
            return o.tolist()
        return str(o)

    def _position_figure(self, fig, x: int, y: int):
        """Attempt to position a matplotlib figure window at (x,y). Non-fatal."""
        if not PLOTTING_AVAILABLE:
            return
        try:
            mgr = plt.get_current_fig_manager()
            if hasattr(mgr, 'window'):
                try:
                    mgr.window.wm_geometry(f'+{x}+{y}')
                except Exception:
                    try:
                        mgr.window.setGeometry(x, y, mgr.window.width(), mgr.window.height())
                    except Exception:
                        pass
            elif hasattr(mgr, 'canvas') and hasattr(mgr.canvas, 'manager'):
                w = mgr.canvas.manager
                try:
                    w.window.wm_geometry(f'+{x}+{y}')
                except Exception:
                    pass
        except Exception:
            pass

    def _normalize_exchange(self, exchange_raw):
        """Return a normalized exchange string (e.g., NASDAQ, NYSE) from raw metadata.

        Uses simple heuristics and a small curated mapping for common tickers.
        """
        if not exchange_raw:
            return None
        e = str(exchange_raw).strip()
        if not e:
            return None
        le = e.lower()
        if 'nas' in le or 'nms' in le:
            return 'NASDAQ'
        if 'nyse' in le:
            return 'NYSE'
        # common yfinance codes
        if le in ('nasdaq', 'nyse'):
            return e.upper()
        return e

    def plot_market_data(self, final_state):
        # Single-window polished dashboard
        if not PLOTTING_AVAILABLE:
            print('Plotting not available: matplotlib not installed or backend unavailable.')
            return

        prices = final_state.get('market_data') or []
        if not prices:
            print('No market data available for plotting.')
            return

        signals = final_state.get('signals') or []
        exec_price = final_state.get('execution_price')
        symbol = final_state.get('data_metadata', {}).get('symbol') or final_state.get('execution', {}).get('symbol') or ''
        exchange = self._normalize_exchange(final_state.get('exchange') or final_state.get('data_metadata', {}).get('exchange'))
        data_source_warning = final_state.get('data_source_warning')

        def sma(arr, window):
            window = max(1, int(window))
            return [sum(arr[max(0, i-window+1):i+1]) / (i - max(0, i-window+1) + 1) for i in range(len(arr))]

        ma20 = sma(prices, 20)
        ma50 = sma(prices, 50)
        x = list(range(len(prices)))

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax_price = axes[0, 0]
        ax_info = axes[0, 1]
        ax_hist = axes[1, 0]
        ax_zoom = axes[1, 1]

        # Price plot
        ax_price.plot(x, prices, label='Price', color='tab:blue')
        ax_price.plot(x, ma20, label='MA20', color='tab:orange', linewidth=1)
        ax_price.plot(x, ma50, label='MA50', color='tab:green', linewidth=1)
        buys = [i for i, s in enumerate(signals) if s and s > 0.5]
        sells = [i for i, s in enumerate(signals) if s and s < -0.5]
        if buys:
            ax_price.scatter(buys, [prices[i] for i in buys], marker='^', color='green', label='Buy', zorder=5)
        if sells:
            ax_price.scatter(sells, [prices[i] for i in sells], marker='v', color='red', label='Sell', zorder=5)
        if exec_price is not None:
            ax_price.axhline(exec_price, color='purple', linestyle='--', label=f'Execution {exec_price:.2f}')
        ax_price.set_title(f'{symbol} Price Series')
        ax_price.set_ylabel('Price')
        ax_price.legend(loc='upper left')

        # Info panel
        ax_info.axis('off')
        info_lines = []
        md = final_state.get('data_metadata') or {}
        info_lines.append(f"Symbol: {symbol}")
        info_lines.append(f"Exchange: {exchange or 'Unknown'}")
        info_lines.append(f"Source: {md.get('source')}")
        info_lines.append(f"Points: {md.get('actual_points')}")
        if data_source_warning:
            info_lines.append(f"WARNING: data source fallback -> {data_source_warning}")
        if exec_price is not None:
            info_lines.append(f"Execution price: {exec_price:.2f}")
        info_text = '\n'.join(info_lines)
        ax_info.text(0.01, 0.99, info_text, va='top', ha='left', fontsize=10, family='monospace')

        # Returns histogram
        returns = []
        for i in range(1, len(prices)):
            try:
                r = (prices[i] - prices[i-1]) / max(1e-12, prices[i-1])
                returns.append(r)
            except Exception:
                continue
        ax_hist.hist(returns, bins=40, color='tab:gray', edgecolor='black')
        ax_hist.set_title('Returns Distribution')
        ax_hist.set_xlabel('Return')

        # Zoomed recent price area
        recent_n = min(60, len(prices))
        ax_zoom.plot(x[-recent_n:], prices[-recent_n:], color='tab:blue')
        ax_zoom.plot(x[-recent_n:], ma20[-recent_n:], color='tab:orange', linewidth=1)
        ax_zoom.set_title(f'Last {recent_n} Points')

        fig.tight_layout()
        plt.show(block=False)
        try:
            self._position_figure(fig, 30, 30)
        except Exception:
            pass

    def summarize_pipeline_result(self, result: dict):
        def _as_float(v):
            try:
                if _np is not None and hasattr(_np, 'generic') and isinstance(v, _np.generic):
                    return float(v.item())
            except Exception:
                pass
            try:
                return float(v)
            except Exception:
                return v

        print('\n--- SUMMARY ---')

        market = result.get('final_state', {}).get('market_data') or result.get('infra', {}).get('result', {}).get('infra', {}).get('data')
        if market:
            try:
                data_len = len(market)
                last_price = _as_float(market[-1])
                mean_price = _as_float(sum(market) / max(1, data_len))
                print(f"Market data: {data_len} points | last={last_price:.4f} | mean={mean_price:.4f}")
            except Exception:
                print('Market data: (could not summarize)')

        research = result.get('research', {}).get('result', {}).get('research')
        if research:
            alpha = _as_float(research.get('alpha'))
            signals = research.get('signals', [])
            pos = sum(1 for s in signals if (float(s) if s is not None else 0) > 0)
            neg = sum(1 for s in signals if (float(s) if s is not None else 0) < 0)
            print(f"Research: alpha={alpha:.6f} | signals +{pos}/-{neg} (n={len(signals)})")

        bt = result.get('backtest', {}).get('result', {}).get('backtest')
        if bt:
            score = _as_float(bt.get('score'))
            metrics = bt.get('metrics', {})
            sharpe = _as_float(metrics.get('sharpe_ratio') or metrics.get('sharpe'))
            dd = _as_float(metrics.get('max_drawdown') or metrics.get('max_dd') or metrics.get('max_drawdown'))
            ret = _as_float(metrics.get('total_returns') or metrics.get('returns') or 0)
            print(f"Backtest: score={score:.6f} | sharpe={sharpe:.3f} | max_dd={dd:.3f} | total_returns={ret:.3f}")

        risk = result.get('risk', {}).get('result', {}).get('risk')
        if risk:
            alloc = _as_float(risk.get('alloc') or risk.get('allocation') or 0)
            var = _as_float(risk.get('var') or 0)
            print(f"Risk: allocation={alloc:.4f} | var={var:.6f}")

        port = result.get('portfolio', {}).get('result', {}).get('portfolio')
        if port:
            summary = port.get('summary', {})
            tv = _as_float(summary.get('total_value') or 0)
            cash = _as_float(summary.get('total_cash') or 0)
            equity = _as_float(summary.get('total_equity') or 0)
            npos = int(summary.get('num_positions') or 0)
            print(f"Portfolio: total_value={tv:.2f} | cash={cash:.2f} | equity={equity:.2f} | positions={npos}")

        execn = result.get('execution', {}).get('result', {}).get('execution')
        if execn:
            oid = execn.get('order_id')
            price = _as_float(execn.get('avg_fill_price') or 0)
            filled = _as_float(execn.get('filled_size') or 0)
            status = execn.get('status')
            print(f"Execution: order={oid} | price={price:.4f} | filled={filled} | status={status}")

        print('--- end summary ---\n')

    def generate_action_paragraph(self, result: dict) -> str:
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
            parts.append("NASDAQ Plan: NASDAQ-listed stocks trade 9:30-16:00 ET; prefer limit/TWAP for large orders to reduce market impact; monitor pre/post-market for earnings-driven moves.")

        if last_price is not None:
            parts.append(f"Last price ${last_price:.4f}.")
        parts.append(f"Backtest score {score:.4f}, Sharpe {sharpe:.3f}, alpha {alpha:.4f}.")
        parts.append(f"Portfolio equity ${equity:,.2f}, current allocation {alloc:.4f}.")
        parts.append(f"Signals: +{pos}/-{neg}; risk var={var:.6f}.")
        parts.append(f"Recommendation: {recommendation}. {action_detail}")
        parts.append(f"Confidence: {confidence*100:.0f}% (algorithmic, not financial advice).")

        return ' '.join(parts)

    def parse_and_execute(self, text: str):
        parsed = self.secretary.parse(text)
        print('\n[Secretary parsed] ->', parsed)
        if parsed.get('action') == 'pipeline' and parsed.get('symbol'):
            print('\n[CEO] Approving pipeline run...')
            params = parsed.get('params') or {}
            # merge UI default price_mode if not provided
            if 'price_mode' not in params and self.default_price_mode:
                params['price_mode'] = self.default_price_mode
            res = self.pipeline.run(parsed['symbol'], params)
            print('\n[PIPELINE RESULTS]')

            # Print full JSON result (with numpy-safe serializer)
            try:
                print(json.dumps(res, default=self._json_default, indent=2))
            except Exception:
                # fallback to naive print
                print(res)

            # Print concise human-readable summary and action paragraph (same as run_demo)
            try:
                self.summarize_pipeline_result(res)
            except Exception as e:
                print(f"(could not generate summary: {e})")

            try:
                paragraph = self.generate_action_paragraph(res)
                print('\n--- Summary ---')
                print(paragraph)
                print('--- end action paragraph ---\n')
            except Exception as e:
                print(f"(could not generate action paragraph: {e})")

            # show synthetic-data banner if applicable
            final_state = res.get('final_state') or {}
            if final_state.get('data_source_warning'):
                print('\n' + '='*60)
                print('WARNING: data source fallback in use:', final_state.get('data_source_warning'))
                print('='*60 + '\n')
            
            # Debug: show data status
            market_data = final_state.get('market_data') or []
            # print(f"\n[DEBUG] Market data points: {len(market_data)}")
            # if market_data:
            #     print(f"[DEBUG] Last 3 prices: {market_data[-3:]}")
            #     print(f"[DEBUG] Data source: {final_state.get('data_metadata', {}).get('source', 'unknown')}")
            
            # show plots in the UI (best-effort)
            try:
                self.plot_market_data(final_state)
            except Exception as e:
                print(f"(plotting error: {e})")

            return
        targets = parsed.get('targets', [])
        if not targets and parsed.get('symbol'):
            targets = list(self.agents.keys())
        if targets:
            res = self.ceo.approve_and_dispatch(targets, 'run_single', parsed.get('symbol') or '', parsed.get('params', {}))
            print('\n[DISPATCH RESULTS]')
            for k, v in res.items():
                print(k, ':', v)
            return
        print('No actionable items found. Try: "run pipeline on AAPL" or "research: find_alpha AAPL"')
        
    def makeSense(self, var, var1):
        client = Client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Return 1 paragraph putting {var} and {var1} into words so its easy to read"}],
            web_search=False
        )
        return (response.choices[0].message.content)
    def makeSense1(self, var):
        client = Client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Return 1 paragraph putting {var} into words so its easy to read"}],
            web_search=False
        )
        return (response.choices[0].message.content)
    def start(self):
        self.clear(); self.banner(); self.help()
        while True:
            cmd = input('\n> ').strip()
            if not cmd:
                continue
            low = cmd.lower()
            # lightweight UI setters
            if low.startswith('set price_mode '):
                try:
                    mode = cmd.split(None, 2)[2].strip()
                    self.default_price_mode = mode
                    print(f"Default price_mode set to '{mode}'")
                except Exception:
                    print('Usage: set price_mode <last|close|intraday>')
                continue
            if low in ('exit', 'quit'):
                break
            if low == 'clear':
                self.clear(); continue
            if low == 'help':
                self.help(); continue
            if low == 'tasks':
                self.list_tasks(); continue
            if low == 'status':
                self.show_status(); continue
            if low == 'history':
                self.view_history(); continue

            if ':' in cmd:
                agent, rest = cmd.split(':', 1)
                agent = agent.strip().lower(); rest = rest.strip()
                if agent == 'secretary':
                    if rest.startswith('llm:'):
                        prompt = rest[len('llm:'):].strip()
                        print(self.secretary.makeSense(prompt))
                    else:
                        self.parse_and_execute(rest)
                    continue

                a = self.agents.get(agent)
                if not a:
                    print('Unknown agent')
                    continue

                tasks = getattr(a, 'tasks', {})
                first_word = rest.split()[0]
                if first_word in tasks:
                    parts = rest.split()
                    sym = None
                    params = {}
                    for p in parts[1:]:
                        if p.isalpha() and p.isupper() and len(p) <= 5:
                            sym = p
                        if '=' in p:
                            k, v = p.split('=', 1)
                            try:
                                v = int(v)
                            except:
                                try:
                                    v = float(v)
                                except:
                                    pass
                            params[k] = v
                    sym = sym or params.get('symbol', '')
                    if not sym:
                        print('No symbol provided. Use: agent: task TICKER or include symbol in params symbol=...')
                        continue
                    out = a.execute_task(first_word, symbol=sym, **params)
                    print(out)
                    continue
                else:
                    out = a.execute_task(rest)
                    print(out)
                    continue

            # default: treat as secretary message
            self.parse_and_execute(cmd)
            


def main():
    print("I got you with the LLM don't worry😉")
    d = ConsoleDashboard()
    d.start()

if __name__ == '__main__':
    
    main()
    
