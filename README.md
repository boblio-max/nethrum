# NETHRUM v4: Autonomous Multi-Agent Quant Trading System
### "The AI-run Quant Firm"

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![AI](https://img.shields.io/badge/AI-Multi_Agent-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Development-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)

---

## 🚀 Overview

**Nethrum v4** is an autonomous, multi-agent quantitative trading framework designed to operate like a fully automated hedge fund.

At the center is the **CEO (You)**. All agents operate underneath your command. The system includes:

* 🔹 **Independent AI Agents** (Research, Backtesting, Risk, Execution, Portfolio, Infrastructure)
* 🔹 **A Secretary LLM** that processes ideas, brainstorms, and acts as your assistant
* 🔹 **A Pipeline** that coordinates all agents
* 🔹 **Real Market Data** via `yfinance`
* 🔹 **Persistent History** tracking all signals and simulations
* 🔹 **CEO Console UI** for commanding the entire firm

---

## 🧠 System Architecture

The system uses a top-down command structure where the CEO directs the firm, the Secretary assists with logic/memory, and the Pipeline orchestrates the specialized agents.

```mermaid
graph TD
    User[CEO You] -->|Commands| UI[CEO Interface ui.py]
    UI -->|Directives| Pipe[Pipeline Orchestrator]
    UI -.->|Brainstorming| Sec[Secretary LLM]
    
    subgraph "The Firm (Agents)"
        Pipe --> Res[Research Agent]
        Pipe --> Infra[Infrastructure Agent]
        
        Res -->|Signals| BT[Backtest Agent]
        BT -->|Results| Risk[Risk Agent]
        Risk -->|Approved Limits| Port[Portfolio Agent]
        Port -->|Allocation| Exec[Execution Agent]
    end

    Res -.-> Data[(Data/History)]
    BT -.-> Data
    Sec -.-> Data
📂 Repository StructurePlaintextnethrum_v4/
│
├── ui.py                     # 🖥️ CEO interface — command bar & dashboard
├── pipeline.py               # ⚙️ Central orchestrator controlling all agents
├── ceo.py                    # 👔 High-level CEO command interpreter
├── secretary.py              # 📝 LLM-based support: brainstorming, tagging, notes
├── history_manager.py        # 🗄️ Persistent backtest + signal storage system
│
├── data/                     # 📂 Auto-created — stores historical runs, signals, logs
│
└── agents/
    ├── base_agent.py         # 🧱 Shared utilities + parent class for all agents
    ├── infrastructure_agent.py # 🛠️ Data integrity, file mgmt, health checks
    ├── quant_algos.py        # 📈 10 quant strategies (The Alpha Engine)
    ├── research_agent.py     # 🔎 Fetch data, run algos, generate signals
    ├── backtest_agent.py     # ⏱️ Historical simulation engine
    ├── risk_agent.py         # 🛡️ VaR, volatility, limits, kill switches
    ├── portfolio_agent.py    # ⚖️ Optimization, rebalancing, weighting
    └── execution_agent.py    # ⚡ Trade routing, slippage modeling, fills
🔬 Agent ResponsibilitiesAgentResponsibilityResearch AgentDownloads real stock data (via yfinance), runs strategies from quant_algos.py, and generates structured signals.Backtest AgentHistorical simulation engine. Computes Sharpe, Drawdown, and CAGR. Stores results in /data.Risk AgentReal-time VaR, position sizing, exposure monitoring, and kill-switch logic.Portfolio AgentDynamic weighting, cross-asset optimization, and rebalancing rules.Execution AgentSlippage modeling, order routing simulation, and fill tracking.Infrastructure AgentEnsures data integrity, file system health checks, and logging support.Secretary LLMBrainstorms alpha ideas, cleans input, creates internal notes, tags ideas, and helps the CEO plan.📈 The 10 Quant StrategiesDefined in agents/quant_algos.pyMean ReversionMomentum / Trend FollowingCross-Sectional FactorsPairs Trading (Cointegration)Machine-Learning ForecastingMicrostructure / Order FlowVolatility ArbitrageRisk-Parity AllocationBayesian Portfolio OptimizationEvent-Driven Models💬 CEO Interface (ui.py)The UI gives you a command bar, status readouts from each agent, and the ability to run full simulations.Example Commands:Bash> run full pipeline on AAPL
> research: test momentum on TSLA
> risk: compute var
> secretary: brainstorm 5 new event-driven models
⚡ Getting Started1. Clone the RepositoryBashgit clone [https://github.com/yourusername/nethrum_v4.git](https://github.com/yourusername/nethrum_v4.git)
cd nethrum_v4
2. Install DependenciesBashpip install yfinance pandas numpy openai colorama
3. Set Your OpenAI API KeyYou can hardcode it in base_agent.py or set it as an environment variable (Recommended):Windows:PowerShellsetx OPENAI_API_KEY "your_key_here"
Mac/Linux:Bashexport OPENAI_API_KEY="your_key_here"
▶️ UsageRun the CEO DashboardThis is the main entry point for the system.Bashpython ui.py
Run a Full Pipeline TestBypass the UI and run a direct simulation on a ticker.Bashpython pipeline.py "run on AAPL"
Talk to the SecretaryLaunch the standalone LLM assistant interface.Bashpython secretary.py
Quick System Test (Ping)Verify all agents are initialized and healthy.Bashpython pipeline.py "ping"
# Expected Output: Every agent initializes and reports status.
🗃️ History Systemhistory_manager.py automatically creates structured logs for every run.Location: /data/ directory.Contents: Signals, model results, backtests, and portfolio weights.🏁 Future Enhancements[ ] Web-based Dashboard (React/Streamlit)[ ] Live Data Feeds (Websockets)[ ] Autonomous Broker Execution (Alpaca/IBKR)[ ] Multi-factor Portfolio Models👤 AuthorNikhil MahankaliLLM CEO of NETHRUM
