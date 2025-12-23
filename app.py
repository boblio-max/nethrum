

import os
import json
from flask import Flask, request, jsonify, Response
from datetime import datetime

from agents.infrastructure_agent import InfrastructureAgent, _CACHE
from agents.research_agent import ResearchAgent
from agents.backtest_agent import BacktestingAgent
from agents.risk_agent import RiskAgent
from agents.portfolio_agent import PortfolioAgent
from agents.execution_agent import ExecutionAgent
from pipeline import Pipeline
from history_manager import HistoryManager

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize agents
infra = InfrastructureAgent()
research = ResearchAgent(infra)
backtest = BacktestingAgent(infra)
risk = RiskAgent()
portfolio = PortfolioAgent()
execution = ExecutionAgent()

agents = {
    'infra': infra,
    'research': research,
    'backtest': backtest,
    'risk': risk,
    'portfolio': portfolio,
    'execution': execution
}
ordered = ['infra', 'research', 'backtest', 'risk', 'portfolio', 'execution']
pipeline = Pipeline(ordered, agents)
history = HistoryManager()


# =============================================================================
# HTML TEMPLATE (embedded)
# =============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nethrum - Quantitative Trading Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-primary: #0a0a1a;
            --bg-secondary: #12122a;
            --bg-card: rgba(255, 255, 255, 0.03);
            --accent: #00d4ff;
            --accent-glow: rgba(0, 212, 255, 0.4);
            --success: #00ff88;
            --danger: #ff4444;
            --warning: #ffc800;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --text-muted: #666666;
            --border: rgba(255, 255, 255, 0.08);
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        /* Header */
        .header {
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo {
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .logo-icon {
            font-size: 1.8rem;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.5rem 1rem;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 50px;
            font-size: 0.85rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
            animation: pulse 2s infinite;
        }
        
        .status-dot.offline {
            background: var(--danger);
            box-shadow: 0 0 10px var(--danger);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Container */
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Search Section */
        .search-section {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            backdrop-filter: blur(10px);
        }
        
        .search-form {
            display: flex;
            gap: 1.5rem;
            align-items: flex-end;
            flex-wrap: wrap;
        }
        
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .input-group label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }
        
        input[type="text"], input[type="number"], select {
            padding: 0.875rem 1.25rem;
            border: 2px solid var(--border);
            border-radius: 10px;
            background: rgba(0, 0, 0, 0.3);
            color: var(--text-primary);
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 20px var(--accent-glow);
        }
        
        #symbol {
            width: 140px;
            text-transform: uppercase;
            font-weight: 700;
            font-size: 1.4rem;
            text-align: center;
            letter-spacing: 2px;
        }
        
        #lookback {
            width: 120px;
        }
        
        .btn {
            padding: 0.875rem 2rem;
            border: none;
            border-radius: 10px;
            font-size: 0.9rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--accent) 0%, #0099cc 100%);
            color: #000;
            box-shadow: 0 4px 15px var(--accent-glow);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px var(--accent-glow);
        }
        
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--accent);
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.875rem 0;
        }
        
        .checkbox-group input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: var(--accent);
        }
        
        .checkbox-group label {
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        /* Loading */
        .loading {
            text-align: center;
            padding: 4rem;
            display: none;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            width: 60px;
            height: 60px;
            border: 3px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1.5rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .loading p {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }
        
        /* Results Section */
        .results-section {
            display: none;
        }
        
        .results-section.active {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Warning Banner */
        .warning-banner {
            background: rgba(255, 200, 0, 0.1);
            border: 1px solid rgba(255, 200, 0, 0.3);
            border-radius: 10px;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            color: var(--warning);
        }
        
        .warning-banner.hidden {
            display: none;
        }
        
        /* Grid */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .grid-2 {
            grid-template-columns: repeat(2, 1fr);
        }
        
        /* Cards */
        .card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: rgba(0, 212, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        
        .card-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-secondary);
            font-weight: 600;
        }
        
        .card-badge {
            padding: 0.35rem 0.85rem;
            border-radius: 50px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        
        .badge-success {
            background: rgba(0, 255, 136, 0.15);
            color: var(--success);
        }
        
        .badge-danger {
            background: rgba(255, 68, 68, 0.15);
            color: var(--danger);
        }
        
        .badge-warning {
            background: rgba(255, 200, 0, 0.15);
            color: var(--warning);
        }
        
        .badge-info {
            background: rgba(0, 212, 255, 0.15);
            color: var(--accent);
        }
        
        /* Price Display */
        .price-display {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #ccc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
        }
        
        .price-change {
            font-size: 1.2rem;
            margin-top: 0.5rem;
            font-weight: 600;
        }
        
        .price-change.positive { color: var(--success); }
        .price-change.negative { color: var(--danger); }
        
        .price-meta {
            margin-top: 1rem;
            display: flex;
            gap: 1.5rem;
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        
        .price-meta span {
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        
        /* Metrics */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .metric {
            text-align: center;
            padding: 1.25rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--accent);
            line-height: 1.2;
        }
        
        .metric-value.positive { color: var(--success); }
        .metric-value.negative { color: var(--danger); }
        
        .metric-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.5rem;
        }
        
        /* Signals */
        .signals-container {
            margin-top: 1.25rem;
        }
        
        .signals-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.75rem;
        }
        
        .signals-bar {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        
        .signal-dot {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .signal-dot:hover {
            transform: scale(1.15);
        }
        
        .signal-dot.buy {
            background: rgba(0, 255, 136, 0.2);
            color: var(--success);
            border: 2px solid var(--success);
        }
        
        .signal-dot.sell {
            background: rgba(255, 68, 68, 0.2);
            color: var(--danger);
            border: 2px solid var(--danger);
        }
        
        .signal-dot.neutral {
            background: rgba(128, 128, 128, 0.2);
            color: var(--text-muted);
            border: 2px solid var(--text-muted);
        }
        
        /* Chart */
        .chart-container {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            margin-bottom: 2rem;
        }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
        }
        
        .chart-title {
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .data-source {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(0, 255, 136, 0.1);
            color: var(--success);
        }
        
        .data-source.synthetic {
            background: rgba(255, 200, 0, 0.1);
            color: var(--warning);
        }
        
        /* Recommendation Card */
        .recommendation-card {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(0, 153, 204, 0.08) 100%);
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .recommendation-action {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .recommendation-action.buy { color: var(--success); }
        .recommendation-action.sell { color: var(--danger); }
        .recommendation-action.hold { color: var(--warning); }
        
        .recommendation-detail {
            color: var(--text-secondary);
            margin-bottom: 1.5rem;
            font-size: 0.95rem;
        }
        
        .confidence-section {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            padding: 1rem;
        }
        
        .confidence-bar {
            height: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--success));
            border-radius: 5px;
            transition: width 0.8s ease;
        }
        
        .confidence-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.75rem;
        }
        
        .confidence-label span:last-child {
            color: var(--accent);
            font-weight: 600;
        }
        
        /* Summary Stats */
        .summary-row {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
        }
        
        .summary-row:last-child {
            border-bottom: none;
        }
        
        .summary-label {
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        
        .summary-value {
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* Footer */
        footer {
            text-align: center;
            padding: 3rem 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }
        
        footer a {
            color: var(--accent);
            text-decoration: none;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .header {
                padding: 1rem;
            }
            
            .search-form {
                flex-direction: column;
                align-items: stretch;
            }
            
            .input-group {
                width: 100%;
            }
            
            #symbol, #lookback {
                width: 100%;
            }
            
            .btn {
                width: 100%;
                justify-content: center;
            }
            
            .grid {
                grid-template-columns: 1fr;
            }
            
            .price-display {
                font-size: 2.2rem;
            }
            
            .metric-value {
                font-size: 1.3rem;
            }
        }
        
        /* Quick Actions Bar */
        .quick-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }
        
        .quick-btn {
            padding: 0.5rem 1rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .quick-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            border-color: var(--accent);
        }
        
        /* Tooltip */
        [data-tooltip] {
            position: relative;
        }
        
        [data-tooltip]:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 0.5rem 0.75rem;
            background: rgba(0, 0, 0, 0.9);
            color: #fff;
            font-size: 0.75rem;
            border-radius: 6px;
            white-space: nowrap;
            z-index: 100;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">
            <span class="logo-icon">⚡</span>
            <span>NETHRUM</span>
        </div>
        <div class="status-indicator">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Connecting...</span>
        </div>
    </header>
    
    <div class="container">
        <!-- Search Section -->
        <section class="search-section">
            <form class="search-form" id="pipelineForm">
                <div class="input-group">
                    <label for="symbol">Stock Symbol</label>
                    <input type="text" id="symbol" placeholder="MSFT" maxlength="5" required autocomplete="off">
                </div>
                
                <div class="input-group">
                    <label for="lookback">Lookback (Days)</label>
                    <input type="number" id="lookback" value="256" min="30" max="1000">
                </div>
                
                <div class="checkbox-group">
                    <input type="checkbox" id="forceLive">
                    <label for="forceLive">Force Live Data</label>
                </div>
                
                <button type="submit" class="btn btn-primary" id="runBtn">
                    <span>▶</span> Run Analysis
                </button>
                
                <button type="button" class="btn btn-secondary" id="clearCacheBtn">
                    🗑 Clear Cache
                </button>
            </form>
            
            <div class="quick-actions">
                <button class="quick-btn" onclick="quickRun('MSFT')">MSFT</button>
                <button class="quick-btn" onclick="quickRun('AAPL')">AAPL</button>
                <button class="quick-btn" onclick="quickRun('GOOGL')">GOOGL</button>
                <button class="quick-btn" onclick="quickRun('AMZN')">AMZN</button>
                <button class="quick-btn" onclick="quickRun('TSLA')">TSLA</button>
                <button class="quick-btn" onclick="quickRun('NVDA')">NVDA</button>
                <button class="quick-btn" onclick="quickRun('META')">META</button>
                <button class="quick-btn" onclick="quickRun('SPY')">SPY</button>
            </div>
        </section>
        
        <!-- Loading Section -->
        <section class="loading" id="loadingSection">
            <div class="spinner"></div>
            <p>Analyzing market data...</p>
        </section>
        
        <!-- Results Section -->
        <section class="results-section" id="resultsSection">
            <!-- Warning Banner -->
            <div class="warning-banner hidden" id="warningBanner">
                <span>⚠️</span>
                <span id="warningText"></span>
            </div>
            
            <!-- Price Chart -->
            <div class="chart-container">
                <div class="chart-header">
                    <h3 class="chart-title" id="chartTitle">Price Chart</h3>
                    <div class="data-source" id="dataSource">
                        <span>●</span>
                        <span id="sourceText">yfinance</span>
                    </div>
                </div>
                <canvas id="priceChart" height="80"></canvas>
            </div>
            
            <!-- Main Grid -->
            <div class="grid">
                <!-- Price Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">📈 Current Price</span>
                        <span class="card-badge badge-info" id="exchangeBadge">NYSE</span>
                    </div>
                    <div class="price-display" id="currentPrice">$0.00</div>
                    <div class="price-change" id="priceChange">+$0.00 (0.00%)</div>
                    <div class="price-meta">
                        <span>📊 <span id="dataPoints">0</span> data points</span>
                        <span>🕐 <span id="timestamp">--</span></span>
                    </div>
                </div>
                
                <!-- Recommendation Card -->
                <div class="card recommendation-card">
                    <div class="card-header">
                        <span class="card-title">🎯 AI Recommendation</span>
                    </div>
                    <div class="recommendation-action" id="recAction">HOLD</div>
                    <div class="recommendation-detail" id="recDetail">Run analysis to get recommendation</div>
                    <div class="confidence-section">
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
                        </div>
                        <div class="confidence-label">
                            <span>Confidence Level</span>
                            <span id="confidenceValue">0%</span>
                        </div>
                    </div>
                </div>
                
                <!-- Research Signals Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">🔬 Research Signals</span>
                        <span class="card-badge badge-info" id="signalBadge">0/10</span>
                    </div>
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value" id="alphaValue">0.0000</div>
                            <div class="metric-label">Alpha Score</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="signalCount">0 / 0</div>
                            <div class="metric-label">Buy / Sell</div>
                        </div>
                    </div>
                    <div class="signals-container">
                        <div class="signals-label">Signal Breakdown</div>
                        <div class="signals-bar" id="signalsBar"></div>
                    </div>
                </div>
                
                <!-- Backtest Metrics Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">📊 Backtest Metrics</span>
                        <span class="card-badge" id="scoreBadge">Score: 0.00</span>
                    </div>
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value" id="sharpeValue">0.00</div>
                            <div class="metric-label">Sharpe Ratio</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="sortinoValue">0.00</div>
                            <div class="metric-label">Sortino Ratio</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value negative" id="drawdownValue">0.00%</div>
                            <div class="metric-label">Max Drawdown</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="returnsValue">0.00%</div>
                            <div class="metric-label">Total Returns</div>
                        </div>
                    </div>
                </div>
                
                <!-- Risk Analysis Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">⚠️ Risk Analysis</span>
                    </div>
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value" id="allocationValue">0.00%</div>
                            <div class="metric-label">Suggested Allocation</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="varValue">0.00%</div>
                            <div class="metric-label">Value at Risk (95%)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="volatilityValue">0.00%</div>
                            <div class="metric-label">Annualized Volatility</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="winRateValue">0.0%</div>
                            <div class="metric-label">Win Rate</div>
                        </div>
                    </div>
                </div>
                
                <!-- Execution Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">⚡ Simulated Execution</span>
                        <span class="card-badge badge-success" id="execStatusBadge">Ready</span>
                    </div>
                    <div class="summary-row">
                        <span class="summary-label">Execution Price</span>
                        <span class="summary-value" id="execPrice">$0.00</span>
                    </div>
                    <div class="summary-row">
                        <span class="summary-label">Order Type</span>
                        <span class="summary-value" id="execType">Market</span>
                    </div>
                    <div class="summary-row">
                        <span class="summary-label">Status</span>
                        <span class="summary-value" id="execStatus">-</span>
                    </div>
                    <div class="summary-row">
                        <span class="summary-label">Order ID</span>
                        <span class="summary-value" id="execOrderId" style="font-size: 0.8rem; font-family: monospace;">-</span>
                    </div>
                </div>
            </div>
        </section>
    </div>
    
    <footer>
        <p>Nethrum Quantitative Trading System v1.0</p>
        <p style="margin-top: 0.5rem;">Data provided by <a href="https://pypi.org/project/yfinance/" target="_blank">yfinance</a> | For educational purposes only | Not financial advice</p>
    </footer>
    
    <script>
        let priceChart = null;
        
        // =====================================================================
        // INITIALIZATION
        // =====================================================================
        
        document.addEventListener('DOMContentLoaded', () => {
            checkStatus();
            setInterval(checkStatus, 30000); // Check every 30 seconds
        });
        
        // =====================================================================
        // API FUNCTIONS
        // =====================================================================
        
        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                const dot = document.getElementById('statusDot');
                const text = document.getElementById('statusText');
                
                if (data.yfinance_available) {
                    dot.classList.remove('offline');
                    text.textContent = `yfinance connected | ${data.cache_entries} cached`;
                } else {
                    dot.classList.add('offline');
                    text.textContent = 'yfinance unavailable';
                }
            } catch (e) {
                document.getElementById('statusDot').classList.add('offline');
                document.getElementById('statusText').textContent = 'Disconnected';
            }
        }
        
        async function runPipeline(symbol, lookback, forceLive) {
            const loadingSection = document.getElementById('loadingSection');
            const resultsSection = document.getElementById('resultsSection');
            const runBtn = document.getElementById('runBtn');
            
            loadingSection.classList.add('active');
            resultsSection.classList.remove('active');
            runBtn.disabled = true;
            
            try {
                const response = await fetch('/api/run_pipeline', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol, lookback, force_live: forceLive })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                displayResults(data);
                
            } catch (e) {
                alert('Error: ' + e.message);
            } finally {
                loadingSection.classList.remove('active');
                runBtn.disabled = false;
            }
        }
        
        async function clearCache() {
            try {
                await fetch('/api/clear_cache', { method: 'POST' });
                checkStatus();
                alert('Cache cleared successfully!');
            } catch (e) {
                alert('Error clearing cache');
            }
        }
        
        // =====================================================================
        // DISPLAY FUNCTIONS
        // =====================================================================
        
        function formatNum(value, decimals = 2) {
            if (value === null || value === undefined || isNaN(value)) return 'N/A';
            return Number(value).toLocaleString(undefined, {
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals
            });
        }
        
        function displayResults(data) {
            const resultsSection = document.getElementById('resultsSection');
            resultsSection.classList.add('active');
            
            // Warning Banner
            const warningBanner = document.getElementById('warningBanner');
            if (data.warning) {
                warningBanner.classList.remove('hidden');
                document.getElementById('warningText').textContent = 'Using synthetic/fallback data: ' + data.warning;
            } else {
                warningBanner.classList.add('hidden');
            }
            
            // Data Source
            const sourceEl = document.getElementById('dataSource');
            const sourceText = document.getElementById('sourceText');
            sourceText.textContent = data.data_source || 'unknown';
            sourceEl.classList.toggle('synthetic', data.data_source === 'synthetic');
            
            // Chart
            document.getElementById('chartTitle').textContent = `${data.symbol} - Price History`;
            updateChart(data.prices || [], data.symbol);
            
            // Price Info
            const prices = data.prices || [];
            const lastPrice = data.last_price || (prices.length ? prices[prices.length - 1] : 0);
            const prevPrice = prices.length > 1 ? prices[prices.length - 2] : lastPrice;
            const change = lastPrice - prevPrice;
            const changePct = prevPrice ? (change / prevPrice * 100) : 0;
            
            document.getElementById('currentPrice').textContent = `$${formatNum(lastPrice, 2)}`;
            
            const changeEl = document.getElementById('priceChange');
            const changeSign = change >= 0 ? '+' : '';
            changeEl.textContent = `${changeSign}$${formatNum(Math.abs(change), 2)} (${changeSign}${formatNum(changePct, 2)}%)`;
            changeEl.className = 'price-change ' + (change >= 0 ? 'positive' : 'negative');
            
            document.getElementById('dataPoints').textContent = data.data_points || 0;
            document.getElementById('exchangeBadge').textContent = data.exchange || 'Unknown';
            document.getElementById('timestamp').textContent = new Date().toLocaleTimeString();
            
            // Recommendation
            const rec = data.recommendation || {};
            const recAction = document.getElementById('recAction');
            const actionText = rec.action || 'HOLD';
            recAction.textContent = actionText;
            recAction.className = 'recommendation-action ' + actionText.toLowerCase().split(' ')[0];
            
            document.getElementById('recDetail').textContent = rec.detail || 'No specific recommendation';
            document.getElementById('confidenceFill').style.width = (rec.confidence || 0) + '%';
            document.getElementById('confidenceValue').textContent = (rec.confidence || 0) + '%';
            
            // Research Signals
            const alpha = data.alpha || 0;
            const alphaEl = document.getElementById('alphaValue');
            alphaEl.textContent = formatNum(alpha, 4);
            alphaEl.className = 'metric-value ' + (alpha > 0 ? 'positive' : alpha < 0 ? 'negative' : '');
            
            const signals = data.signals || [];
            const buySignals = signals.filter(s => s > 0).length;
            const sellSignals = signals.filter(s => s < 0).length;
            document.getElementById('signalCount').textContent = `${buySignals} / ${sellSignals}`;
            document.getElementById('signalBadge').textContent = `${signals.length} signals`;
            
            // Signal dots
            const signalsBar = document.getElementById('signalsBar');
            signalsBar.innerHTML = '';
            signals.forEach((s, i) => {
                const dot = document.createElement('div');
                dot.className = 'signal-dot ' + (s > 0 ? 'buy' : s < 0 ? 'sell' : 'neutral');
                dot.textContent = i + 1;
                dot.setAttribute('data-tooltip', `Signal ${i + 1}: ${formatNum(s, 4)}`);
                signalsBar.appendChild(dot);
            });
            
            // Backtest Metrics
            const score = data.backtest_score || 0;
            const scoreBadge = document.getElementById('scoreBadge');
            scoreBadge.textContent = `Score: ${formatNum(score, 4)}`;
            scoreBadge.className = 'card-badge ' + (score > 0 ? 'badge-success' : score < 0 ? 'badge-danger' : 'badge-warning');
            
            document.getElementById('sharpeValue').textContent = formatNum(data.sharpe_ratio, 3);
            document.getElementById('sortinoValue').textContent = formatNum(data.sortino_ratio, 3);
            
            const drawdown = (data.max_drawdown || 0) * 100;
            const drawdownEl = document.getElementById('drawdownValue');
            drawdownEl.textContent = formatNum(drawdown, 2) + '%';
            drawdownEl.className = 'metric-value negative';
            
            const returns = (data.total_returns || 0) * 100;
            const returnsEl = document.getElementById('returnsValue');
            returnsEl.textContent = formatNum(returns, 2) + '%';
            returnsEl.className = 'metric-value ' + (returns >= 0 ? 'positive' : 'negative');
            
            // Risk
            document.getElementById('allocationValue').textContent = formatNum((data.allocation || 0) * 100, 2) + '%';
            document.getElementById('varValue').textContent = formatNum((data.var || 0) * 100, 4) + '%';
            document.getElementById('volatilityValue').textContent = formatNum((data.volatility || 0) * 100, 2) + '%';
            document.getElementById('winRateValue').textContent = formatNum((data.win_rate || 0) * 100, 1) + '%';
            
            // Execution
            const exec = data.execution || {};
            document.getElementById('execPrice').textContent = `$${formatNum(exec.avg_fill_price, 2)}`;
            document.getElementById('execType').textContent = (exec.order_type || 'market').toUpperCase();
            document.getElementById('execStatus').textContent = (exec.status || '-').toUpperCase();
            document.getElementById('execOrderId').textContent = exec.order_id || '-';
            
            const execStatusBadge = document.getElementById('execStatusBadge');
            execStatusBadge.textContent = exec.status || 'Ready';
            execStatusBadge.className = 'card-badge ' + (exec.status === 'filled' ? 'badge-success' : 'badge-info');
        }
        
        function updateChart(prices, symbol) {
            const ctx = document.getElementById('priceChart').getContext('2d');
            
            if (priceChart) {
                priceChart.destroy();
            }
            
            if (!prices || prices.length === 0) {
                return;
            }
            
            const labels = prices.map((_, i) => i + 1);
            
            // Calculate 20-day moving average
            const ma20 = prices.map((_, i, arr) => {
                const start = Math.max(0, i - 19);
                const slice = arr.slice(start, i + 1);
                return slice.reduce((a, b) => a + b, 0) / slice.length;
            });
            
            // Calculate Bollinger Bands (simple version)
            const bb = prices.map((_, i, arr) => {
                const window = 20;
                const start = Math.max(0, i - window + 1);
                const slice = arr.slice(start, i + 1);
                const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
                const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / slice.length;
                const std = Math.sqrt(variance);
                return { upper: mean + 2 * std, lower: mean - 2 * std };
            });
            
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: symbol,
                            data: prices,
                            borderColor: '#00d4ff',
                            backgroundColor: 'rgba(0, 212, 255, 0.1)',
                            fill: true,
                            tension: 0.1,
                            pointRadius: 0,
                            borderWidth: 2.5
                        },
                        {
                            label: 'MA20',
                            data: ma20,
                            borderColor: '#ff9800',
                            borderWidth: 1.5,
                            pointRadius: 0,
                            borderDash: [5, 5],
                            fill: false
                        },
                        {
                            label: 'Upper BB',
                            data: bb.map(b => b.upper),
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                            borderWidth: 1,
                            pointRadius: 0,
                            fill: false
                        },
                        {
                            label: 'Lower BB',
                            data: bb.map(b => b.lower),
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                            borderWidth: 1,
                            pointRadius: 0,
                            fill: '-1',
                            backgroundColor: 'rgba(255, 255, 255, 0.02)'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: {
                                color: '#888',
                                usePointStyle: true,
                                padding: 20
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#fff',
                            bodyColor: '#ccc',
                            borderColor: '#333',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.05)',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#888',
                                callback: function(value) {
                                    return '$' + value.toFixed(2);
                                }
                            }
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    }
                }
            });
        }
        
        // =====================================================================
        // EVENT HANDLERS
        // =====================================================================
        
        document.getElementById('pipelineForm').addEventListener('submit', (e) => {
            e.preventDefault();
            const symbol = document.getElementById('symbol').value.toUpperCase().trim();
            const lookback = parseInt(document.getElementById('lookback').value) || 256;
            const forceLive = document.getElementById('forceLive').checked;
            
            if (symbol) {
                runPipeline(symbol, lookback, forceLive);
            }
        });
        
        document.getElementById('clearCacheBtn').addEventListener('click', clearCache);
        
        // Quick run buttons
        function quickRun(symbol) {
            document.getElementById('symbol').value = symbol;
            const lookback = parseInt(document.getElementById('lookback').value) || 256;
            const forceLive = document.getElementById('forceLive').checked;
            runPipeline(symbol, lookback, forceLive);
        }
        
        // Symbol input auto-uppercase
        document.getElementById('symbol').addEventListener('input', (e) => {
            e.target.value = e.target.value.toUpperCase();
        });
    </script>
</body>
</html>
'''


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_recommendation(result):
    """Generate trading recommendation from pipeline result"""
    final_state = result.get('final_state', {})
    
    alpha = final_state.get('alpha', 0) or 0
    score = final_state.get('backtest_score', 0) or 0
    sharpe = final_state.get('sharpe_ratio', 0) or 0
    alloc = final_state.get('allocation', 0) or 0
    
    recommendation = 'HOLD'
    detail = 'No immediate trade suggested.'
    confidence = max(0.0, min(1.0, (abs(score) * 10.0 + max(0.0, sharpe)) / 6.0))
    
    if score > 0.05 and sharpe > 0.5 and alpha > 0:
        recommendation = 'BUY'
        suggested_pct = min(0.1, max(0.01, abs(alloc) * 0.5 or 0.02))
        detail = f"Consider building a position (~{suggested_pct*100:.1f}% of portfolio)."
    elif score < -0.02 or sharpe < 0.15:
        if alloc > 0:
            recommendation = 'SELL'
            detail = "Consider reducing exposure due to poor metrics."
        else:
            recommendation = 'AVOID'
            detail = 'Metrics do not support entry at this time.'
    elif alpha > 0.05:
        recommendation = 'BUY (Weak)'
        detail = 'Positive alpha detected; consider small position.'
    elif alpha < -0.05:
        recommendation = 'SELL (Weak)'
        detail = 'Negative alpha; avoid new positions.'
    
    return {
        'action': recommendation,
        'detail': detail,
        'confidence': round(confidence * 100, 1)
    }


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return Response(HTML_TEMPLATE, mimetype='text/html')


@app.route('/api/run_pipeline', methods=['POST'])
def run_pipeline_api():
    """Run the full pipeline on a symbol"""
    data = request.get_json() or {}
    symbol = (data.get('symbol') or '').upper().strip()
    
    if not symbol or len(symbol) > 5 or not symbol.isalpha():
        return jsonify({'error': 'Invalid symbol. Use 1-5 letters (e.g., MSFT, AAPL)'}), 400
    
    lookback = min(max(int(data.get('lookback', 256)), 30), 1000)
    force_live = bool(data.get('force_live', False))
    
    params = {
        'lookback': lookback,
        'force_live': force_live
    }
    
    try:
        if force_live:
            _CACHE.clear()
        
        result = pipeline.run(symbol, params)
        
        final_state = result.get('final_state', {})
        market_data = final_state.get('market_data', []) or []
        
        # Calculate price change
        last_price = market_data[-1] if market_data else None
        
        response = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'data_source': final_state.get('data_metadata', {}).get('source', 'unknown'),
            'exchange': final_state.get('exchange'),
            'data_points': len(market_data),
            'last_price': last_price,
            'prices': market_data[-100:] if market_data else [],
            
            'alpha': final_state.get('alpha'),
            'signals': final_state.get('signals', []),
            
            'backtest_score': final_state.get('backtest_score'),
            'sharpe_ratio': final_state.get('sharpe_ratio'),
            'sortino_ratio': final_state.get('sortino_ratio'),
            'max_drawdown': final_state.get('max_drawdown'),
            'total_returns': final_state.get('all_metrics', {}).get('total_returns'),
            'volatility': final_state.get('all_metrics', {}).get('volatility'),
            'win_rate': final_state.get('all_metrics', {}).get('win_rate'),
            
            'allocation': final_state.get('allocation'),
            'var': final_state.get('var'),
            
            'portfolio': final_state.get('portfolio_metrics', {}).get('summary', {}),
            'execution': final_state.get('execution', {}),
            
            'recommendation': generate_recommendation(result),
            'warning': final_state.get('data_source_warning')
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def get_status():
    """Get system status"""
    from agents.infrastructure_agent import YF_AVAILABLE
    
    return jsonify({
        'yfinance_available': YF_AVAILABLE,
        'agents': list(agents.keys()),
        'cache_entries': len(_CACHE)
    })


@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the data cache"""
    _CACHE.clear()
    return jsonify({'status': 'ok', 'message': 'Cache cleared'})


@app.route('/api/quick_quote', methods=['POST'])
def quick_quote():
    """Get a quick price quote"""
    data = request.get_json() or {}
    symbol = (data.get('symbol') or '').upper().strip()
    
    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400
    
    try:
        result = infra.task_load_data(symbol, lookback=30, force_live=True)
        prices = result.get('data', [])
        
        if not prices:
            return jsonify({'error': 'No data available'}), 404
        
        last_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else last_price
        change = last_price - prev_price
        change_pct = (change / prev_price * 100) if prev_price else 0
        
        return jsonify({
            'symbol': symbol,
            'price': last_price,
            'change': change,
            'change_pct': change_pct,
            'high_30d': max(prices),
            'low_30d': min(prices),
            'exchange': result.get('exchange'),
            'source': result.get('metadata', {}).get('source')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  ⚡ NETHRUM - Quantitative Trading Dashboard")
    print("=" * 60)
    print()
    print("  Open your browser and go to:")
    print()
    print("     👉  http://127.0.0.1:5000")
    print()
    print("=" * 60)
    print()
    
    app.run(debug=True, host='127.0.0.1', port=5000)


