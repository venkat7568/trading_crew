#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
main.py — Trading UI with robust rendering & SSE
- HTML wrapped in {% raw %} ... {% endraw %} to prevent Jinja collisions
- SSE newlines fixed
- /status hardened against broker errors
"""

import os
import json
import time
import queue
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from flask import Flask, render_template_string, jsonify, request, Response
from dotenv import load_dotenv

load_dotenv()

# Local modules
from trading_crew import TradingCrew
import logging
import logging.handlers

# ============================================================================
# LOGGING CONFIGURATION - Comprehensive logging to console + file
# ============================================================================
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

# Create rotating file handler (max 10MB per file, keep 5 files)
log_file = LOG_DIR / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Console handler (less verbose)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Configure specific loggers
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Suppress LiteLLM debug logs (used by CrewAI)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)

# Create a dedicated logger for trading system
trade_logger = logging.getLogger("trading_system")
trade_logger.info("=" * 80)
trade_logger.info(f"🚀 Trading system starting - Log file: {log_file}")
trade_logger.info("=" * 80)

# These imports may fail when tokens are missing; we'll guard their usage
try:
    from upstox_operator import UpstoxOperator
except Exception:
    UpstoxOperator = None

try:
    from upstox_technical import UpstoxTechnicalClient
except Exception:
    UpstoxTechnicalClient = None

try:
    from news_client import NewsClient
except Exception:
    NewsClient = None

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path("./data"); DATA_DIR.mkdir(exist_ok=True)

# Max number of different symbols we’ll trade per cycle (NOT a restriction by name)
MAX_DISCOVERED_SYMBOLS = int(os.environ.get("MAX_DISCOVERED_SYMBOLS", "20"))

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'trading-secret-2024')

logging.getLogger("werkzeug").setLevel(logging.ERROR)
trading_active = False
status_queue = queue.Queue()
current_companies_data = []  # Track companies being analyzed

HTML = r"""{% raw %}
<!DOCTYPE html>
<html>
<head>
    <title>AI Trading System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; border-radius: 15px; padding: 25px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        h1 { font-size: 32px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric { background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 20px; border-radius: 10px; text-align: center; }
        .metric-label { font-size: 14px; color: #666; margin-bottom: 10px; }
        .metric-value { font-size: 28px; font-weight: bold; color: #333; }
        .status-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 600; }
        .status-open { background: #10b981; color: white; }
        .status-closed { background: #ef4444; color: white; }
        .status-running { background: #3b82f6; color: white; animation: pulse 2s infinite; }
        .status-idle { background: #6b7280; color: white; }
        .controls { display: flex; gap: 15px; margin-top: 20px; flex-wrap: wrap; }
        .form-group { flex: 1; min-width: 200px; }
        label { display: block; font-size: 14px; color: #666; margin-bottom: 8px; font-weight: 500; }
        input, select { width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 14px; }
        input:focus, select:focus { outline: none; border-color: #667eea; }
        button { padding: 14px 28px; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; transition: all 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; }
        .btn-success { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; }
        .btn-danger { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .activity-feed { background: #f9fafb; border-radius: 10px; padding: 20px; max-height: 500px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 13px; }
        .activity-item { padding: 12px; margin-bottom: 10px; border-radius: 6px; border-left: 4px solid #667eea; background: white; }
        .activity-time { color: #999; font-size: 11px; }
        .activity-text { color: #333; margin-top: 5px; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🤖 AI Trading System</h1>
            <div style="color: #666; margin-bottom: 20px;">Agent-Based Discovery • Multi-Signal Analysis • Risk Managed</div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Market Status</div>
                    <span id="market-status" class="status-badge status-closed">Loading...</span>
                    <div style="margin-top: 10px; font-size: 14px; color: #666;">
                        <div id="market-time">--:--:--</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="metric">
                    <div class="metric-label">Available Funds</div>
                    <div id="available-margin" class="metric-value">₹0</div>
                </div>
            </div>

            <div class="card">
                <div class="metric">
                    <div class="metric-label">Trading Status</div>
                    <span id="trading-status" class="status-badge status-idle">Idle</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2 style="margin-bottom: 20px; color: #333;">Trading Controls</h2>
            <div class="controls">
                <div class="form-group">
                    <label>Mode</label>
                    <select id="mode">
                        <option value="live">Live Trading</option>
                        <option value="backtest">Backtest</option>
                    </select>
                </div>

                <div class="form-group" id="date-group" style="display: none;">
                    <label>Backtest Date</label>
                    <input type="date" id="backtest-date">
                </div>

                <div class="form-group">
                    <label>Execute Orders</label>
                    <select id="live-mode">
                        <option value="false">Paper Trading (Safe)</option>
                        <option value="true">Live Orders ⚠️</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Max Companies to Analyze</label>
                    <input type="number" id="max-symbols" min="5" max="50" value="20" placeholder="20">
                    <div style="font-size: 11px; color: #666; margin-top: 5px;">Default: 20 symbols per cycle</div>
                </div>

                <div class="form-group">
                    <label>Learning Mode 🧠</label>
                    <select id="learning-mode">
                        <option value="true">Enabled (Learns from trades)</option>
                        <option value="false">Disabled</option>
                    </select>
                    <div style="font-size: 11px; color: #666; margin-top: 5px;">System improves from past trades</div>
                </div>
            </div>

            <div style="margin-top: 20px; display: flex; gap: 15px;">
                <button id="btn-start" class="btn-success" onclick="startTrading()">▶️ Start Auto Trading</button>
                <button id="btn-stop" class="btn-danger" onclick="stopTrading()" disabled>⏹️ Stop</button>
                <button class="btn-primary" onclick="refreshData()">🔄 Refresh</button>
                <button class="btn-primary" onclick="runLearning()">🎓 Run Learning Analysis</button>
            </div>

            <div style="margin-top: 20px; padding: 15px; background: #fffbeb; border-radius: 8px; border-left: 4px solid #f59e0b;">
                <div style="font-weight: 600; color: #92400e; margin-bottom: 5px;">🧠 Agent-Based Intelligence</div>
                <div style="font-size: 13px; color: #78350f;">
                    News Agent → Technical Agent → Risk Agent → Executor. Full multi-agent validation and decision-making.
                </div>
            </div>
        </div>

        <div class="card">
            <h2 style="margin-bottom: 15px; color: #333;">Activity Log</h2>
            <div id="activity-feed" class="activity-feed">
                <div class="activity-item">
                    <div class="activity-time">UI loaded</div>
                    <div class="activity-text">✅ Frontend is ready</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let eventSource = null;

        document.getElementById('mode').addEventListener('change', function() {
            const dateGroup = document.getElementById('date-group');
            dateGroup.style.display = this.value === 'backtest' ? 'block' : 'none';
        });

        document.getElementById('backtest-date').value = new Date().toISOString().split('T')[0];

        function addActivity(text) {
            const feed = document.getElementById('activity-feed');
            const item = document.createElement('div');
            item.className = 'activity-item';
            const time = new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata' });
            item.innerHTML = '<div class="activity-time">' + time + '</div><div class="activity-text">' + text + '</div>';
            feed.insertBefore(item, feed.firstChild);
            while (feed.children.length > 50) { feed.removeChild(feed.lastChild); }
        }

        async function refreshData() {
            try {
                const resp = await fetch('/status');
                const data = await resp.json();

                const marketStatus = document.getElementById('market-status');
                if (data.market && data.market.open) {
                    marketStatus.textContent = 'OPEN';
                    marketStatus.className = 'status-badge status-open';
                } else {
                    marketStatus.textContent = 'CLOSED';
                    marketStatus.className = 'status-badge status-closed';
                }

                document.getElementById('market-time').textContent = (data.market && data.market.time) ? data.market.time : '--:--:--';

                const funds = data.funds || {};
                const available = Number(funds.available_margin || 0);
                document.getElementById('available-margin').textContent = '₹' + available.toLocaleString('en-IN', { maximumFractionDigits: 2 });

                const tradingStatus = document.getElementById('trading-status');
                if (data.trading_active) {
                    tradingStatus.textContent = 'RUNNING';
                    tradingStatus.className = 'status-badge status-running';
                } else {
                    tradingStatus.textContent = 'IDLE';
                    tradingStatus.className = 'status-badge status-idle';
                }
            } catch (error) {
                console.error('Refresh error:', error);
                addActivity('⚠️ Refresh error: ' + error);
            }
        }

        async function startTrading() {
            const mode = document.getElementById('mode').value;
            const date = document.getElementById('backtest-date').value;
            const live = document.getElementById('live-mode').value === 'true';
            const maxSymbols = parseInt(document.getElementById('max-symbols').value) || 20;
            const learningMode = document.getElementById('learning-mode').value === 'true';

            if (live && !confirm('⚠️ WARNING: Live trading will use REAL MONEY! Continue?')) return;

            document.getElementById('btn-start').disabled = true;
            document.getElementById('btn-stop').disabled = false;

            addActivity('🚀 Starting... Mode: ' + mode + ', Live: ' + (live ? 'YES ⚠️' : 'Paper'));
            addActivity('📊 Max symbols: ' + maxSymbols + ', Learning: ' + (learningMode ? 'ON 🧠' : 'OFF'));

            try {
                const resp = await fetch('/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode, date, live, max_symbols: maxSymbols, learning_mode: learningMode })
                });
                const result = await resp.json();
                if (result.status === 'ok') {
                    addActivity('✅ Trading started successfully');
                } else {
                    addActivity('❌ Error: ' + (result.message || 'Unknown'));
                    document.getElementById('btn-start').disabled = false;
                    document.getElementById('btn-stop').disabled = true;
                }
            } catch (error) {
                addActivity('❌ Failed: ' + error);
                document.getElementById('btn-start').disabled = false;
                document.getElementById('btn-stop').disabled = true;
            }
        }

        async function stopTrading() {
            try {
                await fetch('/stop', { method: 'POST' });
                addActivity('⏹️ Trading stopped');
                document.getElementById('btn-start').disabled = false;
                document.getElementById('btn-stop').disabled = true;
            } catch (error) {
                addActivity('❌ Failed to stop: ' + error);
            }
        }

        async function runLearning() {
            addActivity('🎓 Running learning analysis...');
            try {
                const resp = await fetch('/learning', { method: 'POST' });
                const result = await resp.json();
                if (result.status === 'ok') {
                    addActivity('✅ Learning analysis complete!');
                    if (result.insights) {
                        addActivity('📊 Insights: ' + result.insights);
                    }
                } else {
                    addActivity('❌ Learning error: ' + (result.message || 'Unknown'));
                }
            } catch (error) {
                addActivity('❌ Failed to run learning: ' + error);
            }
        }

        function connectSSE() {
            if (eventSource) eventSource.close();
            eventSource = new EventSource('/stream');
            eventSource.onmessage = function(e) {
                try {
                    const payload = JSON.parse(e.data);
                    if (payload.event !== 'heartbeat' && payload.message) addActivity(payload.message);
                } catch (err) {}
            };
            eventSource.onerror = function() { setTimeout(connectSSE, 5000); };
        }

        setInterval(refreshData, 3000);
        setInterval(() => {
            const time = new Date().toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata' });
            document.getElementById('market-time').textContent = time;
        }, 1000);

        connectSSE();
        refreshData();
    </script>
</body>
</html>
{% endraw %}
"""

def emit_status(message: str):
    status_queue.put({
        "event": "update",
        "message": message,
        "timestamp": datetime.now(IST).isoformat()
    })

def _crew_status_to_text(payload: dict) -> str:
    event = payload.get("event", "")
    data = payload.get("data", {}) or {}
    sym = data.get("symbol")
    if event == "market_wait_start": return "⏳ Waiting for market to open…"
    if event == "market_status":     return f"📟 Market status: {'OPEN' if data.get('open') else 'CLOSED'} (phase: {data.get('phase')})"
    if event == "review_holdings_start": return "🔍 Reviewing existing holdings…"
    if event == "reviewing_holding" and sym: return f"🧾 Reviewing holding: {sym}"
    if event == "holding_action" and sym:    return f"📌 {sym}: {data.get('action','').upper()} — {data.get('reason','')[:120]}"
    if event == "decide_start" and sym:      return f"🧮 Deciding trade for {sym}…"
    if event == "decision_analyzing" and sym:return f"🔎 Analyzing news & technicals for {sym}…"
    if event == "decide_complete" and sym:
        dirn = data.get("direction"); conf = data.get("confidence")
        return f"✅ Decision {sym}: {dirn} (conf={conf})" if conf is not None else f"✅ Decision {sym}: {dirn}"
    if event == "sizing_start" and sym:      return f"📐 Sizing position for {sym}…"
    if event == "sizing_complete" and sym:   return f"📝 Plan ready for {sym}"
    if event == "execution_complete" and sym:return f"🚦 Executed flow for {sym}"
    if event == "cycle_complete":            return "✅ Decision cycle complete."
    return f"ℹ️ {event}"

# ---------- NEW: pure news-based discovery using UpstoxTechnicalClient.resolve ----------
def discover_and_validate_symbols(max_symbols=20):
    """
    Discover tradable NSE/BSE symbols purely from latest news.
    No static TRADING_SYMBOLS. We:
      - Fetch recent news
      - Extract hints (symbol/company/headline/title)
      - Pass each hint to UpstoxTechnicalClient.resolve()
      - Keep unique NSE_EQ| / BSE_EQ| instruments
    """
    emit_status("🔍 Discovering stocks from recent news (no static watchlist)…")

    if not (NewsClient and UpstoxTechnicalClient):
        emit_status("❌ News/Technical client not available")
        return []

    news_client = NewsClient()
    tech_client = UpstoxTechnicalClient()

    news_data = news_client.get_recent_news_and_calls(
        lookback_days=2,
        max_items=50,
        compact=True
    )

    if not isinstance(news_data, list):
        news_data = news_data or []

    emit_status(f"📰 Fetched {len(news_data)} news items")

    validated = []
    seen_ik = set()

    def _hint_strings(item):
        """Collect possible company/symbol hints from one news item."""
        hints = []

        # Direct symbol-like fields
        for key in ("symbol", "ticker"):
            v = item.get(key)
            if isinstance(v, str):
                hints.append(v)

        # Arrays / company fields
        for key in ("symbols", "companies", "company"):
            v = item.get(key)
            if isinstance(v, str):
                hints.append(v)
            elif isinstance(v, (list, tuple)):
                hints.extend([x for x in v if isinstance(x, str)])

        # Headlines / titles – this is where your resolve() magic kicks in
        title = item.get("headline") or item.get("title")
        if isinstance(title, str):
            hints.append(title)

        # Deduplicate (case-insensitive) while preserving order
        out = []
        seen_local = set()
        for h in hints:
            h = h.strip()
            if not h:
                continue
            key = h.lower()
            if key in seen_local:
                continue
            seen_local.add(key)
            out.append(h)
        return out

    def _is_valid_symbol(sym: str) -> bool:
        """Validate symbol format to filter junk."""
        if not sym or not isinstance(sym, str):
            return False

        sym = sym.strip().upper()

        # Length check: NSE symbols are typically 1-10 chars
        if len(sym) < 1 or len(sym) > 10:
            return False

        # Format check: Should be alphanumeric, usually starts with letter
        if not sym[0].isalpha():
            return False

        # Should be mostly letters (at least 50%)
        letter_count = sum(1 for c in sym if c.isalpha())
        if letter_count < len(sym) * 0.5:
            return False

        # Common junk patterns to reject
        junk_patterns = [
            lambda s: s.isdigit(),  # All numbers
            lambda s: any(c in s for c in ['_', '-', '.', ' ']),  # Special chars
            lambda s: len(s) > 6 and any(c.isdigit() for c in s[-3:]),  # Ends with numbers (likely truncated text)
        ]

        for pattern in junk_patterns:
            if pattern(sym):
                return False

        return True

    def _is_tradable_stock(sym: str, instrument_key: str) -> tuple[bool, str]:
        """
        Verify stock is actually tradable by checking price, volume, and liquidity.

        Returns: (is_tradable: bool, reason: str)
        """
        try:
            # Fetch technical snapshot to check tradability
            snapshot = tech_client.snapshot(sym, days=2)

            if not snapshot:
                return False, "no_snapshot_data"

            # Check 1: Valid current price (must be > ₹1)
            current_price = snapshot.get("current_price")
            if current_price is None or current_price == 0:
                return False, f"zero_price (price={current_price})"

            current_price = float(current_price)
            if current_price < 1.0:
                return False, f"price_too_low (₹{current_price:.2f})"

            # Check 2: Recent candle data exists (stock is trading)
            candles = snapshot.get("candles", [])
            if not candles or len(candles) == 0:
                return False, "no_candle_data"

            # Check 3: Latest candle has valid OHLC data
            latest_candle = candles[0]
            if not latest_candle:
                return False, "invalid_latest_candle"

            open_price = latest_candle.get("open", 0)
            high_price = latest_candle.get("high", 0)
            low_price = latest_candle.get("low", 0)
            close_price = latest_candle.get("close", 0)

            # All OHLC values must be > 0
            if not all([open_price > 0, high_price > 0, low_price > 0, close_price > 0]):
                return False, f"invalid_ohlc (O:{open_price}, H:{high_price}, L:{low_price}, C:{close_price})"

            # Check 4: Recent trading volume (at least some volume in last candle)
            latest_volume = latest_candle.get("volume")
            if latest_volume is None or latest_volume == 0:
                return False, "zero_volume"

            # Check 5: Price movement sanity check (high >= low, reasonable spread)
            if high_price < low_price:
                return False, "invalid_high_low"

            price_range = high_price - low_price
            if price_range == 0:
                # No price movement at all - likely suspended or not trading
                return False, "no_price_movement"

            # Check 6: Price source validation
            price_source = snapshot.get("price_source", "")
            if price_source == "unavailable":
                return False, "price_unavailable"

            # All checks passed
            return True, f"tradable (₹{current_price:.2f}, vol={latest_volume})"

        except Exception as e:
            # If we can't fetch data, treat as untradable
            return False, f"validation_error: {str(e)}"

    for item in news_data:
        hints = _hint_strings(item)
        for hint in hints:
            try:
                row = tech_client.resolve(hint)
            except Exception:
                continue
            if not row:
                continue

            ik = row.get("instrument_key")
            if not ik or not ik.startswith(("NSE_EQ|", "BSE_EQ|")):
                continue
            if ik in seen_ik:
                continue

            sym = row.get("symbol") or hint
            name = row.get("name") or hint

            # Validate symbol format
            if not _is_valid_symbol(sym):
                emit_status(f"⚠️ Skipping invalid symbol format: {sym}")
                continue

            # CRITICAL: Validate stock is actually tradable (has price, volume, liquidity)
            is_tradable, reason = _is_tradable_stock(sym, ik)
            if not is_tradable:
                emit_status(f"⚠️ Skipping {sym}: {reason}")
                continue

            seen_ik.add(ik)

            validated.append({
                "symbol": sym,
                "name": name,
                "instrument_key": ik,
                "source": hint
            })
            emit_status(f"✅ From news: {hint} → {sym} ({name}) - {reason}")

            if len(validated) >= max_symbols:
                break
        if len(validated) >= max_symbols:
            break

    if not validated:
        emit_status("⚠️ No tradable NSE/BSE equities could be resolved from today's news.")

        # FALLBACK: Use popular liquid NSE stocks when news discovery fails (especially for backtest)
        emit_status("🔄 Falling back to popular NSE stocks for analysis...")
        fallback_symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
            "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN"
        ]

        for sym in fallback_symbols:
            try:
                row = tech_client.resolve(sym)
                if not row:
                    continue

                ik = row.get("instrument_key")
                if not ik or not ik.startswith(("NSE_EQ|", "BSE_EQ|")):
                    continue
                if ik in seen_ik:
                    continue

                resolved_sym = row.get("symbol") or sym
                resolved_name = row.get("name") or sym

                # CRITICAL: Validate fallback stock is also tradable
                is_tradable, reason = _is_tradable_stock(resolved_sym, ik)
                if not is_tradable:
                    emit_status(f"⚠️ Skipping fallback {resolved_sym}: {reason}")
                    continue

                seen_ik.add(ik)
                validated.append({
                    "symbol": resolved_sym,
                    "name": resolved_name,
                    "instrument_key": ik,
                    "source": "fallback"
                })
                emit_status(f"✅ Fallback: {resolved_sym} - {reason}")

                if len(validated) >= max_symbols:
                    break
            except Exception as e:
                emit_status(f"⚠️ Fallback error for {sym}: {str(e)}")
                continue

        if validated:
            emit_status(f"✅ Using {len(validated)} fallback symbols for analysis")
        else:
            emit_status("❌ Fallback also failed - no symbols available")
    else:
        emit_status(f"🎉 Discovered {len(validated)} symbols from news for this cycle")

    return validated

def trading_loop(live, max_symbols=20, learning_mode=True):
    global trading_active, current_companies_data
    try:
        trading_active = True
        emit_status(f"🚀 Trading system started (live={'🔴 LIVE' if live else '📝 PAPER'})")
        emit_status(f"⚙️ Settings: max_symbols={max_symbols}, learning={'ON' if learning_mode else 'OFF'}")
        emit_status(f"🧠 Using AI: 4-Phase Intelligent System (Parallel Analysis → Ranking → Risk Validation → Execution)")

        operator = UpstoxOperator() if UpstoxOperator else None
        if operator:
            try:
                status = operator.market_session_status()
                if not status.get("open"):
                    emit_status("⏳ Market closed, waiting for open...")
                    while not operator.market_session_status().get("open"):
                        if not trading_active: return
                        time.sleep(30)
                    emit_status("✅ Market opened!")
            except Exception as e:
                emit_status(f"⚠️ Market status error: {e}")

        validated_symbols = discover_and_validate_symbols(max_symbols)
        if not validated_symbols:
            emit_status("❌ No valid symbols to trade")
            return

        emit_status("🧠 Initializing AI agents...")
        crew = TradingCrew(live=live)

        def _cb(payload: dict):
            try:
                text = _crew_status_to_text(payload)
                if text: emit_status(text)
            except Exception:
                pass

        crew.add_status_callback(_cb)

        emit_status(f"📊 Running decision cycle for {len(validated_symbols)} symbols...")
        symbols_to_trade = [s["symbol"] for s in validated_symbols]

        # Update companies being analyzed for UI
        current_companies_data = [{"symbol": s["symbol"], "name": s.get("name", s["symbol"]), "decision": None} for s in validated_symbols]

        try:
            results = crew.run_decision_cycle(symbols_to_trade)
            decisions = results.get("decisions", [])
            executions = results.get("executions", [])

            # Update company decisions for UI
            for decision in decisions:
                symbol = decision.get("symbol")
                direction = decision.get("direction")
                for comp in current_companies_data:
                    if comp["symbol"] == symbol:
                        comp["decision"] = direction
                        break
            emit_status(f"✅ Cycle complete: {len(decisions)} decisions, {len(executions)} executions")
            buy_count  = sum(1 for d in decisions if d.get("direction") == "BUY")
            sell_count = sum(1 for d in decisions if d.get("direction") == "SELL")
            skip_count = sum(1 for d in decisions if d.get("direction") == "SKIP")
            emit_status(f"📈 Summary: {buy_count} BUY, {sell_count} SELL, {skip_count} SKIP")

            # Run learning analysis if enabled
            if learning_mode and executions:
                try:
                    emit_status("🎓 Running learning analysis...")
                    learning_result = crew.run_learning_mode(days=7)
                    if learning_result:
                        summary = learning_result.get("summary", "Learning complete")
                        emit_status(f"🧠 {summary}")
                except Exception as learning_error:
                    emit_status(f"⚠️ Learning analysis error: {str(learning_error)}")

        except Exception as e:
            emit_status(f"❌ Trading cycle error: {str(e)}")

        emit_status("✅ Trading complete!")

    except Exception as e:
        emit_status(f"❌ System error: {str(e)}")
    finally:
        trading_active = False
        current_companies_data = []  # Clear on stop
        emit_status("⏹️ System stopped")

def status_stream():
    def event_stream():
        while True:
            try:
                try:
                    payload = status_queue.get(timeout=1)
                    yield "data: " + json.dumps(payload) + "\n\n"
                except queue.Empty:
                    yield "data: " + json.dumps({"event": "heartbeat"}) + "\n\n"
            except GeneratorExit:
                break
    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/")
def index():
    """Serve the enhanced dashboard."""
    try:
        dashboard_path = Path(__file__).parent / "dashboard_ui.html"
        if dashboard_path.exists():
            with open(dashboard_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        app.logger.error(f"Dashboard error: {e}")

    # Fallback to old UI if dashboard not found
    return render_template_string(HTML)

@app.route("/health")
def health():
    return jsonify({"ok": True, "time": datetime.now(IST).isoformat()})

@app.route("/status")
def get_status():
    market = {"open": False, "status": "UNKNOWN", "time": datetime.now(IST).strftime("%H:%M:%S")}
    funds_equity = {"available_margin": 0.0}
    # guard broker calls
    if UpstoxOperator:
        try:
            op = UpstoxOperator()
            market = op.market_session_status()
            funds_data = op.get_funds()
            funds_equity = (funds_data or {}).get("equity", {}) or funds_equity
        except Exception as e:
            app.logger.warning("status error: %s", e)
    return jsonify({
        "market": market,
        "funds": funds_equity,
        "trading_active": trading_active,
        "timestamp": datetime.now(IST).isoformat()
    })

@app.route("/start", methods=["POST"])
def start_trading():
    global trading_active
    if trading_active:
        return jsonify({"status": "error", "message": "Already running"})
    data = request.json or {}
    live = bool(data.get("live", False))
    max_symbols = int(data.get("max_symbols", 20))
    learning_mode = bool(data.get("learning_mode", True))
    threading.Thread(target=trading_loop, args=(live, max_symbols, learning_mode), daemon=True).start()
    return jsonify({"status": "ok"})

@app.route("/stop", methods=["POST"])
def stop_trading():
    global trading_active
    trading_active = False
    return jsonify({"status": "ok"})

@app.route("/learning", methods=["POST"])
def run_learning():
    """Run learning analysis on recent trades."""
    try:
        emit_status("🎓 Starting learning analysis...")
        crew = TradingCrew(live=False)
        learning_result = crew.run_learning_mode(days=30)

        insights = learning_result.get("summary", "Analysis complete")
        emit_status(f"✅ Learning complete: {insights}")

        return jsonify({
            "status": "ok",
            "insights": insights,
            "result": learning_result
        })
    except Exception as e:
        app.logger.error(f"Learning error: {e}")
        emit_status(f"❌ Learning failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route("/wallet")
def get_wallet():
    """Get wallet/money management data for both live and paper trading."""
    try:
        from money_manager import get_money_manager
        from trade_tracker import get_trade_tracker

        mm = get_money_manager()
        tracker = get_trade_tracker()

        wallet_status = mm.get_wallet_status()
        daily_pnl = tracker.get_daily_pnl()

        return jsonify({
            "status": "ok",
            "total_capital": wallet_status.get("total_capital", 0),
            "available_capital": wallet_status.get("available_capital", 0),
            "used_capital": wallet_status.get("used_capital", 0),
            "daily_pnl": daily_pnl.get("net_pnl", 0),
            "can_trade": wallet_status.get("can_trade", False),
            "max_positions": wallet_status.get("max_positions", 0),
        })
    except Exception as e:
        app.logger.error(f"Wallet error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/positions")
def get_positions():
    """Get all open positions."""
    try:
        from trade_tracker import get_trade_tracker
        from upstox_technical import UpstoxTechnicalClient

        tracker = get_trade_tracker()
        tech_client = UpstoxTechnicalClient() if UpstoxTechnicalClient else None

        open_positions = tracker.get_open_positions()

        # Enrich with current price if available
        if tech_client:
            for pos in open_positions:
                try:
                    ltp, _ = tech_client.ltp(pos.get("instrument_key", pos["symbol"]))
                    if ltp:
                        pos["ltp"] = float(ltp)
                        entry = pos.get("entry_price", 0)
                        qty = pos.get("quantity", 0)
                        side = pos.get("side", "BUY")

                        if side == "BUY":
                            pos["unrealized_pnl"] = (ltp - entry) * qty
                        else:
                            pos["unrealized_pnl"] = (entry - ltp) * qty
                except Exception:
                    pass

        return jsonify({
            "status": "ok",
            "positions": open_positions
        })
    except Exception as e:
        app.logger.error(f"Positions error: {e}")
        return jsonify({"status": "error", "message": str(e), "positions": []})

@app.route("/holdings")
def get_holdings():
    """Get current holdings."""
    try:
        if not UpstoxOperator:
            return jsonify({"status": "ok", "holdings": []})

        operator = UpstoxOperator()
        holdings_data = operator.get_holdings()

        if holdings_data.get("status") == "ok":
            return jsonify({
                "status": "ok",
                "holdings": holdings_data.get("holdings", [])
            })
        else:
            return jsonify({"status": "ok", "holdings": []})
    except Exception as e:
        app.logger.error(f"Holdings error: {e}")
        return jsonify({"status": "ok", "holdings": []})

@app.route("/trades")
def get_trades():
    """Get today's trade history."""
    try:
        from trade_tracker import get_trade_tracker

        tracker = get_trade_tracker()
        daily_pnl = tracker.get_daily_pnl()

        return jsonify({
            "status": "ok",
            "trades": daily_pnl.get("trades", [])
        })
    except Exception as e:
        app.logger.error(f"Trades error: {e}")
        return jsonify({"status": "ok", "trades": []})

@app.route("/companies")
def get_companies():
    """Get companies currently being analyzed."""
    global current_companies_data

    try:
        return jsonify({
            "status": "ok",
            "companies": current_companies_data
        })
    except Exception as e:
        return jsonify({"status": "ok", "companies": []})

@app.route("/stream")
def stream():
    return status_stream()

@app.route("/dashboard")
def dashboard():
    """Serve the enhanced dashboard."""
    try:
        dashboard_path = Path(__file__).parent / "dashboard_ui.html"
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Dashboard not found", 404

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 AI TRADING SYSTEM - AGENT-BASED VERSION")
    print("="*70)
    print("🔗 Open: http://localhost:5000")
    print("⏰ Time:", datetime.now(IST).strftime("%H:%M:%S IST"))
    print("="*70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
