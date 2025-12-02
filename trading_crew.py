#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trading_crew.py — Main orchestration for the AI trading system (brace-safe)
===========================================================================

- Uses string.Template with $placeholders so JSON braces never break formatting.
- Matches agents.py contracts (JSON-only agents, Executor uses place_order_tool).
- Streams status via callbacks, persists decisions and ledger.
"""

from __future__ import annotations

import os
import json
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional, Sequence
from pathlib import Path
from string import Template
import concurrent.futures

from crewai import Crew, Task, Process

# Suppress LiteLLM debug logs (used by CrewAI for LLM calls)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Agents & tools
from agents import create_all_agents
from crew_tools import (
    get_recent_news_tool,
    search_news_tool,
    get_technical_snapshot_tool,
    get_market_status_tool,
    get_funds_tool,
    get_positions_tool,
    get_holdings_tool,
    get_portfolio_summary_tool,
    calculate_margin_tool,
    calculate_max_quantity_tool,
    place_order_tool,
    square_off_tool,
    calculate_trade_metrics_tool,
    get_current_time_tool,
    round_to_tick_tool,
    calculate_atr_stop_tool,
)

# Direct clients (optional/fallbacks)
from upstox_technical import UpstoxTechnicalClient
from news_client import NewsClient

# Trade tracker for P&L calculation
from trade_tracker import TradeTracker

# Market context for sentiment and breadth analysis
from market_context import MarketContext

# Position monitor for tracking SL/target hits
from position_monitor import PositionMonitor

# Money manager for capital allocation
from money_manager import MoneyManager

# Learning engine for continuous improvement
from learning_engine import LearningEngine

# Opportunity ranking for multi-stock comparison
from opportunity_ranker import OpportunityRanker

# Portfolio risk management
from portfolio_risk_manager import PortfolioRiskManager

# Imperative operator for direct actions (optional)
try:
    import upstox_operator as upop
    _OpClass = None
    for _name in ("UpstoxOperator", "Operator", "BrokerOperator"):
        _OpClass = getattr(upop, _name, _OpClass)
    UpstoxOperator = _OpClass or None
except Exception:
    UpstoxOperator = None

IST = ZoneInfo(os.environ.get("TZ", "Asia/Kolkata"))
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


def _tmpl(text: str, **kw) -> str:
    """Brace-safe tiny templater using $placeholders (no str.format)."""
    skw = {
        k: (
            v
            if isinstance(v, str)
            else json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
            else str(v)
        )
        for k, v in kw.items()
    }
    return Template(text).substitute(**skw)


class TradingCrew:
    """Main trading crew orchestrator."""

    def __init__(
        self,
        live: bool = False,
        wait_for_open: bool = False,
        min_confidence_gate: Optional[float] = None,
        crew_verbose: bool = None,
    ):
        self.today = datetime.now(IST).strftime("%Y-%m-%d")
        self.live = bool(live)
        self.wait_for_open = bool(wait_for_open)
        self.min_confidence_gate = min_confidence_gate

        # Crew verbosity (default from env, can be overridden)
        if crew_verbose is None:
            self.crew_verbose = os.environ.get("CREW_VERBOSE", "true").lower() in ("true", "1", "yes")
        else:
            self.crew_verbose = bool(crew_verbose)

        # File paths
        self.holdings_file = DATA_DIR / "holdings.json"
        self.decisions_file = DATA_DIR / f"decisions-{self.today}.json"
        self.ledger_file = DATA_DIR / "ledger.jsonl"
        self.memory_file = DATA_DIR / "memory.json"
        self.incidents_file = DATA_DIR / "incidents.jsonl"

        # Load memory
        self.memory = self._load_memory()
        if self.min_confidence_gate is not None:
            self.memory["confidence_gate"] = float(self.min_confidence_gate)

        # Initialize clients
        print("🔧 Initializing clients...")
        try:
            self.tech = UpstoxTechnicalClient()
            print("✅ Technical client initialized")
        except Exception as e:
            print(f"⚠️ Technical client init failed: {e}")
            self.tech = None

        try:
            self.news = NewsClient()
            print("✅ News client initialized")
        except Exception as e:
            print(f"⚠️ News client init failed: {e}")
            self.news = None

        try:
            self.operator = UpstoxOperator() if UpstoxOperator else None
            print("✅ Operator initialized" if self.operator else "⚠️ Operator not available")
        except Exception as e:
            print(f"⚠️ Operator init failed: {e}")
            self.operator = None

        # Initialize trade tracker for P&L tracking
        try:
            self.tracker = TradeTracker()
            print("✅ Trade tracker initialized")
        except Exception as e:
            print(f"⚠️ Trade tracker init failed: {e}")
            self.tracker = None

        # Initialize market context analyzer
        try:
            self.market_context = MarketContext(tech_client=self.tech)
            print("✅ Market context analyzer initialized")
        except Exception as e:
            print(f"⚠️ Market context init failed: {e}")
            self.market_context = None

        # Initialize position monitor
        try:
            self.position_monitor = PositionMonitor(
                operator=self.operator,
                tech_client=self.tech,
                trade_tracker=self.tracker
            )
            print("✅ Position monitor initialized")
        except Exception as e:
            print(f"⚠️ Position monitor init failed: {e}")
            self.position_monitor = None

        # Initialize money manager
        try:
            self.money_manager = MoneyManager(operator=self.operator)
            print("✅ Money manager initialized")
        except Exception as e:
            print(f"⚠️ Money manager init failed: {e}")
            self.money_manager = None

        # Initialize learning engine
        try:
            self.learning_engine = LearningEngine(trade_tracker=self.tracker)
            print("✅ Learning engine initialized")
        except Exception as e:
            print(f"⚠️ Learning engine init failed: {e}")
            self.learning_engine = None

        # Initialize opportunity ranker
        try:
            self.opportunity_ranker = OpportunityRanker(learning_engine=self.learning_engine)
            print("✅ Opportunity ranker initialized")
        except Exception as e:
            print(f"⚠️ Opportunity ranker init failed: {e}")
            self.opportunity_ranker = None

        # Initialize portfolio risk manager (sector concentration disabled)
        try:
            self.portfolio_risk = PortfolioRiskManager(max_sector_concentration=100.0)
            print("✅ Portfolio risk manager initialized (sector limits disabled)")
        except Exception as e:
            print(f"⚠️ Portfolio risk manager init failed: {e}")
            self.portfolio_risk = None

        # Initialize agents with tools
        all_tools = [
            get_recent_news_tool,
            search_news_tool,
            get_technical_snapshot_tool,
            get_market_status_tool,
            get_funds_tool,
            get_positions_tool,
            get_holdings_tool,
            get_portfolio_summary_tool,
            calculate_margin_tool,
            calculate_max_quantity_tool,
            place_order_tool,
            square_off_tool,
            calculate_trade_metrics_tool,
            get_current_time_tool,
            round_to_tick_tool,
            calculate_atr_stop_tool,
        ]
        self.agents = create_all_agents(all_tools)
        print(f"✅ Initialized {len(self.agents)} agents with {len(all_tools)} tools")

        # Status stream callbacks (UI bridge)
        self.status_callbacks = []

    # -------------------------------
    # Persistence helpers
    # -------------------------------
    def _load_memory(self) -> Dict[str, Any]:
        if self.memory_file.exists():
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "w_news": 0.45,
            "w_tech": 0.55,
            "confidence_gate": 0.50,
            "risk_base_pct": 0.60,
            "blacklist": [],
            "symbol_notes": {},
            "last_update": None,
        }

    def _save_memory(self):
        self.memory["last_update"] = datetime.now(IST).isoformat()
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    def _load_holdings(self) -> List[Dict[str, Any]]:
        if self.holdings_file.exists():
            with open(self.holdings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_holdings(self, holdings: List[Dict[str, Any]]):
        with open(self.holdings_file, "w", encoding="utf-8") as f:
            json.dump(holdings, f, indent=2, ensure_ascii=False)

    def _append_ledger(self, entry: Dict[str, Any]):
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _save_decisions(self, decisions: Dict[str, Any]):
        with open(self.decisions_file, "w", encoding="utf-8") as f:
            json.dump(decisions, f, indent=2, ensure_ascii=False)

    def _log_incident(self, incident: Dict[str, Any]):
        incident["timestamp"] = datetime.now(IST).isoformat()
        with open(self.incidents_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(incident, ensure_ascii=False) + "\n")

    # -------------------------------
    # UI status helpers
    # -------------------------------
    def _emit_status(self, event: str, data: Dict[str, Any]):
        payload = {
            "timestamp": datetime.now(IST).isoformat(),
            "event": event,
            "data": data,
        }
        for callback in self.status_callbacks:
            try:
                callback(payload)
            except Exception as e:
                print(f"Status callback error: {e}")

    def add_status_callback(self, callback):
        self.status_callbacks.append(callback)

    # -------------------------------
    # Market session utility
    # -------------------------------
    def _wait_until_open_if_needed(self):
        if not (self.wait_for_open and self.mode == "live"):
            return
        if not self.operator:
            print("⚠️ Operator not available, skipping market wait")
            return

        self._emit_status("market_wait_start", {})
        print("⏳ Waiting for market to open...")

        try:
            while True:
                status = self.operator.market_session_status()
                is_open = status.get("open", False)
                phase = status.get("status", "UNKNOWN")
                self._emit_status("market_status", {"open": is_open, "phase": phase})
                if is_open:
                    print(f"✅ Market is open (phase: {phase})")
                    break
                print(f"⏳ Market closed (phase: {phase}), waiting...")
                time.sleep(30)
        except Exception as e:
            print(f"⚠️ Error waiting for market: {e}")
            self._emit_status("market_wait_error", {"error": str(e)})

    # -------------------------------
    # Holdings review
    # -------------------------------
    def review_holdings(self) -> List[Dict[str, Any]]:
        self._emit_status("review_holdings_start", {})

        holdings = self._load_holdings()
        actions: List[Dict[str, Any]] = []

        if not holdings:
            self._emit_status(
                "review_holdings_complete",
                {"actions": [], "message": "No holdings to review"},
            )
            return []

        for holding in holdings:
            symbol = holding["symbol"]
            self._emit_status("reviewing_holding", {"symbol": symbol})

            desc = _tmpl(
                """Review swing holding for $symbol.

Current position:
$pos

Tasks:
1) Fetch recent news (last 24h) for $symbol
2) Check for negative shocks (downgrades, earnings miss, issues)
3) Check if +1R achieved (current price vs entry/target)
4) Recommend one: HOLD | SQUARE-OFF | TRAIL-TO-BE

Return JSON only:
{"recommendation":"HOLD|SQUARE-OFF|TRAIL-TO-BE","notes":"..."}
""",
                symbol=symbol,
                pos=holding,
            )

            task = Task(
                description=desc, agent=self.agents["monitor"], expected_output="JSON"
            )

            crew = Crew(
                agents=[self.agents["monitor"]],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )

            result = str(crew.kickoff()).strip()

            try:
                parsed = json.loads(result) if result.startswith("{") else {}
                rec = (parsed.get("recommendation") or "").upper()
                if not rec:
                    up = result.upper()
                    if "SQUARE-OFF" in up:
                        rec = "SQUARE-OFF"
                    elif "TRAIL" in up or "BREAKEVEN" in up or "BE" in up:
                        rec = "TRAIL-TO-BE"
                    else:
                        rec = "HOLD"

                action = {
                    "symbol": symbol,
                    "action": rec.replace("-", "_").lower(),
                    "reason": parsed.get("notes") or result,
                    "timestamp": datetime.now(IST).isoformat(),
                }

                if rec == "SQUARE-OFF" and self.live and self.operator:
                    try:
                        exec_res = self.operator.square_off(symbol=symbol, live=True)

                        # Record exit in trade tracker if successful
                        if self.tracker and exec_res.get("status") == "ok":
                            try:
                                # Get current price for P&L calculation
                                current_price = None
                                if self.tech:
                                    px, _ = self.tech.ltp(holding.get("instrument_key", ""))
                                    current_price = float(px) if px else None

                                if current_price:
                                    pnl_record = self.tracker.record_exit(
                                        symbol=symbol,
                                        exit_price=current_price,
                                        exit_reason="SWING_REVIEW_RECOMMENDATION",
                                        order_id=exec_res.get("order_id"),
                                    )
                                    action["pnl_record"] = pnl_record
                                    print(f"💰 P&L recorded: {symbol} → ₹{pnl_record.get('net_pnl', 0):.2f}")
                            except Exception as e:
                                print(f"⚠️ Error recording exit P&L: {e}")

                    except Exception as e:
                        exec_res = {"ok": False, "error": str(e)}
                    action["execution"] = exec_res

                actions.append(action)
                self._emit_status("holding_action", action)

            except Exception as e:
                self._log_incident(
                    {
                        "type": "holding_review_error",
                        "symbol": symbol,
                        "error": str(e),
                        "raw": result,
                    }
                )

        self._emit_status("review_holdings_complete", {"actions": actions})
        return actions

    # REMOVED: square_off_intraday_positions - No time-based square-off needed
    # All positions are delivery with permanent monitoring via position_monitor
    # Position monitor will square off when target or stop-loss is hit (permanent)

    # -------------------------------
    # Decision (news + tech + market context)
    # -------------------------------
    def decide_trade(self, symbol: str, market_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._emit_status("decide_start", {"symbol": symbol})

        if symbol in self.memory.get("blacklist", []):
            decision = {
                "symbol": symbol,
                "direction": "SKIP",
                "reason": "blacklisted",
                "timestamp": datetime.now(IST).isoformat(),
            }
            self._emit_status("decide_complete", decision)
            return decision

        news_desc = _tmpl(
            """Analyze news sentiment for $symbol (Mode $mode, Date $today).
Steps:
1) Fetch 1–2 days of news & broker calls
2) Score sentiment in [-1, +1] with time-decay (half-life ~18h)
3) Summarize key drivers

Return JSON only: {"news_score": <float>, "summary": "..."}
""",
            symbol=symbol,
            mode=self.mode,
            today=self.today,
        )

        tech_desc = _tmpl(
            """Analyze technicals for $symbol (Mode $mode, Date $today).
Use 30m (short-term) and Daily (trend). Extract RSI, EMA20/EMA50, MACD-hist, VWAP gap, ATR%.

Return JSON only:
{
  "ref_price": <float>,
  "indicators": {"rsi14":..,"ema20":..,"ema50":..,"atr_pct":..,"vwap_gap_pct":..},
  "tf": {"m30":{"trend":"UP|DOWN|FLAT","strength":0..1}, "d1":{"trend":"UP|DOWN|FLAT","strength":0..1}}
}
""",
            symbol=symbol,
            mode=self.mode,
            today=self.today,
        )

        mem_view = {
            k: self.memory[k] for k in ["w_news", "w_tech", "confidence_gate"]
        }

        # Prepare market context summary for decision agent
        market_summary = "Not available"
        if market_ctx:
            nifty = market_ctx.get("nifty", {})
            breadth = market_ctx.get("breadth", {})
            combined = market_ctx.get("combined_assessment", "UNKNOWN")

            market_summary = f"""
Nifty: {nifty.get('current_price', 0):.0f} ({nifty.get('change_percent', 0):+.2f}%)
Trend: {nifty.get('trend', 'UNKNOWN')} | Sentiment: {nifty.get('sentiment', 'NEUTRAL')}
Trading Bias: {nifty.get('trading_bias', 'SELECTIVE')}
"""
            if breadth and not breadth.get("error"):
                market_summary += f"""Breadth: {breadth.get('advance_percent', 50):.0f}% advancing, {breadth.get('above_ema_percent', 50):.0f}% above EMA20
Breadth Sentiment: {breadth.get('breadth_sentiment', 'NEUTRAL')}
"""
            market_summary += f"""Combined Assessment: {combined}
Agent Guidance: {nifty.get('agent_guidance', 'No guidance available')}"""

        decision_desc = _tmpl(
            """Make a trading decision for $symbol by synthesizing News + Technicals + Market Context.

Memory:
$memory

MARKET CONTEXT (Nifty & Breadth):
$market_context

IMPORTANT: Consider market context when making decisions:
- In BEARISH markets (Nifty down, weak breadth): Be more conservative, require higher confidence, prefer defensive stocks
- In BULLISH markets (Nifty up, strong breadth): Can be more aggressive with good setups
- In MIXED/NEUTRAL markets: Stock-specific analysis more important, use normal thresholds

CRITICAL: Determine STYLE (intraday vs swing) based on timeframe strengths.

STYLE SELECTION:
- If m30_strength ≥ 0.70 AND aligned with d1 → style="intraday"
- If d1_strength strong but m30 weaker → style="swing"
- If both weak → SKIP

CONFIDENCE CALCULATION (product-specific):

FOR INTRADAY:
- Base confidence = 0.70 × m30_strength + 0.30 × news_score
- Adjust for market: +0.05 if market bullish, -0.05 if bearish
- Gate: ≥0.60 (higher bar)

FOR SWING:
- Base confidence = 0.55 × d1_strength + 0.45 × news_score
- Adjust for market: +0.05 if market bullish, -0.05 if bearish
- Gate: ≥0.50

ALIGNMENT:
1) Both align (news + tech same direction) → use that direction
2) Conflict → require dominance:
   - news_score ≥ ±0.70 OR
   - tech_strength ≥ 0.75
3) Otherwise → SKIP

MARKET OVERRIDE:
- If market is STRONG_BEARISH and stock signal is BUY: increase confidence requirement to 0.70
- If market is STRONG_BULLISH and stock signal is BUY: can lower requirement to 0.55 (intraday) / 0.45 (swing)

Return JSON only (include style!):
{"direction":"BUY|SELL|SKIP","confidence":0..1,"style":"intraday|swing","rationale":"..."}
""",
            symbol=symbol,
            memory=mem_view,
            gate=self.memory.get("confidence_gate", 0.50),
            market_context=market_summary,
        )

        news_task = Task(
            description=news_desc,
            agent=self.agents["news"],
            expected_output="JSON",
        )
        tech_task = Task(
            description=tech_desc,
            agent=self.agents["technical"],
            expected_output="JSON",
        )
        decision_task = Task(
            description=decision_desc,
            agent=self.agents["lead"],
            expected_output="JSON",
            context=[news_task, tech_task],
        )

        crew = Crew(
            agents=[self.agents["news"], self.agents["technical"], self.agents["lead"]],
            tasks=[news_task, tech_task, decision_task],
            process=Process.sequential,
            verbose=self.crew_verbose,
        )

        self._emit_status("decision_analyzing", {"symbol": symbol})
        result = str(crew.kickoff()).strip()

        try:
            parsed = json.loads(result) if result.startswith("{") else {}
            direction = (parsed.get("direction") or "SKIP").upper()
            conf = parsed.get("confidence", None)

            decision = {
                "symbol": symbol,
                "direction": direction,
                "confidence": conf,
                "raw": result,
                "timestamp": datetime.now(IST).isoformat(),
            }
            self._emit_status("decide_complete", decision)
            return decision

        except Exception as e:
            self._log_incident(
                {
                    "type": "decision_parse_error",
                    "symbol": symbol,
                    "error": str(e),
                    "raw_result": result,
                }
            )
            decision = {
                "symbol": symbol,
                "direction": "SKIP",
                "reason": "parse_error",
                "timestamp": datetime.now(IST).isoformat(),
            }
            self._emit_status("decide_complete", decision)
            return decision

    # -------------------------------
    # Sizing + Execution
    # -------------------------------
    def _fresh_snapshot(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        if not self.tech:
            raise RuntimeError("Technical client not initialized")
        try:
            return self.tech.snapshot(symbol, days=days)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch snapshot for {symbol}: {e}")

    def size_and_execute(
        self, symbol: str, direction: str, confidence: float
    ) -> Dict[str, Any]:
        if direction == "SKIP":
            return {
                "symbol": symbol,
                "status": "skipped",
                "reason": "direction_skip",
            }

        self._emit_status("sizing_start", {"symbol": symbol, "direction": direction})

        # 1) Fresh technical snapshot (for ATR etc.)
        try:
            tech_snap = self._fresh_snapshot(symbol, days=7)
            print(f"✅ Snapshot {symbol}: price={tech_snap.get('current_price')}")
        except Exception as e:
            print(f"❌ Snapshot error for {symbol}: {e}")
            self._emit_status("snapshot_error", {"symbol": symbol, "error": str(e)})
            return {
                "symbol": symbol,
                "status": "skipped",
                "reason": "snapshot_error",
                "error": str(e),
            }

        # 2) 🆕 ENTRY VALIDATION - Check if NOW is a good time to enter
        self._emit_status("entry_validation_start", {"symbol": symbol, "direction": direction})

        entry_validation_desc = _tmpl(
            """Entry quality check for $symbol ($direction, conf=$confidence).

Technical Data: $snapshot

SIMPLIFIED SCORING (start at 60, trust decision agent):
- Base: 60 points
- RSI 40-75: +10
- RSI >80 or <20: -20
- Price >3% from VWAP: -10
- Good time (9:20-11:30, 14:00-14:45): +10
- Bad time (last 10 min): -15

RULES:
- Score ≥60: ENTER_NOW (allow trade)
- Score 40-59: WATCHLIST
- Score <40: SKIP

Return ONLY this JSON (NO arrays, NO nested objects):
{{
  "entry_decision": "ENTER_NOW",
  "entry_quality_score": 70,
  "reason": "RSI healthy timing good"
}}
""",
            symbol=symbol,
            direction=direction,
            confidence=f"{confidence:.3f}",
            snapshot=tech_snap,
        )

        entry_task = Task(
            description=entry_validation_desc,
            agent=self.agents["entry_validator"],
            expected_output="JSON",
        )

        entry_crew = Crew(
            agents=[self.agents["entry_validator"]],
            tasks=[entry_task],
            process=Process.sequential,
            verbose=self.crew_verbose,
        )

        entry_result_str = str(entry_crew.kickoff()).strip()
        self._emit_status("entry_validation_complete", {"symbol": symbol, "result": entry_result_str})

        # Parse entry validation result
        try:
            # Extract JSON from text if needed
            if "{" in entry_result_str:
                json_start = entry_result_str.find("{")
                json_end = entry_result_str.rfind("}") + 1
                if json_end > json_start:
                    entry_result_str = entry_result_str[json_start:json_end]

            # Try parsing
            try:
                entry_validation = json.loads(entry_result_str)
            except json.JSONDecodeError:
                # CHANGED: Default to ENTER_NOW if JSON is malformed (trust the decision agent)
                print(f"⚠️ Entry validation malformed JSON, trusting decision agent (confidence={confidence:.2f})")
                entry_validation = {
                    "entry_decision": "ENTER_NOW",
                    "entry_quality_score": 65,
                    "reason": "Validator failed, trusting decision agent"
                }

            entry_decision = (entry_validation.get("entry_decision") or "ENTER_NOW").upper()
            entry_quality = int(entry_validation.get("entry_quality_score") or 65)
            entry_reason = entry_validation.get("reason", "")

            # SAFETY: If validator returns 0 score but decision agent had good confidence, override to ENTER_NOW
            if entry_quality == 0 and confidence >= 0.60:
                print(f"⚠️ Overriding 0 score: decision confidence {confidence:.2f} >= 0.60, allowing entry")
                entry_decision = "ENTER_NOW"
                entry_quality = 65
                entry_reason = "Overridden: trusting decision agent confidence"

            print(f"📊 Entry Quality: {entry_quality}/100 → {entry_decision}")
            print(f"   Reason: {entry_reason}")

            if entry_decision == "SKIP":
                return {
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": "entry_quality_too_low",
                    "entry_quality_score": entry_quality,
                    "entry_reason": entry_reason,
                }

            elif entry_decision == "WATCHLIST":
                # Add to watchlist for monitoring
                print(f"📋 Adding {symbol} to watchlist (quality: {entry_quality})")
                wait_for = entry_validation.get("wait_for", "better entry")

                # Import and use watchlist manager
                try:
                    from watchlist_manager import get_watchlist_manager
                    wm = get_watchlist_manager()
                    wm.add_to_intraday_watchlist(
                        symbol=symbol,
                        signal=direction,
                        reason=entry_reason,
                        entry_target=None,  # Could parse from wait_for
                        current_price=tech_snap.get("current_price"),
                        confidence=confidence,
                        entry_quality=entry_quality,
                        setup_type="pending_entry",
                        technical_data=tech_snap,
                    )
                    print(f"✅ {symbol} added to watchlist: {wait_for}")
                except Exception as e:
                    print(f"⚠️ Error adding to watchlist: {e}")

                return {
                    "symbol": symbol,
                    "status": "watchlisted",
                    "reason": "waiting_for_better_entry",
                    "entry_quality_score": entry_quality,
                    "wait_for": wait_for,
                    "entry_reason": entry_reason,
                }

            # If ENTER_NOW, continue to sizing & execution
            print(f"✅ Entry validated - proceeding with execution")

        except Exception as e:
            print(f"⚠️ Entry validation parse error: {e}")
            # If validation fails, default to cautious (skip)
            return {
                "symbol": symbol,
                "status": "skipped",
                "reason": "entry_validation_parse_error",
                "error": str(e),
            }

        # 3) FETCH REAL FUNDS FIRST - Don't trust agent to call tools reliably!
        try:
            real_funds = self.operator.get_funds()
            real_available = float((real_funds.get("equity") or {}).get("available_margin", 0) or 0)
            if real_available <= 0:
                logger.warning("⚠️ No available funds! Cannot trade. Funds: %s", real_funds)
                return {
                    "decision": "SKIP",
                    "symbol": symbol,
                    "reason": "no_available_funds",
                    "available_funds": real_available,
                }
        except Exception as e:
            logger.exception("Failed to fetch real funds from Upstox")
            return {
                "decision": "SKIP",
                "symbol": symbol,
                "reason": "funds_fetch_failed",
                "error": str(e),
            }

        # 4) Ask Risk agent for a plan
        risk_desc = _tmpl(
            """Build position plan for $symbol.

Inputs:
- Direction: $direction
- Confidence: $confidence
- Technical Snapshot: $snapshot
- REAL AVAILABLE FUNDS (from Upstox account): ₹$available_funds

⚠️ CRITICAL - YOU MUST USE THE REAL FUNDS PROVIDED ABOVE!
The available funds shown above (₹$available_funds) is fetched from your LIVE Upstox account.
DO NOT make up, assume, or hallucinate any other amount!
YOU MUST USE: available_funds = $available_funds

INSTRUCTIONS:
1) Use EXACTLY ₹$available_funds as your available_funds (DO NOT call get_funds_tool, value already provided!)
2) Use current_price from snapshot as entry price
3) Calculate stop_loss from ATR:
   - For intraday: stop_loss = entry × (1 - atr_pct/100 × 0.9) for BUY
   - For swing: stop_loss = entry × (1 - atr_pct/100 × 1.8) for BUY
4) Call calculate_max_quantity_tool with:
   - symbol: $symbol
   - price: entry price from snapshot
   - product: "D" (delivery)
   - risk_pct: 0.5 to 1.0 (based on confidence)
   - stop_loss: calculated stop loss
5) Calculate target for minimum R:R:
   - Intraday: target = entry + (entry - stop_loss) × 1.3
   - Swing: target = entry + (entry - stop_loss) × 1.6
6) Return a FLAT JSON object (NO NESTING!)

REQUIRED OUTPUT FORMAT (must be flat, all fields at root level):
{
  "symbol": "$symbol",
  "direction": "$direction",
  "side": "$direction",
  "style": "swing",
  "product": "D",
  "qty": <integer from calculate_max_quantity_tool>,
  "entry": <float from snapshot>,
  "stop_loss": <float calculated above>,
  "target": <float calculated above>,
  "order_type": "MARKET",
  "rr_ratio": <float>,
  "available_funds": $available_funds,
  "capital_required": <qty × entry>,
  "capital_remaining": <$available_funds - capital_required>,
  "expected_net_profit": <(target - entry) × qty - estimated_charges>,
  "rationale": "Brief 1-line explanation"
}

IMPORTANT:
- Do NOT nest inside "final_choice", "plan", "intraday", or "swing" keys
- Do NOT include alternative plans in the response
- If qty < 1 → return {"decision":"SKIP","reason":"insufficient_capital"}
- ALWAYS use product="D" (delivery only, no intraday)
- You MUST use available_funds = $available_funds (the REAL value from Upstox)
""",
            symbol=symbol,
            direction=direction,
            confidence=f"{confidence:.3f}",
            snapshot=tech_snap,
            available_funds=f"{real_available:.2f}",
        )

        risk_task = Task(
            description=risk_desc,
            agent=self.agents["risk"],
            expected_output="JSON",
        )

        risk_crew = Crew(
            agents=[self.agents["risk"]],
            tasks=[risk_task],
            process=Process.sequential,
            verbose=self.crew_verbose,
        )

        plan_str = str(risk_crew.kickoff()).strip()
        self._emit_status("sizing_complete", {"symbol": symbol, "plan": plan_str})

        # 3) Parse & validate risk plan
        try:
            plan_obj = (
                json.loads(plan_str) if plan_str.lstrip().startswith("{") else {}
            )
        except Exception as e:
            self._log_incident(
                {
                    "type": "risk_plan_parse_error",
                    "symbol": symbol,
                    "error": str(e),
                    "raw_plan": plan_str,
                }
            )
            return {
                "symbol": symbol,
                "status": "skipped",
                "reason": "risk_plan_parse_error",
            }

        # Explicit SKIP from risk agent
        if (plan_obj.get("decision") or "").upper() == "SKIP":
            return {
                "symbol": symbol,
                "status": "skipped",
                "reason": plan_obj.get("reason", "infeasible"),
            }

        # Unwrap possible nested plan structures: {"final_choice": {...}} / {"plan": {...}} etc.
        chosen = None
        for key in ("final_choice", "plan", "intraday", "swing"):
            v = plan_obj.get(key)
            if isinstance(v, dict):
                chosen = v
                break
        if chosen is None:
            chosen = plan_obj  # assume flat

        # Ensure basic fields (also provide 'side' alias for executor)
        chosen.setdefault("symbol", symbol)
        chosen.setdefault("direction", direction)
        chosen.setdefault("side", direction)

        qty = int(chosen.get("qty") or 0)
        entry = chosen.get("entry")
        stop_loss = chosen.get("stop_loss") or chosen.get("stop")
        stop_loss_pct = chosen.get("stop_loss_pct")

        def _is_num(x: Any) -> bool:
            return isinstance(x, (int, float))

        # NEW: allow either absolute stop_loss OR stop_loss_pct (mandatory SL policy)
        if qty < 1 or not _is_num(entry) or (
            stop_loss is None and stop_loss_pct is None
        ):
            self._log_incident(
                {
                    "type": "risk_plan_invalid",
                    "symbol": symbol,
                    "raw_plan": plan_str,
                    "plan_obj": plan_obj,
                }
            )
            return {
                "symbol": symbol,
                "status": "skipped",
                "reason": "invalid_risk_plan",
            }

        # Clean plan JSON that we pass to executor
        cleaned_plan_str = json.dumps(chosen, ensure_ascii=False)

        # 4) Executor agent: place order using cleaned plan
        exec_desc = _tmpl(
            """Execute trade for $symbol using place_order_tool.

IMPORTANT:
- Use "order_type".
- Pass "live": $live.
- product must be "I" or "D".
- Stop-loss is MANDATORY: provide stop_loss OR stop_loss_pct in the payload.
- target/target_pct are OPTIONAL.

Input (Position Plan JSON from previous step):
$plan

DO:
1) If Mode="$mode" and live=$live, verify market via get_market_status_tool.
2) Build payload with keys:
   symbol, side, qty, product, order_type, price,
   stop_loss, stop_loss_pct, target, target_pct, live, tag.
3) Call place_order_tool and return its JSON result.

Return only JSON like:
{"status":"ok|error","order":{...},"notes":"..."}
""",
            symbol=symbol,
            live=str(self.live).lower(),
            mode=self.mode,
            plan=cleaned_plan_str,
        )

        exec_task = Task(
            description=exec_desc,
            agent=self.agents["executor"],
            expected_output="JSON",
        )

        exec_crew = Crew(
            agents=[self.agents["executor"]],
            tasks=[exec_task],
            process=Process.sequential,
            verbose=self.crew_verbose,
        )

        exec_result = str(exec_crew.kickoff()).strip()

        # Parse execution result to extract trade details
        exec_data = None
        try:
            if "{" in exec_result:
                json_start = exec_result.find("{")
                json_end = exec_result.rfind("}") + 1
                if json_end > json_start:
                    exec_json_str = exec_result[json_start:json_end]
                    exec_data = json.loads(exec_json_str)
        except Exception as e:
            print(f"⚠️ Could not parse execution result for trade tracking: {e}")

        # Record trade entry in tracker ONLY if we can verify order was placed
        trade_record = None
        if self.tracker and exec_data and exec_data.get("status") in ("success", "ok"):
            try:
                # Extract order details
                order_info = exec_data.get("order") or exec_data.get("entry") or {}
                order_id = order_info.get("order_id")

                # CRITICAL: Only record if we have a valid order_id (proof order was placed)
                if not order_id:
                    print(f"⚠️ No order_id in execution result - order may not have been placed!")
                    print(f"   Execution result: {exec_data}")
                else:
                    # Get entry price from chosen plan
                    entry_price = float(chosen.get("entry", 0))
                    stop_loss = float(chosen.get("stop_loss") or 0) if chosen.get("stop_loss") else None
                    target = float(chosen.get("target") or 0) if chosen.get("target") else None

                    if entry_price > 0 and qty > 0:
                        # VERIFICATION: Check if order was actually placed at broker (optional but recommended)
                        order_verified = True  # Assume true by default if we have order_id

                        # If in live mode, try to verify the order exists
                        if self.live and self.operator:
                            try:
                                print(f"🔍 Verifying order placement at broker...")
                                time.sleep(1)  # Small delay for order to register

                                # Try to get positions to verify
                                pos_data = self.operator.get_positions(include_closed=False)
                                positions = pos_data.get("positions", [])

                                # Check if our symbol appears in positions
                                found = any(
                                    p.get("tradingsymbol", "").upper() == symbol.upper() or
                                    p.get("symbol", "").upper() == symbol.upper()
                                    for p in positions
                                )

                                if found:
                                    print(f"✅ Order verified at broker - position exists")
                                else:
                                    print(f"⚠️ Warning: Order ID exists but position not found in broker positions yet")
                                    print(f"   This might be normal if there's a delay. Order ID: {order_id}")
                                    # Don't fail - order might still be processing
                            except Exception as e:
                                print(f"⚠️ Could not verify order at broker: {e}")
                                # Don't fail - we have order_id so likely succeeded

                        # Record the trade
                        trade_record = self.tracker.record_entry(
                            symbol=symbol,
                            side=direction,
                            quantity=qty,
                            entry_price=entry_price,
                            product=chosen.get("product", "I"),
                            order_id=order_id,
                            stop_loss=stop_loss,
                            target=target,
                            strategy=chosen.get("strategy", "ai_decision"),
                            confidence=confidence,
                            tags=[self.mode, chosen.get("style", "unknown")],
                            metadata={
                                "rationale": chosen.get("rationale"),
                                "rr_ratio": chosen.get("rr_ratio"),
                            }
                        )
                        print(f"📊 Trade recorded in tracker: {trade_record.get('trade_id')}")
            except Exception as e:
                print(f"⚠️ Error recording trade entry: {e}")

        result = {
            "symbol": symbol,
            "direction": direction,
            "plan": cleaned_plan_str,
            "execution": exec_result,
            "timestamp": datetime.now(IST).isoformat(),
            "trade_record": trade_record,  # Add tracker record to result
        }
        self._emit_status("execution_complete", result)
        self._append_ledger(result)
        return result

    # -------------------------------
    # Parallel analysis helper
    # -------------------------------
    def _analyze_stocks_parallel(
        self, symbols: Sequence[str], market_ctx: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple stocks in parallel for speed.

        Args:
            symbols: List of symbols to analyze
            market_ctx: Market context for all decisions

        Returns:
            List of decision dicts (one per symbol)
        """
        print(f"\n🔄 Analyzing {len(symbols)} stocks in parallel (max 5 concurrent)...")

        decisions = []
        completed = 0

        # Use ThreadPoolExecutor for parallel analysis (max 5 concurrent to avoid API limits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.decide_trade, symbol, market_ctx): symbol
                for symbol in symbols
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    decision = future.result()
                    decisions.append(decision)
                    completed += 1

                    # Progress indicator
                    direction = decision.get("direction", "SKIP")
                    conf = decision.get("confidence")
                    if conf is not None:
                        print(f"   [{completed}/{len(symbols)}] {symbol}: {direction} (conf={conf:.2f})")
                    else:
                        print(f"   [{completed}/{len(symbols)}] {symbol}: {direction}")

                except Exception as e:
                    print(f"   ❌ {symbol}: Analysis failed - {e}")
                    decisions.append({
                        "symbol": symbol,
                        "direction": "ERROR",
                        "error": str(e),
                        "timestamp": datetime.now(IST).isoformat(),
                    })
                    completed += 1

        print(f"✅ Parallel analysis complete: {len(decisions)} decisions\n")
        return decisions

    # -------------------------------
    # Full cycle
    # -------------------------------
    def run_decision_cycle(self, symbols: Sequence[str]) -> Dict[str, Any]:
        self._emit_status(
            "cycle_start",
            {"symbols": list(symbols), "mode": self.mode, "date": self.today},
        )
        self._wait_until_open_if_needed()

        cycle_results: Dict[str, Any] = {
            "date": self.today,
            "mode": self.mode,
            "live": self.live,
            "start_time": datetime.now(IST).isoformat(),
            "holdings_review": [],
            "decisions": [],
            "executions": [],
            "capital_tracking": {
                "initial_capital": 0.0,
                "final_capital": 0.0,
                "used_capital": 0.0,
                "max_positions": int(os.environ.get("MAX_POSITIONS", "5")),
            },
        }

        # Get initial capital and wallet status
        try:
            if self.money_manager:
                # Get comprehensive wallet status
                wallet_status = self.money_manager.get_wallet_status()
                cycle_results["wallet_status"] = wallet_status

                initial_capital = wallet_status.get("available_capital", 0)
                cycle_results["capital_tracking"]["initial_capital"] = initial_capital

                print(f"\n💰 Wallet Status:")
                print(f"   Total Capital: ₹{wallet_status.get('total_capital', 0):,.2f}")
                print(f"   Available: ₹{wallet_status.get('available_capital', 0):,.2f}")
                print(f"   Used: ₹{wallet_status.get('used_capital', 0):,.2f} ({wallet_status.get('capital_usage_pct', 0):.1f}%)")
                print(f"   Product: Delivery only (no leverage)")
                print(f"   Daily P&L: ₹{wallet_status.get('daily_pnl', 0):,.2f}")
                print(f"   Daily Trades: {wallet_status.get('daily_trades', 0)}/{wallet_status.get('max_daily_trades', 0)}")

                if not wallet_status.get("can_trade"):
                    print(f"\n⚠️ TRADING BLOCKED: {', '.join(wallet_status.get('blocking_reasons', []))}")
                    cycle_results["trading_blocked"] = True
                    cycle_results["blocking_reasons"] = wallet_status.get("blocking_reasons", [])
                else:
                    print(f"\n✅ Trading allowed - All limits OK")

            elif self.operator:
                funds = self.operator.get_funds()
                initial_capital = float(
                    (funds.get("equity") or {}).get("available_margin", 0) or 0
                )
                cycle_results["capital_tracking"]["initial_capital"] = initial_capital
                print(f"💰 Starting capital: ₹{initial_capital:,.2f}")
            else:
                initial_capital = 0.0
                print("⚠️ Operator not available, capital tracking disabled")
        except Exception as e:
            print(f"⚠️ Error fetching wallet status: {e}")
            initial_capital = 0.0

        # Get market context (Nifty + breadth)
        market_ctx = None
        if self.market_context:
            try:
                print("\n📊 Fetching market context...")
                # Get complete context with breadth analysis on provided symbols
                market_ctx = self.market_context.get_complete_market_context(symbols=list(symbols))

                nifty = market_ctx.get("nifty", {})
                breadth = market_ctx.get("breadth", {})

                print(f"\n🔍 Market Context:")

                # Check if Nifty data is valid
                if nifty.get("error"):
                    print(f"   ❌ NIFTY DATA FAILED: {nifty.get('error')}")
                    print(f"   ⚠️  Details: {nifty.get('details', 'No details available')}")
                    print(f"   ⚠️  WARNING: Trading without market context - using conservative approach!")
                    # Set market_ctx to None to force conservative trading
                    market_ctx = None
                else:
                    # Display Nifty data
                    current_price = nifty.get('current_price', 0)
                    change_pct = nifty.get('change_percent', 0)

                    # CRITICAL: Validate Nifty data is not zero
                    if current_price == 0:
                        print(f"   ❌ CRITICAL: Nifty price is 0! Market data is INVALID!")
                        print(f"   ⚠️  WARNING: Proceeding without market context!")
                        market_ctx = None
                    else:
                        print(f"   Nifty: {current_price:.2f} ({change_pct:+.2f}%)")
                        print(f"   Trend: {nifty.get('trend', 'UNKNOWN')} | Sentiment: {nifty.get('sentiment', 'NEUTRAL')}")
                        print(f"   Trading Bias: {nifty.get('trading_bias', 'SELECTIVE')}")

                if breadth and not breadth.get("error"):
                    print(f"\n📈 Market Breadth ({breadth.get('stocks_analyzed', 0)} stocks):")
                    print(f"   Advancing: {breadth.get('advance_percent', 0):.1f}% | Declining: {breadth.get('decline_percent', 0):.1f}%")
                    print(f"   Above EMA20: {breadth.get('above_ema_percent', 0):.1f}%")
                    print(f"   Breadth Sentiment: {breadth.get('breadth_sentiment', 'NEUTRAL')}")

                if market_ctx.get("combined_assessment"):
                    print(f"\n💡 Combined Assessment: {market_ctx.get('combined_assessment')}")
                    print(f"   {market_ctx.get('recommendation', '')}")

                cycle_results["market_context"] = market_ctx
                self._emit_status("market_context_loaded", market_ctx)

            except Exception as e:
                print(f"⚠️ Error fetching market context: {e}")
                market_ctx = None

        holdings_actions = self.review_holdings()
        cycle_results["holdings_review"] = holdings_actions

        # NO time-based square-off needed - position_monitor handles all exits

        # ====================================================================================
        # NEW FLOW: 4-PHASE INTELLIGENT TRADING SYSTEM
        # ====================================================================================

        print(f"\n{'=' * 80}")
        print(f"🚀 INTELLIGENT TRADING SYSTEM - 4 PHASE EXECUTION")
        print(f"{'=' * 80}\n")

        # PHASE 1: PARALLEL ANALYSIS (analyze all stocks at once)
        # =========================================================================
        print(f"📊 PHASE 1: PARALLEL ANALYSIS")
        print(f"   Analyzing {len(symbols)} stocks simultaneously...")
        print(f"{'=' * 80}")

        all_decisions = self._analyze_stocks_parallel(symbols, market_ctx=market_ctx)
        cycle_results["decisions"] = all_decisions

        # Count signals
        buy_signals = [d for d in all_decisions if d.get("direction") == "BUY"]
        sell_signals = [d for d in all_decisions if d.get("direction") == "SELL"]
        skip_signals = [d for d in all_decisions if d.get("direction") == "SKIP"]

        print(f"\n📈 Analysis Summary:")
        print(f"   BUY signals: {len(buy_signals)} (avg conf: {sum(d.get('confidence', 0) for d in buy_signals) / max(len(buy_signals), 1):.2f})")
        print(f"   SELL signals: {len(sell_signals)}")
        print(f"   SKIP signals: {len(skip_signals)}")
        print(f"")

        # PHASE 2: OPPORTUNITY RANKING (pick best trades)
        # =========================================================================
        print(f"{'=' * 80}")
        print(f"🎯 PHASE 2: OPPORTUNITY RANKING & SELECTION")
        print(f"   Ranking opportunities by quality, R:R, and capital efficiency...")
        print(f"{'=' * 80}\n")

        max_positions = cycle_results["capital_tracking"]["max_positions"]

        # Get current positions
        current_positions = []
        if self.operator:
            try:
                pos_data = self.operator.get_positions(include_closed=False)
                current_positions = pos_data.get("positions", [])
            except Exception as e:
                print(f"⚠️ Error fetching positions: {e}")

        # Rank opportunities
        if self.opportunity_ranker:
            ranking_result = self.opportunity_ranker.rank_opportunities(
                decisions=all_decisions,
                available_capital=initial_capital,
                max_positions=max_positions,
                current_positions=current_positions,
            )

            selected_opportunities = ranking_result["selected_opportunities"]
            rejected_opportunities = ranking_result["rejected_opportunities"]

            # Store ranking metadata
            cycle_results["ranking_metadata"] = ranking_result["ranking_metadata"]

            print(f"🏆 TOP OPPORTUNITIES (selected {len(selected_opportunities)} of {len(buy_signals + sell_signals)}):")
            for i, opp in enumerate(selected_opportunities, 1):
                print(
                    f"   {i}. {opp['symbol']:12} | {opp['direction']:4} | "
                    f"Score: {opp['composite_score']:.3f} | Conf: {opp['confidence']:.2f} | "
                    f"R:R: {opp['rr_ratio']:.1f} | Cap: ₹{opp['estimated_capital']:,.0f}"
                )

            if rejected_opportunities:
                print(f"\n❌ Rejected ({len(rejected_opportunities)}):")
                for i, opp in enumerate(rejected_opportunities[:5], 1):  # Show top 5 rejected
                    print(f"   {i}. {opp['symbol']:12} | Score: {opp['composite_score']:.3f} | "
                          f"Reason: Not in top {max_positions} or capital limit reached")
        else:
            # Fallback if ranker not available - use simple selection
            print(f"⚠️ Ranker not available, using simple selection...")
            selected_opportunities = []
            for dec in (buy_signals + sell_signals)[:max_positions]:
                selected_opportunities.append({
                    "symbol": dec["symbol"],
                    "decision": dec,
                    "direction": dec["direction"],
                    "confidence": dec.get("confidence", 0.6),
                    "rr_ratio": 1.5,
                    "estimated_capital": 10000.0,
                })

        # PHASE 3: PORTFOLIO RISK VALIDATION (safety check)
        # =========================================================================
        print(f"\n{'=' * 80}")
        print(f"🛡️ PHASE 3: PORTFOLIO RISK VALIDATION")
        print(f"   Checking sector concentration, correlation, and balance...")
        print(f"{'=' * 80}\n")

        if self.portfolio_risk and selected_opportunities:
            risk_validation = self.portfolio_risk.validate_portfolio_risk(
                selected_opportunities=selected_opportunities,
                current_positions=current_positions,
                total_capital=initial_capital,
            )

            cycle_results["portfolio_risk_validation"] = risk_validation

            if risk_validation["warnings"]:
                print(f"⚠️ Portfolio Risk Warnings:")
                for warning in risk_validation["warnings"]:
                    print(f"   • {warning}")
                print("")

            if risk_validation["blocking_issues"]:
                print(f"🚫 BLOCKING ISSUES - Cannot proceed:")
                for issue in risk_validation["blocking_issues"]:
                    print(f"   • {issue}")
                print(f"\n❌ Execution blocked due to portfolio risk issues")
                selected_opportunities = []  # Block execution
            elif risk_validation["approved"]:
                print(f"✅ Portfolio risk validation PASSED")

            if risk_validation["recommendations"]:
                print(f"\n💡 Recommendations:")
                for rec in risk_validation["recommendations"]:
                    print(f"   • {rec}")

        # PHASE 4: EXECUTION (execute top opportunities)
        # =========================================================================
        print(f"\n{'=' * 80}")
        print(f"⚡ PHASE 4: EXECUTION")
        print(f"   Executing {len(selected_opportunities)} positions...")
        print(f"{'=' * 80}\n")

        executed_positions = 0

        for opp in selected_opportunities:
            try:
                symbol = opp["symbol"]
                direction = opp["direction"]
                confidence = float(opp.get("confidence", 0.6))

                print(f"\n{'─' * 80}")
                print(f"📍 Executing {executed_positions + 1}/{len(selected_opportunities)}: {symbol} {direction}")
                print(f"{'─' * 80}")

                execution = self.size_and_execute(symbol, direction, confidence)
                cycle_results["executions"].append(execution)

                # Track executed positions
                if execution.get("status") not in ("skipped", "error"):
                    exec_data = execution.get("execution")
                    if isinstance(exec_data, str):
                        try:
                            exec_data = json.loads(exec_data)
                        except Exception:
                            pass
                    if isinstance(exec_data, dict) and exec_data.get("status") in ("success", "ok"):
                        executed_positions += 1
                        print(f"✅ Position {executed_positions} opened successfully")
                    else:
                        print(f"⚠️ Execution completed but status unclear")
                else:
                    print(f"⚠️ Position skipped or errored: {execution.get('reason', 'unknown')}")

                time.sleep(0.3)  # Small delay between orders

            except Exception as e:
                print(f"❌ Error executing {symbol}: {e}")
                self._log_incident({
                    "type": "execution_error",
                    "symbol": symbol,
                    "error": str(e),
                })

        print(f"\n{'=' * 80}")
        print(f"✅ EXECUTION PHASE COMPLETE")
        print(f"   Positions opened: {executed_positions}/{len(selected_opportunities)}")
        print(f"{'=' * 80}\n")

        # Final capital tracking
        if self.operator and initial_capital > 0:
            try:
                final_funds = self.operator.get_funds()
                final_capital = float(
                    (final_funds.get("equity") or {}).get("available_margin", 0) or 0
                )
                cycle_results["capital_tracking"]["final_capital"] = final_capital
                cycle_results["capital_tracking"]["used_capital"] = (
                    initial_capital - final_capital
                )
                print(f"\n💰 Final capital: ₹{final_capital:,.2f} (used: ₹{initial_capital - final_capital:,.2f})")
            except Exception as e:
                print(f"⚠️ Error fetching final capital: {e}")

        # Get P&L summary for today
        if self.tracker:
            try:
                daily_pnl = self.tracker.get_daily_pnl(self.today)
                cycle_results["pnl_summary"] = daily_pnl

                print(f"\n📊 Today's P&L Summary ({self.today}):")
                print(f"   Total Trades: {daily_pnl.get('total_trades', 0)}")
                print(f"   Gross P&L: ₹{daily_pnl.get('gross_pnl', 0):,.2f}")
                print(f"   Charges: ₹{daily_pnl.get('charges', 0):,.2f}")
                print(f"   Net P&L: ₹{daily_pnl.get('net_pnl', 0):,.2f}")
                print(f"   Win Rate: {daily_pnl.get('win_rate', 0):.1f}%")
                print(f"   Winners: {daily_pnl.get('winning_trades', 0)} | Losers: {daily_pnl.get('losing_trades', 0)}")

                # Get overall trading statistics
                stats = self.tracker.get_trade_statistics()
                if not stats.get("error"):
                    cycle_results["trading_statistics"] = stats
                    print(f"\n📈 Overall Statistics:")
                    print(f"   Total Trades: {stats.get('total_trades', 0)}")
                    print(f"   Overall Win Rate: {stats.get('win_rate', 0):.1f}%")
                    print(f"   Total P&L: ₹{stats.get('total_pnl', 0):,.2f}")
                    print(f"   Avg P&L/Trade: ₹{stats.get('average_pnl_per_trade', 0):,.2f}")
                    print(f"   Profit Factor: {stats.get('profit_factor', 0):.2f}")

            except Exception as e:
                print(f"⚠️ Error fetching P&L summary: {e}")

        # Check open positions for SL/target hits
        if self.position_monitor:
            try:
                print(f"\n🔍 Checking open positions for SL/target hits...")
                position_check = self.position_monitor.check_positions(live=self.live)

                cycle_results["position_check"] = position_check

                if position_check.get("actions_taken"):
                    print(f"\n⚡ Position Actions:")
                    for action in position_check["actions_taken"]:
                        symbol = action.get("symbol")
                        reason = action.get("exit_reason")
                        pnl_record = action.get("pnl_record", {})
                        net_pnl = pnl_record.get("net_pnl", 0)

                        print(f"   {symbol}: {reason} → P&L: ₹{net_pnl:,.2f}")

                        # Update money manager with this trade result
                        if self.money_manager and pnl_record:
                            self.money_manager.record_trade_result(
                                net_pnl=net_pnl,
                                product=pnl_record.get("product", "I")
                            )
                else:
                    print(f"   No position exits triggered")

                # Show position summary
                pos_summary = self.position_monitor.get_position_summary()
                if pos_summary.get("total_positions", 0) > 0:
                    print(f"\n📊 Open Positions: {pos_summary.get('total_positions', 0)} (all delivery)")
                    print(f"   Unrealized P&L: ₹{pos_summary.get('total_unrealized_pnl', 0):,.2f}")

            except Exception as e:
                print(f"⚠️ Error checking positions: {e}")

        cycle_results["end_time"] = datetime.now(IST).isoformat()
        self._save_decisions(cycle_results)
        self._emit_status("cycle_complete", cycle_results)
        return cycle_results

    # -------------------------------
    # Learning mode - Enhanced with Learning Engine
    # -------------------------------
    def run_learning_mode(self, days: int = 30) -> Dict[str, Any]:
        """
        Run comprehensive learning analysis on recent trades.

        Analyzes what worked, what didn't, and adjusts parameters.
        This makes the system better every day!

        Args:
            days: Number of days of history to analyze

        Returns:
            Learning analysis with recommendations and adjustments
        """
        self._emit_status("learning_start", {"days": days})

        print(f"\n🧠 Starting Learning Mode (analyzing last {days} days)...")
        print("=" * 80)

        result = {
            "status": "complete",
            "timestamp": datetime.now(IST).isoformat(),
        }

        # 1. Get current learning state
        if self.learning_engine:
            try:
                learning_summary = self.learning_engine.get_learning_summary()
                result["current_state"] = learning_summary

                print(f"\n📚 Current Learning State:")
                print(f"   Last Analysis: {learning_summary.get('last_analysis', 'Never')}")
                print(f"   Trades Analyzed: {learning_summary.get('total_trades_analyzed', 0)}")
                print(f"   Confidence Threshold: {learning_summary.get('confidence_threshold', 0.60):.2f}")
                print(f"   Patterns Learned: {learning_summary.get('patterns_learned', {}).get('winning', 0)} winning, {learning_summary.get('patterns_learned', {}).get('losing', 0)} losing")

            except Exception as e:
                print(f"⚠️ Error getting learning state: {e}")

        # 2. Analyze trade history
        if self.learning_engine:
            try:
                print(f"\n🔍 Analyzing trade history...")
                analysis = self.learning_engine.analyze_trade_history(days=days)

                result["analysis"] = analysis

                if analysis.get("status") == "insufficient_data":
                    print(f"\n⚠️ {analysis.get('message')}")
                    return result

                # Display key metrics
                print(f"\n📊 Performance Metrics:")
                print(f"   Total Trades: {analysis.get('total_trades', 0)}")
                print(f"   Winners: {analysis.get('winners', 0)} | Losers: {analysis.get('losers', 0)} | Breakeven: {analysis.get('breakeven', 0)}")
                print(f"   Win Rate: {analysis.get('win_rate', 0):.1f}%")
                print(f"   Total P&L: ₹{analysis.get('total_pnl', 0):,.2f}")
                print(f"   Avg Win: ₹{analysis.get('avg_win', 0):,.2f}")
                print(f"   Avg Loss: ₹{analysis.get('avg_loss', 0):,.2f}")
                print(f"   Profit Factor: {analysis.get('profit_factor', 0):.2f}")

                # Display symbol analysis
                symbol_analysis = analysis.get("symbol_analysis", {})
                if symbol_analysis.get("best_symbols"):
                    print(f"\n🏆 Best Performing Symbols:")
                    for symbol, stats in list(symbol_analysis["best_symbols"].items())[:3]:
                        print(f"   {symbol}: ₹{stats['total_pnl']:,.2f} ({stats['win_rate']:.1f}% win rate, {stats['trades']} trades)")

                if symbol_analysis.get("worst_symbols"):
                    print(f"\n📉 Worst Performing Symbols:")
                    for symbol, stats in list(symbol_analysis["worst_symbols"].items())[:3]:
                        print(f"   {symbol}: ₹{stats['total_pnl']:,.2f} ({stats['win_rate']:.1f}% win rate, {stats['trades']} trades)")

                # Display timing analysis
                timing = analysis.get("timing_analysis", {})
                if timing:
                    print(f"\n⏱️ Timing Analysis:")
                    print(f"   Stop-Loss Hits: {timing.get('stop_loss_hits', 0)}")
                    print(f"   Target Hits: {timing.get('target_hits', 0)}")
                    print(f"   Avg Hold Time: {timing.get('avg_holding_time_minutes', 0):.1f} min")
                    print(f"   {timing.get('insight', '')}")

                # Display patterns
                winning_patterns = analysis.get("winning_patterns", [])
                losing_patterns = analysis.get("losing_patterns", [])

                if winning_patterns:
                    print(f"\n✅ Winning Patterns:")
                    for pattern in winning_patterns:
                        print(f"   • {pattern}")

                if losing_patterns:
                    print(f"\n❌ Losing Patterns:")
                    for pattern in losing_patterns:
                        print(f"   • {pattern}")

                # Display recommendations
                recommendations = analysis.get("recommendations", {}).get("recommendations", [])
                if recommendations:
                    print(f"\n💡 Recommendations ({len(recommendations)}):")
                    for rec in recommendations:
                        priority = rec.get("priority", "MEDIUM")
                        category = rec.get("category", "general")
                        recommendation = rec.get("recommendation", "")
                        reason = rec.get("reason", "")

                        print(f"\n   [{priority}] {recommendation}")
                        print(f"   Category: {category}")
                        print(f"   Reason: {reason}")
                        print(f"   Action: {rec.get('action', 'N/A')}")

                # Apply recommended adjustments
                adjustments = analysis.get("recommendations", {}).get("adjustments", {})
                if adjustments:
                    print(f"\n🔧 Applying Parameter Adjustments:")

                    # Update confidence threshold
                    if "confidence_threshold" in adjustments:
                        old_threshold = self.memory.get("confidence_gate", 0.60)
                        new_threshold = adjustments["confidence_threshold"]
                        self.memory["confidence_gate"] = new_threshold
                        print(f"   Confidence threshold: {old_threshold:.2f} → {new_threshold:.2f}")

                    # Update blacklist
                    if "blacklist_add" in adjustments:
                        for symbol in adjustments["blacklist_add"]:
                            if symbol not in self.memory.get("blacklist", []):
                                self.memory.setdefault("blacklist", []).append(symbol)
                                print(f"   Added to blacklist: {symbol}")

                    self._save_memory()
                    result["adjustments_applied"] = adjustments

            except Exception as e:
                print(f"❌ Error during trade analysis: {e}")
                result["error"] = str(e)

        # 3. Update money manager if needed
        if self.money_manager and analysis and not analysis.get("error"):
            try:
                # Check if circuit breaker should be adjusted based on performance
                win_rate = analysis.get("win_rate", 50)
                if win_rate < 40:
                    print(f"\n⚠️ Low win rate detected - consider reviewing strategy")
                elif win_rate > 70:
                    print(f"\n🎉 Excellent win rate - strategy is working well!")

            except Exception as e:
                print(f"⚠️ Error updating money manager: {e}")

        print(f"\n{'=' * 80}")
        print(f"✅ Learning mode complete!")
        print(f"{'=' * 80}\n")

        self._emit_status("learning_complete", result)
        return result


# -------------------------------
# Convenience wrapper for main.py
# -------------------------------
def run_decision_cycle(symbols: Sequence[str], **kwargs) -> Dict[str, Any]:
    crew = TradingCrew(**kwargs)
    return crew.run_decision_cycle(symbols)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Trading Crew Orchestrator")
    ap.add_argument("--live", type=int, default=0)
    ap.add_argument("--wait-open", action="store_true")
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument("--symbols", nargs="*", default=["ITC", "TCS", "RELIANCE"])
    args = ap.parse_args()

    crew = TradingCrew(
        live=bool(args.live),
        wait_for_open=args.wait_open,
        min_confidence_gate=args.min_confidence,
    )
    summary = crew.run_decision_cycle(args.symbols)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
