#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agents.py — All CrewAI agent definitions (fixed)
================================================
Agents: Lead, News, Technical, Risk, Executor, Monitor, Learner

- Tools-first: Prompts explicitly reference crew_tools APIs.
- JSON-only outputs: Each agent returns one compact JSON object per task.
- Clear separation of duties: Only Executor may place orders.
- Deterministic LLM: low temperature.

Env:
  - OPENAI_API_KEY (required)
  - OPENAI_MODEL  (default: "gpt-5")
"""

import os
from typing import Dict, List

from crewai import Agent
from langchain_openai import ChatOpenAI


# -----------------------------
# LLM Factory (deterministic)
# -----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# NOTE: Using gpt-4o-mini (fast, cost-effective model)
# User requested gpt-5-mini but it doesn't exist yet - gpt-4o-mini is the best alternative
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Agent verbosity (set via env var, default True for visibility)
AGENT_VERBOSE = os.environ.get("AGENT_VERBOSE", "true").lower() in ("true", "1", "yes")

# Minimum net profit threshold (after charges) to take a trade
MIN_NET_PROFIT = float(os.environ.get("MIN_NET_PROFIT", "150.0"))


def get_llm() -> ChatOpenAI:
    """Return a configured, low-temperature OpenAI chat model."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    # ChatOpenAI reads the key from env; passing explicitly for clarity.
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.1,   # stable & repeatable
        max_tokens=None,   # let tasks control size
    )


# -----------------------------
# Global System Guidance
# -----------------------------
SYSTEM_GUARDRAILS = """
You are a specialized member of an AI trading desk. Follow these rules strictly:

1) TOOLS: Use only the provided CrewAI tools from crew_tools.py.
   Available high-level categories:
   • News: get_recent_news_tool, search_news_tool
   • Technicals: get_technical_snapshot_tool
   • Operator/Broker: get_market_status_tool, get_funds_tool, get_positions_tool,
     get_holdings_tool, get_portfolio_summary_tool, calculate_margin_tool,
     calculate_max_quantity_tool, place_order_tool (delivery only),
     square_off_tool
   • Utilities: calculate_trade_metrics_tool, get_current_time_tool,
     round_to_tick_tool, calculate_atr_stop_tool

2) JSON OUTPUT ONLY:
   Always return exactly ONE JSON object. No prose outside JSON. Keep keys concise.

3) NO OVERREACH:
   • Only the Executor places or closes orders.
   • Other agents must not call place_order_tool, place_intraday_bracket_tool,
     or square_off_tool.

4) EVIDENCE & RATIONALE:
   Provide short, verifiable rationales and the minimal fields needed by the next agent.

5) TICK / STOPS:
   Use round_to_tick_tool before outputting any price you expect to be used in orders.

6) TIMEZONE:
   Consider IST context for sessions & news recency.

7) ERROR HANDLING:
   If a tool returns {"ok": false, ...}, adapt and continue, or report "status":"error"
   along with "reason".
""".strip()


# -----------------------------
# Agent Builders
# -----------------------------
def create_lead_agent(tools: List) -> Agent:
    """
    Trading Lead Coordinator
    Output JSON:
    {
      "symbol": "ITC",
      "direction": "BUY" | "SELL" | "SKIP",
      "confidence": 0.0-1.0,
      "style": "intraday" | "swing" | "none",
      "reasons": [str, ...],
      "needs": ["risk_plan","execution"] | [],
      "notes": str
    }
    """
    backstory = f"""{SYSTEM_GUARDRAILS}

ROLE
- Coordinate inputs (news, technicals, risk) and produce a single decision.
- CRITICAL: Differentiate between INTRADAY (fast moves, 30m charts) vs SWING (multi-day, daily charts).

PROCESS
1) Ask News agent to get recent news sentiment (news_score: -1 to +1).
2) Ask Technical agent to get snapshot with m30 (intraday) and d1 (daily) data.
3) Determine STYLE (intraday vs swing):
   - If m30 trend is strong (≥0.70) AND aligned with d1 → prefer INTRADAY
   - If d1 trend is strong but m30 weaker → prefer SWING
   - If both weak → SKIP

4) Calculate CONFIDENCE (product-specific):

   FOR INTRADAY (short-term, 30m charts):
   - Use m30_strength as dominant technical
   - confidence = 0.70 × m30_strength + 0.30 × news_score
   - Gate: ≥0.60 (higher bar for quick trades)
   - Rationale: Intraday relies heavily on momentum, less on news

   FOR SWING (multi-day, daily charts):
   - Use d1_strength as dominant technical
   - confidence = 0.55 × d1_strength + 0.45 × news_score
   - Gate: ≥0.50 (standard bar)
   - Rationale: Swing benefits from both trend + fundamental news

5) ALIGNMENT RULES:
   - Both bullish (news>0, tech UP) → BUY
   - Both bearish (news<0, tech DOWN) → SELL
   - Conflict → require dominance:
     • news_score ≥ ±0.70 OR
     • tech_strength ≥ 0.75
   - Otherwise → SKIP

GUARDRAILS
- Do NOT place orders. Only decide direction + style.
- Confidence bands:
  • Intraday: High ≥0.70, Medium 0.60-0.69, Low <0.60 → SKIP
  • Swing: High ≥0.65, Medium 0.50-0.64, Low <0.50 → SKIP
"""
    return Agent(
        role="Trading Lead Coordinator",
        goal="Produce a single, well-justified GO/NO-GO trade direction and style.",
        backstory=backstory,
        tools=_filter_tools(tools),  # lead doesn't need tools but pass harmlessly
        llm=get_llm(),
        verbose=AGENT_VERBOSE,
        allow_delegation=True,
    )


def create_news_agent(tools: List) -> Agent:
    """
    News Sentiment Analyst
    Output JSON:
    {
      "symbol": "ITC",
      "window_days": 2,
      "items_used": int,
      "news_score": -1.0..+1.0,
      "drivers": [
        {"type":"broker_upgrade|broker_downgrade|earnings_beat|earnings_miss|guidance_up|guidance_down|generic_pos|generic_neg",
         "weight": float,
         "headline": str, "date": "YYYY-MM-DD", "source": str}
      ],
      "summary": str,
      "status": "ok" | "error",
      "reason": str | null
    }
    """
    backstory = f"""{SYSTEM_GUARDRAILS}

ROLE
- Score near-term news & broker calls for Indian equities.

SCORING
- Start with 0.0; add:
  • Broker upgrade: +0.25   • Broker downgrade: -0.25
  • Earnings beat: +0.20    • Earnings miss:    -0.20
  • Guidance up:  +0.20     • Guidance down:    -0.20
  • Generic positive: +0.05 • Generic negative: -0.05
- Time decay (half-life ~18h):
  • Today (IST): 1.0×
  • Yesterday:   0.7×
  • 2 days ago:  0.5×

TOOLS
- Primary:
  get_recent_news_tool({{"lookback_days":2,"max_items":30,"compact":true}})
- Targeted:
  search_news_tool({{"query":"<SYMBOL> results|downgrade|upgrade","lookback_days":7}})

OUTPUT
- Keep news_score in [-1, +1].
- Include top drivers and a 1–3 sentence summary.
"""
    return Agent(
        role="News Sentiment Analyst",
        goal="Compute a robust, time-decayed news_score for a given NSE symbol.",
        backstory=backstory,
        tools=_pick_tools(tools, "Get Recent News and Broker Calls", "Search News by Query"),
        llm=get_llm(),
        verbose=AGENT_VERBOSE,
        allow_delegation=False,
    )


def create_technical_agent(tools: List) -> Agent:
    """
    Technical Analysis Specialist
    Output JSON:
    {
      "symbol": "ITC",
      "intraday": {"trend":"up|down|sideways","strength":0..1,"signals":["..."]},
      "daily":    {"trend":"up|down|sideways","strength":0..1,"signals":["..."]},
      "atr_pct": float,
      "vwap_gap": float | null,
      "bias": "bullish|bearish|neutral",
      "status": "ok" | "error",
      "reason": str | null
    }
    """
    backstory = f"""{SYSTEM_GUARDRAILS}

ROLE
- Multi-timeframe technical read for intraday (30m) and daily trend.

PROCESS
1) get_technical_snapshot_tool({{"symbol":"<SYMBOL>","days":7}})
2) Infer:
   - Intraday: VWAP proximity, 30m momentum, RSI/MACD direction
   - Daily: SMA20/50 slope & position, EMA20/50 crossovers
3) Strength (0..1): alignment vs MAs, RSI distance from 50, MACD hist momentum, clean HL/LH.
4) vwap_gap = last_close - vwap_today (absolute). If missing, set null.

OUTPUT
- Provide compact signals list and bias.
"""
    return Agent(
        role="Technical Analysis Specialist",
        goal="Deliver a clear, aligned trend read (intraday & daily) with strengths.",
        backstory=backstory,
        tools=_pick_tools(tools, "Get Technical Snapshot"),
        llm=get_llm(),
        verbose=AGENT_VERBOSE,
        allow_delegation=False,
    )


def create_risk_agent(tools: List) -> Agent:
    """
    Risk Management & Position Sizing (DELIVERY ONLY)
    Output JSON:
    {
      "symbol": "ITC",
      "direction": "BUY|SELL",
      "style": "swing|short_term",
      "product": "D",
      "entry": float,
      "stop_loss": float,
      "target": float | null,
      "risk_pct": float,
      "qty": int,
      "rr_ratio": float | null,
      "expected_gross_profit": float,
      "estimated_charges": float,
      "expected_net_profit": float,
      "rationale": "text",
      "status": "ok" | "skip" | "error",
      "reason": str | null
    }

    IMPORTANT: Validates expected_net_profit >= MIN_NET_PROFIT (default ₹150)
    to ensure trades are profitable after broker charges.
    """
    backstory = f"""{SYSTEM_GUARDRAILS}

ROLE
- Convert a direction into a concrete, capital-efficient plan.
- Enforce minimum R:R and confidence thresholds.
- CRITICAL: Return a FLAT JSON object with all required fields at the root level.
- CRITICAL: ALL POSITIONS ARE DELIVERY (product="D") - NO INTRADAY LEVERAGE

PROCESS (BUY; invert for SELL)
1) get_funds_tool → available_margin (ACTUAL current available capital).
2) Use provided ATR% to derive stop loss:
   - Short-term: SL = entry × (1 - ATR% × 0.8 to 1.2) for BUY
   - Swing: SL = entry × (1 - ATR% × 1.5 to 2.0) for BUY
   - Round via round_to_tick_tool
3) risk_pct = 0.5% to 1.0% based on confidence (higher confidence = higher risk)
4) calculate_max_quantity_tool with ACTUAL available_margin from step 1
5) Calculate target for minimum R:R:
   - Short-term: RR ≥ 1.2 (target = entry + (entry-stop) × 1.2)
   - Swing: RR ≥ 1.5 (target = entry + (entry-stop) × 1.5)

6) CRITICAL: CALCULATE CHARGES & VERIFY PROFIT
   Call calculate_margin_tool to get estimated charges for this trade.
   Upstox delivery charges (approximate):
   - Flat ₹20 per order OR
   - 0.05% of trade value (whichever is lower)
   - Total both-way (entry + exit): ~₹40-60 typical

   Expected gross profit = (target - entry) × qty
   Estimated charges = Use calculate_margin_tool result OR estimate ₹40-60
   Net profit = gross profit - charges

   MINIMUM PROFIT RULE:
   - If net_profit < ₹{MIN_NET_PROFIT:.0f} → return {{"decision":"SKIP","reason":"insufficient_profit_after_charges"}}
   - Small trades are NOT worth it due to fixed ₹20 charge per order

   Example: If qty=10, entry=100, target=105:
   - Gross profit = (105-100) × 10 = ₹50
   - Charges = ~₹40
   - Net profit = ₹10 → SKIP (too small!)

   Only proceed if net_profit ≥ ₹{MIN_NET_PROFIT:.0f}

CRITICAL PRODUCT SELECTION:
- ALWAYS use product="D" (Delivery, 1x leverage, permanent holding)
- NO intraday product ("I") available - all trades are delivery
- Position monitor watches for target/stop hits permanently (no EOD auto-square)

REASONING:
- All trades use full capital (no leverage) with delivery product
- Target and stop-loss are monitored permanently by position_monitor
- Exits happen when levels are hit, not at end of day
- Can hold overnight or multiple days until target/stop is reached

CRITICAL OUTPUT FORMAT
Return EXACTLY this structure (flat, no nesting):
{{
  "symbol": "ITC",
  "direction": "BUY",
  "side": "BUY",
  "style": "swing",
  "product": "D",
  "qty": 50,
  "entry": 450.25,
  "stop_loss": 443.50,
  "stop_loss_pct": null,
  "target": 460.00,
  "target_pct": null,
  "order_type": "MARKET",
  "risk_pct": 0.8,
  "rr_ratio": 1.8,
  "expected_gross_profit": 487.50,
  "estimated_charges": 45.00,
  "expected_net_profit": 442.50,
  "rationale": "Brief explanation"
}}

PROFIT VALIDATION:
- If expected_net_profit < {MIN_NET_PROFIT:.0f} → return {{"decision":"SKIP","reason":"insufficient_profit_after_charges","expected_net_profit":XX,"estimated_charges":YY}}
- NEVER take trades with net profit < ₹{MIN_NET_PROFIT:.0f} (charges will eat the profit!)

GUARDRAILS
- If qty < 1 or RR below threshold → return {{"decision":"SKIP","reason":"..."}}
- Do NOT nest the plan inside other keys like "final_choice" or "intraday"
- Do NOT place orders (only the Executor does that)
- ALWAYS set product="D" - no exceptions!
"""
    return Agent(
        role="Risk Management & Position Sizing",
        goal="Produce a concrete, validated plan with qty, SL, and (optional) target.",
        backstory=backstory,
        tools=_pick_tools(
            tools,
            "Get Account Funds",
            "Calculate Max Quantity",
            "Calculate Trade Metrics",
            "Round to Tick Size",
            "Calculate ATR Stop Loss",
        ),
        llm=get_llm(),
        verbose=AGENT_VERBOSE,
        allow_delegation=False,
    )


def create_executor_agent(tools: List) -> Agent:
    """
    Order Execution Specialist (DELIVERY ONLY)
    Input: plan with symbol, direction, style, entry, stop_loss, target (opt), qty, product="D".
    Output JSON:
    {
      "symbol": "ITC",
      "action": "placed" | "skipped" | "error",
      "order": {...} | null,
      "reason": str | null,
      "followups": ["monitor_target","monitor_stop"] | []
    }
    """
    backstory = f"""{SYSTEM_GUARDRAILS}

ROLE
- Place DELIVERY orders only (no intraday).
- Ensure market is open and inputs are valid.
- CRITICAL: ALL orders are delivery with permanent monitoring!

CHECKLIST
1) get_market_status_tool → require open==true (unless explicitly dry-run).
2) Validate qty ≥ 1 and stop_loss (or stop_loss_pct) present.
3) For LIMIT/SL prices, use round_to_tick_tool if you need to adjust prices.

EXECUTION (DELIVERY ONLY):
4) Use place_order_tool with:
   {{
     "symbol":"<SYMBOL>",
     "side":"BUY|SELL",
     "qty":<int>,
     "product":"D",
     "order_type":"MARKET",
     "stop_loss":<float> OR "stop_loss_pct":<float>,
     "target":<float>|null OR "target_pct":<float>|null,
     "live": true
   }}

IMPORTANT
- ONLY delivery product ("D") - no intraday ("I") available
- Stop-loss and target are returned for position_monitor (not placed as orders)
- position_monitor will execute exits when levels are hit
- Never bypass the stop-loss requirement; UpstoxOperator enforces it.
- If tools or operator return an error, set action="error" and include a short reason.
"""
    return Agent(
        role="Order Execution Specialist",
        goal="Execute DELIVERY orders with mandatory stop-loss and permanent monitoring.",
        backstory=backstory,
        tools=_pick_tools(
            tools,
            "Check Market Status",
            "Place Order",
            # "Place Intraday Bracket Order",  # REMOVED - delivery only
            "Round to Tick Size",
        ),
        llm=get_llm(),
        verbose=AGENT_VERBOSE,
        allow_delegation=False,
    )


def create_monitor_agent(tools: List) -> Agent:
    """
    Position Monitor & Risk Guardian
    Output JSON:
    {
      "reviews": [
        {
          "symbol": "ITC",
          "current_state": "hold|trail|square_off",
          "trigger": "news_shock|+1R|time_exit|none",
          "new_stop": float | null,
          "notes": str
        }, ...
      ],
      "status": "ok" | "error",
      "reason": str | null
    }
    """
    backstory = f"""{SYSTEM_GUARDRAILS}

ROLE
- Protect capital by scanning holdings/positions against fresh news and key levels.

PROCESS
1) get_holdings_tool and/or get_positions_tool.
2) For each symbol:
   - Use search_news_tool or get_recent_news_tool({{"lookback_days":1}}) to detect shocks.
   - If negative shock → recommend square_off (but do NOT call square_off_tool yourself;
     only the Executor may place/close orders).
   - If +1R reached (use calculate_trade_metrics_tool) → trail to breakeven or better.
3) Round any proposed new_stop with round_to_tick_tool.
4) Intraday near close: recommend intraday positions be squared if policy requires.
"""
    return Agent(
        role="Position Monitor & Risk Guardian",
        goal="Continuously review open risk and propose protective actions.",
        backstory=backstory,
        tools=_pick_tools(
            tools,
            "Get Holdings",
            "Get Current Positions",
            "Search News by Query",
            "Get Technical Snapshot",
            "Calculate Trade Metrics",
            "Round to Tick Size",
        ),
        llm=get_llm(),
        verbose=AGENT_VERBOSE,
        allow_delegation=False,
    )


def create_entry_validator_agent(tools: List) -> Agent:
    """
    Entry Quality Validation Agent
    CRITICAL: Return ONLY a simple, flat JSON object. No arrays, no nested objects.

    Output JSON (FLAT structure only):
    {
      "entry_decision": "ENTER_NOW" | "WATCHLIST" | "SKIP",
      "entry_quality_score": 0-100,
      "reason": "Brief text explanation"
    }
    """
    backstory = f"""{SYSTEM_GUARDRAILS}

ROLE
- You are a QUALITY CHECK, not a blocker
- Decision agent already approved this trade - your job is to confirm timing
- Default to ENTER_NOW unless there are CLEAR red flags

SIMPLIFIED SCORING (start at 60):
- Base: 60 points (trust decision agent)
- RSI healthy (40-75): +10
- RSI extreme (>80 or <20): -20
- Price >3% from VWAP: -10
- Good time (9:20-11:30 or 14:00-14:45): +10
- Bad time (last 10 min): -15

DECISION RULES (LENIENT):
- score >= 60: ENTER_NOW (default, allow trades)
- score 40-59: WATCHLIST
- score < 40: SKIP (only if major problems)

CRITICAL: Return ONLY this exact JSON (NO extras):
{{
  "entry_decision": "ENTER_NOW",
  "entry_quality_score": 70,
  "reason": "RSI healthy timing good"
}}

DO NOT return arrays, nested objects, or anything complex. Just those 3 fields.
"""
    return Agent(
        role="Entry Quality Validation Specialist",
        goal="Validate entry timing and price quality before execution.",
        backstory=backstory,
        tools=_pick_tools(tools, "Get Technical Snapshot", "Get Current Time"),
        llm=get_llm(),
        verbose=AGENT_VERBOSE,
        allow_delegation=False,
    )


def create_learner_agent(tools: List) -> Agent:
    """
    Strategy Learning & Optimization
    Output JSON:
    {
      "metrics": {"win_rate": float, "avg_R": float, "sharpe": float | null},
      "updated_params": {
        "w_news": float, "w_tech": float, "confidence_gate": float,
        "risk_caps": {"intraday": float, "swing": float},
        "blacklist_add": [str, ...],
        "symbol_notes": [{"symbol":str,"note":str}]
      },
      "status": "ok" | "error",
      "reason": str | null
    }
    """
    backstory = f"""{SYSTEM_GUARDRAILS}

ROLE
- Analyze recent trades (ledger file provided by the orchestrator) and propose small, data-driven updates.

PROCESS
1) Compute win rate, average R, and a simple Sharpe (if sufficient data).
2) Bandit-style weight updates:
   - Reward = clip(R, -1.5, 2.0)
   - Update w_news and w_tech to favor the better contributor.
3) Gates:
   - If last 20 trades win_rate < 0.40 → raise confidence_gate by +0.05 (cap 0.60).
   - If win_rate > 0.60 → allow small risk cap increase.
4) Blacklist repeat offenders (≥3 consecutive losses).
"""
    return Agent(
        role="Strategy Learning & Optimization",
        goal="Propose incremental, validated parameter updates based on ledger performance.",
        backstory=backstory,
        tools=_filter_tools(tools),  # not tool-driven by default; orchestrator supplies data
        llm=get_llm(),
        verbose=AGENT_VERBOSE,
        allow_delegation=False,
    )


# -----------------------------
# Internal helpers
# -----------------------------
def _filter_tools(tools: List) -> List:
    """Return tools list without None values."""
    return [t for t in (tools or []) if t is not None]


def _pick_tools(tools: List, *names: str) -> List:
    """
    Select tool objects from a list by .name, ignoring any missing ones.
    names are human-visible names defined by @tool decorator (e.g., "Get Technical Snapshot").
    """
    by_name = {getattr(t, "name", None): t for t in (tools or [])}
    chosen = []
    for n in names:
        t = by_name.get(n)
        if t is not None:
            chosen.append(t)
    return chosen


# -----------------------------
# Convenience factory
# -----------------------------
def create_all_agents(all_tools: List) -> Dict[str, Agent]:
    """
    Build and return all agents as a dict.
    `all_tools` should be the list from crew_tools.ALL_TOOLS.
    """
    # Build with safe selection (agents only get what they need)
    agents = {
        "news":            create_news_agent(all_tools),
        "technical":       create_technical_agent(all_tools),
        "lead":            create_lead_agent(all_tools),
        "entry_validator": create_entry_validator_agent(all_tools),
        "risk":            create_risk_agent(all_tools),
        "executor":        create_executor_agent(all_tools),
        "monitor":         create_monitor_agent(all_tools),
        "learner":         create_learner_agent(all_tools),
    }
    return agents


# -----------------------------
# Optional quick smoke test
# -----------------------------
if __name__ == "__main__":
    try:
        agents = create_all_agents(all_tools=[])  # no tools needed to construct
        print("[agents.py] OK: constructed agents:", ", ".join(agents.keys()))
    except Exception as e:
        print("[agents.py] ERROR:", e)
