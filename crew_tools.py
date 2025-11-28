#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crew_tools.py — All CrewAI tools (imports at top, docstrings added)
===================================================================
Directly imports your core modules:
- news_client.NewsClient
- upstox_technical.UpstoxTechnicalClient
- upstox_operator.UpstoxOperator

All tools return JSON strings. Self-test:  python crew_tools.py --self-test

EXTRA:
- Logs to ./data/crew_tools.log (or CREW_LOG_DIR)
- Logs all news/technical/order calls + errors for easier debugging
"""
from __future__ import annotations

import os
import json
import math
import time
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

from crewai.tools import tool

# ---- hard imports (as requested) ----
from news_client import NewsClient
from upstox_technical import UpstoxTechnicalClient
from upstox_operator import UpstoxOperator

IST = ZoneInfo(os.environ.get("TZ", "Asia/Kolkata"))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = Path(os.environ.get("CREW_LOG_DIR", "./data"))
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("crew_tools")
if not logger.handlers:
    log_path = LOG_DIR / "crew_tools.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)  # detailed log to file

    # Console handler: default WARNING to avoid spam, can override via env
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    console_level = os.environ.get("CREW_TOOLS_CONSOLE_LEVEL", "WARNING").upper()
    sh.setLevel(console_level)

    logger.addHandler(fh)
    logger.addHandler(sh)

logger.setLevel(os.environ.get("CREW_TOOLS_LOG_LEVEL", "INFO").upper())
logger.info("crew_tools initialized, log file: %s", LOG_DIR / "crew_tools.log")

# ---- singletons ----
NEWS = NewsClient()
TECH = UpstoxTechnicalClient()
OP = UpstoxOperator()

# ---- tiny in-memory caches & debounce state ----
_SNAP_CACHE: Dict[Tuple[str, int], Tuple[float, dict]] = {}             # (symbol, days) -> (ts, snapshot)
_NEWS_CACHE: Dict[Tuple[int, int, bool, str], Tuple[float, dict]] = {}  # (lookback_days, max_items, compact, mode)
_LAST_CALL_TS: Dict[str, int] = {}                                      # per-key debounce ms timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json_fail(reason: str, **extra) -> str:
    """Return a failure JSON string with error and optional extras."""
    payload = {"ok": False, "error": reason}
    if extra:
        payload.update(extra)
    logger.debug("json_fail: %s", payload)
    return json.dumps(payload, ensure_ascii=False)


def _json_ok(**data) -> str:
    """Return a success JSON string with arbitrary payload fields."""
    payload = {"ok": True}
    payload.update(data)
    logger.debug("json_ok: %s", {k: payload[k] for k in list(payload)[:6]})
    return json.dumps(payload, ensure_ascii=False)


def _parse(s: Optional[str]) -> Dict[str, Any]:
    """Parse JSON string; fallback to 'k=v,k2=v2' simple CSV pairs."""
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        if "=" in s and ":" not in s:
            d = {}
            for kv in s.split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    d[k.strip()] = v.strip()
            return d
        return {}


def _coerce_args(payload: Any = None, **kwargs) -> Dict[str, Any]:
    """
    Accepts dict | JSON string | bytes | "k=v ..." text | None plus **kwargs.
    Returns a dict; kwargs override payload.
    """
    args: Dict[str, Any] = {}
    if isinstance(payload, dict):
        args.update(payload)
    elif isinstance(payload, (bytes, bytearray)):
        try:
            s = payload.decode("utf-8", "ignore")
        except Exception:
            s = ""
        args.update(_parse(s))
    elif isinstance(payload, str) and payload.strip():
        s = payload.strip()
        # JSON?
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                args.update(json.loads(s))
            except Exception:
                pass
        else:
            # allow "a=1 b=2" or "a=1,b=2"
            try:
                items = [tok for tok in s.replace(",", " ").split() if "=" in tok]
                kvs = dict(kv.split("=", 1) for kv in items)
                if kvs:
                    args.update({k.strip(): v.strip() for k, v in kvs.items()})
            except Exception:
                pass
    # overlay kwargs
    args.update({k: v for k, v in kwargs.items() if v is not None})
    return args


def _extract_available_margin(funds: dict) -> float:
    """Heuristically extract available margin/cash from a funds dict."""
    if not isinstance(funds, dict):
        return 0.0
    eq = funds.get("equity")
    if isinstance(eq, dict):
        for k in ("available_margin", "cash", "available"):
            if k in eq:
                try:
                    return float(eq[k])
                except Exception:
                    pass
    for k in ("available_margin", "cash", "available"):
        if k in funds:
            try:
                return float(funds[k])
            except Exception:
                pass
    raw = funds.get("raw")
    if isinstance(raw, dict):
        eqr = raw.get("equity")
        if isinstance(eqr, dict):
            for k in ("available_margin", "cash", "available"):
                if k in eqr:
                    try:
                        return float(eqr[k])
                    except Exception:
                        pass
    return 0.0


def _retry(plan_ms, fn):
    """
    plan_ms: iterable of sleep milliseconds (0 for immediate)
    fn:      callable with no args
    returns (ok, value_or_errstr)
    """
    last_err = None
    for ms in plan_ms:
        try:
            return True, fn()
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            logger.warning("retry step failed (%sms): %s", ms, last_err)
            if ms:
                time.sleep(ms / 1000.0)
    return False, last_err


def _debounce(key: str, min_ms: int = 150) -> None:
    """Sleep a tiny bit if the same key was called very recently (to avoid hammering)."""
    now = int(time.time() * 1000)
    last = _LAST_CALL_TS.get(key, 0)
    if now - last < min_ms:
        time.sleep((min_ms - (now - last)) / 1000.0)
    _LAST_CALL_TS[key] = int(time.time() * 1000)


def _round_to_tick(px: float, tick: float) -> float:
    if tick <= 0:
        return float(f"{px:.2f}")
    steps = round(px / tick)
    return float(f"{steps * tick:.4f}")


# ---------------------------------------------------------------------------
# NEWS
# ---------------------------------------------------------------------------
@tool("Get Recent News and Broker Calls")
def get_recent_news_tool(input_str: Any = None, **kw) -> str:
    """
    Quickly fetch *fresh, India-focused* market news & broker calls.

    Input (dict | JSON | kwargs):
      {
        "lookback_days": 2,
        "max_items": 20,
        "mode": "live",
        "today": "YYYY-MM-DD" | null,
        "compact": true
      }
    """
    try:
        args = _coerce_args(input_str, **kw)
        lookback_days = int(args.get("lookback_days", 2) or 2)
        max_items     = int(args.get("max_items", 20) or 20)
        mode          = str(args.get("mode", "live"))
        today         = args.get("today")
        compact       = bool(args.get("compact", True))

        logger.info(
            "get_recent_news_tool called: lookback_days=%s max_items=%s mode=%s today=%s compact=%s",
            lookback_days, max_items, mode, today, compact
        )

        # Debounce & short cache
        _debounce(f"news:{lookback_days}:{max_items}:{compact}:{mode}", 150)
        now = time.time()
        ck = (lookback_days, max_items, compact, mode)
        cached = _NEWS_CACHE.get(ck)
        if cached and (now - cached[0] <= 20):
            items = cached[1]["items"]
            logger.info("get_recent_news_tool returning %d cached items", len(items))
            return _json_ok(items=items, count=len(items), cached=True)

        # Retry ladder: shrink ask on retries
        ladder = [(max_items, 0), (max_items // 2 or 10, 250), (10, 500)]
        last_err = None
        for mi, sleep_ms in ladder:
            def _call():
                return NEWS.get_recent_news_and_calls(
                    today=today, lookback_days=lookback_days, max_items=mi, mode=mode, compact=compact
                )
            ok, items_or_err = _retry([0], _call)
            if ok:
                items = items_or_err or []
                payload = {"items": items}
                _NEWS_CACHE[ck] = (now, payload)
                logger.info("get_recent_news_tool fetched %d items (mi=%s)", len(items), mi)
                return _json_ok(items=items, count=len(items))
            last_err = items_or_err
            logger.warning("get_recent_news_tool attempt failed (mi=%s): %s", mi, last_err)
            if sleep_ms:
                time.sleep(sleep_ms / 1000.0)

        logger.error("get_recent_news_tool failed after retries: %s", last_err)
        return _json_fail("news_fetch_failed", detail=str(last_err))
    except Exception as e:
        logger.exception("get_recent_news_tool exception")
        return _json_fail("news_tool_error", detail=str(e))


@tool("Search News by Query")
def search_news_tool(input_str: Any = None, **kw) -> str:
    """
    Targeted news search to investigate a specific symbol, event, or theme.

    Input (dict | JSON | kwargs):
      {
        "query": "HFCL results",   # required
        "lookback_days": 7,
        "max_results": 25,
        "mode": "live",
        "compact": true,
        "domain": "moneycontrol.com" | null,
        "offset": int | null
      }
    """
    try:
        args = _coerce_args(input_str, **kw)
        q = str(args.get("query") or args.get("q") or "").strip()
        if not q:
            logger.warning("search_news_tool missing query")
            return _json_fail("missing_query")

        lookback_days = int(args.get("lookback_days", 7) or 7)
        max_results   = int(args.get("max_results", 25) or 25)
        mode          = str(args.get("mode", "live"))
        compact       = bool(args.get("compact", True))
        domain        = args.get("domain")
        offset        = args.get("offset")

        logger.info(
            "search_news_tool called: q=%r lookback_days=%s max_results=%s mode=%s domain=%s",
            q, lookback_days, max_results, mode, domain
        )

        _debounce(f"search:{q}:{lookback_days}", 150)

        ladder = [(max_results, 0), (max_results // 2 or 10, 250), (10, 500)]
        last_err = None
        for mr, sleep_ms in ladder:
            def _call():
                return NEWS.search_news(
                    q, lookback_days=lookback_days, max_results=mr,
                    sort_by_date=True, offset=offset, sitedomain=domain,
                    compact=compact, mode=mode
                )
            ok, res_or_err = _retry([0], _call)
            if ok:
                out = res_or_err
                if isinstance(out, dict) and "items" in out:
                    count = len(out.get("items") or [])
                elif isinstance(out, list):
                    count = len(out)
                else:
                    count = None
                logger.info("search_news_tool got %s results (mr=%s)", count, mr)
                return _json_ok(items=out, count=count)
            last_err = res_or_err
            logger.warning("search_news_tool attempt failed (mr=%s): %s", mr, last_err)
            if sleep_ms:
                time.sleep(sleep_ms / 1000.0)

        logger.error("search_news_tool failed after retries: %s", last_err)
        return _json_fail("news_search_failed", detail=str(last_err))
    except Exception as e:
        logger.exception("search_news_tool exception")
        return _json_fail("news_search_error", detail=str(e))


# ---------------------------------------------------------------------------
# TECHNICALS
# ---------------------------------------------------------------------------
@tool("Get Technical Snapshot")
def get_technical_snapshot_tool(input_str: Any = None, **kw) -> str:
    """
    Pull a compact technical view for a single NSE cash symbol.

    Input (dict | JSON | kwargs):
      { "symbol": "HFCL", "days": 7 }
    """
    try:
        args = _coerce_args(input_str, **kw)
        symbol = (args.get("symbol") or args.get("query") or "").strip()
        if not symbol:
            logger.warning("get_technical_snapshot_tool: missing symbol")
            return _json_fail("missing_symbol")
        days = int(args.get("days", 7) or 7)

        logger.info("get_technical_snapshot_tool called: symbol=%s days=%s", symbol, days)

        _debounce(f"snap:{symbol}:{days}", 150)

        # short cache (~25s)
        key = (symbol.upper(), days)
        now = time.time()
        cached = _SNAP_CACHE.get(key)
        if cached and (now - cached[0] <= 25):
            logger.info("get_technical_snapshot_tool returning cached snapshot for %s", symbol)
            return _json_ok(symbol=symbol, snapshot=cached[1], cached=True)

        def _call():
            return TECH.snapshot(symbol, days=days)

        ok, snap_or_err = _retry([0, 200, 400], _call)
        if not ok:
            logger.error("get_technical_snapshot_tool failed: %s", snap_or_err)
            return _json_fail("technical_failed", detail=str(snap_or_err))

        _SNAP_CACHE[key] = (now, snap_or_err or {})
        logger.info("get_technical_snapshot_tool fetched fresh snapshot for %s", symbol)
        return _json_ok(symbol=symbol, snapshot=snap_or_err or {})
    except Exception as e:
        logger.exception("get_technical_snapshot_tool exception")
        return _json_fail("technical_error", detail=str(e))


# ---------------------------------------------------------------------------
# OPERATOR / BROKER
# ---------------------------------------------------------------------------
@tool("Check Market Status")
def get_market_status_tool(input_str: Any = None, **kw) -> str:
    """
    Returns the current NSE market session state to gate trading actions.
    Input: none (or {})
    """
    try:
        res = OP.market_session_status()
        logger.info("get_market_status_tool: %s", res)
        return _json_ok(market_session=res)
    except Exception as e:
        logger.exception("get_market_status_tool exception")
        return _json_fail("market_status_failed", detail=str(e))


@tool("Get Account Funds")
def get_funds_tool(input_str: Any = None, **kw) -> str:
    """
    Fetch available/used margin to size trades and enforce risk budgets.
    Input: none (or {})
    """
    try:
        funds = OP.get_funds()
        logger.info(
            "get_funds_tool: available_margin=%s",
            (funds.get("equity") or {}).get("available_margin", 0)
        )
        return _json_ok(funds=funds)
    except Exception as e:
        logger.exception("get_funds_tool exception")
        return _json_fail("funds_failed", detail=str(e))


@tool("Get Current Positions")
def get_positions_tool(input_str: Any = None, **kw) -> str:
    """
    Retrieve current positions.
    Input: { "include_closed": false }
    """
    args = _coerce_args(input_str, **kw)
    include_closed = bool(args.get("include_closed", False))
    try:
        pos = OP.get_positions(include_closed=include_closed) if "include_closed" in inspect.signature(OP.get_positions).parameters else OP.get_positions()
        count = len(pos.get("positions") or []) if isinstance(pos, dict) else None
        logger.info("get_positions_tool: include_closed=%s count=%s", include_closed, count)
        return _json_ok(positions=pos)
    except Exception as e:
        logger.exception("get_positions_tool exception")
        return _json_fail("positions_failed", detail=str(e))


@tool("Get Holdings")
def get_holdings_tool(input_str: Any = None, **kw) -> str:
    """
    Fetch delivery holdings for swing management and mark-to-market.
    """
    try:
        holds = OP.get_holdings()
        count = len(holds.get("holdings") or []) if isinstance(holds, dict) else None
        logger.info("get_holdings_tool: count=%s", count)
        return _json_ok(holdings=holds)
    except Exception as e:
        logger.exception("get_holdings_tool exception")
        return _json_fail("holdings_failed", detail=str(e))


@tool("Get Portfolio Summary")
def get_portfolio_summary_tool(input_str: Any = None, **kw) -> str:
    """
    One-call overview for funds, positions, and holdings.
    """
    try:
        if hasattr(OP, "get_portfolio_summary"):
            summ = OP.get_portfolio_summary()
        else:
            funds = OP.get_funds()
            positions = OP.get_positions()
            holdings = OP.get_holdings()
            summ = {"funds": funds, "positions": positions, "holdings": holdings}
        logger.info("get_portfolio_summary_tool called")
        return _json_ok(summary=summ)
    except Exception as e:
        logger.exception("get_portfolio_summary_tool exception")
        return _json_fail("portfolio_summary_failed", detail=str(e))


@tool("Calculate Required Margin")
def calculate_margin_tool(input_str: Any = None, **kw) -> str:
    """
    Estimate broker-required margin for a prospective order.
    Input: {"symbol": "...", "qty": 10, "side": "BUY", "product": "I", "price": 123.45}
    """
    p = _coerce_args(input_str, **kw)
    try:
        logger.info(
            "calculate_margin_tool called: symbol=%s qty=%s price=%s product=%s",
            p.get("symbol"), p.get("qty"), p.get("price"), p.get("product")
        )
        if hasattr(OP, "calculate_required_margin"):
            m = OP.calculate_required_margin(**p)
            return _json_ok(margin=m)
        return _json_fail("operator_missing_calculate_required_margin")
    except Exception as e:
        logger.exception("calculate_margin_tool exception")
        return _json_fail("calc_margin_failed", detail=str(e))


@tool("Calculate Max Quantity")
def calculate_max_quantity_tool(input_str: Any = None, **kw) -> str:
    """
    Compute the maximum affordable quantity given funds/leverage and optional risk.

    Input:
      {
        "symbol": "HFCL" | null,
        "price": 406.2,        # required
        "product": "I"|"D"="I",
        "risk_pct": 0.5,       # optional % (used in fallback if broker doesn't support it)
        "stop_loss": 401.3,    # needed if risk_pct > 0
        "tick_size": 0.05
      }

    NOTE:
    - If UpstoxOperator.calculate_max_quantity exposes richer semantics,
      we introspect its signature and forward only supported params.
    """
    args = _coerce_args(input_str, **kw)
    try:
        price = float(args["price"])
    except Exception:
        logger.warning("calculate_max_quantity_tool: missing/invalid price arg=%r", args.get("price"))
        return _json_fail("missing_or_invalid_price")

    symbol     = args.get("symbol")
    product    = args.get("product", "I")
    risk_pct   = args.get("risk_pct", 0.0)
    stop_loss  = args.get("stop_loss")
    tick_size  = float(args.get("tick_size", os.environ.get("TICK_SIZE", 0.05)))

    try:
        risk_pct  = float(risk_pct or 0.0)
        stop_loss = float(stop_loss) if stop_loss is not None else None
    except Exception:
        logger.warning("calculate_max_quantity_tool: invalid risk_pct/stop_loss")
        return _json_fail("invalid_risk_or_stop_loss")

    try:
        logger.info(
            "calculate_max_quantity_tool called: symbol=%s price=%s product=%s risk_pct=%s stop_loss=%s",
            symbol, price, product, risk_pct, stop_loss
        )

        funds = {}
        try:
            funds = OP.get_funds() or {}
        except Exception as e_f:
            logger.warning("calculate_max_quantity_tool: get_funds failed: %s", e_f)
            funds = {}
        available = _extract_available_margin(funds)

        if hasattr(OP, "calculate_max_quantity"):
            fn = getattr(OP, "calculate_max_quantity")
            sig = inspect.signature(fn)
            forward = {}
            if "symbol" in sig.parameters: forward["symbol"] = symbol
            if "price" in sig.parameters: forward["price"] = price
            if "product" in sig.parameters: forward["product"] = product
            if "risk_pct" in sig.parameters and risk_pct: forward["risk_pct"] = risk_pct
            if "stop_loss" in sig.parameters and stop_loss is not None: forward["stop_loss"] = stop_loss
            if "tick_size" in sig.parameters: forward["tick_size"] = tick_size
            if "available_margin" in sig.parameters: forward["available_margin"] = available
            if "funds" in sig.parameters: forward["funds"] = funds
            if "equity_available_margin" in sig.parameters: forward["equity_available_margin"] = available

            resp = fn(**forward)
            logger.info("calculate_max_quantity_tool: operator response=%s", resp)
            if isinstance(resp, dict):
                return _json_ok(**resp) if resp.get("ok", True) else _json_fail("calc_max_qty_failed", **resp)
            return _json_ok(result=resp)

        # Fallback estimator
        leverage = 3.0 if str(product).upper().startswith("I") else 1.0
        gross_budget = max(0.0, available * leverage)
        max_qty_by_budget = int(math.floor(gross_budget / price)) if price > 0 else 0

        max_qty_by_risk = None
        if risk_pct > 0.0 and stop_loss is not None:
            per_share_risk = abs(price - stop_loss)
            risk_cap = available * (risk_pct / 100.0)
            max_qty_by_risk = int(math.floor(risk_cap / per_share_risk)) if per_share_risk > 0 else 0

        qty = max(0, min(max_qty_by_budget, max_qty_by_risk)) if max_qty_by_risk is not None else max(0, max_qty_by_budget)
        logger.info(
            "calculate_max_quantity_tool fallback: qty=%s available=%s leverage=%s",
            qty, available, leverage
        )
        return _json_ok(
            qty=qty,
            symbol=symbol,
            price=price,
            product=product,
            leverage_used=leverage,
            available_funds=round(available, 2),
            method="fallback",
            notes=("Risk sizing applied" if max_qty_by_risk is not None else "Budget-only sizing"),
        )
    except Exception as e:
        logger.exception("calculate_max_quantity_tool exception")
        return _json_fail("calc_max_qty_exception", detail=str(e))


@tool("Place Order")
def place_order_tool(input_str: Any = None, **kw) -> str:
    """
    Submit a broker order with a mandatory stop-loss policy (operator-enforced).

    Input (dict | JSON | kwargs):
      {
        "symbol":"HFCL",
        "side":"BUY",
        "qty":50,
        "product":"I",
        "order_type":"MARKET",
        "stop_loss_pct":1.0,     # or stop_loss absolute
        "target_pct":2.0,        # or target absolute (optional)
        "live":true,             # REQUIRED for real order
        "auto_size":false
      }

    Underlying behavior (from UpstoxOperator.place_order):
    - Places a single entry order (MARKET or LIMIT).
    - Then places *two separate exit orders*:
        • target (LIMIT)
        • stop_loss (SL-M)
      using the same product (I/D) as entry.
    - Returns a dict like:
        {
          "entry": {...},
          "exits": {
            "target": {...} | null,
            "stop_loss": {...}
          },
          "exit_status": "ok" | "partial" | "failed",
          "computed_levels": {"stop_loss": ..., "target": ...},
          "warnings": [...optional...]
        }

    IMPORTANT:
    - If UpstoxOperator returns {"error": ...}, we now return ok=false so
      agents/UI can see that the order actually failed.
    """
    p = _coerce_args(input_str, **kw)
    if "order_type" not in p and "entry_type" in p:
        p["order_type"] = p.pop("entry_type")

    # Light-weight log that won't leak full secrets
    logger.info(
        "place_order_tool called: symbol=%s side=%s qty=%s product=%s order_type=%s live=%s "
        "stop_loss=%s stop_loss_pct=%s target=%s target_pct=%s",
        p.get("symbol"), p.get("side"), p.get("qty"), p.get("product"),
        p.get("order_type"), p.get("live"),
        p.get("stop_loss"), p.get("stop_loss_pct"),
        p.get("target"), p.get("target_pct")
    )

    try:
        res = OP.place_order(**p)

        # If operator returned an error dict, surface as failure
        if isinstance(res, dict) and res.get("error"):
            logger.error("place_order_tool: operator error response: %s", res)
            return _json_fail("place_order_failed", **res)

        logger.info("place_order_tool: success response: %s", res)
        return _json_ok(order=res)
    except Exception as e:
        logger.exception("place_order_tool exception")
        return _json_fail("place_order_exception", detail=str(e))


@tool("Place Intraday Bracket Order")
def place_intraday_bracket_tool(input_str: Any = None, **kw) -> str:
    """
    Convenience wrapper for a typical *intraday bracket-style* order:

    - Entry: MARKET, product="I"
    - Exits: separate LIMIT target + SL-M stop-loss (created by UpstoxOperator)
    - Stop-loss is MANDATORY (absolute or %); target is optional.

    Input (dict | JSON | kwargs):
      {
        "symbol": "HFCL",
        "side": "BUY",              # BUY or SELL
        "qty": 1,                   # optional if auto_size=true (then broker may cap)
        "stop_loss_pct": 0.5,       # OR "stop_loss": 75.4
        "target_pct": 1.0,          # OR "target": 76.5 (optional)
        "live": false,              # default false; MUST be true for real orders
        "auto_size": false          # optional, passed through to operator
      }

    Returns:
      {
        "ok": true/false,
        "order": { ... full UpstoxOperator.place_order result ... }
      }
    """
    p = _coerce_args(input_str, **kw)

    # Hard intraday + MARKET defaults
    p.setdefault("product", "I")
    p.setdefault("order_type", "MARKET")
    p.setdefault("live", False)

    # Validate side/symbol
    symbol = (p.get("symbol") or "").strip()
    side   = str(p.get("side", "BUY") or "BUY").upper()
    if not symbol:
        return _json_fail("missing_symbol", detail="symbol is required")
    if side not in ("BUY", "SELL"):
        return _json_fail("invalid_side", detail=f"side must be BUY/SELL, got {side}")

    # Require SL (absolute or pct)
    if not (p.get("stop_loss") or p.get("stop_loss_pct")):
        return _json_fail(
            "missing_stop_loss",
            detail="Provide stop_loss or stop_loss_pct for bracket order."
        )

    logger.info(
        "place_intraday_bracket_tool called: symbol=%s side=%s qty=%s product=%s live=%s "
        "stop_loss=%s stop_loss_pct=%s target=%s target_pct=%s auto_size=%s",
        symbol, side, p.get("qty"), p.get("product"), p.get("live"),
        p.get("stop_loss"), p.get("stop_loss_pct"),
        p.get("target"), p.get("target_pct"),
        p.get("auto_size")
    )

    try:
        res = OP.place_order(**p)

        if isinstance(res, dict) and res.get("error"):
            logger.error("place_intraday_bracket_tool: operator error response: %s", res)
            return _json_fail("place_intraday_bracket_failed", **res)

        logger.info("place_intraday_bracket_tool: success response: %s", res)
        return _json_ok(order=res)
    except Exception as e:
        logger.exception("place_intraday_bracket_tool exception")
        return _json_fail("place_intraday_bracket_exception", detail=str(e))


@tool("Square Off Position")
def square_off_tool(input_str: Any = None, **kw) -> str:
    """
    Close open position in the symbol and clean related GTTs (per operator).

    NOTE: In your current UpstoxOperator version, exits are placed as
    separate orders (not GTT). However, this tool still calls
    OP.get_gtt_list()/cancel_gtt() if the operator implements that
    behavior internally.

    Input: { "symbol": "HFCL", "live": true }
    """
    p = _coerce_args(input_str, **kw)
    try:
        logger.info("square_off_tool called: %s", {k: p.get(k) for k in ("symbol", "instrument_key", "live")})
        if "instrument_key" in inspect.signature(OP.square_off).parameters:
            res = OP.square_off(**p)
        else:
            # fallback name if your operator uses a different function
            if hasattr(OP, "square_off_symbol"):
                res = OP.square_off_symbol(**p)
            else:
                res = OP.square_off(**p)  # let it raise if truly unavailable
        logger.info("square_off_tool result: %s", res)
        return _json_ok(result=res)
    except Exception as e:
        logger.exception("square_off_tool exception")
        return _json_fail("square_off_failed", detail=str(e))


@tool("Calculate Trade Metrics")
def calculate_trade_metrics_tool(input_str: Any = None, **kw) -> str:
    """
    Fast, deterministic risk math for agent decisions and logs.

    Input:
      {
        "side":"BUY","entry":406.2,"qty":100,
        # Provide one of:
        "stop_loss":401.3,                      # absolute
        "stop_loss_pct":1.2,                    # or percent
        # or ATR-based:
        "atr":0.75, "atr_mult":1.5,
        # Optional target:
        "target":416.0, "target_rr":2.0,
        "fees_total":8.5
      }
    """
    p = _coerce_args(input_str, **kw)
    try:
        side = str(p["side"]).upper()
        if side not in ("BUY", "SELL"):
            logger.warning("calculate_trade_metrics_tool: invalid side=%r", side)
            return _json_fail("side_invalid")
        entry = float(p["entry"])
        qty   = int(p["qty"])

        sl = p.get("stop_loss")
        sl_pct = p.get("stop_loss_pct")
        atr = p.get("atr"); atr_mult = p.get("atr_mult")
        target = p.get("target"); target_rr = p.get("target_rr")
        fees_total = float(p.get("fees_total", 0.0))

        if sl is None and sl_pct is not None:
            sl_pct = float(sl_pct)
            sl = entry * (1 - sl_pct/100.0) if side == "BUY" else entry * (1 + sl_pct/100.0)
        if sl is None and (atr is not None and atr_mult is not None):
            sl = entry - float(atr) * float(atr_mult) if side == "BUY" else entry + float(atr) * float(atr_mult)
        if sl is None:
            logger.warning("calculate_trade_metrics_tool: missing all stop inputs")
            return _json_fail("metrics_failed", detail="Provide stop_loss or stop_loss_pct or (atr & atr_mult).")

        sl = float(sl)
        rps = abs(entry - sl)
        if rps <= 0:
            logger.warning("calculate_trade_metrics_tool: non-positive risk per share")
            return _json_fail("metrics_failed", detail="Non-positive risk per share.")

        if target is None and target_rr is not None:
            rr = float(target_rr)
            target = entry + rr * rps if side == "BUY" else entry - rr * rps

        reward_ps = None; rr_ratio = None
        if target is not None:
            target = float(target)
            reward_ps = abs(target - entry)
            rr_ratio  = reward_ps / rps if rps > 0 else None

        gross_risk   = rps * qty
        gross_reward = (reward_ps * qty) if reward_ps is not None else None
        net_risk   = gross_risk + fees_total
        net_reward = (gross_reward - fees_total) if gross_reward is not None else None

        be = None
        if qty > 0:
            fees_ps = fees_total / qty
            be = entry + fees_ps if side == "BUY" else entry - fees_ps

        out = {
            "side": side,
            "entry": entry,
            "qty": qty,
            "stop_loss": round(sl, 4),
            "risk_per_share": round(rps, 4),
            "gross_risk": round(gross_risk, 2),
            "fees_total": round(fees_total, 2),
            "net_risk": round(net_risk, 2),
            "breakeven": round(be, 4) if be is not None else None
        }
        if target is not None:
            out.update({
                "target": round(target, 4),
                "reward_per_share": round(reward_ps, 4),
                "gross_reward": round(gross_reward, 2),
                "net_reward": round(net_reward, 2),
                "rr_ratio": round(rr_ratio, 3) if rr_ratio is not None else None
            })
        logger.info(
            "calculate_trade_metrics_tool: side=%s entry=%s qty=%s stop=%s target=%s rr=%s",
            side, entry, qty, out["stop_loss"], out.get("target"), out.get("rr_ratio")
        )
        return _json_ok(**out)
    except Exception as e:
        logger.exception("calculate_trade_metrics_tool exception")
        return _json_fail("metrics_failed", detail=str(e))


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
@tool("Get Current IST Time")
def get_current_time_tool(input_str: Any = None, **kw) -> str:
    """
    Provide a stable IST clock source for time-based decisions and logging.
    """
    now = datetime.now(IST)
    logger.debug("get_current_time_tool: %s", now.isoformat())
    return _json_ok(now_ist=now.isoformat(), date=now.date().isoformat(), time=now.time().isoformat())


@tool("Round to Tick Size")
def round_to_tick_tool(input_str: Any = None, **kw) -> str:
    """
    Normalize any price to exchange/broker tick size for valid order placement.
    Input: { "price": 406.23, "tick_size": 0.05 }
    """
    p = _coerce_args(input_str, **kw)
    try:
        price = float(p["price"])
        tick = p.get("tick_size", p.get("tick", os.environ.get("TICK_SIZE", 0.05)))
        tick = float(tick)
        if tick <= 0:
            logger.warning("round_to_tick_tool: invalid tick_size=%s", tick)
            return _json_fail("invalid_tick_size")
        rounded = _round_to_tick(price, tick)
        logger.debug("round_to_tick_tool: price=%s tick=%s rounded=%s", price, tick, rounded)
        return _json_ok(price=price, tick_size=tick, rounded=float(f"{rounded:.4f}"))
    except Exception as e:
        logger.exception("round_to_tick_tool exception")
        return _json_fail("round_failed", detail=str(e))


@tool("Calculate ATR Stop Loss")
def calculate_atr_stop_tool(input_str: Any = None, **kw) -> str:
    """
    Convert a %ATR rule into an absolute stop (robust: can infer entry/ATR from symbol).

    Accepts (dict | JSON | kwargs):
      {
        "symbol": "ITC",        # optional; allows inference
        "entry": 406.2,         # optional; will infer from snapshot if symbol given
        "atr": 4.8,             # optional; absolute ATR; else infer "atr14" if available
        "atr_pct": 1.0,         # % of entry; if missing and atr known -> derive; else fallback 1.0
        "k": 1.0,               # ATR multiplier
        "side": "BUY"|"SELL"="BUY",
        "tick_size": 0.05
      }
    """
    try:
        args = _coerce_args(input_str, **kw)
        symbol    = args.get("symbol")
        entry     = args.get("entry", None)
        atr       = args.get("atr", None)
        atr_pct   = args.get("atr_pct", None)
        k         = float(args.get("k", 1.0) or 1.0)
        side      = str(args.get("side", "BUY") or "BUY").upper().strip()
        tick_size = args.get("tick_size")
        if tick_size is None:
            tick_size = float(os.environ.get("TICK_SIZE", "0.05"))

        logger.info(
            "calculate_atr_stop_tool called: symbol=%s entry=%s atr=%s atr_pct=%s k=%s side=%s",
            symbol, entry, atr, atr_pct, k, side
        )

        if side not in ("BUY", "SELL"):
            return _json_fail("invalid_side", detail=f"'{side}'")

        # infer entry/ATR from technical snapshot if needed
        if (entry is None or (atr is None and atr_pct is None)) and symbol:
            try:
                _debounce(f"snap:{symbol}:7", 150)
                snap = TECH.snapshot(symbol, days=int(args.get("days", 7) or 7))
                if entry is None:
                    entry = snap.get("current_price") or snap.get("last_close")
                indi = (snap or {}).get("indicators", {})
                if atr is None:
                    # support key variants from your client
                    atr = indi.get("ATR14") or indi.get("atr14") or indi.get("atr") or None
                logger.debug("calculate_atr_stop_tool inferred from snapshot: entry=%s atr=%s", entry, atr)
            except Exception as e_snap:
                logger.warning("calculate_atr_stop_tool: snapshot inference failed: %s", e_snap)

        if entry is None:
            return _json_fail("missing_entry", detail="Provide entry or symbol to infer it.")
        entry = float(entry)

        atr_val = float(atr) if atr is not None else None
        if atr_pct is not None:
            atr_pct_used = float(atr_pct)
        else:
            if atr_val is not None and entry > 0:
                atr_pct_used = (atr_val / entry) * 100.0
            else:
                atr_pct_used = 1.0  # sane default

        dist = entry * (atr_pct_used / 100.0) * k
        if side == "BUY":
            stop = entry - dist
            stop_pct = -(dist / entry) * 100.0
        else:
            stop = entry + dist
            stop_pct = (dist / entry) * 100.0

        stop_rounded = _round_to_tick(stop, float(tick_size))

        out = {
            "entry": round(float(entry), 4),
            "side": side,
            "atr": None if atr_val is None else round(float(atr_val), 4),
            "atr_pct_used": round(float(atr_pct_used), 4),
            "k": round(float(k), 4),
            "stop_loss": stop_rounded,
            "stop_loss_pct": round(float(stop_pct), 4)
        }
        logger.info("calculate_atr_stop_tool output: %s", out)
        return _json_ok(**out)
    except Exception as e:
        logger.exception("calculate_atr_stop_tool exception")
        return _json_fail("atr_failed", detail=str(e))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ALL_TOOLS = [
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
    place_intraday_bracket_tool,
    square_off_tool,
    calculate_trade_metrics_tool,
    get_current_time_tool,
    round_to_tick_tool,
    calculate_atr_stop_tool,
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="crew_tools self-test")
    ap.add_argument("--self-test", action="store_true", help="Run a simple smoke test")
    args = ap.parse_args()

    if args.self_test:
        print("Using Tool: Get Current IST Time")
        print("[SELF-TEST] get_current_time:", get_current_time_tool.run(""))

        print("Using Tool: Check Market Status")
        print("[SELF-TEST] market_status:", get_market_status_tool.run("{}"))

        print("Using Tool: Get Recent News and Broker Calls")
        print("[SELF-TEST] recent_news:", get_recent_news_tool.run(
            json.dumps({"lookback_days": 2, "max_items": 10, "mode": "live", "compact": True})
        ))

        print("Using Tool: Search News by Query")
        print("[SELF-TEST] search_news:", search_news_tool.run(
            json.dumps({"query": "HFCL results", "lookback_days": 7, "max_results": 10, "mode": "live", "compact": True})
        ))

        print("Using Tool: Get Technical Snapshot")
        print("[SELF-TEST] technical snapshot:", get_technical_snapshot_tool.run(
            json.dumps({"symbol": "HFCL", "days": 7})
        ))

        print("Using Tool: Get Account Funds")
        print("[SELF-TEST] funds:", get_funds_tool.run("{}"))

        print("Using Tool: Get Current Positions")
        print("[SELF-TEST] positions:", get_positions_tool.run("{}"))

        print("Using Tool: Get Holdings")
        print("[SELF-TEST] holdings:", get_holdings_tool.run("{}"))

        print("Using Tool: Get Portfolio Summary")
        print("[SELF-TEST] portfolio:", get_portfolio_summary_tool.run("{}"))

        print("Using Tool: Calculate Required Margin")
        print("[SELF-TEST] margin sample:", calculate_margin_tool.run(
            json.dumps({"symbol": "HFCL", "price": 406.2, "qty": 10, "product": "I"})
        ))

        print("Using Tool: Calculate Max Quantity")
        print("[SELF-TEST] max qty sample:", calculate_max_quantity_tool.run(
            json.dumps({"symbol": "HFCL", "price": 406.2, "product": "I", "risk_pct": 0.5, "stop_loss": 401.3})
        ))

        print("Using Tool: Round to Tick Size")
        print("[SELF-TEST] round_to_tick:", round_to_tick_tool.run(
            json.dumps({"price": 406.23, "tick_size": 0.05})
        ))

        print("Using Tool: Calculate ATR Stop Loss")
        print("[SELF-TEST] ATR stop:", calculate_atr_stop_tool.run(
            json.dumps({"entry": 406.2, "atr_pct": 1.2, "side": "BUY"})
        ))

        print("Using Tool: Calculate Trade Metrics")
        print("[SELF-TEST] metrics:", calculate_trade_metrics_tool.run(
            json.dumps({"side": "BUY", "entry": 406.2, "qty": 100, "stop_loss": 401.3, "target_rr": 2.0, "fees_total": 8.5})
        ))

        print("[SELF-TEST] OK")
