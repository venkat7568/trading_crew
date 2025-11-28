#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upstox_operator.py — Delivery-only trading with permanent monitoring
=====================================================================

Key features:
- ✅ DELIVERY ORDERS ONLY - No intraday leverage (product always "D")
- ✅ Stop-loss is REQUIRED for every live order.
  Provide either:
    • absolute: stop_loss=445.5
    • or percent: stop_loss_pct=0.5  (means 0.5%)
- 🎯 Target is OPTIONAL (absolute or percent via target / target_pct).
- 🧾 Entry can be MARKET or LIMIT (optional price).
- 📊 PERMANENT MONITORING: Target and stop-loss levels are returned for
     position_monitor to watch. NO automatic exit orders placed.
     Exits are executed by position_monitor when levels are hit.
- 💰 NO LEVERAGE: All trades use full capital (1x), no intraday margin.

Env (optional)
- UPSTOX_API_BASE       (default https://api.upstox.com)
- UPSTOX_ACCESS_TOKEN   (required for live)
- STRICT_LIVE_MODE      (default 1)
- MODE                  (default live)
- TZ                    (default Asia/Kolkata)
- ALLOW_INSECURE_SSL    (default false)
- TICK_SIZE             (default 0.05)

CLI examples:
  # Dry-run BUY with mandatory SL (0.5%) and optional target (1%)
  python upstox_operator.py --place --symbol ITC --side BUY --qty 1 \
      --order-type MARKET --product D --stop-loss-pct 0.5 --target-pct 1

  # Live LIMIT SELL with absolute SL/target  python upstox_operator.py --place --symbol RELIANCE --side SELL --qty 2 \
      --order-type LIMIT --price 2515 --product D \
      --stop-loss 2540 --target 2465 --live
"""

from __future__ import annotations

import os, json, time, logging
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

# ---------- logging ----------
def _mk_logger(name: str, level_env: str = "TECH_LOG_LEVEL", default_level: str = "WARNING") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(h)
    log.setLevel(os.environ.get(level_env, default_level).upper())
    return log

log = _mk_logger("upstox_operator", "OPERATOR_LOG_LEVEL", "INFO")

# ---------- optional dotenv ----------
try:
    from dotenv import load_dotenv, find_dotenv
    _DOT = find_dotenv(usecwd=True)
    if _DOT:
        load_dotenv(_DOT, override=False)
except Exception:
    pass

# ---------- deps ----------
try:
    import requests
    _HAS_REQ = True
except Exception:
    _HAS_REQ = False

# Bridge to your technical client
from upstox_technical import UpstoxTechnicalClient


# ---------- price helpers ----------
def _tick_size() -> float:
    try:
        return float(os.environ.get("TICK_SIZE", "0.05"))
    except Exception:
        return 0.05

def _round_to_tick(x: float) -> float:
    t = _tick_size()
    if t <= 0:
        return float(x)
    return round(round(float(x) / t) * t, 2)  # 2 decimals is okay for EQ


class UpstoxOperator:
    """High-level trading operations for Upstox with mandatory SL."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        api_base: Optional[str] = None,
        strict_live_mode: Optional[bool] = None,
        mode: Optional[str] = None,
        allow_insecure_ssl: Optional[bool] = None,
        tz: Optional[str] = None,
        session: Optional["requests.Session"] = None,
        tech: Optional[UpstoxTechnicalClient] = None,
    ) -> None:
        if not _HAS_REQ:
            raise RuntimeError("The 'requests' package is required for UpstoxOperator.")

        self.api_base = (api_base or os.environ.get("UPSTOX_API_BASE", "https://api.upstox.com")).rstrip("/")
        self.access_token = (access_token or os.environ.get("UPSTOX_ACCESS_TOKEN") or "").strip()
        self.strict_live_mode = bool(
            os.environ.get("STRICT_LIVE_MODE", "1").strip().lower() in ("1", "true", "yes", "on")
            if strict_live_mode is None else strict_live_mode
        )
        self.mode = (mode or os.environ.get("MODE", "live")).strip().lower()
        self.allow_insecure_ssl = bool(
            os.environ.get("ALLOW_INSECURE_SSL", "").lower() == "true"
            if allow_insecure_ssl is None else allow_insecure_ssl
        )
        self.IST = ZoneInfo(os.environ.get("TZ", tz or "Asia/Kolkata"))

        self.sess = session or requests.Session()
        self._last_req_ts = 0.0
        self._rate_gap = 0.10

        # technical client (resolver + quotes + candles)
        self.tech = tech or UpstoxTechnicalClient()

    # ---------- internals: HTTP ----------
    def _headers(self) -> Dict[str, str]:
        h = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "TradingCrew/Operator/3.2"
        }
        if self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        return h

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_req_ts
        if elapsed < self._rate_gap:
            time.sleep(self._rate_gap - elapsed)
        self._last_req_ts = time.time()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Tuple[int, Any]:
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        try:
            self._rate_limit()
            r = self.sess.get(
                url,
                params=params or {},
                headers=self._headers(),
                timeout=timeout,
                verify=not self.allow_insecure_ssl,
            )
            if r.status_code == 200 and r.content:
                try:
                    return r.status_code, r.json()
                except Exception:
                    return r.status_code, {}
            return r.status_code, (r.json() if r.content else {})
        except Exception as e:
            if not self.allow_insecure_ssl:
                log.warning("GET %s failed: %s", url, e)
                return 0, {"error": str(e)}
            # retry insecure
            try:
                r = self.sess.get(
                    url,
                    params=params or {},
                    headers=self._headers(),
                    timeout=timeout,
                    verify=False,
                )
                if r.status_code == 200 and r.content:
                    try:
                        return r.status_code, r.json()
                    except Exception:
                        return r.status_code, {}
                return r.status_code, (r.json() if r.content else {})
            except Exception as e2:
                log.error("GET %s failed (insecure): %s", url, e2)
                return 0, {"error": str(e2)}

    def _post(self, path: str, payload: Dict[str, Any], timeout: int = 30) -> Tuple[int, Any]:
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        try:
            self._rate_limit()
            r = self.sess.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=timeout,
                verify=not self.allow_insecure_ssl,
            )
            if r.status_code == 200 and r.content:
                try:
                    return r.status_code, r.json()
                except Exception:
                    return r.status_code, {}
            return r.status_code, (r.json() if r.content else {})
        except Exception as e:
            if not self.allow_insecure_ssl:
                log.warning("POST %s failed: %s", url, e)
                return 0, {"error": str(e)}
            # retry insecure
            try:
                r = self.sess.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=timeout,
                    verify=False,
                )
                if r.status_code == 200 and r.content:
                    try:
                        return r.status_code, r.json()
                    except Exception:
                        return r.status_code, {}
                return r.status_code, (r.json() if r.content else {})
            except Exception as e2:
                log.error("POST %s failed (insecure): %s", url, e2)
                return 0, {"error": str(e2)}

    def _delete(self, path: str, timeout: int = 30) -> Tuple[int, Any]:
        url = path if path.startswith("http") else f"{self.api_base}{path}"
        try:
            self._rate_limit()
            r = self.sess.delete(
                url,
                headers=self._headers(),
                timeout=timeout,
                verify=not self.allow_insecure_ssl,
            )
            if r.status_code == 200 and r.content:
                try:
                    return r.status_code, r.json()
                except Exception:
                    return r.status_code, {}
            return r.status_code, (r.json() if r.content else {})
        except Exception as e:
            if not self.allow_insecure_ssl:
                log.warning("DELETE %s failed: %s", url, e)
                return 0, {"error": str(e)}
            try:
                r = self.sess.delete(
                    url,
                    headers=self._headers(),
                    timeout=timeout,
                    verify=False,
                )
                if r.status_code == 200 and r.content:
                    try:
                        return r.status_code, r.json()
                    except Exception:
                        return r.status_code, {}
                return r.status_code, (r.json() if r.content else {})
            except Exception as e2:
                log.error("DELETE %s failed (insecure): %s", url, e2)
                return 0, {"error": str(e2)}

    # ---------- safety ----------
    def _live_guard(self, live: bool) -> Optional[Dict[str, Any]]:
        if not live:
            return None
        if self.strict_live_mode and self.mode not in ("live", "1", "true", "on"):
            return {
                "error": "strict_live_mode_block",
                "message": "MODE!=live and STRICT_LIVE_MODE enabled.",
            }
        if self.strict_live_mode and not self.access_token:
            return {
                "error": "strict_live_mode_block",
                "message": "UPSTOX_ACCESS_TOKEN missing",
            }
        return None

    # ---------- helpers ----------
    def _resolve(self, symbol_or_name_or_isin_or_key: str) -> Dict[str, Any]:
        q = (symbol_or_name_or_isin_or_key or "").strip()
        if "|" in q:  # already instrument_key
            return {"instrument_key": q, "symbol": None, "name": None}
        return self.tech.resolve(q)

    def _current_price(self, instrument_key: str) -> Optional[float]:
        px, _ = self.tech.ltp(instrument_key)
        if px is not None:
            return float(px)
        candles = self.tech.ohlc_30m_with_today(instrument_key, days=2)
        if candles and len(candles[0]) >= 5:
            return float(candles[0][4])
        return None

    def _compute_levels(
        self,
        side: str,
        entry_px: float,
        stop_loss: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        target: Optional[float] = None,
        target_pct: Optional[float] = None,
    ) -> Tuple[float, Optional[float]]:
        """
        Returns: (stop_price, target_price_optional)
        - stop_loss OR stop_loss_pct is mandatory.
        - target / target_pct optional.
        """
        s = side.upper()
        px = float(entry_px)

        if stop_loss is None and (stop_loss_pct is None or stop_loss_pct <= 0):
            raise ValueError("Stop-loss is mandatory: provide stop_loss or stop_loss_pct (>0).")

        # Compute stop
        if stop_loss is None:
            if s == "BUY":
                stop = px * (1.0 - float(stop_loss_pct) / 100.0)
            else:  # SELL
                stop = px * (1.0 + float(stop_loss_pct) / 100.0)
        else:
            stop = float(stop_loss)

        # Compute target (optional)
        tgt_val: Optional[float] = None
        if target is not None:
            tgt_val = float(target)
        elif target_pct is not None and target_pct > 0:
            if s == "BUY":
                tgt_val = px * (1.0 + float(target_pct) / 100.0)
            else:
                tgt_val = px * (1.0 - float(target_pct) / 100.0)

        # Directional sanity
        stop = _round_to_tick(stop)
        if s == "BUY" and stop >= px:
            # force below entry
            stop = _round_to_tick(px * 0.999)
        if s == "SELL" and stop <= px:
            # force above entry
            stop = _round_to_tick(px * 1.001)

        if tgt_val is not None:
            tgt_val = _round_to_tick(tgt_val)
            if s == "BUY" and tgt_val <= px:
                tgt_val = _round_to_tick(px * 1.001)
            if s == "SELL" and tgt_val >= px:
                tgt_val = _round_to_tick(px * 0.999)

        return stop, tgt_val

    # ---------- public API ----------
    def market_session_status(self) -> Dict[str, Any]:
        st, data = self._get("/v2/market/status/NSE")

        def _clock_open() -> bool:
            now = datetime.now(tz=self.IST)
            return (now.weekday() < 5) and (dtime(9, 15) <= now.time() <= dtime(15, 30))

        if st == 200 and isinstance(data, dict):
            d = data.get("data") or data
            status = str(d.get("status") or d.get("market_status") or "").lower()
            phase = str(d.get("phase") or d.get("market_phase") or "").lower()
            flags = [
                d.get("open"),
                d.get("is_open"),
                d.get("exchange_open"),
                d.get("is_market_open"),
                d.get("isTrading"),
                d.get("trading"),
            ]
            api_open = any(bool(x) for x in flags)
            text_open = status in {"open", "trading", "continuous", "normal_trading"} or phase in {
                "open",
                "trading",
                "continuous",
            }
            open_now = bool(api_open or text_open)
            if not open_now and _clock_open():
                return {
                    "open": True,
                    "status": status or "unknown",
                    "phase": phase or "unknown",
                    "source": "hybrid(api+clock)",
                    "reason": "api_closed_but_clock_open",
                }
            return {
                "open": open_now,
                "status": status or ("open" if open_now else "closed"),
                "phase": phase or ("open" if open_now else "closed"),
                "source": "upstox",
            }

        # fallback
        now = datetime.now(tz=self.IST)
        open_now = (now.weekday() < 5) and (dtime(9, 15) <= now.time() <= dtime(15, 30))
        return {"open": open_now, "status": ("open" if open_now else "closed"), "source": "local"}

    def wait_for_market_open(self, timeout_minutes: int = 120, poll_seconds: int = 5) -> Dict[str, Any]:
        deadline = time.time() + timeout_minutes * 60
        while time.time() < deadline:
            st = self.market_session_status()
            if st.get("open"):
                return {"ok": True, "status": st}
            time.sleep(poll_seconds)
        return {"ok": False, "status": self.market_session_status(), "error": "timeout"}

    # --- Funds & portfolio ---
    def get_funds(self, segment: str = "SEC") -> Dict[str, Any]:
        st, data = self._get("/v2/user/get-funds-and-margin", {"segment": segment})
        if st != 200 or not isinstance(data, dict):
            return {"error": "failed_to_fetch", "status": st, "response": data}
        d = data.get("data") or {}
        eq = d.get("equity") or {}
        return {
            "status": "ok",
            "equity": {
                "available_margin": float(eq.get("available_margin", 0) or 0),
                "used_margin": float(eq.get("used_margin", 0) or 0),
                "total_margin": float(eq.get("total_margin", 0) or 0),
                "opening_balance": float(eq.get("opening_balance", 0) or 0),
            },
            "commodity": d.get("commodity"),
            "raw": d,
        }

    def get_positions(self, include_closed: bool = False) -> Dict[str, Any]:
        st, data = self._get("/v2/portfolio/short-term-positions")
        if st != 200 or not isinstance(data, dict):
            return {"error": "failed_to_fetch", "status": st, "response": data}
        pos = data.get("data") or []
        if not include_closed:
            pos = [p for p in pos if int(p.get("quantity", 0) or 0) != 0]
        total_pnl = sum(float(p.get("unrealised_profit") or p.get("realised_profit") or 0) for p in pos)
        total_val = sum(float(p.get("quantity", 0) or 0) * float(p.get("last_price", 0) or 0) for p in pos)
        return {
            "status": "ok",
            "positions": pos,
            "count": len(pos),
            "total_pnl": total_pnl,
            "total_value": total_val,
        }

    def get_holdings(self) -> Dict[str, Any]:
        st, data = self._get("/v2/portfolio/long-term-holdings")
        if st != 200 or not isinstance(data, dict):
            return {"error": "failed_to_fetch", "status": st, "response": data}
        h = [x for x in (data.get("data") or []) if float(x.get("quantity", 0) or 0) > 0]
        invested = sum(float(x.get("quantity", 0) or 0) * float(x.get("average_price", 0) or 0) for x in h)
        current = sum(float(x.get("quantity", 0) or 0) * float(x.get("last_price", 0) or 0) for x in h)
        pnl = current - invested
        return {
            "status": "ok",
            "holdings": h,
            "count": len(h),
            "total_invested": invested,
            "total_current_value": current,
            "total_pnl": pnl,
            "pnl_percent": (pnl / invested * 100.0) if invested > 0 else 0.0,
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        f = self.get_funds()
        p = self.get_positions()
        h = self.get_holdings()
        return {
            "funds": f.get("equity", {}) if f.get("status") == "ok" else {},
            "positions": {
                "count": p.get("count", 0),
                "pnl": p.get("total_pnl", 0),
                "value": p.get("total_value", 0),
            }
            if p.get("status") == "ok"
            else {"count": 0, "pnl": 0, "value": 0},
            "holdings": {
                "count": h.get("count", 0),
                "invested": h.get("total_invested", 0),
                "current": h.get("total_current_value", 0),
                "pnl": h.get("total_pnl", 0),
                "pnl_percent": h.get("pnl_percent", 0),
            }
            if h.get("status") == "ok"
            else {"count": 0, "invested": 0, "current": 0, "pnl": 0, "pnl_percent": 0},
            "timestamp": datetime.now(tz=self.IST).isoformat(),
        }

    # --- Margin ---
    def calculate_required_margin(
        self,
        symbol: str,
        qty: int,
        price: float,
        side: str = "BUY",
        product: str = "D",
        instrument_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not instrument_key:
            instrument_key = self._resolve(symbol)["instrument_key"]
        payload = {
            "instruments": [
                {
                    "instrument_key": instrument_key,
                    "quantity": int(qty),
                    "transaction_type": side.upper(),
                    "product": "D",  # Always delivery
                    "price": float(price),
                }
            ]
        }
        st, data = self._post("/v2/charges/margin", payload)
        if st == 200 and isinstance(data, dict):
            charges = (data.get("data") or {}).get("charges") or {}
            return {
                "status": "ok",
                "required_margin": float(charges.get("total", 0) or 0),
                "charges": {
                    "transaction_charges": float(charges.get("transaction_charge", 0) or 0),
                    "gst": float(charges.get("gst", 0) or 0),
                    "stt": float(charges.get("stt", 0) or 0),
                    "stamp_duty": float(charges.get("stamp_duty", 0) or 0),
                    "total": float(charges.get("total", 0) or 0),
                },
                "product": "D",
                "leverage": 1.0,
                "raw": data.get("data"),
            }
        # fallback model - always delivery, no leverage
        req = price * qty * 1.00
        return {
            "required_margin": req,
            "charges": {"total": req * 0.001},
            "product": "D",
            "leverage": 1.0,
            "note": "estimated (margin API unavailable)",
        }

    def calculate_max_quantity(
        self,
        symbol: str,
        price: float,
        available_margin: float,
        product: str = "D",
        safety_buffer: float = 0.90,
    ) -> Dict[str, Any]:
        """
        Simple margin-based sizing for delivery orders (no leverage).

        :param available_margin: equity.available_margin
        :return: dict with max_quantity, required_margin, remaining_margin, etc.
        """
        usable = max(0.0, available_margin) * max(0.0, min(1.0, safety_buffer))
        # Always delivery, no leverage - full amount required
        max_qty = int(usable / max(price, 0.01))
        if max_qty <= 0:
            return {
                "max_quantity": 0,
                "required_margin": 0,
                "remaining_margin": available_margin,
                "error": "insufficient_funds",
            }
        m = self.calculate_required_margin(symbol, max_qty, price, "BUY", "D")
        req = float(m.get("required_margin", 0) or 0)
        return {
            "max_quantity": max_qty,
            "required_margin": req,
            "remaining_margin": available_margin - req,
            "product": "D",
            "leverage": 1.0,
        }

    # --- Orders / exits (persistent monitoring) ---
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: Optional[float] = None,
        order_type: str = "MARKET",
        product: str = "D",
        target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        instrument_key: Optional[str] = None,
        live: bool = False,
        auto_size: bool = False,
        tag: Optional[str] = None,
        *,
        target_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place DELIVERY order with mandatory stop-loss. Target/stop monitored by position_monitor.

        Behaviour:
        - Entry: /v2/order/place (delivery product only)
        - NO automatic exit orders - position_monitor watches for target/stop hits
        - Returns computed stop_loss and target levels for monitoring
        """
        blk = self._live_guard(live)
        if blk:
            return blk

        row = self._resolve(instrument_key or symbol)
        ik = row["instrument_key"]

        # Determine entry price (for % computations)
        entry_px = None
        if order_type.upper() == "MARKET" and price is None:
            entry_px = self._current_price(ik)
        else:
            entry_px = price

        if entry_px is None:
            entry_px = self._current_price(ik)

        if entry_px is None:
            return {"error": "price_unavailable_for_stop_calc", "symbol": symbol}

        # Compute mandatory SL and optional target
        try:
            stop_px, tgt_px = self._compute_levels(
                side=side,
                entry_px=float(entry_px),
                stop_loss=stop_loss,
                stop_loss_pct=stop_loss_pct,
                target=target,
                target_pct=target_pct,
            )
        except ValueError as ve:
            return {"error": "stop_loss_required", "message": str(ve)}

        # Auto-size if requested (BUY only, by funds)
        if auto_size and side.upper() == "BUY":
            funds = self.get_funds()
            avail = float(funds.get("equity", {}).get("available_margin", 0) or 0)
            px_for_size = float(entry_px) if price is not None else (self._current_price(ik) or entry_px)
            cap = self.calculate_max_quantity(symbol, px_for_size, avail, product)
            cand = int(cap.get("max_quantity", 0))
            if cand < qty:
                log.warning("Auto-size: requested %d, allowed %d by margin", qty, cand)
                qty = cand
            if qty <= 0:
                return {"error": "insufficient_funds_for_order", "suggested_qty": cand}

        # Build entry payload - ALWAYS delivery product
        payload = {
            "instrument_token": ik,
            "instrument_key": ik,
            "transaction_type": side.upper(),
            "quantity": int(qty),
            "order_type": order_type.upper(),
            "product": "D",  # Always delivery, no intraday
            "validity": "DAY",
            "price": float(price or 0),
            "trigger_price": 0.0,
            "disclosed_quantity": 0,
            "is_amo": False,
            "tag": tag or "delivery_monitored",
        }

        if not live:
            # Dry-run shows computed SL / target to confirm
            return {
                "live": False,
                "dry_run": True,
                "request": payload,
                "computed": {"stop_loss": stop_px, "target": tgt_px},
                "note": "Set live=True to execute",
            }

        # 1) Place entry
        st, data = self._post("/v2/order/place", payload)
        if st != 200 or not isinstance(data, dict):
            return {
                "error": "order_failed",
                "status": st,
                "response": data,
                "request": payload,
            }

        entry = {
            "status": "success",
            "order_id": (data.get("data") or {}).get("order_id"),
            "data": data.get("data"),
        }

        # NO exit orders placed - position_monitor will watch for target/stop hits
        # Target and stop levels are returned for monitoring
        result: Dict[str, Any] = {
            "status": "success",
            "live": True,
            "symbol": row.get("symbol") or symbol,
            "side": side.upper(),
            "quantity": qty,
            "product": "D",
            "entry": entry,
            "monitoring": {
                "stop_loss": stop_px,
                "target": tgt_px,
                "note": "Levels monitored by position_monitor - will execute when hit",
            },
            "computed_levels": {"stop_loss": stop_px, "target": tgt_px},
            "timestamp": datetime.now(tz=self.IST).isoformat(),
        }
        return result

    def square_off(
        self,
        symbol: Optional[str] = None,
        instrument_key: Optional[str] = None,
        live: bool = False,
    ) -> Dict[str, Any]:
        """
        Square-off: MARKET order opposite to current position (delivery only).
        Used by position_monitor when target or stop-loss is hit.
        """
        blk = self._live_guard(live)
        if blk:
            return blk

        if not instrument_key:
            if not symbol:
                return {"error": "instrument_key_or_symbol_required"}
            instrument_key = self._resolve(symbol)["instrument_key"]

        pos = self.get_positions().get("positions") or []
        position = None
        for p in pos:
            if p.get("instrument_token") == instrument_key:
                position = p
                break

        if not position:
            return {
                "status": "ok",
                "message": "no_open_position",
                "instrument_key": instrument_key,
                "symbol": symbol,
            }

        qty = int(position.get("quantity", 0) or 0)
        if qty == 0:
            return {
                "status": "ok",
                "message": "already_flat",
                "instrument_key": instrument_key,
                "symbol": symbol,
            }

        side = "SELL" if qty > 0 else "BUY"

        payload = {
            "instrument_token": instrument_key,
            "instrument_key": instrument_key,
            "transaction_type": side,
            "quantity": abs(qty),
            "order_type": "MARKET",
            "product": "D",  # Always delivery
            "validity": "DAY",
            "price": 0.0,
            "trigger_price": 0.0,
            "disclosed_quantity": 0,
            "is_amo": False,
            "tag": "square_off_monitored",
        }

        if not live:
            return {
                "live": False,
                "dry_run": True,
                "request": payload,
                "note": "Set live=True to execute square-off",
            }

        st, data = self._post("/v2/order/place", payload)
        if st != 200 or not isinstance(data, dict):
            return {
                "status": "error",
                "message": "square_off_failed",
                "http_status": st,
                "response": data,
                "request": payload,
            }

        return {
            "status": "ok",
            "symbol": symbol,
            "instrument_key": instrument_key,
            "squared_qty": abs(qty),
            "side": side,
            "order_id": (data.get("data") or {}).get("order_id"),
            "data": data.get("data"),
            "timestamp": datetime.now(tz=self.IST).isoformat(),
        }


# ---------------- CLI (minimal; optional) ----------------
def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="UpstoxOperator — mandatory SL, optional target")
    ap.add_argument("--funds", action="store_true")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--holdings", action="store_true")
    ap.add_argument("--portfolio", action="store_true")
    ap.add_argument("--market-status", action="store_true")

    ap.add_argument("--place", action="store_true")
    ap.add_argument("--square", action="store_true")

    ap.add_argument("--symbol", default="ITC", help="Name / symbol / ISIN / instrument_key")
    ap.add_argument("--instrument-key", help="If provided, skips resolver")
    ap.add_argument("--side", default="BUY", choices=["BUY", "SELL"])
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--price", type=float)
    ap.add_argument("--order-type", default="MARKET", choices=["MARKET", "LIMIT", "SL", "SL-M"])
    ap.add_argument("--product", default="D", help="Product type (always D for delivery)")

    # target/SL absolute
    ap.add_argument("--target", type=float)
    ap.add_argument("--stop-loss", type=float)

    # target/SL percent (e.g., 0.1 means 0.1%)
    ap.add_argument("--target-pct", type=float)
    ap.add_argument("--stop-loss-pct", type=float)

    ap.add_argument("--live", action="store_true")
    ap.add_argument("--auto-size", action="store_true")

    args = ap.parse_args()
    op = UpstoxOperator()

    if args.market_status:
        print(json.dumps(op.market_session_status(), indent=2))
        return
    if args.funds:
        print(json.dumps(op.get_funds(), indent=2))
        return
    if args.positions:
        print(json.dumps(op.get_positions(), indent=2))
        return
    if args.holdings:
        print(json.dumps(op.get_holdings(), indent=2))
        return
    if args.portfolio:
        print(json.dumps(op.get_portfolio_summary(), indent=2))
        return

    if args.place:
        out = op.place_order(
            symbol=args.symbol,
            side=args.side,
            qty=args.qty,
            price=args.price,
            order_type=args.order_type,
            product=args.product,
            target=args.target,
            stop_loss=args.stop_loss,
            instrument_key=args.instrument_key,
            live=args.live,
            auto_size=args.auto_size,
            target_pct=args.target_pct,
            stop_loss_pct=args.stop_loss_pct,
        )
        print(json.dumps(out, indent=2))
        return

    if args.square:
        out = op.square_off(symbol=args.symbol, instrument_key=args.instrument_key, live=args.live)
        print(json.dumps(out, indent=2))
        return

    ap.print_help()


if __name__ == "__main__":
    _cli()
