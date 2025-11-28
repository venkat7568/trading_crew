#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brokerage.py — Simple India cash-equity fee model (approx.)
All numbers are adjustable via env to match your statement.

Env overrides (optional):
  BROKERAGE_PER_ORDER         (₹) default 20.0
  EXCHANGE_TXN_BPS            (bps of turnover) default 3.25
  SEBI_CHARGES_PER_CR         (₹ per crore) default 10.0
  GST_PCT                     (%) default 18.0
  STT_DELIVERY_BPS            (bps) default 10.0
  STT_INTRADAY_BPS            (bps) default 2.5
  STAMP_BPS                   (bps on buy turnover) default 1.5
"""

import os
from dataclasses import dataclass
from typing import Literal, Dict

def _f(env, default):
    try:
        return float(os.environ.get(env, default))
    except Exception:
        return default

@dataclass
class BrokerageModel:
    brokerage_per_order: float = _f("BROKERAGE_PER_ORDER", 20.0)
    exchange_txn_bps: float    = _f("EXCHANGE_TXN_BPS", 3.25)       # exch txn + clearing (approx)
    sebi_per_cr: float         = _f("SEBI_CHARGES_PER_CR", 10.0)    # ₹ per crore
    gst_pct: float             = _f("GST_PCT", 18.0)
    stt_delivery_bps: float    = _f("STT_DELIVERY_BPS", 10.0)
    stt_intraday_bps: float    = _f("STT_INTRADAY_BPS", 2.5)
    stamp_bps: float           = _f("STAMP_BPS", 1.5)

    def estimate(
        self,
        side: Literal["BUY","SELL"],
        price: float,
        qty: int,
        product: Literal["I","D"],
        include_exit: bool = True,
        slippage_pct: float = 0.0
    ) -> Dict[str, float]:
        """
        Returns one-way or round-trip (include_exit=True) cost estimate, in ₹.
        - product "I" (intraday): lower STT; "D" (delivery): higher STT.
        - slippage_pct: optional % applied to entry+exit turnover.
        """
        price = float(price); qty = int(qty)
        notional = price * qty

        # Brokerage: per order
        brok_entry = self.brokerage_per_order
        brok_exit  = self.brokerage_per_order if include_exit else 0.0

        # Exchange txn + clearing (~bps of turnover)
        exch_entry = notional * (self.exchange_txn_bps / 10_000.0)
        exch_exit  = (notional * (self.exchange_txn_bps / 10_000.0)) if include_exit else 0.0

        # SEBI charges (₹ per crore of turnover)
        sebi_entry = (notional / 10_000_000.0) * self.sebi_per_cr
        sebi_exit  = ((notional / 10_000_000.0) * self.sebi_per_cr) if include_exit else 0.0

        # STT: depends on product; delivery far higher on sell (officially on sell side)
        if product == "D":
            stt_entry = 0.0
            stt_exit  = notional * (self.stt_delivery_bps / 10_000.0) if include_exit else 0.0
        else:
            # intraday (both legs tiny; conservatively apply on both)
            stt_entry = notional * (self.stt_intraday_bps / 10_000.0)
            stt_exit  = (notional * (self.stt_intraday_bps / 10_000.0)) if include_exit else 0.0

        # Stamp duty: on BUY turnover only (varies by state; simplified bps)
        stamp_entry = notional * (self.stamp_bps / 10_000.0)
        stamp_exit  = 0.0

        # GST % on (brokerage + exchange + clearing). Many brokers apply on brok + exch+clearing.
        gst_base_entry = brok_entry + exch_entry
        gst_base_exit  = brok_exit + exch_exit
        gst = (gst_base_entry + gst_base_exit) * (self.gst_pct / 100.0)

        # Slippage: apply % on both legs (entry+exit) turnover if include_exit
        slip = (notional * (slippage_pct/100.0)) * (2 if include_exit else 1)

        total = (
            brok_entry + brok_exit +
            exch_entry + exch_exit +
            sebi_entry + sebi_exit +
            stt_entry + stt_exit +
            stamp_entry + stamp_exit +
            gst + slip
        )
        return {
            "one_way": brok_entry + exch_entry + sebi_entry + stt_entry + stamp_entry + (gst_base_entry * self.gst_pct/100.0) + (notional * (slippage_pct/100.0)),
            "round_trip": total,
            "components": {
                "brokerage": brok_entry + brok_exit,
                "exchange": exch_entry + exch_exit,
                "sebi": sebi_entry + sebi_exit,
                "stt": stt_entry + stt_exit,
                "stamp": stamp_entry,
                "gst": gst,
                "slippage": slip
            }
        }
