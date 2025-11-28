#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trade_tracker.py — Comprehensive P&L tracking and trade management
===================================================================

Tracks all trades from entry to exit with complete P&L calculation.
Provides market-ready JSON structure for all trading operations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# Set up logging
logger = logging.getLogger("trade_tracker")

# Timezone
IST = ZoneInfo("Asia/Kolkata")

# Data directory
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


class TradeTracker:
    """
    Manages trade lifecycle from entry to exit with P&L calculation.

    Features:
    - Track open positions with entry details
    - Calculate P&L on exit (realized profit/loss)
    - Store complete trade history with proper JSON structure
    - Support both intraday and swing trades
    - Provide trade analytics and reporting
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(exist_ok=True)

        # Files
        self.open_positions_file = self.data_dir / "open_positions.json"
        self.trades_history_file = self.data_dir / "trades_history.jsonl"
        self.daily_pnl_file = self.data_dir / "daily_pnl.jsonl"

        # Load open positions
        self.open_positions = self._load_open_positions()

    def _load_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Load currently open positions from file."""
        if not self.open_positions_file.exists():
            return {}
        try:
            with open(self.open_positions_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading open positions: {e}")
            return {}

    def _save_open_positions(self):
        """Save open positions to file."""
        try:
            with open(self.open_positions_file, "w", encoding="utf-8") as f:
                json.dump(self.open_positions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving open positions: {e}")

    def _append_trade_history(self, trade: Dict[str, Any]):
        """Append completed trade to history file."""
        try:
            with open(self.trades_history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error appending trade history: {e}")

    def _append_daily_pnl(self, pnl_entry: Dict[str, Any]):
        """Append daily P&L entry."""
        try:
            with open(self.daily_pnl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(pnl_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error appending daily P&L: {e}")

    def record_entry(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        product: str,
        order_id: Optional[str] = None,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        strategy: Optional[str] = None,
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a new position entry.

        Returns market-ready JSON structure for the entry.
        """
        trade_id = f"{symbol}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}"

        entry_data = {
            # Trade identification
            "trade_id": trade_id,
            "symbol": symbol,
            "order_id": order_id or "PENDING",

            # Entry details
            "side": side.upper(),  # BUY or SELL
            "quantity": quantity,
            "entry_price": float(entry_price),
            "entry_time": datetime.now(IST).isoformat(),
            "entry_timestamp": datetime.now(IST).timestamp(),

            # Position details
            "product": product.upper(),  # I (intraday) or D (delivery/swing)
            "position_type": "INTRADAY" if product.upper() == "I" else "SWING",

            # Risk management
            "stop_loss": float(stop_loss) if stop_loss else None,
            "target": float(target) if target else None,
            "risk_per_share": abs(entry_price - stop_loss) if stop_loss else None,
            "reward_per_share": abs(target - entry_price) if target else None,
            "rr_ratio": (
                abs(target - entry_price) / abs(entry_price - stop_loss)
                if (target and stop_loss and stop_loss != entry_price)
                else None
            ),

            # Capital allocation
            "position_value": float(entry_price * quantity),
            "max_risk": (
                float(abs(entry_price - stop_loss) * quantity)
                if stop_loss else None
            ),
            "potential_profit": (
                float(abs(target - entry_price) * quantity)
                if target else None
            ),

            # Strategy metadata
            "strategy": strategy or "default",
            "confidence": float(confidence) if confidence else None,
            "tags": tags or [],

            # Status
            "status": "OPEN",

            # Additional metadata
            "metadata": metadata or {},
        }

        # Store in open positions
        self.open_positions[trade_id] = entry_data
        self._save_open_positions()

        logger.info(f"✅ Position opened: {symbol} {side} {quantity}@{entry_price} (ID: {trade_id})")

        return entry_data

    def record_exit(
        self,
        trade_id: Optional[str] = None,
        symbol: Optional[str] = None,
        exit_price: Optional[float] = None,
        exit_reason: str = "MANUAL",
        order_id: Optional[str] = None,
        partial_quantity: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Record position exit and calculate P&L.

        Can identify position by trade_id or symbol.
        Supports partial exits.

        Returns complete trade record with P&L.
        """
        # Find the position
        if trade_id:
            if trade_id not in self.open_positions:
                logger.error(f"Trade ID {trade_id} not found in open positions")
                return {"error": "trade_not_found", "trade_id": trade_id}
            position = self.open_positions[trade_id]
        elif symbol:
            # Find most recent position for this symbol
            matching = [
                (tid, pos) for tid, pos in self.open_positions.items()
                if pos["symbol"] == symbol
            ]
            if not matching:
                logger.error(f"No open position found for {symbol}")
                return {"error": "position_not_found", "symbol": symbol}
            # Use most recent (latest timestamp)
            trade_id, position = max(matching, key=lambda x: x[1]["entry_timestamp"])
        else:
            return {"error": "trade_id_or_symbol_required"}

        # Get exit price (use provided or fetch current)
        if exit_price is None:
            logger.warning("Exit price not provided, P&L calculation will be incomplete")
            exit_price = position["entry_price"]  # Fallback

        exit_price = float(exit_price)

        # Calculate P&L
        entry_price = position["entry_price"]
        quantity = partial_quantity or position["quantity"]
        side = position["side"]

        if side == "BUY":
            # Long position: profit when exit > entry
            price_diff = exit_price - entry_price
            gross_pnl = price_diff * quantity
        else:  # SELL
            # Short position: profit when exit < entry
            price_diff = entry_price - exit_price
            gross_pnl = price_diff * quantity

        # Calculate returns
        pnl_percent = (price_diff / entry_price) * 100 if side == "BUY" else (price_diff / entry_price) * 100

        # Estimate charges (approximate)
        turnover = (entry_price + exit_price) * quantity
        brokerage = min(turnover * 0.0003, 40)  # 0.03% or ₹40 per order, whichever is lower
        stt = turnover * 0.00025 if side == "SELL" else 0  # 0.025% on sell side
        transaction_charges = turnover * 0.0000325  # NSE: 0.00325%
        gst = (brokerage + transaction_charges) * 0.18  # 18% GST
        sebi_charges = turnover * 0.000001  # ₹10 per crore
        stamp_duty = turnover * 0.00003 if side == "BUY" else 0  # 0.003% on buy side

        total_charges = brokerage + stt + transaction_charges + gst + sebi_charges + stamp_duty
        net_pnl = gross_pnl - total_charges

        # Create complete trade record
        trade_record = {
            **position,  # Include all entry data

            # Exit details
            "exit_price": exit_price,
            "exit_time": datetime.now(IST).isoformat(),
            "exit_timestamp": datetime.now(IST).timestamp(),
            "exit_reason": exit_reason.upper(),
            "exit_order_id": order_id,

            # P&L calculation
            "price_diff": round(price_diff, 2),
            "gross_pnl": round(gross_pnl, 2),
            "charges": {
                "brokerage": round(brokerage, 2),
                "stt": round(stt, 2),
                "transaction_charges": round(transaction_charges, 2),
                "gst": round(gst, 2),
                "sebi_charges": round(sebi_charges, 2),
                "stamp_duty": round(stamp_duty, 2),
                "total": round(total_charges, 2),
            },
            "net_pnl": round(net_pnl, 2),
            "pnl_percent": round(pnl_percent, 2),
            "return_on_capital": round((net_pnl / position["position_value"]) * 100, 2),

            # Trade duration
            "duration_seconds": datetime.now(IST).timestamp() - position["entry_timestamp"],
            "duration_minutes": round((datetime.now(IST).timestamp() - position["entry_timestamp"]) / 60, 1),

            # Status
            "status": "CLOSED",

            # Target/SL hit analysis
            "target_hit": (
                (side == "BUY" and exit_price >= position["target"]) or
                (side == "SELL" and exit_price <= position["target"])
            ) if position.get("target") else False,
            "sl_hit": (
                (side == "BUY" and exit_price <= position["stop_loss"]) or
                (side == "SELL" and exit_price >= position["stop_loss"])
            ) if position.get("stop_loss") else False,

            # R multiple (how many R's of risk did we make/lose)
            "r_multiple": (
                round(price_diff / position["risk_per_share"], 2)
                if position.get("risk_per_share") and position["risk_per_share"] != 0
                else None
            ),
        }

        # Handle partial exits
        if partial_quantity and partial_quantity < position["quantity"]:
            # Partial exit - update position quantity
            position["quantity"] -= partial_quantity
            self._save_open_positions()
            logger.info(f"📊 Partial exit: {symbol} {partial_quantity} units @ {exit_price} (P&L: ₹{net_pnl:.2f})")
        else:
            # Full exit - remove from open positions
            del self.open_positions[trade_id]
            self._save_open_positions()
            logger.info(f"✅ Position closed: {symbol} @ {exit_price} (P&L: ₹{net_pnl:.2f}, {pnl_percent:+.2f}%)")

        # Save to trade history
        self._append_trade_history(trade_record)

        # Update daily P&L
        self._append_daily_pnl({
            "date": datetime.now(IST).date().isoformat(),
            "time": datetime.now(IST).isoformat(),
            "symbol": position["symbol"],
            "trade_id": trade_id,
            "net_pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "charges": total_charges,
            "product": position["product"],
        })

        return trade_record

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all currently open positions."""
        return list(self.open_positions.values())

    def get_position(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get specific open position by trade_id."""
        return self.open_positions.get(trade_id)

    def get_position_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get most recent open position for a symbol."""
        matching = [
            pos for pos in self.open_positions.values()
            if pos["symbol"] == symbol
        ]
        if not matching:
            return None
        return max(matching, key=lambda x: x["entry_timestamp"])

    def get_daily_pnl(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get P&L summary for a specific date.

        Args:
            date: Date in YYYY-MM-DD format. Defaults to today.
        """
        if date is None:
            date = datetime.now(IST).date().isoformat()

        if not self.daily_pnl_file.exists():
            return {
                "date": date,
                "total_trades": 0,
                "gross_pnl": 0.0,
                "net_pnl": 0.0,
                "charges": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
                "trades": [],
            }

        trades = []
        gross_pnl = 0.0
        net_pnl = 0.0
        charges = 0.0
        winning = 0
        losing = 0

        try:
            with open(self.daily_pnl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("date") == date:
                            trades.append(entry)
                            net_pnl += entry.get("net_pnl", 0)
                            gross_pnl += entry.get("gross_pnl", 0)
                            charges += entry.get("charges", 0)
                            if entry.get("net_pnl", 0) > 0:
                                winning += 1
                            elif entry.get("net_pnl", 0) < 0:
                                losing += 1
                    except:
                        continue
        except Exception as e:
            logger.error(f"Error reading daily P&L: {e}")

        return {
            "date": date,
            "total_trades": len(trades),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "charges": round(charges, 2),
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": round((winning / len(trades)) * 100, 1) if trades else 0.0,
            "trades": trades,
        }

    def get_trade_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get trading statistics for the last N days."""
        if not self.trades_history_file.exists():
            return {"error": "no_trade_history"}

        trades = []
        try:
            with open(self.trades_history_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        trade = json.loads(line.strip())
                        trades.append(trade)
                    except:
                        continue
        except Exception as e:
            logger.error(f"Error reading trade history: {e}")
            return {"error": str(e)}

        if not trades:
            return {"total_trades": 0, "message": "no_completed_trades"}

        # Filter by date range if needed
        # For now, use all trades

        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get("net_pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("net_pnl", 0) < 0]

        total_pnl = sum(t.get("net_pnl", 0) for t in trades)
        total_charges = sum(t.get("charges", {}).get("total", 0) for t in trades)

        avg_win = sum(t.get("net_pnl", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.get("net_pnl", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": round((len(winning_trades) / total_trades) * 100, 1),

            "total_pnl": round(total_pnl, 2),
            "total_charges": round(total_charges, 2),
            "average_pnl_per_trade": round(total_pnl / total_trades, 2),

            "average_win": round(avg_win, 2),
            "average_loss": round(avg_loss, 2),
            "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else float('inf'),

            "largest_win": round(max((t.get("net_pnl", 0) for t in trades), default=0), 2),
            "largest_loss": round(min((t.get("net_pnl", 0) for t in trades), default=0), 2),
        }


# Global singleton instance
_tracker_instance: Optional[TradeTracker] = None


def get_trade_tracker() -> TradeTracker:
    """Get the global trade tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = TradeTracker()
    return _tracker_instance


# Convenience functions
def record_entry(**kwargs) -> Dict[str, Any]:
    """Record a position entry."""
    return get_trade_tracker().record_entry(**kwargs)


def record_exit(**kwargs) -> Dict[str, Any]:
    """Record a position exit."""
    return get_trade_tracker().record_exit(**kwargs)


def get_open_positions() -> List[Dict[str, Any]]:
    """Get all open positions."""
    return get_trade_tracker().get_open_positions()


def get_daily_pnl(date: Optional[str] = None) -> Dict[str, Any]:
    """Get daily P&L summary."""
    return get_trade_tracker().get_daily_pnl(date)


def get_trade_statistics(days: int = 30) -> Dict[str, Any]:
    """Get trade statistics."""
    return get_trade_tracker().get_trade_statistics(days)
