#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watchlist_manager.py — Intelligent Watchlist & Memory System
=============================================================

Manages:
1. Intraday watchlist (monitor during session for entry opportunities)
2. Tomorrow's queue (carry forward incomplete setups)
3. Pattern memory (learn what works: best stocks, times, setups)
4. Entry quality tracking

Professional trading firms don't trade everything at once - they:
- Build a watchlist of promising setups
- Monitor continuously for optimal entry
- Learn from patterns over time
- Carry forward incomplete setups to next session
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional

IST = ZoneInfo(os.environ.get("TZ", "Asia/Kolkata"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DATA_DIR.mkdir(exist_ok=True)


class WatchlistManager:
    """Manages intraday watchlist, tomorrow's queue, and pattern learning"""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.watchlist_file = self.data_dir / "watchlist.json"
        self.patterns_file = self.data_dir / "patterns.json"

        self.watchlist = self._load_watchlist()
        self.patterns = self._load_patterns()

    # ========== Persistence ==========

    def _load_watchlist(self) -> Dict[str, Any]:
        if self.watchlist_file.exists():
            with open(self.watchlist_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "intraday": [],        # Active monitoring during session
            "tomorrow": [],         # Carry forward to next day
            "completed": [],        # Archive of today's completed setups
        }

    def _save_watchlist(self):
        with open(self.watchlist_file, "w", encoding="utf-8") as f:
            json.dump(self.watchlist, f, indent=2, ensure_ascii=False)

    def _load_patterns(self) -> Dict[str, Any]:
        if self.patterns_file.exists():
            with open(self.patterns_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "best_performers": {
                "intraday_morning": [],   # 9:15-11:30
                "intraday_afternoon": [],  # 11:30-15:15
                "swing": [],
            },
            "learned_setups": {},  # symbol -> {best_entry_time, avg_move, success_rate}
            "time_patterns": {},   # symbol -> best times to trade
        }

    def _save_patterns(self):
        with open(self.patterns_file, "w", encoding="utf-8") as f:
            json.dump(self.patterns, f, indent=2, ensure_ascii=False)

    # ========== Intraday Watchlist ==========

    def add_to_intraday_watchlist(
        self,
        symbol: str,
        signal: str,  # "BUY" or "SELL"
        reason: str,
        entry_target: Optional[float] = None,
        current_price: Optional[float] = None,
        confidence: float = 0.0,
        entry_quality: int = 0,
        setup_type: str = "pullback",  # pullback, breakout, reversal, etc.
        technical_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add stock to intraday watchlist for monitoring.

        Use cases:
        - BUY signal but price too extended (wait for pullback)
        - Good setup but not optimal entry yet
        - Strong signal but need volume confirmation
        """
        now = datetime.now(IST)

        # Check if already in watchlist
        for item in self.watchlist["intraday"]:
            if item["symbol"] == symbol:
                # Update existing entry
                item.update({
                    "signal": signal,
                    "reason": reason,
                    "entry_target": entry_target,
                    "current_price": current_price,
                    "confidence": confidence,
                    "entry_quality": entry_quality,
                    "setup_type": setup_type,
                    "updated_at": now.isoformat(),
                })
                if technical_data:
                    item["technical_data"] = technical_data
                self._save_watchlist()
                return item

        # Add new entry
        entry = {
            "symbol": symbol,
            "signal": signal,
            "reason": reason,
            "entry_target": entry_target,
            "current_price": current_price,
            "confidence": confidence,
            "entry_quality": entry_quality,
            "setup_type": setup_type,
            "technical_data": technical_data or {},
            "added_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "expires_at": now.replace(hour=15, minute=15).isoformat(),  # Market close
            "checks": 0,  # How many times we've checked this setup
        }

        self.watchlist["intraday"].append(entry)
        self._save_watchlist()
        return entry

    def check_intraday_watchlist(
        self,
        tech_client: Any,
        current_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check intraday watchlist for entry triggers.

        Returns list of stocks ready for entry.
        Call this every 15-30 minutes during market hours.
        """
        current_time = current_time or datetime.now(IST)
        ready_for_entry = []

        for item in self.watchlist["intraday"][:]:  # Copy to allow removal
            symbol = item["symbol"]
            item["checks"] += 1

            # Check expiration
            expires_at = datetime.fromisoformat(item["expires_at"])
            if current_time >= expires_at:
                # Move to completed
                item["status"] = "expired"
                self.watchlist["completed"].append(item)
                self.watchlist["intraday"].remove(item)
                continue

            try:
                # Get current price and technical data
                current_price, _ = tech_client.ltp(item["technical_data"].get("instrument_key", ""))
                if current_price is None:
                    continue

                item["current_price"] = current_price
                item["last_check_at"] = current_time.isoformat()

                # Check if entry target is hit
                entry_target = item.get("entry_target")
                signal = item["signal"]

                if entry_target:
                    tolerance = entry_target * 0.003  # 0.3% tolerance

                    if signal == "BUY":
                        # Waiting for price to pull back to entry_target
                        if current_price <= entry_target + tolerance:
                            item["entry_triggered"] = True
                            item["trigger_price"] = current_price
                            item["trigger_time"] = current_time.isoformat()
                            ready_for_entry.append(item)

                    elif signal == "SELL":
                        # Waiting for price to rise to entry_target
                        if current_price >= entry_target - tolerance:
                            item["entry_triggered"] = True
                            item["trigger_price"] = current_price
                            item["trigger_time"] = current_time.isoformat()
                            ready_for_entry.append(item)

            except Exception as e:
                item["error"] = str(e)
                continue

        self._save_watchlist()
        return ready_for_entry

    def mark_watchlist_item_executed(self, symbol: str, execution_data: Dict[str, Any]):
        """Mark a watchlist item as executed and archive it"""
        for item in self.watchlist["intraday"][:]:
            if item["symbol"] == symbol:
                item["status"] = "executed"
                item["execution"] = execution_data
                item["executed_at"] = datetime.now(IST).isoformat()
                self.watchlist["completed"].append(item)
                self.watchlist["intraday"].remove(item)
                break

        self._save_watchlist()

    def clear_expired_intraday(self):
        """Clean up expired intraday watchlist items (call at end of day)"""
        now = datetime.now(IST)

        for item in self.watchlist["intraday"][:]:
            expires_at = datetime.fromisoformat(item["expires_at"])
            if now >= expires_at:
                item["status"] = "expired"
                self.watchlist["completed"].append(item)
                self.watchlist["intraday"].remove(item)

        self._save_watchlist()

    # ========== Tomorrow's Queue ==========

    def add_to_tomorrow_queue(
        self,
        symbol: str,
        setup: str,  # "breakout_watch", "pullback_continuation", "reversal", etc.
        trigger_price: Optional[float] = None,
        reason: str = "",
        technical_data: Optional[Dict[str, Any]] = None,
        priority: int = 50,  # 0-100 (higher = more important)
    ) -> Dict[str, Any]:
        """
        Add incomplete setup to tomorrow's queue.

        Use cases:
        - Consolidation pattern forming, watch for breakout tomorrow
        - Good fundamental news but no technical setup yet
        - End of day - didn't get entry but setup still valid
        """
        now = datetime.now(IST)

        # Check if already in queue
        for item in self.watchlist["tomorrow"]:
            if item["symbol"] == symbol:
                # Update existing
                item.update({
                    "setup": setup,
                    "trigger_price": trigger_price,
                    "reason": reason,
                    "priority": priority,
                    "updated_at": now.isoformat(),
                })
                if technical_data:
                    item["technical_data"] = technical_data
                self._save_watchlist()
                return item

        # Add new
        entry = {
            "symbol": symbol,
            "setup": setup,
            "trigger_price": trigger_price,
            "reason": reason,
            "technical_data": technical_data or {},
            "priority": priority,
            "added_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "days_waiting": 0,
        }

        self.watchlist["tomorrow"].append(entry)
        self._save_watchlist()
        return entry

    def get_tomorrow_priority_list(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N priority stocks for tomorrow.
        Call this at start of day.
        """
        # Increment days_waiting
        for item in self.watchlist["tomorrow"]:
            item["days_waiting"] = item.get("days_waiting", 0) + 1

        # Sort by priority (desc) and days_waiting (asc for freshness)
        sorted_queue = sorted(
            self.watchlist["tomorrow"],
            key=lambda x: (x["priority"], -x.get("days_waiting", 0)),
            reverse=True,
        )

        self._save_watchlist()
        return sorted_queue[:top_n]

    def remove_from_tomorrow_queue(self, symbol: str):
        """Remove symbol from tomorrow's queue (e.g., setup invalidated)"""
        self.watchlist["tomorrow"] = [
            item for item in self.watchlist["tomorrow"] if item["symbol"] != symbol
        ]
        self._save_watchlist()

    # ========== Pattern Learning ==========

    def record_trade_result(
        self,
        symbol: str,
        style: str,  # "intraday" or "swing"
        entry_time: str,
        exit_time: Optional[str],
        pnl_pct: float,
        setup_type: str,
    ):
        """Record trade result for pattern learning"""
        now = datetime.now(IST)
        entry_dt = datetime.fromisoformat(entry_time)

        # Determine time bucket
        entry_hour = entry_dt.hour
        if 9 <= entry_hour < 11:
            time_bucket = "morning"
        elif 11 <= entry_hour < 13:
            time_bucket = "midday"
        else:
            time_bucket = "afternoon"

        # Update learned setups for this symbol
        if symbol not in self.patterns["learned_setups"]:
            self.patterns["learned_setups"][symbol] = {
                "trades": [],
                "success_rate": 0.0,
                "avg_move_pct": 0.0,
                "best_time_bucket": "",
            }

        setup_data = self.patterns["learned_setups"][symbol]
        setup_data["trades"].append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl_pct": pnl_pct,
            "setup_type": setup_type,
            "time_bucket": time_bucket,
        })

        # Calculate stats
        trades = setup_data["trades"][-50:]  # Last 50 trades
        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        setup_data["success_rate"] = wins / len(trades) if trades else 0.0
        setup_data["avg_move_pct"] = sum(t["pnl_pct"] for t in trades) / len(trades) if trades else 0.0

        # Find best time bucket
        time_buckets = {}
        for t in trades:
            bucket = t["time_bucket"]
            if bucket not in time_buckets:
                time_buckets[bucket] = []
            time_buckets[bucket].append(t["pnl_pct"])

        best_bucket = ""
        best_avg = -999
        for bucket, pnls in time_buckets.items():
            avg = sum(pnls) / len(pnls) if pnls else 0
            if avg > best_avg:
                best_avg = avg
                best_bucket = bucket

        setup_data["best_time_bucket"] = best_bucket

        # Update best performers
        if style == "intraday":
            key = f"intraday_{time_bucket}"
            performers = self.patterns["best_performers"].get(key, [])

            # Add/update this symbol
            found = False
            for perf in performers:
                if perf["symbol"] == symbol:
                    perf["success_rate"] = setup_data["success_rate"]
                    perf["avg_move_pct"] = setup_data["avg_move_pct"]
                    found = True
                    break

            if not found and setup_data["success_rate"] > 0.5:  # Only add if > 50% success
                performers.append({
                    "symbol": symbol,
                    "success_rate": setup_data["success_rate"],
                    "avg_move_pct": setup_data["avg_move_pct"],
                })

            # Sort by success_rate * avg_move_pct (quality score)
            performers.sort(
                key=lambda x: x["success_rate"] * abs(x["avg_move_pct"]),
                reverse=True,
            )

            self.patterns["best_performers"][key] = performers[:20]  # Top 20

        self._save_patterns()

    def get_best_symbols_for_time(self, current_time: Optional[datetime] = None) -> List[str]:
        """Get best symbols to trade based on current time of day"""
        current_time = current_time or datetime.now(IST)
        hour = current_time.hour

        if 9 <= hour < 11:
            time_bucket = "intraday_morning"
        elif 11 <= hour < 13:
            time_bucket = "intraday_midday"
        else:
            time_bucket = "intraday_afternoon"

        performers = self.patterns["best_performers"].get(time_bucket, [])
        return [p["symbol"] for p in performers[:10]]  # Top 10

    def should_trade_symbol_now(self, symbol: str, current_time: Optional[datetime] = None) -> bool:
        """Check if this is a good time to trade this symbol based on historical patterns"""
        current_time = current_time or datetime.now(IST)
        hour = current_time.hour

        if 9 <= hour < 11:
            time_bucket = "morning"
        elif 11 <= hour < 13:
            time_bucket = "midday"
        else:
            time_bucket = "afternoon"

        if symbol not in self.patterns["learned_setups"]:
            return True  # No data, allow trade

        setup_data = self.patterns["learned_setups"][symbol]
        best_bucket = setup_data.get("best_time_bucket", "")

        # If we have data and this is not the best time, return cautious
        if best_bucket and best_bucket != time_bucket:
            success_rate = setup_data.get("success_rate", 0.0)
            # Only allow if overall success rate is very high (> 70%)
            return success_rate > 0.70

        return True

    # ========== Summary & Cleanup ==========

    def get_status(self) -> Dict[str, Any]:
        """Get current watchlist status"""
        return {
            "intraday_count": len(self.watchlist["intraday"]),
            "tomorrow_count": len(self.watchlist["tomorrow"]),
            "completed_today": len(self.watchlist["completed"]),
            "intraday_items": self.watchlist["intraday"],
            "tomorrow_items": self.watchlist["tomorrow"][:5],  # Top 5
        }

    def end_of_day_cleanup(self):
        """
        Call at end of trading day (e.g., 3:30 PM).

        - Clear expired intraday items
        - Archive today's completed setups
        - Prepare tomorrow's queue
        """
        self.clear_expired_intraday()

        # Archive completed items to a daily file
        today = datetime.now(IST).strftime("%Y-%m-%d")
        archive_file = self.data_dir / f"watchlist_archive_{today}.json"

        with open(archive_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "date": today,
                    "completed": self.watchlist["completed"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Clear completed for fresh start tomorrow
        self.watchlist["completed"] = []
        self._save_watchlist()

        print(f"✅ End of day cleanup complete. Archive saved to {archive_file}")


# Convenience function
def get_watchlist_manager() -> WatchlistManager:
    """Get singleton watchlist manager instance"""
    return WatchlistManager()


if __name__ == "__main__":
    # Self-test
    wm = WatchlistManager()

    # Test adding to intraday watchlist
    wm.add_to_intraday_watchlist(
        symbol="IRFC",
        signal="BUY",
        reason="Bullish but RSI overbought (77), wait for pullback",
        entry_target=121.0,
        current_price=123.35,
        confidence=0.65,
        entry_quality=45,
        setup_type="pullback",
    )

    # Test adding to tomorrow's queue
    wm.add_to_tomorrow_queue(
        symbol="AAVAS",
        setup="breakout_watch",
        trigger_price=1850,
        reason="Daily consolidation + positive news, watch for breakout",
        priority=80,
    )

    print(json.dumps(wm.get_status(), indent=2))
