#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
position_monitor.py — Active monitoring of open positions
==========================================================

Monitors open positions to detect:
- Stop-loss hits
- Target hits
- Time-based exits (intraday square-off)
- Position health checks

This is CRITICAL because after placing an order, the system needs to know
what happened - did SL hit? Did target hit? Without this, the system is blind.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("position_monitor")

IST = ZoneInfo("Asia/Kolkata")


class PositionMonitor:
    """
    Monitors open positions and takes action when SL/target is hit.

    Features:
    - Check if stop-loss or target prices are hit
    - Auto-close positions when exit conditions met
    - Monitor intraday positions for time-based square-off
    - Track position health and unrealized P&L
    - Update trade tracker when positions exit
    """

    def __init__(self, operator=None, tech_client=None, trade_tracker=None):
        """
        Initialize position monitor.

        Args:
            operator: UpstoxOperator for checking positions and placing orders
            tech_client: UpstoxTechnicalClient for fetching current prices
            trade_tracker: TradeTracker for recording exits
        """
        self.operator = operator
        self.tech = tech_client
        self.tracker = trade_tracker

        # Initialize if not provided
        if not self.tech:
            try:
                from upstox_technical import UpstoxTechnicalClient
                self.tech = UpstoxTechnicalClient()
            except Exception as e:
                logger.error(f"Failed to init tech client: {e}")

        if not self.operator:
            try:
                from upstox_operator import UpstoxOperator
                self.operator = UpstoxOperator()
            except Exception as e:
                logger.error(f"Failed to init operator: {e}")

        if not self.tracker:
            try:
                from trade_tracker import get_trade_tracker
                self.tracker = get_trade_tracker()
            except Exception as e:
                logger.error(f"Failed to init tracker: {e}")

    def check_positions(self, live: bool = False) -> Dict[str, Any]:
        """
        Check all open positions and take action if SL/target hit.

        This is the main function that should be called periodically.

        Args:
            live: If True, will actually place orders to close positions

        Returns:
            Summary of actions taken
        """
        if not self.tech:
            return {"error": "tech_client_not_available"}

        try:
            # In backtest/paper mode, use trade tracker as primary source
            # In live mode, use broker positions
            if not live and self.tracker:
                # Backtest/paper trading mode: use tracker positions
                tracked_open = self.tracker.get_open_positions()

                if not tracked_open:
                    logger.info("No open positions to monitor")
                    return {
                        "open_positions": 0,
                        "actions_taken": [],
                        "message": "no_open_positions"
                    }

                actions_taken = []
                positions_checked = 0

                for tracked in tracked_open:
                    symbol = tracked.get("symbol")
                    if not symbol:
                        continue

                    positions_checked += 1

                    # Get current price (for backtest, use historical data or simulated price)
                    try:
                        # Try to get current price from tech client
                        instrument_key = tracked.get("instrument_key") or symbol
                        current_price, _ = self.tech.ltp(instrument_key)
                        if current_price is None:
                            # In backtest mode, if LTP fails, skip this position
                            logger.warning(f"Could not get LTP for {symbol} in backtest mode")
                            continue
                        current_price = float(current_price)
                    except Exception as e:
                        logger.error(f"Error getting price for {symbol}: {e}")
                        continue

                    # Get position details from tracker
                    stop_loss = tracked.get("stop_loss")
                    target = tracked.get("target")
                    side = tracked.get("side", "BUY").upper()
                    entry_price = tracked.get("entry_price", 0)
                    quantity = tracked.get("quantity", 0)
                    product = tracked.get("product", "I").upper()
                    trade_id = tracked.get("trade_id")

                    # Check for exit conditions
                    exit_reason = None
                    should_exit = False

                    # Check SL/Target hits
                    if side == "BUY":
                        if stop_loss and current_price <= stop_loss:
                            should_exit = True
                            exit_reason = "STOP_LOSS"
                        elif target and current_price >= target:
                            should_exit = True
                            exit_reason = "TARGET"
                    elif side == "SELL":
                        if stop_loss and current_price >= stop_loss:
                            should_exit = True
                            exit_reason = "STOP_LOSS"
                        elif target and current_price <= target:
                            should_exit = True
                            exit_reason = "TARGET"

                    # Check time-based exit for intraday
                    if not should_exit and product == "I":
                        should_exit, time_reason = self._check_time_exit()
                        if should_exit:
                            exit_reason = time_reason

                    # Execute exit if needed
                    if should_exit and exit_reason:
                        try:
                            # Record the exit in tracker
                            exit_record = self.tracker.record_exit(
                                trade_id=trade_id,
                                exit_price=current_price,
                                exit_reason=exit_reason
                            )

                            actions_taken.append({
                                "symbol": symbol,
                                "trade_id": trade_id,
                                "exit_price": current_price,
                                "exit_reason": exit_reason,
                                "pnl_record": exit_record
                            })

                            logger.info(f"✅ Exited {symbol}: {exit_reason} at {current_price}")

                        except Exception as e:
                            logger.error(f"Error recording exit for {symbol}: {e}")

                return {
                    "open_positions": positions_checked,
                    "actions_taken": actions_taken,
                    "positions_checked": positions_checked,
                    "message": f"checked_{positions_checked}_positions"
                }

            else:
                # Live mode: use broker positions
                if not self.operator:
                    return {"error": "operator_not_available_for_live_mode"}

                positions_data = self.operator.get_positions(include_closed=False)
                broker_positions = positions_data.get("positions", [])

                if not broker_positions:
                    logger.info("No open positions to monitor")
                    return {
                        "open_positions": 0,
                        "actions_taken": [],
                        "message": "no_open_positions"
                    }

                # Get tracked positions (with SL/target info)
                tracked_positions = {}
                if self.tracker:
                    for pos in self.tracker.get_open_positions():
                        symbol = pos.get("symbol")
                        if symbol:
                            tracked_positions[symbol] = pos

                actions_taken = []
                positions_checked = 0

                for broker_pos in broker_positions:
                    symbol = broker_pos.get("tradingsymbol") or broker_pos.get("symbol")
                    if not symbol:
                        continue

                    instrument_key = broker_pos.get("instrument_token") or broker_pos.get("instrument_key")
                    quantity = int(broker_pos.get("quantity", 0) or 0)
                    product = broker_pos.get("product", "I").upper()

                    if quantity == 0:
                        continue

                    positions_checked += 1

                    # Get current price
                    try:
                        current_price, _ = self.tech.ltp(instrument_key)
                        if current_price is None:
                            logger.warning(f"Could not get LTP for {symbol}")
                            continue
                        current_price = float(current_price)
                    except Exception as e:
                        logger.error(f"Error getting price for {symbol}: {e}")
                        continue

                    # Check if we have tracking info for this position
                    tracked = tracked_positions.get(symbol)
                    if not tracked:
                        logger.warning(f"Position {symbol} not found in tracker - cannot check SL/target")
                        continue

                    # Get SL and target from tracked position
                    stop_loss = tracked.get("stop_loss")
                    target = tracked.get("target")
                    side = tracked.get("side", "BUY").upper()
                    entry_price = tracked.get("entry_price", 0)

                    # Check for exit conditions
                    exit_reason = None
                    should_exit = False

                    # Check stop-loss
                    if stop_loss:
                        if side == "BUY" and current_price <= stop_loss:
                            exit_reason = "STOP_LOSS_HIT"
                            should_exit = True
                            logger.info(f"🛑 {symbol} STOP-LOSS HIT: {current_price:.2f} <= {stop_loss:.2f}")
                        elif side == "SELL" and current_price >= stop_loss:
                            exit_reason = "STOP_LOSS_HIT"
                            should_exit = True
                            logger.info(f"🛑 {symbol} STOP-LOSS HIT: {current_price:.2f} >= {stop_loss:.2f}")

                    # Check target
                    if not should_exit and target:
                        if side == "BUY" and current_price >= target:
                            exit_reason = "TARGET_HIT"
                            should_exit = True
                            logger.info(f"🎯 {symbol} TARGET HIT: {current_price:.2f} >= {target:.2f}")
                        elif side == "SELL" and current_price <= target:
                            exit_reason = "TARGET_HIT"
                            should_exit = True
                            logger.info(f"🎯 {symbol} TARGET HIT: {current_price:.2f} <= {target:.2f}")

                    # Check time-based exit for intraday (after 3:15 PM)
                    if not should_exit and product == "I":
                        now = datetime.now(IST)
                        square_off_time = now.replace(hour=15, minute=15, second=0, microsecond=0)
                        if now >= square_off_time:
                            exit_reason = "INTRADAY_TIME_BASED_EXIT"
                            should_exit = True
                            logger.info(f"⏰ {symbol} Intraday time-based exit (after 3:15 PM)")

                    # Execute exit if needed
                    if should_exit:
                        action_result = {
                            "symbol": symbol,
                            "exit_reason": exit_reason,
                            "current_price": current_price,
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "target": target,
                            "timestamp": datetime.now(IST).isoformat(),
                        }

                        if live:
                            # Place square-off order
                            try:
                                square_off_result = self.operator.square_off(
                                    symbol=symbol,
                                    instrument_key=instrument_key,
                                    live=True
                                )

                                action_result["square_off_result"] = square_off_result

                                if square_off_result.get("status") == "ok":
                                    logger.info(f"✅ {symbol} squared off successfully")

                                    # Record exit in tracker
                                    if self.tracker:
                                        try:
                                            pnl_record = self.tracker.record_exit(
                                                symbol=symbol,
                                                exit_price=current_price,
                                                exit_reason=exit_reason,
                                                order_id=square_off_result.get("order_id"),
                                            )
                                            action_result["pnl_record"] = pnl_record

                                            net_pnl = pnl_record.get("net_pnl", 0)
                                            pnl_pct = pnl_record.get("pnl_percent", 0)
                                            logger.info(f"💰 P&L: {symbol} → ₹{net_pnl:.2f} ({pnl_pct:+.2f}%)")

                                        except Exception as e:
                                            logger.error(f"Error recording exit for {symbol}: {e}")
                                else:
                                    logger.error(f"❌ Failed to square off {symbol}: {square_off_result.get('message')}")

                            except Exception as e:
                                logger.error(f"Error squaring off {symbol}: {e}")
                                action_result["error"] = str(e)
                        else:
                            action_result["action"] = "DRY_RUN (set live=True to execute)"

                        actions_taken.append(action_result)
                    else:
                        # Position is healthy, calculate unrealized P&L
                        if side == "BUY":
                            unrealized_pnl = (current_price - entry_price) * quantity
                        else:
                            unrealized_pnl = (entry_price - current_price) * quantity

                        logger.debug(
                            f"✓ {symbol}: {current_price:.2f} | "
                            f"SL: {stop_loss:.2f if stop_loss else 'N/A'} | "
                            f"Target: {target:.2f if target else 'N/A'} | "
                            f"Unrealized P&L: ₹{unrealized_pnl:.2f}"
                        )

                return {
                    "open_positions": positions_checked,
                    "actions_taken": actions_taken,
                    "exits_executed": len([a for a in actions_taken if a.get("square_off_result")]),
                    "timestamp": datetime.now(IST).isoformat(),
                }

        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return {"error": str(e)}

    def _check_time_exit(self) -> tuple[bool, Optional[str]]:
        """
        Check if it's time for intraday square-off (after 3:15 PM IST).

        Returns:
            (should_exit, exit_reason) tuple
        """
        now = datetime.now(IST)
        square_off_time = now.replace(hour=15, minute=15, second=0, microsecond=0)

        if now >= square_off_time:
            return True, "INTRADAY_TIME_BASED_EXIT"

        return False, None

    def monitor_loop(
        self,
        interval_seconds: int = 30,
        max_iterations: Optional[int] = None,
        live: bool = False
    ):
        """
        Continuously monitor positions in a loop.

        Args:
            interval_seconds: How often to check (default 30 seconds)
            max_iterations: Maximum iterations (None = infinite)
            live: If True, will actually execute exits

        This should be run in a separate thread/process during trading hours.
        """
        logger.info(f"Starting position monitor (interval={interval_seconds}s, live={live})")

        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            try:
                # Check if market is open
                if self.operator:
                    market_status = self.operator.market_session_status()
                    if not market_status.get("open"):
                        logger.info("Market closed, sleeping...")
                        time.sleep(60)  # Sleep 1 minute if market closed
                        continue

                # Check positions
                result = self.check_positions(live=live)

                if result.get("actions_taken"):
                    logger.info(f"Actions taken: {len(result['actions_taken'])}")
                    for action in result["actions_taken"]:
                        logger.info(f"  - {action['symbol']}: {action['exit_reason']}")

                iteration += 1
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Position monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(interval_seconds)

        logger.info(f"Position monitor stopped after {iteration} iterations")

    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get summary of all open positions with current status.

        Returns:
            Summary including unrealized P&L, position health, etc.
        """
        if not self.operator or not self.tech or not self.tracker:
            return {"error": "services_not_available"}

        try:
            # Get broker positions
            positions_data = self.operator.get_positions(include_closed=False)
            broker_positions = positions_data.get("positions", [])

            # Get tracked positions
            tracked = {p.get("symbol"): p for p in self.tracker.get_open_positions()}

            summary = {
                "total_positions": len(broker_positions),
                "intraday_count": 0,
                "swing_count": 0,
                "total_unrealized_pnl": 0.0,
                "positions": [],
                "timestamp": datetime.now(IST).isoformat(),
            }

            for broker_pos in broker_positions:
                symbol = broker_pos.get("tradingsymbol") or broker_pos.get("symbol")
                if not symbol:
                    continue

                instrument_key = broker_pos.get("instrument_token")
                quantity = int(broker_pos.get("quantity", 0) or 0)
                product = broker_pos.get("product", "I").upper()

                if quantity == 0:
                    continue

                # Get current price
                try:
                    current_price, _ = self.tech.ltp(instrument_key)
                    current_price = float(current_price) if current_price else 0
                except:
                    current_price = 0

                # Get tracking info
                track_info = tracked.get(symbol, {})
                entry_price = track_info.get("entry_price", 0)
                stop_loss = track_info.get("stop_loss")
                target = track_info.get("target")
                side = track_info.get("side", "BUY").upper()

                # Calculate unrealized P&L
                if entry_price and current_price:
                    if side == "BUY":
                        unrealized_pnl = (current_price - entry_price) * quantity
                    else:
                        unrealized_pnl = (entry_price - current_price) * quantity
                else:
                    unrealized_pnl = 0

                summary["total_unrealized_pnl"] += unrealized_pnl

                if product == "I":
                    summary["intraday_count"] += 1
                else:
                    summary["swing_count"] += 1

                # Calculate distance to SL/target
                sl_distance_pct = None
                target_distance_pct = None
                if current_price and entry_price:
                    if stop_loss:
                        sl_distance_pct = ((current_price - stop_loss) / entry_price) * 100
                    if target:
                        target_distance_pct = ((target - current_price) / entry_price) * 100

                summary["positions"].append({
                    "symbol": symbol,
                    "product": product,
                    "side": side,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "stop_loss": stop_loss,
                    "target": target,
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "unrealized_pnl_pct": round((unrealized_pnl / (entry_price * quantity)) * 100, 2) if entry_price else 0,
                    "sl_distance_pct": round(sl_distance_pct, 2) if sl_distance_pct else None,
                    "target_distance_pct": round(target_distance_pct, 2) if target_distance_pct else None,
                })

            summary["total_unrealized_pnl"] = round(summary["total_unrealized_pnl"], 2)

            return summary

        except Exception as e:
            logger.error(f"Error getting position summary: {e}")
            return {"error": str(e)}


# Global instance
_monitor_instance: Optional[PositionMonitor] = None


def get_position_monitor() -> PositionMonitor:
    """Get global position monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PositionMonitor()
    return _monitor_instance


# Convenience functions
def check_positions(live: bool = False) -> Dict[str, Any]:
    """Check all positions and take action if SL/target hit."""
    return get_position_monitor().check_positions(live=live)


def get_position_summary() -> Dict[str, Any]:
    """Get summary of all open positions."""
    return get_position_monitor().get_position_summary()


def start_monitor_loop(interval: int = 30, live: bool = False):
    """Start monitoring loop."""
    return get_position_monitor().monitor_loop(interval_seconds=interval, live=live)
