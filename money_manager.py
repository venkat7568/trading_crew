#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
money_manager.py — Capital allocation and risk management
==========================================================

Manages wallet, capital limits, and risk allocation.

Features:
- Track total capital and available balance
- Allocate capital to intraday vs swing trading
- Enforce position size limits
- Track capital usage and exposure
- Risk management (max loss per trade, max daily loss)
- Portfolio-level risk controls
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("money_manager")

IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


class MoneyManager:
    """
    Manages capital allocation and risk limits.

    This is CRITICAL - the system needs to know:
    - How much capital is available
    - What can be allocated to each trade
    - Position limits (max positions, max exposure)
    - Risk limits (max loss per trade, max daily loss)
    """

    def __init__(self, operator=None):
        """
        Initialize money manager.

        Args:
            operator: UpstoxOperator for fetching account balance
        """
        self.operator = operator

        if not self.operator:
            try:
                from upstox_operator import UpstoxOperator
                self.operator = UpstoxOperator()
            except Exception as e:
                logger.error(f"Failed to init operator: {e}")

        # Configuration file
        self.config_file = DATA_DIR / "money_management_config.json"
        self.state_file = DATA_DIR / "money_management_state.json"

        # Load config
        self.config = self._load_config()
        self.state = self._load_state()

    def _load_config(self) -> Dict[str, Any]:
        """Load money management configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        # Default configuration (DOES NOT include total_capital - always fetch from broker)
        return {
            # Position limits
            "max_positions": 5,  # Max open positions at once

            # Capital usage limits
            "max_capital_usage_pct": 95.0,  # Max % of capital to use (increased from 90%)
            "min_cash_reserve": 1000.0,  # Minimum cash to keep aside (reduced from 10000)

            # Risk limits per trade
            "max_risk_per_trade_pct": 1.0,  # Max 1% risk per trade
            "max_position_size_pct": 50.0,  # Max 50% of capital per position (increased from 20%)

            # Daily limits
            "max_daily_loss": 2000.0,  # Stop trading if daily loss > this
            "max_daily_trades": 10,  # Maximum trades per day
            "profit_target_daily": 3000.0,  # Daily profit target (optional)

            # Risk controls
            "use_trailing_stop": False,  # Enable trailing stop-loss
            "max_correlation_exposure": 50.0,  # Max exposure to correlated stocks

            # Emergency controls
            "circuit_breaker_loss_pct": 3.0,  # Stop all trading if loss > 3% of capital
            "emergency_square_off": False,  # Emergency mode - close all positions
        }

    def _save_config(self):
        """Save configuration."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _load_state(self) -> Dict[str, Any]:
        """Load current state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading state: {e}")

        # Default state
        return {
            "last_updated": None,
            "current_date": None,
            "daily_pnl": 0.0,
            "daily_trades": 0,
            "circuit_breaker_triggered": False,
        }

    def _save_state(self):
        """Save current state."""
        self.state["last_updated"] = datetime.now(IST).isoformat()
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _reset_daily_state_if_needed(self):
        """Reset daily counters if it's a new day."""
        today = datetime.now(IST).date().isoformat()
        if self.state.get("current_date") != today:
            logger.info(f"New trading day: {today}, resetting daily state")
            self.state["current_date"] = today
            self.state["daily_pnl"] = 0.0
            self.state["daily_trades"] = 0
            self.state["circuit_breaker_triggered"] = False
            self._save_state()

    def get_wallet_status(self) -> Dict[str, Any]:
        """
        Get complete wallet and capital status.

        Returns detailed information about available capital,
        allocations, and limits.
        """
        self._reset_daily_state_if_needed()

        # Get account balance from broker (REQUIRED - no fallback)
        available_capital = 0.0
        used_capital = 0.0

        if not self.operator:
            logger.error("No operator configured - cannot fetch funds!")
            raise RuntimeError("MoneyManager requires UpstoxOperator to fetch real funds")

        try:
            funds = self.operator.get_funds()
            equity = funds.get("equity", {})
            available_capital = float(equity.get("available_margin", 0) or 0)
            used_capital = float(equity.get("used_margin", 0) or 0)

            if available_capital <= 0:
                logger.error(f"Invalid available_capital from broker: {available_capital}")
                raise ValueError("Available capital must be > 0")

        except Exception as e:
            logger.error(f"Error fetching funds from broker: {e}")
            raise RuntimeError(f"Failed to fetch real funds from Upstox: {e}")

        total_capital = available_capital + used_capital

        if total_capital <= 0:
            logger.error(f"Invalid total_capital: {total_capital}")
            raise ValueError("Total capital must be > 0")

        # Calculate usage
        capital_usage_pct = (used_capital / total_capital * 100) if total_capital > 0 else 0
        max_usable_capital = (total_capital * self.config["max_capital_usage_pct"]) / 100

        # Check limits
        can_trade = True
        blocking_reasons = []

        if self.state.get("circuit_breaker_triggered"):
            can_trade = False
            blocking_reasons.append("circuit_breaker_triggered")

        if self.state.get("daily_pnl", 0) <= -self.config["max_daily_loss"]:
            can_trade = False
            blocking_reasons.append("max_daily_loss_reached")

        if self.state.get("daily_trades", 0) >= self.config["max_daily_trades"]:
            can_trade = False
            blocking_reasons.append("max_daily_trades_reached")

        if available_capital < self.config["min_cash_reserve"]:
            can_trade = False
            blocking_reasons.append("below_minimum_cash_reserve")

        if capital_usage_pct >= self.config["max_capital_usage_pct"]:
            can_trade = False
            blocking_reasons.append("max_capital_usage_exceeded")

        return {
            # Capital
            "total_capital": round(total_capital, 2),
            "available_capital": round(available_capital, 2),
            "used_capital": round(used_capital, 2),
            "capital_usage_pct": round(capital_usage_pct, 2),
            "max_usable_capital": round(max_usable_capital, 2),
            "remaining_usable_capital": round(max_usable_capital - used_capital, 2),

            # Limits
            "max_positions": self.config["max_positions"],
            "min_cash_reserve": self.config["min_cash_reserve"],

            # Daily state
            "daily_pnl": round(self.state.get("daily_pnl", 0), 2),
            "daily_trades": self.state.get("daily_trades", 0),
            "max_daily_loss": self.config["max_daily_loss"],
            "max_daily_trades": self.config["max_daily_trades"],

            # Trading status
            "can_trade": can_trade,
            "blocking_reasons": blocking_reasons,
            "circuit_breaker_triggered": self.state.get("circuit_breaker_triggered", False),

            "timestamp": datetime.now(IST).isoformat(),
        }

    def can_open_position(
        self,
        product: str,
        position_value: float,
        risk_amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Check if a new position can be opened given limits.

        Args:
            product: "I" for intraday, "D" for swing
            position_value: Total value of position (price * qty)
            risk_amount: Amount at risk (optional, for risk check)

        Returns:
            Dict with can_open (bool) and reasons
        """
        self._reset_daily_state_if_needed()

        wallet = self.get_wallet_status()

        can_open = True
        reasons = []

        # Check if trading is allowed
        if not wallet["can_trade"]:
            can_open = False
            reasons.extend(wallet["blocking_reasons"])

        # Check capital availability
        available = wallet["remaining_usable_capital"]
        if position_value > available:
            can_open = False
            reasons.append(f"insufficient_capital (need ₹{position_value:.2f}, have ₹{available:.2f})")

        # Check position size limit
        max_position_value = (wallet["total_capital"] * self.config["max_position_size_pct"]) / 100
        if position_value > max_position_value:
            can_open = False
            reasons.append(f"position_too_large (₹{position_value:.2f} > ₹{max_position_value:.2f} max)")

        # Check risk limit
        if risk_amount:
            max_risk = (wallet["total_capital"] * self.config["max_risk_per_trade_pct"]) / 100
            if risk_amount > max_risk:
                can_open = False
                reasons.append(f"risk_too_high (₹{risk_amount:.2f} > ₹{max_risk:.2f} max)")

        return {
            "can_open": can_open,
            "reasons": reasons if not can_open else ["all_checks_passed"],
            "position_value": position_value,
            "available_capital": available,
            "max_position_value": max_position_value,
        }

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        product: str = "I",
        risk_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on risk.

        Uses the formula:
        Position Size = Risk Amount / (Entry Price - Stop Loss)

        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            product: "I" for intraday, "D" for swing
            risk_pct: Custom risk % (defaults to config)

        Returns:
            Recommended position size and details
        """
        wallet = self.get_wallet_status()
        total_capital = wallet["total_capital"]

        # Determine risk amount
        if risk_pct is None:
            risk_pct = self.config["max_risk_per_trade_pct"]

        risk_amount = (total_capital * risk_pct) / 100

        # Calculate position size based on risk
        price_diff = abs(entry_price - stop_loss)
        if price_diff == 0:
            return {
                "error": "entry_price_equals_stop_loss",
                "message": "Entry price and stop loss cannot be the same"
            }

        quantity = int(risk_amount / price_diff)
        position_value = quantity * entry_price

        # Check against limits
        max_position_value = (total_capital * self.config["max_position_size_pct"]) / 100
        if position_value > max_position_value:
            # Reduce quantity to fit limit
            quantity = int(max_position_value / entry_price)
            position_value = quantity * entry_price
            risk_amount = price_diff * quantity

        # Check if position can be opened
        can_open_check = self.can_open_position(product, position_value, risk_amount)

        return {
            "recommended_quantity": quantity,
            "position_value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_pct": round((risk_amount / total_capital) * 100, 2),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "price_diff": round(price_diff, 2),
            "can_open": can_open_check["can_open"],
            "blocking_reasons": can_open_check["reasons"],
        }

    def record_trade_result(self, net_pnl: float, product: str = "I"):
        """
        Record the result of a closed trade.

        Updates daily P&L and checks circuit breakers.

        Args:
            net_pnl: Net profit/loss from the trade
            product: "I" for intraday, "D" for swing
        """
        self._reset_daily_state_if_needed()

        # Update daily P&L
        self.state["daily_pnl"] = self.state.get("daily_pnl", 0) + net_pnl
        self.state["daily_trades"] = self.state.get("daily_trades", 0) + 1

        # Check circuit breaker
        wallet = self.get_wallet_status()
        total_capital = wallet["total_capital"]
        loss_pct = abs(self.state["daily_pnl"] / total_capital * 100) if total_capital > 0 else 0

        if self.state["daily_pnl"] < 0 and loss_pct >= self.config["circuit_breaker_loss_pct"]:
            logger.warning(f"🚨 CIRCUIT BREAKER TRIGGERED! Loss: ₹{self.state['daily_pnl']:.2f} ({loss_pct:.2f}%)")
            self.state["circuit_breaker_triggered"] = True

        self._save_state()

        logger.info(f"Trade recorded: P&L ₹{net_pnl:+.2f} | Daily P&L: ₹{self.state['daily_pnl']:+.2f} | Trades: {self.state['daily_trades']}")

    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration parameters.

        Args:
            updates: Dict of config keys to update
        """
        for key, value in updates.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")

        self._save_config()

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker (use with caution)."""
        logger.info("Circuit breaker manually reset")
        self.state["circuit_breaker_triggered"] = False
        self._save_state()

    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get risk management summary.

        Shows current risk exposure, limits, and recommendations.
        """
        wallet = self.get_wallet_status()

        # Calculate risk metrics
        total_capital = wallet["total_capital"]
        daily_pnl = wallet["daily_pnl"]
        daily_loss_pct = (abs(daily_pnl) / total_capital * 100) if (daily_pnl < 0 and total_capital > 0) else 0

        return {
            "total_capital": wallet["total_capital"],
            "capital_at_risk": wallet["used_capital"],
            "capital_at_risk_pct": wallet["capital_usage_pct"],

            "daily_pnl": daily_pnl,
            "daily_loss_pct": round(daily_loss_pct, 2),
            "daily_trades": wallet["daily_trades"],

            "max_risk_per_trade": round((total_capital * self.config["max_risk_per_trade_pct"]) / 100, 2),
            "max_position_size": round((total_capital * self.config["max_position_size_pct"]) / 100, 2),
            "max_daily_loss": self.config["max_daily_loss"],

            "remaining_daily_loss_buffer": round(self.config["max_daily_loss"] - abs(daily_pnl), 2) if daily_pnl < 0 else self.config["max_daily_loss"],
            "remaining_daily_trades": max(0, self.config["max_daily_trades"] - wallet["daily_trades"]),

            "circuit_breaker": {
                "triggered": wallet["circuit_breaker_triggered"],
                "threshold_pct": self.config["circuit_breaker_loss_pct"],
                "current_loss_pct": round(daily_loss_pct, 2),
            },

            "can_trade": wallet["can_trade"],
            "blocking_reasons": wallet["blocking_reasons"],

            "timestamp": datetime.now(IST).isoformat(),
        }


# Global instance
_money_manager_instance: Optional[MoneyManager] = None


def get_money_manager() -> MoneyManager:
    """Get global money manager instance."""
    global _money_manager_instance
    if _money_manager_instance is None:
        _money_manager_instance = MoneyManager()
    return _money_manager_instance


# Convenience functions
def get_wallet_status() -> Dict[str, Any]:
    """Get wallet status."""
    return get_money_manager().get_wallet_status()


def can_open_position(product: str, position_value: float, risk_amount: Optional[float] = None) -> Dict[str, Any]:
    """Check if position can be opened."""
    return get_money_manager().can_open_position(product, position_value, risk_amount)


def calculate_position_size(entry_price: float, stop_loss: float, product: str = "I") -> Dict[str, Any]:
    """Calculate position size based on risk."""
    return get_money_manager().calculate_position_size(entry_price, stop_loss, product)


def record_trade_result(net_pnl: float, product: str = "I"):
    """Record trade result."""
    return get_money_manager().record_trade_result(net_pnl, product)
