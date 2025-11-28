#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learning_engine.py — Learn from trades and improve strategies
==============================================================

Analyzes trade history to identify patterns and improve decision-making.

This is CRITICAL for the system to get better every day:
- Analyze winning vs losing trades
- Identify what patterns work
- Adjust strategy parameters
- Learn from mistakes
- Improve over time

The system MUST learn from:
- Which stocks are profitable
- Which strategies work best
- What market conditions favor which approaches
- Common mistakes to avoid
- Optimal entry/exit timing
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger("learning_engine")

IST = ZoneInfo("Asia/Kolkata")

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


class LearningEngine:
    """
    Analyzes trade history and improves trading strategies.

    Features:
    - Analyze win/loss patterns
    - Identify profitable vs unprofitable setups
    - Adjust parameters based on results
    - Track strategy performance
    - Generate insights and recommendations
    """

    def __init__(self, trade_tracker=None):
        """
        Initialize learning engine.

        Args:
            trade_tracker: TradeTracker instance for accessing trade history
        """
        self.tracker = trade_tracker

        if not self.tracker:
            try:
                from trade_tracker import get_trade_tracker
                self.tracker = get_trade_tracker()
            except Exception as e:
                logger.error(f"Failed to init trade tracker: {e}")

        # Learning state file
        self.learning_file = DATA_DIR / "learning_state.json"
        self.insights_file = DATA_DIR / "trading_insights.jsonl"

        # Load learning state
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load learning state."""
        if self.learning_file.exists():
            try:
                with open(self.learning_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading learning state: {e}")

        # Default learning state
        return {
            "last_analysis": None,
            "total_trades_analyzed": 0,

            # Performance tracking
            "best_performing_symbols": {},  # symbol -> win_rate
            "worst_performing_symbols": {},  # symbol -> win_rate

            # Strategy adjustments
            "confidence_threshold": 0.60,  # Minimum confidence for trades
            "preferred_holding_time": "medium",  # short/medium/long
            "market_condition_bias": "neutral",  # bullish/bearish/neutral

            # Learned patterns
            "winning_patterns": [],
            "losing_patterns": [],

            # Parameter adjustments
            "recommended_adjustments": {},
        }

    def _save_state(self):
        """Save learning state."""
        self.state["last_analysis"] = datetime.now(IST).isoformat()
        try:
            with open(self.learning_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")

    def _save_insight(self, insight: Dict[str, Any]):
        """Save an insight for future reference."""
        insight["timestamp"] = datetime.now(IST).isoformat()
        try:
            with open(self.insights_file, "a") as f:
                f.write(json.dumps(insight, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error saving insight: {e}")

    def analyze_trade_history(
        self,
        days: int = 30,
        min_trades: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze recent trade history and extract insights.

        Args:
            days: Number of days to analyze
            min_trades: Minimum trades needed for analysis

        Returns:
            Analysis results with insights and recommendations
        """
        if not self.tracker:
            return {"error": "trade_tracker_not_available"}

        try:
            # Load trade history
            trades = self._load_recent_trades(days)

            if len(trades) < min_trades:
                return {
                    "status": "insufficient_data",
                    "trades_found": len(trades),
                    "min_required": min_trades,
                    "message": f"Need at least {min_trades} trades for meaningful analysis"
                }

            logger.info(f"Analyzing {len(trades)} trades from last {days} days...")

            # Separate winners and losers
            winners = [t for t in trades if t.get("net_pnl", 0) > 0]
            losers = [t for t in trades if t.get("net_pnl", 0) < 0]
            breakeven = [t for t in trades if t.get("net_pnl", 0) == 0]

            # Calculate metrics
            total_pnl = sum(t.get("net_pnl", 0) for t in trades)
            win_rate = (len(winners) / len(trades)) * 100 if trades else 0

            avg_win = sum(t.get("net_pnl", 0) for t in winners) / len(winners) if winners else 0
            avg_loss = sum(t.get("net_pnl", 0) for t in losers) / len(losers) if losers else 0

            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

            # Analyze by symbol
            symbol_analysis = self._analyze_by_symbol(trades)

            # Analyze by strategy/confidence
            strategy_analysis = self._analyze_by_strategy(trades)

            # Analyze by market conditions
            market_condition_analysis = self._analyze_by_market_condition(trades)

            # Analyze timing (entry/exit quality)
            timing_analysis = self._analyze_timing(trades)

            # Identify patterns
            winning_patterns = self._identify_patterns(winners, "winning")
            losing_patterns = self._identify_patterns(losers, "losing")

            # Generate recommendations
            recommendations = self._generate_recommendations(
                trades, winners, losers, symbol_analysis, strategy_analysis
            )

            # Update learning state
            self.state["total_trades_analyzed"] = len(trades)
            self.state["best_performing_symbols"] = symbol_analysis.get("best_symbols", {})
            self.state["worst_performing_symbols"] = symbol_analysis.get("worst_symbols", {})
            self.state["winning_patterns"] = winning_patterns
            self.state["losing_patterns"] = losing_patterns
            self.state["recommended_adjustments"] = recommendations.get("adjustments", {})

            # Adjust confidence threshold based on results
            if win_rate < 50:
                # Low win rate - increase confidence threshold
                self.state["confidence_threshold"] = min(0.75, self.state.get("confidence_threshold", 0.60) + 0.05)
            elif win_rate > 70:
                # High win rate - can be slightly more aggressive
                self.state["confidence_threshold"] = max(0.55, self.state.get("confidence_threshold", 0.60) - 0.02)

            self._save_state()

            analysis = {
                "period_days": days,
                "total_trades": len(trades),

                # Overall performance
                "winners": len(winners),
                "losers": len(losers),
                "breakeven": len(breakeven),
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),

                # Average metrics
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "avg_pnl_per_trade": round(total_pnl / len(trades), 2),

                # Detailed analysis
                "symbol_analysis": symbol_analysis,
                "strategy_analysis": strategy_analysis,
                "market_condition_analysis": market_condition_analysis,
                "timing_analysis": timing_analysis,

                # Patterns
                "winning_patterns": winning_patterns,
                "losing_patterns": losing_patterns,

                # Recommendations
                "recommendations": recommendations,

                "timestamp": datetime.now(IST).isoformat(),
            }

            # Save this as an insight
            self._save_insight({
                "type": "trade_history_analysis",
                "summary": {
                    "trades": len(trades),
                    "win_rate": round(win_rate, 2),
                    "total_pnl": round(total_pnl, 2),
                    "profit_factor": round(profit_factor, 2),
                },
                "recommendations_count": len(recommendations.get("recommendations", [])),
            })

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing trade history: {e}")
            return {"error": str(e)}

    def _load_recent_trades(self, days: int) -> List[Dict[str, Any]]:
        """Load trades from recent days."""
        trades = []

        if not self.tracker or not self.tracker.trades_history_file.exists():
            return trades

        cutoff_date = datetime.now(IST) - timedelta(days=days)

        try:
            with open(self.tracker.trades_history_file, "r") as f:
                for line in f:
                    try:
                        trade = json.loads(line.strip())
                        # Check if trade is within date range
                        trade_time = datetime.fromisoformat(trade.get("entry_time", ""))
                        if trade_time >= cutoff_date:
                            trades.append(trade)
                    except:
                        continue
        except Exception as e:
            logger.error(f"Error loading trades: {e}")

        return trades

    def _analyze_by_symbol(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by symbol."""
        symbol_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": 0})

        for trade in trades:
            symbol = trade.get("symbol", "UNKNOWN")
            pnl = trade.get("net_pnl", 0)

            symbol_stats[symbol]["trades"] += 1
            symbol_stats[symbol]["total_pnl"] += pnl

            if pnl > 0:
                symbol_stats[symbol]["wins"] += 1
            elif pnl < 0:
                symbol_stats[symbol]["losses"] += 1

        # Calculate win rates
        symbol_performance = {}
        for symbol, stats in symbol_stats.items():
            win_rate = (stats["wins"] / stats["trades"]) * 100 if stats["trades"] > 0 else 0
            symbol_performance[symbol] = {
                "win_rate": round(win_rate, 2),
                "total_pnl": round(stats["total_pnl"], 2),
                "trades": stats["trades"],
                "wins": stats["wins"],
                "losses": stats["losses"],
            }

        # Sort by performance
        sorted_by_pnl = sorted(symbol_performance.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
        sorted_by_win_rate = sorted(symbol_performance.items(), key=lambda x: x[1]["win_rate"], reverse=True)

        best_symbols = dict(sorted_by_pnl[:5]) if len(sorted_by_pnl) >= 5 else dict(sorted_by_pnl)
        worst_symbols = dict(sorted_by_pnl[-5:]) if len(sorted_by_pnl) >= 5 else {}

        return {
            "all_symbols": symbol_performance,
            "best_symbols": best_symbols,
            "worst_symbols": worst_symbols,
            "most_traded": max(symbol_stats.items(), key=lambda x: x[1]["trades"])[0] if symbol_stats else None,
        }

    def _analyze_by_strategy(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by strategy/confidence level."""
        # Group by confidence ranges
        confidence_bins = {
            "very_high": {"range": (0.75, 1.0), "trades": [], "pnl": 0},
            "high": {"range": (0.65, 0.75), "trades": [], "pnl": 0},
            "medium": {"range": (0.55, 0.65), "trades": [], "pnl": 0},
            "low": {"range": (0.0, 0.55), "trades": [], "pnl": 0},
        }

        for trade in trades:
            confidence = trade.get("confidence", 0.5)
            pnl = trade.get("net_pnl", 0)

            for bin_name, bin_data in confidence_bins.items():
                min_conf, max_conf = bin_data["range"]
                if min_conf <= confidence < max_conf:
                    bin_data["trades"].append(trade)
                    bin_data["pnl"] += pnl
                    break

        # Calculate metrics for each bin
        confidence_performance = {}
        for bin_name, bin_data in confidence_bins.items():
            trade_count = len(bin_data["trades"])
            if trade_count > 0:
                winners = sum(1 for t in bin_data["trades"] if t.get("net_pnl", 0) > 0)
                win_rate = (winners / trade_count) * 100
                avg_pnl = bin_data["pnl"] / trade_count

                confidence_performance[bin_name] = {
                    "trades": trade_count,
                    "win_rate": round(win_rate, 2),
                    "total_pnl": round(bin_data["pnl"], 2),
                    "avg_pnl": round(avg_pnl, 2),
                    "confidence_range": bin_data["range"],
                }

        return confidence_performance

    def _analyze_by_market_condition(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by market conditions (if available in trade metadata)."""
        # This would analyze trades based on market context at the time
        # For now, return placeholder
        return {
            "note": "Market condition analysis requires historical market context data",
            "future_enhancement": True,
        }

    def _analyze_timing(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entry/exit timing quality."""
        # Analyze if SL/target hits vs premature exits
        sl_hits = sum(1 for t in trades if t.get("sl_hit", False))
        target_hits = sum(1 for t in trades if t.get("target_hit", False))

        # Analyze holding duration
        durations = [t.get("duration_minutes", 0) for t in trades if t.get("duration_minutes")]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Winners vs losers duration
        winner_durations = [t.get("duration_minutes", 0) for t in trades if t.get("net_pnl", 0) > 0 and t.get("duration_minutes")]
        loser_durations = [t.get("duration_minutes", 0) for t in trades if t.get("net_pnl", 0) < 0 and t.get("duration_minutes")]

        avg_winner_duration = sum(winner_durations) / len(winner_durations) if winner_durations else 0
        avg_loser_duration = sum(loser_durations) / len(loser_durations) if loser_durations else 0

        return {
            "stop_loss_hits": sl_hits,
            "target_hits": target_hits,
            "sl_vs_target_ratio": round(sl_hits / max(target_hits, 1), 2),

            "avg_holding_time_minutes": round(avg_duration, 2),
            "avg_winner_hold_time": round(avg_winner_duration, 2),
            "avg_loser_hold_time": round(avg_loser_duration, 2),

            "insight": (
                "Winners held longer - trend following works"
                if avg_winner_duration > avg_loser_duration
                else "Losers held longer - need faster exits"
            ),
        }

    def _identify_patterns(self, trades: List[Dict[str, Any]], pattern_type: str) -> List[str]:
        """Identify common patterns in winning or losing trades."""
        patterns = []

        if not trades:
            return patterns

        # Analyze R-multiples
        r_multiples = [t.get("r_multiple", 0) for t in trades if t.get("r_multiple")]
        if r_multiples:
            avg_r = sum(r_multiples) / len(r_multiples)
            if pattern_type == "winning" and avg_r > 2:
                patterns.append(f"Strong R-multiples (avg: {avg_r:.2f}R) - good risk/reward setups")
            elif pattern_type == "losing" and avg_r < -1:
                patterns.append(f"Negative R-multiples (avg: {avg_r:.2f}R) - SL too tight or entries poor")

        # Analyze product type
        intraday_count = sum(1 for t in trades if t.get("product") == "I")
        swing_count = sum(1 for t in trades if t.get("product") == "D")

        if intraday_count > swing_count * 1.5:
            patterns.append(f"Mostly intraday trades ({intraday_count}/{len(trades)}) - {pattern_type}")
        elif swing_count > intraday_count * 1.5:
            patterns.append(f"Mostly swing trades ({swing_count}/{len(trades)}) - {pattern_type}")

        # Analyze common exit reasons
        if pattern_type == "winning":
            target_hits = sum(1 for t in trades if t.get("target_hit", False))
            if target_hits / len(trades) > 0.6:
                patterns.append(f"Targets frequently hit ({target_hits}/{len(trades)}) - good target placement")

        return patterns

    def _generate_recommendations(
        self,
        all_trades: List[Dict[str, Any]],
        winners: List[Dict[str, Any]],
        losers: List[Dict[str, Any]],
        symbol_analysis: Dict[str, Any],
        strategy_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        adjustments = {}

        win_rate = (len(winners) / len(all_trades)) * 100 if all_trades else 0

        # Recommendation 1: Confidence threshold
        if win_rate < 50:
            recommendations.append({
                "priority": "HIGH",
                "category": "confidence",
                "recommendation": "Increase confidence threshold",
                "reason": f"Win rate is only {win_rate:.1f}%. Being more selective could improve results.",
                "action": "Raise minimum confidence from 0.60 to 0.70"
            })
            adjustments["confidence_threshold"] = 0.70
        elif win_rate > 70:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "confidence",
                "recommendation": "Can slightly lower confidence threshold",
                "reason": f"Win rate is {win_rate:.1f}%. System is being overly cautious.",
                "action": "Lower minimum confidence to 0.55 for more opportunities"
            })
            adjustments["confidence_threshold"] = 0.55

        # Recommendation 2: Symbol blacklist/whitelist
        worst_symbols = symbol_analysis.get("worst_symbols", {})
        if worst_symbols:
            worst_symbol = min(worst_symbols.items(), key=lambda x: x[1]["total_pnl"])
            if worst_symbol[1]["trades"] >= 3 and worst_symbol[1]["total_pnl"] < -500:
                recommendations.append({
                    "priority": "HIGH",
                    "category": "blacklist",
                    "recommendation": f"Avoid {worst_symbol[0]}",
                    "reason": f"{worst_symbol[0]} has lost ₹{abs(worst_symbol[1]['total_pnl']):.2f} over {worst_symbol[1]['trades']} trades",
                    "action": f"Add {worst_symbol[0]} to blacklist"
                })
                adjustments["blacklist_add"] = [worst_symbol[0]]

        # Recommendation 3: Strategy adjustments
        best_strategy = max(strategy_analysis.items(), key=lambda x: x[1].get("win_rate", 0)) if strategy_analysis else None
        if best_strategy and best_strategy[1]["win_rate"] > 65:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "strategy",
                "recommendation": f"Focus on {best_strategy[0]} confidence trades",
                "reason": f"{best_strategy[0]} confidence has {best_strategy[1]['win_rate']:.1f}% win rate",
                "action": "Prioritize trades in this confidence range"
            })

        # Recommendation 4: Risk management
        avg_loss = sum(t.get("net_pnl", 0) for t in losers) / len(losers) if losers else 0
        avg_win = sum(t.get("net_pnl", 0) for t in winners) / len(winners) if winners else 0

        if abs(avg_loss) > avg_win:
            recommendations.append({
                "priority": "HIGH",
                "category": "risk",
                "recommendation": "Tighten stop-losses",
                "reason": f"Average loss (₹{abs(avg_loss):.2f}) > average win (₹{avg_win:.2f})",
                "action": "Use tighter stop-losses or smaller position sizes"
            })

        return {
            "recommendations": recommendations,
            "adjustments": adjustments,
            "summary": f"Generated {len(recommendations)} recommendations based on {len(all_trades)} trades"
        }

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get current learning state summary."""
        return {
            "last_analysis": self.state.get("last_analysis"),
            "total_trades_analyzed": self.state.get("total_trades_analyzed", 0),
            "confidence_threshold": self.state.get("confidence_threshold", 0.60),
            "best_performing_symbols": self.state.get("best_performing_symbols", {}),
            "worst_performing_symbols": self.state.get("worst_performing_symbols", {}),
            "recommended_adjustments": self.state.get("recommended_adjustments", {}),
            "patterns_learned": {
                "winning": len(self.state.get("winning_patterns", [])),
                "losing": len(self.state.get("losing_patterns", [])),
            },
        }


# Global instance
_learning_engine_instance: Optional[LearningEngine] = None


def get_learning_engine() -> LearningEngine:
    """Get global learning engine instance."""
    global _learning_engine_instance
    if _learning_engine_instance is None:
        _learning_engine_instance = LearningEngine()
    return _learning_engine_instance


# Convenience functions
def analyze_trades(days: int = 30) -> Dict[str, Any]:
    """Analyze recent trades."""
    return get_learning_engine().analyze_trade_history(days=days)


def get_learning_summary() -> Dict[str, Any]:
    """Get learning summary."""
    return get_learning_engine().get_learning_summary()
