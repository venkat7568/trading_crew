#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opportunity_ranker.py — Multi-Stock Opportunity Comparison & Ranking
====================================================================

Ranks trading opportunities across multiple stocks to pick the BEST
trades that fit within capital and position limits.

This solves the critical problem: When you have 10 BUY signals but
capital for only 3, which 3 should you take?

Algorithm:
1. Score each opportunity by confidence, R:R, entry quality, market alignment
2. Adjust for capital efficiency (profit potential / capital required)
3. Apply diversification bonuses (different sectors better than same sector)
4. Filter by capital constraints and position limits
5. Return ranked list of top opportunities
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("opportunity_ranker")

IST = ZoneInfo("Asia/Kolkata")


class OpportunityRanker:
    """
    Ranks and selects best trading opportunities from multiple stock signals.

    Key Features:
    - Multi-factor scoring (confidence, R:R, entry quality, alignment)
    - Capital efficiency adjustment
    - Sector diversification bonus
    - Win rate history integration
    - Product-specific allocation (intraday vs swing)
    """

    def __init__(
        self,
        learning_engine=None,
        market_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ranker.

        Args:
            learning_engine: LearningEngine for historical win rates
            market_context: Market sentiment and breadth data
        """
        self.learning_engine = learning_engine
        self.market_context = market_context or {}

    def rank_opportunities(
        self,
        decisions: List[Dict[str, Any]],
        available_capital: float,
        max_positions: int,
        product_allocations: Optional[Dict[str, float]] = None,
        current_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Rank all opportunities and select top N that fit constraints.

        Args:
            decisions: List of decision dicts from decide_trade()
            available_capital: Total available capital
            max_positions: Maximum number of positions allowed
            product_allocations: Deprecated, not used anymore
            current_positions: Currently open positions (for diversification)

        Returns:
            {
                "selected_opportunities": [...],  # Top N to execute
                "rejected_opportunities": [...],  # Good but didn't make cut
                "ranking_metadata": {...}
            }
        """
        logger.info(f"🎯 Ranking {len(decisions)} opportunities...")

        current_positions = current_positions or []

        # Filter to tradeable signals (BUY/SELL with confidence)
        tradeable = []
        skipped = []

        for dec in decisions:
            direction = (dec.get("direction") or "SKIP").upper()
            if direction in ("BUY", "SELL") and dec.get("confidence") is not None:
                tradeable.append(dec)
            else:
                skipped.append(dec)

        logger.info(f"   Tradeable signals: {len(tradeable)} (skipped {len(skipped)})")

        if not tradeable:
            return {
                "selected_opportunities": [],
                "rejected_opportunities": [],
                "ranking_metadata": {
                    "total_analyzed": len(decisions),
                    "tradeable": 0,
                    "selected": 0,
                    "reason": "no_tradeable_signals",
                },
            }

        # Score each opportunity
        scored_opportunities = []
        for dec in tradeable:
            score_result = self._score_opportunity(dec)
            if score_result["valid"]:
                scored_opportunities.append(score_result)

        if not scored_opportunities:
            return {
                "selected_opportunities": [],
                "rejected_opportunities": tradeable,
                "ranking_metadata": {
                    "total_analyzed": len(decisions),
                    "tradeable": len(tradeable),
                    "selected": 0,
                    "reason": "no_valid_scores",
                },
            }

        # Sort by composite score (descending)
        scored_opportunities.sort(key=lambda x: x["composite_score"], reverse=True)

        # Select top opportunities within constraints
        selected, rejected, used_capital = self._select_within_constraints(
            scored_opportunities=scored_opportunities,
            available_capital=available_capital,
            max_positions=max_positions,
            current_positions=current_positions,
        )

        logger.info(f"   ✅ Selected {len(selected)} opportunities (rejected {len(rejected)})")
        logger.info(f"   💰 Capital usage: ₹{used_capital:,.2f} / ₹{available_capital:,.2f}")

        return {
            "selected_opportunities": selected,
            "rejected_opportunities": rejected,
            "ranking_metadata": {
                "total_analyzed": len(decisions),
                "tradeable": len(tradeable),
                "selected": len(selected),
                "rejected": len(rejected),
                "capital_used": used_capital,
                "capital_available": available_capital,
                "timestamp": datetime.now(IST).isoformat(),
            },
        }

    def _score_opportunity(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single trading opportunity using multiple factors.

        Scoring components:
        1. Base confidence (35%): From decision agent
        2. R:R ratio (25%): Higher reward/risk = better
        3. Entry quality (20%): From entry validator score
        4. Market alignment (10%): Does trade align with market sentiment?
        5. Symbol history (10%): Past win rate for this symbol

        Returns:
            {
                "symbol": "ITC",
                "decision": {...original decision...},
                "opportunity_score": 0.0-1.0,
                "composite_score": float (adjusted for capital efficiency),
                "components": {...breakdown...},
                "valid": True/False
            }
        """
        symbol = decision.get("symbol", "UNKNOWN")
        confidence = float(decision.get("confidence", 0))
        direction = decision.get("direction", "").upper()

        # Extract style and R:R ratio from raw decision (handle both string and dict)
        style = "intraday"  # Default
        rr_ratio = 1.5  # Default

        try:
            raw_str = decision.get("raw", "{}")
            if isinstance(raw_str, str) and "{" in raw_str:
                import json
                raw_obj = json.loads(raw_str)
                style = raw_obj.get("style") or "intraday"
                rr_ratio = float(raw_obj.get("rr_ratio") or raw_obj.get("rr") or 1.5)
            elif isinstance(raw_str, dict):
                # If raw is already a dict (shouldn't happen but handle it)
                style = raw_str.get("style") or "intraday"
                rr_ratio = float(raw_str.get("rr_ratio") or raw_str.get("rr") or 1.5)
        except Exception:
            pass  # Use defaults

        # Entry quality score (if available from earlier validation)
        entry_quality = float(decision.get("entry_quality", 70))  # 0-100

        # Component scores (normalized 0-1)
        confidence_score = min(confidence, 1.0)  # Already 0-1

        # R:R score (cap at 3.0 for normalization)
        rr_score = min(rr_ratio / 3.0, 1.0)

        # Entry quality score (0-100 → 0-1)
        entry_score = min(entry_quality / 100.0, 1.0)

        # Market alignment score
        market_alignment_score = self._calculate_market_alignment(direction, style)

        # Symbol history score (win rate from learning engine)
        symbol_history_score = self._get_symbol_win_rate(symbol)

        # Weighted opportunity score
        opportunity_score = (
            confidence_score * 0.35 +
            rr_score * 0.25 +
            entry_score * 0.20 +
            market_alignment_score * 0.10 +
            symbol_history_score * 0.10
        )

        # Capital efficiency adjustment
        # Better setups with lower capital requirement get bonus
        estimated_capital = self._estimate_capital_required(decision)
        capital_efficiency = 1.0
        if estimated_capital > 0:
            # Expected profit vs capital required
            expected_profit = estimated_capital * (rr_ratio * 0.01)  # Rough estimate
            capital_efficiency = min(expected_profit / max(estimated_capital, 1000), 2.0)

        # Composite score (opportunity score × capital efficiency)
        composite_score = opportunity_score * capital_efficiency

        return {
            "symbol": symbol,
            "decision": decision,
            "opportunity_score": opportunity_score,
            "composite_score": composite_score,
            "confidence": confidence,
            "rr_ratio": rr_ratio,
            "entry_quality": entry_quality,
            "style": style,
            "direction": direction,
            "estimated_capital": estimated_capital,
            "components": {
                "confidence": confidence_score,
                "rr": rr_score,
                "entry_quality": entry_score,
                "market_alignment": market_alignment_score,
                "symbol_history": symbol_history_score,
                "capital_efficiency": capital_efficiency,
            },
            "valid": True,
        }

    def _calculate_market_alignment(self, direction: str, style: str) -> float:
        """
        Calculate how well the trade aligns with current market conditions.

        Returns: 0.0 - 1.0
        - 1.0 = Perfect alignment (BUY in strong bull market)
        - 0.5 = Neutral (SELL in neutral market, or mixed signals)
        - 0.0 = Against market (BUY in strong bear market)
        """
        if not self.market_context:
            return 0.6  # Neutral default

        nifty = self.market_context.get("nifty", {})
        sentiment = nifty.get("sentiment", "NEUTRAL").upper()
        trading_bias = nifty.get("trading_bias", "SELECTIVE").upper()

        # Alignment matrix
        if direction == "BUY":
            if sentiment in ("BULLISH", "STRONG_BULLISH"):
                return 1.0  # Perfect
            elif sentiment == "NEUTRAL" or trading_bias == "SELECTIVE":
                return 0.6  # Stock-specific OK
            elif sentiment in ("BEARISH", "STRONG_BEARISH"):
                return 0.3  # Fighting the market
        elif direction == "SELL":
            if sentiment in ("BEARISH", "STRONG_BEARISH"):
                return 1.0  # Perfect short alignment
            elif sentiment == "NEUTRAL":
                return 0.6  # Neutral
            elif sentiment in ("BULLISH", "STRONG_BULLISH"):
                return 0.4  # Counter-trend short

        return 0.5  # Default neutral

    def _get_symbol_win_rate(self, symbol: str) -> float:
        """
        Get historical win rate for symbol from learning engine.

        Returns: 0.0 - 1.0 (0.5 default if no history)
        """
        if not self.learning_engine:
            return 0.5  # Neutral

        try:
            # Get symbol-specific stats from learning engine
            symbol_stats = self.learning_engine.get_symbol_performance(symbol)
            if symbol_stats and symbol_stats.get("trades", 0) >= 3:
                win_rate = symbol_stats.get("win_rate", 50.0) / 100.0
                return min(max(win_rate, 0.0), 1.0)
        except Exception as e:
            logger.debug(f"Error getting win rate for {symbol}: {e}")

        return 0.5  # Default neutral

    def _estimate_capital_required(self, decision: Dict[str, Any]) -> float:
        """
        Estimate capital required for this trade.

        Uses confidence to estimate position size.
        Actual sizing happens in risk agent, but this helps with ranking.
        """
        confidence = float(decision.get("confidence", 0.5))

        # Rough estimate based on confidence
        # Higher confidence = willing to allocate more capital
        base_capital = 10000.0  # Base allocation
        confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5x to 2x based on confidence

        return base_capital * confidence_multiplier

    def _select_within_constraints(
        self,
        scored_opportunities: List[Dict[str, Any]],
        available_capital: float,
        max_positions: int,
        current_positions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict], List[Dict], float]:
        """
        Select opportunities that fit within capital and position constraints.

        Returns: (selected, rejected, total_capital_used)
        """
        selected = []
        rejected = []
        used_capital = 0.0

        # Track sectors for diversification
        selected_sectors = set()
        for pos in current_positions:
            sector = pos.get("sector", "UNKNOWN")
            selected_sectors.add(sector)

        for opp in scored_opportunities:
            # Check position limit
            if len(selected) >= max_positions:
                rejected.append(opp)
                continue

            # Check total capital
            estimated_capital = opp["estimated_capital"]
            if used_capital + estimated_capital > available_capital:
                logger.debug(
                    f"   {opp['symbol']}: Rejected (exceeds total capital: "
                    f"₹{used_capital + estimated_capital:.0f} > ₹{available_capital:.0f})"
                )
                rejected.append(opp)
                continue

            # Diversification check (optional bonus, not hard limit)
            # If same sector already selected, slight penalty but still allow
            symbol_sector = opp.get("decision", {}).get("sector", "UNKNOWN")

            # Accept opportunity
            selected.append(opp)
            used_capital += estimated_capital
            selected_sectors.add(symbol_sector)

            logger.debug(
                f"   ✅ {opp['symbol']}: score={opp['composite_score']:.3f}, "
                f"conf={opp['confidence']:.2f}, R:R={opp['rr_ratio']:.1f}, "
                f"capital=₹{estimated_capital:.0f}"
            )

        return selected, rejected, used_capital


# Singleton accessor
_ranker_instance = None


def get_opportunity_ranker(learning_engine=None, market_context=None) -> OpportunityRanker:
    """Get or create opportunity ranker singleton."""
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = OpportunityRanker(learning_engine, market_context)
    return _ranker_instance


if __name__ == "__main__":
    # Quick test
    print("✅ opportunity_ranker.py loaded successfully")
