#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_context.py — Market-wide context and sentiment analysis
===============================================================

Provides overall market context including:
- Nifty 50 trend and momentum
- Market breadth (advance/decline)
- Sectoral strength
- Overall market sentiment

This helps agents make better decisions based on market conditions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

# Set up logging
logger = logging.getLogger("market_context")

# Timezone
IST = ZoneInfo("Asia/Kolkata")


class MarketContext:
    """
    Provides market-wide context for trading decisions.

    Features:
    - Nifty 50 trend analysis
    - Market sentiment (bullish/bearish/neutral)
    - Market breadth calculation
    - Trend strength indicators
    """

    def __init__(self, tech_client=None):
        """
        Initialize market context analyzer.

        Args:
            tech_client: UpstoxTechnicalClient instance for fetching data
        """
        self.tech = tech_client
        if not self.tech:
            try:
                from upstox_technical import UpstoxTechnicalClient
                self.tech = UpstoxTechnicalClient()
            except Exception as e:
                logger.error(f"Failed to initialize technical client: {e}")
                self.tech = None

    def get_nifty_context(self, days: int = 7) -> Dict[str, Any]:
        """
        Get Nifty 50 index analysis with trend and momentum.

        Args:
            days: Number of days of historical data to analyze

        Returns:
            Dictionary containing Nifty analysis
        """
        if not self.tech:
            return {"error": "technical_client_not_available"}

        try:
            # Fetch Nifty 50 data
            # Try different possible names for Nifty 50
            nifty_snapshot = None
            last_error = None

            for nifty_name in ["NIFTY 50", "NIFTY", "NIFTY50", "Nifty 50", "^NSEI"]:
                try:
                    print(f"🔍 Trying to fetch Nifty data with name: {nifty_name}")
                    nifty_snapshot = self.tech.snapshot(nifty_name, days=days)

                    # CRITICAL: Validate we got REAL data, not zeros
                    current_price = nifty_snapshot.get("current_price")
                    if current_price and float(current_price) > 0:
                        print(f"✅ Successfully fetched Nifty data: {nifty_name} @ {current_price}")
                        break
                    else:
                        print(f"⚠️ Got snapshot for {nifty_name} but current_price is {current_price} (invalid)")
                        nifty_snapshot = None  # Treat as failed
                except Exception as e:
                    last_error = str(e)
                    print(f"❌ Failed to fetch {nifty_name}: {e}")
                    continue

            if not nifty_snapshot:
                error_msg = f"nifty_data_not_available (tried all symbols, last error: {last_error})"
                print(f"❌ {error_msg}")
                return {"error": error_msg}

            current_price = nifty_snapshot.get("current_price") or 0
            indicators = nifty_snapshot.get("indicators", {})

            # Extract key indicators with proper null handling
            rsi = indicators.get("rsi14") or 50
            ema20 = indicators.get("ema20") or 0
            ema50 = indicators.get("ema50") or 0
            macd = indicators.get("macd") or 0
            macd_signal = indicators.get("signal") or 0
            change_pct = nifty_snapshot.get("change_percent") or 0

            # Convert to float and handle None values
            try:
                current_price = float(current_price) if current_price else 0
                ema20 = float(ema20) if ema20 else 0
                ema50 = float(ema50) if ema50 else 0
                rsi = float(rsi) if rsi else 50
                macd = float(macd) if macd else 0
                macd_signal = float(macd_signal) if macd_signal else 0
                change_pct = float(change_pct) if change_pct else 0
            except (TypeError, ValueError) as e:
                # If conversion fails, return error
                print(f"❌ Failed to convert Nifty data to float: {e}")
                return {"error": "invalid_nifty_data", "details": f"Price data is not numeric: {e}"}

            # CRITICAL VALIDATION: Don't continue with zero data!
            if current_price == 0:
                print(f"❌ CRITICAL: Nifty current_price is 0! Market data is invalid!")
                return {
                    "error": "invalid_nifty_price",
                    "details": "Got zero price from API - market data feed may be down",
                    "snapshot": nifty_snapshot  # Include raw data for debugging
                }

            print(f"📊 Nifty context: price={current_price:.2f}, change={change_pct:.2f}%, RSI={rsi:.1f}")

            # Determine trend (with null-safe comparisons)
            trend = "NEUTRAL"
            trend_strength = 0

            # Only perform comparisons if all values are valid (non-zero)
            if current_price > 0 and ema20 > 0 and ema50 > 0:
                if current_price > ema20 > ema50:
                    trend = "BULLISH"
                    trend_strength = min(100, abs(((current_price - ema50) / ema50) * 100) * 10)
                elif current_price < ema20 < ema50:
                    trend = "BEARISH"
                    trend_strength = min(100, abs(((current_price - ema50) / ema50) * 100) * 10)
                elif current_price > ema20:
                    trend = "WEAK_BULLISH"
                    trend_strength = 40
                elif current_price < ema20:
                    trend = "WEAK_BEARISH"
                    trend_strength = 40

            # Determine momentum
            momentum = "NEUTRAL"
            if rsi > 60 and macd > macd_signal:
                momentum = "STRONG_BULLISH"
            elif rsi > 50 and macd > macd_signal:
                momentum = "BULLISH"
            elif rsi < 40 and macd < macd_signal:
                momentum = "STRONG_BEARISH"
            elif rsi < 50 and macd < macd_signal:
                momentum = "BEARISH"

            # Overall market sentiment
            sentiment_score = 0
            if trend in ["BULLISH", "WEAK_BULLISH"]:
                sentiment_score += 2 if trend == "BULLISH" else 1
            elif trend in ["BEARISH", "WEAK_BEARISH"]:
                sentiment_score -= 2 if trend == "BEARISH" else 1

            if momentum in ["STRONG_BULLISH", "BULLISH"]:
                sentiment_score += 2 if momentum == "STRONG_BULLISH" else 1
            elif momentum in ["STRONG_BEARISH", "BEARISH"]:
                sentiment_score -= 2 if momentum == "STRONG_BEARISH" else 1

            if change_pct > 0.5:
                sentiment_score += 1
            elif change_pct < -0.5:
                sentiment_score -= 1

            # Classify sentiment
            if sentiment_score >= 3:
                overall_sentiment = "STRONG_BULLISH"
            elif sentiment_score >= 1:
                overall_sentiment = "BULLISH"
            elif sentiment_score <= -3:
                overall_sentiment = "STRONG_BEARISH"
            elif sentiment_score <= -1:
                overall_sentiment = "BEARISH"
            else:
                overall_sentiment = "NEUTRAL"

            # Trading recommendations based on market
            trading_bias = {
                "STRONG_BULLISH": "AGGRESSIVE_LONG",
                "BULLISH": "LONG_BIASED",
                "NEUTRAL": "SELECTIVE",
                "BEARISH": "CAUTIOUS",
                "STRONG_BEARISH": "DEFENSIVE",
            }.get(overall_sentiment, "SELECTIVE")

            return {
                "index": "NIFTY_50",
                "current_price": round(current_price, 2),
                "change_percent": round(change_pct, 2),

                # Trend analysis
                "trend": trend,
                "trend_strength": round(trend_strength, 1),

                # Momentum
                "momentum": momentum,
                "rsi": round(rsi, 2),

                # Moving averages
                "price_vs_ema20": round(((current_price - ema20) / ema20) * 100, 2) if ema20 else None,
                "price_vs_ema50": round(((current_price - ema50) / ema50) * 100, 2) if ema50 else None,
                "ema20": round(ema20, 2),
                "ema50": round(ema50, 2),

                # MACD
                "macd_trend": "BULLISH" if macd > macd_signal else "BEARISH",
                "macd_divergence": round(macd - macd_signal, 2),

                # Overall assessment
                "sentiment": overall_sentiment,
                "sentiment_score": sentiment_score,
                "trading_bias": trading_bias,

                # Context for agents
                "agent_guidance": self._get_agent_guidance(overall_sentiment, trend, momentum),

                "timestamp": datetime.now(IST).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error fetching Nifty context: {e}")
            return {"error": str(e)}

    def _get_agent_guidance(self, sentiment: str, trend: str, momentum: str) -> str:
        """
        Generate guidance text for trading agents based on market context.
        """
        guidance_map = {
            "STRONG_BULLISH": (
                "Market is strongly bullish. Favor long positions. "
                "Look for breakout opportunities and momentum stocks. "
                "Tight stop-losses can be used as trend is strong."
            ),
            "BULLISH": (
                "Market has bullish bias. Long positions preferred. "
                "Be selective and wait for good entry points. "
                "Watch for support levels to hold."
            ),
            "NEUTRAL": (
                "Market is range-bound or mixed. Be selective. "
                "Focus on stock-specific opportunities. "
                "Use wider stop-losses as market lacks clear direction."
            ),
            "BEARISH": (
                "Market has bearish bias. Be cautious with longs. "
                "Consider defensive stocks or wait for better market conditions. "
                "Keep position sizes smaller."
            ),
            "STRONG_BEARISH": (
                "Market is strongly bearish. Avoid new long positions. "
                "Focus on capital preservation. "
                "Only high-conviction trades with tight stop-losses."
            ),
        }

        return guidance_map.get(sentiment, "Market assessment unavailable. Trade with caution.")

    def calculate_market_breadth(
        self,
        symbols: List[str],
        look_back_periods: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate market breadth by analyzing multiple stocks.

        Market breadth shows how many stocks are participating in the move.
        A healthy bull market has broad participation.

        Args:
            symbols: List of stock symbols to analyze
            look_back_periods: Number of recent candles to analyze

        Returns:
            Market breadth analysis
        """
        if not self.tech or not symbols:
            return {"error": "insufficient_data"}

        try:
            advancing = 0
            declining = 0
            unchanged = 0
            above_ema20 = 0
            below_ema20 = 0
            rsi_overbought = 0  # RSI > 70
            rsi_oversold = 0  # RSI < 30
            strong_momentum = 0  # RSI > 60

            analyzed = 0

            for symbol in symbols:
                try:
                    snapshot = self.tech.snapshot(symbol, days=7)

                    # Get values with proper null handling
                    change_pct = snapshot.get("change_percent") or 0
                    indicators = snapshot.get("indicators", {})
                    current_price = snapshot.get("current_price") or 0
                    ema20 = indicators.get("ema20") or 0
                    rsi = indicators.get("rsi14") or 50

                    # Convert to float and handle None
                    try:
                        change_pct = float(change_pct) if change_pct else 0
                        current_price = float(current_price) if current_price else 0
                        ema20 = float(ema20) if ema20 else 0
                        rsi = float(rsi) if rsi else 50
                    except (TypeError, ValueError):
                        # Skip this symbol if data is invalid
                        continue

                    analyzed += 1

                    # Advance/Decline
                    if change_pct > 0.1:
                        advancing += 1
                    elif change_pct < -0.1:
                        declining += 1
                    else:
                        unchanged += 1

                    # Price vs EMA20 (only if both values are valid)
                    if current_price > 0 and ema20 > 0:
                        if current_price > ema20:
                            above_ema20 += 1
                        else:
                            below_ema20 += 1

                    # RSI analysis
                    if rsi > 70:
                        rsi_overbought += 1
                    elif rsi < 30:
                        rsi_oversold += 1

                    if rsi > 60:
                        strong_momentum += 1

                except Exception as e:
                    logger.debug(f"Error analyzing {symbol} for breadth: {e}")
                    continue

            if analyzed == 0:
                return {"error": "no_stocks_analyzed"}

            # Calculate ratios
            advance_decline_ratio = advancing / max(declining, 1)
            advance_pct = (advancing / analyzed) * 100
            decline_pct = (declining / analyzed) * 100
            above_ema_pct = (above_ema20 / analyzed) * 100
            strong_momentum_pct = (strong_momentum / analyzed) * 100

            # Determine market breadth sentiment
            if advance_pct > 70 and above_ema_pct > 70:
                breadth_sentiment = "VERY_STRONG"
            elif advance_pct > 55 and above_ema_pct > 60:
                breadth_sentiment = "STRONG"
            elif advance_pct > 45 and above_ema_pct > 45:
                breadth_sentiment = "NEUTRAL"
            elif advance_pct < 35 or above_ema_pct < 35:
                breadth_sentiment = "WEAK"
            else:
                breadth_sentiment = "VERY_WEAK"

            return {
                "stocks_analyzed": analyzed,

                # Advance/Decline
                "advancing": advancing,
                "declining": declining,
                "unchanged": unchanged,
                "advance_percent": round(advance_pct, 1),
                "decline_percent": round(decline_pct, 1),
                "advance_decline_ratio": round(advance_decline_ratio, 2),

                # Trend participation
                "above_ema20": above_ema20,
                "below_ema20": below_ema20,
                "above_ema_percent": round(above_ema_pct, 1),

                # Momentum distribution
                "strong_momentum_count": strong_momentum,
                "strong_momentum_percent": round(strong_momentum_pct, 1),
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold,

                # Overall assessment
                "breadth_sentiment": breadth_sentiment,

                # Interpretation
                "interpretation": self._interpret_breadth(
                    breadth_sentiment,
                    advance_pct,
                    above_ema_pct,
                    strong_momentum_pct
                ),

                "timestamp": datetime.now(IST).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error calculating market breadth: {e}")
            return {"error": str(e)}

    def _interpret_breadth(
        self,
        sentiment: str,
        advance_pct: float,
        above_ema_pct: float,
        momentum_pct: float
    ) -> str:
        """Generate interpretation text for market breadth."""
        interpretations = {
            "VERY_STRONG": (
                f"Market breadth is very strong with {advance_pct:.0f}% stocks advancing "
                f"and {above_ema_pct:.0f}% above their 20-EMA. Broad-based rally in progress. "
                "High probability of trend continuation."
            ),
            "STRONG": (
                f"Market breadth is positive with {advance_pct:.0f}% stocks advancing. "
                "Good participation in the move. Favorable for long positions."
            ),
            "NEUTRAL": (
                f"Market breadth is mixed with {advance_pct:.0f}% advancing vs {100-advance_pct:.0f}% declining. "
                "Selective stock-specific moves. Choose stocks carefully."
            ),
            "WEAK": (
                f"Market breadth is weak with only {advance_pct:.0f}% stocks advancing. "
                "Narrow leadership or declining market. Be cautious."
            ),
            "VERY_WEAK": (
                f"Market breadth is very weak with {advance_pct:.0f}% advancing. "
                "Broad-based selling or very narrow rally. High risk environment."
            ),
        }

        return interpretations.get(sentiment, "Market breadth analysis unavailable.")

    def get_complete_market_context(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get complete market context including Nifty and breadth.

        Args:
            symbols: Optional list of symbols for breadth calculation

        Returns:
            Complete market context
        """
        context = {
            "timestamp": datetime.now(IST).isoformat(),
        }

        # Get Nifty context
        nifty_context = self.get_nifty_context()
        context["nifty"] = nifty_context

        # Get market breadth if symbols provided
        if symbols and len(symbols) > 3:
            breadth = self.calculate_market_breadth(symbols)
            context["breadth"] = breadth

            # Combine assessments
            nifty_sentiment = nifty_context.get("sentiment", "NEUTRAL")
            breadth_sentiment = breadth.get("breadth_sentiment", "NEUTRAL")

            # Determine if they align
            bullish_signals = sum([
                "BULLISH" in nifty_sentiment,
                "STRONG" in breadth_sentiment or breadth.get("advance_percent", 0) > 60,
            ])

            bearish_signals = sum([
                "BEARISH" in nifty_sentiment,
                "WEAK" in breadth_sentiment or breadth.get("advance_percent", 0) < 40,
            ])

            if bullish_signals >= 2:
                combined_assessment = "BULLISH_CONFIRMED"
            elif bearish_signals >= 2:
                combined_assessment = "BEARISH_CONFIRMED"
            elif bullish_signals == 1 and bearish_signals == 0:
                combined_assessment = "MILDLY_BULLISH"
            elif bearish_signals == 1 and bullish_signals == 0:
                combined_assessment = "MILDLY_BEARISH"
            else:
                combined_assessment = "MIXED"

            context["combined_assessment"] = combined_assessment
            context["recommendation"] = self._get_combined_recommendation(combined_assessment)

        return context

    def _get_combined_recommendation(self, assessment: str) -> str:
        """Get trading recommendation based on combined market assessment."""
        recommendations = {
            "BULLISH_CONFIRMED": (
                "Market conditions are bullish (Nifty + Breadth confirming). "
                "Favor long positions with normal risk allocation."
            ),
            "BEARISH_CONFIRMED": (
                "Market conditions are bearish (Nifty + Breadth confirming). "
                "Be defensive, reduce position sizes, avoid fresh longs."
            ),
            "MILDLY_BULLISH": (
                "Market has mild bullish bias but not fully confirmed. "
                "Selective longs with careful stock selection."
            ),
            "MILDLY_BEARISH": (
                "Market has mild bearish bias. Exercise caution, "
                "wait for better setups or reduce exposure."
            ),
            "MIXED": (
                "Market signals are mixed (Nifty vs Breadth divergence). "
                "Trade with heightened caution, smaller sizes, focus on high-conviction setups."
            ),
        }
        return recommendations.get(assessment, "No clear market direction. Trade cautiously.")


# Global singleton instance
_market_context_instance: Optional[MarketContext] = None


def get_market_context() -> MarketContext:
    """Get the global market context instance."""
    global _market_context_instance
    if _market_context_instance is None:
        _market_context_instance = MarketContext()
    return _market_context_instance


# Convenience functions
def get_nifty_context(days: int = 7) -> Dict[str, Any]:
    """Get Nifty market context."""
    return get_market_context().get_nifty_context(days)


def get_complete_context(symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get complete market context."""
    return get_market_context().get_complete_market_context(symbols)


def calculate_breadth(symbols: List[str]) -> Dict[str, Any]:
    """Calculate market breadth."""
    return get_market_context().calculate_market_breadth(symbols)
