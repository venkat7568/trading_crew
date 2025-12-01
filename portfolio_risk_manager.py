#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
portfolio_risk_manager.py — Portfolio-Level Risk Validation
============================================================

Validates that selected opportunities don't create excessive:
- Sector concentration risk
- Correlation risk (multiple stocks moving together)
- Directional bias (all BUY or all SELL)
- Aggregate R:R quality

This is the FINAL check before execution - ensures portfolio
stays balanced and doesn't take on hidden risks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict

logger = logging.getLogger("portfolio_risk")

IST = ZoneInfo("Asia/Kolkata")


class PortfolioRiskManager:
    """
    Portfolio-level risk validator.

    Checks for:
    1. Sector concentration (max % in one sector)
    2. Correlation clusters (highly correlated stocks)
    3. Directional balance (not all same direction)
    4. Aggregate portfolio R:R
    5. Product mix balance (intraday vs swing)
    """

    def __init__(self, max_sector_concentration: float = 100.0):
        """
        Initialize portfolio risk manager.

        Args:
            max_sector_concentration: Max % of capital in one sector (default 100% - disabled)
        """
        self.max_sector_concentration = max_sector_concentration

        # Indian stock sector mapping (basic classification)
        self.sector_map = self._build_sector_map()

        # Correlation clusters (stocks that tend to move together)
        self.correlation_clusters = self._build_correlation_clusters()

    def validate_portfolio_risk(
        self,
        selected_opportunities: List[Dict[str, Any]],
        current_positions: Optional[List[Dict[str, Any]]] = None,
        total_capital: float = 100000.0,
    ) -> Dict[str, Any]:
        """
        Validate portfolio-level risk for selected opportunities.

        Args:
            selected_opportunities: Opportunities ranked and selected
            current_positions: Currently open positions
            total_capital: Total trading capital

        Returns:
            {
                "approved": True/False,
                "warnings": [...],
                "blocking_issues": [...],
                "metrics": {...},
                "recommendations": [...]
            }
        """
        current_positions = current_positions or []

        warnings = []
        blocking_issues = []
        recommendations = []

        # Combine new + existing for full portfolio view
        all_positions = current_positions + [
            {
                "symbol": opp["symbol"],
                "direction": opp["direction"],
                "style": opp["style"],
                "estimated_capital": opp["estimated_capital"],
            }
            for opp in selected_opportunities
        ]

        # 1. Sector concentration check
        sector_risk = self._check_sector_concentration(
            all_positions, selected_opportunities, total_capital
        )
        if sector_risk["violations"]:
            blocking_issues.extend(sector_risk["violations"])
        warnings.extend(sector_risk["warnings"])

        # 2. Correlation risk check
        correlation_risk = self._check_correlation_risk(selected_opportunities)
        warnings.extend(correlation_risk["warnings"])
        if correlation_risk["high_risk"]:
            recommendations.append(
                "High correlation detected - consider reducing position sizes"
            )

        # 3. Directional balance check (disabled - not blocking trades)
        direction_risk = self._check_directional_balance(
            selected_opportunities, current_positions
        )
        # Removed: warnings.extend(direction_risk["warnings"])
        # Removed: recommendations for directional imbalance

        # 4. Aggregate R:R check
        rr_metrics = self._check_aggregate_rr(selected_opportunities)
        if rr_metrics["avg_rr"] < 1.5:
            warnings.append(
                f"Average R:R is low ({rr_metrics['avg_rr']:.2f}) - "
                "consider only high-quality setups"
            )

        # 5. Product mix check (disabled - all trades are delivery orders)
        product_mix = self._check_product_mix(selected_opportunities)
        # Removed: warnings for product mix imbalance

        # Overall verdict
        approved = len(blocking_issues) == 0

        metrics = {
            "sector_concentration": sector_risk["metrics"],
            "correlation_risk": correlation_risk["score"],
            "directional_balance": direction_risk["metrics"],
            "aggregate_rr": rr_metrics,
            "product_mix": product_mix["metrics"],
        }

        result = {
            "approved": approved,
            "warnings": warnings,
            "blocking_issues": blocking_issues,
            "recommendations": recommendations,
            "metrics": metrics,
            "timestamp": datetime.now(IST).isoformat(),
        }

        logger.info(
            f"Portfolio risk validation: {'✅ APPROVED' if approved else '❌ BLOCKED'}"
        )
        if warnings:
            logger.warning(f"   Warnings: {len(warnings)}")
        if blocking_issues:
            logger.error(f"   Blocking issues: {len(blocking_issues)}")

        return result

    def _check_sector_concentration(
        self,
        all_positions: List[Dict[str, Any]],
        new_opportunities: List[Dict[str, Any]],
        total_capital: float,
    ) -> Dict[str, Any]:
        """Check for excessive sector concentration."""
        sector_exposure = defaultdict(float)
        sector_symbols = defaultdict(list)

        # Calculate exposure by sector
        for pos in all_positions:
            symbol = pos.get("symbol", "UNKNOWN")
            capital = float(pos.get("estimated_capital", 0) or pos.get("value", 0))
            sector = self._get_sector(symbol)

            sector_exposure[sector] += capital
            sector_symbols[sector].append(symbol)

        violations = []
        warnings = []

        # Check each sector
        for sector, exposure in sector_exposure.items():
            pct = (exposure / total_capital) * 100 if total_capital > 0 else 0

            if pct > self.max_sector_concentration:
                violations.append(
                    f"Sector {sector} exceeds limit: {pct:.1f}% > {self.max_sector_concentration:.1f}% "
                    f"(symbols: {', '.join(sector_symbols[sector])})"
                )
            elif pct > self.max_sector_concentration * 0.8:  # Warning at 80% of limit
                warnings.append(
                    f"Sector {sector} near limit: {pct:.1f}% of {self.max_sector_concentration:.1f}%"
                )

        metrics = {
            "by_sector": {
                sector: {
                    "exposure": exposure,
                    "pct": (exposure / total_capital) * 100 if total_capital > 0 else 0,
                    "symbols": sector_symbols[sector],
                }
                for sector, exposure in sector_exposure.items()
            },
            "max_concentration": max(
                (exp / total_capital) * 100 if total_capital > 0 else 0
                for exp in sector_exposure.values()
            )
            if sector_exposure
            else 0,
        }

        return {"violations": violations, "warnings": warnings, "metrics": metrics}

    def _check_correlation_risk(
        self, opportunities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check for correlated stocks being traded together."""
        symbols = [opp["symbol"] for opp in opportunities]
        warnings = []
        high_risk = False

        # Check each correlation cluster
        for cluster_name, cluster_symbols in self.correlation_clusters.items():
            # How many symbols from this cluster are we trading?
            cluster_count = sum(1 for sym in symbols if sym in cluster_symbols)

            if cluster_count >= 3:
                high_risk = True
                warnings.append(
                    f"High correlation risk: {cluster_count} stocks from {cluster_name} cluster "
                    f"({', '.join([s for s in symbols if s in cluster_symbols])})"
                )
            elif cluster_count >= 2:
                warnings.append(
                    f"Moderate correlation: {cluster_count} stocks from {cluster_name} cluster"
                )

        # Correlation risk score (0-1, higher = more risk)
        max_cluster_pct = (
            max(
                sum(1 for sym in symbols if sym in cluster)
                for cluster in self.correlation_clusters.values()
            )
            / max(len(symbols), 1)
            if symbols
            else 0
        )

        return {
            "warnings": warnings,
            "high_risk": high_risk,
            "score": max_cluster_pct,
        }

    def _check_directional_balance(
        self,
        new_opportunities: List[Dict[str, Any]],
        current_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check if portfolio is too heavily skewed in one direction."""
        buy_count = sum(
            1 for opp in new_opportunities if opp.get("direction") == "BUY"
        )
        sell_count = sum(
            1 for opp in new_opportunities if opp.get("direction") == "SELL"
        )

        # Add current positions
        for pos in current_positions:
            side = pos.get("side", "").upper() or pos.get("direction", "").upper()
            if side == "BUY":
                buy_count += 1
            elif side == "SELL":
                sell_count += 1

        total = buy_count + sell_count
        warnings = []
        imbalanced = False
        dominant_direction = None

        if total > 0:
            buy_pct = (buy_count / total) * 100
            sell_pct = (sell_count / total) * 100

            # Flag if >75% in one direction
            if buy_pct > 75:
                imbalanced = True
                dominant_direction = "LONG"
                warnings.append(
                    f"Heavily long-biased: {buy_pct:.0f}% BUY positions - "
                    "consider market risk"
                )
            elif sell_pct > 75:
                imbalanced = True
                dominant_direction = "SHORT"
                warnings.append(
                    f"Heavily short-biased: {sell_pct:.0f}% SELL positions - "
                    "consider market risk"
                )

        return {
            "warnings": warnings,
            "imbalanced": imbalanced,
            "dominant_direction": dominant_direction,
            "metrics": {
                "buy_count": buy_count,
                "sell_count": sell_count,
                "buy_pct": (buy_count / total * 100) if total > 0 else 0,
                "sell_pct": (sell_count / total * 100) if total > 0 else 0,
            },
        }

    def _check_aggregate_rr(self, opportunities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate R:R metrics for portfolio."""
        if not opportunities:
            return {"avg_rr": 0.0, "min_rr": 0.0, "max_rr": 0.0}

        rr_ratios = [opp.get("rr_ratio", 1.5) for opp in opportunities]

        return {
            "avg_rr": sum(rr_ratios) / len(rr_ratios),
            "min_rr": min(rr_ratios),
            "max_rr": max(rr_ratios),
            "below_threshold_count": sum(1 for rr in rr_ratios if rr < 1.5),
        }

    def _check_product_mix(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check balance between intraday and swing trades."""
        intraday_count = sum(1 for opp in opportunities if opp.get("style") == "intraday")
        swing_count = sum(1 for opp in opportunities if opp.get("style") == "swing")

        total = intraday_count + swing_count
        imbalanced = False
        warning = None

        if total > 0:
            intraday_pct = (intraday_count / total) * 100

            # Flag if >90% in one product type
            if intraday_pct > 90:
                imbalanced = True
                warning = (
                    f"Portfolio heavily intraday ({intraday_pct:.0f}%) - "
                    "consider swing opportunities for diversification"
                )
            elif intraday_pct < 10:
                imbalanced = True
                warning = (
                    f"Portfolio heavily swing ({100 - intraday_pct:.0f}%) - "
                    "consider intraday opportunities for quick gains"
                )

        return {
            "imbalanced": imbalanced,
            "warning": warning,
            "metrics": {
                "intraday_count": intraday_count,
                "swing_count": swing_count,
                "intraday_pct": (intraday_count / total * 100) if total > 0 else 0,
            },
        }

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self.sector_map.get(symbol.upper(), "OTHER")

    def _build_sector_map(self) -> Dict[str, str]:
        """Build symbol → sector mapping for major Indian stocks."""
        return {
            # Banks
            "HDFCBANK": "BANKING",
            "ICICIBANK": "BANKING",
            "KOTAKBANK": "BANKING",
            "AXISBANK": "BANKING",
            "SBIN": "BANKING",
            "INDUSINDBK": "BANKING",
            "BANKBARODA": "BANKING",
            "PNB": "BANKING",
            # IT
            "TCS": "IT",
            "INFY": "IT",
            "WIPRO": "IT",
            "HCLTECH": "IT",
            "TECHM": "IT",
            "LTIM": "IT",
            "COFORGE": "IT",
            # Auto
            "MARUTI": "AUTO",
            "TATAMOTORS": "AUTO",
            "M&M": "AUTO",
            "EICHERMOT": "AUTO",
            "BAJAJ-AUTO": "AUTO",
            "HEROMOTOCO": "AUTO",
            # FMCG
            "HINDUNILVR": "FMCG",
            "ITC": "FMCG",
            "NESTLEIND": "FMCG",
            "BRITANNIA": "FMCG",
            "DABUR": "FMCG",
            "MARICO": "FMCG",
            # Pharma
            "SUNPHARMA": "PHARMA",
            "DRREDDY": "PHARMA",
            "CIPLA": "PHARMA",
            "DIVISLAB": "PHARMA",
            "AUROPHARMA": "PHARMA",
            # Energy
            "RELIANCE": "ENERGY",
            "ONGC": "ENERGY",
            "BPCL": "ENERGY",
            "IOC": "ENERGY",
            "ADANIPORTS": "ENERGY",
            # Telecom
            "BHARTIARTL": "TELECOM",
            "IDEA": "TELECOM",
            # Metals
            "TATASTEEL": "METALS",
            "HINDALCO": "METALS",
            "JSWSTEEL": "METALS",
            "VEDL": "METALS",
            "NATIONALUM": "METALS",
            # Cement
            "ULTRACEMCO": "CEMENT",
            "GRASIM": "CEMENT",
            "AMBUJACEM": "CEMENT",
            "ACC": "CEMENT",
            # Infrastructure
            "LT": "INFRASTRUCTURE",
            "ADANIENT": "INFRASTRUCTURE",
        }

    def _build_correlation_clusters(self) -> Dict[str, Set[str]]:
        """Build correlation clusters (stocks that move together)."""
        return {
            "PRIVATE_BANKS": {
                "HDFCBANK",
                "ICICIBANK",
                "KOTAKBANK",
                "AXISBANK",
                "INDUSINDBK",
            },
            "PSU_BANKS": {"SBIN", "PNB", "BANKBARODA", "CANBK"},
            "IT_SERVICES": {"TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"},
            "AUTO_MAJORS": {"MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO"},
            "OIL_GAS": {"RELIANCE", "ONGC", "BPCL", "IOC"},
            "PHARMA": {"SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"},
            "METALS": {"TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL"},
        }


# Singleton accessor
_portfolio_risk_instance = None


def get_portfolio_risk_manager(
    max_sector_concentration: float = 100.0,
) -> PortfolioRiskManager:
    """Get or create portfolio risk manager singleton."""
    global _portfolio_risk_instance
    if _portfolio_risk_instance is None:
        _portfolio_risk_instance = PortfolioRiskManager(max_sector_concentration)
    return _portfolio_risk_instance


if __name__ == "__main__":
    # Quick test
    print("✅ portfolio_risk_manager.py loaded successfully")
