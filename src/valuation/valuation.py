from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.analysis.financial_analysis import AnalysisContext


def _numeric(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else None


@dataclass
class DCFValuationEngine:
    mapping: Any
    assumptions: dict[str, float] | None = None

    def __post_init__(self) -> None:
        defaults: dict[str, float | None] = {
            "revenue_growth": 0.08,
            "ebitda_margin": 0.14,
            "tax_rate": 0.24,
            "capex": 0.04,
            "working_capital": 0.02,
            "wacc": None,
            "terminal_growth": None,
        }
        defaults.update(self.assumptions or {})
        self.assumptions = defaults

    def _mapped_df(self, field: str) -> pd.DataFrame:
        df = self.mapping.to_dataframe() if hasattr(self.mapping, "to_dataframe") else pd.DataFrame(self.mapping)
        filtered = df[df["standard_name"] == field].copy()
        if filtered.empty:
            return filtered
        filtered["numeric_value"] = pd.to_numeric(filtered["value"], errors="coerce")
        return filtered.sort_values(["period", "period_type"], na_position="last")

    def _latest(self, field: str) -> float | None:
        point = AnalysisContext(self.mapping).latest_point(field)
        return float(point["value"]) if point is not None else None

    def _required_inputs(self) -> list[str]:
        required = {
            "revenue": self._latest("revenue"),
            "ebitda": self._latest("ebitda"),
            "debt": self._latest("debt"),
            "cash": self._latest("cash"),
            "shares_outstanding": self._latest("shares_outstanding"),
            "market_price": self._latest("market_price"),
        }
        return [name for name, value in required.items() if value is None]

    def build_forecast(self, revenue_growth: float, ebitda_margin: float, tax_rate: float, capex: float, working_capital: float) -> pd.DataFrame:
        revenue_base = self._latest("revenue")
        ebitda_base = self._latest("ebitda")
        if revenue_base is None or ebitda_base is None:
            return pd.DataFrame()
        d_and_a = self._latest("depreciation")
        capex_base = self._latest("capex")
        wc_base = self._latest("working_capital")
        if d_and_a is None or capex_base is None or wc_base is None:
            return pd.DataFrame()
        rows = []
        for year_idx in range(1, 6):
            revenue = revenue_base * ((1 + revenue_growth) ** year_idx)
            ebitda = revenue * ebitda_margin
            ebit = ebitda * 0.75
            nopat = ebit * (1 - tax_rate)
            fcf = nopat + d_and_a - max(abs(capex_base), 0.0) - max(wc_base * (0.05 * year_idx), 0.0)
            rows.append({
                "year": year_idx,
                "revenue": revenue,
                "ebitda": ebitda,
                "ebit": ebit,
                "nopat": nopat,
                "d_and_a": d_and_a,
                "capex": max(abs(capex_base), 0.0),
                "working_capital_change": max(wc_base * (0.05 * year_idx), 0.0),
                "fcf": fcf,
            })
        return pd.DataFrame(rows)

    def estimate(self, assumptions: dict[str, float] | None = None) -> dict[str, Any]:
        settings = {**self.assumptions, **(assumptions or {})}
        missing_inputs = self._required_inputs()
        forecast = self.build_forecast(
            settings["revenue_growth"],
            settings["ebitda_margin"],
            settings["tax_rate"],
            settings["capex"],
            settings["working_capital"],
        )
        shares_outstanding = self._latest("shares_outstanding")
        market_price = self._latest("market_price")
        if shares_outstanding is not None and shares_outstanding <= 0:
            missing_inputs.append("positive shares outstanding")
        if market_price is not None and market_price <= 0:
            missing_inputs.append("positive market price")
        if (
            missing_inputs
            or forecast.empty
            or settings["wacc"] is None
            or settings["terminal_growth"] is None
            or settings["wacc"] <= settings["terminal_growth"]
        ):
            missing_inputs = list(missing_inputs)
            if settings["wacc"] is None:
                missing_inputs.append("wacc assumption")
            if settings["terminal_growth"] is None:
                missing_inputs.append("terminal growth assumption")
            if settings["wacc"] is not None and settings["terminal_growth"] is not None and settings["wacc"] <= settings["terminal_growth"]:
                missing_inputs.append("WACC greater than terminal growth")
            return {
                "status": "NOT_AVAILABLE",
                "missing_inputs": missing_inputs or ["depreciation", "capex", "working_capital"],
                "assumptions": settings,
                "forecast": forecast,
                "enterprise_value": None,
                "net_debt": None,
                "equity_value": None,
                "shares_outstanding": self._latest("shares_outstanding"),
                "intrinsic_value_per_share": None,
                "market_price": self._latest("market_price"),
                "upside_percent": None,
            }
        discount_factors = [(1 + settings["wacc"]) ** int(year) for year in forecast["year"]]
        pv_fcf = (forecast["fcf"] / pd.Series(discount_factors)).sum()
        terminal_value = (forecast["fcf"].iloc[-1] * (1 + settings["terminal_growth"])) / (settings["wacc"] - settings["terminal_growth"])
        pv_terminal = terminal_value / ((1 + settings["wacc"]) ** int(forecast["year"].iloc[-1]))
        enterprise_value = float(pv_fcf + pv_terminal)
        debt = self._latest("debt")
        cash = self._latest("cash")
        if debt is None or cash is None or shares_outstanding is None or market_price is None:
            raise ValueError("Required DCF inputs became unavailable after validation")
        net_debt = debt - cash
        equity_value = enterprise_value - net_debt
        intrinsic_value_per_share = equity_value / shares_outstanding
        upside = ((intrinsic_value_per_share - market_price) / market_price) * 100 if market_price else None
        return {
            "status": "CALCULATED",
            "missing_inputs": [],
            "assumptions": settings,
            "forecast": forecast,
            "enterprise_value": enterprise_value,
            "net_debt": net_debt,
            "equity_value": equity_value,
            "shares_outstanding": shares_outstanding,
            "intrinsic_value_per_share": intrinsic_value_per_share,
            "market_price": market_price,
            "upside_percent": upside,
        }


@dataclass
class ComparableEngine:
    mapping: Any

    def _latest_value(self, field: str) -> float | None:
        df = self.mapping.to_dataframe() if hasattr(self.mapping, "to_dataframe") else pd.DataFrame(self.mapping)
        filtered = df[df["standard_name"] == field].copy()
        if filtered.empty:
            return None
        filtered["numeric_value"] = pd.to_numeric(filtered["value"], errors="coerce")
        latest = filtered["numeric_value"].dropna().iloc[-1]
        return float(latest) if pd.notna(latest) else None

    def peer_rows(self) -> pd.DataFrame:
        rows = []
        for field in ["revenue", "ebitda", "pat", "market_price", "shares_outstanding"]:
            value = self._latest_value(field)
            if value is not None:
                rows.append({"metric": field, "value": value})
        return pd.DataFrame(rows)


@dataclass
class DashboardData:
    mapping: Any
    validation: Any
    analysis: Any

    def _context(self) -> AnalysisContext:
        analysis_context = getattr(self.analysis, "context", None)
        if isinstance(analysis_context, AnalysisContext):
            return analysis_context
        return AnalysisContext(self.mapping, self.validation)

    def overview(self) -> dict[str, Any]:
        context = self._context()
        revenue = context.latest_point("revenue")
        ebitda = context.latest_point("ebitda")
        pat = context.latest_point("pat")
        fcf = context.latest_point("free_cash_flow")
        return {
            "revenue_latest": revenue["value"] if revenue else None,
            "ebitda_latest": ebitda["value"] if ebitda else None,
            "pat_latest": pat["value"] if pat else None,
            "fcf_latest": fcf["value"] if fcf else None,
            "validation_pass": self.validation.count("PASS") if hasattr(self.validation, "count") else 0,
            "validation_warning": self.validation.count("WARNING") if hasattr(self.validation, "count") else 0,
            "validation_fail": self.validation.count("FAIL") if hasattr(self.validation, "count") else 0,
        }
