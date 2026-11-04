from __future__ import annotations

from .financial_analysis import AnalysisContext, AnalysisMetric


def _ratio(name: str, numerator: dict, denominator: dict, expression: str) -> AnalysisMetric:
    value = numerator["value"] / denominator["value"] if denominator["value"] != 0 else None
    flags = ["Denominator is zero"] if denominator["value"] == 0 else []
    if name in {"debt_to_equity", "debt_to_assets"} and value is not None and value > 1:
        flags.append("Elevated leverage")
    return AnalysisMetric(
        name, "Leverage", numerator.get("period"), numerator.get("period_type", "Unknown"),
        value, "CALCULATED" if value is not None else "NOT_AVAILABLE", expression,
        AnalysisContext.source_inputs([numerator, denominator]), flags=flags,
    )


def leverage_analysis(context: AnalysisContext) -> list[AnalysisMetric]:
    metrics = []
    for name, numerator, denominator in (
        ("debt_to_equity", "debt", "equity"),
        ("debt_to_assets", "debt", "total_assets"),
    ):
        for _, _, points in context.aligned(numerator, denominator):
            metrics.append(_ratio(name, points[0], points[1], f"{numerator} / {denominator}"))
    for period, period_type, points in context.aligned("debt", "equity"):
        debt, equity = points
        capital = debt["value"] + equity["value"]
        metrics.append(AnalysisMetric(
            "debt_to_capital", "Leverage", period, period_type,
            debt["value"] / capital if capital else None,
            "CALCULATED" if capital else "NOT_AVAILABLE", "Debt / (Debt + Equity)",
            AnalysisContext.source_inputs(points), flags=["Capital is zero"] if not capital else [],
        ))
    for period, period_type, points in context.aligned("net_debt", "ebitda"):
        net_debt, ebitda = points
        value = net_debt["value"] / ebitda["value"] if ebitda["value"] else None
        metrics.append(AnalysisMetric(
            "net_debt_to_ebitda", "Leverage", period, period_type, value,
            "CALCULATED" if value is not None else "NOT_AVAILABLE", "Net Debt / EBITDA",
            AnalysisContext.source_inputs(points),
            flags=["EBITDA is zero"] if ebitda["value"] == 0 else [],
        ))
    return metrics


def calculate_leverage(context: AnalysisContext) -> list[AnalysisMetric]:
    return leverage_analysis(context)
