from __future__ import annotations

from .financial_analysis import AnalysisContext, AnalysisMetric


def _ratio(name: str, numerator: dict, denominator: dict, expression: str) -> AnalysisMetric:
    value = numerator["value"] / denominator["value"] if denominator["value"] != 0 else None
    flags = ["Denominator is zero"] if denominator["value"] == 0 else []
    if value is not None and value < 1:
        flags.append("Below 1.0x")
    return AnalysisMetric(
        name, "Liquidity", numerator.get("period"), numerator.get("period_type", "Unknown"),
        value, "CALCULATED" if value is not None else "NOT_AVAILABLE", expression,
        AnalysisContext.source_inputs([numerator, denominator]), flags=flags,
    )


def liquidity_analysis(context: AnalysisContext) -> list[AnalysisMetric]:
    metrics = []
    for name, numerator, denominator in (
        ("current_ratio", "current_assets", "current_liabilities"),
        ("cash_ratio", "cash", "current_liabilities"),
    ):
        for _, _, points in context.aligned(numerator, denominator):
            metrics.append(_ratio(name, points[0], points[1], f"{numerator} / {denominator}"))
    for period, period_type, points in context.aligned("current_assets", "current_liabilities"):
        assets, liabilities = points
        value = assets["value"] - liabilities["value"]
        metrics.append(AnalysisMetric(
            "working_capital_calculated", "Liquidity", period, period_type, value, "CALCULATED",
            "Current Assets - Current Liabilities", AnalysisContext.source_inputs(points),
            flags=["Negative working capital"] if value < 0 else [],
        ))
    return metrics


def calculate_liquidity(context: AnalysisContext) -> list[AnalysisMetric]:
    return liquidity_analysis(context)
