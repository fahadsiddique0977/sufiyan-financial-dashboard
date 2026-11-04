from __future__ import annotations

from .financial_analysis import AnalysisContext, AnalysisMetric


def _ratio(name: str, numerator: dict, denominator: dict, expression: str) -> AnalysisMetric:
    value = numerator["value"] / denominator["value"] if denominator["value"] else None
    return AnalysisMetric(
        name, "Cash Flow", numerator.get("period"), numerator.get("period_type", "Unknown"),
        value, "CALCULATED" if value is not None else "NOT_AVAILABLE", expression,
        AnalysisContext.source_inputs([numerator, denominator]),
        flags=["Denominator is zero"] if denominator["value"] == 0 else [],
    )


def cash_flow_analysis(context: AnalysisContext) -> list[AnalysisMetric]:
    metrics = []
    for name, numerator, denominator in (
        ("operating_cash_flow_margin", "operating_cash_flow", "revenue"),
        ("free_cash_flow_margin", "free_cash_flow", "revenue"),
        ("operating_cash_flow_conversion", "operating_cash_flow", "pat"),
        ("free_cash_flow_conversion", "free_cash_flow", "pat"),
    ):
        for _, _, points in context.aligned(numerator, denominator):
            metrics.append(_ratio(name, points[0], points[1], f"{numerator} / {denominator}"))

    for period, period_type, points in context.aligned("operating_cash_flow", "capex"):
        cfo, capex = points
        value = cfo["value"] - abs(capex["value"])
        metrics.append(AnalysisMetric(
            "free_cash_flow_calculated", "Cash Flow", period, period_type, value, "CALCULATED",
            "Operating Cash Flow - absolute(Capex)", AnalysisContext.source_inputs(points),
            flags=["Negative free cash flow"] if value < 0 else [],
        ))

    for period, period_type, points in context.aligned("pat", "shares_outstanding"):
        pat, shares = points
        value = pat["value"] / shares["value"] if shares["value"] else None
        metrics.append(AnalysisMetric(
            "eps_calculated", "Per Share", period, period_type, value,
            "CALCULATED" if value is not None else "NOT_AVAILABLE", "PAT / Shares Outstanding",
            AnalysisContext.source_inputs(points),
            flags=["Shares outstanding is zero"] if shares["value"] == 0 else [],
        ))
    for period, period_type, points in context.aligned("market_price", "shares_outstanding"):
        price, shares = points
        value = price["value"] * shares["value"]
        metrics.append(AnalysisMetric(
            "market_capitalization_calculated", "Per Share", period, period_type, value,
            "CALCULATED", "Market Price × Shares Outstanding", AnalysisContext.source_inputs(points),
        ))
    for item in ("eps", "market_price"):
        for point in context.unique_points(item):
            metrics.append(AnalysisMetric(
                item, "Per Share", point.get("period"), point.get("period_type", "Unknown"),
                point["value"], str(point.get("status") or "MAPPED").upper(),
                "Mapped source value", AnalysisContext.source_inputs([point]),
            ))
    return metrics


def calculate_cash_flow(context: AnalysisContext) -> list[AnalysisMetric]:
    return cash_flow_analysis(context)
