from __future__ import annotations

from typing import Any

from .financial_analysis import AnalysisContext, AnalysisMetric


def _ratio(
    name: str, category: str, numerator: dict[str, Any], denominator: dict[str, Any],
    expression: str, flags: list[str] | None = None,
) -> AnalysisMetric:
    value = numerator["value"] / denominator["value"] if denominator["value"] != 0 else None
    local_flags = list(flags or [])
    if denominator["value"] == 0:
        local_flags.append("Denominator is zero")
    return AnalysisMetric(
        name, category, numerator.get("period"), numerator.get("period_type", "Unknown"),
        value, "CALCULATED" if value is not None else "NOT_AVAILABLE", expression,
        AnalysisContext.source_inputs([numerator, denominator]), flags=local_flags,
    )


def profitability_analysis(context: AnalysisContext) -> list[AnalysisMetric]:
    metrics: list[AnalysisMetric] = []
    for numerator, name in (
        ("gross_profit", "gross_margin"),
        ("ebitda", "ebitda_margin"),
        ("ebit", "ebit_margin"),
        ("pat", "net_margin"),
    ):
        for period, period_type, points in context.aligned(numerator, "revenue"):
            metrics.append(_ratio(
                name, "Profitability", points[0], points[1],
                f"{numerator} / revenue", ["Margin is negative"] if points[0]["value"] < 0 else [],
            ))

    for numerator, denominator, name in (
        ("pat", "equity", "roe"),
        ("pat", "total_assets", "roa"),
    ):
        for _, _, points in context.aligned(numerator, denominator):
            metrics.append(_ratio(name, "Profitability", points[0], points[1], f"{numerator} / {denominator}"))

    for period, period_type, points in context.aligned("ebit", "debt", "equity", "cash"):
        ebit, debt, equity, cash = points
        invested_capital = debt["value"] + equity["value"] - cash["value"]
        flags = []
        if invested_capital == 0:
            flags.append("Invested capital is zero")
        tax_points = {
            (p.get("period"), p.get("period_type")): p for p in context.unique_points("tax")
        }
        ebt_points = {
            (p.get("period"), p.get("period_type")): p for p in context.unique_points("ebt")
        }
        tax = tax_points.get((period, period_type))
        ebt = ebt_points.get((period, period_type))
        nopat = ebit["value"]
        source_points = list(points)
        expression = "EBIT / (Debt + Equity - Cash)"
        if tax is not None:
            nopat = ebit["value"] - tax["value"]
            source_points.append(tax)
            expression = "(EBIT - Tax) / (Debt + Equity - Cash)"
        else:
            flags.append("Tax unavailable; EBIT proxy used for NOPAT")
        if ebt is not None and ebt["value"] != 0:
            flags.append(f"Effective tax rate: {tax['value'] / ebt['value']:.2%}" if tax else "Tax input unavailable")
            source_points.append(ebt)
        value = nopat / invested_capital if invested_capital != 0 else None
        metrics.append(AnalysisMetric(
            "roic", "Profitability", period, period_type, value,
            "CALCULATED" if value is not None else "NOT_AVAILABLE", expression,
            AnalysisContext.source_inputs(source_points), flags=flags,
        ))
    return metrics


def calculate_margins(context: AnalysisContext) -> list[AnalysisMetric]:
    return profitability_analysis(context)


def calculate_roe(context: AnalysisContext) -> list[AnalysisMetric]:
    return [metric for metric in profitability_analysis(context) if metric.metric_name == "roe"]


def calculate_roa(context: AnalysisContext) -> list[AnalysisMetric]:
    return [metric for metric in profitability_analysis(context) if metric.metric_name == "roa"]


def calculate_roic(context: AnalysisContext) -> list[AnalysisMetric]:
    return [metric for metric in profitability_analysis(context) if metric.metric_name == "roic"]
