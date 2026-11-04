from __future__ import annotations

from typing import Any

import pandas as pd

from .financial_analysis import AnalysisContext, AnalysisMetric


def _validation_checks(validation_results: Any) -> list[dict[str, Any]]:
    if validation_results is None:
        return []
    if hasattr(validation_results, "to_dataframe"):
        dataframe = validation_results.to_dataframe()
    elif hasattr(validation_results, "checks"):
        dataframe = pd.DataFrame([check.__dict__ for check in validation_results.checks])
    else:
        dataframe = validation_results
    if dataframe is None or getattr(dataframe, "empty", True):
        return []
    return dataframe.to_dict("records")


def analysis_flags(context: AnalysisContext, metrics: list[AnalysisMetric]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for check in _validation_checks(context.validation_results):
        status = str(check.get("status", ""))
        if status in {"WARNING", "FAIL"}:
            flags.append({
                "flag": check.get("check_name", "Validation check"),
                "category": check.get("category", "Validation"),
                "severity": check.get("severity", status),
                "status": status,
                "message": check.get("message", ""),
                "source": check.get("source", ""),
            })
    for metric in metrics:
        for flag in metric.flags:
            flags.append({
                "flag": f"{metric.metric_name}: {flag}",
                "category": metric.category,
                "severity": "MEDIUM",
                "status": "WARNING",
                "message": flag,
                "source": metric.source,
            })
    return flags


def build_insights(
    metrics: list[AnalysisMetric],
    flags: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> list[str]:
    insights: list[str] = []
    for metric in metrics:
        if "growth" in metric.metric_name and metric.value is not None and metric.value < 0:
            insights.append(f"{metric.metric_name} declined in {metric.period}.")
        if metric.metric_name in {"gross_margin", "ebitda_margin", "net_margin"} and metric.value is not None:
            insights.append(f"{metric.metric_name} is {metric.value:.1%} in {metric.period}.")
    if any(str(flag.get("category")) == "Liquidity" for flag in flags):
        insights.append("Liquidity metrics include a below-threshold or unavailable indicator.")
    if any(str(flag.get("category")) == "Leverage" for flag in flags):
        insights.append("Leverage metrics include an elevated or unavailable indicator.")
    if not insights and summaries:
        insights.append("No analytical flags were generated from the available mapped periods.")
    return list(dict.fromkeys(insights))


def validate_analysis(context: AnalysisContext, metrics: list[AnalysisMetric]) -> list[dict[str, Any]]:
    return analysis_flags(context, metrics)
