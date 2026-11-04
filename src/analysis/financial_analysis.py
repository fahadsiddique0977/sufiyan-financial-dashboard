from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Iterable

import pandas as pd


@dataclass
class AnalysisMetric:
    """One calculated metric, retaining the periods and mapped source inputs."""

    metric_name: str
    category: str
    period: str | None
    period_type: str
    value: float | None
    status: str
    calculation: str
    source_inputs: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""
    flags: list[str] = field(default_factory=list)
    insight: str = ""

    def __post_init__(self) -> None:
        if not self.source and self.source_inputs:
            locations = []
            for source_input in self.source_inputs:
                sheet = source_input.get("source_sheet")
                column = source_input.get("source_column")
                row = source_input.get("source_row")
                if sheet and column and row:
                    locations.append(f"{sheet}!{column}{row}")
            self.source = "; ".join(dict.fromkeys(locations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "category": self.category,
            "period": self.period,
            "period_type": self.period_type,
            "value": self.value,
            "status": self.status,
            "calculation": self.calculation,
            "source_inputs": self.source_inputs,
            "source": self.source,
            "flags": self.flags,
            "insight": self.insight,
        }


@dataclass
class AnalysisResult:
    metrics: list[AnalysisMetric]
    summaries: list[dict[str, Any]]
    flags: list[dict[str, Any]]
    insights: list[str]

    def to_dataframe(self) -> pd.DataFrame:
        columns = [
            "metric_name", "category", "period", "period_type", "value",
            "status", "calculation", "source_inputs", "source", "flags", "insight",
        ]
        return pd.DataFrame([metric.to_dict() for metric in self.metrics], columns=columns)

    def summary_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.summaries)

    def flags_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.flags)


def _period_key(period: object, period_type: object) -> tuple[int, str, str]:
    text = str(period or "")
    match = re.search(r"(19|20)\d{2}", text)
    year = int(match.group(0)) if match else 9999
    type_text = str(period_type or "Unknown")
    type_order = {"Historical": 0, "Unknown": 1, "Forecast": 2, "LTM": 3}.get(type_text, 1)
    return year, type_order, text


def _number(value: object) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(converted):
        return None
    number = float(converted)
    return number if math.isfinite(number) else None


class AnalysisContext:
    """Normalized, read-only view of MappingResult records for analysis modules."""

    def __init__(self, mapped_data: Any, validation_results: Any = None) -> None:
        if hasattr(mapped_data, "to_dataframe"):
            dataframe = mapped_data.to_dataframe()
        elif isinstance(mapped_data, pd.DataFrame):
            dataframe = mapped_data.copy()
        else:
            dataframe = pd.DataFrame(mapped_data or [])
        self.dataframe = dataframe.copy()
        self.validation_results = validation_results
        for column, default in (
            ("standard_name", None), ("value", None), ("period", None),
            ("period_type", "Unknown"), ("status", "Unknown"),
            ("source_sheet", None), ("source_row", None), ("source_column", None),
            ("detected_label", None), ("confidence", None),
        ):
            if column not in self.dataframe.columns:
                self.dataframe[column] = default
        if not self.dataframe.empty:
            self.dataframe["_numeric_value"] = self.dataframe["value"].map(_number)
        else:
            self.dataframe["_numeric_value"] = pd.Series(dtype=float)
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def points(self, item: str, numeric_only: bool = True) -> list[dict[str, Any]]:
        if item in self._cache and numeric_only:
            return self._cache[item]
        if self.dataframe.empty:
            return []
        rows = self.dataframe[self.dataframe["standard_name"].astype(str) == item]
        points: list[dict[str, Any]] = []
        for _, row in rows.iterrows():
            number = row.get("_numeric_value")
            if numeric_only and (number is None or pd.isna(number) or not math.isfinite(float(number))):
                continue
            point = {column: row.get(column) for column in self.dataframe.columns}
            point["value"] = number
            period = point.get("period")
            period_text = "" if pd.isna(period) else str(period).strip()
            point["period"] = None if period_text.lower() in {"", "nan", "nat", "none"} else period_text
            point["period_type"] = str(point.get("period_type") or "Unknown")
            points.append(point)
        points.sort(key=lambda p: _period_key(p.get("period"), p.get("period_type")))
        if numeric_only:
            self._cache[item] = points
        return points

    def unique_points(self, item: str) -> list[dict[str, Any]]:
        """Select one source row per period while keeping the selected traceability."""
        selected: dict[tuple[object, object], dict[str, Any]] = {}
        confidence = {"High": 3, "Medium": 2, "Low": 1, "Unmapped": 0}
        points = self.points(item)
        dated_points = [point for point in points if point.get("period")]
        if dated_points:
            points = dated_points
        for point in points:
            key = (point.get("period"), point.get("period_type"))
            current = selected.get(key)
            if current is None or confidence.get(str(point.get("confidence")), 0) > confidence.get(
                str(current.get("confidence")), 0
            ):
                selected[key] = point
        return sorted(selected.values(), key=lambda p: _period_key(p.get("period"), p.get("period_type")))

    def latest_point(self, item: str) -> dict[str, Any] | None:
        points = self.unique_points(item)
        for period_type in ("Historical", "LTM", "Forecast", "Unknown"):
            candidates = [point for point in points if point["period_type"] == period_type]
            if candidates:
                return candidates[-1]
        return None

    def aligned(self, *items: str) -> list[tuple[str | None, str, list[dict[str, Any]]]]:
        """Return periods present for every requested item without filling gaps."""
        if not items:
            return []
        maps = []
        for item in items:
            maps.append({
                (p.get("period"), p.get("period_type")): p for p in self.unique_points(item)
            })
        keys = set(maps[0])
        for mapping in maps[1:]:
            keys &= set(mapping)
        return [
            (key[0], str(key[1]), [mapping[key] for mapping in maps])
            for key in sorted(keys, key=lambda value: _period_key(value[0], value[1]))
        ]

    @staticmethod
    def source_inputs(points: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for point in points:
            result.append({
                "standard_name": point.get("standard_name"),
                "value": point.get("value"),
                "period": point.get("period"),
                "period_type": point.get("period_type"),
                "status": point.get("status"),
                "detected_label": point.get("detected_label"),
                "source_sheet": point.get("source_sheet"),
                "source_row": point.get("source_row"),
                "source_column": point.get("source_column"),
                "confidence": point.get("confidence"),
            })
        return result

    def summary(self, item: str) -> dict[str, Any]:
        points = self.unique_points(item)
        historical = [p for p in points if p["period_type"] == "Historical"]
        forecast = [p for p in points if p["period_type"] == "Forecast"]
        return {
            "item": item,
            "historical_start": historical[0]["period"] if historical else None,
            "historical_end": historical[-1]["period"] if historical else None,
            "historical_latest": historical[-1]["value"] if historical else None,
            "forecast_start": forecast[0]["period"] if forecast else None,
            "forecast_end": forecast[-1]["period"] if forecast else None,
            "forecast_latest": forecast[-1]["value"] if forecast else None,
            "status": "Available" if points else "NOT_AVAILABLE",
        }


class FinancialAnalysisEngine:
    """Run calculation-only analysis over a MappingResult and ValidationResult."""

    def __init__(self, mapped_data: Any, validation_results: Any = None) -> None:
        self.context = AnalysisContext(mapped_data, validation_results)
        self.mapped_data = mapped_data
        self.validation_results = validation_results
        self._result: AnalysisResult | None = None

    def analyze(self) -> AnalysisResult:
        if self._result is not None:
            return self._result
        # Local imports avoid making the analysis submodules depend on import order.
        from .cash_flow import cash_flow_analysis
        from .growth import growth_analysis
        from .leverage import leverage_analysis
        from .liquidity import liquidity_analysis
        from .profitability import profitability_analysis
        from .validation import analysis_flags, build_insights

        metrics: list[AnalysisMetric] = []
        for calculator in (
            growth_analysis, profitability_analysis, liquidity_analysis,
            leverage_analysis, cash_flow_analysis,
        ):
            metrics.extend(calculator(self.context))
        expected_metrics = {
            "revenue_growth", "revenue_cagr", "ebitda_growth", "ebitda_cagr",
            "ebit_growth", "ebit_cagr", "pat_growth", "pat_cagr",
            "eps_growth", "eps_cagr", "free_cash_flow_growth", "free_cash_flow_cagr",
            "gross_margin", "ebitda_margin", "ebit_margin", "net_margin", "roe", "roa", "roic",
            "current_ratio", "cash_ratio", "working_capital_calculated",
            "debt_to_equity", "debt_to_assets", "debt_to_capital", "net_debt_to_ebitda",
            "operating_cash_flow_margin", "free_cash_flow_margin",
            "operating_cash_flow_conversion", "free_cash_flow_conversion",
            "free_cash_flow_calculated", "eps", "market_price", "eps_calculated",
            "market_capitalization_calculated",
        }
        existing = {metric.metric_name for metric in metrics}
        for name in sorted(expected_metrics - existing):
            category = (
                "Growth" if name.endswith(("_growth", "_cagr")) else
                "Profitability" if name in {"gross_margin", "ebitda_margin", "ebit_margin", "net_margin", "roe", "roa", "roic"} else
                "Liquidity" if name in {"current_ratio", "cash_ratio", "working_capital_calculated"} else
                "Leverage" if name in {"debt_to_equity", "debt_to_assets", "debt_to_capital", "net_debt_to_ebitda"} else
                "Per Share" if name in {"eps", "market_price", "eps_calculated", "market_capitalization_calculated"} else
                "Cash Flow"
            )
            metrics.append(AnalysisMetric(
                name, category, None, "Unknown", None, "NOT_AVAILABLE",
                "Required source inputs are unavailable", flags=["Missing source inputs"],
            ))
        items = (
            "revenue", "ebitda", "ebit", "pat", "eps", "free_cash_flow",
            "operating_cash_flow", "cash", "debt", "equity", "total_assets",
        )
        summaries = [self.context.summary(item) for item in items]
        flags = analysis_flags(self.context, metrics)
        insights = build_insights(metrics, flags, summaries)
        self._result = AnalysisResult(metrics, summaries, flags, insights)
        return self._result

    run = analyze
    calculate = analyze

    def to_dataframe(self) -> pd.DataFrame:
        return self.analyze().to_dataframe()

    def summary_to_dataframe(self) -> pd.DataFrame:
        return self.analyze().summary_to_dataframe()

    def flags_to_dataframe(self) -> pd.DataFrame:
        return self.analyze().flags_to_dataframe()

    def to_dict(self) -> dict[str, Any]:
        result = self.analyze()
        return {
            "metrics": [metric.to_dict() for metric in result.metrics],
            "summaries": result.summaries,
            "flags": result.flags,
            "insights": result.insights,
        }

    @property
    def result(self) -> AnalysisResult:
        return self.analyze()

    @property
    def metrics(self) -> list[AnalysisMetric]:
        return self.analyze().metrics

    @property
    def summaries(self) -> list[dict[str, Any]]:
        return self.analyze().summaries

    @property
    def flags(self) -> list[dict[str, Any]]:
        return self.analyze().flags

    @property
    def insights(self) -> list[str]:
        return self.analyze().insights
