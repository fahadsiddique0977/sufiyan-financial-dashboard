from __future__ import annotations

from typing import Any

from .financial_analysis import AnalysisMetric, AnalysisContext


GROWTH_ITEMS = (
    "revenue",
    "ebitda",
    "ebit",
    "pat",
    "eps",
    "free_cash_flow",
)


def _period_year(period: object) -> int | None:
    """
    Extract a usable year from common financial-model period labels.

    Examples:
        FY20   -> 2020
        FY21   -> 2021
        FY2020 -> 2020
        FY2025E -> 2025
        LTM2025 -> 2025
    """
    text = str(period or "").strip().upper()

    # Prefer a four-digit year.
    for start in range(len(text) - 3):
        chunk = text[start:start + 4]
        if chunk.isdigit():
            year = int(chunk)
            if 1900 <= year <= 2100:
                return year

    # Fall back to two-digit FY labels such as FY20.
    digits = "".join(character for character in text if character.isdigit())

    if len(digits) >= 2:
        year = int(digits[:2])

        # Financial-model convention: 00-49 => 2000-2049,
        # 50-99 => 1950-1999.
        return 2000 + year if year <= 49 else 1900 + year

    return None


def _metric(
    name: str,
    period: object,
    period_type: str,
    value: float | None,
    calculation: str,
    points: list[dict[str, Any]],
    flags: list[str] | None = None,
) -> AnalysisMetric:

    status = "CALCULATED" if value is not None else "NOT_AVAILABLE"

    return AnalysisMetric(
        name,
        "Growth",
        None if period is None else str(period),
        period_type,
        value,
        status,
        calculation,
        AnalysisContext.source_inputs(points),
        flags=flags or [],
    )


def growth_analysis(context: AnalysisContext) -> list[AnalysisMetric]:
    metrics: list[AnalysisMetric] = []

    for item in GROWTH_ITEMS:

        # Get unique valid points.
        points = [
            point
            for point in context.unique_points(item)
            if _period_year(point.get("period")) is not None
        ]

        # Separate Historical / Forecast / LTM.
        by_type: dict[str, list[dict[str, Any]]] = {}

        for point in points:
            period_type = str(point.get("period_type") or "Unknown")
            by_type.setdefault(period_type, []).append(point)

        for period_type, typed_points in by_type.items():

            # IMPORTANT:
            # Sort chronologically before calculating growth.
            typed_points = sorted(
                typed_points,
                key=lambda point: _period_year(point.get("period")) or 0,
            )

            # ---------------------------------------------------------
            # YEAR-OVER-YEAR GROWTH
            # ---------------------------------------------------------
            for previous, current in zip(
                typed_points,
                typed_points[1:],
            ):

                previous_value = previous.get("value")
                current_value = current.get("value")

                period = current.get("period")
                previous_period = previous.get("period")

                flags: list[str] = []
                value: float | None = None

                if previous_value is None or current_value is None:
                    flags.append("Missing value")

                elif previous_value != 0:
                    value = current_value / previous_value - 1

                    if value < 0:
                        flags.append("Negative growth")

                else:
                    flags.append("Prior period is zero")

                metrics.append(
                    _metric(
                        f"{item}_growth",
                        period,
                        period_type,
                        value,
                        (
                            f"({item}[{period}] / "
                            f"{item}[{previous_period}]) - 1"
                        ),
                        [previous, current],
                        flags,
                    )
                )

            # ---------------------------------------------------------
            # CAGR
            # ---------------------------------------------------------
            if len(typed_points) >= 2:

                first = typed_points[0]
                last = typed_points[-1]

                start_year = _period_year(first.get("period"))
                end_year = _period_year(last.get("period"))

                years = (
                    end_year - start_year
                    if start_year is not None and end_year is not None
                    else 0
                )

                start_value = first.get("value")
                end_value = last.get("value")

                value = None

                if (
                    start_value is not None
                    and end_value is not None
                ):
                    value = calculate_cagr(
                        start_value,
                        end_value,
                        years,
                    )

                flags = (
                    []
                    if value is not None
                    else [
                        "CAGR requires positive starting value "
                        "and distinct years"
                    ]
                )

                metrics.append(
                    _metric(
                        f"{item}_cagr",
                        f"{first['period']}–{last['period']}",
                        period_type,
                        value,
                        (
                            f"({item}[{last['period']}] / "
                            f"{item}[{first['period']}]) "
                            f"^ (1 / {years}) - 1"
                        ),
                        [first, last],
                        flags,
                    )
                )

    return metrics


def calculate_growth(context: AnalysisContext) -> list[AnalysisMetric]:
    return growth_analysis(context)


def calculate_cagr(
    start_value: float,
    end_value: float,
    years: float,
) -> float | None:

    if years <= 0:
        return None

    if start_value <= 0:
        return None

    if end_value < 0:
        return None

    return (end_value / start_value) ** (1 / years) - 1