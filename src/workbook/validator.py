from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .mapper import MappingResult


VALIDATION_STATUSES = ("PASS", "WARNING", "FAIL", "NOT_AVAILABLE")
SEVERITIES = ("INFO", "MEDIUM", "HIGH", "CRITICAL")

REQUIRED_ITEMS = (
    "revenue",
    "ebitda",
    "ebit",
    "pat",
    "eps",
    "cash",
    "debt",
    "total_assets",
    "total_liabilities",
    "equity",
    "operating_cash_flow",
    "capex",
    "working_capital",
    "free_cash_flow",
    "shares_outstanding",
    "market_price",
)


@dataclass(frozen=True)
class ValidationCheck:
    check_name: str
    category: str
    status: str
    value: object
    expected: object
    difference: object
    source: str
    message: str
    severity: str


@dataclass
class ValidationResult:
    checks: list[ValidationCheck]
    dcf_readiness: str
    dcf_missing_inputs: list[str]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([check.__dict__ for check in self.checks])

    def count(self, status: str) -> int:
        return sum(check.status == status for check in self.checks)

    @property
    def overall_status(self) -> str:
        if self.count("FAIL"):
            return "🔴 Critical Issues"
        if self.count("WARNING"):
            return "🟡 Review Required"
        return "🟢 Strong"


def _records_by_item(mapping: MappingResult) -> dict[str, pd.DataFrame]:
    dataframe = mapping.to_dataframe()
    if dataframe.empty:
        return {}
    return {
        name: group[group["status"] != "Not found"].copy()
        for name, group in dataframe.groupby("standard_name", dropna=False)
    }


def _available(records: dict[str, pd.DataFrame], item: str) -> bool:
    return item in records and not records[item].empty and records[item]["value"].notna().any()


def _first_value(records: dict[str, pd.DataFrame], item: str) -> float | None:
    if not _available(records, item):
        return None
    values = pd.to_numeric(records[item]["value"], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def _check(
    checks: list[ValidationCheck],
    name: str,
    category: str,
    status: str,
    message: str,
    *,
    value: object = None,
    expected: object = None,
    difference: object = None,
    source: str = "",
    severity: str = "INFO",
) -> None:
    checks.append(
        ValidationCheck(
            check_name=name,
            category=category,
            status=status,
            value=value,
            expected=expected,
            difference=difference,
            source=source,
            message=message,
            severity=severity,
        )
    )


def _comparable_values(records: dict[str, pd.DataFrame], first: str, second: str) -> tuple[float, float] | None:
    if not (_available(records, first) and _available(records, second)):
        return None
    left = pd.to_numeric(records[first]["value"], errors="coerce").dropna()
    right = pd.to_numeric(records[second]["value"], errors="coerce").dropna()
    if left.empty or right.empty:
        return None
    return float(left.iloc[0]), float(right.iloc[0])


def _aligned_values(
    records: dict[str, pd.DataFrame],
    first: str,
    second: str,
    third: str,
) -> tuple[float, float, float, str] | None:
    """Return values for the same mapped period, rather than unrelated first rows."""
    if not all(_available(records, item) for item in (first, second, third)):
        return None

    left = records[first].copy()
    middle = records[second].copy()
    right = records[third].copy()
    join_columns = ["period", "period_type"]
    merged = left.merge(middle, on=join_columns, suffixes=("_first", "_second"))
    merged = merged.merge(right, on=join_columns)
    for _, row in merged.iterrows():
        values = [
            pd.to_numeric(row.get("value_first"), errors="coerce"),
            pd.to_numeric(row.get("value_second"), errors="coerce"),
            pd.to_numeric(row.get("value"), errors="coerce"),
        ]
        if all(pd.notna(value) for value in values):
            return float(values[0]), float(values[1]), float(values[2]), str(row["period"])
    return None


def _validate_completeness(records: dict[str, pd.DataFrame], checks: list[ValidationCheck]) -> None:
    for item in REQUIRED_ITEMS:
        display_name = item.replace("_", " ").title()
        if _available(records, item):
            _check(checks, f"{display_name} mapping", "Data Completeness", "PASS",
                   f"{display_name} is available in the standardized mapping.", severity="INFO")
        else:
            _check(checks, f"{display_name} mapping", "Data Completeness", "NOT_AVAILABLE",
                   f"{display_name} is not available; missing data was not replaced with zero.",
                   severity="MEDIUM")


def _validate_statement_checks(records: dict[str, pd.DataFrame], checks: list[ValidationCheck]) -> None:
    balance = _aligned_values(records, "total_assets", "total_liabilities", "equity")
    if balance:
        assets, liabilities, equity, period = balance
        difference = assets - (liabilities + equity)
        tolerance = max(abs(assets) * 0.01, 1.0)
        status = "PASS" if abs(difference) <= tolerance else "WARNING" if abs(difference) <= tolerance * 5 else "FAIL"
        _check(checks, "Balance Sheet consistency", "Financial Statement Consistency", status,
               f"Assets compared with liabilities plus shareholders' equity for {period}.",
               value=assets, expected=liabilities + equity, difference=difference,
               source="Mapped Total Assets, Total Liabilities, Shareholders Equity",
               severity="INFO" if status == "PASS" else "HIGH")
    else:
        _check(checks, "Balance Sheet consistency", "Financial Statement Consistency", "NOT_AVAILABLE",
               "Insufficient data for validation.", severity="MEDIUM")

    fcf_values = _aligned_values(records, "operating_cash_flow", "capex", "free_cash_flow")
    if fcf_values:
        capex_rows = records["capex"]
        if capex_rows["confidence"].astype(str).eq("Low").all():
            _check(
                checks,
                "Free Cash Flow construction",
                "Financial Statement Consistency",
                "NOT_AVAILABLE",
                "Capex is only available from a low-confidence source match; "
                "the FCF consistency check was not used to avoid a misleading failure.",
                source="Mapped Operating Cash Flow, Capex, Free Cash Flow",
                severity="MEDIUM",
            )
            return
        operating_cash_flow, capex, mapped_fcf, period = fcf_values
        calculated_fcf = operating_cash_flow - abs(capex)
        difference = calculated_fcf - mapped_fcf
        tolerance = max(abs(calculated_fcf) * 0.01, 1.0)
        status = (
            "PASS"
            if abs(difference) <= tolerance
            else "WARNING"
            if abs(difference) <= tolerance * 5
            else "FAIL"
        )
        _check(checks, "Free Cash Flow construction", "Financial Statement Consistency", status,
               "Compared Operating Cash Flow minus absolute Capital Expenditure with mapped "
               f"Free Cash Flow for the aligned {period} period.",
               value=calculated_fcf, expected=mapped_fcf, difference=difference,
               source=f"Mapped Operating Cash Flow, Capex, Free Cash Flow ({period})",
               severity="INFO" if status == "PASS" else "MEDIUM" if status == "WARNING" else "HIGH")
    else:
        _check(checks, "Free Cash Flow construction", "Financial Statement Consistency", "NOT_AVAILABLE",
               "Insufficient data for FCF construction validation.", severity="MEDIUM")


def _validate_periods(records: dict[str, pd.DataFrame], checks: list[ValidationCheck]) -> None:
    period_rows = []
    for item, dataframe in records.items():
        for _, row in dataframe.iterrows():
            if pd.notna(row.get("period")):
                period_rows.append((item, str(row["period"]), str(row.get("period_type", "Unknown"))))
    if not period_rows:
        _check(checks, "Period identification", "Period Consistency", "NOT_AVAILABLE",
               "No recognizable financial periods were found in mapped records.", severity="MEDIUM")
        return

    period_keys = [
        (item, period, period_type)
        for item, period, period_type in period_rows
        if period_type != "LTM"
    ]
    duplicate_keys = sorted({key for key in period_keys if period_keys.count(key) > 1})
    unknown = sum(period_type == "Unknown" for _, _, period_type in period_rows)
    if duplicate_keys:
        duplicate_sources = []
        for item, period, period_type in duplicate_keys:
            dataframe = records[item]
            matching = dataframe[
                (dataframe["period"].astype(str) == period)
                & (dataframe["period_type"].astype(str) == period_type)
            ]
            locations = matching.apply(
                lambda row: f"{row.get('source_sheet')}!{row.get('source_column')}{row.get('source_row')}",
                axis=1,
            ).tolist()
            duplicate_sources.append(f"{item} {period} {period_type}: {', '.join(locations)}")
        _check(checks, "Duplicate periods", "Period Consistency", "WARNING",
               "Duplicate mappings exist for the same standard item, period, and period type: "
               + "; ".join(duplicate_sources),
               value=len(duplicate_keys), expected="No duplicate item-period keys", severity="MEDIUM")
    else:
        _check(checks, "Duplicate periods", "Period Consistency", "PASS",
               "No duplicate mapped periods were detected.", severity="INFO")
    if unknown:
        _check(checks, "Historical / forecast period labels", "Period Consistency", "WARNING",
               f"{unknown} mapped period values have an unknown period type; no assumption was applied.",
               value=unknown, expected=0, severity="MEDIUM")
    else:
        _check(checks, "Historical / forecast period labels", "Period Consistency", "PASS",
               "Mapped periods have identifiable period types.", severity="INFO")


def _validate_mapping_accuracy(mapping: MappingResult, checks: list[ValidationCheck]) -> None:
    dataframe = mapping.to_dataframe()
    if dataframe.empty:
        return
    suspicious = []
    for _, row in dataframe[dataframe["status"] != "Not found"].iterrows():
        label = str(row.get("detected_label") or "").lower()
        item = str(row.get("standard_name") or "")
        if item == "shares_outstanding" and any(term in label for term in ("eps", "price")):
            suspicious.append("Shares Outstanding")
        if item == "market_price" and any(term in label for term in ("revenue", "eps", "share")):
            suspicious.append("Market Price")
        if item == "revenue" and any(term in label for term in ("margin", "rate", "price")):
            suspicious.append("Revenue")
        if item == "tax" and "rate" in label:
            suspicious.append("Tax")
    if suspicious:
        _check(checks, "Mapping accuracy", "Mapping Accuracy", "WARNING",
               f"Potentially suspicious mapped labels: {', '.join(sorted(set(suspicious)))}.",
               source="Standardized mapping", severity="HIGH")
    else:
        _check(checks, "Mapping accuracy", "Mapping Accuracy", "PASS",
               "No prohibited label-to-item combinations were detected.", severity="INFO")


def _validate_values(records: dict[str, pd.DataFrame], checks: list[ValidationCheck]) -> None:
    for item in ("shares_outstanding", "market_price"):
        value = _first_value(records, item)
        if value is None:
            continue
        display_name = item.replace("_", " ").title()
        status = "FAIL" if value <= 0 else "PASS"
        _check(checks, f"{display_name} sign check", "Financial Logic Checks", status,
               f"{display_name} must be positive." if status == "FAIL" else f"{display_name} is positive.",
               value=value, expected="> 0", severity="CRITICAL" if status == "FAIL" else "INFO")

    for item in ("revenue", "total_assets"):
        value = _first_value(records, item)
        if value is None:
            continue
        if value < 0:
            _check(checks, f"{item.replace('_', ' ').title()} abnormal sign", "Abnormal Value Detection", "WARNING",
                   "Negative value detected; this can be legitimate but requires analyst review.",
                   value=value, expected="Usually non-negative", severity="MEDIUM")
        else:
            _check(checks, f"{item.replace('_', ' ').title()} sign check", "Abnormal Value Detection", "PASS",
                   "No negative value detected.", value=value, severity="INFO")


def _validate_units(mapping: MappingResult, checks: list[ValidationCheck]) -> None:
    dataframe = mapping.to_dataframe()
    units = dataframe["unit"].dropna().astype(str).str.lower().unique().tolist() if not dataframe.empty else []
    if len(units) <= 1:
        _check(checks, "Unit consistency", "Unit Consistency", "NOT_AVAILABLE",
               "Units were not consistently identified in the source workbook; no conversion was performed.",
               value=units, expected="Known consistent unit", severity="MEDIUM")
    else:
        _check(checks, "Unit consistency", "Unit Consistency", "WARNING",
               f"Multiple source units were detected: {', '.join(units)}.",
               value=units, expected="One consistent unit", severity="HIGH")


def _dcf_readiness(records: dict[str, pd.DataFrame]) -> tuple[str, list[str]]:
    required = {
        "Forecast Revenue": "revenue",
        "Forecast EBITDA / EBIT": "ebitda",
        "Tax": "tax",
        "D&A": "depreciation",
        "Capex": "capex",
        "Working Capital": "working_capital",
        "FCF": "free_cash_flow",
        "WACC": "wacc",
        "Terminal Growth": "terminal_growth",
        "Cash": "cash",
        "Debt": "debt",
        "Shares Outstanding": "shares_outstanding",
        "Market Price": "market_price",
    }
    missing = [label for label, item in required.items() if not _available(records, item)]
    if not missing:
        return "Ready", []
    if len(missing) < len(required):
        return "Partially Ready", missing
    return "Not Ready", missing


def validate_mapping(mapping: MappingResult) -> ValidationResult:
    records = _records_by_item(mapping)
    checks: list[ValidationCheck] = []
    _validate_completeness(records, checks)
    _validate_statement_checks(records, checks)
    _validate_periods(records, checks)
    _validate_mapping_accuracy(mapping, checks)
    _validate_values(records, checks)
    _validate_units(mapping, checks)
    readiness, missing = _dcf_readiness(records)
    _check(checks, "DCF input readiness", "Valuation Input Readiness",
           "PASS" if readiness == "Ready" else "WARNING" if readiness == "Partially Ready" else "NOT_AVAILABLE",
           f"DCF input readiness: {readiness}. No DCF calculation was performed.",
           value=readiness, expected="Ready", difference=", ".join(missing),
           source="Standardized mapping", severity="INFO" if readiness == "Ready" else "MEDIUM")
    return ValidationResult(checks=checks, dcf_readiness=readiness, dcf_missing_inputs=missing)
