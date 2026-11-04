from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any
import re

import numpy as np
import pandas as pd


@dataclass
class MappingRecord:
    standard_name: str
    display_name: str
    detected_label: str | None
    value: Any
    period: str | None
    period_type: str
    source_sheet: str | None
    source_row: int | None
    source_column: str | None
    confidence: str
    mapping_method: str
    status: str
    unit: str | None = None


@dataclass
class MappingResult:
    records: list[MappingRecord]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(record) for record in self.records])

    def get(self, standard_name: str) -> list[MappingRecord]:
        return [
            record
            for record in self.records
            if record.standard_name == standard_name
        ]


# ================================================================
# LINE ITEM DEFINITIONS
# ================================================================

LINE_ITEM_RULES: dict[str, dict[str, Any]] = {

    "revenue": {
        "display": "Revenue",
        "aliases": [
            "revenue",
            "sales",
            "net sales",
            "total revenue",
            "revenue from operations",
        ],
    },

    "cogs": {
        "display": "COGS",
        "aliases": [
            "cogs",
            "cost of goods sold",
            "cost of revenue",
            "cost of sales",
        ],
    },

    "gross_profit": {
        "display": "Gross Profit",
        "aliases": [
            "gross profit",
            "gross margin",
        ],
    },

    "ebitda": {
        "display": "EBITDA",
        "aliases": [
            "ebitda",
            "earnings before interest tax depreciation amortization",
        ],
    },

    "ebit": {
        "display": "EBIT",
        "aliases": [
            "ebit",
            "operating profit",
            "operating income",
        ],
    },

    "ebt": {
        "display": "EBT",
        "aliases": [
            "ebt",
            "profit before tax",
            "profit before income tax",
            "pbt",
        ],
    },

    "tax": {
        "display": "Tax",
        "aliases": [
            "tax",
            "income tax",
            "tax expense",
            "provision for tax",
        ],
    },

    "pat": {
        "display": "Net Income / PAT",
        "aliases": [
            "pat",
            "net income",
            "net profit",
            "profit after tax",
            "profit for the year",
        ],
    },

    "eps": {
        "display": "EPS",
        "aliases": [
            "eps",
            "earnings per share",
            "basic eps",
            "diluted eps",
        ],
    },

    "cash": {
        "display": "Cash",
        "aliases": [
            "cash",
            "cash and cash equivalents",
            "cash & cash equivalents",
            "cash equivalents",
        ],
    },

    "debt": {
        "display": "Debt",
        "aliases": [
            "debt",
            "borrowings",
            "total borrowings",
            "interest bearing debt",
        ],
    },

    "total_debt": {
        "display": "Total Debt",
        "aliases": [
            "total debt",
            "total borrowings",
            "total debt outstanding",
        ],
    },

    "net_debt": {
        "display": "Net Debt",
        "aliases": [
            "net debt",
            "net borrowings",
        ],
    },

    "total_assets": {
        "display": "Total Assets",
        "aliases": [
            "total assets",
        ],
    },

    "current_assets": {
        "display": "Current Assets",
        "aliases": [
            "current assets",
            "total current assets",
        ],
    },

    "current_liabilities": {
        "display": "Current Liabilities",
        "aliases": [
            "current liabilities",
            "total current liabilities",
        ],
    },

    "total_liabilities": {
        "display": "Total Liabilities",
        "aliases": [
            "total liabilities",
            "liabilities total",
        ],
    },

    "equity": {
        "display": "Shareholders Equity",
        "aliases": [
            "shareholders equity",
            "shareholders' equity",
            "total equity",
            "shareholder equity",
            "owners equity",
            "owners' equity",
            "net worth",
        ],
    },

    "operating_cash_flow": {
        "display": "Operating Cash Flow",
        "aliases": [
            "operating cash flow",
            "cash flow from operating activities",
            "net cash from operating activities",
        ],
    },

    "capex": {
        "display": "Capital Expenditure / Capex",
        "aliases": [
            "capex",
            "capital expenditure",
            "capital expenditures",
            "purchase of property plant equipment",
            "purchase of fixed assets",
        ],
    },

    "depreciation": {
        "display": "Depreciation",
        "aliases": [
            "depreciation",
            "depreciation expense",
            "depreciation and amortization",
            "depreciation & amortization",
            "d&a",
        ],
    },

    "working_capital": {
        "display": "Working Capital",
        "aliases": [
            "working capital",
            "net working capital",
        ],
    },

    "change_in_working_capital": {
        "display": "Change in Working Capital",
        "aliases": [
            "change in working capital",
            "change in net working capital",
            "increase decrease in working capital",
        ],
    },

    "free_cash_flow": {
        "display": "Free Cash Flow",
        "aliases": [
            "free cash flow",
            "fcf",
            "free cash flows",
        ],
    },

    "shares_outstanding": {
        "display": "Shares Outstanding",
        "aliases": [
            "shares outstanding",
            "number of shares outstanding",
            "number of shares",
            "equity shares outstanding",
        ],
    },

    "market_price": {
        "display": "Market Price",
        "aliases": [
            "market price",
            "share price",
            "current share price",
            "current price",
            "stock price",
            "cmp",
        ],
    },

    "market_capitalization": {
        "display": "Market Capitalization",
        "aliases": [
            "market capitalization",
            "market cap",
        ],
    },

    "wacc": {
        "display": "WACC",
        "aliases": [
            "wacc",
            "weighted average cost of capital",
        ],
    },

    "terminal_growth": {
        "display": "Terminal Growth",
        "aliases": [
            "terminal growth",
            "terminal growth rate",
            "perpetuity growth",
            "perpetuity growth rate",
        ],
    },
}


# ================================================================
# SHEET PRIORITY
# ================================================================

SHEET_PRIORITY = [
    "historical financial statements",
    "historical fs",
    "raw fs",
    "financial statements",
    "balance sheet",
    "cash flow",
    "cash flow data",
    "financial data",
    "forecast",
    "wacc",
    "dcf",
    "valuation",
    "market",
    "market price history",
    "automation control",
]


# ================================================================
# HELPERS
# ================================================================

def _normalize(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("’", "'")

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _sheet_priority(sheet_name: str | None) -> int:
    normalized = _normalize(sheet_name)

    for index, name in enumerate(SHEET_PRIORITY):
        if normalized == _normalize(name):
            return index

    return len(SHEET_PRIORITY)


def _period_type(period: Any) -> str:
    if period is None:
        return "Unknown"

    text = str(period).strip().lower()

    if "ltm" in text or "ttm" in text:
        return "LTM"

    # FY25E / FY2025E / 2025E
    if re.search(r"(fy)?\d{2,4}e$", text):
        return "Forecast"

    if any(word in text for word in [
        "estimate",
        "forecast",
        "projected",
        "projection",
    ]):
        return "Forecast"

    # FY20 / FY2020 / 2020
    if re.match(r"^fy\d{2,4}$", text):
        return "Historical"

    if re.match(r"^20\d{2}$", text):
        return "Historical"

    if any(word in text for word in [
        "actual",
        "historical",
    ]):
        return "Historical"

    return "Unknown"


def _is_numeric(value: Any) -> bool:
    if value is None or pd.isna(value) or isinstance(
        value,
        (date, datetime, pd.Timestamp, np.datetime64),
    ):
        return False

    if isinstance(value, str) and not value.strip():
        return False

    try:
        number = pd.to_numeric(
            pd.Series([value]),
            errors="coerce"
        ).iloc[0]

        return pd.notna(number)

    except Exception:
        return False


def _alias_match(label: str, alias: str) -> tuple[bool, str]:

    normalized_label = _normalize(label)
    normalized_alias = _normalize(alias)

    if not normalized_label or not normalized_alias:
        return False, ""

    # ------------------------------------------------------------
    # EXACT MATCH
    # ------------------------------------------------------------

    if normalized_label == normalized_alias:
        return True, "exact label"

    # ------------------------------------------------------------
    # FALSE-MATCH PROTECTION
    # ------------------------------------------------------------

    protected_combinations = {

        "debt": [
            "net debt",
            "cost of debt",
            "debt to equity",
            "debt to assets",
            "debt to capital",
            "net debt to ebitda",
        ],

        "equity": [
            "equity risk premium",
            "return on equity",
            "equity beta",
        ],
    }

    blocked = protected_combinations.get(
        normalized_alias,
        []
    )

    if any(
        normalized_label == item
        or normalized_label.startswith(item + " ")
        for item in blocked
    ):
        return False, ""

    # ------------------------------------------------------------
    # QUALIFIED EXACT MATCH
    # ------------------------------------------------------------

    allowed_suffixes = {
        "amount",
        "value",
        "balance",
        "total",
        "expense",
    }

    if normalized_label.startswith(
        normalized_alias + " "
    ):

        remainder = normalized_label[
            len(normalized_alias):
        ].strip()

        if remainder in allowed_suffixes:
            return True, "qualified exact label"

    # ------------------------------------------------------------
    # DEBT SPECIAL PROTECTION
    # ------------------------------------------------------------

    if normalized_alias == "debt":
        return False, ""

    # ------------------------------------------------------------
    # GENERAL CONTEXTUAL MATCH
    # ------------------------------------------------------------

    if f" {normalized_alias} " in f" {normalized_label} ":

        blocked_terms = {
            "risk premium",
            "growth rate",
            "margin",
            "ratio",
            "percentage",
            "percent",
            "rate",
            "cost of",
            "net debt",
        }

        if not any(
            term in normalized_label
            for term in blocked_terms
        ):
            return True, "contextual label match"

    return False, ""


def _extract_period_headers(
    df: pd.DataFrame
) -> dict[int, str]:

    headers: dict[int, str] = {}

    for col_idx in range(df.shape[1]):

        for row_idx in range(min(30, df.shape[0])):

            value = df.iat[row_idx, col_idx]

            if value is None or pd.isna(value):
                continue

            if isinstance(value, (date, datetime, pd.Timestamp)):
                headers[col_idx] = str(value.year)
                break

            text = str(value).strip()

            if re.match(r"^20\d{2}-\d{2}-\d{2}", text):
                headers[col_idx] = text[:4]
                break

            if re.search(
                r"\b20\d{2}\b",
                text
            ):
                headers[col_idx] = text
                break

            if re.match(
                r"^FY\d{2,4}[A-Za-z]*$",
                text,
                re.IGNORECASE,
            ):
                headers[col_idx] = text
                break

            if text.upper() in {
                "LTM",
                "TTM",
            }:
                headers[col_idx] = text
                break

    return headers


def _find_best_match(
    label: str,
    standard_name: str,
) -> tuple[bool, str, str]:

    rule = LINE_ITEM_RULES[standard_name]

    best_method = ""
    best_alias = ""

    for alias in rule["aliases"]:

        matched, method = _alias_match(
            label,
            alias
        )

        if matched:

            if method == "exact label":
                return True, alias, method

            best_alias = alias
            best_method = method

    if best_alias:
        return True, best_alias, best_method

    return False, "", ""


def _confidence(
    method: str,
    sheet_name: str | None
) -> str:

    priority = _sheet_priority(
        sheet_name
    )

    if (
        method == "exact label"
        and priority <= 3
    ):
        return "High"

    if method == "exact label":
        return "Medium"

    if method == "qualified exact label":
        return "Medium"

    return "Low"


def _row_label(
    df: pd.DataFrame,
    row_idx: int
) -> tuple[str, int]:

    if df.shape[1] == 0:
        return "", -1

    for column_idx in range(min(5, df.shape[1])):
        value = df.iat[row_idx, column_idx]
        if value is None or pd.isna(value):
            continue
        label = str(value).strip()
        if label and len(label) <= 80 and not re.search(r"[.!?;]", label):
            return label, column_idx

    return "", -1


def _get_sheet_name(sheet: Any) -> str | None:

    return (
        getattr(
            sheet,
            "sheet_name",
            None
        )
        or getattr(
            sheet,
            "name",
            None
        )
        or getattr(
            sheet,
            "title",
            None
        )
    )


def _get_sheet_dataframe(
    sheet: Any
) -> pd.DataFrame | None:

    df = getattr(
        sheet,
        "data",
        None
    )

    if df is None:
        df = getattr(
            sheet,
            "dataframe",
            None
        )

    if df is None:
        df = getattr(
            sheet,
            "preview",
            None
        )

    if df is None:
        return None

    if not isinstance(
        df,
        pd.DataFrame
    ):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return None

    return df


# ================================================================
# MAIN MAPPER
# ================================================================

def map_financial_data(
    scan: Any
) -> MappingResult:

    records: list[
        MappingRecord
    ] = []

    # Scanner compatibility
    sheets = (
        getattr(
            scan,
            "sheets",
            None
        )
        or getattr(
            scan,
            "sheet_results",
            None
        )
        or []
    )

    for standard_name, rule in (
        LINE_ITEM_RULES.items()
    ):

        candidates: list[
            MappingRecord
        ] = []

        for sheet in sheets:

            sheet_name = _get_sheet_name(
                sheet
            )

            df = _get_sheet_dataframe(
                sheet
            )

            if df is None or df.empty:
                continue

            period_headers = (
                _extract_period_headers(df)
            )

            for row_idx in range(
                df.shape[0]
            ):

                label, label_column = _row_label(
                    df,
                    row_idx
                )

                if not label:
                    continue

                matched, alias, method = (
                    _find_best_match(
                        label,
                        standard_name
                    )
                )

                if not matched:
                    continue

                for col_idx in range(label_column + 1, df.shape[1]):

                    value = df.iat[
                        row_idx,
                        col_idx
                    ]

                    if not _is_numeric(
                        value
                    ):
                        continue

                    period = (
                        period_headers.get(
                            col_idx
                        )
                    )
                    period_type = _period_type(period)
                    if _normalize(sheet_name).startswith("dcf") and period_type != "LTM":
                        period_type = "Forecast"

                    source_column = None

                    if col_idx < 26:
                        source_column = chr(
                            65 + col_idx
                        )

                    candidates.append(
                        MappingRecord(
                            standard_name=standard_name,
                            display_name=rule[
                                "display"
                            ],
                            detected_label=label,
                            value=value,
                            period=period,
                            period_type=period_type,
                            source_sheet=(
                                sheet_name
                            ),
                            source_row=(
                                row_idx + 1
                            ),
                            source_column=(
                                source_column
                            ),
                            confidence=(
                                _confidence(
                                    method,
                                    sheet_name
                                )
                            ),
                            mapping_method=(
                                method
                            ),
                            status="Mapped",
                            unit=None,
                        )
                    )

        candidates.sort(
            key=lambda record: (
                _sheet_priority(
                    record.source_sheet
                ),
                0
                if record.confidence == "High"
                else 1
                if record.confidence == "Medium"
                else 2,
                record.source_row
                or 999999,
                record.source_column
                or "",
            )
        )

        if candidates:

            records.extend(
                candidates
            )

        else:

            records.append(
                MappingRecord(
                    standard_name=(
                        standard_name
                    ),
                    display_name=(
                        rule["display"]
                    ),
                    detected_label=None,
                    value=None,
                    period=None,
                    period_type="Unknown",
                    source_sheet=None,
                    source_row=None,
                    source_column=None,
                    confidence="Unmapped",
                    mapping_method=(
                        "No supported label found"
                    ),
                    status="Not found",
                    unit=None,
                )
            )

    return MappingResult(
        records
    )
