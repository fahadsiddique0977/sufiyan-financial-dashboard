from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_WORKBOOK_PATH = str(Path(__file__).resolve().parents[2] / "Larsen-Toubro-financial-model.xlsx")

CLASSIFICATION_RULES = {
    "Cover": (("cover",), ("cover page", "company profile", "about company")),
    "Historical Financial Statements": (
        ("historical fs", "historical financial", "historical statement"),
        ("historical financial statement", "historical fs", "balance sheet", "income statement"),
    ),
    "Ratio Analysis": (("ratio analysis", "ratios"), ("ratio", "margin", "return on equity", "return on assets")),
    "Common Size Financial Statements": (
        ("common size statement", "common size"),
        ("common size", "percentage of sales", "% of revenue"),
    ),
    "Forecast / Financial Forecast": (
        ("forecasting", "forecast", "projections"),
        ("forecast", "projected", "estimates", "assumptions"),
    ),
    "WACC / Cost of Capital": (("wacc",), ("weighted average cost of capital", "cost of capital", "wacc")),
    "Growth Analysis": (("intrinsic growth", "growth analysis"), ("intrinsic growth", "growth rate", "roic")),
    "DCF Valuation": (("dcf", "discounted cash flow"), ("discounted cash flow", "pv of fcff", "terminal value")),
    "Peer Comparison": (("peer comparison", "peer comps", "peer analysis"), ("peer comparison", "peer median", "peer group")),
    "Comparable Companies Valuation": (
        ("comps valuation", "comparable", "comps"),
        ("comparable company", "ev/ebitda", "ev/revenue", "p/e"),
    ),
    "Football Field Valuation": (("football field", "footbal feild"), ("football field", "valuation range")),
    "Risk Analysis": (("var", "value at risk"), ("value at risk", "simulated return", "sorted return")),
    "DuPont Analysis": (("duopont", "duopoint", "dupont"), ("dupont", "du pont", "asset turnover")),
    "Credit Risk Analysis": (("altman z score", "altman"), ("altman z score", "working capital/assets", "credit risk")),
    "Raw Financial Statements": (("raw fs", "raw financial"), ("equity capital", "reserves", "borrowings")),
    "Cash Flow Data": (("cash flow data", "cash flow"), ("cash from operating activity", "cash flow", "capital expenditure")),
    "Financial Data": (("data sheet", "financial data"), ("number of shares", "current price", "face value")),
    "Market / Stock Data": (
        ("list of stock", "stock list", "market data"),
        ("company_name", "industry", "stock exchange", "ticker"),
    ),
    "Market Price History": (
        ("weekly historical data", "market price history", "price history"),
        ("weekly return", "adj close", "historical price"),
    ),
}


@dataclass
class SheetScanResult:
    name: str
    row_count: int
    column_count: int
    detected_type: str
    preview: pd.DataFrame
    dataframe: pd.DataFrame


@dataclass
class WorkbookScanResult:
    workbook_path: str
    sheet_names: List[str]
    sheets: List[SheetScanResult] = field(default_factory=list)
    company_name: str | None = None

    @property
    def sheet_count(self) -> int:
        return len(self.sheet_names)

    def get_sheet(self, sheet_name: str) -> SheetScanResult | None:
        for sheet in self.sheets:
            if sheet.name == sheet_name:
                return sheet
        return None


def _content_text(dataframe: pd.DataFrame | None) -> str:
    if dataframe is None or dataframe.empty:
        return ""
    relevant_cells = dataframe.iloc[:30, :30].to_numpy().flatten()
    return " ".join(str(value) for value in relevant_cells if pd.notna(value)).lower()


def detect_sheet_type(sheet_name: str, dataframe: pd.DataFrame | None = None) -> str:
    normalized_name = (sheet_name or "").strip().lower()
    content = _content_text(dataframe)
    scores: dict[str, int] = {}

    for label, (name_keywords, content_keywords) in CLASSIFICATION_RULES.items():
        name_score = sum(5 for keyword in name_keywords if keyword in normalized_name)
        content_score = sum(1 for keyword in content_keywords if keyword in content)
        if name_score or content_score:
            scores[label] = name_score + content_score

    if not scores:
        return "Other / Unclassified"

    name_matches = {
        label: sum(5 for keyword in rules[0] if keyword in normalized_name)
        for label, rules in CLASSIFICATION_RULES.items()
        if any(keyword in normalized_name for keyword in rules[0])
    }
    content_matches = {
        label: sum(1 for keyword in rules[1] if keyword in content)
        for label, rules in CLASSIFICATION_RULES.items()
        if any(keyword in content for keyword in rules[1])
    }

    name_label = max(name_matches, key=name_matches.get) if name_matches else None
    content_label = max(content_matches, key=content_matches.get) if content_matches else None

    # A recognized name is the default; only strong, multi-keyword content
    # evidence can override it. This prevents shared financial headers from
    # misclassifying otherwise well-labelled worksheets.
    if content_label and (
        (not name_label and content_matches[content_label] >= 2)
        or (name_label and content_label != name_label and content_matches[content_label] >= 6)
    ):
        return content_label
    return name_label or content_label or "Other / Unclassified"


def _sheet_preview(dataframe: pd.DataFrame, max_rows: int = 10, max_columns: int = 10) -> pd.DataFrame:
    return dataframe.iloc[:max_rows, :max_columns].copy()


def detect_company_name(workbook_data: dict[str, pd.DataFrame]) -> str | None:
    """Find a company name from common label/value pairs without hard-coding a company."""
    exact_labels = {"company name", "company name:", "issuer", "entity"}
    fallback_labels = {"company"}
    for labels in (exact_labels, fallback_labels):
        for dataframe in workbook_data.values():
            for row_index in range(min(len(dataframe), 40)):
                for column_index in range(min(dataframe.shape[1], 20)):
                    value = dataframe.iat[row_index, column_index]
                    if pd.isna(value) or str(value).strip().lower() not in labels:
                        continue
                    for next_column in range(column_index + 1, min(dataframe.shape[1], column_index + 4)):
                        candidate = dataframe.iat[row_index, next_column]
                        candidate_text = str(candidate).strip() if pd.notna(candidate) else ""
                        if candidate_text and candidate_text.lower() not in {"ticker", "n/a", "na"}:
                            return candidate_text
    return None


def scan_workbook(workbook_path: str | Path) -> WorkbookScanResult:
    workbook_file = Path(workbook_path)
    if not workbook_file.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_file}")

    workbook_data = pd.read_excel(
        workbook_file,
        sheet_name=None,
        header=None,
        engine="openpyxl",
    )
    sheet_names = list(workbook_data.keys())
    sheet_results: List[SheetScanResult] = []

    for sheet_name, dataframe in workbook_data.items():
        row_count, column_count = dataframe.shape
        sheet_results.append(
            SheetScanResult(
                name=sheet_name,
                row_count=int(row_count),
                column_count=int(column_count),
                detected_type=detect_sheet_type(sheet_name, dataframe),
                preview=_sheet_preview(dataframe),
                dataframe=dataframe,
            )
        )

    return WorkbookScanResult(
        workbook_path=str(workbook_file),
        sheet_names=sheet_names,
        sheets=sheet_results,
        company_name=detect_company_name(workbook_data),
    )
