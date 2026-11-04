from __future__ import annotations

import os
import tempfile
import math
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import FinancialAnalysisEngine
from src.valuation import ComparableEngine, DCFValuationEngine, DashboardData
from src.workbook.mapper import map_financial_data
from src.workbook.scanner import DEFAULT_WORKBOOK_PATH, scan_workbook
from src.workbook.validator import VALIDATION_STATUSES, validate_mapping


st.set_page_config(
    page_title="Investment Banking Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _display_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed workbook cell types before Streamlit Arrow serialization."""
    display = dataframe.copy()
    for column in display.columns:
        if pd.api.types.is_object_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: value.isoformat() if hasattr(value, "isoformat") else value
            ).astype("string")
    return display


def _format_amount(value, decimals: int = 0) -> str:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return "N/A"
    return f"{value:,.{decimals}f}"


@st.cache_data(show_spinner=False)
def _scan_cached(workbook_path: str):
    return scan_workbook(workbook_path)


@st.cache_data(show_spinner=False)
def _map_cached(workbook, mapper_version: str):
    return map_financial_data(workbook)


def _latest_mapping_value(
    mapping,
    standard_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    dataframe = mapping.to_dataframe()
    rows = dataframe[
        (dataframe["standard_name"] == standard_name)
        & dataframe["value"].notna()
    ].copy()
    if rows.empty:
        return None
    rows["numeric_value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows = rows.dropna(subset=["numeric_value"])
    if minimum is not None:
        rows = rows[rows["numeric_value"] >= minimum]
    if maximum is not None:
        rows = rows[rows["numeric_value"] <= maximum]
    if rows.empty:
        return None
    return float(rows.iloc[-1]["numeric_value"])


def _dcf_assumptions(workbook, mapping) -> tuple[float, float, str]:
    sheet = workbook.get_sheet("DCF")
    workbook_values: dict[str, float] = {}
    if sheet is not None and not sheet.dataframe.empty:
        data = sheet.dataframe
        for row_index in range(len(data)):
            labels = {
                re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
                for value in data.iloc[row_index, :5]
                if pd.notna(value)
            }
            if labels.intersection({"wacc", "weighted average cost of capital"}):
                assumption = "wacc"
            elif labels.intersection({"terminal growth", "terminal growth rate"}):
                assumption = "terminal_growth"
            else:
                continue
            for value in data.iloc[row_index, 1:5]:
                number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(number) and math.isfinite(float(number)):
                    workbook_values[assumption] = float(number)
                    break

    wacc = workbook_values.get("wacc")
    if wacc is None or not 0.0 < wacc < 1.0:
        wacc = _latest_mapping_value(mapping, "wacc", minimum=0.0, maximum=1.0)
    if wacc is None or not 0.0 < wacc < 1.0:
        wacc = 0.10

    terminal_growth = workbook_values.get("terminal_growth")
    if terminal_growth is None or not -0.5 < terminal_growth < 0.5:
        terminal_growth = _latest_mapping_value(
            mapping, "terminal_growth", minimum=-0.5, maximum=0.5
        )
    if (
        terminal_growth is None
        or not -0.5 < terminal_growth < 0.5
        or terminal_growth >= wacc
    ):
        terminal_growth = min(0.03, wacc - 0.01)

    source = (
        "DCF worksheet"
        if {"wacc", "terminal_growth"}.issubset(workbook_values)
        else "validated fallback assumptions"
    )
    return wacc, terminal_growth, source


def _points_frame(context, items, selected_periods):
    rows = []
    selected = set(selected_periods)
    for item, label in items:
        for point in context.unique_points(item):
            if point.get("period") in selected:
                rows.append(
                    {
                        "period": point.get("period"),
                        "metric": label,
                        "value": point.get("value"),
                    }
                )
    return pd.DataFrame(rows)


def _workbook_sheet(workbook, sheet_name: str, rows: int = 18):
    sheet = workbook.get_sheet(sheet_name)
    if sheet is None or sheet.dataframe.empty:
        st.info(f"{sheet_name} is not available in this workbook.")
        return
    st.caption(f"Source: {sheet_name} · {sheet.detected_type}")
    st.dataframe(
        _display_frame(sheet.dataframe.iloc[:rows, :].copy()),
        use_container_width=True,
        hide_index=True,
    )


def _find_sheet(workbook, name: str):
    target = name.strip().lower()
    return next((sheet for sheet in workbook.sheets if sheet.name.strip().lower() == target), None)


def _peer_numeric_table(workbook) -> pd.DataFrame:
    sheet = workbook.get_sheet("Peer Comparison Comps")
    if sheet is None:
        return pd.DataFrame()
    raw = sheet.dataframe
    header_rows = raw.index[
        raw.apply(lambda row: row.astype(str).str.strip().eq("Name").any(), axis=1)
    ].tolist()
    if not header_rows:
        return pd.DataFrame()
    header_row = header_rows[0]
    headers = [str(value).strip() if pd.notna(value) else "" for value in raw.iloc[header_row]]
    data = raw.iloc[header_row + 1 :].copy()
    name_column = headers.index("Name")
    serial_column = headers.index("S.No.")
    data = data[data[serial_column].map(lambda value: pd.notna(value) and str(value).strip().isdigit())]
    if data.empty:
        return pd.DataFrame()
    result = pd.DataFrame({"Company": data[name_column].astype(str).str.strip()})
    for index, header in enumerate(headers):
        if not header or header in {"Name", "S.No.", "Ticker"}:
            continue
        numeric = pd.to_numeric(data[index], errors="coerce")
        if numeric.notna().all():
            result[header] = numeric.to_numpy()
    return result.reset_index(drop=True)


def _forecast_table(workbook) -> pd.DataFrame:
    sheet = workbook.get_sheet("Forecasting")
    if sheet is None or sheet.dataframe.empty:
        return pd.DataFrame()
    raw = sheet.dataframe
    rows = []
    for row_index in range(len(raw)):
        year_value = raw.iat[row_index, 2] if raw.shape[1] > 2 else None
        if not hasattr(year_value, "year") or year_value.year < 2027 or year_value.year > 2031:
            continue
        rows.append(
            {
                "Year": year_value.year,
                "Revenue": pd.to_numeric(raw.iat[row_index, 3], errors="coerce"),
                "Revenue Growth": pd.to_numeric(raw.iat[row_index, 4], errors="coerce"),
                "EBITDA": pd.to_numeric(raw.iat[row_index, 8], errors="coerce"),
                "EBITDA Growth": pd.to_numeric(raw.iat[row_index, 9], errors="coerce"),
                "EPS": pd.to_numeric(raw.iat[row_index, 13], errors="coerce"),
                "EPS Growth": pd.to_numeric(raw.iat[row_index, 14], errors="coerce"),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.dropna(axis=1, how="all").dropna(subset=["Year"]).reset_index(drop=True)


def _football_field_table(workbook) -> pd.DataFrame:
    sheet = _find_sheet(workbook, "Footbal Feild")
    if sheet is None or sheet.dataframe.empty:
        return pd.DataFrame()
    raw = sheet.dataframe
    rows = []
    for row_index in range(len(raw)):
        label = raw.iat[row_index, 10] if raw.shape[1] > 10 else None
        values = [pd.to_numeric(raw.iat[row_index, column], errors="coerce") for column in range(11, 15)]
        if pd.notna(label) and all(pd.notna(value) for value in values):
            rows.append(
                {
                    "Scenario": str(label).strip(),
                    "Low": min(values[0], values[1]),
                    "High": max(values[2], values[3]),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["Midpoint"] = (result["Low"] + result["High"]) / 2
    return result[["Scenario", "Low", "Midpoint", "High"]]


def _render_investment_demo(workbook, mapping, validation):
    st.markdown("### Demo · Investment decision simulator")
    st.write(
        "Explore how operating assumptions change valuation. This is a transparent model exercise, "
        "not an automatic investment recommendation."
    )
    mapped_wacc = _latest_mapping_value(mapping, "wacc", minimum=0.0, maximum=1.0)
    mapped_growth = _latest_mapping_value(mapping, "terminal_growth", minimum=-0.5, maximum=0.5)
    base_wacc = min(0.20, max(0.05, mapped_wacc if mapped_wacc is not None else 0.10))
    base_terminal_growth = mapped_growth if mapped_growth is not None else 0.03
    base_terminal_growth = min(base_terminal_growth, base_wacc - 0.01)
    st.subheader("1. What-if scenario")
    inputs = st.columns(4)
    revenue_growth = inputs[0].slider("Revenue growth (%)", 0.0, 30.0, 8.0, 0.5, format="%.1f")
    ebitda_margin = inputs[1].slider("EBITDA margin (%)", 5.0, 35.0, 14.0, 0.5, format="%.1f")
    wacc_pct = inputs[2].slider("WACC (%)", 5.0, 20.0, float(base_wacc * 100), 0.5, format="%.1f")
    terminal_growth_pct = inputs[3].slider(
        "Terminal growth",
        -2.0,
        min(10.0, max(-1.5, wacc_pct - 0.5)),
        min(float(base_terminal_growth * 100), wacc_pct - 1.0),
        0.5,
        format="%.1f%%",
    )
    revenue_growth /= 100
    ebitda_margin /= 100
    wacc = wacc_pct / 100
    terminal_growth = terminal_growth_pct / 100
    dcf_table = _football_field_table(workbook)
    market_price = _latest_mapping_value(mapping, "market_price") or 0.0
    reference_values = (
        dcf_table.loc[dcf_table["Scenario"].str.strip().eq("DCF Base"), "Midpoint"]
        if not dcf_table.empty
        else pd.Series(dtype=float)
    )
    reference_value = float(reference_values.iloc[0]) if not reference_values.empty else market_price

    def scenario_value(case_wacc: float, case_growth: float, case_margin: float, case_terminal: float) -> float:
        if reference_value <= 0:
            return 0.0
        return reference_value * (1 + case_growth) / 1.08 * (case_margin / 0.14) * (
            (1 + 0.10) / (1 + case_wacc)
        ) * ((1 + case_terminal) / 1.03)

    value = scenario_value(wacc, revenue_growth, ebitda_margin, terminal_growth)
    upside = ((value - market_price) / market_price) * 100 if market_price else 0.0
    cards = st.columns(4)
    cards[0].metric("Indicative value / share", f"{value:,.2f}")
    cards[1].metric("Market price", f"{market_price:,.2f}")
    cards[2].metric("Upside / downside", f"{upside:+.1f}%")
    view = "Attractive" if upside >= 15 else "Watchlist" if upside >= 0 else "Caution"
    cards[3].metric("Model view", view)
    st.caption(
        "How to use: move the four controls above, then compare the model value with the market price. "
        "Green upside means the selected assumptions produce a value above today’s price; red means downside."
    )
    st.caption(
        f"Reference point: DCF Base midpoint from the workbook ({reference_value:,.2f}). "
        "The demo scales this reference for your selected assumptions."
    )
    st.info(
        f"If revenue grows {revenue_growth:.1%} and EBITDA margin is {ebitda_margin:.1%}, "
        f"the model estimates {value:,.2f} per share versus a market price of {market_price:,.2f}."
    )
    scenario_table = pd.DataFrame(
        [
            {"Measure": "Market price today", "Value": market_price},
            {"Measure": "Model value under your assumptions", "Value": value},
            {"Measure": "Difference per share", "Value": value - market_price},
            {"Measure": "Expected return from today", "Value": upside / 100},
        ]
    )
    scenario_display = pd.DataFrame(
        [
            {"Measure": row["Measure"], "Value": f"{row['Value']:+.1%}" if row["Measure"] == "Expected return from today" else f"{row['Value']:,.2f}"}
            for _, row in scenario_table.iterrows()
        ]
    )
    st.dataframe(_display_frame(scenario_display), use_container_width=True, hide_index=True)

    st.subheader("2. Valuation sensitivity")
    st.caption(
        "Read across for terminal growth and down for WACC. Each cell is the estimated value per share; "
        "the percentage in brackets is the upside or downside versus today’s market price."
    )
    wacc_values = [round(value, 3) for value in [0.08, 0.10, 0.12, 0.14, 0.16]]
    growth_values = [round(value, 3) for value in [0.01, 0.02, 0.03, 0.04, 0.05]]
    sensitivity = []
    for row_wacc in wacc_values:
        row = []
        for row_growth in growth_values:
            row.append(scenario_value(row_wacc, revenue_growth, ebitda_margin, row_growth))
        sensitivity.append(row)
    heatmap = go.Figure(
        go.Heatmap(
            z=sensitivity,
            x=[f"{value:.1%}" for value in growth_values],
            y=[f"{value:.1%}" for value in wacc_values],
            colorscale="RdYlGn",
            text=[
                [
                    f"{cell:,.0f}<br>({(cell / market_price - 1):+.0%})" if cell is not None else "N/A"
                    for cell in row
                ]
                for row in sensitivity
            ],
            texttemplate="%{text}",
            hovertemplate="Terminal growth: %{x}<br>WACC: %{y}<br>Value/share: %{z:,.2f}<br>Compare with market price shown above<extra></extra>",
            colorbar_title="Value/share",
        )
    )
    heatmap.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Terminal growth", yaxis_title="WACC")
    st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("3. Transparent attractiveness scorecard")
    st.caption("This is not another valuation table. It explains what is driving the view and what the customer should verify.")
    score_items = [
        ("Valuation vs market price", 2 if upside >= 15 else 1 if upside >= 0 else 0, "The selected case implies meaningful upside." if upside >= 15 else "The selected case is close to market price." if upside >= 0 else "The selected case implies downside."),
        ("Revenue growth assumption", 2 if revenue_growth >= 0.10 else 1 if revenue_growth >= 0.05 else 0, "The case assumes strong expansion." if revenue_growth >= 0.10 else "The case assumes moderate expansion." if revenue_growth >= 0.05 else "The case assumes limited expansion."),
        ("EBITDA margin assumption", 2 if ebitda_margin >= 0.15 else 1 if ebitda_margin >= 0.10 else 0, "The case assumes strong profitability." if ebitda_margin >= 0.15 else "The case assumes reasonable profitability." if ebitda_margin >= 0.10 else "The case assumes weak profitability."),
        ("Source-model reliability", 2 if validation.count("FAIL") == 0 and validation.count("WARNING") <= 2 else 1, "No material validation failure is present." if validation.count("FAIL") == 0 and validation.count("WARNING") <= 2 else "Review the data-quality warnings before relying on this case."),
    ]
    score_table = pd.DataFrame(
        [{"Decision factor": factor, "What it means": assessment, "Score": f"{score}/2"} for factor, score, assessment in score_items]
    )
    st.dataframe(_display_frame(score_table), use_container_width=True, hide_index=True)
    total_score = sum(item[1] for item in score_items)
    st.success(f"Demo score: {total_score}/8 · Indicative view: {view}")
    st.write(
        "Decision guide: **Attractive** means the selected assumptions show upside and acceptable model support. "
        "**Watchlist** means the result is mixed or depends on stronger execution. "
        "**Caution** means the selected case does not justify the current price."
    )

    st.subheader("4. Target-return calculator")
    st.write(
        "Use this after choosing a target price. Enter the amount you may invest, enter the price you think "
        "the share could reach, and select how long you expect to hold it."
    )
    calculator = st.columns(3)
    investment = calculator[0].number_input("1. Amount to invest", min_value=1000.0, value=100000.0, step=5000.0, help="Example: enter 100000 for an investment of 1 lakh.")
    target_price = calculator[1].number_input(
        "2. Your expected future price",
        min_value=0.01,
        value=max(0.01, float(value)),
        step=10.0,
        help="Use the model value above or enter your own target price.",
    )
    holding_period = calculator[2].number_input("3. Holding period (years)", min_value=1.0, max_value=20.0, value=3.0, step=1.0, help="How many years you expect to hold the shares.")
    shares = investment / market_price if market_price else 0.0
    profit = shares * (target_price - market_price)
    annualized = ((target_price / market_price) ** (1 / holding_period) - 1) if market_price and target_price else 0.0
    result_cards = st.columns(3)
    result_cards[0].metric("Estimated shares", f"{shares:,.1f}")
    result_cards[1].metric("Potential profit / loss", f"{profit:+,.0f}")
    result_cards[2].metric("Annualized return", f"{annualized:+.1%}")
    st.caption(
        "Calculation: shares = investment amount ÷ market price; potential profit/loss = shares × (target price − market price). "
        "Outputs exclude brokerage, taxes, dividends, and liquidity effects."
    )


def _render_client_dashboard(workbook, source_label, mapping, validation, analysis, overview, context):
    page = st.sidebar.radio(
        "Client dashboard",
        ["1 · Deal Overview", "2 · Operating Performance", "3 · Valuation", "4 · Risk & Confidence", "5 · Demo Simulator"],
        key="client_dashboard_page",
    )
    st.sidebar.caption("A client-ready investment banking readout")

    periods = [point["period"] for point in context.unique_points("revenue") if point.get("period")]
    selected_periods = st.sidebar.multiselect(
        "Period focus",
        periods,
        default=periods[-8:],
        key="client_dashboard_periods",
    )
    selected_periods = selected_periods or periods

    if page == "5 · Demo Simulator":
        _render_investment_demo(workbook, mapping, validation)

    elif page == "1 · Deal Overview":
        st.markdown("### Executive briefing")
        st.write(
            "A concise view of the company for management, investors, and transaction discussions. "
            "Start here to understand scale, earnings quality, cash generation, and what needs attention."
        )
        cards = st.columns(4)
        for column, label, key in (
            (cards[0], "Revenue", "revenue_latest"),
            (cards[1], "EBITDA", "ebitda_latest"),
            (cards[2], "PAT", "pat_latest"),
            (cards[3], "Free Cash Flow", "fcf_latest"),
        ):
            value = overview[key]
            column.metric(label, _format_amount(value))
        st.divider()
        st.subheader("Investment banking takeaway")
        takeaways = [
            ("Scale", f"{workbook.company_name or 'The company'} reports latest revenue of {_format_amount(overview['revenue_latest'])}."),
            ("Profitability", f"Latest EBITDA is {_format_amount(overview['ebitda_latest'])}, indicating the operating earnings base used for valuation."),
            ("Cash generation", f"Latest free cash flow is {_format_amount(overview['fcf_latest'])}; cash conversion should be discussed with management."),
            (
                "Model confidence",
                f"{overview['validation_pass']} checks pass; "
                f"{overview['validation_warning']} warning"
                f"{'s' if overview['validation_warning'] != 1 else ''} "
                f"{'requires' if overview['validation_warning'] == 1 else 'require'} review before final conclusions.",
            ),
        ]
        for heading, message in takeaways:
            st.markdown(f"**{heading}**  \n{message}")
        st.subheader("Selected-period trend")
        data = _points_frame(context, [("revenue", "Revenue"), ("ebitda", "EBITDA"), ("pat", "PAT")], selected_periods)
        if not data.empty:
            figure = go.Figure()
            for metric in data["metric"].unique():
                subset = data[data["metric"] == metric]
                figure.add_trace(go.Scatter(x=subset["period"], y=subset["value"], mode="lines+markers", name=metric))
            figure.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Amount")
            st.plotly_chart(figure, use_container_width=True)

    elif page == "2 · Operating Performance":
        st.markdown("### Operating performance")
        st.write("Use this page to explain the company’s growth, earnings trajectory, and cash conversion over the selected periods.")
        data = _points_frame(
            context,
            [("revenue", "Revenue"), ("ebitda", "EBITDA"), ("pat", "PAT"), ("free_cash_flow", "Free Cash Flow")],
            selected_periods,
        )
        if not data.empty:
            left, right = st.columns([2, 1])
            with left:
                figure = go.Figure()
                for metric in data["metric"].unique():
                    subset = data[data["metric"] == metric]
                    figure.add_trace(go.Scatter(x=subset["period"], y=subset["value"], mode="lines+markers", name=metric))
                figure.update_layout(height=420, hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Amount")
                st.plotly_chart(figure, use_container_width=True)
            with right:
                latest = data.sort_values("period").groupby("metric", as_index=False).tail(1)
                st.dataframe(_display_frame(latest[["metric", "period", "value"]]), use_container_width=True, hide_index=True)
        st.info("Customer message: revenue shows scale, EBITDA shows operating profitability, PAT shows earnings after financing and tax, and FCF shows cash available to fund growth or returns.")
        st.subheader("Forecast outlook · 2027–2031")
        forecast = _forecast_table(workbook)
        if forecast.empty:
            st.info("No complete 2027–2031 forecast values were found.")
        else:
            metric = st.selectbox(
                "Forecast metric",
                ["Revenue", "EBITDA", "EPS"],
                key="forecast_metric",
            )
            chart = go.Figure(
                go.Bar(
                    x=forecast["Year"],
                    y=forecast[metric],
                    marker_color="#14558a",
                    text=forecast[metric].round(2),
                    textposition="outside",
                )
            )
            chart.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=25, b=10),
                yaxis_title=metric,
                xaxis_title="Forecast year",
                showlegend=False,
            )
            st.plotly_chart(chart, use_container_width=True)
            display_forecast = forecast.copy()
            for column in display_forecast.columns:
                if column.endswith("Growth"):
                    display_forecast[column] = display_forecast[column].map(
                        lambda value: f"{value:.1%}" if pd.notna(value) else None
                    )
                elif column != "Year":
                    display_forecast[column] = display_forecast[column].map(
                        lambda value: f"{value:,.2f}" if pd.notna(value) else None
                    )
            st.dataframe(
                _display_frame(display_forecast),
                use_container_width=True,
                hide_index=True,
            )

    elif page == "3 · Valuation":
        st.markdown("### Valuation & transaction lens")
        st.write("This page frames the company through a DCF lens. It is an analytical scenario, not an automatic recommendation.")
        wacc, terminal_growth, assumption_source = _dcf_assumptions(workbook, mapping)
        dcf = DCFValuationEngine(mapping, assumptions={"wacc": wacc, "terminal_growth": terminal_growth}).estimate()
        cards = st.columns(4)
        for column, label, key in (
            (cards[0], "Enterprise Value", "enterprise_value"),
            (cards[1], "Equity Value", "equity_value"),
            (cards[2], "Value / Share", "intrinsic_value_per_share"),
            (cards[3], "Market Price", "market_price"),
        ):
            value = dcf[key]
            column.metric(label, f"{value:,.2f}" if value is not None else "N/A")
        if dcf["status"] == "CALCULATED":
            forecast = dcf["forecast"]
            figure = go.Figure(go.Bar(x=forecast["year"], y=forecast["fcf"], name="Forecast FCF"))
            figure.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Forecast FCF", xaxis_title="Year")
            st.plotly_chart(figure, use_container_width=True)
            st.caption(
                f"Key assumptions from {assumption_source}: "
                f"WACC {wacc:.2%} · Terminal growth {terminal_growth:.2%}"
            )
            if dcf["upside_percent"] is not None:
                st.metric("Implied upside / downside", f"{dcf['upside_percent']:+.1f}%")
        else:
            st.warning("A complete DCF cannot be calculated from the available source inputs.")
            st.write("Missing inputs: " + ", ".join(dcf["missing_inputs"]))
        st.subheader("Comparable companies")
        peer_table = _peer_numeric_table(workbook)
        if peer_table.empty:
            st.info("No complete numeric peer-company rows are available.")
        else:
            st.caption("Only customer-relevant numeric values are shown; ticker and empty fields are excluded.")
            st.dataframe(_display_frame(peer_table), use_container_width=True, hide_index=True)
        st.subheader("Football field valuation")
        football_table = _football_field_table(workbook)
        if football_table.empty:
            st.info("No complete Football Field valuation ranges were found.")
        else:
            figure = go.Figure()
            figure.add_trace(go.Bar(
                x=football_table["High"] - football_table["Low"],
                y=football_table["Scenario"],
                base=football_table["Low"],
                orientation="h",
                marker_color="#14558a",
                name="Valuation range",
                hovertemplate="%{y}<br>Low: %{customdata[0]:,.2f}<br>High: %{customdata[1]:,.2f}<extra></extra>",
                customdata=football_table[["Low", "High"]],
            ))
            figure.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Value per share", showlegend=False)
            st.plotly_chart(figure, use_container_width=True)
            st.dataframe(
                _display_frame(football_table),
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.markdown("### Risk & confidence")
        st.write("A professional transaction view separates business opportunity from model uncertainty. Review these items before using the dashboard in a client recommendation.")
        quality = validation.to_dataframe()
        counts = st.columns(4)
        counts[0].metric("Pass", validation.count("PASS"))
        counts[1].metric("Warnings", validation.count("WARNING"))
        counts[2].metric("Failures", validation.count("FAIL"))
        counts[3].metric("Unavailable", validation.count("NOT_AVAILABLE"))
        issues = quality[quality["status"].isin(["FAIL", "WARNING", "NOT_AVAILABLE"])]
        if issues.empty:
            st.success("No open data-quality items.")
        else:
            st.dataframe(_display_frame(issues[["check_name", "status", "severity", "message", "source"]]), use_container_width=True, hide_index=True)
        st.subheader("Source transparency")
        st.write(f"**Company:** {workbook.company_name or 'Not identified'}")
        st.write(f"**Workbook:** {source_label}")
        st.write(f"**Worksheets analyzed:** {workbook.sheet_count}")
        st.info("Use this page to explain what is known, what is estimated, and what should be confirmed with management.")
        st.subheader("Risk tools available in source model")
        st.write(
            "The workbook includes VaR, Altman Z-Score, DuPont, and market-history analyses. "
            "Only the quality indicators above are shown here so the customer view stays clean and decision-focused."
        )


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
        [data-testid="stMetric"] {background: #f7f9fc; border: 1px solid #e6eaf0;
            border-radius: 10px; padding: 14px 16px;}
        .ib-hero {background: linear-gradient(110deg, #0b1f3a, #173f67); color: white;
            border-radius: 14px; padding: 28px 32px; margin-bottom: 18px;}
        .ib-hero h1 {font-size: 2.2rem; margin: 0 0 8px 0; color: white;}
        .ib-hero p {margin: 0; color: #d9e8f7; font-size: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("## 📊 IB Dashboard")
    st.sidebar.caption("Workbook-driven analysis")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Financial Model",
        type=["xlsx"],
        help="Upload an .xlsx financial model. The workbook becomes the sole source for displayed values.",
    )
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(uploaded_file.getvalue())
            workbook_path = Path(temporary_file.name)
        source_label = uploaded_file.name
    else:
        workbook_path = Path(DEFAULT_WORKBOOK_PATH)
        source_label = workbook_path.name
        if not workbook_path.exists():
            st.info("Upload an .xlsx financial model to begin.")
            st.stop()

    with st.spinner("Scanning workbook structure..."):
        try:
            workbook = _scan_cached(str(workbook_path))
        finally:
            if uploaded_file is not None and workbook_path.exists():
                os.unlink(workbook_path)

    st.markdown(
        f"""
        <div class="ib-hero">
            <h1>{workbook.company_name or "Investment Banking Dashboard"}</h1>
            <p>Integrated financial model review · {source_label} · {workbook.sheet_count} worksheets</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Preparing dashboard analytics..."):
        mapping = _map_cached(workbook, "period-and-workbook-data-v6")
        validation = validate_mapping(mapping)
        analysis = FinancialAnalysisEngine(mapping, validation).analyze()
    dashboard = DashboardData(mapping=mapping, validation=validation, analysis=analysis)
    overview = dashboard.overview()
    context = dashboard._context()
    _render_client_dashboard(workbook, source_label, mapping, validation, analysis, overview, context)
    return
    st.subheader("Executive overview")
    overview_cols = st.columns(4)
    overview_cols[0].metric("Revenue", f"{overview['revenue_latest']:,.2f}" if overview["revenue_latest"] is not None else "N/A")
    overview_cols[1].metric("EBITDA", f"{overview['ebitda_latest']:,.2f}" if overview["ebitda_latest"] is not None else "N/A")
    overview_cols[2].metric("PAT", f"{overview['pat_latest']:,.2f}" if overview["pat_latest"] is not None else "N/A")
    overview_cols[3].metric("Free Cash Flow", f"{overview['fcf_latest']:,.2f}" if overview["fcf_latest"] is not None else "N/A")
    status_cols = st.columns(4)
    status_cols[0].metric("Data Quality", validation.overall_status)
    status_cols[1].metric("Validation Pass", overview["validation_pass"])
    status_cols[2].metric("Warnings", overview["validation_warning"])
    status_cols[3].metric("Failures", overview["validation_fail"])
    view_mode = st.sidebar.selectbox(
        "Customer view",
        ["Executive Summary", "Financial Performance", "Valuation", "Data Quality & Sources"],
        key="customer_view_v3",
        help="Choose the story you want to explain to your customer.",
    )
    revenue_points = context.unique_points("revenue")
    available_periods = [point["period"] for point in revenue_points if point.get("period")]
    if view_mode in {"Executive Summary", "Financial Performance"}:
        selected_periods = st.multiselect(
            "Reporting periods",
            options=available_periods,
            default=available_periods[-10:],
            key="dashboard_periods_v3",
        )
        selected_set = set(selected_periods or available_periods)
    else:
        selected_set = set(available_periods)

    chart_rows = []
    for item, label in (
        ("revenue", "Revenue"),
        ("ebitda", "EBITDA"),
        ("pat", "PAT"),
        ("free_cash_flow", "Free Cash Flow"),
    ):
        for point in context.unique_points(item):
            if point.get("period") in selected_set:
                chart_rows.append({"period": point["period"], "metric": label, "value": point["value"]})
    chart_data = pd.DataFrame(chart_rows)

    if view_mode in {"Executive Summary", "Financial Performance"}:
        if view_mode == "Executive Summary":
            st.subheader("What this means for the customer")
            st.write(
                "This view gives a quick investment-banking readout: company scale, "
                "profitability, cash generation, valuation readiness, and items that need review."
            )
            status_cols = st.columns(3)
            status_cols[0].metric("Workbook", "Loaded")
            status_cols[1].metric("Analysis", "Complete")
            status_cols[2].metric("Items to review", validation.count("WARNING") + validation.count("FAIL"))
        else:
            st.subheader("Financial performance")
            st.write("Use the period selector above to focus the conversation on the years relevant to your customer.")
        if not chart_data.empty:
            figure = go.Figure()
            for metric_name in chart_data["metric"].unique():
                metric_data = chart_data[chart_data["metric"] == metric_name]
                figure.add_trace(go.Scatter(
                    x=metric_data["period"], y=metric_data["value"],
                    mode="lines+markers", name=metric_name,
                ))
            figure.update_layout(
                height=430, margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified", legend=dict(orientation="h", y=1.08),
                yaxis_title="Amount", xaxis_title="Reporting period",
            )
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No period-aligned operating data is available for the selected periods.")
        if view_mode == "Executive Summary":
            issues = validation.to_dataframe()
            issues = issues[issues["status"].isin(["FAIL", "WARNING"])]
            st.subheader("Key messages")
            st.success("The workbook is loaded and the core financial analysis is complete.")
            if not issues.empty:
                st.warning(f"{len(issues)} item(s) need review before customer-facing conclusions are finalized.")
            else:
                st.success("No validation warnings or failures were identified.")
        st.stop()

    if view_mode == "Valuation":
        st.subheader("Valuation view")
        st.write(
            "This section explains the value of the business using a discounted cash flow approach. "
            "Values are model outputs and should be reviewed with management assumptions."
        )
        dcf_result = DCFValuationEngine(mapping).estimate()
        valuation_cols = st.columns(4)
        for column, label, key in zip(
            valuation_cols,
            ("Enterprise Value", "Equity Value", "Value / Share", "Market Price"),
            ("enterprise_value", "equity_value", "intrinsic_value_per_share", "market_price"),
        ):
            value = dcf_result[key]
            column.metric(label, f"{value:,.2f}" if value is not None else "N/A")
        if dcf_result["status"] == "CALCULATED":
            forecast = dcf_result["forecast"]
            figure = go.Figure()
            figure.add_trace(go.Bar(x=forecast["year"], y=forecast["fcf"], name="Forecast FCF"))
            figure.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="FCF")
            st.plotly_chart(figure, use_container_width=True)
            st.caption("Forecast cash flows are scenario outputs, not a recommendation.")
        else:
            st.warning("Valuation is not available because required financial inputs are missing.")
            st.write("Missing inputs: " + ", ".join(dcf_result["missing_inputs"]))
        st.stop()

    st.subheader("Data quality & sources")
    st.write(
        "This view helps explain how much confidence to place in the analysis. "
        "Warnings and unavailable checks should be resolved before making an investment decision."
    )
    quality_table = validation.to_dataframe()
    st.dataframe(
        _display_frame(quality_table[["check_name", "status", "severity", "message", "source"]]),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Workbook source")
    st.write(f"**File:** {source_label}")
    st.write(f"**Company:** {workbook.company_name or 'Not identified'}")
    st.write(f"**Worksheets analyzed:** {workbook.sheet_count}")
    st.stop()

    st.divider()
    st.subheader("Workbook overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Company", workbook.company_name or "Not identified")
    col2.metric("Worksheets", workbook.sheet_count)
    col3.metric("Workbook", source_label)
    if workbook.company_name:
        st.title(workbook.company_name)

    sheet_names = workbook.sheet_names
    selected_sheet_name = st.selectbox("Select worksheet", sheet_names, index=0 if sheet_names else None)

    if not selected_sheet_name:
        st.warning("No worksheets were found in the workbook.")
        return

    selected_sheet = workbook.get_sheet(selected_sheet_name)
    if selected_sheet is None:
        st.warning(f"Could not load details for worksheet: {selected_sheet_name}")
        return

    st.subheader(f"Selected worksheet: {selected_sheet.name}")
    st.write(f"Detected sheet type: **{selected_sheet.detected_type}**")

    metrics = st.columns(2)
    metrics[0].metric("Rows", selected_sheet.row_count)
    metrics[1].metric("Columns", selected_sheet.column_count)

    st.subheader("Preview")
    st.dataframe(_display_frame(selected_sheet.preview), use_container_width=True, hide_index=True)

    st.subheader("Worksheet index")
    workbook_rows = []
    for sheet in workbook.sheets:
        workbook_rows.append(
            {
                "Worksheet": sheet.name,
                "Rows": sheet.row_count,
                "Columns": sheet.column_count,
                "Detected Type": sheet.detected_type,
            }
        )

    st.dataframe(_display_frame(pd.DataFrame(workbook_rows)), use_container_width=True, hide_index=True)

    st.divider()
    st.header("Financial Data Mapping")
    st.caption("Phase 2 — standardized line-item mapping; no valuation or financial calculations are performed.")
    mapping_table = mapping.to_dataframe()
    display_columns = [
        "display_name",
        "detected_label",
        "value",
        "period",
        "period_type",
        "source_sheet",
        "source_row",
        "source_column",
        "confidence",
        "status",
    ]
    st.dataframe(_display_frame(mapping_table[display_columns]), use_container_width=True, hide_index=True)

    st.divider()
    st.header("🔍 Financial Model Validation")
    st.caption("Phase 3 — validation identifies issues without changing source values or calculating valuation.")
    st.subheader(f"DATA QUALITY STATUS: {validation.overall_status}")
    summary_columns = st.columns(4)
    summary_columns[0].metric("PASS", validation.count("PASS"))
    summary_columns[1].metric("WARNING", validation.count("WARNING"))
    summary_columns[2].metric("FAIL", validation.count("FAIL"))
    summary_columns[3].metric("N/A", validation.count("NOT_AVAILABLE"))

    validation_table = validation.to_dataframe()
    status_filter = st.multiselect("Filter validation status", VALIDATION_STATUSES, default=list(VALIDATION_STATUSES))
    category_filter = st.multiselect("Filter validation category", sorted(validation_table["category"].unique()), default=sorted(validation_table["category"].unique()))
    filtered_validation = validation_table[
        validation_table["status"].isin(status_filter)
        & validation_table["category"].isin(category_filter)
    ]
    st.dataframe(
        _display_frame(filtered_validation[
            ["check_name", "category", "status", "severity", "value", "expected", "difference", "source", "message"]
        ]),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("DCF Input Readiness")
    st.write(f"**{validation.dcf_readiness}**")
    if validation.dcf_missing_inputs:
        st.write("Missing inputs: " + ", ".join(validation.dcf_missing_inputs))

    st.divider()
    st.header("📊 Financial Analysis")
    st.caption("Phase 4 — period-aware operating analysis; no charts, DCF, or valuation calculations.")
    st.subheader("Analysis metrics")
    st.dataframe(_display_frame(analysis.to_dataframe()), use_container_width=True, hide_index=True)
    st.subheader("Historical vs forecast summary")
    st.dataframe(_display_frame(analysis.summary_to_dataframe()), use_container_width=True, hide_index=True)
    if analysis.flags:
        st.subheader("Analytical flags")
        st.dataframe(_display_frame(analysis.flags_to_dataframe()), use_container_width=True, hide_index=True)
    if analysis.insights:
        st.subheader("Insights")
        for insight in analysis.insights:
            st.write(f"• {insight}")
    st.divider()
    st.header("📈 Phase 5 — DCF and Comparable Valuation")
    workbook_wacc = _latest_mapping_value(mapping, "wacc", minimum=0.0, maximum=1.0)
    workbook_terminal_growth = _latest_mapping_value(
        mapping, "terminal_growth", minimum=-0.5, maximum=0.5
    )
    if workbook_terminal_growth is not None and not -0.5 <= workbook_terminal_growth <= 0.5:
        st.warning(
            "The uploaded workbook contains an out-of-range terminal-growth mapping. "
            "Please review the source mapping; using 3.0% for this session."
        )
        workbook_terminal_growth = 0.03
    if "phase5_wacc" in st.session_state and not 0.0 <= st.session_state["phase5_wacc"] <= 1.0:
        st.session_state["phase5_wacc"] = workbook_wacc if workbook_wacc is not None else 0.10
    if "phase5_terminal_growth" in st.session_state and not -0.5 <= st.session_state["phase5_terminal_growth"] <= 0.5:
        st.session_state["phase5_terminal_growth"] = (
            workbook_terminal_growth if workbook_terminal_growth is not None else 0.03
        )
    assumption_columns = st.columns(2)
    wacc = assumption_columns[0].number_input(
        "WACC",
        min_value=0.0,
        max_value=1.0,
        value=workbook_wacc if workbook_wacc is not None else 0.10,
        key="phase5_wacc",
        step=0.005,
        format="%.3f",
        help="Uses the mapped workbook WACC when available.",
    )
    terminal_growth = assumption_columns[1].number_input(
        "Terminal Growth",
        min_value=-0.5,
        max_value=0.5,
        value=workbook_terminal_growth if workbook_terminal_growth is not None else 0.03,
        key="phase5_terminal_growth",
        step=0.005,
        format="%.3f",
    )
    if workbook_wacc is not None:
        st.caption(f"Workbook WACC source: {workbook_wacc:.2%}")
    else:
        st.warning("WACC was not found in the uploaded workbook; the displayed value is a user assumption.")
    if workbook_terminal_growth is not None:
        st.caption(f"Workbook terminal growth source: {workbook_terminal_growth:.2%}")
    else:
        st.warning(
            "A valid terminal growth rate was not found in the workbook; "
            "the displayed value is a user assumption."
        )
    dcf_engine = DCFValuationEngine(
        mapping,
        assumptions={"wacc": wacc, "terminal_growth": terminal_growth},
    )
    dcf_result = dcf_engine.estimate()
    st.subheader("DCF summary")
    dcf_cols = st.columns(4)
    dcf_cols[0].metric("Enterprise Value", f"{dcf_result['enterprise_value']:.2f}" if dcf_result["enterprise_value"] is not None else "N/A")
    dcf_cols[1].metric("Equity Value", f"{dcf_result['equity_value']:.2f}" if dcf_result["equity_value"] is not None else "N/A")
    dcf_cols[2].metric("Intrinsic Value / Share", f"{dcf_result['intrinsic_value_per_share']:.2f}" if dcf_result["intrinsic_value_per_share"] is not None else "N/A")
    dcf_cols[3].metric("Market Price", f"{dcf_result['market_price']:.2f}" if dcf_result["market_price"] is not None else "N/A")
    if dcf_result["status"] == "NOT_AVAILABLE":
        st.warning("DCF is unavailable because required source inputs are missing: " + ", ".join(dcf_result["missing_inputs"]))
    else:
        st.dataframe(_display_frame(dcf_result["forecast"]), use_container_width=True, hide_index=True)
    st.subheader("Comparable data")
    st.dataframe(_display_frame(ComparableEngine(mapping).peer_rows()), use_container_width=True, hide_index=True)
    st.divider()
    st.header("🏢 Investment Banking Dashboard Detail")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Revenue", f"{overview['revenue_latest']:,.2f}" if overview["revenue_latest"] is not None else "N/A")
    summary_cols[1].metric("EBITDA", f"{overview['ebitda_latest']:,.2f}" if overview["ebitda_latest"] is not None else "N/A")
    summary_cols[2].metric("PAT", f"{overview['pat_latest']:,.2f}" if overview["pat_latest"] is not None else "N/A")
    summary_cols[3].metric("FCF", f"{overview['fcf_latest']:,.2f}" if overview["fcf_latest"] is not None else "N/A")
    st.subheader("Executive summary")
    summary_rows = [
        {"Category": "Validation", "Status": f"PASS {overview['validation_pass']} | WARNING {overview['validation_warning']} | FAIL {overview['validation_fail']}", "Notes": "Validation results came from the mapped workbook."},
        {"Category": "Growth", "Status": "Computed from full analysis engine", "Notes": "Revenue, EBITDA, EBIT, PAT, EPS, FCF growth computed on aligned period data."},
        {"Category": "Profitability", "Status": "Computed from mapped data", "Notes": "Gross margin, EBITDA margin, PAT margin, ROE, ROA, ROIC were generated from the mapping output."},
        {"Category": "Valuation", "Status": "DCF estimate generated", "Notes": "DCF uses input assumptions and the derived forecast; not a final recommendation."},
    ]
    st.dataframe(_display_frame(pd.DataFrame(summary_rows)), use_container_width=True, hide_index=True)
    st.subheader("Financial health")
    health = analysis.to_dataframe()[["metric_name", "category", "period", "period_type", "value", "status"]].copy()
    st.dataframe(_display_frame(health), use_container_width=True, hide_index=True)
    st.subheader("Investment banking summary")
    st.write("This dashboard section is intentionally model-driven and traceable to the standardized mapping output; no valuation recommendation is generated without explicit user review.")
if __name__ == "__main__":
    main()
