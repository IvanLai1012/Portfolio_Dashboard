from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Portfolio Analytics & Stress Test", layout="wide")

CRISIS_PERIODS = {
    "2008 Global Financial Crisis": ("2007-10-01", "2009-03-09"),
    "2020 COVID Liquidity Shock": ("2020-02-19", "2020-03-23"),
    "2022 Inflation and Rate-Hike Cycle": ("2022-01-03", "2022-12-30"),
    "2000-02 Dot-Com Drawdown": ("2000-03-24", "2002-10-09"),
}

CAPM = "Capital Asset Pricing Model (CAPM)"
HISTORICAL = "Historical Geometric Mean (5Y)"
BLENDED = "Smart Estimate (CAPM + Mean-Reverting Alpha)"

CUSTOM_ASSET_RULES = {
    "WS2": {"remark": "HSBC World Selection 2", "beta": 0.35, "expected_return": 5.50},
    "WS3": {"remark": "HSBC World Selection 3", "beta": 0.55, "expected_return": 6.80},
    "MMF": {"remark": "Money Market Fund (MMF)", "beta": 0.01},
    "HKD_CASH": {"remark": "Pure Cash", "beta": 0.00},
}

BYPASS_REMARKS = {
    "Pure Cash",
    "Money Market Fund (MMF)",
    "HSBC World Selection 2",
    "HSBC World Selection 3",
}

DEFAULT_ASSETS = pd.DataFrame(
    {
        "Ticker/Asset": [
            "QQQ", "SCHD", "GOOGL", "SAP", "KO", "MCD", "MSFT",
            "VEEV", "NVDA", "IAU", "WS2", "MMF", "HKD_CASH",
        ],
        "Asset Type (Remark)": [
            "Equity (Core)", "Equity (Core)", "Equity (Satellite)", "Equity (Satellite)",
            "Equity (Core)", "Equity (Core)", "Equity (Satellite)", "Equity (Satellite)",
            "Equity (Satellite)", "Alternative (Gold)", "HSBC World Selection 2",
            "Money Market Fund (MMF)", "Pure Cash",
        ],
        "Methodology": [
            CAPM, CAPM, CAPM, CAPM, CAPM, CAPM, CAPM, CAPM, CAPM,
            HISTORICAL, HISTORICAL, HISTORICAL, HISTORICAL,
        ],
        "Current Value": [15.0, 15.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 7.0, 10.0, 5.0, 10.0, 10.0],
    }
)


def _extract_close(raw: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    tickers = list(tickers)
    if raw.empty:
        return pd.DataFrame(columns=tickers)

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if "Close" in level0:
            close = raw["Close"].copy()
        elif "Close" in level1:
            close = raw.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame(columns=tickers)
    else:
        if "Close" not in raw.columns:
            return pd.DataFrame(columns=tickers)
        close = raw[["Close"]].copy()
        if len(tickers) == 1:
            close.columns = tickers

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    available = [ticker for ticker in tickers if ticker in close.columns]
    return close.reindex(columns=available).sort_index().dropna(how="all")


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    wealth = (1 + clean).cumprod()
    return float((wealth / wealth.cummax() - 1).min() * 100)


def _geometric_annual_return(prices: pd.Series) -> float:
    clean = prices.dropna()
    if len(clean) < 2:
        return np.nan
    years = (clean.index[-1] - clean.index[0]).days / 365.25
    if years <= 0 or clean.iloc[0] <= 0 or clean.iloc[-1] <= 0:
        return np.nan
    return float(((clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1) * 100)


def _safe_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    aligned = pd.concat(
        [asset_returns.rename("asset"), market_returns.rename("market")], axis=1
    ).dropna()
    if len(aligned) < 26 or aligned["market"].var() <= 0:
        return np.nan
    return float(aligned["asset"].cov(aligned["market"]) / aligned["market"].var())


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_tbill_proxy() -> tuple[float, str]:
    """Use the 13-week US T-bill yield as a configurable risk-free proxy."""
    try:
        raw = yf.download(
            "^IRX", period="10d", auto_adjust=True, progress=False, threads=False
        )
        close = _extract_close(raw, ["^IRX"])
        value = float(close["^IRX"].dropna().iloc[-1])
        return value, "Live ^IRX 13-week T-bill proxy"
    except Exception as exc:
        return 4.0, f"Fallback 4.0% because ^IRX was unavailable: {exc}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_panels(tickers: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    all_tickers = tuple(dict.fromkeys((*tickers, "^GSPC")))
    end = pd.Timestamp.now().normalize()
    five_year_start = end - pd.DateOffset(years=5)
    stress_start = min(pd.Timestamp(start) for start, _ in CRISIS_PERIODS.values())

    weekly = pd.DataFrame()
    daily = pd.DataFrame()

    try:
        raw_weekly = yf.download(
            list(all_tickers),
            start=five_year_start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1wk",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
        weekly = _extract_close(raw_weekly, all_tickers)
    except Exception as exc:
        warnings.append(f"Five-year weekly data failed: {exc}")

    try:
        raw_daily = yf.download(
            list(all_tickers),
            start=stress_start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
        daily = _extract_close(raw_daily, all_tickers)
    except Exception as exc:
        warnings.append(f"Long-history stress data failed: {exc}")

    return weekly, daily, warnings


def normalise_asset_table(edited: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    table = edited.copy()
    messages: list[str] = []
    table = table.dropna(subset=["Ticker/Asset"]).reset_index(drop=True)
    table["Ticker/Asset"] = table["Ticker/Asset"].astype(str).str.strip().str.upper()
    table["Current Value"] = pd.to_numeric(table["Current Value"], errors="coerce").fillna(0.0)
    table = table[table["Current Value"] >= 0].copy()

    for index, row in table.iterrows():
        ticker = row["Ticker/Asset"]
        rule = CUSTOM_ASSET_RULES.get(ticker)
        if not rule:
            continue
        if row["Asset Type (Remark)"] != rule["remark"]:
            table.at[index, "Asset Type (Remark)"] = rule["remark"]
            messages.append(f"{ticker}: asset type restored to the preset mapping")
        if row["Methodology"] not in {HISTORICAL, BLENDED, CAPM}:
            table.at[index, "Methodology"] = HISTORICAL

    table["Row ID"] = [f"R{i + 1:02d}" for i in range(len(table))]
    return table, messages


def build_ticker_analytics(
    weekly_prices: pd.DataFrame,
    asset_table: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    analytics: dict[str, dict[str, Any]] = {}
    market_returns = (
        weekly_prices["^GSPC"].pct_change(fill_method=None).dropna()
        if "^GSPC" in weekly_prices
        else pd.Series(dtype=float)
    )

    for _, row in asset_table.iterrows():
        ticker = row["Ticker/Asset"]
        remark = row["Asset Type (Remark)"]
        if ticker in analytics:
            continue

        rule = CUSTOM_ASSET_RULES.get(ticker)
        if rule:
            analytics[ticker] = {
                "beta": float(rule["beta"]),
                "geo_return": float(rule.get("expected_return", np.nan)),
                "annual_volatility": np.nan,
                "max_drawdown": np.nan,
                "source": "Preset model assumption",
            }
            continue

        if ticker not in weekly_prices or weekly_prices[ticker].dropna().empty:
            analytics[ticker] = {
                "beta": 1.0,
                "geo_return": np.nan,
                "annual_volatility": np.nan,
                "max_drawdown": np.nan,
                "source": "No history; beta fallback = 1.0",
            }
            continue

        prices = weekly_prices[ticker].dropna()
        returns = prices.pct_change(fill_method=None).dropna()
        beta = _safe_beta(returns, market_returns)
        annual_vol = float(returns.std() * np.sqrt(52) * 100) if len(returns) >= 26 else np.nan
        analytics[ticker] = {
            "beta": beta if np.isfinite(beta) else 1.0,
            "geo_return": _geometric_annual_return(prices),
            "annual_volatility": annual_vol,
            "max_drawdown": _max_drawdown_from_returns(returns),
            "source": "Observed adjusted-price history" if np.isfinite(beta) else "History available; beta fallback = 1.0",
        }

    return analytics


def build_stress_matrix(
    daily_prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    analytics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    matrix: dict[str, dict[str, dict[str, Any]]] = {}

    for scenario, (start_text, end_text) in CRISIS_PERIODS.items():
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        scenario_prices = daily_prices.loc[start:end] if not daily_prices.empty else pd.DataFrame()
        market_prices = scenario_prices.get("^GSPC", pd.Series(dtype=float)).dropna()
        market_return = (
            float((market_prices.iloc[-1] / market_prices.iloc[0] - 1) * 100)
            if len(market_prices) >= 2
            else -25.0
        )
        market_max_dd = (
            _max_drawdown_from_returns(market_prices.pct_change(fill_method=None))
            if len(market_prices) >= 2
            else -25.0
        )

        matrix[scenario] = {}
        for _, row in asset_table.iterrows():
            row_id = row["Row ID"]
            ticker = row["Ticker/Asset"]
            remark = row["Asset Type (Remark)"]
            beta = float(np.clip(analytics[ticker]["beta"], -0.5, 2.5))

            if remark in {"Pure Cash", "Money Market Fund (MMF)"}:
                result = {
                    "return": 0.0,
                    "max_drawdown": 0.0,
                    "source": "Capital-value assumption; income excluded",
                }
            elif remark in {"HSBC World Selection 2", "HSBC World Selection 3"}:
                result = {
                    "return": market_return * beta,
                    "max_drawdown": market_max_dd * abs(beta),
                    "source": f"S&P 500 beta proxy ({beta:.2f})",
                }
            elif ticker in scenario_prices and len(scenario_prices[ticker].dropna()) >= 2:
                prices = scenario_prices[ticker].dropna()
                result = {
                    "return": float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
                    "max_drawdown": _max_drawdown_from_returns(prices.pct_change(fill_method=None)),
                    "source": "Observed adjusted-price history",
                }
            else:
                result = {
                    "return": market_return * beta,
                    "max_drawdown": market_max_dd * abs(beta),
                    "source": f"S&P 500 beta proxy ({beta:.2f}); asset history unavailable",
                }

            matrix[scenario][row_id] = result

    return matrix


def build_portfolio_history(
    weekly_prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    analytics: dict[str, dict[str, Any]],
    rf_rate: float,
) -> tuple[pd.Series, dict[str, float]]:
    if "^GSPC" not in weekly_prices:
        return pd.Series(dtype=float), {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
        }

    market_returns = weekly_prices["^GSPC"].pct_change(fill_method=None).dropna()
    row_returns: list[pd.Series] = []

    for _, row in asset_table.iterrows():
        ticker = row["Ticker/Asset"]
        remark = row["Asset Type (Remark)"]
        weight = row["Weight (%)"] / 100
        beta = float(np.clip(analytics[ticker]["beta"], -0.5, 2.5))

        if remark == "Pure Cash":
            returns = pd.Series(0.0, index=market_returns.index)
        elif remark == "Money Market Fund (MMF)":
            returns = pd.Series((1 + rf_rate / 100) ** (1 / 52) - 1, index=market_returns.index)
        elif remark in {"HSBC World Selection 2", "HSBC World Selection 3"}:
            returns = market_returns * beta
        elif ticker in weekly_prices:
            observed = weekly_prices[ticker].pct_change(fill_method=None).reindex(market_returns.index)
            returns = observed.where(observed.notna(), market_returns * beta)
        else:
            returns = market_returns * beta

        row_returns.append((returns * weight).rename(row["Row ID"]))

    if not row_returns:
        return pd.Series(dtype=float), {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
        }

    portfolio_returns = pd.concat(row_returns, axis=1).sum(axis=1, min_count=1).dropna()
    if len(portfolio_returns) < 52:
        return portfolio_returns, {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "max_drawdown": _max_drawdown_from_returns(portfolio_returns),
            "sharpe": np.nan,
        }

    years = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days / 365.25
    terminal = float((1 + portfolio_returns).prod())
    annual_return = (terminal ** (1 / years) - 1) * 100 if years > 0 and terminal > 0 else np.nan
    annual_volatility = float(portfolio_returns.std() * np.sqrt(52) * 100)
    sharpe = (
        (annual_return - rf_rate) / annual_volatility
        if np.isfinite(annual_return) and annual_volatility > 0
        else np.nan
    )
    metrics = {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": _max_drawdown_from_returns(portfolio_returns),
        "sharpe": sharpe,
    }
    return portfolio_returns, metrics


def apply_expected_return_models(
    asset_table: pd.DataFrame,
    analytics: dict[str, dict[str, Any]],
    rf_rate: float,
    market_premium: float,
    alpha_retention: float,
    cash_return: float,
    mmf_return: float,
) -> pd.DataFrame:
    table = asset_table.copy()
    output_rows: list[dict[str, Any]] = []

    for _, row in table.iterrows():
        ticker = row["Ticker/Asset"]
        remark = row["Asset Type (Remark)"]
        model = row["Methodology"]
        metrics = analytics[ticker]
        raw_beta = float(metrics["beta"])
        beta_used = float(np.clip(raw_beta, -0.5, 2.5))
        capm_return = rf_rate + beta_used * market_premium
        historical_return = float(metrics["geo_return"]) if np.isfinite(metrics["geo_return"]) else np.nan
        note = metrics["source"]

        if remark == "Pure Cash":
            expected_return = cash_return
            model_used = "User cash-return assumption"
        elif remark == "Money Market Fund (MMF)":
            expected_return = mmf_return
            model_used = "User MMF-return assumption"
        elif ticker in CUSTOM_ASSET_RULES and "expected_return" in CUSTOM_ASSET_RULES[ticker]:
            expected_return = float(CUSTOM_ASSET_RULES[ticker]["expected_return"])
            model_used = "Preset strategic-return assumption"
        elif model == CAPM:
            expected_return = capm_return
            model_used = CAPM
        elif model == HISTORICAL and np.isfinite(historical_return):
            expected_return = historical_return
            model_used = HISTORICAL
        elif model == BLENDED and np.isfinite(historical_return):
            historical_alpha = historical_return - capm_return
            expected_return = capm_return + alpha_retention * historical_alpha
            model_used = BLENDED
        else:
            expected_return = capm_return
            model_used = f"CAPM fallback; {model} unavailable"
            note = f"{note}; historical return unavailable"

        unclipped = expected_return
        expected_return = float(np.clip(expected_return, -10.0, 30.0))
        if expected_return != unclipped:
            note = f"{note}; expected return clipped to [-10%, 30%]"

        output_rows.append(
            {
                **row.to_dict(),
                "Beta (Raw)": raw_beta,
                "Beta Used": beta_used,
                "Historical Return (%)": historical_return,
                "CAPM Return (%)": capm_return,
                "Expected Return (%)": expected_return,
                "Model Used": model_used,
                "Data / Assumption Source": note,
                "Historical Volatility (%)": metrics["annual_volatility"],
                "Historical Max Drawdown (%)": metrics["max_drawdown"],
            }
        )

    output = pd.DataFrame(output_rows)
    output["Weighted Return (%)"] = output["Weight (%)"] / 100 * output["Expected Return (%)"]
    return output


# -----------------------------
# Streamlit application
# -----------------------------
st.title("Portfolio Analytics and Historical Stress Test")
st.caption(
    "Observed history and proxy assumptions are labelled separately. Expected returns are model inputs, not guarantees."
)

st.sidebar.header("Portfolio Settings")
live_rf_rate, rf_source = fetch_tbill_proxy()
rf_rate = st.sidebar.number_input(
    "Risk-free rate proxy (%)",
    min_value=-2.0,
    max_value=15.0,
    value=float(round(live_rf_rate, 2)),
    step=0.10,
    help=rf_source,
)
market_premium = st.sidebar.number_input(
    "Equity market risk premium (%)", min_value=0.0, max_value=12.0, value=5.5, step=0.25
)
alpha_retention = st.sidebar.slider(
    "Historical alpha retained in blended estimate",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
    help="0.30 retains 30% of historical alpha and mean-reverts the remaining 70%.",
)
cash_return = st.sidebar.number_input("Cash expected return (%)", -2.0, 15.0, 0.0, 0.10)
mmf_return = st.sidebar.number_input("MMF expected return (%)", -2.0, 15.0, float(round(rf_rate, 2)), 0.10)
currency_label = st.sidebar.text_input("Portfolio value currency", value="HKD")

st.sidebar.subheader("Assets")
edited = st.sidebar.data_editor(
    DEFAULT_ASSETS,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker/Asset": st.column_config.TextColumn(required=True),
        "Current Value": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
        "Asset Type (Remark)": st.column_config.SelectboxColumn(
            options=[
                "Equity (Core)",
                "Equity (Satellite)",
                "Alternative (Gold)",
                "HSBC World Selection 2",
                "HSBC World Selection 3",
                "Money Market Fund (MMF)",
                "Pure Cash",
            ],
            required=True,
        ),
        "Methodology": st.column_config.SelectboxColumn(
            options=[CAPM, HISTORICAL, BLENDED], required=True
        ),
    },
)

asset_table, normalisation_messages = normalise_asset_table(edited)
for message in normalisation_messages:
    st.sidebar.info(message)

if asset_table.empty or asset_table["Current Value"].sum() <= 0:
    st.warning("Add at least one asset with a positive current value.")
    st.stop()

total_value = float(asset_table["Current Value"].sum())
asset_table["Weight (%)"] = asset_table["Current Value"] / total_value * 100

api_tickers = tuple(
    sorted(
        {
            row["Ticker/Asset"]
            for _, row in asset_table.iterrows()
            if row["Asset Type (Remark)"] not in BYPASS_REMARKS
        }
    )
)

with st.spinner("Downloading adjusted-price history and calculating portfolio analytics..."):
    weekly_prices, daily_prices, data_warnings = fetch_price_panels(api_tickers)
    analytics = build_ticker_analytics(weekly_prices, asset_table)
    stress_matrix = build_stress_matrix(daily_prices, asset_table, analytics)
    model_table = apply_expected_return_models(
        asset_table,
        analytics,
        rf_rate,
        market_premium,
        alpha_retention,
        cash_return,
        mmf_return,
    )
    portfolio_returns, historical_metrics = build_portfolio_history(
        weekly_prices, model_table, analytics, rf_rate
    )

for warning in data_warnings:
    st.warning(warning)

portfolio_expected_return = float(model_table["Weighted Return (%)"].sum())
portfolio_beta = float((model_table["Weight (%)"] / 100 * model_table["Beta Used"]).sum())

st.subheader("Portfolio Overview")
metric_columns = st.columns(6)
metric_columns[0].metric("Portfolio Value", f"{currency_label} {total_value:,.2f}")
metric_columns[1].metric("Expected Return", f"{portfolio_expected_return:.2f}%")
metric_columns[2].metric("Portfolio Beta", f"{portfolio_beta:.2f}")
metric_columns[3].metric(
    "Historical Volatility",
    "N/A" if not np.isfinite(historical_metrics["annual_volatility"]) else f"{historical_metrics['annual_volatility']:.2f}%",
)
metric_columns[4].metric(
    "Historical Max Drawdown",
    "N/A" if not np.isfinite(historical_metrics["max_drawdown"]) else f"{historical_metrics['max_drawdown']:.2f}%",
)
metric_columns[5].metric(
    "Historical Sharpe",
    "N/A" if not np.isfinite(historical_metrics["sharpe"]) else f"{historical_metrics['sharpe']:.2f}",
)

st.caption(f"Risk-free-rate source: {rf_source}")

left, right = st.columns([3, 2])
with left:
    st.subheader("Modelled Asset Allocation")
    display_columns = [
        "Ticker/Asset",
        "Asset Type (Remark)",
        "Current Value",
        "Weight (%)",
        "Beta (Raw)",
        "Expected Return (%)",
        "Historical Volatility (%)",
        "Historical Max Drawdown (%)",
        "Model Used",
        "Data / Assumption Source",
    ]
    st.dataframe(
        model_table[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Current Value": st.column_config.NumberColumn(format="%.2f"),
            "Weight (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Beta (Raw)": st.column_config.NumberColumn(format="%.2f"),
            "Expected Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Historical Volatility (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Historical Max Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

with right:
    st.subheader("Allocation")
    allocation_figure = px.pie(
        model_table,
        values="Current Value",
        names="Ticker/Asset",
        hole=0.42,
    )
    allocation_figure.update_traces(textposition="inside", textinfo="percent+label")
    allocation_figure.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(allocation_figure, use_container_width=True)

st.subheader("Historical Scenario Analysis")
selected_scenario = st.selectbox("Scenario detail", list(CRISIS_PERIODS.keys()))
scenario_rows: list[dict[str, Any]] = []
for _, row in model_table.iterrows():
    result = stress_matrix[selected_scenario][row["Row ID"]]
    scenario_rows.append(
        {
            "Ticker/Asset": row["Ticker/Asset"],
            "Weight (%)": row["Weight (%)"],
            "Scenario Return (%)": result["return"],
            "Scenario Max Drawdown (%)": result["max_drawdown"],
            "Source": result["source"],
        }
    )
scenario_detail = pd.DataFrame(scenario_rows)
scenario_detail["Weighted Scenario Return (%)"] = (
    scenario_detail["Weight (%)"] / 100 * scenario_detail["Scenario Return (%)"]
)
selected_portfolio_return = float(scenario_detail["Weighted Scenario Return (%)"].sum())

scenario_left, scenario_right = st.columns([3, 2])
with scenario_left:
    st.dataframe(
        scenario_detail,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Weight (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Scenario Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Scenario Max Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Weighted Scenario Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
with scenario_right:
    st.metric("Portfolio Scenario Return", f"{selected_portfolio_return:.2f}%")
    observed_weight = scenario_detail.loc[
        scenario_detail["Source"].eq("Observed adjusted-price history"), "Weight (%)"
    ].sum()
    st.metric("Weight Covered by Observed History", f"{observed_weight:.1f}%")
    st.caption("The remaining weight uses labelled proxy assumptions.")

matrix_rows: list[dict[str, Any]] = []
for scenario in CRISIS_PERIODS:
    portfolio_return = 0.0
    observed_weight = 0.0
    for _, row in model_table.iterrows():
        result = stress_matrix[scenario][row["Row ID"]]
        portfolio_return += row["Weight (%)"] / 100 * result["return"]
        if result["source"] == "Observed adjusted-price history":
            observed_weight += row["Weight (%)"]
    matrix_rows.append(
        {
            "Scenario": scenario,
            "Portfolio Return (%)": portfolio_return,
            "Observed-History Coverage (%)": observed_weight,
        }
    )

matrix_df = pd.DataFrame(matrix_rows)
stress_figure = px.bar(
    matrix_df,
    x="Portfolio Return (%)",
    y="Scenario",
    orientation="h",
    text="Portfolio Return (%)",
    hover_data=["Observed-History Coverage (%)"],
)
stress_figure.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
stress_figure.update_layout(xaxis_ticksuffix="%", height=380)
st.plotly_chart(stress_figure, use_container_width=True)

if not portfolio_returns.empty:
    wealth = (1 + portfolio_returns).cumprod().rename("Growth of 1.0")
    wealth.index.name = "Date"
    wealth_figure = px.line(
        wealth.reset_index(),
        x="Date",
        y="Growth of 1.0",
        title="Modelled Historical Portfolio Growth",
    )
    st.plotly_chart(wealth_figure, use_container_width=True)
    st.caption(
        "For periods without asset history, the model backfills with the labelled beta proxy; this is not a true backtest."
    )

export_columns = [column for column in model_table.columns if column != "Row ID"]
export_csv = model_table[export_columns].to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Download portfolio analytics CSV",
    data=export_csv,
    file_name=f"portfolio_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

st.info(
    "Model limitations: CAPM is a single-factor estimate; historical returns may not repeat; beta proxies understate some idiosyncratic and regime risks; taxes, fees, FX, and liquidity are not modelled."
)
