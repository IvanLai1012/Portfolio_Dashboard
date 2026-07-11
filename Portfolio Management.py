from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# 1. Page Configuration (Consolidated to a single call)
st.set_page_config(
    page_title="Portfolio Risk Tool",
    layout="wide"  # Enforce wide mode for spacious viewports
)

st.title("🏛️ Live Technical Portfolio Risk & Stress-Testing Tool")
st.markdown("Please Don't Adjust the Preset non-stock and non-ETF items")

# --------------------------------------------------------
# 1. 歷史危機時間窗口定義
# --------------------------------------------------------
CRISIS_PERIODS = {
    "2008 Financial Crisis (Lehman Collapse)": ("2007-10-01", "2009-03-01"),
    "2020 COVID-19 Crash (Liquidity Shock)": ("2020-02-15", "2020-04-01"),
    "2022 Inflation & Rate Hike Cycle": ("2022-01-01", "2022-12-31"),
    "2000 Dot-Com Bubble (Tech Meltdown)": ("2000-03-01", "2002-10-01"),
}

# --------------------------------------------------------
# 2. SIDEBAR：資產輸入與核心設定
# --------------------------------------------------------
st.sidebar.header("🛠️ Portfolio Asset Setup")

return_model = st.sidebar.radio(
    "Select Expected Return Methodology",
    [
        "Capital Asset Pricing Model (CAPM)",
        "Historical Geometric Mean (5Y)",
        "Smart Realistic Estimate (CAPM + Blended Alpha) 🌟",
    ],
)

alpha_lambda = 0.3
if "Smart Realistic Estimate" in return_model:
    alpha_lambda = st.sidebar.slider(
        "Alpha Decay Factor (λ)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="0.3 代表保留 30% 的歷史超額報酬 (Alpha)，將 70% 進行均值回歸。數值越低越理性保守。",
    )

if "rf_rate" not in st.session_state: 
    try:
        ten_year_bond = yf.Ticker("^TNX")
        st.session_state.rf_rate = (
            ten_year_bond.history(period="1d")["Close"].iloc[-1]
        )
    except:
        st.session_state.rf_rate = 4.0
market_premium = 5.5 

default_assets = pd.DataFrame(
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
        "Current Value": [
            15.0, 15.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0, 7.0, 10.0, 5.0, 10.0, 10.0,
        ],
    }
)

st.sidebar.subheader("1. Input Live Assets & Remarks")
edited_df = st.sidebar.data_editor(
    default_assets,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Current Value": st.column_config.NumberColumn(format="$%.2f"),
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
    },
)

edited_df = edited_df.dropna(subset=["Ticker/Asset"])
edited_df["Ticker/Asset"] = edited_df["Ticker/Asset"].str.strip().str.upper()

if edited_df.empty or edited_df["Current Value"].sum() == 0:
    st.warning("Please add at least one valid asset and market value.")
    st.stop()

total_val = edited_df["Current Value"].sum()
edited_df["Weight (%)"] = (edited_df["Current Value"] / total_val) * 100

# --------------------------------------------------------
# 3. 核心數據運算引擎
# --------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_portfolio_analytics(ticker_rows, rf_rate):
    bypass_remarks = [
        "Pure Cash",
        "Money Market Fund (MMF)",
        "HSBC World Selection 2",
        "HSBC World Selection 3",
    ]
    api_tickers = [
        row["Ticker/Asset"]
        for row in ticker_rows
        if row["Asset Type (Remark)"] not in bypass_remarks
    ]
    unique_tickers = list(set(api_tickers + ["^GSPC"]))

    start_date = (datetime.now() - pd.Timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    raw_data = pd.DataFrame()

    if api_tickers:
        try:
            raw_data = yf.download(unique_tickers, start=start_date, interval="1wk")["Close"]
        except Exception as e:
            st.error(f"Error fetching data from Yahoo Finance: {e}")

    analytics = {}
    benchmark_returns = raw_data["^GSPC"].pct_change().dropna() if "^GSPC" in raw_data else pd.Series()

    for row in ticker_rows:
        tk = row["Ticker/Asset"]
        remark = row["Asset Type (Remark)"]

        if remark == "Pure Cash":
            analytics[tk] = {"beta": 0.00, "geo_return": 0.00}
        elif remark == "Money Market Fund (MMF)":
            analytics[tk] = {"beta": 0.01, "geo_return": float(rf_rate)}
        elif remark == "HSBC World Selection 2":
            analytics[tk] = {"beta": 0.35, "geo_return": 5.50}
        elif remark == "HSBC World Selection 3":
            analytics[tk] = {"beta": 0.55, "geo_return": 6.80}
        elif tk in raw_data and not raw_data[tk].dropna().empty:
            asset_prices = raw_data[tk].dropna()
            asset_returns = asset_prices.pct_change().dropna()
            combined = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
            combined.columns = ["asset", "market"]

            if len(combined) > 10 and combined["market"].var() != 0:
                beta = combined["asset"].cov(combined["market"]) / combined["market"].var()
            else:
                beta = 1.0

            total_ret = asset_prices.iloc[-1] / asset_prices.iloc[0]
            n_years = len(asset_prices) / 52.14
            geo_return = ((total_ret ** (1 / n_years) - 1) * 100 if n_years > 0 else rf_rate)
            analytics[tk] = {"beta": float(beta), "geo_return": float(geo_return)}
        else:
            analytics[tk] = {"beta": 1.0, "geo_return": 8.0}

    stress_results = {}
    for scenario_name, (s_start, s_end) in CRISIS_PERIODS.items():
        stress_results[scenario_name] = {}
        scen_data = pd.DataFrame()
        if api_tickers:
            try:
                scen_data = yf.download(unique_tickers, start=s_start, end=s_end)["Close"]
            except Exception:
                pass

        for row in ticker_rows:
            tk = row["Ticker/Asset"]
            remark = row["Asset Type (Remark)"]

            if remark == "Pure Cash":
                stress_results[scenario_name][tk] = 0.00
            elif remark == "Money Market Fund (MMF)":
                stress_results[scenario_name][tk] = (1.20 if "2008" not in scenario_name else 0.50)
            elif remark in ["HSBC World Selection 2", "HSBC World Selection 3"]:
                beta_proxy = 0.35 if remark == "HSBC World Selection 2" else 0.55
                if "^GSPC" in scen_data and not scen_data["^GSPC"].dropna().empty:
                    mkt_d = ((scen_data["^GSPC"].dropna().iloc[-1] / scen_data["^GSPC"].dropna().iloc[0]) - 1) * 100
                    stress_results[scenario_name][tk] = mkt_d * beta_proxy
                else:
                    stress_results[scenario_name][tk] = (-8.0 if remark == "HSBC World Selection 2" else -13.0)
            else:
                if tk in scen_data and len(scen_data[tk].dropna()) > 2:
                    clean_s = scen_data[tk].dropna()
                    stress_results[scenario_name][tk] = ((clean_s.iloc[-1] / clean_s.iloc[0]) - 1) * 100
                else:
                    if "^GSPC" in scen_data and not scen_data["^GSPC"].dropna().empty:
                        mkt_drawdown = ((scen_data["^GSPC"].dropna().iloc[-1] / scen_data["^GSPC"].dropna().iloc[0]) - 1) * 100
                    else:
                        mkt_drawdown = -25.0
                    stress_results[scenario_name][tk] = mkt_drawdown * analytics[tk]["beta"]

    return analytics, stress_results


ticker_input_tuples = edited_df[["Ticker/Asset", "Asset Type (Remark)"]].to_dict(orient="records")
with st.spinner("Calculating live analytics & mapping accounting remarks..."):
    live_analytics, stress_matrix = fetch_portfolio_analytics(ticker_input_tuples, st.session_state.rf_rate)

# --------------------------------------------------------
# 4. 反寫資料庫與三軌制 Methodology 運算
# --------------------------------------------------------
calculated_returns = []
live_betas = []

for idx, row in edited_df.iterrows():
    tk = row["Ticker/Asset"]
    ana = live_analytics.get(tk, {"beta": 1.0, "geo_return": 8.0})
    live_betas.append(ana["beta"])

    if row["Asset Type (Remark)"] == "Pure Cash":
        capm_ret = 0.00
    else:
        capm_ret = st.session_state.rf_rate + ana["beta"] * market_premium

    geo_ret = ana["geo_return"]

    if return_model == "Capital Asset Pricing Model (CAPM)":
        expected_ret = capm_ret
    elif return_model == "Historical Geometric Mean (5Y)":
        expected_ret = geo_ret
    else:
        if row["Asset Type (Remark)"] in ["Pure Cash", "Money Market Fund (MMF)"]:
            expected_ret = geo_ret
        else:
            hist_alpha = geo_ret - capm_ret
            expected_ret = capm_ret + (hist_alpha * alpha_lambda)

    calculated_returns.append(expected_ret)

edited_df["Beta"] = live_betas
edited_df["Expected Return (%)"] = calculated_returns
edited_df["Weighted Return (%)"] = (edited_df["Weight (%)"] / 100) * edited_df["Expected Return (%)"]

portfolio_return_baseline = edited_df["Weighted Return (%)"].sum()
portfolio_beta = ((edited_df["Weight (%)"] / 100) * edited_df["Beta"]).sum()

# --------------------------------------------------------
# 5. INTERACTIVE SCENARIO SELECTOR
# --------------------------------------------------------
st.subheader("🎯 Macro Economic Scenario Selection")
scenario_list = ["Current Baseline Projections"] + list(CRISIS_PERIODS.keys())
selected_scen = st.selectbox("Select Active Macro Scenario Focus", scenario_list)

if selected_scen == "Current Baseline Projections":
    portfolio_scen_return = portfolio_return_baseline
    edited_df["Scenario Return (%)"] = edited_df["Expected Return (%)"]
else:
    scen_returns = [stress_matrix[selected_scen].get(row["Ticker/Asset"], -10.0) for idx, row in edited_df.iterrows()]
    edited_df["Scenario Return (%)"] = scen_returns
    portfolio_scen_return = ((edited_df["Weight (%)"] / 100) * edited_df["Scenario Return (%)"]).sum()

# --------------------------------------------------------
# 6. DASHBOARD CARDS DISPLAY
# --------------------------------------------------------
st.markdown(" ")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Portfolio Value", f"${total_val:,.2f}")
c2.metric("Portfolio Volatility (Beta)", f"{portfolio_beta:.2f}")
c3.metric("Baseline Expected Return", f"{portfolio_return_baseline:.2f}%")

if selected_scen == "Current Baseline Projections":
    c4.metric("Scenario Projected Impact", f"{portfolio_scen_return:.2f}%", "No Active Stress")
else:
    delta_perf = portfolio_scen_return - portfolio_return_baseline
    c4.metric("Scenario Return", f"{portfolio_scen_return:.2f}%", f"{delta_perf:.2f}% Stress Shift", delta_color="inverse")

st.markdown("---")

# --------------------------------------------------------
# 7. VISUALIZATION: DATA TABLE & TICKER PIE CHART ONLY
# --------------------------------------------------------
# Left-hand table expanded, Right-hand holds only the clean Ticker Pie
main_col1, main_col2 = st.columns([5, 5])

with main_col1:
    st.subheader("📋 Dynamically Modeled Asset Allocation")
    fmt_df = edited_df.copy()
    fmt_df["Current Value"] = fmt_df["Current Value"].map("${:,.2f}".format)
    fmt_df["Weight (%)"] = fmt_df["Weight (%)"].map("{:.2f}%".format)
    fmt_df["Beta"] = fmt_df["Beta"].map("{:.2f}".format)
    fmt_df["Expected Return (%)"] = fmt_df["Expected Return (%)"].map("{:.2f}%".format)
    fmt_df["Scenario Return (%)"] = fmt_df["Scenario Return (%)"].map("{:.2f}%".format)

    st.dataframe(
        fmt_df[[
            "Ticker/Asset",
            "Asset Type (Remark)",
            "Current Value",
            "Weight (%)",
            "Beta",
            "Expected Return (%)",
            "Scenario Return (%)",
        ]],
        use_container_width=True,
        hide_index=True,
    )

with main_col2:
    st.subheader("🍕 Asset Allocation (Ticker)")
    fig_ticker = px.pie(
        edited_df,
        values="Current Value",
        names="Ticker/Asset",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_ticker.update_traces(textposition="outside", textinfo="percent+label")
    fig_ticker.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_ticker, use_container_width=True)

# --------------------------------------------------------
# 8. CROSS-REGIME STRESS MATRIX
# --------------------------------------------------------
st.subheader("📈 Cross-Regime Macro Stress Test Matrix")

matrix_data = [{"Scenario/Regime": "Current Baseline Projections", "Portfolio Stressed Return": portfolio_return_baseline}]
for name in CRISIS_PERIODS.keys():
    temp_ret = 0
    for idx, row in edited_df.iterrows():
        tk = row["Ticker/Asset"]
        temp_ret += (row["Weight (%)"] / 100) * stress_matrix[name].get(tk, 0)
    matrix_data.append({"Scenario/Regime": name, "Portfolio Stressed Return": temp_ret})

matrix_df = pd.DataFrame(matrix_data)

fig_bar = px.bar(
    matrix_df,
    x="Portfolio Stressed Return",
    y="Scenario/Regime",
    orientation="h",
    text="Portfolio Stressed Return",
    color="Portfolio Stressed Return",
    color_continuous_scale=px.colors.sequential.RdBu_r,
    labels={"Portfolio Stressed Return": "Portfolio Return under Scenario (%)"},
)
fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
fig_bar.update_layout(
    xaxis_ticksuffix="%",
    coloraxis_showscale=False,
    yaxis={"categoryorder": "total descending"},
    height=350,
)
st.plotly_chart(fig_bar, use_container_width=True)
