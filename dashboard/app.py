"""
Streamlit dashboard for the IPO Listing Gain Prediction strategy.

Run:
    streamlit run dashboard/app.py

Requires artifacts built via:
    python -m dashboard.build_artifacts
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import get_project_root, load_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRICS = {
    "cum_return": "Cumulative Return (%)",
    "mean_daily_return": "Mean Daily Return (%)",
    "win_rate": "Win Rate",
    "avg_allocation": "Average Allocation",
    "pct_days_traded": "% Days Traded",
    "volatility": "Volatility (%)",
    "sharpe_like": "Sharpe-like Ratio",
}
DEFAULT_METRIC = "cum_return"

PERIODS = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
DEFAULT_PERIOD = "Monthly"

# Metrics whose definition requires a minimum number of traded days per period.
# Everything else just needs num_days >= 1.
MIN_DAYS_FOR = {"volatility": 5, "sharpe_like": 5}

METRIC_DESCRIPTIONS = {
    "cum_return": (
        "Compounded return from the first decision day, in %. "
        "Computed as (∏(1 + r_d / 100) − 1) × 100 over all daily portfolio returns up to the point."
    ),
    "mean_daily_return": (
        "Average portfolio return per decision day; overall profitability. "
        "Formula: mean_daily_return = (1 / N) · Σ (Σ wᵢ × rᵢ)_day."
    ),
    "win_rate": (
        "Fraction of days where the portfolio generated positive return; consistency. "
        "Formula: win_rate = (#days with portfolio_return_day > 0) / N."
    ),
    "avg_allocation": (
        "Average fraction of capital deployed per day; utilization. "
        "Formula: avg_allocation = (1 / N) · Σ (Σ wᵢ)_day."
    ),
    "pct_days_traded": (
        "Fraction of days where at least some capital was deployed; participation. "
        "Formula: pct_days_traded = (#days with Σ wᵢ > 0) / N."
    ),
    "volatility": (
        "Standard deviation of daily portfolio returns; risk. "
        "Formula: volatility = std(portfolio_return_day). Needs ≥ 5 days to be meaningful."
    ),
    "sharpe_like": (
        "Risk-adjusted return (no risk-free adjustment); return per unit risk. "
        "Formula: sharpe_like = mean_daily_return / volatility. Needs ≥ 5 days."
    ),
}

API_URL = os.environ.get("DASHBOARD_API_URL", "http://127.0.0.1:8000/predict")


# ---------------------------------------------------------------------------
# Data loading & aggregation (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_artifacts() -> tuple[pd.DataFrame, dict, str]:
    root = get_project_root()
    cfg = load_config()
    art_dir = root / cfg["paths"]["dashboard_artifacts"]
    trades = pd.read_csv(art_dir / "trades.csv", parse_dates=["date"])
    with open(art_dir / "meta.json") as f:
        meta = json.load(f)
    github_url = cfg.get("dashboard", {}).get("github_url", "#")
    return trades, meta, github_url


@st.cache_data
def compute_daily(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-IPO trades into per-day portfolio metrics + baselines."""
    def _day_row(g: pd.DataFrame) -> pd.Series:
        above = g[g["prob"] >= 0.5]
        return pd.Series({
            "num_ipos": len(g),
            "strategy_return_pct": g["contribution_pct"].sum(),
            "strategy_allocation": g["weight"].sum(),
            "equal_weight_return_pct": g["actual_return_pct"].mean(),
            "equal_weight_above_05_pct": (
                above["actual_return_pct"].mean() if len(above) else np.nan
            ),
        })

    daily = trades.groupby("date").apply(_day_row).reset_index()
    return daily.sort_values("date").reset_index(drop=True)


@st.cache_data
def aggregate_by_period(daily: pd.DataFrame, period_code: str) -> pd.DataFrame:
    """Roll up daily series into period-level metrics, including cum_return."""
    if daily.empty:
        return daily.assign(period=pd.Series(dtype="datetime64[ns]"))

    df = daily.copy()
    df["period"] = df["date"].dt.to_period(period_code).dt.to_timestamp()

    def _agg(g: pd.DataFrame) -> pd.Series:
        rets = g["strategy_return_pct"].values
        allocs = g["strategy_allocation"].values
        num_days = len(rets)
        mean_r = float(np.mean(rets)) if num_days else 0.0
        vol = float(np.std(rets)) if num_days else 0.0
        return pd.Series({
            "num_days": num_days,
            "num_ipos": int(g["num_ipos"].sum()),
            "mean_daily_return": mean_r,
            "win_rate": float(np.mean(rets > 0)) if num_days else 0.0,
            "avg_allocation": float(np.mean(allocs)) if num_days else 0.0,
            "pct_days_traded": float(np.mean(allocs > 0)) if num_days else 0.0,
            "volatility": vol,
            "sharpe_like": (mean_r / vol) if vol > 0 else 0.0,
        })

    agg = df.groupby("period").apply(_agg).reset_index()

    # Cumulative return is a running total across the raw daily series,
    # sampled at each period's last day.
    cum = (1 + daily["strategy_return_pct"] / 100).cumprod() - 1
    cum_df = pd.DataFrame({
        "date": daily["date"],
        "period": daily["date"].dt.to_period(period_code).dt.to_timestamp(),
        "cum_pct": cum * 100,
    })
    cum_by_period = cum_df.groupby("period")["cum_pct"].last().reset_index()
    cum_by_period = cum_by_period.rename(columns={"cum_pct": "cum_return"})
    agg = agg.merge(cum_by_period, on="period", how="left")

    return agg.sort_values("period").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def window_bounds(
    daily: pd.DataFrame,
    selected_year,
    selected_quarter: str,
    selected_month: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if selected_year == "All" or daily.empty:
        return daily["date"].min(), daily["date"].max()

    y = int(selected_year)
    if selected_month != "All":
        m = int(selected_month)
        start = pd.Timestamp(year=y, month=m, day=1)
        end = start + pd.offsets.MonthEnd(0)
    elif selected_quarter != "All":
        q = int(selected_quarter[1])
        start = pd.Timestamp(year=y, month=(q - 1) * 3 + 1, day=1)
        end = start + pd.offsets.QuarterEnd(0)
    else:
        start = pd.Timestamp(year=y, month=1, day=1)
        end = pd.Timestamp(year=y, month=12, day=31)
    return start, end


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def render_header(meta: dict, github_url: str) -> None:
    st.title("IPO Listing Gain Prediction")
    st.markdown(
        f"A machine-learning trading strategy that predicts the probability of Indian IPOs "
        f"exceeding a **{meta['listing_gain_threshold_perc']}% listing-day gain**, then allocates "
        f"capital across each day's IPOs proportional to `probability^α` above a minimum-confidence "
        f"threshold. Results below are from a walk-forward backtest over "
        f"**{meta['num_ipos_backtested']} IPOs** between "
        f"**{meta['data_range']['from']}** and **{meta['data_range']['to']}**. "
        f"[View on GitHub]({github_url})."
    )


def render_holdout_kpis(meta: dict) -> None:
    hd = meta.get("holdout", {})
    st.subheader("Holdout set metrics (unseen data)")
    cols = st.columns(7)
    cols[0].metric("Days", hd.get("num_days", "—"))
    cols[1].metric("IPOs", hd.get("num_ipos", "—"))
    cols[2].metric("Mean daily return", f"{hd.get('mean_daily_return', 0):+.2f}%")
    cols[3].metric("Win rate", f"{hd.get('win_rate', 0):.1%}")
    cols[4].metric("Avg allocation", f"{hd.get('avg_allocation', 0):.2f}")
    cols[5].metric("Volatility", f"{hd.get('volatility', 0):.2f}")
    cols[6].metric("Sharpe-like", f"{hd.get('sharpe_like', 0):+.2f}")


def render_body(trades: pd.DataFrame, daily: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("Strategy performance over time")

    # Placeholders so navigation dropdowns can be rendered AFTER the chart
    # while their values are read BEFORE the chart builds.
    top_box = st.container()
    chart_box = st.container()
    window_kpis_box = st.container()
    nav_box = st.container()
    drilldown_box = st.container()

    # Top: metric + timeline
    with top_box:
        c1, c2 = st.columns(2)
        metric_key = c1.selectbox(
            "Metric",
            options=list(METRICS.keys()),
            format_func=lambda k: METRICS[k],
            index=list(METRICS.keys()).index(DEFAULT_METRIC),
        )
        period_label = c2.selectbox(
            "Timeline",
            options=list(PERIODS.keys()),
            index=list(PERIODS.keys()).index(DEFAULT_PERIOD),
        )
    period_code = PERIODS[period_label]

    # Bottom nav (declared here so values are available to chart)
    with nav_box:
        years = sorted({d.year for d in daily["date"]})
        n1, n2, n3 = st.columns(3)
        selected_year = n1.selectbox("Jump to year", ["All"] + years, index=0)
        if selected_year == "All":
            n2.selectbox("Quarter", ["All"], index=0, disabled=True)
            n3.selectbox("Month", ["All"], index=0, disabled=True)
            selected_quarter, selected_month = "All", "All"
        else:
            selected_quarter = n2.selectbox("Quarter", ["All", "Q1", "Q2", "Q3", "Q4"], index=0)
            selected_month = n3.selectbox(
                "Month", ["All"] + [str(m) for m in range(1, 13)], index=0
            )

    # Filter daily to the selected window, then aggregate
    start, end = window_bounds(daily, selected_year, selected_quarter, selected_month)
    daily_view = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    agg_view = aggregate_by_period(daily_view, period_code)

    # Chart
    with chart_box:
        fig = _build_metric_chart(agg_view, metric_key, period_label)
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="main_chart",
        )

    # Window KPIs
    with window_kpis_box:
        st.caption("Averaged metrics for data currently in view:")
        _render_window_kpis(daily_view)

    # Drilldown
    with drilldown_box:
        points = _extract_selected_points(event)
        if points:
            sel_x = points[0].get("x")
            if sel_x is not None:
                _render_drilldown(trades, daily, pd.Timestamp(sel_x), period_code, period_label)

    # Baselines
    st.markdown("---")
    st.subheader("Baseline comparison")
    st.caption(
        "**Strategy**: probability-weighted allocation above `t_min`. "
        "**Equal-weight (all)**: 1/n on every IPO that day, uncapped. "
        "**Equal-weight (prob ≥ 0.5)**: 1/k on IPOs with prob ≥ 0.5; cash on zero-k days."
    )
    _render_baseline_chart(daily)


def _extract_selected_points(event) -> list:
    """Robustly pull selected points regardless of Streamlit return shape."""
    if event is None:
        return []
    try:
        return event["selection"]["points"]
    except (KeyError, TypeError):
        pass
    try:
        return event.selection.get("points", [])
    except AttributeError:
        return []


def _build_metric_chart(view: pd.DataFrame, metric_key: str, period_label: str) -> go.Figure:
    fig = go.Figure()

    if view.empty:
        fig.update_layout(
            xaxis_title=period_label,
            yaxis_title=METRICS[metric_key],
            annotations=[dict(
                text="No data in selected window",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False,
            )],
            height=450,
        )
        return fig

    values, hovers = [], []
    min_days = MIN_DAYS_FOR.get(metric_key, 1)
    for _, row in view.iterrows():
        period_label_str = _fmt_period(row["period"], period_label)
        if metric_key == "cum_return":
            val = float(row.get("cum_return", 0.0)) if pd.notna(row.get("cum_return", np.nan)) else 0.0
            hover = (
                f"<b>{period_label_str}</b><br>"
                f"{METRICS[metric_key]}: {val:+.2f}%<br>"
                f"Days: {int(row['num_days'])} · IPOs: {int(row['num_ipos'])}"
            )
        elif row["num_days"] < min_days:
            val = 0.0
            hover = (
                f"<b>{period_label_str}</b><br>"
                f"<i>Not defined</i>: needs ≥{min_days} traded days, had {int(row['num_days'])}<br>"
                f"IPOs: {int(row['num_ipos'])}"
            )
        else:
            val = float(row[metric_key])
            hover = (
                f"<b>{period_label_str}</b><br>"
                f"{METRICS[metric_key]}: {val:+.3f}<br>"
                f"Days: {int(row['num_days'])} · IPOs: {int(row['num_ipos'])}"
            )
        values.append(val)
        hovers.append(hover)

    fig.add_trace(go.Scatter(
        x=view["period"],
        y=values,
        mode="lines+markers",
        marker=dict(size=10, line=dict(width=1, color="white")),
        line=dict(width=2),
        hovertext=hovers,
        hovertemplate="%{hovertext}<extra></extra>",
        name=METRICS[metric_key],
    ))

    fig.update_layout(
        xaxis_title=period_label,
        yaxis_title=METRICS[metric_key],
        hovermode="closest",
        clickmode="event+select",
        margin=dict(l=20, r=20, t=30, b=20),
        height=450,
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", griddash="dot")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", griddash="dot", zeroline=True, zerolinecolor="gray")
    return fig


def _fmt_period(period: pd.Timestamp, period_label: str) -> str:
    p = pd.Timestamp(period)
    if period_label == "Monthly":
        return p.strftime("%Y-%m")
    if period_label == "Quarterly":
        q = (p.month - 1) // 3 + 1
        return f"{p.year}-Q{q}"
    return p.strftime("%Y")


def _render_window_kpis(daily_view: pd.DataFrame) -> None:
    if daily_view.empty:
        st.info("No data in selected window.")
        return

    returns = daily_view["strategy_return_pct"].values
    allocs = daily_view["strategy_allocation"].values
    num_days = len(returns)
    num_ipos = int(daily_view["num_ipos"].sum())
    mean_r = float(np.mean(returns))
    vol = float(np.std(returns))
    wr = float(np.mean(returns > 0))
    alloc = float(np.mean(allocs))
    traded = float(np.mean(allocs > 0))
    shr = (mean_r / vol) if vol > 0 else 0.0
    cum = float(((1 + returns / 100).prod() - 1) * 100)

    cols = st.columns(9)
    cols[0].metric("Days", num_days)
    cols[1].metric("IPOs", num_ipos)
    cols[2].metric("Cum return", f"{cum:+.2f}%")
    cols[3].metric("Mean return", f"{mean_r:+.2f}%")
    cols[4].metric("Win rate", f"{wr:.1%}")
    cols[5].metric("Avg alloc", f"{alloc:.2f}")
    cols[6].metric("% Days traded", f"{traded:.1%}")
    cols[7].metric("Volatility", f"{vol:.2f}")
    cols[8].metric("Sharpe-like", f"{shr:+.2f}")


def _render_drilldown(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    sel_period: pd.Timestamp,
    period_code: str,
    period_label: str,
) -> None:
    start = sel_period.to_period(period_code).to_timestamp(how="start")
    end = sel_period.to_period(period_code).to_timestamp(how="end")

    period_daily = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    period_trades = trades[(trades["date"] >= start) & (trades["date"] <= end)]

    if period_trades.empty:
        return

    st.markdown("---")
    st.subheader(f"Drilldown · {_fmt_period(sel_period, period_label)}")

    _render_window_kpis(period_daily)

    with st.expander("Show individual IPO trades (grouped by day)", expanded=True):
        show = period_trades.sort_values(
            ["date", "contribution_pct"], ascending=[True, False]
        ).copy()
        show["date"] = show["date"].dt.strftime("%Y-%m-%d")
        show["prob"] = show["prob"].round(4)
        show["weight"] = show["weight"].round(4)
        show["actual_return_pct"] = show["actual_return_pct"].round(2)
        show["contribution_pct"] = show["contribution_pct"].round(4)
        show = show.rename(columns={
            "date": "Date", "company": "Company", "prob": "Prob",
            "weight": "Weight", "actual_return_pct": "Actual Return %",
            "contribution_pct": "Contribution %", "allocated": "Allocated",
        })[["Date", "Company", "Prob", "Weight",
            "Actual Return %", "Contribution %", "Allocated"]]
        st.dataframe(show, use_container_width=True, hide_index=True)


def _render_baseline_chart(daily: pd.DataFrame) -> None:
    d = daily.copy()
    d["strategy_cum"] = ((1 + d["strategy_return_pct"] / 100).cumprod() - 1) * 100
    d["ew_cum"] = ((1 + d["equal_weight_return_pct"].fillna(0) / 100).cumprod() - 1) * 100
    d["ew05_cum"] = ((1 + d["equal_weight_above_05_pct"].fillna(0) / 100).cumprod() - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["strategy_cum"], mode="lines", name="Strategy"))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ew_cum"], mode="lines", name="Equal-weight (all)"))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["ew05_cum"], mode="lines",
        name="Equal-weight (prob ≥ 0.5)",
    ))
    fig.update_layout(
        yaxis_title="Cumulative return (%)",
        xaxis_title="Date",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", griddash="dot")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", griddash="dot", zeroline=True, zerolinecolor="gray")
    st.plotly_chart(fig, use_container_width=True)


def render_metric_descriptions() -> None:
    st.markdown("---")
    st.subheader("Metric definitions")
    for key, label in METRICS.items():
        with st.expander(label):
            st.write(METRIC_DESCRIPTIONS[key])


def render_example_day(meta: dict) -> None:
    ex = meta.get("example_day")
    if not ex:
        return
    st.markdown("---")
    st.subheader(f"Example trading day · {ex['date']}")
    st.caption(
        "A real day sampled from the backtest showing each IPO's predicted probability, "
        "allocated weight, and realized listing-day return."
    )
    df = pd.DataFrame(ex["ipos"]).rename(columns={
        "company": "Company", "prob": "Prob", "weight": "Weight",
        "actual_return_pct": "Actual Return %",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_api_form() -> None:
    st.markdown("---")
    st.subheader("Try the prediction API")
    st.caption(
        f"POSTs to `{API_URL}`. Start the FastAPI server with `python -m app.main` "
        f"(override via env var `DASHBOARD_API_URL`)."
    )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        nii = c1.number_input("NII (x)", min_value=0.0, value=50.0)
        qib = c2.number_input("QIB (x)", min_value=0.0, value=30.0)
        retail = c3.number_input("Retail (x)", min_value=0.0, value=10.0)

        c1, c2, c3 = st.columns(3)
        total = c1.number_input("Total (x)", min_value=0.01, value=25.0)
        year = c2.number_input("Year", min_value=2000, max_value=2100, value=2025)
        issue_amount = c3.number_input("Issue amount (Rs. cr.)", min_value=0.01, value=500.0)

        c1, c2, c3 = st.columns(3)
        price_band_high = c1.number_input("Price band high (Rs.)", min_value=0.01, value=500.0)
        price_band_low = c2.number_input("Price band low (Rs.)", min_value=0.01, value=475.0)
        gmp_raw = c3.text_input("GMP (Rs., blank if unknown)", value="")

        submitted = st.form_submit_button("Predict")

    if submitted:
        gmp = None
        if gmp_raw.strip():
            try:
                gmp = float(gmp_raw)
            except ValueError:
                st.error(f"GMP must be numeric or blank; got {gmp_raw!r}")
                return

        payload = {"ipos": [{
            "nii": nii, "qib": qib, "retail": retail, "total": total,
            "year": int(year), "issue_amount": issue_amount,
            "price_band_high": price_band_high, "price_band_low": price_band_low,
            "gmp": gmp,
        }]}

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
        except Exception as e:
            st.error(f"Could not reach API: {e}")
            return

        if resp.status_code == 200:
            st.success("Prediction received")
            st.json(resp.json())
        else:
            st.error(f"API returned {resp.status_code}: {resp.text}")


def render_disclaimers(meta: dict) -> None:
    st.markdown("---")
    st.subheader("Disclaimers")
    st.warning(
        f"- Assumes **100% IPO allotment** — real retail allotments are lottery-based.\n"
        f"- Strategy parameters: **α = {meta['alpha']:.4f}**, "
        f"**t_min = {meta['t_min']:.4f}**, "
        f"**listing-gain threshold = {meta['listing_gain_threshold_perc']}%**.\n"
        f"- Last training date: **{meta.get('last_training_date', '—')}**.\n"
        f"- Transaction costs and taxes assumed zero.\n"
        f"- Not investment advice. Past performance does not guarantee future results."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="IPO Listing Gain Prediction",
        layout="wide",
    )

    try:
        trades, meta, github_url = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Dashboard artifacts not found. Build them first:\n\n"
            "    python -m dashboard.build_artifacts"
        )
        return

    daily = compute_daily(trades)

    render_header(meta, github_url)
    render_holdout_kpis(meta)
    render_body(trades, daily)
    render_metric_descriptions()
    render_example_day(meta)
    render_api_form()
    render_disclaimers(meta)


if __name__ == "__main__":
    main()
