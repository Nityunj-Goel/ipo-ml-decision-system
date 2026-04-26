"""Dashboard entrypoint — see ``dashboard/__init__.py`` for package docs."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import get_project_root, load_config


METRICS = {
    "cum_return": "Cumulative Return",
    "mean_daily_return": "Mean Daily Return",
    "win_rate": "Win Rate",
    "pct_days_traded": "% Days Traded",
    "volatility": "Volatility",
    "sharpe_like": "Sharpe-like Ratio",
}
DEFAULT_METRIC = "mean_daily_return"

# Metrics whose raw value is a fraction in [0, 1] — we multiply by 100 for display.
FRACTIONAL_METRICS = {"win_rate", "pct_days_traded"}
# Metrics already expressed in %. Displayed with a "%" suffix, no scaling.
ALREADY_PERCENT_METRICS = {"mean_daily_return", "volatility"}

PERIODS = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
DEFAULT_PERIOD = "Yearly"

# Metrics whose definition requires a minimum number of traded days per period.
MIN_DAYS_FOR = {"volatility": 5, "sharpe_like": 5}

METRIC_DESCRIPTIONS = {
    "cum_return": (
        "Compounded return from the first decision day, expressed in %. "
        "Formula: cum_return_t = (∏(1 + r_d / 100) − 1) × 100 over all daily portfolio returns up to t."
    ),
    "mean_daily_return": (
        "Average portfolio return per decision day; overall profitability. "
        "Formula: mean_daily_return = (1 / N) · Σ (Σ wᵢ × rᵢ)_day."
    ),
    "win_rate": (
        "Fraction of days with positive portfolio return; consistency. "
        "Formula: win_rate = (#days with portfolio_return_day > 0) / N."
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

# This won't likely work in prod
API_URL = os.environ.get("DASHBOARD_API_URL", "http://127.0.0.1:8000/predict")


@st.cache_data
def load_artifacts() -> tuple[pd.DataFrame, dict, str]:
    root = get_project_root()
    cfg = load_config()
    art_dir = root / cfg["paths"]["dashboard_artifacts"]
    trades = pd.read_csv(art_dir / "trades.csv", parse_dates=["date"])
    # Derived: daily-portfolio contribution.
    # Stripped from CSV to avoid storing a redundant column.
    trades["contribution_pct"] = (
        trades["weight"] * trades["allotment_ratio"] * trades["actual_return_pct"]
    )
    with open(art_dir / "meta.json") as f:
        meta = json.load(f)
    github_url = cfg.get("dashboard", {}).get("github_url", "#")
    return trades, meta, github_url


@st.cache_data
def compute_daily(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-IPO trades into per-day portfolio metrics + baselines."""
    def _day_row(g: pd.DataFrame) -> pd.Series:
        # Equal-weight baseline: 1/n on every IPO that day, allotment-adjusted
        # (1/n × allotmentᵢ × rᵢ summed across IPOs == mean of allotment × return).
        ew_contrib = g["allotment_ratio"] * g["actual_return_pct"]
        return pd.Series({
            "num_ipos": len(g),
            "strategy_return_pct": g["contribution_pct"].sum(),
            "strategy_allocation": g["weight"].sum(),
            "equal_weight_return_pct": ew_contrib.mean(),
        })

    daily = trades.groupby("date").apply(_day_row).reset_index()
    return daily.sort_values("date").reset_index(drop=True)


@st.cache_data
def aggregate_by_period(daily: pd.DataFrame, period_code: str) -> pd.DataFrame:
    """Roll up daily series into period-level metrics, including cum_return."""
    if daily.empty:
        return pd.DataFrame(columns=[
            "period", "num_days", "num_ipos", "mean_daily_return",
            "win_rate", "avg_allocation", "pct_days_traded",
            "volatility", "sharpe_like", "cum_return",
        ])

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

    # Cumulative return: running total across the raw daily series, sampled at period end.
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
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if selected_year == "All" or daily.empty:
        return daily["date"].min(), daily["date"].max()

    y = int(selected_year)
    if selected_quarter != "All":
        q = int(selected_quarter[1])
        start = pd.Timestamp(year=y, month=(q - 1) * 3 + 1, day=1)
        end = start + pd.offsets.QuarterEnd(0)
    else:
        start = pd.Timestamp(year=y, month=1, day=1)
        end = pd.Timestamp(year=y, month=12, day=31)
    return start, end


def _reset_nav_defaults(period_code: str, period_label: str, max_year: int) -> None:
    """Called when the timeline dropdown changes."""
    st.session_state["nav_year"] = "All" if period_label == "Yearly" else max_year
    st.session_state["nav_quarter"] = "All"
    st.session_state["_prev_period"] = period_code


def _shift_year(delta: int, min_year: int, max_year: int) -> None:
    cur = st.session_state.get("nav_year")
    if not isinstance(cur, int):
        return
    st.session_state["nav_year"] = max(min_year, min(max_year, cur + delta))


# ---------------------------------------------------------------------------
# Chart / KPI helpers
# ---------------------------------------------------------------------------

def _extract_selected_points(event) -> list:
    """Robust across Streamlit return shapes for on_select='rerun'."""
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


def _fmt_period(period, period_label: str) -> str:
    p = pd.Timestamp(period)
    if period_label == "Monthly":
        return p.strftime("%Y-%m")
    if period_label == "Quarterly":
        q = (p.month - 1) // 3 + 1
        return f"{p.year}-Q{q}"
    return p.strftime("%Y")


def _fmt_metric_value(metric_key: str, value: float) -> str:
    """Format a metric for hover / KPI display."""
    if metric_key == "cum_return":
        return f"{value:+.2f}%"
    if metric_key in FRACTIONAL_METRICS:
        return f"{value * 100:.1f}%"
    if metric_key in ALREADY_PERCENT_METRICS:
        return f"{value:+.2f}%" if metric_key == "mean_daily_return" else f"{value:.2f}%"
    return f"{value:+.3f}"  # sharpe_like


def _metric_y_title(metric_key: str) -> str:
    if metric_key == "cum_return":
        return "Cumulative Return (%)"
    if metric_key in FRACTIONAL_METRICS or metric_key in ALREADY_PERCENT_METRICS:
        return f"{METRICS[metric_key]} (%)"
    return METRICS[metric_key]


def _build_metric_chart(view: pd.DataFrame, metric_key: str, period_label: str) -> go.Figure:
    fig = go.Figure()

    if view.empty:
        fig.update_layout(
            xaxis_title=period_label,
            yaxis_title=_metric_y_title(metric_key),
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
        plabel = _fmt_period(row["period"], period_label)
        num_days = int(row["num_days"])
        num_ipos = int(row["num_ipos"])

        if metric_key == "cum_return":
            raw = float(row.get("cum_return", 0.0)) if pd.notna(row.get("cum_return", np.nan)) else 0.0
            values.append(raw)
            hovers.append(
                f"<b>{plabel}</b><br>"
                f"Cumulative Return: {raw:+.2f}%<br>"
                f"Days: {num_days} · IPOs: {num_ipos}"
            )
        elif num_days < min_days:
            values.append(0.0)
            hovers.append(
                f"<b>{plabel}</b><br>"
                f"<i>Not defined</i>: needs ≥{min_days} traded days, had {num_days}<br>"
                f"IPOs: {num_ipos}"
            )
        else:
            raw = float(row[metric_key])
            display_val = raw * 100 if metric_key in FRACTIONAL_METRICS else raw
            values.append(display_val)
            hovers.append(
                f"<b>{plabel}</b><br>"
                f"{METRICS[metric_key]}: {_fmt_metric_value(metric_key, raw)}<br>"
                f"Days: {num_days} · IPOs: {num_ipos}"
            )

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

    layout_kwargs = dict(
        xaxis_title=period_label,
        yaxis_title=_metric_y_title(metric_key),
        hovermode="closest",
        clickmode="event+select",
        height=450,
    )
    if period_label != "Yearly":
        years = sorted({pd.Timestamp(p).year for p in view["period"]})
        year_str = str(years[0]) if len(years) == 1 else f"{years[0]}–{years[-1]}"
        layout_kwargs["title"] = dict(
            text=year_str, x=0.5, xanchor="center", font=dict(size=18),
        )
        layout_kwargs["margin"] = dict(l=20, r=20, t=50, b=60)
    else:
        layout_kwargs["margin"] = dict(l=20, r=20, t=30, b=60)
    fig.update_layout(**layout_kwargs)
    grid = "lightgray"
    tickvals = list(view["period"])
    ticktext = [_fmt_period(p, period_label) for p in view["period"]]
    fig.update_xaxes(
        showgrid=True, gridcolor=grid, griddash="dot",
        tickmode="array", tickvals=tickvals, ticktext=ticktext,
        tickangle=-45,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=grid, griddash="dot",
        zeroline=True, zerolinecolor="gray",
        type="linear",
        rangemode="tozero",
    )
    return fig


def _render_window_kpis(daily_view: pd.DataFrame) -> None:
    if daily_view.empty:
        st.info("No data in selected window.")
        return

    returns = daily_view["strategy_return_pct"].values
    allocs = daily_view["strategy_allocation"].values
    num_ipo_days = len(returns)  # days on which IPO(s) were available
    calendar_days = int((daily_view["date"].max() - daily_view["date"].min()).days) + 1
    num_ipos = int(daily_view["num_ipos"].sum())
    mean_r = float(np.mean(returns))
    vol = float(np.std(returns))
    wr = float(np.mean(returns > 0))
    traded = float(np.mean(allocs > 0))
    shr = (mean_r / vol) if vol > 0 else 0.0
    cum = float(((1 + returns / 100).prod() - 1) * 100)

    avg_per_calendar = (mean_r * num_ipo_days / calendar_days) if calendar_days else 0.0

    r1 = st.columns(5)
    r1[0].metric("Calendar Days", calendar_days)
    r1[1].metric(
        "IPO Trade Days", num_ipo_days,
        help="Trading days on which at least one IPO was available.",
    )
    r1[2].metric("IPOs Evaluated", num_ipos)
    r1[3].metric(
        "Cumulative Return", f"{cum:+.2f}%",
        help="Total compounded portfolio return over the period.",
    )
    r1[4].metric(
        "Avg Return per Trade Day", f"{mean_r:+.2f}%",
        help="Average portfolio return across IPO trading days.",
    )

    r2 = st.columns(5)
    r2[0].metric(
        "Avg Return per Calendar Day", f"{avg_per_calendar:+.2f}%",
        help="Avg return adjusted for inactive days (no IPOs). "
             "Reflects true time-based performance.",
    )
    r2[1].metric("Win Rate (Trade Days)", f"{wr * 100:.1f}%")
    r2[2].metric(
        "% Trade Days Deployed", f"{traded * 100:.1f}%",
        help="Fraction of IPO trading days on which capital was actually allocated.",
    )
    r2[3].metric("Return Volatility (Trade Days)", f"{vol:.2f}%")
    r2[4].metric("Sharpe Ratio (Trade-Day, Approx.)", f"{shr:+.2f}")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def render_header(meta: dict, github_url: str) -> None:
    st.title("IPO Investment Decision System Performance Dashboard")
    st.markdown(
        f"A machine-learning based trading strategy that predicts the probability of Indian IPOs exceeding "
        f"a **{meta['listing_gain_threshold_perc']}% listing-day gain** and constructs a daily portfolio by "
        f"**equally splitting capital across all IPOs whose predicted probability passes a learned threshold**. "
        f"Rather than forecasting exact returns, the system treats model outputs as confidence signals to filter "
        f"the daily opportunity set down to high-quality bets. "
        f"Results below are from a walk-forward backtest over **{meta['num_ipos_backtested']} IPOs** between "
        f"**{meta['data_range']['from']}** and **{meta['data_range']['to']}**. "
        f"For more details around the modeling approach and decision logic, "
        f"check out the [GitHub repository]({github_url}) 🔗."
    )
    st.caption("Tip: switch between light and dark themes from the ⋮ menu (top-right) → Settings → Choose app theme.")


def render_holdout_kpis(meta: dict) -> None:
    hd = meta.get("holdout", {})
    st.subheader("Live Simulation Performance on unseen data")

    r1 = st.columns(5)
    r1[0].metric("Calendar Days", hd.get("calendar_days", "—"))
    r1[1].metric(
        "IPO Trade Days", hd.get("num_days", "—"),
        help="Trading days on which at least one IPO was available.",
    )
    r1[2].metric("IPOs Evaluated", hd.get("num_ipos", "—"))
    r1[3].metric(
        "Cumulative  Return",
        f"{hd.get('cum_return', 0):+.2f}%",
        help="Total compounded portfolio return over the period.",
    )
    r1[4].metric(
        "Avg Return per Trade Day", f"{hd.get('mean_daily_return', 0):+.2f}%",
        help="Average portfolio return across IPO trading days.",
    )

    r2 = st.columns(5)
    r2[0].metric(
        "Avg Return per Calendar Day",
        f"{hd.get('avg_return_per_calendar_day', 0):+.2f}%",
        help="Avg return adjusted for inactive days (no IPOs). "
             "Reflects true time-based performance.",
    )
    r2[1].metric("Win Rate (Trade Days)", f"{hd.get('win_rate', 0) * 100:.1f}%")
    r2[2].metric(
        "% Trade Days Deployed", f"{hd.get('pct_days_traded', 0) * 100:.1f}%",
        help="Fraction of IPO trading days on which capital was actually allocated.",
    )
    r2[3].metric("Return Volatility (Trade Days)", f"{hd.get('volatility', 0):.2f}%")
    r2[4].metric("Sharpe Ratio (Trade-Day, Approx.)", f"{hd.get('sharpe_like', 0):+.2f}")


def render_body(trades: pd.DataFrame, daily: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("Strategy performance over time")
    st.caption(
        "💡 Tip: click any point on the chart and scroll down to see the "
        "individual IPO trades that produced it. Use the ◄ ► arrows beside "
        "the chart to navigate between consecutive years when viewing a monthly or quarterly timeline."
    )

    # Top: metric + timeline selectors. The ? tooltip next to the "Metric"
    # label shows definitions for all metrics.
    metric_help = "\n\n".join(
        f"**{METRICS[k]}** — {METRIC_DESCRIPTIONS[k]}" for k in METRICS
    )
    c1, c2 = st.columns(2)
    metric_key = c1.selectbox(
        "Metric",
        options=list(METRICS.keys()),
        format_func=lambda k: METRICS[k],
        index=list(METRICS.keys()).index(DEFAULT_METRIC),
        key="metric",
        help=metric_help,
    )
    period_label = c2.selectbox(
        "Timeline",
        options=list(PERIODS.keys()),
        index=list(PERIODS.keys()).index(DEFAULT_PERIOD),
        key="period",
    )
    period_code = PERIODS[period_label]

    years = sorted({d.year for d in daily["date"]})
    if not years:
        st.info("No data to display.")
        return
    min_year, max_year = years[0], years[-1]

    # Reset nav state on timeline change (also runs on first render)
    if st.session_state.get("_prev_period") != period_code:
        _reset_nav_defaults(period_code, period_label, max_year)

    is_yearly = period_label == "Yearly"
    is_quarterly = period_label == "Quarterly"

    nav_year = st.session_state.get("nav_year", "All")
    arrows_active = (not is_yearly) and isinstance(nav_year, int)
    left_disabled = (not arrows_active) or (arrows_active and nav_year <= min_year)
    right_disabled = (not arrows_active) or (arrows_active and nav_year >= max_year)

    # Arrow | chart | arrow layout
    col_l, col_c, col_r = st.columns(
        [1, 18, 1],
        vertical_alignment="center" if hasattr(st, "columns") else None,
    )
    with col_l:
        st.button(
            "◄", key="btn_prev_year", disabled=left_disabled,
            on_click=_shift_year, args=(-1, min_year, max_year),
            use_container_width=True,
        )
    with col_r:
        st.button(
            "►", key="btn_next_year", disabled=right_disabled,
            on_click=_shift_year, args=(1, min_year, max_year),
            use_container_width=True,
        )

    selected_year = st.session_state.get("nav_year", "All")
    selected_quarter = st.session_state.get("nav_quarter", "All")
    start, end = window_bounds(daily, selected_year, selected_quarter)
    daily_view = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    agg_view = aggregate_by_period(daily_view, period_code)

    with col_c:
        fig = _build_metric_chart(agg_view, metric_key, period_label)
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="main_chart",
        )

    # Jump-to dropdowns (below chart)
    n1, n2 = st.columns(2)
    n1.selectbox(
        "Jump to year",
        options=["All"] + years,
        key="nav_year",
        disabled=is_yearly,
    )
    n2.selectbox(
        "Jump to quarter",
        options=["All", "Q1", "Q2", "Q3", "Q4"],
        key="nav_quarter",
        disabled=is_yearly or is_quarterly,
    )

    # Averaged metrics for current view (below jump-to)
    st.caption("Averaged metrics for data currently in view:")
    _render_window_kpis(daily_view)

    # Drilldown (below KPIs)
    points = _extract_selected_points(event)
    if points:
        sel_x = points[0].get("x")
        if sel_x is not None:
            _render_drilldown(trades, daily, pd.Timestamp(sel_x), period_code, period_label)


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

    # Anchor so we can smoothly scroll here on chart-point click.
    st.markdown('<div id="drilldown-anchor"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        f"""
        <div style="
            margin: 8px 0 18px 0;
            padding: 14px 20px;
            background: linear-gradient(90deg,
                rgba(99,102,241,0.18) 0%,
                rgba(99,102,241,0.04) 100%);
            border-left: 6px solid #6366f1;
            border-radius: 6px;
            font-size: 1.45rem;
            font-weight: 600;
            letter-spacing: 0.2px;">
            🔍 Drilldown &nbsp;·&nbsp;
            <span style="color:#6366f1;">
              {_fmt_period(sel_period, period_label)}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_window_kpis(period_daily)

    with st.expander("Show individual IPO trades (grouped by day)", expanded=True):
        st.caption(
            "Each day rendered as its own block with a date header. "
            "🟢 Profitable Trade · 🔴 Losing Trade · ⚪ Not Traded (weight = 0)."
        )
        show = period_trades.sort_values(
            ["date", "contribution_pct"], ascending=[True, False]
        ).copy()
        show["date_str"] = show["date"].dt.strftime("%Y-%m-%d (%a)")
        show["prob"] = show["prob"].round(4)
        show["weight"] = show["weight"].round(4)
        show["actual_return_pct"] = show["actual_return_pct"].round(2)
        show["contribution_pct"] = show["contribution_pct"].round(4)

        for date_str, day_df in show.groupby("date_str", sort=True):
            day_portfolio_ret = float(day_df["contribution_pct"].sum())
            color = "#1fa868" if day_portfolio_ret > 0 else ("#d93a3a" if day_portfolio_ret < 0 else "#888")
            st.markdown(
                f"""
                <div style="margin: 18px 0 6px 0; padding: 8px 14px;
                            background: rgba(100,100,100,0.08);
                            border-left: 6px solid {color};
                            border-radius: 4px; font-size: 1.05rem;">
                    📅 <b>{date_str}</b> &nbsp;·&nbsp; {len(day_df)} IPO(s)
                    &nbsp;·&nbsp; Day Return:
                    <span style="color:{color};"><b>{day_portfolio_ret:+.2f}%</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            day_view = day_df.rename(columns={
                "company": "Company",
                "prob": "Probability",
                "weight": "Weight",
                "actual_return_pct": "Listing Gain (%)",
                "contribution_pct": "Portfolio Contribution (%)",
                "allocated": "Traded",
            })[["Company", "Probability", "Weight",
                "Listing Gain (%)", "Portfolio Contribution (%)",
                "Traded"]].reset_index(drop=True)
            st.write(_style_drilldown(day_view).to_html(), unsafe_allow_html=True)

    # Smooth-scroll the user to this section on rerun (triggered by point click).
    components.html(
        """
        <script>
        const el = window.parent.document.getElementById('drilldown-anchor');
        if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        </script>
        """,
        height=0,
    )


def _style_drilldown(df: pd.DataFrame):
    """Color rows bright green/red by profitability; gray for non-traded."""
    def row_styles(row: pd.Series) -> list[str]:
        if "Traded" in row.index:
            allocated = bool(row["Traded"])
        else:
            allocated = float(row.get("Weight", 0) or 0) > 0
        ret = row["Listing Gain (%)"]
        if not allocated:
            base = ("background-color: rgba(130,130,130,0.18);"
                    " color: #888; font-style: italic;")
        elif pd.notna(ret) and ret > 0:
            base = "background-color: rgba(40,200,100,0.55); color: #0a2a14;"
        else:
            base = "background-color: rgba(240,70,70,0.55); color: #2a0a0a;"
        return [base] * len(row)

    styler = df.style.apply(row_styles, axis=1)
    styler = styler.set_table_styles([
        {"selector": "th", "props": "text-align: left; padding: 6px 10px;"
                                    " background: rgba(100,100,100,0.15);"},
        {"selector": "td", "props": "padding: 6px 10px;"},
        {"selector": "table", "props": "border-collapse: collapse; width: 100%;"
                                       " margin-bottom: 8px;"},
    ])
    styler = styler.hide(axis="index")
    return styler


def render_baseline_chart(daily: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("Baseline comparison")
    st.caption(
        "**Strategy**: 1/k on IPOs with prob ≥ learned `t_min`. "
        "**Equal-weight (all)**: 1/n on every IPO that day (no filter)."
    )
    d = daily.copy()
    d["strategy_cum"] = ((1 + d["strategy_return_pct"] / 100).cumprod() - 1) * 100
    d["ew_cum"] = ((1 + d["equal_weight_return_pct"].fillna(0) / 100).cumprod() - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["strategy_cum"], mode="lines", name="Strategy"))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ew_cum"], mode="lines", name="Equal-weight (all)"))
    fig.update_layout(
        yaxis_title="Cumulative Return (%)",
        xaxis_title="Date",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.6)",
        ),
    )
    grid = "lightgray"
    fig.update_xaxes(showgrid=True, gridcolor=grid, griddash="dot")
    fig.update_yaxes(
        showgrid=True, gridcolor=grid, griddash="dot",
        zeroline=True, zerolinecolor="gray", rangemode="tozero",
    )
    st.plotly_chart(fig, use_container_width=True)


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
    df = pd.DataFrame(ex["ipos"])
    # Derive Portfolio Contribution % to match the drilldown schema. allotment_ratio
    # is available when example_day is regenerated against the latest build_artifacts.
    if "allotment_ratio" in df.columns:
        df["contribution_pct"] = (
            df["weight"] * df["allotment_ratio"] * df["actual_return_pct"]
        ).round(4)
    else:
        df["contribution_pct"] = (df["weight"] * df["actual_return_pct"]).round(4)
    df = df.rename(columns={
        "company": "Company",
        "prob": "Probability",
        "weight": "Weight",
        "actual_return_pct": "Listing Gain (%)",
        "contribution_pct": "Portfolio Contribution (%)",
        "allocated": "Traded",
    })
    cols = ["Company", "Probability", "Weight",
            "Listing Gain (%)", "Portfolio Contribution (%)"]
    if "Traded" in df.columns:
        cols.append("Traded")
    df = df[cols]
    st.caption("🟢 Profitable Trade · 🔴 Losing Trade · ⚪ Not Traded (weight = 0).")
    st.write(_style_drilldown(df).to_html(), unsafe_allow_html=True)


def render_api_form() -> None:
    st.markdown("---")
    st.subheader("Try the prediction API")
    st.caption(
        "Test the live model end-to-end. Fill in an IPO's public details as they "
        "stand at the close of its bidding window and submit. The API returns the "
        "model's predicted probability of a positive listing-day gain along with "
        "the portfolio weight the strategy would assign on the spot."
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


def render_methodology(meta: dict) -> None:
    """Methodology + assumptions, shown right after the holdout KPI strip
    so readers see the framing alongside the numbers."""
    st.info(
        f"**📈 How the strategy works**\n\n"
        f"- Investments are simulated from the **HNI (Non-Institutional Investor)** "
        f"category. Per-IPO allotment is approximated as "
        f"`1 / max(1, NII subscription multiple)` — oversubscribed issues yield a "
        f"pro-rata fraction. Each IPO's contribution to the daily portfolio return is "
        f"`weight × allotment × listing-day return`.\n"
        f"- On any given day, capital is split **equally** across all IPOs whose "
        f"predicted probability exceeds `t_min`.\n"
        f"- Strategy parameters: **t_min = {meta['t_min']:.4f}**, "
        f"**listing-gain threshold = {meta['listing_gain_threshold_perc']}%**.\n"
        f"- Last training date: **{meta.get('last_training_date', '—')}**.\n"
        f"- Transaction costs and taxes assumed zero."
    )


def render_disclaimer(github_url: str) -> None:
    st.markdown("---")
    st.caption(
        f"📂 Source code & methodology: [GitHub repository]({github_url})."
    )
    st.caption(
        "** Not investment advice. Past performance does not guarantee future results."
    )


def main() -> None:
    st.set_page_config(
        page_title="IPO Investment Decision System Performance Dashboard",
        layout="wide",
    )

    # Hint that chart points are clickable: pointer cursor over plotly plot area.
    # Plotly's drag layer (.nsewdrag) captures pointer events and overrides marker
    # cursor styles — so we style it directly. Result: finger cursor anywhere over
    # the plotting area, signalling interactivity.
    st.markdown(
        """
        <style>
        .js-plotly-plot .nsewdrag,
        .js-plotly-plot .nsewdrag.drag,
        .js-plotly-plot .plot .scatterlayer .trace .points path {
            cursor: pointer !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
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
    render_methodology(meta)
    render_body(trades, daily)
    render_baseline_chart(daily)
    render_example_day(meta)
    render_api_form()
    render_disclaimer(github_url)


if __name__ == "__main__":
    main()
