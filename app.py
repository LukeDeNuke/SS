"""Live Streamlit dashboard for the stock scanner."""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from stock_scanner import FILTERS, POLL_SECONDS, WATCHLIST, apply_filters, fetch_snapshot


st.set_page_config(
    page_title="Market Watch",
    page_icon="🗣️🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #102a43; --muted: #627d98; --line: #d9e2ec; --accent: #147d92; }
    .block-container { max-width: 1440px; padding-top: 2.5rem; }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .eyebrow { color: var(--accent); font-size: .78rem; font-weight: 700;
               letter-spacing: .12em; text-transform: uppercase; }
    .subtitle { color: var(--muted); margin-top: -.7rem; }
    .chart-card { border: 1px solid var(--line); border-radius: 8px; padding: .75rem .9rem .25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=45, show_spinner=False)
def load_snapshot():
    return fetch_snapshot(WATCHLIST)


@st.cache_data(ttl=45, show_spinner=False)
def load_chart(symbol):
    bars = yf.download(
        symbol,
        period="1d",
        interval="5m",
        progress=False,
        auto_adjust=False,
        group_by="column",
    )
    if bars.empty:
        return pd.DataFrame()
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    bars = bars.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    bars["SMA 5"] = bars["Close"].rolling(5).mean()
    return bars


def format_volume(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value / 1_000:.0f}K"


def make_chart(bars, symbol, chart_type, dark_mode):
    background = "#101820" if dark_mode else "#ffffff"
    foreground = "#f4f7f9" if dark_mode else "#102a43"
    grid = "#263640" if dark_mode else "#edf2f7"
    chart = go.Figure()
    if chart_type == "Candlestick":
        chart.add_trace(
            go.Candlestick(
                x=bars.index,
                open=bars["Open"],
                high=bars["High"],
                low=bars["Low"],
                close=bars["Close"],
                name=symbol,
                increasing_line_color="#39c28f",
                increasing_fillcolor="#39c28f",
                decreasing_line_color="#e66b6b",
                decreasing_fillcolor="#e66b6b",
                hovertext=symbol,
                hoverinfo="x+y+name",
            )
        )
    else:
        chart.add_trace(
            go.Scatter(
                x=bars.index,
                y=bars["Close"],
                name="Price",
                mode="lines",
                line={"color": "#39c28f" if dark_mode else "#147d92", "width": 3},
                hovertemplate="$%{y:.2f}<extra></extra>",
            )
        )
    chart.add_trace(
        go.Scatter(
            x=bars.index,
            y=bars["SMA 5"],
            name="5-bar average",
            mode="lines",
            line={"color": "#f2b134", "width": 1.5, "dash": "dot"},
            hovertemplate="$%{y:.2f}<extra></extra>",
        )
    )
    chart.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        plot_bgcolor=background,
        paper_bgcolor=background,
        font={"color": foreground},
        xaxis={"showgrid": False, "rangeslider": {"visible": False}},
        yaxis={"showgrid": True, "gridcolor": grid, "tickprefix": "$"},
    )
    return chart


st.markdown('<div class="eyebrow">Live market monitor</div>', unsafe_allow_html=True)
st.title("Market Watch")
st.markdown(
    "<p class='subtitle'>A quiet view of your watchlist, refreshed from Yahoo Finance as the market moves.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    with st.expander("Watchlist", expanded=True):
        selected_symbols = st.multiselect(
            "Stocks to chart",
            WATCHLIST,
            default=WATCHLIST[:6],
            help="Choose the stocks shown in the chart grid.",
        )

    with st.expander("Chart display", expanded=True):
        charts_per_page = st.slider("Charts per page", 2, 8, 4)
        chart_type = st.radio("Chart type", ["Line", "Candlestick"], horizontal=True)
        dark_mode = st.toggle("Dark chart mode", value=True)

    page_count = max(1, (len(selected_symbols) + charts_per_page - 1) // charts_per_page)

    with st.expander("Refresh", expanded=False):
        page_number = st.number_input(
            "Chart page",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            disabled=page_count == 1,
        )
        refresh_seconds = st.slider("Refresh interval", 30, 300, POLL_SECONDS, step=15)
        st.caption(f"Yahoo Finance may lag during busy periods. Updates run every {refresh_seconds}s.")


@st.fragment(run_every=f"{refresh_seconds}s")
def live_dashboard():
    timestamp = datetime.now().strftime("%H:%M:%S")
    with st.spinner("Updating market data..."):
        try:
            rows = load_snapshot()
        except Exception as error:
            st.error(f"Could not fetch market data: {error}")
            return

    matches = apply_filters(rows, FILTERS)
    match_symbols = {row["symbol"] for row in matches}
    lead_symbol = selected_symbols[0] if selected_symbols else None
    selected = next((row for row in rows if row["symbol"] == lead_symbol), None)

    overview_tab, charts_tab = st.tabs(["Overview", "Charts"])

    with overview_tab:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Symbols tracked", len(rows))
        metric_columns[1].metric("Scanner matches", len(matches))
        metric_columns[2].metric("Lead price", f"${selected['price']:.2f}" if selected else "--")
        metric_columns[3].metric(
            "Today's move",
            f"{selected['pct_change']:+.2f}%" if selected else "--",
            delta_color="normal",
        )

        with st.expander("Watchlist details", expanded=True):
            table = pd.DataFrame(rows)
            if not table.empty:
                table["status"] = table["symbol"].map(lambda symbol: "MATCH" if symbol in match_symbols else "watch")
                table["price"] = table["price"].map(lambda value: f"${value:.2f}")
                table["pct_change"] = table["pct_change"].map(lambda value: f"{value:+.2f}%")
                table["volume"] = table["volume"].map(format_volume)
                table["vol_ratio"] = table["vol_ratio"].map(lambda value: f"{value:.2f}x")
                table = table.rename(columns={
                    "symbol": "Symbol", "price": "Price", "pct_change": "Today",
                    "volume": "Volume", "vol_ratio": "Vol / avg", "status": "Status",
                })
                st.dataframe(
                    table[["Symbol", "Price", "Today", "Volume", "Vol / avg", "Status"]],
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )

    with charts_tab:
        st.subheader(f"Price tracking · page {page_number} of {page_count}")
        page_start = (page_number - 1) * charts_per_page
        page_symbols = selected_symbols[page_start:page_start + charts_per_page]
        if not page_symbols:
            st.info("Choose one or more stocks in the Watchlist sidebar section to populate the chart grid.")
        else:
            for row_start in range(0, len(page_symbols), 2):
                chart_columns = st.columns(2, gap="large")
                for column, symbol in zip(chart_columns, page_symbols[row_start:row_start + 2]):
                    with column:
                        with st.container(border=True):
                            st.markdown(f"### {symbol}")
                            bars = load_chart(symbol)
                            if bars.empty:
                                st.info(f"No intraday bars are available for {symbol} right now.")
                            else:
                                st.plotly_chart(
                                    make_chart(bars, symbol, chart_type, dark_mode),
                                    use_container_width=True,
                                    config={"displaylogo": False},
                                )
                                latest_bar = bars.iloc[-1]
                                st.caption(
                                    f"Last bar: {bars.index[-1].strftime('%H:%M')}  ·  "
                                    f"Close ${latest_bar['Close']:.2f}  ·  Updated {timestamp}"
                                )


live_dashboard()