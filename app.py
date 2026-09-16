"""Live Streamlit dashboard for the stock scanner."""

from datetime import datetime
import json
import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
except ImportError:
    TradingClient = None
    OrderSide = None
    TimeInForce = None
    LimitOrderRequest = None
    MarketOrderRequest = None

from stock_scanner import FILTERS, POLL_SECONDS, WATCHLIST, apply_filters, fetch_snapshot


LARGE_CAP_SYMBOLS = [
    "AVGO", "ORCL", "COST", "WMT", "V", "MA", "LLY", "JNJ",
    "PG", "KO", "PEP", "DIS", "ADBE", "QCOM", "INTC", "IBM",
    "GE", "CAT", "BA", "GS", "C", "T", "VZ", "PFE",
]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]
CHART_SYMBOLS = list(dict.fromkeys(WATCHLIST + LARGE_CAP_SYMBOLS + CRYPTO_SYMBOLS))
TRADING_SYMBOLS = list(dict.fromkeys(WATCHLIST + LARGE_CAP_SYMBOLS))
DEFAULT_PRESETS = {
    "Balanced": FILTERS.copy(),
    "Momentum": {
        **FILTERS,
        "min_abs_change_pct": 2.0,
        "min_vol_ratio": 1.0,
        "macd_bullish_only": True,
    },
    "Unusual volume": {
        **FILTERS,
        "min_vol_ratio": 1.0,
        "unusual_volume_only": True,
    },
}


st.set_page_config(
    page_title="Stock Scanner",
    page_icon="🗣️🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink: #102a43; --muted: #627d98; --line: #d9e2ec; --accent: #147d92; }
    .block-container { max-width: 1440px; padding-top: 1rem; padding-bottom: 1rem; }
    h1 { font-size: 2rem !important; margin-bottom: .2rem !important; }
    h2 { font-size: 1.35rem !important; margin-top: .65rem !important; margin-bottom: .35rem !important; }
    h3 { font-size: 1.05rem !important; margin-top: .35rem !important; margin-bottom: .15rem !important; }
    [data-testid="stVerticalBlock"] { gap: .45rem; }
    [data-testid="stMetric"] { padding: .35rem .5rem; }
    [data-testid="stExpander"] { margin-bottom: .35rem; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .3rem; }
    [data-testid="stMetricValue"] { color: #e8f1f5; }
    [data-testid="stMetricLabel"] { color: #b8cbd5; }
    [data-testid="stMetricDelta"] { color: #d7e6ee; }
    .eyebrow { color: var(--accent); font-size: .78rem; font-weight: 700;
               letter-spacing: .12em; text-transform: uppercase; }
    .subtitle { color: #d7e6ee; margin-top: -.7rem; }
    .chart-card { border: 1px solid var(--line); border-radius: 8px; padding: .4rem .6rem .15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=45, show_spinner=False)
def load_snapshot():
    return fetch_snapshot(WATCHLIST)


@st.cache_data(ttl=45, show_spinner=False)
def load_chart(symbol, period, interval):
    bars = yf.download(
        symbol,
        period=period,
        interval=interval,
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


@st.cache_data(ttl=30, show_spinner=False)
def load_market_depth(symbol):
    """Return Yahoo's latest top-of-book quote for one symbol."""
    info = yf.Ticker(symbol).info
    bid = info.get("bid")
    ask = info.get("ask")
    if bid is None or ask is None:
        return None
    return {
        "symbol": symbol,
        "bid": float(bid),
        "bid_size": info.get("bidSize"),
        "ask": float(ask),
        "ask_size": info.get("askSize"),
        "last": info.get("regularMarketPrice"),
        "volume": info.get("regularMarketVolume"),
    }


@st.cache_data(ttl=120, show_spinner=False)
def load_trending_tickers():
    """Return Yahoo's current most-active equity screen."""
    result = yf.screen("most_actives", count=8)
    quotes = result.get("quotes", []) if result else []
    return [
        {
            "Ticker": quote.get("symbol", "--"),
            "Price": quote.get("regularMarketPrice"),
            "Move": quote.get("regularMarketChangePercent"),
            "Volume": quote.get("regularMarketVolume"),
        }
        for quote in quotes
        if quote.get("symbol") and quote.get("regularMarketPrice") is not None
    ]


@st.cache_data(ttl=300, show_spinner=False)
def load_company_context(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        "sector": info.get("sector", "--"),
        "industry": info.get("industry", "--"),
        "market_cap": info.get("marketCap"),
        "pe": info.get("trailingPE"),
        "dividend_yield": info.get("dividendYield"),
        "recommendation": info.get("recommendationKey", "--"),
        "earnings_date": info.get("earningsTimestampStart"),
        "news": (ticker.news or [])[:5],
    }


def load_session_credentials():
    credentials_path = st.session_state.get("alpaca_credentials_path")
    if not credentials_path:
        return None, None
    try:
        with open(credentials_path, encoding="utf-8") as credentials_file:
            credentials = json.load(credentials_file)
        return credentials.get("api_key"), credentials.get("secret_key")
    except (OSError, json.JSONDecodeError):
        return None, None


def save_session_credentials(api_key, secret_key):
    delete_session_credentials()
    credentials_file = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="market_watch_alpaca_",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    )
    try:
        json.dump({"api_key": api_key, "secret_key": secret_key}, credentials_file)
        credentials_file.flush()
        os.chmod(credentials_file.name, 0o600)
        st.session_state.alpaca_credentials_path = credentials_file.name
    finally:
        credentials_file.close()


def delete_session_credentials():
    credentials_path = st.session_state.pop("alpaca_credentials_path", None)
    if credentials_path:
        try:
            os.remove(credentials_path)
        except FileNotFoundError:
            pass


def get_paper_client():
    if TradingClient is None:
        return None, "Install the alpaca-py package to enable paper trading."
    api_key, secret_key = load_session_credentials()
    api_key = api_key or os.getenv("ALPACA_API_KEY")
    secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        try:
            api_key = api_key or st.secrets.get("ALPACA_API_KEY")
            secret_key = secret_key or st.secrets.get("ALPACA_SECRET_KEY")
        except Exception:
            pass
    if not api_key or not secret_key:
        return None, "Alpaca paper credentials were not detected. Set both API key and secret, then restart Streamlit."
    try:
        return TradingClient(api_key, secret_key, paper=True), None
    except Exception as error:
        return None, f"Could not connect to Alpaca paper trading: {error}"


def format_money(value):
    return f"${float(value):,.2f}"


def format_volume(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value / 1_000:.0f}K"


def make_chart(bars, symbol, chart_type, dark_mode):
    background = "#101820" if dark_mode else "#ffffff"
    foreground = "#f4f7f9" if dark_mode else "#085fb1"
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
        height=270,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
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
st.title("Stock Scanner")
st.markdown(
    "<p class='subtitle'>watchist of stocks from yahoo finance (cuz were poor)</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    with st.expander("Watchlist", expanded=True):
        selected_symbols = st.multiselect(
            "Stocks to chart",
            CHART_SYMBOLS,
            default=WATCHLIST[:6],
            help="Choose the stocks shown in the chart grid.",
        )
        st.caption("Includes large-cap equities and crypto pairs.")

    with st.expander("Scanner filters", expanded=False):
        if "saved_presets" not in st.session_state:
            st.session_state.saved_presets = DEFAULT_PRESETS.copy()
        preset_names = ["Custom"] + list(st.session_state.saved_presets)
        selected_preset = st.selectbox("Preset", preset_names)
        active_filters = (
            FILTERS.copy()
            if selected_preset == "Custom"
            else st.session_state.saved_presets[selected_preset].copy()
        )
        price_min = st.number_input(
            "Minimum price",
            min_value=0.0,
            value=float(active_filters["price_min"]),
            step=1.0,
        )
        price_max = st.number_input(
            "Maximum price",
            min_value=0.0,
            value=float(active_filters["price_max"]),
            step=100.0,
        )
        min_abs_change_pct = st.number_input(
            "Minimum daily move (%)",
            min_value=0.0,
            value=float(active_filters["min_abs_change_pct"]),
            step=0.1,
        )
        min_vol_ratio = st.number_input(
            "Minimum volume ratio",
            min_value=0.0,
            value=float(active_filters["min_vol_ratio"]),
            step=0.1,
        )
        gap_min_pct = st.number_input(
            "Minimum gap (%)",
            min_value=-100.0,
            max_value=100.0,
            value=float(active_filters.get("gap_min_pct", -100.0)),
            step=0.5,
        )
        gap_max_pct = st.number_input(
            "Maximum gap (%)",
            min_value=-100.0,
            max_value=100.0,
            value=float(active_filters.get("gap_max_pct", 100.0)),
            step=0.5,
        )
        breakout_only = st.checkbox(
            "20-day breakouts only",
            value=active_filters.get("breakout_only", False),
        )
        unusual_volume_only = st.checkbox(
            "Unusual volume only (2x+)",
            value=active_filters.get("unusual_volume_only", False),
        )
        rsi_min = st.number_input(
            "Minimum RSI",
            min_value=0.0,
            max_value=100.0,
            value=float(active_filters.get("rsi_min", 0.0)),
            step=1.0,
        )
        rsi_max = st.number_input(
            "Maximum RSI",
            min_value=0.0,
            max_value=100.0,
            value=float(active_filters.get("rsi_max", 100.0)),
            step=1.0,
        )
        macd_bullish_only = st.checkbox(
            "Bullish MACD only",
            value=active_filters.get("macd_bullish_only", False),
        )
        bullish_cross_only = st.checkbox(
            "Bullish crossovers only",
            value=active_filters["bullish_cross_only"],
        )
        scanner_filters = {
            "price_min": price_min,
            "price_max": price_max,
            "min_abs_change_pct": min_abs_change_pct,
            "min_vol_ratio": min_vol_ratio,
            "gap_min_pct": gap_min_pct,
            "gap_max_pct": gap_max_pct,
            "breakout_only": breakout_only,
            "unusual_volume_only": unusual_volume_only,
            "rsi_min": rsi_min,
            "rsi_max": rsi_max,
            "macd_bullish_only": macd_bullish_only,
            "bullish_cross_only": bullish_cross_only,
        }
        preset_to_save = st.text_input("Save current filters as", placeholder="Preset name")
        if st.button("Save preset", use_container_width=True) and preset_to_save.strip():
            st.session_state.saved_presets[preset_to_save.strip()] = scanner_filters.copy()
            st.success(f"Saved {preset_to_save.strip()}")

    with st.expander("Chart display", expanded=True):
        charts_per_page = st.slider("Charts per page", 2, 8, 4)
        chart_type = st.radio("Chart type", ["Line", "Candlestick"], horizontal=True)
        dark_mode = st.toggle("Dark chart mode", value=True)
        chart_period = st.selectbox("Chart range", ["1d", "5d", "1mo"], index=0)
        chart_interval = st.selectbox("Chart timeframe", ["5m", "15m", "1h", "1d"], index=0)

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

    matches = apply_filters(rows, scanner_filters)
    match_symbols = {row["symbol"] for row in matches}
    previous_match_symbols = st.session_state.get("previous_match_symbols", set())
    new_match_symbols = match_symbols - previous_match_symbols
    st.session_state.previous_match_symbols = match_symbols
    lead_symbol = selected_symbols[0] if selected_symbols else None
    selected = next((row for row in rows if row["symbol"] == lead_symbol), None)

    def render_chart_grid(symbols, key_prefix):
        if not symbols:
            st.info("No current scanner matches. Matching charts will appear here automatically.")
            return

        for row_start in range(0, len(symbols), 2):
            chart_columns = st.columns(2, gap="large")
            for column_index, (column, symbol) in enumerate(
                zip(chart_columns, symbols[row_start:row_start + 2])
            ):
                with column:
                    with st.container(border=True):
                        st.markdown(f"### {symbol}")
                        bars = load_chart(symbol, chart_period, chart_interval)
                        if bars.empty:
                            st.info(f"No intraday bars are available for {symbol} right now.")
                        else:
                            st.plotly_chart(
                                make_chart(bars, symbol, chart_type, dark_mode),
                                use_container_width=True,
                                config={"displaylogo": False},
                                key=f"{key_prefix}-{row_start + column_index}-{symbol}-{chart_period}-{chart_interval}",
                            )
                            latest_bar = bars.iloc[-1]
                            st.caption(
                                f"Last bar: {bars.index[-1].strftime('%H:%M')}  ·  "
                                f"Close ${latest_bar['Close']:.2f}  ·  Updated {timestamp}"
                            )

    main_column, right_column = st.columns([3.4, 1.2], gap="large")

    with right_column:
        st.subheader("Market info")
        with st.expander("Market depth", expanded=True):
            st.caption("Top-of-book data. Yahoo Finance does not provide the full Level 2 order book.")
            depth_symbol = st.selectbox("Depth symbol", CHART_SYMBOLS, label_visibility="collapsed")
            try:
                depth = load_market_depth(depth_symbol)
            except Exception as error:
                st.warning(f"Quote depth unavailable: {error}")
                depth = None

            if depth is None:
                st.info(f"No bid/ask quote is available for {depth_symbol} right now.")
            else:
                midpoint = (depth["bid"] + depth["ask"]) / 2
                spread = depth["ask"] - depth["bid"]
                quote_columns = st.columns(2)
                quote_columns[0].metric("Bid", f"${depth['bid']:.2f}")
                quote_columns[1].metric("Ask", f"${depth['ask']:.2f}")
                st.dataframe(
                    pd.DataFrame([
                        {"Quote": "Bid size", "Value": depth["bid_size"] or "--"},
                        {"Quote": "Ask size", "Value": depth["ask_size"] or "--"},
                        {"Quote": "Spread", "Value": f"${spread:.4f}"},
                        {"Quote": "Midpoint", "Value": f"${midpoint:.2f}"},
                    ]),
                    use_container_width=True,
                    hide_index=True,
                    height=170,
                )

        with st.expander("Company context", expanded=False):
            try:
                context = load_company_context(depth_symbol)
            except Exception as error:
                st.warning(f"Company information unavailable: {error}")
                context = None

            if context:
                context_rows = [
                    {"Field": "Sector", "Value": context["sector"]},
                    {"Field": "Industry", "Value": context["industry"]},
                    {"Field": "Market cap", "Value": format_volume(context["market_cap"]) if context["market_cap"] else "--"},
                    {"Field": "Trailing P/E", "Value": f"{context['pe']:.2f}" if context["pe"] else "--"},
                    {"Field": "Dividend yield", "Value": f"{context['dividend_yield']:.2%}" if context["dividend_yield"] else "--"},
                    {"Field": "Analyst view", "Value": context["recommendation"]},
                    {
                        "Field": "Earnings date",
                        "Value": datetime.fromtimestamp(context["earnings_date"]).strftime("%Y-%m-%d")
                        if context["earnings_date"] else "--",
                    },
                ]
                st.dataframe(
                    pd.DataFrame(context_rows),
                    use_container_width=True,
                    hide_index=True,
                    height=225,
                )
                with st.expander("Latest news", expanded=False):
                    if context["news"]:
                        for article in context["news"]:
                            content = article.get("content", article)
                            title = content.get("title", "Untitled")
                            canonical_url = content.get("canonicalUrl", {})
                            click_url = content.get("clickThroughUrl", {})
                            link = canonical_url.get("url") or click_url.get("url")
                            st.markdown(f"- [{title}]({link})" if link else f"- {title}")
                    else:
                        st.caption("No recent news available.")

        with st.expander("Trending tickers", expanded=True):
            st.caption("Yahoo's most-active equity screen, refreshed periodically.")
            try:
                trending = load_trending_tickers()
            except Exception as error:
                st.warning(f"Trending data unavailable: {error}")
                trending = []

            if trending:
                trending_table = pd.DataFrame(trending)
                trending_table["Price"] = trending_table["Price"].map(lambda value: f"${value:.2f}")
                trending_table["Move"] = trending_table["Move"].map(lambda value: f"{value:+.2f}%")
                trending_table["Volume"] = trending_table["Volume"].map(format_volume)
                st.dataframe(
                    trending_table,
                    use_container_width=True,
                    hide_index=True,
                    height=min(320, 38 + len(trending_table) * 35),
                )
            else:
                st.info("No trending ticker data is available right now.")

    with main_column:
        overview_tab, charts_tab, matches_tab, paper_tab = st.tabs(
            ["Overview", "Charts", "Matches", "Paper Trading"]
        )

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
            if new_match_symbols:
                st.success(f"New scanner matches: {', '.join(sorted(new_match_symbols))}")

            with st.expander("Watchlist details", expanded=True):
                table = pd.DataFrame(rows)
                if not table.empty:
                    table["status"] = table["symbol"].map(lambda symbol: "MATCH" if symbol in match_symbols else "watch")
                    table["price"] = table["price"].map(lambda value: f"${value:.2f}")
                    table["pct_change"] = table["pct_change"].map(lambda value: f"{value:+.2f}%")
                    table["gap_pct"] = table["gap_pct"].map(lambda value: f"{value:+.2f}%")
                    table["volume"] = table["volume"].map(format_volume)
                    table["vol_ratio"] = table["vol_ratio"].map(lambda value: f"{value:.2f}x")
                    table["rsi"] = table["rsi"].map(lambda value: f"{value:.1f}")
                    table["macd"] = table["macd"].map(lambda value: f"{value:.2f}")
                    table = table.rename(columns={
                        "symbol": "Symbol", "price": "Price", "pct_change": "Today",
                        "gap_pct": "Gap", "volume": "Volume", "vol_ratio": "Vol / avg",
                        "rsi": "RSI", "macd": "MACD", "status": "Status",
                    })
                    st.dataframe(
                        table[["Symbol", "Price", "Today", "Gap", "Volume", "Vol / avg", "RSI", "MACD", "Status"]],
                        use_container_width=True,
                        hide_index=True,
                        height=230,
                    )
                    st.download_button(
                        "Download snapshot CSV",
                        pd.DataFrame(rows).to_csv(index=False),
                        file_name="market_snapshot.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        with charts_tab:
            st.subheader(f"Price tracking · page {page_number} of {page_count}")
            page_start = (page_number - 1) * charts_per_page
            page_symbols = selected_symbols[page_start:page_start + charts_per_page]
            render_chart_grid(page_symbols, f"charts-page-{page_number}")

        with matches_tab:
            st.subheader(f"Current matches · {len(matches)}")
            st.caption("This tab updates automatically and only shows stocks that currently satisfy the scanner filters.")
            render_chart_grid([row["symbol"] for row in matches], "matches")

        with paper_tab:
            st.subheader("Alpaca paper trading")
            st.caption("Paper environment only. Orders are simulated and never sent to a live brokerage account.")

            has_session_credentials = bool(st.session_state.get("alpaca_credentials_path"))
            with st.expander("Connect paper account", expanded=not has_session_credentials):
                st.caption("Keys are kept in a temporary owner-only file for this session and are not saved to the project.")
                with st.form("alpaca_credentials_form", clear_on_submit=False):
                    entered_api_key = st.text_input("Alpaca API key", type="password")
                    entered_secret_key = st.text_input("Alpaca secret key", type="password")
                    connect_account = st.form_submit_button("Connect paper account", type="primary")

                if connect_account:
                    if not entered_api_key.strip() or not entered_secret_key.strip():
                        st.warning("Enter both the Alpaca API key and secret key.")
                    else:
                        save_session_credentials(entered_api_key.strip(), entered_secret_key.strip())
                        st.session_state.alpaca_connected = False
                        st.rerun(scope="fragment")

                if has_session_credentials:
                    if st.button("Disconnect paper account"):
                        delete_session_credentials()
                        st.session_state.alpaca_connected = False
                        st.rerun(scope="fragment")

            paper_client, connection_error = get_paper_client()
            if connection_error:
                st.info(connection_error)
                st.code("export ALPACA_API_KEY=your_paper_key\nexport ALPACA_SECRET_KEY=your_paper_secret")
            else:
                try:
                    account = paper_client.get_account()
                    st.session_state.alpaca_connected = True
                    account_columns = st.columns(3)
                    account_columns[0].metric("Buying power", format_money(account.buying_power))
                    account_columns[1].metric("Portfolio value", format_money(account.portfolio_value))
                    account_columns[2].metric("Account status", str(account.status))

                    positions = paper_client.get_all_positions()
                    with st.expander(f"Positions · {len(positions)}", expanded=True):
                        if positions:
                            position_rows = [
                                {
                                    "Symbol": position.symbol,
                                    "Qty": position.qty,
                                    "Value": format_money(position.market_value),
                                    "P/L": format_money(position.unrealized_pl),
                                }
                                for position in positions
                            ]
                            st.dataframe(pd.DataFrame(position_rows), use_container_width=True, hide_index=True)
                        else:
                            st.caption("No paper positions yet.")

                    with st.form("paper_order_form", clear_on_submit=False):
                        st.markdown("#### Submit paper order")
                        order_symbol = st.selectbox("Symbol", TRADING_SYMBOLS)
                        order_side = st.radio("Side", ["Buy", "Sell"], horizontal=True)
                        order_type = st.radio("Order type", ["Market", "Limit"], horizontal=True)
                        order_quantity = st.number_input("Quantity", min_value=0.0001, value=1.0, step=1.0)
                        limit_price = st.number_input("Limit price", min_value=0.01, value=100.0, step=0.01, disabled=order_type == "Market")
                        confirm_order = st.checkbox("I understand this submits an order to my Alpaca paper account.")
                        submit_order = st.form_submit_button("Submit paper order", type="primary")

                    if submit_order:
                        if not confirm_order:
                            st.warning("Confirm the paper-order checkbox before submitting.")
                        else:
                            side = OrderSide.BUY if order_side == "Buy" else OrderSide.SELL
                            if order_type == "Market":
                                order_request = MarketOrderRequest(
                                    symbol=order_symbol,
                                    qty=order_quantity,
                                    side=side,
                                    time_in_force=TimeInForce.DAY,
                                )
                            else:
                                order_request = LimitOrderRequest(
                                    symbol=order_symbol,
                                    qty=order_quantity,
                                    side=side,
                                    time_in_force=TimeInForce.DAY,
                                    limit_price=limit_price,
                                )
                            try:
                                order = paper_client.submit_order(order_data=order_request)
                                st.success(f"Paper order submitted: {order.id} ({order.status})")
                            except Exception as error:
                                st.error(f"Paper order rejected: {error}")
                except Exception as error:
                    st.error(f"Could not load Alpaca paper account: {error}")


live_dashboard()