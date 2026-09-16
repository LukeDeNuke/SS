"""Live Streamlit dashboard for the stock scanner.

Layout redesigned as a compact trading-terminal: a scrolling ticker tape up top,
a chart toolbar (symbol / timeframe / type), a three-pane workspace
(watchlist | chart | order entry + account), and a tabbed bottom panel for
everything else (grid of charts, scanner matches, orders, paper account
details, company/news, trending tickers) -- similar to the panel layout used
by TradingView / Warrior Trading's software.
"""

from datetime import datetime
import json
import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    TradingClient = None
    OrderSide = None
    TimeInForce = None
    LimitOrderRequest = None
    MarketOrderRequest = None
    GetOrdersRequest = None
    QueryOrderStatus = None
    StockHistoricalDataClient = None
    DataFeed = None
    StockBarsRequest = None
    StockLatestQuoteRequest = None
    StockLatestTradeRequest = None
    TimeFrame = None
    TimeFrameUnit = None

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

# --------------------------------------------------------------------------
# Terminal theme -- dark, dense, monospace numerics, colour-coded up/down,
# scrolling ticker tape. This is intentionally applied regardless of the
# system theme so the app always reads like trading software.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0a0e14;
        --panel: #10151d;
        --panel-alt: #141b25;
        --border: #232b36;
        --text: #e6edf3;
        --text-dim: #7c8896;
        --up: #26a69a;
        --down: #ef5350;
        --accent: #2f81f7;
        --amber: #f2b134;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--bg) !important;
    }
    [data-testid="stHeader"] { background-color: transparent !important; }
    [data-testid="stSidebar"] {
        background-color: var(--panel) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .3rem; }
    .block-container { max-width: 1700px; padding-top: .5rem; padding-bottom: 1rem; }

    h1, h2, h3, h4, h5, label, p, span, .stMarkdown, .stCaption {
        font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    }
    h1 { font-size: 1.3rem !important; margin: 0 !important; color: var(--text); }
    h2 { font-size: 1.0rem !important; margin: .2rem 0 !important; color: var(--text); }
    h3 { font-size: .85rem !important; margin: 0 0 .3rem 0 !important;
         text-transform: uppercase; letter-spacing: .04em; color: var(--text-dim); }
    [data-testid="stVerticalBlock"] { gap: .4rem; }
    [data-testid="stExpander"] { margin-bottom: .3rem; background-color: var(--panel-alt);
        border: 1px solid var(--border); border-radius: 6px; }

    .eyebrow { color: var(--accent); font-size: .72rem; font-weight: 700;
               letter-spacing: .12em; text-transform: uppercase; }

    /* ---- ticker tape ---- */
    .ticker-tape-wrap { border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
        background: var(--panel); overflow-x: auto; white-space: nowrap; padding: .35rem .5rem;
        margin: .35rem 0 .6rem 0; scrollbar-width: thin; }
    .ticker-tape-wrap::-webkit-scrollbar { height: 4px; }
    .ticker-item { display: inline-flex; align-items: baseline; gap: .35rem; margin-right: 1.3rem;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: .78rem; }
    .ticker-sym { color: var(--text); font-weight: 700; }
    .ticker-px { color: var(--text-dim); }
    .up { color: var(--up); font-weight: 600; }
    .down { color: var(--down); font-weight: 600; }

    /* ---- panel framing for the three-pane workspace ---- */
    .panel-card { border: 1px solid var(--border); background: var(--panel);
        border-radius: 8px; padding: .55rem .65rem .35rem; height: 100%; }
    .panel-card-tight { border: 1px solid var(--border); background: var(--panel);
        border-radius: 8px; padding: .4rem .55rem .25rem; margin-bottom: .4rem; }

    /* numeric fonts everywhere data shows up */
    [data-testid="stMetricValue"] { color: var(--text) !important;
        font-family: "SFMono-Regular", Consolas, monospace; font-size: 1.15rem !important; }
    [data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: .68rem !important;
        text-transform: uppercase; letter-spacing: .05em; }
    [data-testid="stMetricDelta"] { font-family: "SFMono-Regular", Consolas, monospace; }
    [data-testid="stCaptionContainer"] { color: var(--text-dim) !important; font-size: .7rem !important; }
    [data-testid="stDataFrame"] { font-size: .74rem; }

    /* buy / sell buttons */
    .buy-btn button { background-color: var(--up) !important; color: #05130f !important;
        border: none !important; font-weight: 700 !important; }
    .sell-btn button { background-color: var(--down) !important; color: #170505 !important;
        border: none !important; font-weight: 700 !important; }

    .status-dot { display: inline-block; width: .5rem; height: .5rem; border-radius: 50%;
        margin-right: .35rem; }
    .status-on { background-color: var(--up); }
    .status-off { background-color: var(--text-dim); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Data loaders (unchanged from the original scanner -- caching + network I/O)
# ==========================================================================

@st.cache_data(ttl=45, show_spinner=False)
def load_snapshot():
    return fetch_snapshot(WATCHLIST)


@st.cache_data(ttl=15, show_spinner=False)
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


def load_alpaca_chart(symbol, period, interval):
    api_key, secret_key = load_session_credentials()
    api_key = api_key or os.getenv("ALPACA_API_KEY")
    secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key or StockHistoricalDataClient is None:
        return pd.DataFrame(), "Alpaca credentials or the Alpaca data SDK are unavailable."

    timeframe_map = {
        "1m": TimeFrame.Minute,
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "1h": TimeFrame.Hour,
        "1d": TimeFrame.Day,
    }
    period_days = {"1d": 1, "5d": 5, "1mo": 30}[period]
    end = datetime.now().astimezone()
    start = end - pd.Timedelta(days=period_days)
    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            timeframe=timeframe_map[interval],
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(request).df
    except Exception as error:
        return pd.DataFrame(), str(error)
    if bars.empty:
        return pd.DataFrame(), "Alpaca returned no bars for this symbol and timeframe."
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.droplevel("symbol")
    bars = bars.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume",
    })
    bars = bars.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    bars["SMA 5"] = bars["Close"].rolling(5).mean()
    return bars, None


@st.cache_data(ttl=5, show_spinner=False)
def load_yahoo_market_depth(symbol):
    """Return Yahoo's latest top-of-book quote for one symbol."""
    info = yf.Ticker(symbol).info
    bid = info.get("bid")
    ask = info.get("ask")
    if bid is None or ask is None:
        return None
    return {
        "source": "Yahoo Finance",
        "symbol": symbol,
        "bid": float(bid),
        "bid_size": info.get("bidSize"),
        "ask": float(ask),
        "ask_size": info.get("askSize"),
        "last": info.get("regularMarketPrice"),
        "volume": info.get("regularMarketVolume"),
    }


@st.cache_data(ttl=5, show_spinner=False)
def load_alpaca_market_depth(symbol):
    api_key, secret_key = load_session_credentials()
    api_key = api_key or os.getenv("ALPACA_API_KEY")
    secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key or StockHistoricalDataClient is None:
        return None, "Alpaca credentials or the Alpaca data SDK are unavailable."
    client = StockHistoricalDataClient(api_key, secret_key)
    errors = []
    candidates = []
    for feed in (DataFeed.SIP, DataFeed.IEX):
        try:
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=feed)
            trade_request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=feed)
            quote = client.get_stock_latest_quote(quote_request).get(symbol)
            trade = client.get_stock_latest_trade(trade_request).get(symbol)
            if quote is not None:
                candidates.append((feed, quote, trade))
                if quote.ask_size not in (None, 0):
                    break
        except Exception as error:
            errors.append(f"{feed.value}: {error}")

    if not candidates:
        return None, "Alpaca returned no latest quote. " + "; ".join(errors)

    feed, quote, trade = next(
        (candidate for candidate in reversed(candidates) if candidate[1].ask_size not in (None, 0)),
        candidates[0],
    )
    bid = float(quote.bid_price) if quote.bid_price is not None else None
    ask = float(quote.ask_price) if quote.ask_price is not None else None
    return {
        "source": f"Alpaca {feed.value.upper()}",
        "symbol": symbol,
        "bid": bid,
        "bid_size": quote.bid_size,
        "ask": ask,
        "ask_size": quote.ask_size,
        "last": float(trade.price) if trade else None,
        "last_size": trade.size if trade else None,
        "quote_time": quote.timestamp,
        "trade_time": trade.timestamp if trade else None,
    }, None


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


def session_credentials_available():
    api_key, secret_key = load_session_credentials()
    return bool(api_key and secret_key)


def alpaca_credentials_available():
    if StockHistoricalDataClient is None:
        return False
    return session_credentials_available() or bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))


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


def record_portfolio_value(value):
    history = st.session_state.setdefault("portfolio_value_history", [])
    history.append({"Time": datetime.now(), "Portfolio value": float(value)})
    st.session_state.portfolio_value_history = history[-500:]


def make_portfolio_chart(history):
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=[point["Time"] for point in history],
            y=[point["Portfolio value"] for point in history],
            mode="lines+markers",
            name="Portfolio value",
            line={"color": "#39c28f", "width": 2},
            marker={"size": 5},
            hovertemplate="$%{y:,.2f}<extra></extra>",
        )
    )
    chart.update_layout(
        height=230,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        plot_bgcolor="#0a0e14",
        paper_bgcolor="#0a0e14",
        font={"color": "#d7e6ee"},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "#232b36", "tickprefix": "$"},
        showlegend=False,
    )
    return chart


def make_orders_chart(orders):
    order_rows = []
    for order in orders:
        event_time = order.filled_at or order.submitted_at or order.created_at
        if event_time is None:
            continue
        quantity = float(order.filled_qty or order.qty or 0)
        price = float(order.filled_avg_price or order.limit_price or 0)
        order_rows.append({
            "time": event_time,
            "notional": quantity * price,
            "side": str(order.side).split(".")[-1].lower(),
            "symbol": order.symbol,
            "status": str(order.status).split(".")[-1].lower(),
        })

    chart = go.Figure()
    for side, color in (("buy", "#26a69a"), ("sell", "#ef5350")):
        side_rows = [row for row in order_rows if row["side"] == side]
        chart.add_trace(
            go.Scatter(
                x=[row["time"] for row in side_rows],
                y=[row["notional"] for row in side_rows],
                text=[f"{row['symbol']} · {row['status']}" for row in side_rows],
                name=side.title(),
                mode="markers",
                marker={"color": color, "size": 10},
                hovertemplate="%{text}<br>$%{y:,.2f}<extra></extra>",
            )
        )
    chart.update_layout(
        height=280,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        plot_bgcolor="#0a0e14",
        paper_bgcolor="#0a0e14",
        font={"color": "#d7e6ee"},
        xaxis={"showgrid": False, "title": "Order time"},
        yaxis={"showgrid": True, "gridcolor": "#232b36", "tickprefix": "$", "title": "Order value"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return chart, order_rows


def format_volume(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value / 1_000:.0f}K"


def make_chart(bars, symbol, chart_type, dark_mode):
    background = "#0a0e14" if dark_mode else "#ffffff"
    foreground = "#e6edf3" if dark_mode else "#085fb1"
    grid = "#232b36" if dark_mode else "#edf2f7"
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
                increasing_line_color="#26a69a",
                increasing_fillcolor="#26a69a",
                decreasing_line_color="#ef5350",
                decreasing_fillcolor="#ef5350",
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
                line={"color": "#26a69a" if dark_mode else "#147d92", "width": 3},
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
        height=460,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.05, "x": 0},
        plot_bgcolor=background,
        paper_bgcolor=background,
        font={"color": foreground},
        xaxis={"showgrid": False, "rangeslider": {"visible": False}},
        yaxis={"showgrid": True, "gridcolor": grid, "tickprefix": "$"},
    )
    return chart


def make_small_chart(bars, symbol, chart_type, dark_mode):
    """Same as make_chart but sized for the multi-chart grid panels."""
    figure = make_chart(bars, symbol, chart_type, dark_mode)
    figure.update_layout(height=270)
    return figure


# ==========================================================================
# Terminal-specific render helpers
# ==========================================================================

def render_ticker_tape(rows):
    items = []
    for row in sorted(rows, key=lambda r: r["symbol"]):
        direction = "up" if row["pct_change"] >= 0 else "down"
        arrow = "▲" if row["pct_change"] >= 0 else "▼"
        items.append(
            f'<span class="ticker-item"><span class="ticker-sym">{row["symbol"]}</span>'
            f'<span class="ticker-px">${row["price"]:.2f}</span>'
            f'<span class="{direction}">{arrow} {row["pct_change"]:+.2f}%</span></span>'
        )
    st.markdown(
        f'<div class="ticker-tape-wrap">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )


def style_watchlist_table(table, match_symbols):
    def colorize_change(value):
        try:
            numeric = float(str(value).replace("%", "").replace("+", ""))
        except ValueError:
            return ""
        color = "#26a69a" if numeric >= 0 else "#ef5350"
        return f"color: {color}; font-weight: 600;"

    def highlight_match(row):
        if row.get("Status") == "MATCH":
            return ["background-color: rgba(47,129,247,0.12)"] * len(row)
        return [""] * len(row)

    styler = table.style.map(colorize_change, subset=["Today"])
    styler = styler.apply(highlight_match, axis=1)
    return styler


st.markdown('<div class="eyebrow">Live market monitor</div>', unsafe_allow_html=True)
st.title("Stock Scanner")

with st.sidebar:
    st.markdown("#### Scanner")
    with st.expander("Filters", expanded=True):
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

    st.markdown("#### Chart grid")
    with st.expander("Symbols & paging", expanded=False):
        selected_symbols = st.multiselect(
            "Symbols in the chart grid",
            CHART_SYMBOLS,
            default=WATCHLIST[:6],
            help="Shown in the bottom 'Charts Grid' tab.",
        )
        charts_per_page = st.slider("Charts per page", 2, 8, 4)
        page_count = max(1, (len(selected_symbols) + charts_per_page - 1) // charts_per_page)
        page_number = st.number_input(
            "Chart page",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            disabled=page_count == 1,
        )

    st.markdown("#### Refresh")
    with st.expander("Auto-refresh", expanded=False):
        refresh_seconds = st.slider("Refresh interval", 15, 300, 30, step=15)
        if st.button("Refresh now", use_container_width=True):
            load_snapshot.clear()
            load_chart.clear()
            load_yahoo_market_depth.clear()
            load_alpaca_market_depth.clear()
            st.rerun()
        st.caption(f"Updates run every {refresh_seconds}s. Yahoo Finance may still be delayed.")

# --------------------------------------------------------------------------
# Chart toolbar -- symbol / timeframe / chart type, TradingView-style, sits
# directly above the three-pane workspace.
# --------------------------------------------------------------------------
chart_source_options = ["Yahoo Finance"]
if alpaca_credentials_available():
    chart_source_options.append("Alpaca (stocks)")

toolbar = st.columns([2.2, .9, .9, 1.1, .8, 1.3, .9])
with toolbar[0]:
    default_index = CHART_SYMBOLS.index(WATCHLIST[0]) if WATCHLIST and WATCHLIST[0] in CHART_SYMBOLS else 0
    lead_symbol = st.selectbox("Symbol", CHART_SYMBOLS, index=default_index, label_visibility="collapsed")
with toolbar[1]:
    chart_interval = st.selectbox("TF", ["1m", "5m", "15m", "1h", "1d"], label_visibility="collapsed")
with toolbar[2]:
    chart_period_options = ["1d"] if chart_interval == "1m" else ["1d", "5d", "1mo"]
    chart_period = st.selectbox("Range", chart_period_options, label_visibility="collapsed")
with toolbar[3]:
    chart_type = st.selectbox("Type", ["Candlestick", "Line"], label_visibility="collapsed")
with toolbar[4]:
    dark_mode = st.toggle("Dark", value=True)
with toolbar[5]:
    chart_source = st.selectbox("Source", chart_source_options, label_visibility="collapsed")
with toolbar[6]:
    audio_alerts = st.toggle("Alerts", value=False)


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
    for symbol in sorted(new_match_symbols):
        st.toast(f"Scanner match: {symbol}", icon="🚨")

    lead_row = next((row for row in rows if row["symbol"] == lead_symbol), None)
    paper_client_for_orders, paper_client_error = get_paper_client()

    render_ticker_tape(rows)

    # --------------------------------------------------------------------
    # helper shared by the chart-grid and matches tabs at the bottom
    # --------------------------------------------------------------------
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
                        use_alpaca = chart_source == "Alpaca (stocks)" and symbol in TRADING_SYMBOLS
                        alpaca_error = None
                        if use_alpaca:
                            bars, alpaca_error = load_alpaca_chart(symbol, chart_period, chart_interval)
                        else:
                            bars = load_chart(symbol, chart_period, chart_interval)
                        if bars.empty:
                            if alpaca_error:
                                st.warning(f"Alpaca chart unavailable for {symbol}: {alpaca_error}")
                            else:
                                st.info(f"No bars are available for {symbol} from Yahoo Finance right now.")
                        else:
                            rising_now = len(bars) >= 2 and float(bars["Close"].iloc[-1]) > float(bars["Close"].iloc[-2])
                            trend_history = st.session_state.setdefault("chart_rising_state", {})
                            was_rising = trend_history.get(symbol)
                            is_chart_page = key_prefix.startswith("charts-page-")
                            upturn_started = is_chart_page and rising_now and was_rising is False
                            trend_history[symbol] = rising_now
                            st.plotly_chart(
                                make_small_chart(bars, symbol, chart_type, dark_mode),
                                use_container_width=True,
                                config={"displaylogo": False},
                                key=f"{key_prefix}-{row_start + column_index}-{symbol}-{chart_period}-{chart_interval}-{chart_source}",
                            )
                            latest_bar = bars.iloc[-1]
                            st.caption(
                                f"Last bar: {bars.index[-1].strftime('%H:%M')}  ·  "
                                f"Close ${latest_bar['Close']:.2f}  ·  Updated {timestamp}"
                            )
                            if upturn_started:
                                st.success(f"Upturn detected: {symbol}")
                                if audio_alerts:
                                    components.html(
                                        """
                                        <script>
                                        const context = new (window.AudioContext || window.webkitAudioContext)();
                                        const oscillator = context.createOscillator();
                                        const gain = context.createGain();
                                        oscillator.frequency.value = 880;
                                        gain.gain.setValueAtTime(0.0001, context.currentTime);
                                        gain.gain.exponentialRampToValueAtTime(0.18, context.currentTime + 0.02);
                                        gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.28);
                                        oscillator.connect(gain).connect(context.destination);
                                        oscillator.start();
                                        oscillator.stop(context.currentTime + 0.3);
                                        </script>
                                        """,
                                        height=0,
                                    )

                            if paper_client_for_orders is not None and symbol in TRADING_SYMBOLS:
                                with st.form(f"paper-orders-{key_prefix}-{symbol}"):
                                    order_quantity = st.number_input(
                                        "Quantity",
                                        min_value=0.0001,
                                        value=1.0,
                                        step=1.0,
                                        key=f"chart-qty-{key_prefix}-{symbol}",
                                    )
                                    confirm_order = st.checkbox(
                                        "Confirm paper order",
                                        key=f"chart-confirm-{key_prefix}-{symbol}",
                                    )
                                    buy_button, sell_button = st.columns(2)
                                    buy_order = buy_button.form_submit_button("Buy", type="primary")
                                    sell_order = sell_button.form_submit_button("Sell")
                                if buy_order or sell_order:
                                    if not confirm_order:
                                        st.warning("Confirm the paper order before submitting.")
                                    else:
                                        order_side = OrderSide.BUY if buy_order else OrderSide.SELL
                                        order_request = MarketOrderRequest(
                                            symbol=symbol,
                                            qty=order_quantity,
                                            side=order_side,
                                            time_in_force=TimeInForce.DAY,
                                        )
                                        try:
                                            order = paper_client_for_orders.submit_order(order_data=order_request)
                                            st.success(f"Paper {order_side.value} order submitted: {order.id}")
                                        except Exception as error:
                                            st.error(f"Paper order rejected: {error}")

    # ======================================================================
    # THREE-PANE WORKSPACE: watchlist | main chart | order entry & account
    # ======================================================================
    watchlist_col, chart_col, order_col = st.columns([1.2, 3.3, 1.3], gap="small")

    # ---- LEFT: watchlist / scanner table ---------------------------------
    with watchlist_col:
        with st.container(key="watchlist-panel"):
            st.markdown("### Watchlist")
            table = pd.DataFrame(rows)
            if not table.empty:
                table = table.sort_values("pct_change", ascending=False, key=lambda s: s.abs())
                table["status_flag"] = table["symbol"].map(lambda symbol: "MATCH" if symbol in match_symbols else "")
                display_table = pd.DataFrame({
                    "Symbol": table["symbol"],
                    "Price": table["price"].map(lambda v: f"{v:.2f}"),
                    "Today": table["pct_change"].map(lambda v: f"{v:+.2f}%"),
                    "Status": table["status_flag"],
                })
                st.dataframe(
                    style_watchlist_table(display_table, match_symbols),
                    use_container_width=True,
                    hide_index=True,
                    height=560,
                )
            st.caption(f"{len(matches)} of {len(rows)} match current filters · updated {timestamp}")

    # ---- CENTER: main chart -----------------------------------------------
    with chart_col:
        with st.container(key="main-chart-panel"):
            header_cols = st.columns([2, 1, 1, 1])
            header_cols[0].markdown(f"### {lead_symbol}")
            if lead_row:
                header_cols[1].metric("Last", f"${lead_row['price']:.2f}")
                header_cols[2].metric("Change", f"{lead_row['pct_change']:+.2f}%")
                header_cols[3].metric("Vol / avg", f"{lead_row['vol_ratio']:.2f}x")

            use_alpaca_main = chart_source == "Alpaca (stocks)" and lead_symbol in TRADING_SYMBOLS
            main_alpaca_error = None
            if use_alpaca_main:
                main_bars, main_alpaca_error = load_alpaca_chart(lead_symbol, chart_period, chart_interval)
            else:
                main_bars = load_chart(lead_symbol, chart_period, chart_interval)

            if main_bars.empty:
                if main_alpaca_error:
                    st.warning(f"Alpaca chart unavailable for {lead_symbol}: {main_alpaca_error}")
                else:
                    st.info(f"No bars are available for {lead_symbol} right now.")
            else:
                st.plotly_chart(
                    make_chart(main_bars, lead_symbol, chart_type, dark_mode),
                    use_container_width=True,
                    config={"displaylogo": False},
                    key=f"main-chart-{lead_symbol}-{chart_period}-{chart_interval}-{chart_source}",
                )
                st.caption(
                    f"Last bar: {main_bars.index[-1].strftime('%H:%M')}  ·  "
                    f"Close ${main_bars['Close'].iloc[-1]:.2f}  ·  Updated {timestamp}"
                )

    # ---- RIGHT: market depth + order entry + account ----------------------
    with order_col:
        with st.container(key="order-panel"):
            st.markdown("### Market depth")
            depth_uses_alpaca = chart_source == "Alpaca (stocks)" and lead_symbol in TRADING_SYMBOLS
            depth_error = None
            try:
                if depth_uses_alpaca:
                    depth, depth_error = load_alpaca_market_depth(lead_symbol)
                else:
                    depth = load_yahoo_market_depth(lead_symbol)
            except Exception as error:
                depth = None
                depth_error = str(error)

            alpaca_fallback_used = False
            if (
                depth_uses_alpaca
                and depth is not None
                and (depth.get("ask") in (None, 0) or depth.get("ask_size") in (None, 0))
            ):
                yahoo_depth = load_yahoo_market_depth(lead_symbol)
                if yahoo_depth is not None and yahoo_depth.get("ask") not in (None, 0):
                    yahoo_depth = dict(yahoo_depth)
                    yahoo_depth["source"] = "Yahoo Finance fallback"
                    depth = yahoo_depth
                    alpaca_fallback_used = True

            if depth is None:
                st.warning(f"Quote unavailable for {lead_symbol}: {depth_error or 'no quote returned'}")
            else:
                bid_col, ask_col = st.columns(2)
                bid_col.metric("Bid", f"${depth['bid']:.2f}" if depth["bid"] is not None else "--")
                ask_col.metric("Ask", f"${depth['ask']:.2f}" if depth["ask"] is not None else "--")
                st.caption(
                    f"{depth['source']}" + (" · Alpaca ask unavailable" if alpaca_fallback_used else "")
                )

            st.markdown("### Order entry")
            if paper_client_error:
                st.info("Connect a paper account in the 'Paper Account' tab below to trade.")
            else:
                order_symbol_default = lead_symbol if lead_symbol in TRADING_SYMBOLS else TRADING_SYMBOLS[0]
                with st.form("order_entry_form", clear_on_submit=False):
                    order_symbol = st.selectbox(
                        "Symbol", TRADING_SYMBOLS,
                        index=TRADING_SYMBOLS.index(order_symbol_default),
                    )
                    order_type = st.radio("Type", ["Market", "Limit"], horizontal=True)
                    order_quantity = st.number_input("Qty", min_value=0.0001, value=1.0, step=1.0)
                    limit_price = st.number_input(
                        "Limit price", min_value=0.01, value=100.0, step=0.01,
                        disabled=order_type == "Market",
                    )
                    confirm_order = st.checkbox("Confirm paper order")
                    buy_col, sell_col = st.columns(2)
                    with buy_col:
                        st.markdown('<div class="buy-btn">', unsafe_allow_html=True)
                        submit_buy = st.form_submit_button("BUY", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    with sell_col:
                        st.markdown('<div class="sell-btn">', unsafe_allow_html=True)
                        submit_sell = st.form_submit_button("SELL", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                if submit_buy or submit_sell:
                    if not confirm_order:
                        st.warning("Confirm the paper-order checkbox before submitting.")
                    else:
                        side = OrderSide.BUY if submit_buy else OrderSide.SELL
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
                            order = paper_client_for_orders.submit_order(order_data=order_request)
                            st.success(f"Paper order submitted: {order.id} ({order.status})")
                        except Exception as error:
                            st.error(f"Paper order rejected: {error}")

            st.markdown("### Account")
            if paper_client_error:
                st.caption(paper_client_error)
            else:
                try:
                    account = paper_client_for_orders.get_account()
                    record_portfolio_value(account.portfolio_value)
                    acct_col1, acct_col2 = st.columns(2)
                    acct_col1.metric("Buying power", format_money(account.buying_power))
                    acct_col2.metric("Portfolio", format_money(account.portfolio_value))

                    positions = paper_client_for_orders.get_all_positions()
                    if positions:
                        position_rows = [
                            {
                                "Sym": position.symbol,
                                "Qty": position.qty,
                                "Value": format_money(position.market_value),
                                "P/L": format_money(position.unrealized_pl),
                            }
                            for position in positions
                        ]
                        st.dataframe(
                            pd.DataFrame(position_rows),
                            use_container_width=True, hide_index=True, height=160,
                        )
                    else:
                        st.caption("No open positions.")
                except Exception as error:
                    st.error(f"Could not load account: {error}")

    st.divider()

    # ======================================================================
    # BOTTOM PANEL: everything else, tabbed, like a terminal's bottom dock
    # ======================================================================
    grid_tab, matches_tab, orders_tab, paper_tab, info_tab, trending_tab = st.tabs(
        ["Charts Grid", "Scanner Matches", "Orders", "Paper Account", "Company & News", "Trending"]
    )

    with grid_tab:
        st.subheader(f"Price tracking · page {page_number} of {page_count}")
        page_start = (page_number - 1) * charts_per_page
        page_symbols = selected_symbols[page_start:page_start + charts_per_page]
        render_chart_grid(page_symbols, f"charts-page-{page_number}")
        st.download_button(
            "Download snapshot CSV",
            pd.DataFrame(rows).to_csv(index=False),
            file_name="market_snapshot.csv",
            mime="text/csv",
        )

    with matches_tab:
        st.subheader(f"Current matches · {len(matches)}")
        st.caption("Updates automatically and only shows stocks that currently satisfy the scanner filters.")
        render_chart_grid([row["symbol"] for row in matches], "matches")

    with orders_tab:
        st.subheader("Paper-account orders")
        st.caption("This graph shows orders from the connected Alpaca paper account only.")
        orders_client, orders_error = get_paper_client()
        if orders_error:
            st.info(orders_error)
        else:
            try:
                order_filter = GetOrdersRequest(
                    status=QueryOrderStatus.ALL,
                    limit=100,
                    nested=True,
                )
                account_orders = orders_client.get_orders(filter=order_filter)
                if not account_orders:
                    st.info("No paper-account orders are available yet.")
                else:
                    order_chart, order_rows = make_orders_chart(account_orders)
                    st.plotly_chart(
                        order_chart,
                        use_container_width=True,
                        config={"displaylogo": False},
                        key="paper-account-orders-chart",
                    )
                    order_table = pd.DataFrame([
                        {
                            "Time": row["time"].strftime("%Y-%m-%d %H:%M"),
                            "Symbol": row["symbol"],
                            "Side": row["side"].title(),
                            "Value": format_money(row["notional"]),
                            "Status": row["status"].title(),
                        }
                        for row in order_rows
                    ])
                    st.dataframe(order_table, use_container_width=True, hide_index=True)
            except Exception as error:
                st.error(f"Could not load paper-account orders: {error}")

    with paper_tab:
        st.subheader("Alpaca paper trading account")
        st.caption("Paper environment only. Orders are simulated and never sent to a live brokerage account.")

        credentials_available = session_credentials_available()
        with st.expander("Connect paper account", expanded=not credentials_available):
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
                    st.rerun()

            if credentials_available:
                if st.button("Disconnect paper account"):
                    delete_session_credentials()
                    st.session_state.alpaca_connected = False
                    st.rerun()

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

                with st.expander("Portfolio value history", expanded=True):
                    history = st.session_state.get("portfolio_value_history", [])
                    st.caption("Live history collected during this Streamlit session. It resets when the session ends.")
                    if len(history) >= 1:
                        st.plotly_chart(
                            make_portfolio_chart(history),
                            use_container_width=True,
                            config={"displaylogo": False},
                            key="paper-portfolio-value-history",
                        )
                    else:
                        st.info("Portfolio history will appear after the first account refresh.")

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
            except Exception as error:
                st.error(f"Could not load Alpaca paper account: {error}")

    with info_tab:
        st.subheader(f"Company context · {lead_symbol}")
        try:
            context = load_company_context(lead_symbol)
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
            info_col1, info_col2 = st.columns([1, 1.4])
            with info_col1:
                st.dataframe(
                    pd.DataFrame(context_rows),
                    use_container_width=True, hide_index=True, height=260,
                )
            with info_col2:
                st.markdown("##### Latest news")
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

    with trending_tab:
        st.subheader("Trending tickers")
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


live_dashboard()

