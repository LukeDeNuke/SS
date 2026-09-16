# Stock Scanner

This project includes both the original terminal scanner and a live browser dashboard.

## Live dashboard

Install the dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. The dashboard refreshes the watchlist and
selected intraday charts automatically. Use the sidebar to choose multiple chart
symbols from the default watchlist, additional large-cap equities, and crypto
pairs, set the number of charts per page, move between pages, or change the
refresh interval. Each chart is labeled with its ticker and can be shown as a line
chart or candlestick chart, with dark chart mode enabled by default.
The main area is organized into Overview, Charts, and Matches tabs. The Matches
tab automatically displays charts only for stocks currently passing the scanner
filters, while sidebar controls are grouped into expandable sections.
The sidebar Market depth panel shows the selected symbol's latest bid, ask, sizes,
spread, and midpoint. This is top-of-book data; Yahoo Finance's public feed does
not provide the full Level 2 order book.
The main dashboard also includes a right-side Market info panel with top-of-book
data and a live list of Yahoo's most-active trending tickers.
The Scanner filters section lets you edit the default minimum and maximum price,
minimum daily move, minimum volume ratio, unusual volume, RSI range, MACD momentum,
and bullish-crossover setting. It includes starter presets and lets you save custom
presets for the current session. The Overview tab also supports CSV snapshot export.
The Overview tab highlights symbols that newly enter the scanner matches on each
refresh.
The Paper Trading tab connects to an Alpaca paper account, shows buying power and
positions, and supports guarded market or limit stock orders. Install `alpaca-py`
and set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` to paper-account credentials
before launching the app. The integration forces `paper=True` and does not use
live trading endpoints.
You can also use the Connect paper account form in the Paper Trading tab. Its
masked credentials are kept in a temporary owner-only file for the current
Streamlit session; use Disconnect to delete that file. The temporary file is not
part of the project and is not committed to Git.
The Paper Trading tab also records the account portfolio value on each refresh
and displays a live session-only value history chart. That chart resets when the
Streamlit session ends.
The Orders tab separately graphs the connected Alpaca paper account's submitted
orders, with buy/sell markers, order value, status, and timestamp.
When an Alpaca paper account is connected, the Chart display controls include an
`Alpaca (stocks)` data source. Stock charts can then use Alpaca historical bars;
crypto charts continue to use Yahoo Finance because this integration uses
Alpaca's stock-data endpoint.
The Charts page also supports optional audio alerts when a tracked stock changes
from falling/flat to rising. Browser autoplay permissions may require interacting
with the page first. Connected Alpaca stock charts include confirmation-gated
paper Buy and Sell buttons below each graph.

Alternatively, create `.streamlit/secrets.toml` with:

```toml
ALPACA_API_KEY = "your_paper_key"
ALPACA_SECRET_KEY = "your_paper_secret"
```

Restart Streamlit after changing credentials so the running process reloads them.
Additional scanner signals include opening gaps and prior-20-day breakouts. Chart
range and timeframe can be changed from the sidebar. The right-side Market info
panel includes cached sector, valuation, dividend, analyst, earnings/news context
for the selected symbol.

Yahoo Finance data is unofficial and may be delayed or rate-limited. It is suitable
for learning and monitoring, not latency-sensitive trading.

## Terminal scanner

```bash
python stock_scanner.py
```