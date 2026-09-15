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
symbols, set the number of charts per page, move between pages, or change the
refresh interval. Each chart is labeled with its ticker and can be shown as a
line chart or candlestick chart, with dark chart mode enabled by default.
The main area is organized into Overview and Charts tabs, while sidebar controls
are grouped into expandable sections.

Yahoo Finance data is unofficial and may be delayed or rate-limited. It is suitable
for learning and monitoring, not latency-sensitive trading.

## Terminal scanner

```bash
python stock_scanner.py
```