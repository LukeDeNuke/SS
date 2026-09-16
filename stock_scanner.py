"""
Simple Stock Scanner — real market data (via yfinance)
─────────────────────────────────────────────────────────────────────────
Same three-part structure as the in-chat demo, now backed by real data:

  1. DATA FEED     — fetch_snapshot() pulls intraday + daily bars from
                      Yahoo Finance for your watchlist.
  2. FILTER ENGINE — apply_filters(), a pure function: (rows, criteria)
                      -> matching rows. Extend this with your own logic.
  3. ALERTING      — the main loop diffs each poll's matches against the
                      last poll and prints anything newly matched.

SETUP
  pip install yfinance pandas

RUN
  python stock_scanner.py

Edit WATCHLIST and FILTERS below to change what it scans and for what.
Yahoo's free data is end-of-day for volume/fundamentals and can lag
intraday prices by a minute or so — fine for learning, not for
latency-sensitive trading. See the notes at the bottom of this file for
what to change if you outgrow it.
─────────────────────────────────────────────────────────────────────────
"""

import time
from datetime import datetime

import pandas as pd
import yfinance as yf

# ── Configuration ──────────────────────────────────────────────────────

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA",
    "GOOGL", "AMZN", "META", "NFLX", "CRM",
    "XOM", "CVX", "JPM", "BAC", "UNH",
]

FILTERS = {
    "price_min": 0,
    "price_max": 10_000,
    "min_abs_change_pct": 1.0,   # only show stocks moving at least this much today
    "min_vol_ratio": 0.5,        # today's volume-so-far vs its recent daily average
    "gap_min_pct": -100.0,
    "gap_max_pct": 100.0,
    "breakout_only": False,      # price must be above its prior 20-day high
    "unusual_volume_only": False, # require volume to be at least 2x its recent average
    "rsi_min": 0.0,
    "rsi_max": 100.0,
    "macd_bullish_only": False,
    "bullish_cross_only": False, # only show 5/20-day SMA bullish crossovers
}

POLL_SECONDS = 60  # Yahoo's free intraday data doesn't update much faster than this


# ── 1. DATA FEED ────────────────────────────────────────────────────────
def fetch_snapshot(tickers):
    """Pull the data needed for one scan pass and return a list of dicts,
    one per symbol. Two calls: 1-minute bars for today's price/volume,
    and daily bars for moving averages and the average-volume baseline."""

    intraday = yf.download(
        tickers, period="1d", interval="1m",
        group_by="ticker", progress=False, auto_adjust=False,
    )
    daily = yf.download(
        tickers, period="3mo", interval="1d",
        group_by="ticker", progress=False, auto_adjust=False,
    )

    rows = []
    for symbol in tickers:
        try:
            intra = intraday[symbol].dropna()
            day = daily[symbol].dropna()
            if intra.empty or len(day) < 20:
                continue

            price = float(intra["Close"].iloc[-1])
            open_price = float(intra["Open"].iloc[0])
            previous_close = float(day["Close"].iloc[-2])
            volume_today = float(intra["Volume"].sum())
            avg_volume = float(day["Volume"].tail(20).mean())
            prior_20_high = float(day["High"].iloc[-21:-1].max())

            sma5_series = day["Close"].rolling(5).mean()
            sma20_series = day["Close"].rolling(20).mean()
            sma5, sma20 = sma5_series.iloc[-1], sma20_series.iloc[-1]
            prev_sma5, prev_sma20 = sma5_series.iloc[-2], sma20_series.iloc[-2]

            delta = day["Close"].diff()
            gains = delta.clip(lower=0).rolling(14).mean()
            losses = (-delta.clip(upper=0)).rolling(14).mean()
            relative_strength = gains / losses.replace(0, pd.NA)
            rsi = (100 - (100 / (1 + relative_strength))).fillna(100).iloc[-1]

            ema12 = day["Close"].ewm(span=12, adjust=False).mean()
            ema26 = day["Close"].ewm(span=26, adjust=False).mean()
            macd_series = ema12 - ema26
            macd_signal_series = macd_series.ewm(span=9, adjust=False).mean()
            macd = macd_series.iloc[-1]
            macd_signal = macd_signal_series.iloc[-1]

            cross_signal = None
            if prev_sma5 <= prev_sma20 and sma5 > sma20:
                cross_signal = "up"
            elif prev_sma5 >= prev_sma20 and sma5 < sma20:
                cross_signal = "down"

            rows.append({
                "symbol": symbol,
                "price": price,
                "pct_change": (price - open_price) / open_price * 100,
                "gap_pct": (open_price - previous_close) / previous_close * 100,
                "breakout": price >= prior_20_high,
                "volume": volume_today,
                "vol_ratio": volume_today / avg_volume if avg_volume else 0,
                "sma5": float(sma5),
                "sma20": float(sma20),
                "cross_signal": cross_signal,
                "rsi": float(rsi),
                "macd": float(macd),
                "macd_signal": float(macd_signal),
            })
        except (KeyError, IndexError):
            # Symbol had no data this pass (delisted, no trades yet, etc).
            continue

    return rows


# ── 2. FILTER ENGINE ─────────────────────────────────────────────────────
def apply_filters(rows, f):
    """Pure function: (rows, criteria) -> matching rows. Add your own
    conditions here the same way (RSI, gap %, earnings date, etc)."""
    matches = []
    for r in rows:
        if not (f["price_min"] <= r["price"] <= f["price_max"]):
            continue
        if abs(r["pct_change"]) < f["min_abs_change_pct"]:
            continue
        if r["vol_ratio"] < f["min_vol_ratio"]:
            continue
        if not (f.get("gap_min_pct", -100.0) <= r.get("gap_pct", 0.0) <= f.get("gap_max_pct", 100.0)):
            continue
        if f.get("breakout_only", False) and not r.get("breakout", False):
            continue
        if f.get("unusual_volume_only", False) and r["vol_ratio"] < 2.0:
            continue
        if not (f.get("rsi_min", 0.0) <= r.get("rsi", 50.0) <= f.get("rsi_max", 100.0)):
            continue
        if f.get("macd_bullish_only", False) and r.get("macd", 0.0) <= r.get("macd_signal", 0.0):
            continue
        if f["bullish_cross_only"] and r["cross_signal"] != "up":
            continue
        matches.append(r)
    return matches


# ── Display helpers ───────────────────────────────────────────────────
def fmt_volume(v):
    return f"{v/1_000_000:.2f}M" if v >= 1_000_000 else f"{v/1000:.0f}K"


def print_table(rows):
    if not rows:
        print("  (no matches)")
        return
    header = f"{'SYMBOL':<8}{'PRICE':>10}{'CHG%':>9}{'VOLUME':>11}{'VOL/AVG':>10}{'SMA5':>10}{'SMA20':>10}  SIGNAL"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: x["vol_ratio"], reverse=True):
        signal = "▲ bull cross" if r["cross_signal"] == "up" else \
                 "▼ bear cross" if r["cross_signal"] == "down" else ""
        print(
            f"{r['symbol']:<8}"
            f"{r['price']:>10.2f}"
            f"{r['pct_change']:>+8.2f}%"
            f"{fmt_volume(r['volume']):>11}"
            f"{r['vol_ratio']:>9.2f}x"
            f"{r['sma5']:>10.2f}"
            f"{r['sma20']:>10.2f}"
            f"  {signal}"
        )


# ── 3. ALERTING + main loop ─────────────────────────────────────────────
def run():
    print(f"Scanning {len(WATCHLIST)} symbols every {POLL_SECONDS}s. Ctrl+C to stop.\n")
    prev_matches = set()

    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            rows = fetch_snapshot(WATCHLIST)
        except Exception as e:
            print(f"[{timestamp}] fetch failed: {e}")
            time.sleep(POLL_SECONDS)
            continue

        matches = apply_filters(rows, FILTERS)
        current_symbols = {r["symbol"] for r in matches}

        newly_matched = current_symbols - prev_matches
        for r in matches:
            if r["symbol"] in newly_matched:
                print(f"[{timestamp}] ALERT: {r['symbol']} entered scan "
                      f"({r['pct_change']:+.2f}%, {r['vol_ratio']:.2f}x volume)")

        print(f"\n[{timestamp}] {len(matches)}/{len(rows)} matching:")
        print_table(matches)
        print()

        prev_matches = current_symbols
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nStopped.")


# ─────────────────────────────────────────────────────────────────────────
# NOTES / WHERE TO GO NEXT
# ─────────────────────────────────────────────────────────────────────────
# - Yahoo's free intraday data is unofficial and occasionally rate-limits
#   or lags. For real trading use, a paid/licensed feed (Polygon.io,
#   Alpaca, IEX Cloud) with a proper API key is the reliable path.
# - Sector filtering: yfinance can fetch it via yf.Ticker(sym).info["sector"],
#   but .info is slow (one call per symbol) — fine to fetch once at
#   startup and cache in a dict if you want a sector filter back.
# - To persist alerts, write the `matches` list to a CSV each pass:
#     pd.DataFrame(matches).to_csv("alerts.csv", mode="a", header=False)
# - To scan the whole market instead of a fixed watchlist, you'd need a
#   symbol list from an exchange (e.g. NASDAQ's published ticker file)
#   and batch fetch_snapshot() calls in chunks — Yahoo will throttle
#   very large single requests.
# ─────────────────────────────────────────────────────────────────────────