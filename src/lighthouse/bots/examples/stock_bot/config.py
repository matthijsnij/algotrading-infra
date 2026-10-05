"""
================================================================================
STOCK BOT CONFIGURATION
================================================================================

Configuration dict for the StockBot — an Example Bot (CONTEXT.md) running a
long-only moving-average crossover strategy on a holiday-aware, non-24/7
instrument.

================================================================================
"""

stock_config = {

    # ── Exchange / data ───────────────────────────────────────────────────────

    "symbol":       "AAPL",            # trading symbol (native ticker, no ccxt notation)
    "quote_currency": "USD",           # currency equity is denominated in
    "calendar":     "nyse",            # holiday-aware NYSE trading calendar
    "timeframe":    "1d",              # primary OHLCV timeframe (drives tick interval and main data fetch)
    "limit":        200,               # number of candles to fetch per tick
    "scan_interval":     86400,             # seconds between ticks when IDLE
    "position_interval": 3600,              # seconds between ticks when IN_POSITION

    # ── Strategy ──────────────────────────────────────────────────────────────

    "fast_period":      20,            # fast SMA period
    "slow_period":      50,            # slow SMA period
    "risk_pct":         0.01,          # fraction of account balance to risk per trade
    "sizing_stop_pct":  0.05,          # synthetic stop-loss distance used only for position sizing (no real SL order is placed)

    # ── Error handling ────────────────────────────────────────────────────────

    "max_consecutive_errors": 5,       # halt the bot after this many consecutive tick errors
    "max_entry_failures":     3,       # halt (non-restartable) after this many consecutive entry attempt failures

}
