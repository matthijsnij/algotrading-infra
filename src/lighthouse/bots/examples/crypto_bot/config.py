"""
================================================================================
CRYPTO BOT CONFIGURATION
================================================================================

Configuration dict for the CryptoBot strategy.

================================================================================
"""

crypto_config = {

    # ── Exchange / data ───────────────────────────────────────────────────────

    "symbol":       "BTC/USDT:USDT",   # trading symbol
    "quote_currency": "USDT",          # currency equity is denominated in
    "timeframe":    "1h",              # primary OHLCV timeframe (drives tick interval and main data fetch)
    "limit":        200,               # number of candles to fetch per tick
    "scan_interval":     3600,              # seconds between ticks when IDLE
    "position_interval": 60,                # seconds between ticks when IN_POSITION

    # ── Direction ─────────────────────────────────────────────────────────────

    "direction":    "both",            # "long", "short", or "both"

    # ── Breakout signal ───────────────────────────────────────────────────────

    "breakout_window":   20,           # bars to look back for swing high/low
    "atr_period":        14,           # ATR calculation period
    "atr_buffer_mult":   0.5,          # ATR multiplier for breakout confirmation buffer
                                       # close must exceed swing high/low by atr_buffer_mult * ATR

    # ── Stop loss / take profit ───────────────────────────────────────────────

    "sl_atr_mult":       1.5,          # SL distance = sl_atr_mult * ATR from entry price
    "rr_ratio":          2.0,          # TP distance = rr_ratio * SL distance (reward:risk)

    # ── Entry filters ─────────────────────────────────────────────────────────

    "volume_period":     20,           # lookback period for rolling volume average
    "volume_mult":       1.2,          # volume must be >= volume_mult * rolling average
    "atr_threshold":     50.0,         # minimum ATR value in price units to allow entry
                                       # prevents trading in flat / low-volatility conditions

    # ── Risk management ───────────────────────────────────────────────────────

    "risk_pct":          0.01,         # fraction of account balance to risk per trade (1%)
    "equity_scope":      "exchange",   # "exchange" (default) sizes against this bot's own exchange
                                       # equity; "system" sizes against Portfolio-wide equity (live-only,
                                       # see runtime/portfolio.py)
    "capital_fraction":  1.0,          # fraction of the scoped equity this bot may size against
                                       # (partitions system-wide equity across multiple bots)

    # ── Error handling ────────────────────────────────────────────────────────

    "max_consecutive_errors": 5,       # halt the bot after this many consecutive tick errors
    "max_entry_failures":     3,       # halt (non-restartable) after this many consecutive entry attempt failures

}
