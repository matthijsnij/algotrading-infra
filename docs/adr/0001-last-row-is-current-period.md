# The last OHLCV row is always the current period

An OHLCV DataFrame handed to a bot always ends with the **current** period — complete or not — in both live and backtest. Bots that need settled data call `closed_bars(df)` to drop it; `limit` counts closed bars, so a request for 100 bars returns 101 rows while a bar is forming.

## Context

Live and backtest silently disagreed. ccxt returns the in-progress candle and `PhemexExchange.fetch_ohlcv` passed it through untouched, while `DataWindow.get_window()` deliberately discarded the incomplete bucket. The same strategy therefore evaluated signals intrabar when live and only at bar close when backtested, and nobody had chosen that.

## Considered options

A per-bot `include_forming_bar` flag was rejected: it would have made the divergence configurable rather than fixing it, and the safe default would have kept live and backtest disagreeing for any bot that didn't opt in. Always excluding the forming bar was rejected because it makes intrabar entry impossible — a bot on a `4h` timeframe scanning every 10 minutes would see identical data on every scan.

## Consequences

The tick must fire *after* a period boundary, never on the final base bar of the period. A backtest gated on `is_signal_bar_close()` fires one base bar too early: at that cursor the just-completed period is the last row and no new period has opened, so dropping the last row yields data one full period stale. The polling backtest fires on scan boundaries and lands after the close, matching live. This off-by-one is the primary thing to test.

`BaseExchange.fetch_ohlcv`'s `limit` contract is now "closed bars", which third-party exchange adapters must honour.
