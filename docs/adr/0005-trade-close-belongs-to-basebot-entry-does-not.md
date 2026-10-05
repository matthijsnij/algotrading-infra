# Trade close belongs to BaseBot, entry does not

`BaseBot.close_trade()` is a concrete method every bot calls to finalize a closed trade — P&L,
fees, funding, `TradeRecord` construction and write, `state.reset()` — all generic across
strategies. There is no equivalent `enter_position()`; entry stays bot-owned, as ADR 0002 already
decided.

Read alongside ADR 0002, which this does not overturn (see Consequences).

## Context

Before this change, `CryptoBot._close_trade()` contained ~70 lines of accounting logic —
direction sign, gross/net P&L, fee handling, funding lookup, outcome derivation, `TradeRecord`
construction, logging, `self.trade_logger.log()`, `self.state.reset()` — that referenced only
`BaseState` fields and `BaseBot`-owned collaborators (`self.exchange`, `self.trade_logger`,
`self.config`). None of it read anything CryptoBot-specific. `StockBot`, on tick dispatch and
halt cleanup, duplicated the same shape by hand.

Meanwhile `on_tick()`'s mode dispatch (`IDLE` → scan, `IN_POSITION` → monitor, `WAITING_FILL` →
poll fill) is also identical across every bot that uses these three modes — it isn't a strategy
decision, it's just "which hook fires for which mode." `BaseBot.on_tick()` became a concrete
dispatcher to `on_scan()` / `on_position()` / `on_pending_fill()`, so subclasses implement only
the hooks, not the branch.

## Decision

Trade close generalizes because there is exactly one way to close a trade once you know
`(reason, exit_price, exit_fee, exit_time)`: compute P&L, subtract fees and funding, derive an
outcome, write a record, reset state. Every bot's close path funnels down to that same shape
regardless of *why* the trade closed — SL fill, TP fill, emergency close — because by the time
`close_trade()` is called, the caller has already resolved the exit terms. The strategy-specific
part (which order filled, what price to fall back to) stays in the bot; only the accounting that
follows moved to `BaseBot`.

Trade entry does not generalize, for the reason ADR 0002 already gives: a bot's data cadence and
signal evaluation aren't knowable in general (`CryptoBot` reads two different rows of the same
DataFrame for its filters versus its signal), and there is exactly one entry-side bot written
today. `record_entry_failure()` remains a call the bot author makes explicitly on every bail-out
path, not something `BaseBot` can detect on its own.

## Consequences

This does not overturn ADR 0002's rejection of wrapping order-placement helpers in `BaseBot`.
ADR 0002 rejected a *shallow* wrapper — one delegating method per `execution/orders.py` helper,
concentrating no complexity, bought at the cost of widening `BaseBot`'s surface for no benefit.
`close_trade()` is not that: it is the deep version ADR 0002 said was worth wanting for entry
too ("a single `enter_position()` owning the whole sequence") — except on the close side, one
bot's implementation already generalized cleanly with zero strategy-specific branches, so there
was no seam left to defer. Entry has no such existing generalization to promote; `close_trade()`
does.

Bots must resolve `exit_price` (including any fallback to a planned SL/TP price) before calling
`close_trade()` — the fallback is bot-specific (which price to fall back to, and whether a
fallback makes sense for a given exit reason at all), so it stays out of `BaseBot`.

Issue #14 may revisit the open path as a second bot is added; if its entry sequence turns out to
match `CryptoBot`'s shape, that is the point to promote entry the same way close was promoted
here.
