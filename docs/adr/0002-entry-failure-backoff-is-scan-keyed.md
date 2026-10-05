# Entry failure backoff is keyed to scans, not bars

When an entry attempt fails and leaves the bot flat, the bot waits a growing number of **scans** before attempting again (1, then 2, then 4), and halts once it has failed `max_entry_failures` times in a row without a successful entry in between. The wait is deliberately *not* keyed to bar boundaries.

Read alongside ADR 0003, which resolves the superficially identical post-exit case the opposite way.

See also ADR 0005: `close_trade()` promotes the *close* side of the sequence this ADR keeps
bot-owned on the *entry* side — that does not overturn the rejection below.

## Context

`scan_interval` and `timeframe` are independent, so a bot can wake many times per bar. The entry paths that bail out with `mode` still `IDLE` re-ran on every one of those scans against unchanged data. The damaging case was entry order fills, protective-order placement fails, emergency close — two fees per attempt, repeated every scan for the rest of the bar.

## Considered options

**Suppressing further attempts for the remainder of the bar** was the obvious candidate and was rejected for two reasons. First, every cause of an entry failure is exchange- or account-side (desync, unsizeable position, rejected order), so the bar clock — which governs when the *signal* changes — tracks the wrong thing entirely; waiting for the next bar just delays the alert without making a retry any more likely to succeed. Second, a bot's data cadence is not knowable even in principle: `CryptoBot` reads the forming bar for its ATR and volume filters (`iloc[-1]`) and the last closed bar for its breakout signal (`iloc[-2]`), so "one attempt per bar" has no single meaning for it.

**Wall-clock backoff** was rejected in favour of counting scans, so the mechanism behaves identically in backtest, where simulated time is the only time there is.

Note also that the per-attempt cost is bounded by the failure cap, not by the wait: because the counter only resets on a successful entry, the total number of fee-burning attempts is the same under either scheme. The wait only controls how quickly you are told.

## Consequences

`can_trade()` returning `False` is exempt: it means the bot's state and the exchange disagree, so it halts on the first occurrence rather than backing off — and it halts `restartable=True`, because `_reconcile()` on restart is purpose-built to resolve exactly that disagreement. The backoff exhaustion halt is `restartable=False`, since a restart there would only repeat the same failing attempt.

`BaseBot` has no chokepoint on order placement — bots import `execution/orders.py` helpers directly — so it cannot detect a failed attempt on its own and the bot must call `record_entry_failure()` on each bail-out path. A contract test over the bot registry guards against authors forgetting.

Routing order placement through `BaseBot` would remove the manual call, and was rejected: wrapping each order helper in a delegating method is a shallow layer that concentrates no complexity, bought at the cost of widening `BaseBot`, which is meant to own orchestration only. The version worth wanting is a single `enter_position()` owning the whole sequence — size, enter, fetch fill, place protective orders, emergency-close on failure, update state — but with one strategy bot written, the seam is hypothetical. Revisit when a third bot duplicates that sequence.
