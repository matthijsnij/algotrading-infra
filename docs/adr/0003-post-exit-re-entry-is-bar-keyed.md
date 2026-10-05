# Post-exit re-entry is keyed to bars, not scans

After a trade closes, a bot may not open a new position until a bar has closed that had not closed at the moment of the exit. The guard is stamped from the **exit**, not the entry, and it is universal: no bot opts out, and its duration is exactly one bar with no configuration.

Read alongside ADR 0002, which resolves the superficially identical failed-entry case the opposite way.

## Context

Closed-bar signals do not change until the bar does. `is_long_breakout` reads the last closed bar, so a bot that exits at 12:30 on a `4h` timeframe re-evaluates at 12:40 against byte-identical data and re-enters with no new information. `CryptoBot` papered over this with `cooldown_minutes`, a wall-clock stand-in that is per-bot, has to be retuned by hand whenever the timeframe changes, and silently drifts wrong when it isn't.

The two clocks are genuinely different. What must change before a *retry* of a failed attempt is sensible is exchange- or account-side, so ADR 0002 counts scans. What must change before a *repeat entry* is sensible is the data the decision reads, so this counts bars.

## Considered options

**Keying the guard to the bar the closed trade was entered on** was the framing the issue started from, and it is wrong whenever a trade is held for more than one bar. Enter at 12:10 with the 08:00 bar newest, exit at 21:30 with the 16:00 bar newest: the entry-bar key is already satisfied, so the bot re-enters ten minutes after being stopped out, on a bar that had closed hours earlier and was fully visible for the entire life of the trade. The exit-bar key blocks until the 20:00 bar closes — the bar that contains the stop-out itself. For a fast stop-out inside a single bar the two keys behave identically, so the exit key blocks a strict superset, and the extra thing it blocks is the whole problem.

**Letting each bot declare its entry cadence** was rejected as wrong-by-default. It would have had every bot author restate something the framework cannot verify, on a library whose filters read the Forming Bar and whose example signals read the last Closed Bar without marking either — precisely the mix `CryptoBot` fell into unnoticed.

**Distinguishing Forming Bar from Closed Bar in the guard itself** turned out to be a non-question. The Forming Bar's open time and the newest Closed Bar's open time advance at the same instants, one timeframe apart, so a guard keyed to either releases at the same moment. The distinction changes only the label, never the timing.

**A configurable number of bars** was rejected: a tunable count is how `cooldown_minutes` grows back.

## Consequences

The guard is a **correctness fix** for any entry decision containing a Closed Bar input, because those inputs are provably unchanged across an exit. For a decision reading only the Forming Bar it degrades to a **frequency policy** — one trade per bar — which a bot could legitimately reject. It is universal anyway: no such bot exists, every mixed decision inherits the Closed Bar case, and a purely intrabar strategy runs a short timeframe so one bar costs it little. Revisit if a bot appears whose entry decision reads no Closed Bar input at all.

`BaseBot.tick()` already snapshots the mode around `on_tick()` to reset the Entry Failure Backoff and fire the position-opened notification. Stamping the exit bar on the mirror transition *out of* `IN_POSITION` costs a few lines in the same block, needs no cooperation from the bot, and needs no contract test — unlike `record_entry_failure()`, which ADR 0002 had to leave as a manual call guarded by a registry test. It also covers exit paths a bot author would forget, including the emergency close inside `on_halt()`, which is exactly the case where instant re-entry after a watchdog restart is worst.

This required `BaseState.reset()` to stop clearing `last_closed_bar`. That field is a fact about the data feed, not about the trade, and clearing it also cost the calendar staleness backstop a tick after every trade close.

The guard does not prevent a bot from re-entering the same signal on every subsequent bar. `is_long_breakout` and the whole `domain/signals/` library are level-triggered — "price is above the level" persists for as long as price stays there — so a sustained breakout can produce an entry on each new bar, every one of them legitimate under this guard. Making signals edge-triggered is a separate concern living in `domain/signals/`, and neither mechanism subsumes the other: edge-triggering alone still re-enters on the same bar after an exit, because the crossover is evaluated on the same unchanged pair of bars.
