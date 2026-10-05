# Lighthouse

An asset-agnostic algorithmic trading framework: the same bot runs live against an exchange or against historical data through a simulated exchange.

## Language

**Timeframe**:
The fixed duration of a single price bar, written as a count and a unit (`5m`, `4h`, `1d`, `2w`). Minutes, hours, days and weeks only — calendar months are excluded because they have no fixed duration.
_Avoid_: Interval, period, resolution, granularity, candle size

**Base Timeframe**:
The timeframe of the historical bars a backtest is driven by. Every other timeframe in a backtest is aggregated up from it, so it is the finest resolution available to that run.
_Avoid_: Source timeframe, raw timeframe, data timeframe

**Metrics Timeframe**:
The timeframe the equity curve is resampled to when performance is reported. A presentation concern only — it never affects how a bot trades.
_Avoid_: Reporting interval, equity curve timeframe, resample rule

**Forming Bar**:
The current, still-incomplete period. It is always the last row of any OHLCV DataFrame handed to a bot, in live and backtest alike (ADR 0001), and its values change on every fetch until the period ends.
_Avoid_: Current bar, live bar, partial candle, in-progress bar

**Closed Bar**:
A period that has completed and whose values will never change again. `closed_bars(df)` drops the Forming Bar to leave only these, and a request for a `limit` of 100 bars means 100 Closed Bars.
_Avoid_: Settled bar, confirmed bar, finished candle, historical bar

**Tick**:
One iteration of a bot's run loop: fetch data, then call `on_tick()` unless suppressed (a stale calendar bar, or Entry Failure Backoff). `on_tick()` dispatches by `state.mode` — `on_scan()` while IDLE, `on_position()` while IN_POSITION, `on_pending_fill()` while WAITING_FILL — so a bot implements the hooks, not the branch. Every tick advances `state.lifetime_tick_count` whether or not `on_tick()` actually ran. How often ticks happen is controlled by whichever interval matches the bot's current mode — Scan Interval while IDLE, Position Interval while IN_POSITION — never by Timeframe directly.
_Avoid_: Cycle, poll, iteration, heartbeat

**Example Bot**:
A minimal, complete reference implementation of the bot lifecycle that shows how a strategy plugs into the framework. It is not a trading strategy recommendation, and nothing in shared code may depend on its specifics.
_Avoid_: Reference implementation (as a name), sample bot, demo bot, template bot

**Scan Interval**:
How often a bot takes a tick while IDLE (scanning for entry). A polling cadence, not a bar duration: it is unrelated to the bot's timeframe in live trading and is measured in seconds. In a backtest it must be a whole multiple of the base timeframe (ADR 0001): the simulation is bar-driven with no wall clock to poll, so it converts the Scan Interval into a fixed number of bars to advance between ticks instead.
_Avoid_: Scan timeframe, poll timeframe, tick interval

**Position Interval**:
How often a bot takes a tick while IN_POSITION (monitoring exit), replacing the Scan Interval for the duration of the position. Also a polling cadence in seconds, decoupled from timeframe — but unlike Scan Interval it carries no divisibility constraint against the base timeframe, because backtests never switch to it: `run_single()` drives every tick on Scan-Interval-derived bar boundaries regardless of mode, so Position Interval only takes effect in live trading.
_Avoid_: Monitoring interval, exit interval, position timeframe

**Entry Attempt**:
A bot acting on a live entry signal: everything from sizing the position through placing the entry order and its protective orders. It succeeds when the bot ends up holding a position and fails when the bot is left flat. A scan on which no signal fired is not an attempt.
_Avoid_: Entry, trade attempt, order attempt

**Entry Failure Backoff**:
The growing wait a bot observes after a failed Entry Attempt before it may attempt entry again, together with the cap on consecutive failures past which it stops trying and halts. Measured in scans rather than bars, because what makes an attempt fail is exchange- or account-side and so has nothing to do with the bot's Timeframe.
_Avoid_: Cooldown, retry limit, entry throttle

**Same-Bar Re-Entry Guard**:
The block on opening a new position while the newest Closed Bar is still the one that was newest when the last trade exited. Keyed to bars rather than the clock, because what must change before a repeat entry carries any new information is the data, not the elapsed time. Distinct from Entry Failure Backoff, which follows a failed Entry Attempt rather than a completed trade and is scan-keyed for the opposite reason (ADR 0002, ADR 0003).
_Avoid_: Cooldown, re-entry delay, trade spacing, same-bar entry guard

**Trade Close**:
The accounting event that ends a trade: realizing P&L from the entry and exit prices, netting fees and funding, writing the TradeRecord, and resetting per-trade state. Distinct from the exchange-side close — an SL or TP order filling, or an emergency close flattening the position — which is what *causes* it. Every exchange-side close should produce exactly one Trade Close, whatever the exit reason.
_Avoid_: Close (bare), trade exit, settlement, booking

**Signal**:
A boolean function in `domain/signals/` reporting a market condition on the last row of the OHLCV DataFrame it is handed — the caller picks the frame (ADR 0001). Level-triggered by default — it stays true for as long as the condition holds, e.g. `is_price_above_ema` for the whole time price holds above the average. `became_true()` (`domain/signals/triggers.py`) edge-triggers any Signal by comparing it one bar earlier; a bot chooses level or edge per call, nothing enforces either.
_Avoid_: Indicator, condition, trigger (bare)

**Filter**:
A Signal used to gate an Entry Attempt without choosing its side, e.g. `is_atr_above_threshold`, `is_volume_above_average`. A bot ANDs one or more filters with a directional Signal before acting; the filter narrows *when* a signal may act, not *which way*.
_Avoid_: Condition, gate

**Cold Start**:
A bot starting with no persisted state to load, so it begins from defaults — no tick history, no re-entry guard, no accumulated entry failures. Happens on a bot's very first run, after its persisted state is cleared, or whenever persistence is switched off. The only kind of start that carries no history.
_Avoid_: Fresh start, first run, clean start

**Warm Start**:
A bot starting from persisted state, so its counters, re-entry guard and trade-in-progress facts carry over from before it stopped. Deliberately indistinguishable from the bot's point of view whether the process is new or the watchdog rebuilt the bot in place — a bot never branches on how it was started, only on what state it loaded.
_Avoid_: Restart, resume, recovery start

**Restart Budget**:
The number of times the watchdog may rebuild a bot after a halt before it gives up and leaves the bot down for a human. Spent per halt and returned in full once a rebuilt bot has stayed healthy long enough, so it caps the rate of restarts rather than their lifetime total. Exhausting it is a terminal state for that bot: it survives the process ending, and only a human clearing the bot's persisted state brings it back.
_Avoid_: Retry limit, restart cap, attempt counter

**Reconciliation**:
The synchronous check `BaseBot._reconcile()` performs for each bot before its tick loop starts, comparing what the exchange reports (positions, open orders) against the bot's persisted state. The exchange is authoritative about what exists; persisted state is authoritative about intent and history (entry price, planned SL/TP, which order IDs are this bot's own) (ADR 0004).
_Avoid_: Recovery, restart check, startup sync

**Orphaned Order**:
An open order the exchange reports with no corresponding open position — a leftover from before the bot last stopped (e.g. an SL left resting after its TP filled) or a cancelled/failed Entry Attempt. Found in Reconciliation's no-position branch and cancelled, except a still-open pending entry order the bot is legitimately waiting on. Distinct from Orphaned Position, which is the reverse mismatch.
_Avoid_: Stale order, leftover order, dangling order

**Orphaned Position**:
An open position the exchange reports that the bot's persisted state cannot explain when Reconciliation runs — whether because state was never persisted, was reset to IDLE, or was opened before the bot last stopped and then lost track of. Whether it counts as a Protected Position is checked separately, after it is identified as orphaned. Distinct from Orphaned Order, which is a position missing rather than an order missing.
_Avoid_: Unmanaged position, unknown position, leftover position

**Protected Position**:
An Orphaned Position whose persisted `sl_order_id` resolves to a currently open order on the exchange. Reconciliation trusts the order's *existence*, not its *adequacy* — it does not check opposite side, covering size, or trigger price (ADR 0004). A position with no persisted `sl_order_id`, or one that no longer resolves to an open order, is unprotected regardless of what other orders happen to be open.
_Avoid_: Covered position, safe position, hedged position

