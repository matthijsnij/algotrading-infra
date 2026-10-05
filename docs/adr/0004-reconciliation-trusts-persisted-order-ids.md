# Reconciliation trusts persisted order IDs, not order-type strings

`BaseBot._reconcile()` identifies a position's SL/TP orders by looking up `state.sl_order_id` /
`state.tp_order_id` in the orders the exchange reports as open. It never inspects `order.type`.

## Context

The previous implementation guessed protection from the order-type string: `type == "stop"` was
read as the SL, `type == "limit"` as the TP. `NormalizedOrder.type` is a free-form `str`
populated independently by each adapter (`"stop_limit"` in `exchanges/live/phemex.py`,
`TYPE_STOP_LIMIT` in `exchanges/backtest/order_book.py`, Alpaca's strings unverified), with no
contract test pinning that vocabulary across adapters. A stop-limit-protected position — the
expected shape for stocks — read as unprotected under that guess and was needlessly
market-closed.

Order type doesn't establish protection anyway. A `stop`-type order can be a stop-*entry* in the
same direction as the position (`place_stop_buy()` is an entry tool, not just an exit tool).
Real protection requires opposite side, covering size, and a trigger on the losing side — none
of which the type string encodes.

## Decision

**Source of truth rule:** the exchange is authoritative about *what exists* — positions and open
orders. Persisted state is authoritative about *intent and history* — entry price, planned SL/TP
prices, trade count, and which order IDs this bot placed. Where they disagree about existence,
the exchange wins and the dependent persisted intent is discarded. Where the exchange has no
opinion (at what price did I enter? what was my planned target?), persisted state is the only
source there is.

This is why persisted IDs and exchange data are both needed: the persisted ID says *which order
was mine*, the exchange says *whether it still exists*. Neither alone answers the question — an
order-type guess answers a different, weaker question than either.

A discarded intent is not recoverable, which is why the paths below log CRITICAL and alert
rather than quietly self-correcting.

### When the bot closes versus resumes

The `Still open?` column refers to the persisted **SL order ID**; the position is open on the
exchange in every row below.

| Persisted mode | `sl_order_id` | Still open? | Action |
|---|---|---|---|
| `IN_POSITION` | set | yes | `on_reconcile()` → resume `IN_POSITION` |
| `IN_POSITION` | set | no | `halt(restartable=False)` → `on_halt()` closes + writes record |
| `IN_POSITION` | none | — | `halt(restartable=False)` → `on_halt()` closes + writes record |
| `IDLE` / fresh | — | — | `_reconcile()` closes → `halt(restartable=False)`, no record possible |

Close ownership follows the position/no-position split already in `_reconcile()`: `on_halt()`
owns the close whenever the bot has a persisted `IN_POSITION` record of the trade;
`_reconcile()` owns it only when it does not (mode isn't `IN_POSITION`, so `on_halt()` would be
a no-op). Exactly one owner closes — never both.

The unprotected-position close is `restartable=False` in every row: reaching it means the bot
found real money in a state it could not explain, and auto-resuming trading would be the
framework deciding an unexplained anomaly was fine. `prepare_bot()` leaves `restartable=False`
rows `HALTED`, requiring a manual DB clear — that is the intended "human reviews first" gate.

### Orphan order cleanup (no-position branch)

When no position is found, every open order on the symbol is cancelled except a still-open
pending entry order the bot itself is waiting on. This is deliberately blunt — it does not try
to distinguish "this bot's leftover SL" from "some other order" — and is only safe because
`runtime/live.py::_check_duplicate_symbols()` enforces one bot per symbol per account. If that
invariant is ever relaxed, this cleanup must be revisited first; nothing in `_reconcile()` itself
checks order ownership.

## Considered options

**A structural protection predicate** (stop price set, opposite side, covering size) was
rejected for now. It's the right upgrade path if unaccounted positions turn out to be common in
practice, but it leans on adapter fields (`stop_price`, `side`, `size` semantics) that are
equally unverified today. It needs a contract test over the adapters first — the same gap that
made the order-type guess unsound in the first place.

## Consequences

The bot trusts an SL order's **existence**, not its **adequacy** — an SL order covering only
part of the position still reads as protected. This is a narrower claim than "protected" often
implies, but it's the same claim `on_reconcile()` already made before this change.

On a size mismatch between the persisted position size and the exchange's reported size, the
exchange's size wins (`on_reconcile()` sets `state.position_size` from `position.size`); the
persisted SL/TP prices are kept as-is, since price is intent-and-history data the exchange has
no opinion on.

Row 2 (`IN_POSITION`, `sl_order_id` set, no longer open) becomes **routine rather than
exceptional** for a bot using day-only stops: a stock position held overnight whose stop expires
at market close reconciles as unprotected and halts for human review. That reading is correct —
the position genuinely was unprotected — and the fix belongs in the bot, which should use GTC
stops. Not actionable now; `StockBot.on_tick()` is still a no-op placeholder.

## Glossary

See `CONTEXT.md` for **Reconciliation**, **Orphaned Order**, **Orphaned Position**, and
**Protected Position**.
