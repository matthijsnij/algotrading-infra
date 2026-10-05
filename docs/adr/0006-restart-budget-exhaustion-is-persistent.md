# Restart budget exhaustion is persistent

**Status:** accepted — implemented

The watchdog's Restart Budget is in-memory bookkeeping, rebuilt from zero every time the
live process starts. A bot that exhausted its budget was therefore resurrected with a full
budget on the next process start, and Reconciliation cleared its halted state back to IDLE
because the halt was still marked restartable — so under a service manager, a crash-looping
bot loops forever and the CRITICAL "manual intervention required" alert never actually
requires any. We decided that exhausting the budget writes a non-restartable marker into the
bot's persisted state, making that halt terminal until a human clears it.

## Considered Options

We rejected persisting the budget itself (the attempt counter and the healthy-uptime clock).
It would close the same hole, but the healthy-reset rule is expressed against a monotonic
clock that is meaningless across processes, so persisting it would force a second, wall-clock
notion of "healthy for long enough" to exist alongside the first. Keeping the budget purely
in-memory and persisting only its *terminal outcome* needs no new time semantics.

We also rejected leaving it alone on the grounds that a human typing the start command is
itself the manual intervention. That holds while the process is started by hand and stops
holding the moment it is started by a service manager — which is the intended deployment.

## Consequences

The Restart Budget now caps the *rate* of restarts within a process, while exhaustion is a
per-bot terminal state that outlives it. Recovery requires editing the bot's persisted state,
so the state store needs a supported way to do that rather than hand-edited SQLite.

Under a service manager this converts an infinite trading loop into an infinite
start-skip-exit loop when every bot is terminally halted: the process starts, skips the
halted bots, finds nothing alive, and exits cleanly. That is noisy but safe, and it is the
intended failure mode.
