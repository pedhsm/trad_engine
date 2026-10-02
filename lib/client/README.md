# `lib/client/` — supply the Interactive Brokers TWS API yourself

This directory is intentionally **empty** in the repository. `cpp_engine` is built
against the Interactive Brokers TWS API C++ client, and that code **cannot be
redistributed**: every IB header carries

> `Copyright (C) Interactive Brokers LLC. All rights reserved. This code is subject
> to the terms and conditions of the IB API Non-Commercial License or the IB API
> Commercial License, as applicable.`

"All rights reserved" means re-hosting it here would violate that license, no matter
how it is credited. So you install it locally instead — it takes a couple of minutes.

## How to populate it

This integration was written against **TwsApiC++** by **Jan Boonen** — a portable,
vendor-independent C++ wrapper around IB's POSIX API (the `TwsApiL0` / `EWrapperL0`
layer). Full credit to that project; it is what made this integration straightforward.

1. Get the **IB API** from Interactive Brokers:
   <https://interactivebrokers.github.io/> (or the TWS API download in your IBKR
   account). Accept IB's API license.
2. Get **TwsApiC++** (Jan Boonen's wrapper) and place its client sources here, so that
   this directory contains the headers `cpp_engine` includes, e.g.:
   ```
   lib/client/EClient.h  EWrapper.h  Contract.h  Order.h  Execution.h
   lib/client/TwsApiL0.h TwsApiL0.cpp  TwsApiDefs.h  CommonDefs.h  …
   ```
3. Build `cpp_engine` (see `cpp_engine/CMakeLists.txt`). The include path
   `../../lib/client/*` will now resolve.

## Delayed market data needs a 5-line patch

Without a real-time market-data subscription (the usual case on a fresh paper
account) IBKR sends **delayed** ticks: price types 66/67/68 instead of 1/2/4. The
size of each tick rides on the price message, and the client is what turns it into
a separate `tickSize` callback, but this client version only does that for the
real-time types. The engine publishes a trade on `tickSize(LAST_SIZE)`, so on delayed
data **no trade ever reaches the strategy**: the engine connects, reconciles and
trades, but the bars stay empty.

Fix in your local copy (found and verified live against a paper account): in
`EClientSocketBaseImpl.h`, case `TICK_PRICE`, the `switch` that maps a price tick
type to its size tick type ends with the `LAST` case. Add a `default` branch after it:

```cpp
default:
    // Delayed market data: 66/67/68 (bid/ask/last) -> 69/70/71 (their sizes).
    if (tickTypeInt >= 66 && tickTypeInt <= 68)
        sizeTickType = (TickType)(tickTypeInt + 3);
break;
```

With a real-time subscription you do not need it. Newer official IB API clients
already map the delayed types.

Everything in this directory except this README is git-ignored, precisely so the IB
code you drop here never gets committed or redistributed.
