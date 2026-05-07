# Hyperliquid WTI Oil — Goldman Roll Pricing Model

A first-principles **no-arbitrage pricing model** for [Hyperliquid's WTI perpetual contract](https://app.hyperliquid.xyz/trade/xyz:CL) (`xyz:CL`) during the monthly **Goldman Roll** window.

The model takes the two underlying CME WTI futures prices (front-month F, next-month N) and produces the fair perpetual price at every hour of the roll cycle, using only three axioms:

1. **Oracle formula**: `O(t) = w(t)·F + (1−w(t))·N` where `w(t)` shifts by 20% on each of the 5 roll days
2. **Funding rate**: `FR = 0.5 · [P + clamp(r−P, ±0.05%)] / 8` per hour where `P = basis/oracle, r = 0.01%/8h`
3. **No-arbitrage**: a hedged short-perp + long-N position has zero expected PnL at every hour

The model solves the resulting system via **iterative backward induction with Newton's method** — Oracle depends on perp during CME closure (EMA tracking with `τ=1h`), perp depends on Oracle via funding, so we iterate until convergence.

![preview](docs/preview.png)

## Features

- **Live data recording** — IBKR (CME front + next month) + Hyperliquid orderbook websocket, aligned to 3-second ticks, written to `live_log.csv`
- **Interactive web dashboard** (`pricing_tool.html`) — model curves, basis, delta hedge ratios, live history; scroll-zoom, drag-pan, click crosshair
- **Hyperliquid XYZ model**: 0.5× funding scaling, hourly settlement, EMA Oracle during CME closure (τ=30min)
- **Boros implied APR** — cumulative funding from now to expiry, useful for comparing against [Pendle Boros yield markets](https://boros.pendle.finance/)
- **Standalone Python model** (`model.py`) — call `solve(F, N, entry, exit, rolls)` to get the full hourly path
- **No trading code** — read-only dashboard

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set up IBKR

You need TWS or IB Gateway running with API access enabled:

- **Edit → Global Configuration → API → Settings**
- ✅ Enable ActiveX and Socket Clients
- ✅ Read-Only API (this project never sends orders)
- Socket port: `7496` (live) or `7497` (paper)
- Trusted IP: `127.0.0.1`
- Restart TWS

You also need market data subscriptions for **NYMEX CL futures** (front + next month).

### 3. Configure

```bash
cp .env.example .env
# edit .env to match your IBKR setup
```

### 4. Edit roll dates for the current cycle

Open `recorder.py` and update these constants for the current month's roll cycle:

```python
ENTRY = datetime(2026, 5, 1, 0, 0)   # start of the model window
EXIT  = datetime(2026, 5, 16, 0, 0)  # end of the model window
ROLLS = [
    datetime(2026, 5, 8,  22, 0),    # RD1 (5:30PM ET → snap to 22:00 UTC)
    datetime(2026, 5, 11, 22, 0),    # RD2
    datetime(2026, 5, 12, 22, 0),    # RD3
    datetime(2026, 5, 13, 22, 0),    # RD4
    datetime(2026, 5, 14, 22, 0),    # RD5
]
BOROS_EXPIRY = datetime(2026, 5, 20, 0, 0)
CL_FRONT_MONTH = '202606'  # CLM26
CL_NEXT_MONTH  = '202607'  # CLN26
```

The Goldman Roll always runs **business days 6–10** of the calendar month at **5:30 PM ET** (snapped to the next UTC hour). Update the same constants in `pricing_tool.html` (`RD`, `ENTRY`, `END`).

### 5. Run

```bash
# Terminal 1: data recorder + Flask API
python recorder.py

# Terminal 2: serve the frontend
python -m http.server 8888
```

Then open `http://localhost:8888/pricing_tool.html` and click **▶ 实时模式** (live mode).

### Standalone model usage

You don't need IBKR or the dashboard — just import and solve:

```python
from datetime import datetime
from model import solve

result = solve(
    F=95.78, N=89.13,
    entry=datetime(2026, 5, 1),
    exit_dt=datetime(2026, 5, 16),
    roll_datetimes=[
        datetime(2026, 5, 8, 22), datetime(2026, 5, 11, 22),
        datetime(2026, 5, 12, 22), datetime(2026, 5, 13, 22),
        datetime(2026, 5, 14, 22),
    ],
)

print(result['perp'])      # hourly perp prices
print(result['oracle'])    # hourly oracle prices
print(result['basis'])     # perp - oracle
print(result['funding_rate'])  # hourly FR
```

## Files

| File | Purpose |
|------|---------|
| `recorder.py` | Live data + model + Flask API (port 5111) |
| `pricing_tool.html` | Frontend dashboard |
| `model.py` | Standalone pricing model (clean, self-contained) |
| `exchange_v3.py` | Same model used by `recorder.py` |
| `requirements.txt` | Python deps |
| `.env.example` | Template for IBKR config |

## API endpoints

The recorder exposes two read-only endpoints on `localhost:5111`:

- `GET /api/prices` — current snapshot (F/N/oracle/perp/basis/FR/HL/deviation/...)
- `GET /api/history?n=3000` — historical CSV rows (server-side downsampled)

## Model verification

The model satisfies no-arbitrage to machine precision. Verified by Monte Carlo: 12,000+ random `(entry, exit)` pairs all give max PnL error of ~$5×10⁻⁶.

## License

MIT
