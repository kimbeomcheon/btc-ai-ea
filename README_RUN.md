# BTC AI EA — Phase 1 Research

This repository package implements the first validation stage for a BTC trend-following
EA research program.

## Strategy architecture

`4H Regime -> 1H Entry -> Anti-Martingale -> ATR Stop/Trailing -> Risk Locks`

- **Bull:** long entries only
- **Bear:** short entries only
- **Range:** no new entries
- **Shock:** no new entries; existing trailing stop tightens
- Signal is generated from a **completed 1h candle**
- Execution is no earlier than the **next 1h open**
- Initial stop: ~2 ATR (candidate-specific)
- Winner-only adds: +0.5 ATR and +1 ATR
- No averaging down / no martingale / no simultaneous hedge
- 1.0x notional cap in Phase 1
- Daily loss lock: 2%
- Weekly loss lock: 5%
- Soft drawdown: 10%
- Hard drawdown: 15%

## Candidates

- V1: Donchian 20 / ADX 20 / 2 ATR stop / 3 ATR trail
- V2: Donchian 30 / ADX 22 / 2.2 ATR stop / 3.2 ATR trail
- V3: Donchian 55 / ADX 18 / 2.5 ATR stop / 4 ATR trail

The full-history winner is **not automatically accepted**. Walk-forward OOS and
2x-cost stress are generated separately.

## Cost assumptions

Defaults are configurable:
- Trading fee: 5.5 bps **per fill**
- Slippage: 2.0 bps **per fill**
- Funding: not included in Phase 1

Phase 1 deliberately uses Binance public BTCUSDT spot 1h data as a long-history
price proxy. A later stage must re-validate on perpetual futures with funding/basis.

## Run

```bash
python -m pip install -r requirements.txt
python backtest.py --start 2017-08-17 --initial 10000
```

Outputs go to `results/`:
- `REPORT.md`
- `candidate_summary.csv`
- `walk_forward.csv`
- `cost_stress_2x.csv`
- `yearly.csv`
- `trades.csv`
- `events.csv`
- `equity.csv`
- `summary.json`

## Live deployment gate

Do **not** use live capital merely because the in-sample result looks good.
Minimum next gates:
1. Walk-forward OOS survives.
2. 2x fees/slippage survives.
3. Futures funding-aware validation survives.
4. Parameters remain stable around nearby values.
5. At least 4 weeks of Bybit paper trading.
6. Then capital scale-up in stages rather than all-in deployment.
