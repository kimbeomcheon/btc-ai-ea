#!/usr/bin/env python3
"""
BTC AI EA — V7.2 Volatility-Budget Backtester
==============================================

Why V7.2 exists
-------------
V5.1 produced the strongest return engine so far, while V6 showed that cutting
risk only after account drawdown is already visible sacrifices too much upside.
V7.1 proved that the return engine has value, but its best candidate still
carried a roughly 27% shadow drawdown. V7.2 keeps the V7.1D return/risk engine
fixed and adds a separate volatility-budget layer that caps exposure during
realized-volatility expansion, including while the broader trend remains bullish.

Architecture
------------
1) CORE: slow daily BTC bull-trend exposure inherited from V5.1.
2) TACTICAL: 4H breakout sleeve, long only.
3) PROACTIVE MARKET RISK: V7.1 trend/momentum/shock state machine.
4) VOLATILITY BUDGET: independent NORMAL/WARN/HIGH state from absolute 30-day
   realized volatility plus volatility expansion versus its 90-day median.
   This layer caps both core and tactical exposure even in a bullish regime.
5) HYSTERESIS: market risk and volatility budget escalate immediately but
   exposure is restored only after completed daily bars confirm recovery.
6) CIRCUIT BREAKERS: daily/weekly loss locks and a separate 15% terminal
   research gate. The 15% gate does not drive normal position sizing.

Risk states
-----------
NORMAL -> CAUTION -> DEFENSIVE -> PANIC

The design objective is to keep at least ~20% research CAGR if possible while
compressing shadow MDD toward 15%, without using account drawdown as an ordinary
position-sizing input. No account high-water-mark
throttle is used in the ordinary shadow run.

Anti-lookahead rules
--------------------
- Daily risk/regime features use only a fully completed prior daily candle.
- 4H tactical signals use only a fully completed prior 4H candle.
- Rebalances execute at a later 4H OPEN.

Phase-1 price source
--------------------
Binance BTCUSDT spot 1H public archives are used as a long-history PRICE proxy.
Funding, basis, liquidation and perpetual-specific microstructure remain deferred
to later futures validation.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BINANCE_VISION = "https://data.binance.vision/data/spot"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


@dataclass(frozen=True)
class Strategy:
    name: str
    fast_days: int
    slow_days: int
    slope_days: int
    strong_long: float
    weak_long: float
    tactical_long: float
    breakout_4h: int
    exit_4h: int
    trail_atr_4h: float
    breakout_buffer_atr: float
    vol_target: float
    vol_floor_scale: float
    max_long: float
    budget_rv_warn: float
    budget_rv_high: float
    budget_ratio_warn: float
    budget_ratio_high: float
    budget_scale_warn: float
    budget_scale_high: float
    budget_recovery_days: int
    risk_fast_days: int
    risk_mid_days: int
    risk_slow_days: int
    mom5_cut: float
    mom20_cut: float
    high20_cut: float
    rv_ratio_cut: float
    caution_scale: float
    defense_scale: float
    panic_scale: float
    recovery_days: int


CANDIDATES = [
    # V7.1D_STICKY engine is intentionally frozen. Only volatility-budget
    # thresholds/scales vary so the V7.2 experiment isolates the new layer.
    Strategy(
        "V72A_MILD",
        fast_days=100, slow_days=250, slope_days=30,
        strong_long=0.72, weak_long=0.36, tactical_long=0.26,
        breakout_4h=40, exit_4h=20, trail_atr_4h=4.5,
        breakout_buffer_atr=0.08, vol_target=0.55, vol_floor_scale=0.55,
        max_long=0.98,
        budget_rv_warn=0.85, budget_rv_high=1.10,
        budget_ratio_warn=1.35, budget_ratio_high=1.75,
        budget_scale_warn=0.90, budget_scale_high=0.68, budget_recovery_days=2,
        risk_fast_days=20, risk_mid_days=50, risk_slow_days=200,
        mom5_cut=-0.10, mom20_cut=-0.17, high20_cut=-0.15, rv_ratio_cut=1.45,
        caution_scale=0.68, defense_scale=0.32, panic_scale=0.00, recovery_days=5,
    ),
    Strategy(
        "V72B_BALANCED",
        fast_days=100, slow_days=250, slope_days=30,
        strong_long=0.72, weak_long=0.36, tactical_long=0.26,
        breakout_4h=40, exit_4h=20, trail_atr_4h=4.5,
        breakout_buffer_atr=0.08, vol_target=0.55, vol_floor_scale=0.55,
        max_long=0.98,
        budget_rv_warn=0.75, budget_rv_high=1.00,
        budget_ratio_warn=1.25, budget_ratio_high=1.60,
        budget_scale_warn=0.82, budget_scale_high=0.55, budget_recovery_days=3,
        risk_fast_days=20, risk_mid_days=50, risk_slow_days=200,
        mom5_cut=-0.10, mom20_cut=-0.17, high20_cut=-0.15, rv_ratio_cut=1.45,
        caution_scale=0.68, defense_scale=0.32, panic_scale=0.00, recovery_days=5,
    ),
    Strategy(
        "V72C_DEFENSIVE",
        fast_days=100, slow_days=250, slope_days=30,
        strong_long=0.72, weak_long=0.36, tactical_long=0.26,
        breakout_4h=40, exit_4h=20, trail_atr_4h=4.5,
        breakout_buffer_atr=0.08, vol_target=0.55, vol_floor_scale=0.55,
        max_long=0.98,
        budget_rv_warn=0.70, budget_rv_high=0.92,
        budget_ratio_warn=1.20, budget_ratio_high=1.50,
        budget_scale_warn=0.76, budget_scale_high=0.45, budget_recovery_days=3,
        risk_fast_days=20, risk_mid_days=50, risk_slow_days=200,
        mom5_cut=-0.10, mom20_cut=-0.17, high20_cut=-0.15, rv_ratio_cut=1.45,
        caution_scale=0.68, defense_scale=0.32, panic_scale=0.00, recovery_days=5,
    ),
    Strategy(
        "V72D_STICKY_BUDGET",
        fast_days=100, slow_days=250, slope_days=30,
        strong_long=0.72, weak_long=0.36, tactical_long=0.26,
        breakout_4h=40, exit_4h=20, trail_atr_4h=4.5,
        breakout_buffer_atr=0.08, vol_target=0.55, vol_floor_scale=0.55,
        max_long=0.98,
        budget_rv_warn=0.78, budget_rv_high=1.00,
        budget_ratio_warn=1.22, budget_ratio_high=1.55,
        budget_scale_warn=0.84, budget_scale_high=0.50, budget_recovery_days=5,
        risk_fast_days=20, risk_mid_days=50, risk_slow_days=200,
        mom5_cut=-0.10, mom20_cut=-0.17, high20_cut=-0.15, rv_ratio_cut=1.45,
        caution_scale=0.68, defense_scale=0.32, panic_scale=0.00, recovery_days=5,
    ),
]


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 5.5
    slippage_bps: float = 2.0


@dataclass(frozen=True)
class RiskRules:
    daily_loss_lock: float = 0.02
    weekly_loss_lock: float = 0.05
    soft_drawdown: float = 0.10
    hard_drawdown: float = 0.15


def _fetch_bytes(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "btc-ai-ea-v7.2/1.0"})
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except Exception as e:
            last = e
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Download failed: {url}: {last}")


def _parse_kline_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if df.shape[1] < 6:
        raise ValueError("Unexpected Binance kline archive")
    df = df.iloc[:, :min(12, df.shape[1])]
    df.columns = KLINE_COLS[:df.shape[1]]
    t = pd.to_numeric(df["open_time"], errors="coerce")
    unit = "us" if t.dropna().median() > 1e14 else "ms"
    idx = pd.to_datetime(t, unit=unit, utc=True, errors="coerce")
    out = pd.DataFrame(index=idx)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    return out.dropna().sort_index()


def _month_starts(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    last = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= last:
        yield cur
        cur += pd.offsets.MonthBegin(1)


def download_btc_1h(start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    current_month = pd.Timestamp(now.year, now.month, 1, tz="UTC")
    last_complete_month_end = current_month - pd.Timedelta(hours=1)
    frames = []

    monthly_end = min(end_ts, last_complete_month_end)
    for m in _month_starts(start_ts, monthly_end):
        ym = m.strftime("%Y-%m")
        zpath = cache_dir / f"BTCUSDT-1h-{ym}.zip"
        url = f"{BINANCE_VISION}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{ym}.zip"
        if not zpath.exists():
            try:
                zpath.write_bytes(_fetch_bytes(url))
                print(f"[DATA] {ym}", flush=True)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                raise
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            frames.append(_parse_kline_csv(z.read(name)))

    # Current month from completed daily archives.
    if end_ts >= current_month:
        d0 = max(start_ts.normalize(), current_month)
        d1 = min(end_ts.normalize(), (now - pd.Timedelta(days=1)).normalize())
        if d1 >= d0:
            for d in pd.date_range(d0, d1, freq="1D", tz="UTC"):
                ds = d.strftime("%Y-%m-%d")
                zpath = cache_dir / f"BTCUSDT-1h-{ds}.zip"
                url = f"{BINANCE_VISION}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{ds}.zip"
                if not zpath.exists():
                    try:
                        zpath.write_bytes(_fetch_bytes(url))
                    except urllib.error.HTTPError as e:
                        if e.code == 404:
                            continue
                        raise
                with zipfile.ZipFile(zpath) as z:
                    name = next(n for n in z.namelist() if n.endswith(".csv"))
                    frames.append(_parse_kline_csv(z.read(name)))

    if not frames:
        raise RuntimeError("No BTC data available")
    x = pd.concat(frames).sort_index()
    x = x[~x.index.duplicated(keep="last")]
    x = x.loc[(x.index >= start_ts) & (x.index <= end_ts)]
    bad = ((x["high"] < x[["open", "close", "low"]].max(axis=1)) |
           (x["low"] > x[["open", "close", "high"]].min(axis=1)))
    if bad.any():
        raise RuntimeError(f"OHLC integrity failure: {int(bad.sum())} rows")
    print(f"[DATA] rows={len(x):,} {x.index.min()} -> {x.index.max()}")
    return x


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return wilder(tr, n)


def build_features(df1h: pd.DataFrame, s: Strategy) -> pd.DataFrame:
    # 4H bars are labeled at bar OPEN. The bar's signal can only be acted upon
    # at a later 4H open.
    h4 = df1h[["open", "high", "low", "close", "volume"]].resample(
        "4h", closed="left", label="left"
    ).agg({"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"}).dropna()
    h4["atr4h"] = atr(h4, 14)
    h4["entry_hi"] = h4["high"].shift(1).rolling(s.breakout_4h).max()
    h4["entry_lo"] = h4["low"].shift(1).rolling(s.breakout_4h).min()
    h4["exit_hi"] = h4["high"].shift(1).rolling(s.exit_4h).max()
    h4["exit_lo"] = h4["low"].shift(1).rolling(s.exit_4h).min()

    # Daily bar is labeled at NEXT midnight. Therefore these features are fully
    # known at that timestamp before merge_asof propagates them to later 4H bars.
    d = df1h[["open", "high", "low", "close", "volume"]].resample(
        "1D", closed="left", label="right"
    ).agg({"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"}).dropna()

    # Slow return-engine regime.
    d["fast"] = d["close"].ewm(
        span=s.fast_days, adjust=False, min_periods=s.fast_days
    ).mean()
    d["slow"] = d["close"].ewm(
        span=s.slow_days, adjust=False, min_periods=s.slow_days
    ).mean()
    d["slow_slope"] = d["slow"] - d["slow"].shift(s.slope_days)

    strong_bull = (
        (d["close"] > d["slow"]) &
        (d["fast"] > d["slow"]) &
        (d["slow_slope"] > 0)
    )
    weak_bull = (d["close"] > d["slow"]) & ~strong_bull
    d["regime"] = "NEUTRAL"
    d.loc[weak_bull, "regime"] = "BULL_WEAK"
    d.loc[strong_bull, "regime"] = "BULL_STRONG"

    # Faster V7 proactive risk sensors.
    d["risk_fast"] = d["close"].ewm(
        span=s.risk_fast_days, adjust=False, min_periods=s.risk_fast_days
    ).mean()
    d["risk_mid"] = d["close"].ewm(
        span=s.risk_mid_days, adjust=False, min_periods=s.risk_mid_days
    ).mean()
    d["risk_slow"] = d["close"].ewm(
        span=s.risk_slow_days, adjust=False, min_periods=s.risk_slow_days
    ).mean()
    ret = d["close"].pct_change()
    d["ret5"] = d["close"].pct_change(5)
    d["ret20"] = d["close"].pct_change(20)
    d["high20"] = d["close"].shift(1).rolling(20, min_periods=10).max()
    d["dd20"] = d["close"] / d["high20"] - 1.0
    d["rv30"] = ret.rolling(30, min_periods=20).std() * np.sqrt(365)
    d["rv90_med"] = d["rv30"].rolling(90, min_periods=45).median()
    d["rv_ratio"] = d["rv30"] / d["rv90_med"].replace(0, np.nan)
    d["atr_pct"] = atr(d, 14) / d["close"]
    d["shock_cut"] = d["atr_pct"].rolling(540, min_periods=180).quantile(0.95)
    d["shock"] = (
        (d["atr_pct"] > d["shock_cut"]) |
        (ret.abs() > 3.0 * ret.rolling(60, min_periods=30).std())
    ).fillna(False)
    d["dclose"] = d["close"]
    d["daily_seq"] = np.arange(len(d), dtype=int)

    left = h4.reset_index().rename(columns={h4.index.name or "index":"time"})
    right = d[[
        "regime", "rv30", "rv_ratio", "shock", "fast", "slow", "slow_slope",
        "risk_fast", "risk_mid", "risk_slow", "ret5", "ret20", "dd20", "dclose", "daily_seq"
    ]].reset_index()
    right = right.rename(columns={right.columns[0]:"time"})
    x = pd.merge_asof(
        left.sort_values("time"), right.sort_values("time"),
        on="time", direction="backward"
    )
    return x.set_index("time")


def week_key(ts: pd.Timestamp):
    i = ts.isocalendar()
    return (int(i.year), int(i.week))


def core_exposure(regime: str, s: Strategy) -> float:
    if regime == "BULL_STRONG":
        return s.strong_long
    if regime == "BULL_WEAK":
        return s.weak_long
    return 0.0


def vol_scale(rv: float, s: Strategy) -> float:
    if not np.isfinite(rv) or rv <= 0:
        return 1.0
    return float(np.clip(s.vol_target / rv, s.vol_floor_scale, 1.0))


def raw_market_risk_state(dclose: float, risk_fast: float, risk_mid: float,
                          risk_slow: float, ret5: float, ret20: float,
                          dd20: float, rv_ratio: float, shock: bool,
                          s: Strategy):
    """Immediate market-risk classification from completed daily data only."""
    warnings = 0
    if np.isfinite(dclose) and np.isfinite(risk_fast) and dclose < risk_fast:
        warnings += 1
    if np.isfinite(dclose) and np.isfinite(risk_mid) and dclose < risk_mid:
        warnings += 1
    if np.isfinite(dclose) and np.isfinite(risk_slow) and dclose < risk_slow:
        warnings += 1
    if np.isfinite(risk_fast) and np.isfinite(risk_mid) and risk_fast < risk_mid:
        warnings += 1
    if np.isfinite(risk_mid) and np.isfinite(risk_slow) and risk_mid < risk_slow:
        warnings += 1
    if np.isfinite(ret5) and ret5 < -0.05:
        warnings += 1
    if np.isfinite(ret20) and ret20 < -0.10:
        warnings += 1
    if np.isfinite(rv_ratio) and rv_ratio > s.rv_ratio_cut:
        warnings += 1

    panic = (
        (np.isfinite(ret5) and ret5 <= s.mom5_cut) or
        (np.isfinite(ret20) and ret20 <= s.mom20_cut) or
        (np.isfinite(dd20) and dd20 <= s.high20_cut and
         np.isfinite(dclose) and np.isfinite(risk_mid) and dclose < risk_mid) or
        (shock and np.isfinite(ret5) and ret5 < -0.035)
    )
    if panic:
        return "PANIC", warnings

    defensive = (
        warnings >= 4 or
        (np.isfinite(dclose) and np.isfinite(risk_mid) and dclose < risk_mid and
         np.isfinite(ret20) and ret20 < -0.07)
    )
    if defensive:
        return "DEFENSIVE", warnings
    if warnings >= 1:
        return "CAUTION", warnings
    return "NORMAL", warnings


STATE_LEVEL = {"NORMAL":0, "CAUTION":1, "DEFENSIVE":2, "PANIC":3}
LEVEL_STATE = {v:k for k,v in STATE_LEVEL.items()}


def hysteresis_update(current: str | None, raw: str, recovery_count: int,
                      recovery_days: int):
    """Escalate immediately; recover only after consecutive completed days."""
    if current is None:
        return raw, 0
    c = STATE_LEVEL[current]; r = STATE_LEVEL[raw]
    if r >= c:
        # Worsening is immediate; equal state clears any stale recovery count.
        return raw, 0
    recovery_count += 1
    if recovery_count < recovery_days:
        return current, recovery_count
    # Step down one risk level at a time; this prevents snap-back to full risk.
    new_level = max(r, c - 1)
    return LEVEL_STATE[new_level], 0


BUDGET_LEVEL = {"NORMAL":0, "WARN":1, "HIGH":2}
BUDGET_STATE = {v:k for k,v in BUDGET_LEVEL.items()}


def raw_vol_budget_state(rv: float, rv_ratio: float, s: Strategy) -> str:
    """Independent volatility-budget severity from completed daily data."""
    high = (
        (np.isfinite(rv) and rv >= s.budget_rv_high) or
        (np.isfinite(rv_ratio) and rv_ratio >= s.budget_ratio_high)
    )
    if high:
        return "HIGH"
    warn = (
        (np.isfinite(rv) and rv >= s.budget_rv_warn) or
        (np.isfinite(rv_ratio) and rv_ratio >= s.budget_ratio_warn)
    )
    return "WARN" if warn else "NORMAL"


def budget_hysteresis_update(current: str | None, raw: str, recovery_count: int,
                             recovery_days: int):
    if current is None:
        return raw, 0
    c = BUDGET_LEVEL[current]; r = BUDGET_LEVEL[raw]
    if r >= c:
        return raw, 0
    recovery_count += 1
    if recovery_count < recovery_days:
        return current, recovery_count
    return BUDGET_STATE[max(r, c - 1)], 0


def volatility_budget_scale(state: str, s: Strategy) -> float:
    if state == "WARN":
        return s.budget_scale_warn
    if state == "HIGH":
        return s.budget_scale_high
    return 1.0


def market_scale_for_state(state: str, s: Strategy) -> float:
    if state == "CAUTION":
        return s.caution_scale
    if state == "DEFENSIVE":
        return s.defense_scale
    if state == "PANIC":
        return s.panic_scale
    return 1.0


def target_exposure(regime: str, rv: float, rv_ratio: float, tactical_side: int,
                    s: Strategy, market_state: str, budget_state: str,
                    market_risk_enabled: bool = True,
                    vol_budget_enabled: bool = True):
    vscale = vol_scale(rv, s)
    core = core_exposure(regime, s) * vscale
    tactical = 0.0
    if tactical_side > 0 and regime in ("BULL_STRONG", "BULL_WEAK"):
        if market_state in ("NORMAL", "CAUTION") and budget_state != "HIGH":
            tactical = s.tactical_long * vscale
    mscale = market_scale_for_state(market_state, s) if market_risk_enabled else 1.0
    bscale = volatility_budget_scale(budget_state, s) if vol_budget_enabled else 1.0
    exp = (core + tactical) * mscale * bscale
    return float(np.clip(exp, 0.0, s.max_long)), mscale, bscale


def trade_cost(delta_notional: float, costs: CostModel) -> float:
    # Fee plus an explicit slippage penalty. Both are charged on every rebalance.
    return abs(delta_notional) * (costs.fee_bps + costs.slippage_bps) / 10_000.0


def backtest(df1h: pd.DataFrame, s: Strategy, costs: CostModel, rules: RiskRules,
             initial: float = 10_000.0, signal_delay_bars: int = 1,
             tactical_enabled: bool = True, enforce_hard_stop: bool = True,
             market_risk_enabled: bool = True, hysteresis_enabled: bool = True,
             vol_budget_enabled: bool = True,
             trade_start: pd.Timestamp | None = None):
    x = build_features(df1h, s).dropna(
        subset=["rv30", "atr4h", "regime", "risk_fast", "risk_mid",
                "risk_slow", "ret20", "dd20"]
    )
    # Walk-forward OOS windows need pre-test history to warm slow daily
    # indicators, but portfolio PnL/risk state must start fresh at the OOS
    # boundary. Build features on the warm-up slice, then discard all bars
    # before trade_start. This fixes the V7 one-year OOS failure where a
    # 250-day slow regime left fewer than 1,500 usable 4H bars.
    if trade_start is not None:
        t0 = pd.Timestamp(trade_start)
        if t0.tzinfo is None:
            t0 = t0.tz_localize("UTC")
        else:
            t0 = t0.tz_convert("UTC")
        x = x.loc[x.index >= t0]
    if len(x) < 1500:
        raise RuntimeError("Insufficient usable 4H history")

    equity = initial
    qty = 0.0
    prev_close = None
    peak_equity = initial

    current_day = x.index[0].date()
    current_week = week_key(x.index[0])
    day_start = initial
    week_start = initial
    day_lock = False
    week_lock = False

    hard_stopped = False
    hard_breached = False
    hard_breach_count = 0
    first_hard_breach_time = None
    was_below_hard = False

    tactical_side = 0
    tactical_peak = np.nan
    pending_tactical = None

    total_costs = 0.0
    rebalance_count = 0
    bars_exposed = 0
    rows = []
    trades = []
    events = []
    last_exposure = 0.0
    last_regime = None
    last_risk_state = None
    last_budget_state = None

    # Hysteresis updates only when a NEW completed daily bar becomes available.
    effective_risk_state = None
    recovery_count = 0
    last_daily_seq = None
    last_raw_state = "NORMAL"
    last_warning_count = 0
    effective_budget_state = None
    budget_recovery_count = 0
    last_raw_budget_state = "NORMAL"

    def set_position(ts, px, desired_exp, reason):
        nonlocal equity, qty, total_costs, rebalance_count, last_exposure
        desired_qty = equity * desired_exp / px if px > 0 else 0.0
        delta = desired_qty - qty
        notional = delta * px
        cost = trade_cost(notional, costs)
        equity -= cost
        total_costs += cost
        if abs(delta) > 1e-12:
            rebalance_count += 1
            events.append({
                "time":ts, "event":"REBALANCE", "reason":reason,
                "price":px, "delta_qty":delta, "cost":cost,
                "target_exposure":desired_exp,
            })
        qty = desired_qty
        last_exposure = desired_exp

    for ts, r in x.iterrows():
        o, h, l, c, a = map(float, [r.open, r.high, r.low, r.close, r.atr4h])
        regime = str(r.regime)
        rv = float(r.rv30)
        rv_ratio = float(r.rv_ratio) if np.isfinite(r.rv_ratio) else 1.0
        shock = bool(r.shock)
        dclose = float(r.dclose)
        rfast = float(r.risk_fast)
        rmid = float(r.risk_mid)
        rslow = float(r.risk_slow)
        ret5 = float(r.ret5) if np.isfinite(r.ret5) else 0.0
        ret20 = float(r.ret20) if np.isfinite(r.ret20) else 0.0
        dd20 = float(r.dd20) if np.isfinite(r.dd20) else 0.0
        daily_seq = int(r.daily_seq)

        if prev_close is not None:
            equity += qty * (o - prev_close)

        if ts.date() != current_day:
            current_day = ts.date(); day_start = equity; day_lock = False
        w = week_key(ts)
        if w != current_week:
            current_week = w; week_start = equity; week_lock = False

        peak_equity = max(peak_equity, equity)
        dd_open = equity / peak_equity - 1.0

        if daily_seq != last_daily_seq:
            raw_state, warning_count = raw_market_risk_state(
                dclose, rfast, rmid, rslow, ret5, ret20, dd20, rv_ratio, shock, s
            )
            last_raw_state = raw_state
            last_warning_count = warning_count
            if hysteresis_enabled:
                effective_risk_state, recovery_count = hysteresis_update(
                    effective_risk_state, raw_state, recovery_count, s.recovery_days
                )
            else:
                effective_risk_state = raw_state
                recovery_count = 0

            raw_budget = raw_vol_budget_state(rv, rv_ratio, s)
            last_raw_budget_state = raw_budget
            if hysteresis_enabled:
                effective_budget_state, budget_recovery_count = budget_hysteresis_update(
                    effective_budget_state, raw_budget, budget_recovery_count,
                    s.budget_recovery_days
                )
            else:
                effective_budget_state = raw_budget
                budget_recovery_count = 0
            last_daily_seq = daily_seq

        risk_state = effective_risk_state or last_raw_state
        budget_state = effective_budget_state or last_raw_budget_state
        warning_count = last_warning_count

        if pending_tactical is not None:
            action, pside, remaining = pending_tactical
            remaining -= 1
            if remaining <= 0:
                if action == "ENTER":
                    if (regime in ("BULL_STRONG", "BULL_WEAK") and
                            risk_state in ("NORMAL", "CAUTION") and
                            budget_state != "HIGH" and not hard_stopped):
                        tactical_side = pside
                        tactical_peak = o
                else:
                    tactical_side = 0
                    tactical_peak = np.nan
                pending_tactical = None
            else:
                pending_tactical = (action, pside, remaining)

        desired = 0.0
        reason = "MODEL"
        mscale = 0.0
        bscale = 0.0
        if hard_stopped or day_lock or week_lock:
            reason = "LOCK"
        else:
            desired, mscale, bscale = target_exposure(
                regime, rv, rv_ratio, tactical_side if tactical_enabled else 0, s,
                risk_state, budget_state,
                market_risk_enabled=market_risk_enabled,
                vol_budget_enabled=vol_budget_enabled,
            )

        if (abs(desired - last_exposure) >= 0.025 or
                regime != last_regime or risk_state != last_risk_state or
                budget_state != last_budget_state or
                (desired == 0 and abs(last_exposure) > 1e-12)):
            set_position(ts, o, desired, reason)
        last_regime = regime
        last_risk_state = risk_state
        last_budget_state = budget_state

        if abs(qty) > 0:
            bars_exposed += 1

        equity += qty * (c - o)
        peak_equity = max(peak_equity, equity)
        dd = equity / peak_equity - 1.0

        if day_start > 0 and equity / day_start - 1 <= -rules.daily_loss_lock:
            day_lock = True
        if week_start > 0 and equity / week_start - 1 <= -rules.weekly_loss_lock:
            week_lock = True

        below_hard = dd <= -rules.hard_drawdown
        if below_hard and not was_below_hard:
            hard_breached = True
            hard_breach_count += 1
            if first_hard_breach_time is None:
                first_hard_breach_time = ts
            if enforce_hard_stop:
                hard_stopped = True
        was_below_hard = below_hard

        if tactical_enabled and not hard_stopped:
            buf = s.breakout_buffer_atr * a
            if (risk_state in ("DEFENSIVE", "PANIC") or budget_state == "HIGH") and tactical_side > 0:
                if pending_tactical is None:
                    pending_tactical = ("EXIT", +1, max(1, signal_delay_bars))
            elif tactical_side == 0 and pending_tactical is None:
                if (regime in ("BULL_STRONG", "BULL_WEAK") and
                        risk_state in ("NORMAL", "CAUTION") and
                        budget_state != "HIGH" and
                        c > float(r.entry_hi) + buf):
                    pending_tactical = ("ENTER", +1, max(1, signal_delay_bars))
            elif tactical_side > 0:
                tactical_peak = max(tactical_peak, h) if np.isfinite(tactical_peak) else h
                trail = tactical_peak - s.trail_atr_4h * a
                if (regime not in ("BULL_STRONG", "BULL_WEAK") or
                        c < float(r.exit_lo) or c < trail):
                    if pending_tactical is None:
                        pending_tactical = ("EXIT", +1, max(1, signal_delay_bars))

        rows.append({
            "time":ts, "equity":equity, "qty":qty,
            "target_exposure":last_exposure, "regime":regime,
            "raw_risk_state":last_raw_state, "risk_state":risk_state,
            "risk_warning_count":warning_count, "market_scale":mscale,
            "raw_budget_state":last_raw_budget_state, "budget_state":budget_state,
            "vol_budget_scale":bscale, "budget_recovery_count":budget_recovery_count,
            "rv30":rv, "rv_ratio":rv_ratio,
            "recovery_count":recovery_count, "drawdown":dd,
            "tactical_side":tactical_side, "day_lock":day_lock, "week_lock":week_lock,
        })
        prev_close = c

    eqdf = pd.DataFrame(rows).set_index("time")
    if abs(qty) > 0 and not eqdf.empty:
        px = float(x.iloc[-1].close)
        cost = trade_cost(qty * px, costs)
        equity -= cost
        total_costs += cost
        eqdf.iloc[-1, eqdf.columns.get_loc("equity")] = equity
        eqdf.iloc[-1, eqdf.columns.get_loc("qty")] = 0.0
        eqdf.iloc[-1, eqdf.columns.get_loc("target_exposure")] = 0.0

    active = eqdf.target_exposure.abs() > 1e-9
    starts = active & ~active.shift(1, fill_value=False)
    ends = active & ~active.shift(-1, fill_value=False)
    for st, en in zip(list(eqdf.index[starts]), list(eqdf.index[ends])):
        g = eqdf.loc[st:en]
        p0 = float(g.equity.iloc[0]); p1 = float(g.equity.iloc[-1])
        trades.append({
            "entry_time":st, "exit_time":en, "direction":"LONG",
            "avg_exposure":float(g.target_exposure.mean()),
            "return":p1/p0 - 1 if p0 > 0 else np.nan,
        })

    trdf = pd.DataFrame(trades)
    evdf = pd.DataFrame(events)
    extra = {
        "costs":total_costs, "hard_stopped":hard_stopped,
        "hard_breached":hard_breached, "hard_breach_count":hard_breach_count,
        "first_hard_breach_time":(
            str(first_hard_breach_time) if first_hard_breach_time is not None else None
        ),
        "bars_exposed":bars_exposed, "bars_total":len(eqdf),
        "rebalances":rebalance_count,
    }
    return eqdf, trdf, evdf, extra


def metrics(eq: pd.DataFrame, trades: pd.DataFrame, extra: dict, initial: float):
    if eq.empty:
        return {}
    final = float(eq.equity.iloc[-1])
    days = max((eq.index[-1] - eq.index[0]).total_seconds()/86400, 1)
    years = days / 365.2425
    cagr = (final/initial)**(1/years)-1 if final > 0 else -1.0
    dd = eq.equity/eq.equity.cummax()-1
    mdd = float(dd.min())
    daily = eq.equity.resample("1D").last().dropna().pct_change().dropna()
    std = daily.std()
    sharpe = float(np.sqrt(365)*daily.mean()/std) if std and std > 0 else np.nan
    downside = daily[daily < 0].std()
    sortino = float(np.sqrt(365)*daily.mean()/downside) if downside and downside > 0 else np.nan
    calmar = cagr/abs(mdd) if mdd < 0 else np.nan
    # For continuous-exposure portfolios PF is computed from daily PnL, not campaign PnL.
    dpnl = eq.equity.resample("1D").last().dropna().diff().dropna()
    gp = float(dpnl[dpnl > 0].sum())
    gl = float(-dpnl[dpnl < 0].sum())
    pf = gp/gl if gl > 0 else np.inf
    return {
        "start":str(eq.index[0]), "end":str(eq.index[-1]),
        "initial_usd":initial, "final_usd":final,
        "total_return":final/initial-1, "cagr":cagr,
        "max_drawdown":mdd, "sharpe_365":sharpe,
        "sortino_365":sortino, "calmar":calmar,
        "profit_factor_daily":pf,
        "campaigns":int(len(trades)),
        "rebalances":int(extra["rebalances"]),
        "costs_paid_usd":float(extra["costs"]),
        "market_exposure":extra["bars_exposed"]/max(extra["bars_total"],1),
        "hard_stopped":bool(extra["hard_stopped"]),
        "hard_breached":bool(extra.get("hard_breached", extra["hard_stopped"])),
        "hard_breach_count":int(extra.get("hard_breach_count", int(extra["hard_stopped"]))),
        "first_hard_breach_time":extra.get("first_hard_breach_time"),
    }


def objective(m):
    """Risk-first research ranking; no equal-score cliff when the 15% gate fails."""
    if not m:
        return -1e9
    sh = m.get("sharpe_365", -1)
    if not np.isfinite(sh):
        sh = -1
    pf = m.get("profit_factor_daily", 0.0)
    if not np.isfinite(pf):
        pf = 5.0
    mdd = abs(m["max_drawdown"])
    excess = max(0.0, mdd - 0.15)
    return (
        m["cagr"]
        - 1.55 * mdd
        + 0.08 * sh
        + 0.02 * min(pf, 5.0)
        - 7.0 * excess
        - 0.10 * int(m.get("hard_breached", False))
    )


def run_candidates(data, costs, rules, initial, delay=1):
    rows=[]; outputs={}
    for s in CANDIDATES:
        print(f"[TEST] {s.name}", flush=True)
        seq,str_,sev,sex = backtest(
            data,s,costs,rules,initial,delay,enforce_hard_stop=False
        )
        sm = metrics(seq,str_,sex,initial)
        req,rtr,rev,rex = backtest(
            data,s,costs,rules,initial,delay,enforce_hard_stop=True
        )
        rm = metrics(req,rtr,rex,initial)
        row = {
            "strategy":s.name,
            **{f"shadow_{k}":v for k,v in sm.items()},
            "risk_final_usd":rm["final_usd"],
            "risk_total_return":rm["total_return"],
            "risk_cagr":rm["cagr"],
            "risk_mdd":rm["max_drawdown"],
            "risk_sharpe":rm["sharpe_365"],
            "risk_pf_daily":rm["profit_factor_daily"],
            "risk_hard_stopped":rm["hard_stopped"],
            "mdd_gate":bool(abs(sm["max_drawdown"]) <= rules.hard_drawdown),
            "score":objective(sm),
        }
        rows.append(row)
        outputs[s.name] = {
            "shadow":(seq,str_,sev,sm),
            "risk":(req,rtr,rev,rm),
        }
    return pd.DataFrame(rows).sort_values(["mdd_gate","score"],ascending=[False,False]), outputs


def walk_forward(data, costs, rules, initial):
    start=data.index.min().normalize(); end=data.index.max().normalize()
    rows=[]; anchor=start
    while anchor + pd.DateOffset(years=4) <= end + pd.Timedelta(days=1):
        train_end=anchor+pd.DateOffset(years=3)-pd.Timedelta(hours=1)
        test_start=train_end+pd.Timedelta(hours=1)
        test_end=anchor+pd.DateOffset(years=4)-pd.Timedelta(hours=1)
        train=data.loc[(data.index>=anchor)&(data.index<=train_end)]
        test=data.loc[(data.index>=test_start)&(data.index<=test_end)]
        # Keep a 600-day pre-test warm-up solely for feature construction.
        # backtest(..., trade_start=test_start) resets account state at the
        # OOS boundary, so no warm-up PnL leaks into the test metrics.
        warmup_start = max(data.index.min(), test_start - pd.Timedelta(days=600))
        test_warm = data.loc[(data.index>=warmup_start)&(data.index<=test_end)]
        if len(train)<15000 or len(test)<4000:
            anchor += pd.DateOffset(years=1); continue

        scored=[]
        for s in CANDIDATES:
            eq,tr,ev,ex=backtest(train,s,costs,rules,initial,enforce_hard_stop=False)
            m=metrics(eq,tr,ex,initial)
            scored.append((abs(m["max_drawdown"]) <= rules.hard_drawdown, objective(m), s, m))
        scored.sort(key=lambda z:(z[0], z[1]), reverse=True)
        _, _, chosen, tm = scored[0]

        seq,str_,sev,sex=backtest(
            test_warm, chosen, costs, rules, initial,
            enforce_hard_stop=False, trade_start=test_start
        )
        sm=metrics(seq,str_,sex,initial)
        req,rtr,rev,rex=backtest(
            test_warm, chosen, costs, rules, initial,
            enforce_hard_stop=True, trade_start=test_start
        )
        rm=metrics(req,rtr,rex,initial)

        rows.append({
            "train_start":anchor,"train_end":train_end,
            "test_start":test_start,"test_end":test_end,
            "chosen":chosen.name,
            "train_shadow_cagr":tm["cagr"],
            "train_shadow_mdd":tm["max_drawdown"],
            "train_shadow_score":objective(tm),
            "test_shadow_return":sm["total_return"],
            "test_shadow_cagr":sm["cagr"],
            "test_shadow_mdd":sm["max_drawdown"],
            "test_shadow_sharpe":sm["sharpe_365"],
            "test_shadow_pf_daily":sm["profit_factor_daily"],
            "test_shadow_hard_breached":sm["hard_breached"],
            "test_risk_return":rm["total_return"],
            "test_risk_cagr":rm["cagr"],
            "test_risk_mdd":rm["max_drawdown"],
            "test_risk_hard_stopped":rm["hard_stopped"],
        })
        anchor += pd.DateOffset(years=1)
    return pd.DataFrame(rows)


def robustness_grid(data, best: Strategy, costs, rules, initial):
    rows=[]
    for budget_mult in (0.8,1.0,1.2):
        for exp_mult in (0.8,1.0,1.2):
            # Scale absolute RV thresholds directly; scale only the excess over
            # 1.0 for RV-ratio thresholds so the neighborhood remains sensible.
            s=replace(
                best,
                name=f"{best.name}_B{budget_mult:.1f}_E{exp_mult:.1f}",
                budget_rv_warn=max(0.35, best.budget_rv_warn*budget_mult),
                budget_rv_high=max(0.50, best.budget_rv_high*budget_mult),
                budget_ratio_warn=1.0+(best.budget_ratio_warn-1.0)*budget_mult,
                budget_ratio_high=1.0+(best.budget_ratio_high-1.0)*budget_mult,
                strong_long=min(1.15, best.strong_long*exp_mult),
                weak_long=min(0.70, best.weak_long*exp_mult),
                tactical_long=min(0.40, best.tactical_long*exp_mult),
                max_long=min(1.25, best.max_long*exp_mult),
            )
            seq,str_,sev,sex=backtest(data,s,costs,rules,initial,enforce_hard_stop=False)
            sm=metrics(seq,str_,sex,initial)
            req,rtr,rev,rex=backtest(data,s,costs,rules,initial,enforce_hard_stop=True)
            rm=metrics(req,rtr,rex,initial)
            rows.append({
                "budget_threshold_mult":budget_mult,"exposure_mult":exp_mult,
                "budget_rv_warn":s.budget_rv_warn,"budget_rv_high":s.budget_rv_high,
                "budget_ratio_warn":s.budget_ratio_warn,"budget_ratio_high":s.budget_ratio_high,
                "max_long":s.max_long,
                "shadow_cagr":sm["cagr"],"shadow_mdd":sm["max_drawdown"],
                "shadow_sharpe":sm["sharpe_365"],
                "shadow_pf_daily":sm["profit_factor_daily"],
                "shadow_hard_breached":sm["hard_breached"],
                "risk_cagr":rm["cagr"],"risk_mdd":rm["max_drawdown"],
                "risk_hard_stopped":rm["hard_stopped"],
            })
    return pd.DataFrame(rows)


def yearly(eq):
    rows=[]
    for y,g in eq.groupby(eq.index.year):
        if len(g)<2: continue
        rows.append({
            "year":int(y),
            "return":float(g.equity.iloc[-1]/g.equity.iloc[0]-1),
            "within_year_mdd":float((g.equity/g.equity.cummax()-1).min()),
            "avg_exposure":float(g.target_exposure.abs().mean()),
            "defensive_panic_fraction":float(g.risk_state.isin(["DEFENSIVE","PANIC"]).mean()) if "risk_state" in g else np.nan,
            "vol_budget_high_fraction":float((g.budget_state=="HIGH").mean()) if "budget_state" in g else np.nan,
        })
    return pd.DataFrame(rows)


def benchmark_sma200(df1h, initial):
    d=df1h[["close"]].resample("1D",closed="left",label="right").last().dropna()
    d["sma200"]=d.close.rolling(200).mean()
    # Prior day's completed signal controls next day's return.
    sig=(d.close>d.sma200).astype(float).shift(1).fillna(0.0)
    r=d.close.pct_change().fillna(0.0)
    eq=initial*(1+sig*r).cumprod()
    dd=eq/eq.cummax()-1
    days=max((eq.index[-1]-eq.index[0]).days,1)
    years=days/365.2425
    return {
        "total_return":float(eq.iloc[-1]/initial-1),
        "cagr":float((eq.iloc[-1]/initial)**(1/years)-1),
        "max_drawdown":float(dd.min()),
    }


def benchmark_buyhold(df1h, initial):
    px=df1h.close
    eq=initial*px/px.iloc[0]
    dd=eq/eq.cummax()-1
    days=max((px.index[-1]-px.index[0]).days,1)
    years=days/365.2425
    return {
        "total_return":float(eq.iloc[-1]/initial-1),
        "cagr":float((eq.iloc[-1]/initial)**(1/years)-1),
        "max_drawdown":float(dd.min()),
    }


def acceptance(best_shadow, best_risk, wf, stress_shadow, stress_risk,
               delay_shadow, delay_risk, robust):
    if not wf.empty:
        oos_total=float((1+wf.test_risk_return).prod()-1)
        years=len(wf)
        oos_cagr=float((1+oos_total)**(1/years)-1) if 1+oos_total>0 else -1.0
        pos_ratio=float((wf.test_risk_return>0).mean())
        worst_oos_mdd=float(wf.test_risk_mdd.min())
        oos_stop_ratio=float(wf.test_risk_hard_stopped.mean())
    else:
        oos_cagr=np.nan; pos_ratio=np.nan; worst_oos_mdd=np.nan; oos_stop_ratio=np.nan

    return {
        "development_shadow_cagr_ge_20pct":bool(best_shadow["cagr"]>=0.20),
        "full_shadow_cagr_ge_25pct":bool(best_shadow["cagr"]>=0.25),
        "full_shadow_mdd_le_15pct":bool(abs(best_shadow["max_drawdown"])<=0.15),
        "full_shadow_sharpe_ge_1_3":bool(best_shadow["sharpe_365"]>=1.3),
        "full_shadow_pf_daily_ge_1_4":bool(best_shadow["profit_factor_daily"]>=1.4),
        "full_risk_policy_not_hard_stopped":bool(not best_risk["hard_stopped"]),
        "oos_risk_compound_cagr":oos_cagr,
        "oos_risk_positive_window_ratio":pos_ratio,
        "oos_risk_worst_window_mdd":worst_oos_mdd,
        "oos_hard_stop_window_ratio":oos_stop_ratio,
        "stress_2x_shadow_positive_cagr":bool(stress_shadow["cagr"]>0),
        "stress_2x_risk_not_hard_stopped":bool(not stress_risk["hard_stopped"]),
        "delay_shadow_positive_cagr":bool(delay_shadow["cagr"]>0),
        "delay_risk_not_hard_stopped":bool(not delay_risk["hard_stopped"]),
        "robust_shadow_positive_cagr_ratio":float((robust.shadow_cagr>0).mean()),
        "robust_risk_survival_ratio":float((~robust.risk_hard_stopped).mean()),
    }



def stress_periods(eq: pd.DataFrame):
    periods = [
        ("2018_bear", "2018-01-01", "2018-12-31"),
        ("2020_crash", "2020-02-01", "2020-05-31"),
        ("2021_cycle", "2021-01-01", "2021-12-31"),
        ("2022_bear", "2022-01-01", "2022-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025_2026_recent", "2025-01-01", "2026-08-25"),
    ]
    rows=[]
    for name,a,b in periods:
        g=eq.loc[(eq.index>=pd.Timestamp(a,tz="UTC")) &
                 (eq.index<=pd.Timestamp(b,tz="UTC"))]
        if len(g)<2:
            continue
        rows.append({
            "period":name,"start":str(g.index[0]),"end":str(g.index[-1]),
            "return":float(g.equity.iloc[-1]/g.equity.iloc[0]-1),
            "within_period_mdd":float((g.equity/g.equity.cummax()-1).min()),
            "avg_abs_exposure":float(g.target_exposure.abs().mean()),
            "defensive_panic_fraction":float(
                g.risk_state.isin(["DEFENSIVE","PANIC"]).mean()
            ) if "risk_state" in g else np.nan,
            "vol_budget_high_fraction":float((g.budget_state=="HIGH").mean()) if "budget_state" in g else np.nan,
        })
    return pd.DataFrame(rows)


def pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{100*x:.2f}%"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",default="2017-08-17")
    ap.add_argument("--end",default=(datetime.now(timezone.utc)-timedelta(days=1)).strftime("%Y-%m-%d"))
    ap.add_argument("--initial",type=float,default=10000.0)
    ap.add_argument("--fee-bps",type=float,default=5.5)
    ap.add_argument("--slippage-bps",type=float,default=2.0)
    ap.add_argument("--cache",default="data_cache")
    ap.add_argument("--results",default="results")
    args=ap.parse_args()

    out=Path(args.results); out.mkdir(parents=True,exist_ok=True)
    data=download_btc_1h(args.start,args.end,Path(args.cache))
    costs=CostModel(args.fee_bps,args.slippage_bps)
    rules=RiskRules()

    cand,outputs=run_candidates(data,costs,rules,args.initial)
    cand.to_csv(out/"candidate_summary.csv",index=False)
    best_name=str(cand.iloc[0].strategy)
    best_s=next(s for s in CANDIDATES if s.name==best_name)

    seq,str_,sev,bsm=outputs[best_name]["shadow"]
    req,rtr,rev,brm=outputs[best_name]["risk"]

    seq.to_csv(out/"equity_shadow.csv")
    req.to_csv(out/"equity_risk_gated.csv")
    str_.to_csv(out/"campaigns_shadow.csv",index=False)
    sev.to_csv(out/"events_shadow.csv",index=False)
    yearly(seq).to_csv(out/"yearly_shadow.csv",index=False)
    yearly(req).to_csv(out/"yearly_risk_gated.csv",index=False)
    stress_periods(seq).to_csv(out/"stress_periods_shadow.csv",index=False)
    if "risk_state" in seq.columns:
        rs = seq.groupby("risk_state").agg(
            bars=("equity","size"),
            avg_exposure=("target_exposure","mean"),
            avg_abs_exposure=("target_exposure",lambda z: z.abs().mean()),
        ).reset_index()
        rs["fraction"] = rs["bars"] / max(len(seq), 1)
        rs.to_csv(out/"risk_state_distribution.csv", index=False)
    if "budget_state" in seq.columns:
        bs = seq.groupby("budget_state").agg(
            bars=("equity","size"),
            avg_exposure=("target_exposure","mean"),
            avg_abs_exposure=("target_exposure",lambda z: z.abs().mean()),
            avg_rv30=("rv30","mean"),
            avg_rv_ratio=("rv_ratio","mean"),
        ).reset_index()
        bs["fraction"] = bs["bars"] / max(len(seq), 1)
        bs.to_csv(out/"vol_budget_distribution.csv", index=False)

    wf=walk_forward(data,costs,rules,args.initial)
    wf.to_csv(out/"walk_forward.csv",index=False)

    stress_costs=CostModel(args.fee_bps*2,args.slippage_bps*2)
    stress_rows=[]; stress_shadow_map={}; stress_risk_map={}
    for s in CANDIDATES:
        se,st,sv,sx=backtest(data,s,stress_costs,rules,args.initial,enforce_hard_stop=False)
        sm=metrics(se,st,sx,args.initial)
        re,rt,rv,rx=backtest(data,s,stress_costs,rules,args.initial,enforce_hard_stop=True)
        rm=metrics(re,rt,rx,args.initial)
        stress_shadow_map[s.name]=sm; stress_risk_map[s.name]=rm
        stress_rows.append({
            "strategy":s.name,
            "shadow_cagr":sm["cagr"],"shadow_mdd":sm["max_drawdown"],
            "shadow_sharpe":sm["sharpe_365"],"shadow_pf":sm["profit_factor_daily"],
            "shadow_hard_breached":sm["hard_breached"],
            "risk_cagr":rm["cagr"],"risk_mdd":rm["max_drawdown"],
            "risk_hard_stopped":rm["hard_stopped"],
        })
    pd.DataFrame(stress_rows).to_csv(out/"cost_stress_2x.csv",index=False)

    delay_rows=[]; delay_shadow_map={}; delay_risk_map={}
    for s in CANDIDATES:
        se,st,sv,sx=backtest(data,s,costs,rules,args.initial,signal_delay_bars=2,enforce_hard_stop=False)
        sm=metrics(se,st,sx,args.initial)
        re,rt,rv,rx=backtest(data,s,costs,rules,args.initial,signal_delay_bars=2,enforce_hard_stop=True)
        rm=metrics(re,rt,rx,args.initial)
        delay_shadow_map[s.name]=sm; delay_risk_map[s.name]=rm
        delay_rows.append({
            "strategy":s.name,
            "shadow_cagr":sm["cagr"],"shadow_mdd":sm["max_drawdown"],
            "shadow_sharpe":sm["sharpe_365"],
            "risk_cagr":rm["cagr"],"risk_mdd":rm["max_drawdown"],
            "risk_hard_stopped":rm["hard_stopped"],
        })
    pd.DataFrame(delay_rows).to_csv(out/"execution_delay_stress.csv",index=False)

    nbe,nbt,nbv,nbx=backtest(
        data,best_s,costs,rules,args.initial,
        enforce_hard_stop=False,vol_budget_enabled=False
    )
    no_budget_m=metrics(nbe,nbt,nbx,args.initial)
    ce,ct,cv,cx=backtest(
        data,best_s,costs,rules,args.initial,
        tactical_enabled=False,enforce_hard_stop=False
    )
    core_m=metrics(ce,ct,cx,args.initial)
    be,bt,bv,bx=backtest(
        data,best_s,costs,rules,args.initial,
        enforce_hard_stop=False, market_risk_enabled=False,
        vol_budget_enabled=False
    )
    base_engine_m=metrics(be,bt,bx,args.initial)
    re0,rt0,rv0,rx0=backtest(
        data,best_s,costs,rules,args.initial,
        enforce_hard_stop=False, hysteresis_enabled=False
    )
    raw_overlay_m=metrics(re0,rt0,rx0,args.initial)
    pd.DataFrame([
        {"variant":"selected full V7.2 proactive + volatility budget",**bsm},
        {"variant":"same engine with volatility budget disabled",**no_budget_m},
        {"variant":"core only with volatility budget",**core_m},
        {"variant":"base return engine (market overlay disabled)",**base_engine_m},
        {"variant":"raw proactive overlay (hysteresis disabled)",**raw_overlay_m},
    ]).to_csv(out/"attribution.csv",index=False)

    robust=robustness_grid(data,best_s,costs,rules,args.initial)
    robust.to_csv(out/"robustness_grid.csv",index=False)

    bh=benchmark_buyhold(data,args.initial)
    sma=benchmark_sma200(data,args.initial)
    gates=acceptance(
        bsm,brm,wf,
        stress_shadow_map[best_name],stress_risk_map[best_name],
        delay_shadow_map[best_name],delay_risk_map[best_name],
        robust
    )

    krw=pd.DataFrame(index=seq.index)
    krw["shadow_equity_krw"]=10_000_000*(seq.equity/args.initial)
    krw["risk_gated_equity_krw"]=10_000_000*(req.equity/args.initial)
    krw.to_csv(out/"krw_10m_projection.csv")

    primary_pass = (
        gates["full_shadow_cagr_ge_25pct"] and
        gates["full_shadow_mdd_le_15pct"] and
        gates["full_shadow_sharpe_ge_1_3"] and
        gates["full_shadow_pf_daily_ge_1_4"] and
        gates["full_risk_policy_not_hard_stopped"] and
        gates["stress_2x_risk_not_hard_stopped"] and
        gates["delay_risk_not_hard_stopped"] and
        gates["oos_hard_stop_window_ratio"] == 0
    )

    summary={
        "version":"V7.2 Volatility Budget",
        "data_start":str(data.index.min()),"data_end":str(data.index.max()),
        "rows_1h":len(data),
        "selected_candidate":best_name,
        "selection_method":"MDD<=15% candidates first, then continuous risk-adjusted score",
        "selected_shadow_metrics":bsm,
        "selected_risk_gated_metrics":brm,
        "no_vol_budget_shadow_metrics":no_budget_m,
        "core_only_shadow_metrics":core_m,
        "base_engine_shadow_metrics":base_engine_m,
        "raw_overlay_shadow_metrics":raw_overlay_m,
        "buy_hold":bh,
        "sma200_long_cash":sma,
        "acceptance":gates,
        "overall_pass":bool(primary_pass),
        "research_note":(
            "V7.2 was designed after observing V1-V7.1 results. Its walk-forward "
            "windows are useful robustness checks but are NOT pristine untouched "
            "out-of-sample evidence for the overall research program."
        ),
        "assumptions":{
            "price_source":"Binance BTCUSDT spot 1H price proxy",
            "fee_bps_per_rebalance":costs.fee_bps,
            "slippage_bps_per_rebalance":costs.slippage_bps,
            "funding_included":False,
            "signal_timing":"completed daily/4H data -> later 4H open",
            "hard_stop_policy":"proactive market-state scaling + independent volatility budget + separate 15% terminal stop in risk-gated run; no progressive account-DD sizing in shadow run",
            "martingale":False,"averaging_down":False,"simultaneous_hedge":False,
        }
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str))

    lines=[
        "# BTC AI EA V7.2 — Volatility Budget","",
        f"- Data: {summary['data_start']} → {summary['data_end']} ({len(data):,} 1H bars)",
        f"- Selected candidate: **{best_name}**",
        f"- Shadow CAGR / MDD: **{pct(bsm['cagr'])} / {pct(bsm['max_drawdown'])}**",
        f"- Shadow Sharpe / PF: **{bsm['sharpe_365']:.2f} / {bsm['profit_factor_daily']:.2f}**",
        f"- Shadow hard-DD breached: **{bsm['hard_breached']}** ({bsm['hard_breach_count']} crossings)",
        f"- Risk-gated CAGR / MDD: **{pct(brm['cagr'])} / {pct(brm['max_drawdown'])}**",
        f"- Risk-gated hard stop: **{brm['hard_stopped']}**",
        f"- Same engine without volatility budget CAGR / MDD: **{pct(no_budget_m['cagr'])} / {pct(no_budget_m['max_drawdown'])}**",
        f"- Core-only shadow CAGR: **{pct(core_m['cagr'])}**",
        f"- Base-engine CAGR (no market overlay): **{pct(base_engine_m['cagr'])}**",
        f"- Raw-overlay CAGR (no hysteresis): **{pct(raw_overlay_m['cagr'])}**",
        f"- Buy & hold CAGR / MDD: **{pct(bh['cagr'])} / {pct(bh['max_drawdown'])}**",
        f"- 200D long/cash CAGR / MDD: **{pct(sma['cagr'])} / {pct(sma['max_drawdown'])}**",
        f"- Overall acceptance: **{'PASS' if primary_pass else 'FAIL'}**",
        "",
        "## Candidate comparison","",
        cand.to_markdown(index=False),
        "",
        "## Walk-forward OOS robustness","",
        wf.to_markdown(index=False) if not wf.empty else "Not enough windows.",
        "",
        "## Nearby-parameter robustness","",
        robust.to_markdown(index=False),
        "",
        "## Acceptance gates","",
        "```json",json.dumps(gates,indent=2,default=str),"```",
        "",
        "## Research integrity note","",
        summary["research_note"],
        "",
        "## Decision rule","",
        "Do not deploy live unless V7.2 keeps full-sample MDD under 15%, the risk-gated policy survives, OOS windows do not "
        "trigger the hard stop, and futures/funding-aware validation plus paper trading pass."
    ]
    (out/"REPORT.md").write_text("\n".join(lines))
    print("\n=== V7.2 COMPLETE ===")
    print((out/"REPORT.md").read_text())


if __name__ == "__main__":
    main()
