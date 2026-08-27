#!/usr/bin/env python3
"""
BTC AI EA — V9.3 Cost-Aware Execution Architecture Backtester
==============================================================

Why V9.3 exists
---------------
V9.2 reduced rebalances and transaction costs while keeping the selected shadow
portfolio inside the 15% drawdown budget, but the selected model still failed
when fee + slippage assumptions were doubled. V9.3 freezes the V9.2C return,
risk and convex-exposure engine and changes only EXECUTION ECONOMICS.

Architecture
------------
1) SIGNAL/RISK ENGINE: frozen from V9.2C_COOLDOWN2.
2) COST-AWARE NO-TRADE BAND: the ordinary rebalance deadband widens when the
   estimated one-way trading cost is large relative to current 4H ATR. Higher
   assumed costs therefore require a larger exposure correction before trading.
3) BATCH REBALANCE INTERVAL: non-urgent exposure increases/drift corrections can
   be deferred for a small number of completed 4H bars. De-risking bypasses it.
4) TACTICAL RE-ENTRY COOLDOWN: after a tactical exit, repeated breakout re-entry
   can be blocked briefly to reduce whipsaw turnover.
5) URGENT DE-RISK BYPASS: zero exposure, worsening market-risk state, pyramid
   stage reduction, tactical exit and regime risk reductions remain immediate.
6) STRESS INTEGRITY: 2x costs are tested in two ways: adaptive (the execution
   engine sees the higher cost) and frozen-decision (actual cost doubles while
   the no-trade band still uses the base cost estimate). This prevents the cost
   stress from being passed solely by mechanically changing behavior in-test.
7) VALIDATION: full history, 15% terminal gate, walk-forward OOS, +4H delay,
   stress periods and an 18-cell execution robustness grid.

Research-integrity note
-----------------------
V9.3 is development-stage research informed by V9.2 full-sample results. Its
walk-forward windows are stability checks, not pristine untouched OOS evidence
for the entire research program.

Anti-lookahead / anti-martingale rules
--------------------------------------
- Daily features use only fully completed prior daily candles.
- 4H decisions use a completed 4H bar and affect only a later 4H open.
- Pyramid stages still unlock only after favorable ATR-normalized movement.
- No averaging down, martingale, simultaneous long/short hedge or fixed high
  leverage is introduced by the execution layer.

Price source
------------
Binance BTCUSDT spot 1H public archives are used as the long-history price proxy.
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
    # Stage-0 core exposure. Unlike V7.1, these are intentionally small.
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
    # Convex exposure staircase. Stages are unlocked only by favorable movement.
    convex_add1: float
    convex_add2: float
    convex_add3: float
    convex_trigger1_atr: float
    convex_trigger2_atr: float
    convex_trigger3_atr: float
    convex_stepdown_atr: float
    tactical_min_stage: int
    # V9.2 execution hysteresis / turnover control.
    rebalance_deadband: float
    stage_up_confirm_bars: int
    stage_reentry_cooldown_bars: int
    normal_boost: float
    state_change_rebalance: bool
    # V9.3 execution economics.
    cost_band_mult: float
    rebalance_min_interval_bars: int
    tactical_reentry_cooldown_bars: int
    legacy_v92_execution: bool
    # Downside fields are retained for feature compatibility but disabled in V9.
    down_sv_warn: float
    down_sv_high: float
    down_ratio_warn: float
    down_ratio_high: float
    down_dd10_warn: float
    down_dd10_high: float
    down_ret3_high: float
    down_scale_warn: float
    down_scale_high: float
    down_recovery_days: int
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


BASE_V92C = dict(
    fast_days=100, slow_days=250, slope_days=30,
    strong_long=0.256, weak_long=0.088, tactical_long=0.144,
    breakout_4h=40, exit_4h=20, trail_atr_4h=4.5,
    breakout_buffer_atr=0.08, vol_target=0.55, vol_floor_scale=0.55,
    max_long=0.704,
    convex_add1=0.128, convex_add2=0.136, convex_add3=0.144,
    convex_trigger1_atr=1.44, convex_trigger2_atr=3.36, convex_trigger3_atr=5.76,
    convex_stepdown_atr=2.16, tactical_min_stage=2,
    rebalance_deadband=0.040, stage_up_confirm_bars=1,
    stage_reentry_cooldown_bars=2, normal_boost=1.00,
    state_change_rebalance=False,
    cost_band_mult=0.0, rebalance_min_interval_bars=0,
    tactical_reentry_cooldown_bars=0, legacy_v92_execution=False,
    down_sv_warn=9.0, down_sv_high=9.0,
    down_ratio_warn=9.0, down_ratio_high=9.0,
    down_dd10_warn=-0.99, down_dd10_high=-0.99, down_ret3_high=-0.99,
    down_scale_warn=1.0, down_scale_high=1.0, down_recovery_days=1,
    risk_fast_days=20, risk_mid_days=50, risk_slow_days=200,
    mom5_cut=-0.10, mom20_cut=-0.17, high20_cut=-0.15, rv_ratio_cut=1.45,
    caution_scale=0.68, defense_scale=0.32, panic_scale=0.00, recovery_days=5,
)


def _mk_strategy(name: str, **overrides) -> Strategy:
    cfg = dict(BASE_V92C)
    cfg.update(overrides)
    return Strategy(name=name, **cfg)


CANDIDATES = [
    # Dynamic band only: isolate the cost/ATR no-trade hypothesis.
    _mk_strategy("V93A_COST05", cost_band_mult=0.50),
    _mk_strategy("V93B_COST075", cost_band_mult=0.75),
    _mk_strategy("V93C_COST10", cost_band_mult=1.00),
    # Batch non-urgent corrections after the cost-aware band is applied.
    _mk_strategy(
        "V93D_BATCH1", cost_band_mult=0.75,
        rebalance_min_interval_bars=1,
    ),
    # Add a brief tactical re-entry cooldown to suppress breakout whipsaw.
    _mk_strategy(
        "V93E_TACTICAL2", cost_band_mult=0.75,
        rebalance_min_interval_bars=1, tactical_reentry_cooldown_bars=2,
    ),
    # V9.2 robustness showed convex re-entry cooldown=4 was a nearby 15%-MDD
    # survivor. Combine that survivor geometry with cost-aware execution.
    _mk_strategy(
        "V93F_K4_COST05", rebalance_deadband=0.030,
        stage_reentry_cooldown_bars=4, cost_band_mult=0.50,
        rebalance_min_interval_bars=1, tactical_reentry_cooldown_bars=2,
    ),
    _mk_strategy(
        "V93G_BALANCED", stage_reentry_cooldown_bars=4,
        cost_band_mult=0.75, rebalance_min_interval_bars=1,
        tactical_reentry_cooldown_bars=2,
    ),
    # Exact V9.2 selected control. No V9.3 execution economics.
    _mk_strategy("V92C_CONTROL", legacy_v92_execution=True),
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
    req = urllib.request.Request(url, headers={"User-Agent": "btc-ai-ea-v9.3/1.0"})
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

    # Asymmetric downside-only risk features. Positive-return days contribute
    # zero to semivolatility instead of being treated as risk.
    neg_ret = ret.clip(upper=0.0)
    d["down_sv10"] = np.sqrt(
        neg_ret.pow(2).rolling(10, min_periods=7).mean()
    ) * np.sqrt(365)
    d["down_sv30"] = np.sqrt(
        neg_ret.pow(2).rolling(30, min_periods=20).mean()
    ) * np.sqrt(365)
    d["down_sv90_med"] = d["down_sv30"].rolling(90, min_periods=45).median()
    d["down_sv_ratio"] = d["down_sv10"] / d["down_sv90_med"].replace(0, np.nan)
    d["ret3"] = d["close"].pct_change(3)
    d["high10"] = d["close"].shift(1).rolling(10, min_periods=5).max()
    d["dd10"] = d["close"] / d["high10"] - 1.0

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
        "risk_fast", "risk_mid", "risk_slow", "ret3", "ret5", "ret20", "dd10", "dd20",
        "down_sv10", "down_sv30", "down_sv_ratio", "dclose", "daily_seq"
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


DOWNSIDE_LEVEL = {"NORMAL":0, "WARN":1, "HIGH":2}
DOWNSIDE_STATE = {v:k for k,v in DOWNSIDE_LEVEL.items()}


def raw_downside_state(down_sv10: float, down_sv30: float, down_sv_ratio: float,
                       ret3: float, ret5: float, dd10: float, shock: bool,
                       s: Strategy) -> str:
    """Classify downside acceleration from completed daily data only."""
    high = (
        (np.isfinite(down_sv10) and down_sv10 >= s.down_sv_high and
         np.isfinite(down_sv_ratio) and down_sv_ratio >= s.down_ratio_warn) or
        (np.isfinite(down_sv_ratio) and down_sv_ratio >= s.down_ratio_high and
         np.isfinite(ret5) and ret5 < -0.02) or
        (np.isfinite(ret3) and ret3 <= s.down_ret3_high) or
        (np.isfinite(dd10) and dd10 <= s.down_dd10_high and
         np.isfinite(ret5) and ret5 < -0.025) or
        (shock and np.isfinite(ret5) and ret5 < -0.025)
    )
    if high:
        return "HIGH"
    warn = (
        (np.isfinite(down_sv10) and down_sv10 >= s.down_sv_warn and
         np.isfinite(down_sv_ratio) and down_sv_ratio >= s.down_ratio_warn) or
        (np.isfinite(down_sv_ratio) and down_sv_ratio >= s.down_ratio_warn and
         np.isfinite(ret5) and ret5 < -0.01) or
        (np.isfinite(dd10) and dd10 <= s.down_dd10_warn) or
        (np.isfinite(ret3) and ret3 < -0.035 and
         np.isfinite(down_sv30) and down_sv30 >= 0.45)
    )
    return "WARN" if warn else "NORMAL"


def downside_hysteresis_update(current: str | None, raw: str, recovery_count: int,
                               recovery_days: int):
    if current is None:
        return raw, 0
    c = DOWNSIDE_LEVEL[current]; r = DOWNSIDE_LEVEL[raw]
    if r >= c:
        return raw, 0
    recovery_count += 1
    if recovery_count < recovery_days:
        return current, recovery_count
    return DOWNSIDE_STATE[max(r, c - 1)], 0


def downside_scale_for_state(state: str, s: Strategy) -> float:
    if state == "WARN":
        return s.down_scale_warn
    if state == "HIGH":
        return s.down_scale_high
    return 1.0


def market_scale_for_state(state: str, s: Strategy) -> float:
    if state == "CAUTION":
        return s.caution_scale
    if state == "DEFENSIVE":
        return s.defense_scale
    if state == "PANIC":
        return s.panic_scale
    return 1.0


def convex_core_exposure(regime: str, stage: int, s: Strategy) -> float:
    base = core_exposure(regime, s)
    if base <= 0:
        return 0.0
    adds = (s.convex_add1, s.convex_add2, s.convex_add3)
    return base + sum(adds[:max(0, min(int(stage), 3))])


def target_exposure(regime: str, rv: float, tactical_side: int, s: Strategy,
                    market_state: str, downside_state: str, convex_stage: int,
                    market_risk_enabled: bool = True,
                    downside_filter_enabled: bool = False,
                    convex_enabled: bool = True):
    vscale = vol_scale(rv, s)
    stage = convex_stage if convex_enabled else 0
    core = convex_core_exposure(regime, stage, s) * vscale
    tactical = 0.0
    tactical_stage_ok = (not convex_enabled or stage >= s.tactical_min_stage)
    if tactical_side > 0 and regime in ("BULL_STRONG", "BULL_WEAK") and tactical_stage_ok:
        if (market_state in ("NORMAL", "CAUTION") and
                (not downside_filter_enabled or downside_state != "HIGH")):
            tactical = s.tactical_long * vscale
    mscale = market_scale_for_state(market_state, s) if market_risk_enabled else 1.0
    dscale = downside_scale_for_state(downside_state, s) if downside_filter_enabled else 1.0
    exp = (core + tactical) * mscale * dscale
    if market_state == "NORMAL":
        exp *= s.normal_boost
    return float(np.clip(exp, 0.0, s.max_long)), mscale, dscale


def trade_cost(delta_notional: float, costs: CostModel) -> float:
    # Fee plus an explicit slippage penalty. Both are charged on every rebalance.
    return abs(delta_notional) * (costs.fee_bps + costs.slippage_bps) / 10_000.0


def backtest(df1h: pd.DataFrame, s: Strategy, costs: CostModel, rules: RiskRules,
             initial: float = 10_000.0, signal_delay_bars: int = 1,
             tactical_enabled: bool = True, enforce_hard_stop: bool = True,
             market_risk_enabled: bool = True, hysteresis_enabled: bool = True,
             downside_filter_enabled: bool = False, convex_enabled: bool = True,
             trade_start: pd.Timestamp | None = None,
             decision_costs: CostModel | None = None):
    x = build_features(df1h, s).dropna(
        subset=["rv30", "atr4h", "regime", "risk_fast", "risk_mid",
                "risk_slow", "ret20", "dd20", "down_sv10", "down_sv30",
                "down_sv_ratio", "ret3", "dd10"]
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

    # Realized costs can be stressed independently from the cost estimate used
    # by the no-trade band. By default both are the same.
    if decision_costs is None:
        decision_costs = costs

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
    tactical_reentry_cooldown = 0
    tactical_reentry_blocks = 0

    # Convex campaign state. Changes are computed after a completed 4H bar and
    # therefore affect the NEXT bar's open exposure only.
    convex_stage = 0
    convex_anchor = np.nan
    convex_peak = np.nan
    convex_cushion_atr = 0.0
    convex_giveback_atr = 0.0
    convex_stage_increases = 0
    convex_stage_decreases = 0
    stage_up_candidate = 0
    stage_up_confirm_count = 0
    stage_reentry_cooldown = 0
    stage_up_deferred = 0
    stage_cooldown_blocks = 0

    total_costs = 0.0
    gross_turnover_usd = 0.0
    rebalance_count = 0
    skipped_rebalances = 0
    cost_band_skips = 0
    interval_blocks = 0
    effective_deadband_sum = 0.0
    effective_deadband_count = 0
    last_rebalance_bar_index = -10**9
    bars_exposed = 0
    rows = []
    trades = []
    events = []
    last_exposure = 0.0
    last_regime = None
    last_risk_state = None
    last_downside_state = None
    last_convex_stage = None
    last_tactical_side = None

    # Hysteresis updates only when a NEW completed daily bar becomes available.
    effective_risk_state = None
    recovery_count = 0
    last_daily_seq = None
    last_raw_state = "NORMAL"
    last_warning_count = 0
    effective_downside_state = None
    downside_recovery_count = 0
    last_raw_downside_state = "NORMAL"

    def set_position(ts, px, desired_exp, reason, bar_index):
        nonlocal equity, qty, total_costs, gross_turnover_usd, rebalance_count, last_exposure
        nonlocal last_rebalance_bar_index
        desired_qty = equity * desired_exp / px if px > 0 else 0.0
        delta = desired_qty - qty
        notional = delta * px
        cost = trade_cost(notional, costs)
        equity -= cost
        total_costs += cost
        gross_turnover_usd += abs(notional)
        if abs(delta) > 1e-12:
            rebalance_count += 1
            events.append({
                "time":ts, "event":"REBALANCE", "reason":reason,
                "price":px, "delta_qty":delta, "cost":cost,
                "target_exposure":desired_exp,
            })
        qty = desired_qty
        last_exposure = desired_exp
        if abs(delta) > 1e-12:
            last_rebalance_bar_index = bar_index

    for bar_index, (ts, r) in enumerate(x.iterrows()):
        o, h, l, c, a = map(float, [r.open, r.high, r.low, r.close, r.atr4h])
        regime = str(r.regime)
        rv = float(r.rv30)
        rv_ratio = float(r.rv_ratio) if np.isfinite(r.rv_ratio) else 1.0
        shock = bool(r.shock)
        dclose = float(r.dclose)
        rfast = float(r.risk_fast)
        rmid = float(r.risk_mid)
        rslow = float(r.risk_slow)
        ret3 = float(r.ret3) if np.isfinite(r.ret3) else 0.0
        ret5 = float(r.ret5) if np.isfinite(r.ret5) else 0.0
        ret20 = float(r.ret20) if np.isfinite(r.ret20) else 0.0
        dd10 = float(r.dd10) if np.isfinite(r.dd10) else 0.0
        dd20 = float(r.dd20) if np.isfinite(r.dd20) else 0.0
        down_sv10 = float(r.down_sv10) if np.isfinite(r.down_sv10) else 0.0
        down_sv30 = float(r.down_sv30) if np.isfinite(r.down_sv30) else 0.0
        down_sv_ratio = float(r.down_sv_ratio) if np.isfinite(r.down_sv_ratio) else 1.0
        daily_seq = int(r.daily_seq)
        tactical_cooldown_was_active = tactical_reentry_cooldown > 0

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

            raw_down = raw_downside_state(
                down_sv10, down_sv30, down_sv_ratio, ret3, ret5, dd10, shock, s
            )
            last_raw_downside_state = raw_down
            if hysteresis_enabled:
                effective_downside_state, downside_recovery_count = downside_hysteresis_update(
                    effective_downside_state, raw_down, downside_recovery_count,
                    s.down_recovery_days
                )
            else:
                effective_downside_state = raw_down
                downside_recovery_count = 0
            last_daily_seq = daily_seq

        risk_state = effective_risk_state or last_raw_state
        downside_state = effective_downside_state or last_raw_downside_state
        warning_count = last_warning_count

        if pending_tactical is not None:
            action, pside, remaining = pending_tactical
            remaining -= 1
            if remaining <= 0:
                if action == "ENTER":
                    if (regime in ("BULL_STRONG", "BULL_WEAK") and
                            risk_state in ("NORMAL", "CAUTION") and
                            (not downside_filter_enabled or downside_state != "HIGH") and
                            (not convex_enabled or convex_stage >= s.tactical_min_stage) and
                            tactical_reentry_cooldown <= 0 and
                            not hard_stopped):
                        tactical_side = pside
                        tactical_peak = o
                else:
                    tactical_side = 0
                    tactical_peak = np.nan
                    tactical_reentry_cooldown = max(
                        tactical_reentry_cooldown,
                        max(0, int(s.tactical_reentry_cooldown_bars))
                    )
                pending_tactical = None
            else:
                pending_tactical = (action, pside, remaining)

        desired = 0.0
        reason = "MODEL"
        mscale = 0.0
        dscale = 0.0
        if hard_stopped or day_lock or week_lock:
            reason = "LOCK"
        else:
            desired, mscale, dscale = target_exposure(
                regime, rv, tactical_side if tactical_enabled else 0, s,
                risk_state, downside_state, convex_stage,
                market_risk_enabled=market_risk_enabled,
                downside_filter_enabled=downside_filter_enabled,
                convex_enabled=convex_enabled,
            )

        # V9.3 cost-aware execution. The dynamic no-trade band expands when
        # trading cost is large relative to 4H ATR. It only throttles ordinary
        # exposure corrections/increases; de-risking remains immediate.
        cost_rate = (decision_costs.fee_bps + decision_costs.slippage_bps) / 10_000.0
        atr_frac = max(a / max(o, 1e-12), 1e-4)
        cost_component = s.cost_band_mult * cost_rate / atr_frac
        effective_deadband = float(np.clip(
            s.rebalance_deadband + cost_component,
            s.rebalance_deadband, 0.14
        ))
        effective_deadband_sum += effective_deadband
        effective_deadband_count += 1

        if s.legacy_v92_execution:
            # Exact V9.2C execution semantics for the control candidate.
            risk_rank = {"NORMAL":0, "CAUTION":1, "DEFENSIVE":2, "PANIC":3}
            risk_worsened = (
                last_risk_state is not None and
                risk_rank.get(risk_state, 0) > risk_rank.get(last_risk_state, 0)
            )
            urgent_zero = desired == 0 and abs(last_exposure) > 1e-12
            stage_changed = (
                last_convex_stage is not None and convex_stage != last_convex_stage
            )
            tactical_changed = (
                last_tactical_side is not None and tactical_side != last_tactical_side
            )
            regime_exit = (
                last_regime in ("BULL_STRONG", "BULL_WEAK") and
                regime not in ("BULL_STRONG", "BULL_WEAK")
            )
            structural = stage_changed or tactical_changed or regime_exit
            drift_large = abs(desired - last_exposure) >= s.rebalance_deadband
            should_rebalance = urgent_zero or risk_worsened or structural or drift_large
        elif s.state_change_rebalance:
            should_rebalance = (
                abs(desired - last_exposure) >= s.rebalance_deadband or
                regime != last_regime or risk_state != last_risk_state or
                downside_state != last_downside_state or
                convex_stage != last_convex_stage or
                (desired == 0 and abs(last_exposure) > 1e-12)
            )
        else:
            risk_rank = {"NORMAL":0, "CAUTION":1, "DEFENSIVE":2, "PANIC":3}
            risk_worsened = (
                last_risk_state is not None and
                risk_rank.get(risk_state, 0) > risk_rank.get(last_risk_state, 0)
            )
            urgent_zero = desired == 0 and abs(last_exposure) > 1e-12
            stage_decreased = (
                last_convex_stage is not None and convex_stage < last_convex_stage
            )
            tactical_exit = (
                last_tactical_side is not None and last_tactical_side > 0 and tactical_side == 0
            )
            regime_risk_reduction = (
                last_regime is not None and regime != last_regime and
                desired < last_exposure - 1e-12
            )
            urgent_derisk = (
                urgent_zero or risk_worsened or stage_decreased or
                tactical_exit or regime_risk_reduction
            )

            gap = abs(desired - last_exposure)
            drift_large = gap >= effective_deadband
            interval_blocked = (
                drift_large and not urgent_derisk and
                (bar_index - last_rebalance_bar_index) <= max(0, int(s.rebalance_min_interval_bars))
            )
            if interval_blocked:
                interval_blocks += 1
            should_rebalance = urgent_derisk or (drift_large and not interval_blocked)

            if not urgent_derisk and gap > 1e-12 and not drift_large:
                cost_band_skips += 1

        if should_rebalance:
            set_position(ts, o, desired, reason, bar_index)
        elif abs(desired - last_exposure) > 1e-12:
            skipped_rebalances += 1
        last_regime = regime
        last_risk_state = risk_state
        last_downside_state = downside_state
        last_convex_stage = convex_stage
        last_tactical_side = tactical_side

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
            if (risk_state in ("DEFENSIVE", "PANIC") or
                    (downside_filter_enabled and downside_state == "HIGH")) and tactical_side > 0:
                if pending_tactical is None:
                    pending_tactical = ("EXIT", +1, max(1, signal_delay_bars))
            elif tactical_side == 0 and pending_tactical is None:
                entry_signal = (
                    regime in ("BULL_STRONG", "BULL_WEAK") and
                    risk_state in ("NORMAL", "CAUTION") and
                    (not downside_filter_enabled or downside_state != "HIGH") and
                    (not convex_enabled or convex_stage >= s.tactical_min_stage) and
                    c > float(r.entry_hi) + buf
                )
                if entry_signal and tactical_reentry_cooldown <= 0:
                    pending_tactical = ("ENTER", +1, max(1, signal_delay_bars))
                elif entry_signal and tactical_reentry_cooldown > 0:
                    tactical_reentry_blocks += 1
            elif tactical_side > 0:
                tactical_peak = max(tactical_peak, h) if np.isfinite(tactical_peak) else h
                trail = tactical_peak - s.trail_atr_4h * a
                if (regime not in ("BULL_STRONG", "BULL_WEAK") or
                        c < float(r.exit_lo) or c < trail):
                    if pending_tactical is None:
                        pending_tactical = ("EXIT", +1, max(1, signal_delay_bars))

        # Convex state update uses this COMPLETED 4H bar. Any resulting
        # exposure change is applied only at the next loop's 4H open.
        # V9.2 keeps stage decreases immediate, but can require confirmation
        # and a short cooldown before re-adding a stage that was just cut.
        old_stage = convex_stage
        eligible_convex = (
            convex_enabled and not hard_stopped and
            regime in ("BULL_STRONG", "BULL_WEAK") and
            risk_state in ("NORMAL", "CAUTION") and
            not day_lock and not week_lock
        )
        cooldown_active = stage_reentry_cooldown > 0
        if not eligible_convex:
            convex_stage = 0
            convex_anchor = np.nan
            convex_peak = np.nan
            convex_cushion_atr = 0.0
            convex_giveback_atr = 0.0
            stage_up_candidate = 0
            stage_up_confirm_count = 0
            stage_reentry_cooldown = 0
        else:
            if not np.isfinite(convex_anchor):
                convex_anchor = c
                convex_peak = h
                convex_stage = 0
                stage_up_candidate = 0
                stage_up_confirm_count = 0
            else:
                convex_peak = max(convex_peak, h) if np.isfinite(convex_peak) else h
                atr_unit = max(a, 1e-12)
                convex_cushion_atr = (c - convex_anchor) / atr_unit
                convex_giveback_atr = max(0.0, (convex_peak - c) / atr_unit)
                cushion_stage = 0
                if convex_cushion_atr >= s.convex_trigger1_atr:
                    cushion_stage = 1
                if convex_cushion_atr >= s.convex_trigger2_atr:
                    cushion_stage = 2
                if convex_cushion_atr >= s.convex_trigger3_atr:
                    cushion_stage = 3
                giveback_steps = int(convex_giveback_atr // max(s.convex_stepdown_atr, 1e-9))
                raw_target_stage = max(0, cushion_stage - giveback_steps)

                if raw_target_stage < convex_stage:
                    convex_stage = raw_target_stage
                    stage_reentry_cooldown = max(0, int(s.stage_reentry_cooldown_bars))
                    stage_up_candidate = 0
                    stage_up_confirm_count = 0
                elif raw_target_stage > convex_stage:
                    if cooldown_active:
                        stage_cooldown_blocks += 1
                        stage_up_candidate = 0
                        stage_up_confirm_count = 0
                    else:
                        if stage_up_candidate == raw_target_stage:
                            stage_up_confirm_count += 1
                        else:
                            stage_up_candidate = raw_target_stage
                            stage_up_confirm_count = 1
                        if stage_up_confirm_count >= max(1, int(s.stage_up_confirm_bars)):
                            convex_stage = raw_target_stage
                            stage_up_candidate = 0
                            stage_up_confirm_count = 0
                        else:
                            stage_up_deferred += 1
                else:
                    stage_up_candidate = 0
                    stage_up_confirm_count = 0

                # Cooldown counts completed 4H observations after the cut.
                # A newly created cooldown is not decremented on the same bar.
                if old_stage <= convex_stage and stage_reentry_cooldown > 0:
                    stage_reentry_cooldown -= 1

        if convex_stage != old_stage:
            if convex_stage > old_stage:
                convex_stage_increases += convex_stage - old_stage
                event_name = "PYRAMID_UP"
            else:
                convex_stage_decreases += old_stage - convex_stage
                event_name = "PYRAMID_DOWN"
            events.append({
                "time":ts, "event":event_name, "reason":"CONVEX_STAGE",
                "price":c, "from_stage":old_stage, "to_stage":convex_stage,
                "anchor":convex_anchor, "peak":convex_peak,
                "cushion_atr":convex_cushion_atr, "giveback_atr":convex_giveback_atr,
            })

        # Existing tactical cooldown counts completed 4H bars. A cooldown
        # created by an exit on this bar is not decremented immediately.
        if tactical_cooldown_was_active and tactical_reentry_cooldown > 0:
            tactical_reentry_cooldown -= 1

        rows.append({
            "time":ts, "equity":equity, "qty":qty,
            "target_exposure":last_exposure, "regime":regime,
            "raw_risk_state":last_raw_state, "risk_state":risk_state,
            "risk_warning_count":warning_count, "market_scale":mscale,
            "raw_downside_state":last_raw_downside_state,
            "downside_state":downside_state, "downside_scale":dscale,
            "downside_recovery_count":downside_recovery_count,
            "recovery_count":recovery_count, "drawdown":dd,
            "convex_stage":convex_stage, "convex_anchor":convex_anchor,
            "convex_peak":convex_peak, "convex_cushion_atr":convex_cushion_atr,
            "convex_giveback_atr":convex_giveback_atr,
            "stage_up_confirm_count":stage_up_confirm_count,
            "stage_reentry_cooldown":stage_reentry_cooldown,
            "tactical_reentry_cooldown":tactical_reentry_cooldown,
            "effective_deadband":effective_deadband,
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
        "costs":total_costs, "gross_turnover_usd":gross_turnover_usd,
        "skipped_rebalances":skipped_rebalances,
        "cost_band_skips":cost_band_skips,
        "interval_blocks":interval_blocks,
        "tactical_reentry_blocks":tactical_reentry_blocks,
        "avg_effective_deadband":(
            effective_deadband_sum / max(effective_deadband_count, 1)
        ),
        "stage_up_deferred":stage_up_deferred,
        "stage_cooldown_blocks":stage_cooldown_blocks,
        "hard_stopped":hard_stopped,
        "hard_breached":hard_breached, "hard_breach_count":hard_breach_count,
        "first_hard_breach_time":(
            str(first_hard_breach_time) if first_hard_breach_time is not None else None
        ),
        "bars_exposed":bars_exposed, "bars_total":len(eqdf),
        "rebalances":rebalance_count,
        "convex_stage_increases":convex_stage_increases,
        "convex_stage_decreases":convex_stage_decreases,
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
        "gross_turnover_usd":float(extra.get("gross_turnover_usd", 0.0)),
        "gross_turnover_x_initial":float(extra.get("gross_turnover_usd", 0.0))/max(initial,1e-12),
        "skipped_rebalances":int(extra.get("skipped_rebalances", 0)),
        "cost_band_skips":int(extra.get("cost_band_skips", 0)),
        "interval_blocks":int(extra.get("interval_blocks", 0)),
        "tactical_reentry_blocks":int(extra.get("tactical_reentry_blocks", 0)),
        "avg_effective_deadband":float(extra.get("avg_effective_deadband", 0.0)),
        "stage_up_deferred":int(extra.get("stage_up_deferred", 0)),
        "stage_cooldown_blocks":int(extra.get("stage_cooldown_blocks", 0)),
        "market_exposure":extra["bars_exposed"]/max(extra["bars_total"],1),
        "hard_stopped":bool(extra["hard_stopped"]),
        "hard_breached":bool(extra.get("hard_breached", extra["hard_stopped"])),
        "hard_breach_count":int(extra.get("hard_breach_count", int(extra["hard_stopped"]))),
        "first_hard_breach_time":extra.get("first_hard_breach_time"),
        "convex_stage_increases":int(extra.get("convex_stage_increases", 0)),
        "convex_stage_decreases":int(extra.get("convex_stage_decreases", 0)),
    }


def objective(m):
    """Base-sample efficiency score; 2x-cost survival is ranked separately."""
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
        m["cagr"] - 0.45*mdd + 0.06*sh + 0.015*min(pf,5.0)
        - 8.0*excess - 0.10*int(m.get("hard_breached", False))
    )


def v93_joint_score(row):
    """Rank return/risk efficiency across base and 2x realized-cost stress."""
    base_mdd = abs(float(row["shadow_max_drawdown"]))
    stress_mdd = abs(float(row["stress2x_shadow_mdd"]))
    base_excess = max(0.0, base_mdd - 0.15)
    stress_excess = max(0.0, stress_mdd - 0.15)
    return (
        0.55*float(row["shadow_cagr"]) + 0.45*float(row["stress2x_shadow_cagr"])
        - 0.30*base_mdd - 0.45*stress_mdd
        + 0.035*float(row["shadow_sharpe_365"])
        + 0.010*min(float(row["shadow_profit_factor_daily"]), 5.0)
        - 6.0*base_excess - 9.0*stress_excess
        - 0.08*int(bool(row["shadow_hard_breached"]))
        - 0.12*int(bool(row["stress2x_shadow_hard_breached"]))
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
    return pd.DataFrame(rows).sort_values(
        ["mdd_gate","score","shadow_gross_turnover_x_initial"],
        ascending=[False,False,True]
    ), outputs


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
            scored.append((
                abs(m["max_drawdown"]) <= rules.hard_drawdown, objective(m),
                -m.get("gross_turnover_x_initial", 0.0), s, m
            ))
        scored.sort(key=lambda z:(z[0], z[1], z[2]), reverse=True)
        _, _, _, chosen, tm = scored[0]

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
    # 18 nearby execution combinations. Signal/risk/exposure parameters stay
    # frozen; only cost-band strength, batch interval and tactical cooldown vary.
    stress_costs = CostModel(costs.fee_bps*2, costs.slippage_bps*2)
    for cost_mult in (0.50,0.75,1.00):
        for interval in (0,1,2):
            for tac_cd in (0,2):
                s=replace(
                    best,
                    name=f"{best.name}_CB{cost_mult:.2f}_I{interval}_T{tac_cd}",
                    cost_band_mult=cost_mult,
                    rebalance_min_interval_bars=interval,
                    tactical_reentry_cooldown_bars=tac_cd,
                )
                seq,str_,sev,sex=backtest(data,s,costs,rules,initial,enforce_hard_stop=False)
                sm=metrics(seq,str_,sex,initial)
                req,rtr,rev,rex=backtest(data,s,costs,rules,initial,enforce_hard_stop=True)
                rm=metrics(req,rtr,rex,initial)
                s2e,s2t,s2v,s2x=backtest(
                    data,s,stress_costs,rules,initial,enforce_hard_stop=False
                )
                s2m=metrics(s2e,s2t,s2x,initial)
                s2re,s2rt,s2rv,s2rx=backtest(
                    data,s,stress_costs,rules,initial,enforce_hard_stop=True
                )
                s2rm=metrics(s2re,s2rt,s2rx,initial)
                rows.append({
                    "cost_band_mult":cost_mult,
                    "rebalance_min_interval_bars":interval,
                    "tactical_reentry_cooldown_bars":tac_cd,
                    "shadow_cagr":sm["cagr"],"shadow_mdd":sm["max_drawdown"],
                    "shadow_sharpe":sm["sharpe_365"],
                    "shadow_pf_daily":sm["profit_factor_daily"],
                    "shadow_rebalances":sm["rebalances"],
                    "shadow_costs_paid_usd":sm["costs_paid_usd"],
                    "shadow_turnover_x_initial":sm["gross_turnover_x_initial"],
                    "shadow_avg_effective_deadband":sm["avg_effective_deadband"],
                    "shadow_hard_breached":sm["hard_breached"],
                    "risk_hard_stopped":rm["hard_stopped"],
                    "stress2x_cagr":s2m["cagr"],
                    "stress2x_mdd":s2m["max_drawdown"],
                    "stress2x_rebalances":s2m["rebalances"],
                    "stress2x_hard_breached":s2m["hard_breached"],
                    "stress2x_risk_hard_stopped":s2rm["hard_stopped"],
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
    stress_costs=CostModel(args.fee_bps*2,args.slippage_bps*2)
    rules=RiskRules()

    # Base candidate runs.
    cand,outputs=run_candidates(data,costs,rules,args.initial)

    # 2x realized costs, with adaptive cost-aware decision band.
    stress_rows=[]; stress_shadow_map={}; stress_risk_map={}
    for s in CANDIDATES:
        se,st,sv,sx=backtest(data,s,stress_costs,rules,args.initial,enforce_hard_stop=False)
        sm=metrics(se,st,sx,args.initial)
        re,rt,rv,rx=backtest(data,s,stress_costs,rules,args.initial,enforce_hard_stop=True)
        rm=metrics(re,rt,rx,args.initial)
        stress_shadow_map[s.name]=sm; stress_risk_map[s.name]=rm
        stress_rows.append({
            "strategy":s.name,
            "stress2x_shadow_cagr":sm["cagr"],
            "stress2x_shadow_mdd":sm["max_drawdown"],
            "stress2x_shadow_sharpe":sm["sharpe_365"],
            "stress2x_shadow_pf":sm["profit_factor_daily"],
            "stress2x_shadow_rebalances":sm["rebalances"],
            "stress2x_shadow_costs_paid_usd":sm["costs_paid_usd"],
            "stress2x_shadow_turnover_x_initial":sm["gross_turnover_x_initial"],
            "stress2x_shadow_avg_effective_deadband":sm["avg_effective_deadband"],
            "stress2x_shadow_hard_breached":sm["hard_breached"],
            "stress2x_risk_cagr":rm["cagr"],
            "stress2x_risk_mdd":rm["max_drawdown"],
            "stress2x_risk_hard_stopped":rm["hard_stopped"],
        })
    stress_df=pd.DataFrame(stress_rows)

    # Frozen-decision 2x stress: actual costs double but execution band still
    # uses the base 1x cost estimate. This is a harder anti-gaming diagnostic.
    frozen_rows=[]; frozen_shadow_map={}; frozen_risk_map={}
    for s in CANDIDATES:
        fe,ft,fv,fx=backtest(
            data,s,stress_costs,rules,args.initial,enforce_hard_stop=False,
            decision_costs=costs
        )
        fm=metrics(fe,ft,fx,args.initial)
        fre,frt,frv,frx=backtest(
            data,s,stress_costs,rules,args.initial,enforce_hard_stop=True,
            decision_costs=costs
        )
        frm=metrics(fre,frt,frx,args.initial)
        frozen_shadow_map[s.name]=fm; frozen_risk_map[s.name]=frm
        frozen_rows.append({
            "strategy":s.name,
            "frozen2x_shadow_cagr":fm["cagr"],
            "frozen2x_shadow_mdd":fm["max_drawdown"],
            "frozen2x_shadow_rebalances":fm["rebalances"],
            "frozen2x_shadow_costs_paid_usd":fm["costs_paid_usd"],
            "frozen2x_shadow_hard_breached":fm["hard_breached"],
            "frozen2x_risk_hard_stopped":frm["hard_stopped"],
        })
    frozen_df=pd.DataFrame(frozen_rows)

    cand=cand.merge(stress_df,on="strategy",how="left").merge(frozen_df,on="strategy",how="left")
    cand["stress2x_gate"]=(
        cand["stress2x_shadow_mdd"].abs() <= rules.hard_drawdown
    ) & (~cand["stress2x_shadow_hard_breached"].astype(bool))
    cand["frozen2x_gate"]=(
        cand["frozen2x_shadow_mdd"].abs() <= rules.hard_drawdown
    ) & (~cand["frozen2x_shadow_hard_breached"].astype(bool))
    cand["v93_score"]=cand.apply(v93_joint_score,axis=1)
    # Primary research ranking: survive adaptive 2x cost first, then base 15%
    # MDD, then frozen-decision 2x diagnostic, then joint efficiency/turnover.
    cand=cand.sort_values(
        ["stress2x_gate","mdd_gate","frozen2x_gate","v93_score","shadow_gross_turnover_x_initial"],
        ascending=[False,False,False,False,True]
    )
    cand.to_csv(out/"candidate_summary.csv",index=False)
    stress_df.to_csv(out/"cost_stress_2x_adaptive.csv",index=False)
    frozen_df.to_csv(out/"cost_stress_2x_frozen_decision.csv",index=False)
    pd.DataFrame([s.__dict__ for s in CANDIDATES]).to_csv(out/"candidate_configs.csv",index=False)
    cand.loc[cand["mdd_gate"]].to_csv(out/"mdd15_survivor_candidates.csv",index=False)
    cand.loc[cand["stress2x_gate"]].to_csv(out/"stress2x_survivor_candidates.csv",index=False)
    cand[[
        "strategy","mdd_gate","stress2x_gate","frozen2x_gate",
        "shadow_cagr","shadow_max_drawdown","shadow_rebalances",
        "shadow_costs_paid_usd","shadow_gross_turnover_x_initial",
        "stress2x_shadow_cagr","stress2x_shadow_mdd",
        "stress2x_shadow_rebalances","v93_score"
    ]].to_csv(out/"cost_aware_candidate_frontier.csv",index=False)

    best_name=str(cand.iloc[0].strategy)
    best_s=next(s for s in CANDIDATES if s.name==best_name)
    seq,str_,sev,bsm=outputs[best_name]["shadow"]
    req,rtr,rev,brm=outputs[best_name]["risk"]
    control_name="V92C_CONTROL"
    control_shadow=outputs[control_name]["shadow"][3]
    control_risk=outputs[control_name]["risk"][3]

    seq.to_csv(out/"equity_shadow.csv")
    req.to_csv(out/"equity_risk_gated.csv")
    str_.to_csv(out/"campaigns_shadow.csv",index=False)
    sev.to_csv(out/"events_shadow.csv",index=False)
    yearly(seq).to_csv(out/"yearly_shadow.csv",index=False)
    yearly(req).to_csv(out/"yearly_risk_gated.csv",index=False)
    stress_periods(seq).to_csv(out/"stress_periods_shadow.csv",index=False)
    if "risk_state" in seq.columns:
        rs=seq.groupby("risk_state").agg(
            bars=("equity","size"),avg_exposure=("target_exposure","mean"),
            avg_abs_exposure=("target_exposure",lambda z:z.abs().mean())
        ).reset_index()
        rs["fraction"]=rs["bars"]/max(len(seq),1)
        rs.to_csv(out/"risk_state_distribution.csv",index=False)
    if "convex_stage" in seq.columns:
        cs=seq.groupby("convex_stage").agg(
            bars=("equity","size"),avg_exposure=("target_exposure","mean"),
            avg_abs_exposure=("target_exposure",lambda z:z.abs().mean())
        ).reset_index()
        cs["fraction"]=cs["bars"]/max(len(seq),1)
        cs.to_csv(out/"convex_stage_distribution.csv",index=False)

    wf=walk_forward(data,costs,rules,args.initial)
    wf.to_csv(out/"walk_forward.csv",index=False)

    delay_rows=[]; delay_shadow_map={}; delay_risk_map={}
    for s in CANDIDATES:
        de,dt,dv,dx=backtest(data,s,costs,rules,args.initial,signal_delay_bars=2,enforce_hard_stop=False)
        dm=metrics(de,dt,dx,args.initial)
        dre,drt,drv,drx=backtest(data,s,costs,rules,args.initial,signal_delay_bars=2,enforce_hard_stop=True)
        drm=metrics(dre,drt,drx,args.initial)
        delay_shadow_map[s.name]=dm; delay_risk_map[s.name]=drm
        delay_rows.append({
            "strategy":s.name,"shadow_cagr":dm["cagr"],
            "shadow_mdd":dm["max_drawdown"],"shadow_sharpe":dm["sharpe_365"],
            "shadow_rebalances":dm["rebalances"],
            "shadow_costs_paid_usd":dm["costs_paid_usd"],
            "risk_cagr":drm["cagr"],"risk_mdd":drm["max_drawdown"],
            "risk_hard_stopped":drm["hard_stopped"],
        })
    pd.DataFrame(delay_rows).to_csv(out/"execution_delay_stress.csv",index=False)

    # Attribution.
    ce,ct,cv,cx=backtest(data,best_s,costs,rules,args.initial,tactical_enabled=False,enforce_hard_stop=False)
    core_m=metrics(ce,ct,cx,args.initial)
    pd.DataFrame([
        {"variant":"selected V9.3 cost-aware execution",**bsm},
        {"variant":"V9.2C exact control",**control_shadow},
        {"variant":"V9.3 selected core-only",**core_m},
        {"variant":"V9.3 selected adaptive 2x cost",**stress_shadow_map[best_name]},
        {"variant":"V9.3 selected frozen-decision 2x cost",**frozen_shadow_map[best_name]},
    ]).to_csv(out/"attribution.csv",index=False)

    robust=robustness_grid(data,best_s,costs,rules,args.initial)
    robust.to_csv(out/"robustness_grid.csv",index=False)

    bh=benchmark_buyhold(data,args.initial)
    sma=benchmark_sma200(data,args.initial)
    gates=acceptance(
        bsm,brm,wf,
        stress_shadow_map[best_name],stress_risk_map[best_name],
        delay_shadow_map[best_name],delay_risk_map[best_name],robust
    )
    gates["frozen2x_risk_not_hard_stopped"] = bool(
        not frozen_risk_map[best_name]["hard_stopped"]
    )
    gates["frozen2x_shadow_mdd_le_15pct"] = bool(
        abs(frozen_shadow_map[best_name]["max_drawdown"]) <= 0.15
    )

    krw=pd.DataFrame(index=seq.index)
    krw["shadow_equity_krw"]=10_000_000*(seq.equity/args.initial)
    krw["risk_gated_equity_krw"]=10_000_000*(req.equity/args.initial)
    krw.to_csv(out/"krw_10m_projection.csv")

    primary_pass=(
        gates["full_shadow_cagr_ge_25pct"] and
        gates["full_shadow_mdd_le_15pct"] and
        gates["full_shadow_sharpe_ge_1_3"] and
        gates["full_shadow_pf_daily_ge_1_4"] and
        gates["full_risk_policy_not_hard_stopped"] and
        gates["stress_2x_risk_not_hard_stopped"] and
        gates["delay_risk_not_hard_stopped"] and
        gates["oos_hard_stop_window_ratio"] == 0
    )
    v93_dev_pass=(
        bsm["cagr"] >= 0.15 and abs(bsm["max_drawdown"]) <= 0.15 and
        not stress_risk_map[best_name]["hard_stopped"] and
        abs(stress_shadow_map[best_name]["max_drawdown"]) <= 0.15 and
        not frozen_risk_map[best_name]["hard_stopped"] and
        bsm["rebalances"] <= 0.85*control_shadow["rebalances"] and
        bsm["costs_paid_usd"] <= 0.85*control_shadow["costs_paid_usd"]
    )

    summary={
        "version":"V9.3 Cost-Aware Execution Architecture",
        "data_start":str(data.index.min()),"data_end":str(data.index.max()),
        "rows_1h":len(data),"selected_candidate":best_name,
        "selection_method":"adaptive 2x-cost survival first; base 15% MDD second; frozen-decision 2x survival third; then joint return/risk score and lower turnover",
        "selected_shadow_metrics":bsm,
        "selected_risk_gated_metrics":brm,
        "v92c_control_shadow_metrics":control_shadow,
        "adaptive_2x_shadow_metrics":stress_shadow_map[best_name],
        "adaptive_2x_risk_metrics":stress_risk_map[best_name],
        "frozen_decision_2x_shadow_metrics":frozen_shadow_map[best_name],
        "frozen_decision_2x_risk_metrics":frozen_risk_map[best_name],
        "turnover_change_vs_v92c_control":{
            "rebalance_delta":int(bsm["rebalances"]-control_shadow["rebalances"]),
            "rebalance_pct":float(bsm["rebalances"]/max(control_shadow["rebalances"],1)-1),
            "cost_delta_usd":float(bsm["costs_paid_usd"]-control_shadow["costs_paid_usd"]),
            "cost_pct":float(bsm["costs_paid_usd"]/max(control_shadow["costs_paid_usd"],1e-12)-1),
            "turnover_delta_x_initial":float(bsm["gross_turnover_x_initial"]-control_shadow["gross_turnover_x_initial"]),
        },
        "v93_development_pass":bool(v93_dev_pass),
        "development_target":{
            "base_cagr_ge_15pct":bool(bsm["cagr"]>=0.15),
            "base_mdd_le_15pct":bool(abs(bsm["max_drawdown"])<=0.15),
            "adaptive_2x_survives":bool(not stress_risk_map[best_name]["hard_stopped"] and abs(stress_shadow_map[best_name]["max_drawdown"])<=0.15),
            "frozen_2x_survives":bool(not frozen_risk_map[best_name]["hard_stopped"]),
            "rebalances_minus_15pct":bool(bsm["rebalances"]<=0.85*control_shadow["rebalances"]),
            "costs_minus_15pct":bool(bsm["costs_paid_usd"]<=0.85*control_shadow["costs_paid_usd"]),
        },
        "buy_hold":bh,"sma200_long_cash":sma,"acceptance":gates,
        "overall_pass":bool(primary_pass),
        "research_note":"V9.3 candidates were designed after observing V9.2 full-sample and cost-stress results. Walk-forward windows are stability checks, not pristine untouched OOS for the research program.",
        "assumptions":{
            "price_source":"Binance BTCUSDT spot 1H price proxy",
            "fee_bps_per_rebalance":costs.fee_bps,
            "slippage_bps_per_rebalance":costs.slippage_bps,
            "funding_included":False,
            "signal_timing":"completed daily/4H data -> later 4H open",
            "cost_aware_rule":"effective deadband = static deadband + cost_band_mult*(estimated one-way cost / 4H ATR fraction), capped at 14 percentage points; urgent de-risk bypasses throttle",
            "martingale":False,"averaging_down":False,"simultaneous_hedge":False,
        }
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str))

    lines=[
        "# BTC AI EA V9.3 — Cost-Aware Execution Architecture","",
        f"- Data: {summary['data_start']} → {summary['data_end']} ({len(data):,} 1H bars)",
        f"- Selected candidate: **{best_name}**",
        f"- Base CAGR / MDD: **{pct(bsm['cagr'])} / {pct(bsm['max_drawdown'])}**",
        f"- Base Sharpe / PF: **{bsm['sharpe_365']:.2f} / {bsm['profit_factor_daily']:.2f}**",
        f"- Rebalances / costs / turnover: **{bsm['rebalances']} / ${bsm['costs_paid_usd']:,.2f} / {bsm['gross_turnover_x_initial']:.2f}x**",
        f"- Avg effective deadband: **{pct(bsm.get('avg_effective_deadband',0.0))}**",
        f"- Cost-band skips / interval blocks / tactical re-entry blocks: **{bsm.get('cost_band_skips',0)} / {bsm.get('interval_blocks',0)} / {bsm.get('tactical_reentry_blocks',0)}**",
        f"- V9.2C control CAGR / MDD: **{pct(control_shadow['cagr'])} / {pct(control_shadow['max_drawdown'])}**",
        f"- V9.2C control rebalances / costs: **{control_shadow['rebalances']} / ${control_shadow['costs_paid_usd']:,.2f}**",
        f"- Adaptive 2x-cost CAGR / MDD: **{pct(stress_shadow_map[best_name]['cagr'])} / {pct(stress_shadow_map[best_name]['max_drawdown'])}**",
        f"- Adaptive 2x-cost hard stop: **{stress_risk_map[best_name]['hard_stopped']}**",
        f"- Frozen-decision 2x-cost CAGR / MDD: **{pct(frozen_shadow_map[best_name]['cagr'])} / {pct(frozen_shadow_map[best_name]['max_drawdown'])}**",
        f"- Frozen-decision 2x hard stop: **{frozen_risk_map[best_name]['hard_stopped']}**",
        f"- V9.3 development target: **{'PASS' if v93_dev_pass else 'FAIL'}**",
        f"- Overall strict acceptance: **{'PASS' if primary_pass else 'FAIL'}**",
        "","## Candidate comparison","",cand.to_markdown(index=False),
        "","## Walk-forward OOS robustness","",wf.to_markdown(index=False) if not wf.empty else "Not enough windows.",
        "","## Execution nearby robustness","",robust.to_markdown(index=False),
        "","## Acceptance gates","","```json",json.dumps(gates,indent=2,default=str),"```",
        "","## Research integrity note","",summary["research_note"],
        "","## Decision rule","",
        "Do not deploy live unless base MDD remains below 15%, both adaptive and frozen-decision 2x-cost tests survive, +4H delay survives, OOS windows avoid hard stops, and later futures/funding-aware validation plus paper trading pass."
    ]
    (out/"REPORT.md").write_text("\n".join(lines))
    print("\n=== V9.3 COMPLETE ===")
    print((out/"REPORT.md").read_text())


if __name__ == "__main__":
    main()
