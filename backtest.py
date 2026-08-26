#!/usr/bin/env python3
"""
BTC AI EA — V5 Core + Tactical Trend Backtester
================================================

Why V5 exists
-------------
V4 reduced turnover and drawdown, but captured too little of BTC's secular upside.
V5 deliberately separates the portfolio into:

  1) CORE: slow daily trend exposure designed to stay invested through major bull legs.
  2) TACTICAL: smaller 4H breakout sleeve that adds only when trend accelerates.

The design is asymmetric by construction:
- long exposure is the primary return engine;
- shorts are small and allowed only in strong daily bear regimes;
- no martingale, no averaging down, no simultaneous long/short hedge.

Anti-lookahead rules
--------------------
- Daily regime uses only a fully completed prior daily candle.
- 4H tactical signals use only a fully completed 4H candle.
- Position changes execute at a later 4H OPEN.

Phase-1 price source
--------------------
Binance BTCUSDT spot 1H public archives are used as a long-history PRICE proxy.
Funding, basis, liquidation and perpetual-specific microstructure are intentionally
reserved for the later futures-validation phase.
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
    strong_short: float
    tactical_long: float
    tactical_short: float
    breakout_4h: int
    exit_4h: int
    trail_atr_4h: float
    breakout_buffer_atr: float
    vol_target: float
    vol_floor_scale: float
    max_long: float
    max_short: float
    shock_scale: float


CANDIDATES = [
    Strategy(
        "V5A", fast_days=100, slow_days=200, slope_days=20,
        strong_long=0.85, weak_long=0.55, strong_short=-0.15,
        tactical_long=0.30, tactical_short=-0.10,
        breakout_4h=30, exit_4h=15, trail_atr_4h=4.0,
        breakout_buffer_atr=0.05, vol_target=0.65, vol_floor_scale=0.65,
        max_long=1.20, max_short=0.25, shock_scale=0.50,
    ),
    Strategy(
        "V5B", fast_days=50, slow_days=200, slope_days=20,
        strong_long=1.00, weak_long=0.65, strong_short=-0.10,
        tactical_long=0.25, tactical_short=-0.08,
        breakout_4h=55, exit_4h=20, trail_atr_4h=5.0,
        breakout_buffer_atr=0.05, vol_target=0.70, vol_floor_scale=0.70,
        max_long=1.25, max_short=0.20, shock_scale=0.55,
    ),
    Strategy(
        "V5C", fast_days=100, slow_days=250, slope_days=30,
        strong_long=0.80, weak_long=0.40, strong_short=-0.20,
        tactical_long=0.30, tactical_short=-0.10,
        breakout_4h=40, exit_4h=20, trail_atr_4h=4.5,
        breakout_buffer_atr=0.08, vol_target=0.55, vol_floor_scale=0.55,
        max_long=1.10, max_short=0.30, shock_scale=0.45,
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
    req = urllib.request.Request(url, headers={"User-Agent": "btc-ai-ea-v5/1.0"})
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
    # 4H bars are labeled at bar OPEN. State changes from the completed bar are
    # applied to a later bar open in backtest().
    h4 = df1h[["open", "high", "low", "close", "volume"]].resample(
        "4h", closed="left", label="left"
    ).agg({"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"}).dropna()
    h4["atr4h"] = atr(h4, 14)
    h4["entry_hi"] = h4["high"].shift(1).rolling(s.breakout_4h).max()
    h4["entry_lo"] = h4["low"].shift(1).rolling(s.breakout_4h).min()
    h4["exit_hi"] = h4["high"].shift(1).rolling(s.exit_4h).max()
    h4["exit_lo"] = h4["low"].shift(1).rolling(s.exit_4h).min()

    # Daily bar labeled at NEXT midnight, meaning all daily features are known at
    # that timestamp and can safely be merge_asof'ed backward onto 4H bars.
    d = df1h[["open", "high", "low", "close", "volume"]].resample(
        "1D", closed="left", label="right"
    ).agg({"open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"}).dropna()
    d["fast"] = d["close"].ewm(span=s.fast_days, adjust=False, min_periods=s.fast_days).mean()
    d["slow"] = d["close"].ewm(span=s.slow_days, adjust=False, min_periods=s.slow_days).mean()
    d["slow_slope"] = d["slow"] - d["slow"].shift(s.slope_days)
    r = d["close"].pct_change()
    d["rv30"] = r.rolling(30, min_periods=20).std() * np.sqrt(365)
    d["atr_pct"] = atr(d, 14) / d["close"]
    d["shock_cut"] = d["atr_pct"].rolling(540, min_periods=180).quantile(0.95)
    d["shock"] = (
        (d["atr_pct"] > d["shock_cut"]) |
        (r.abs() > 3.0 * r.rolling(60, min_periods=30).std())
    ).fillna(False)

    strong_bull = (d["close"] > d["slow"]) & (d["fast"] > d["slow"]) & (d["slow_slope"] > 0)
    weak_bull = (d["close"] > d["slow"]) & ~strong_bull
    strong_bear = (d["close"] < d["slow"]) & (d["fast"] < d["slow"]) & (d["slow_slope"] < 0)
    weak_bear = (d["close"] < d["slow"]) & ~strong_bear

    d["regime"] = "NEUTRAL"
    d.loc[weak_bull, "regime"] = "BULL_WEAK"
    d.loc[strong_bull, "regime"] = "BULL_STRONG"
    d.loc[weak_bear, "regime"] = "BEAR_WEAK"
    d.loc[strong_bear, "regime"] = "BEAR_STRONG"

    left = h4.reset_index().rename(columns={h4.index.name or "index":"time"})
    right = d[["regime", "rv30", "shock", "fast", "slow", "slow_slope"]].reset_index()
    right = right.rename(columns={right.columns[0]:"time"})
    x = pd.merge_asof(left.sort_values("time"), right.sort_values("time"), on="time", direction="backward")
    return x.set_index("time")


def week_key(ts: pd.Timestamp):
    i = ts.isocalendar()
    return (int(i.year), int(i.week))


def core_exposure(regime: str, s: Strategy) -> float:
    if regime == "BULL_STRONG":
        return s.strong_long
    if regime == "BULL_WEAK":
        return s.weak_long
    if regime == "BEAR_STRONG":
        return s.strong_short
    return 0.0


def vol_scale(rv: float, s: Strategy) -> float:
    if not np.isfinite(rv) or rv <= 0:
        return 1.0
    return float(np.clip(s.vol_target / rv, s.vol_floor_scale, 1.0))


def target_exposure(regime: str, rv: float, shock: bool, tactical_side: int,
                    s: Strategy, soft_dd: bool) -> float:
    scale = vol_scale(rv, s)
    core = core_exposure(regime, s) * scale
    tactical = 0.0
    if tactical_side > 0 and regime in ("BULL_STRONG", "BULL_WEAK"):
        tactical = s.tactical_long * scale
    elif tactical_side < 0 and regime == "BEAR_STRONG":
        tactical = s.tactical_short * scale

    exp = core + tactical
    if shock:
        exp *= s.shock_scale
    if soft_dd:
        # Cut only the incremental risk; do not instantly dump the secular core.
        exp *= 0.70
    return float(np.clip(exp, -s.max_short, s.max_long))


def trade_cost(delta_notional: float, costs: CostModel) -> float:
    # Fee plus an explicit slippage penalty. Both are charged on every rebalance.
    return abs(delta_notional) * (costs.fee_bps + costs.slippage_bps) / 10_000.0


def backtest(df1h: pd.DataFrame, s: Strategy, costs: CostModel, rules: RiskRules,
             initial: float = 10_000.0, signal_delay_bars: int = 1,
             tactical_enabled: bool = True, shorts_enabled: bool = True):
    x = build_features(df1h, s).dropna(subset=["rv30", "atr4h", "regime"])
    if len(x) < 1500:
        raise RuntimeError("Insufficient usable 4H history")

    equity = initial
    qty = 0.0  # signed derivative BTC quantity
    prev_close = None
    peak_equity = initial
    current_day = x.index[0].date()
    current_week = week_key(x.index[0])
    day_start = initial
    week_start = initial
    day_lock = False
    week_lock = False
    hard_stopped = False

    tactical_side = 0
    tactical_peak = np.nan
    pending_tactical = None  # tuple(action, side, bars_remaining)

    total_costs = 0.0
    rebalance_count = 0
    bars_exposed = 0
    rows = []
    trades = []
    events = []
    last_exposure = 0.0
    last_regime = None

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
        shock = bool(r.shock)

        # Mark previous close -> current open before any rebalance.
        if prev_close is not None:
            equity += qty * (o - prev_close)

        # Reset day/week locks on the fresh marked-to-market open.
        if ts.date() != current_day:
            current_day = ts.date()
            day_start = equity
            day_lock = False
        w = week_key(ts)
        if w != current_week:
            current_week = w
            week_start = equity
            week_lock = False

        peak_equity = max(peak_equity, equity)
        dd_open = equity / peak_equity - 1.0
        soft_dd = dd_open <= -rules.soft_drawdown

        # Apply a tactical state change that was generated by a COMPLETED prior 4H bar.
        if pending_tactical is not None:
            action, pside, remaining = pending_tactical
            remaining -= 1
            if remaining <= 0:
                if action == "ENTER":
                    tactical_side = pside
                    tactical_peak = o
                else:
                    tactical_side = 0
                    tactical_peak = np.nan
                pending_tactical = None
            else:
                pending_tactical = (action, pside, remaining)

        if not shorts_enabled and tactical_side < 0:
            tactical_side = 0
            tactical_peak = np.nan

        # Locks / hard stop override all model exposures.
        desired = 0.0
        reason = "MODEL"
        if hard_stopped or day_lock or week_lock:
            desired = 0.0
            reason = "LOCK"
        else:
            desired = target_exposure(regime, rv, shock,
                                      tactical_side if tactical_enabled else 0,
                                      s, soft_dd)
            if not shorts_enabled:
                desired = max(0.0, desired)

        # Rebalance whenever exposure target materially changes or daily regime changes.
        if (abs(desired - last_exposure) >= 0.025 or regime != last_regime or
                (desired == 0 and abs(last_exposure) > 1e-12)):
            set_position(ts, o, desired, reason)
        last_regime = regime

        if abs(qty) > 0:
            bars_exposed += 1

        # Mark current 4H open -> close. This deliberately avoids using intrabar
        # highs/lows for portfolio PnL while still allowing them to update a future
        # tactical trailing reference.
        equity += qty * (c - o)
        peak_equity = max(peak_equity, equity)
        dd = equity / peak_equity - 1.0

        # Risk locks are evaluated after the completed 4H candle and flatten on
        # the next bar open, avoiding same-bar hindsight fills.
        if day_start > 0 and equity / day_start - 1 <= -rules.daily_loss_lock:
            day_lock = True
        if week_start > 0 and equity / week_start - 1 <= -rules.weekly_loss_lock:
            week_lock = True
        if dd <= -rules.hard_drawdown:
            hard_stopped = True

        # Generate tactical transition for a FUTURE 4H open.
        if tactical_enabled and not hard_stopped:
            buf = s.breakout_buffer_atr * a
            if tactical_side == 0 and pending_tactical is None:
                if regime in ("BULL_STRONG", "BULL_WEAK") and c > float(r.entry_hi) + buf:
                    pending_tactical = ("ENTER", +1, max(1, signal_delay_bars))
                elif shorts_enabled and regime == "BEAR_STRONG" and c < float(r.entry_lo) - buf:
                    pending_tactical = ("ENTER", -1, max(1, signal_delay_bars))
            elif tactical_side > 0:
                tactical_peak = max(tactical_peak, h) if np.isfinite(tactical_peak) else h
                trail = tactical_peak - s.trail_atr_4h * a
                if (regime not in ("BULL_STRONG", "BULL_WEAK") or
                        c < float(r.exit_lo) or c < trail):
                    if pending_tactical is None:
                        pending_tactical = ("EXIT", +1, max(1, signal_delay_bars))
            elif tactical_side < 0:
                tactical_peak = min(tactical_peak, l) if np.isfinite(tactical_peak) else l
                trail = tactical_peak + s.trail_atr_4h * a
                if regime != "BEAR_STRONG" or c > float(r.exit_hi) or c > trail:
                    if pending_tactical is None:
                        pending_tactical = ("EXIT", -1, max(1, signal_delay_bars))

        rows.append({
            "time":ts, "equity":equity, "qty":qty,
            "target_exposure":last_exposure, "regime":regime,
            "tactical_side":tactical_side, "drawdown":dd,
            "day_lock":day_lock, "week_lock":week_lock,
        })
        prev_close = c

    eqdf = pd.DataFrame(rows).set_index("time")
    # Flat-at-end accounting cost, so reported final equity is realizable.
    if abs(qty) > 0 and not eqdf.empty:
        px = float(x.iloc[-1].close)
        cost = trade_cost(qty * px, costs)
        equity -= cost
        total_costs += cost
        eqdf.iloc[-1, eqdf.columns.get_loc("equity")] = equity
        eqdf.iloc[-1, eqdf.columns.get_loc("qty")] = 0.0
        eqdf.iloc[-1, eqdf.columns.get_loc("target_exposure")] = 0.0

    # Create exposure-regime "campaigns" from contiguous non-zero exposure periods.
    e = eqdf.copy()
    active = e.target_exposure.abs() > 1e-9
    starts = active & ~active.shift(1, fill_value=False)
    ends = active & ~active.shift(-1, fill_value=False)
    start_times = list(e.index[starts])
    end_times = list(e.index[ends])
    for st, en in zip(start_times, end_times):
        g = e.loc[st:en]
        p0 = float(g.equity.iloc[0]); p1 = float(g.equity.iloc[-1])
        avgexp = float(g.target_exposure.mean())
        trades.append({
            "entry_time":st, "exit_time":en,
            "direction":"LONG" if avgexp > 0 else "SHORT",
            "avg_exposure":avgexp,
            "return":p1/p0 - 1 if p0 > 0 else np.nan,
        })

    trdf = pd.DataFrame(trades)
    evdf = pd.DataFrame(events)
    extra = {
        "costs":total_costs, "hard_stopped":hard_stopped,
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
    }


def objective(m):
    if not m or m.get("hard_stopped"):
        return -1e9
    sh = m.get("sharpe_365", -1)
    if not np.isfinite(sh):
        sh = -1
    # V5 intentionally prioritizes CAGR while keeping a strong DD penalty.
    return m["cagr"] - 1.10*abs(m["max_drawdown"]) + 0.06*sh


def run_candidates(data, costs, rules, initial, delay=1):
    rows=[]; outputs={}
    for s in CANDIDATES:
        print(f"[TEST] {s.name}", flush=True)
        eq,tr,ev,ex = backtest(data,s,costs,rules,initial,delay)
        m=metrics(eq,tr,ex,initial)
        rows.append({"strategy":s.name, **m, "score":objective(m)})
        outputs[s.name]=(eq,tr,ev,m)
    return pd.DataFrame(rows).sort_values("score",ascending=False), outputs


def walk_forward(data, costs, rules, initial):
    # 3-year train, 1-year OOS, rolled yearly.
    start=data.index.min().normalize(); end=data.index.max().normalize()
    rows=[]; anchor=start
    while anchor + pd.DateOffset(years=4) <= end + pd.Timedelta(days=1):
        train_end=anchor+pd.DateOffset(years=3)-pd.Timedelta(hours=1)
        test_start=train_end+pd.Timedelta(hours=1)
        test_end=anchor+pd.DateOffset(years=4)-pd.Timedelta(hours=1)
        train=data.loc[(data.index>=anchor)&(data.index<=train_end)]
        test=data.loc[(data.index>=test_start)&(data.index<=test_end)]
        if len(train)<15000 or len(test)<4000:
            anchor += pd.DateOffset(years=1); continue
        scored=[]
        for s in CANDIDATES:
            eq,tr,ev,ex=backtest(train,s,costs,rules,initial)
            m=metrics(eq,tr,ex,initial)
            scored.append((objective(m),s,m))
        scored.sort(key=lambda z:z[0], reverse=True)
        _, chosen, tm = scored[0]
        eq,tr,ev,ex=backtest(test,chosen,costs,rules,initial)
        om=metrics(eq,tr,ex,initial)
        rows.append({
            "train_start":anchor,"train_end":train_end,
            "test_start":test_start,"test_end":test_end,
            "chosen":chosen.name,
            "train_cagr":tm["cagr"],"train_mdd":tm["max_drawdown"],
            "test_return":om["total_return"],"test_cagr":om["cagr"],
            "test_mdd":om["max_drawdown"],"test_sharpe":om["sharpe_365"],
            "test_pf_daily":om["profit_factor_daily"],
            "test_rebalances":om["rebalances"],
            "test_hard_stopped":om["hard_stopped"],
        })
        anchor += pd.DateOffset(years=1)
    return pd.DataFrame(rows)


def robustness_grid(data, best: Strategy, costs, rules, initial):
    rows=[]
    for slow_mult in (0.8,1.0,1.2):
        for breakout_mult in (0.8,1.0,1.2):
            s=replace(
                best,
                name=f"{best.name}_S{slow_mult:.1f}_B{breakout_mult:.1f}",
                slow_days=max(best.fast_days+20, int(round(best.slow_days*slow_mult))),
                breakout_4h=max(12, int(round(best.breakout_4h*breakout_mult))),
            )
            eq,tr,ev,ex=backtest(data,s,costs,rules,initial)
            m=metrics(eq,tr,ex,initial)
            rows.append({
                "slow_mult":slow_mult,"breakout_mult":breakout_mult,
                "slow_days":s.slow_days,"breakout_4h":s.breakout_4h,
                "cagr":m["cagr"],"mdd":m["max_drawdown"],
                "sharpe":m["sharpe_365"],"pf_daily":m["profit_factor_daily"],
                "hard_stopped":m["hard_stopped"],
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


def acceptance(best_m, wf, stress_m, delay_m, robust):
    if not wf.empty:
        oos_total=float((1+wf.test_return).prod()-1)
        years=len(wf)
        oos_cagr=float((1+oos_total)**(1/years)-1) if 1+oos_total>0 else -1.0
        pos_ratio=float((wf.test_return>0).mean())
        worst_oos_mdd=float(wf.test_mdd.min())
    else:
        oos_cagr=np.nan; pos_ratio=np.nan; worst_oos_mdd=np.nan
    return {
        "full_cagr_ge_25pct":bool(best_m["cagr"]>=0.25),
        "full_mdd_le_15pct":bool(abs(best_m["max_drawdown"])<=0.15),
        "full_sharpe_ge_1_3":bool(best_m["sharpe_365"]>=1.3),
        "full_pf_daily_ge_1_4":bool(best_m["profit_factor_daily"]>=1.4),
        "oos_compound_cagr":oos_cagr,
        "oos_positive_window_ratio":pos_ratio,
        "oos_worst_window_mdd":worst_oos_mdd,
        "stress_2x_positive_cagr":bool(stress_m["cagr"]>0),
        "stress_2x_not_hard_stopped":bool(not stress_m["hard_stopped"]),
        "delay_positive_cagr":bool(delay_m["cagr"]>0),
        "robust_positive_cagr_ratio":float((robust.cagr>0).mean()),
        "robust_all_survive_hard_stop":bool((~robust.hard_stopped).all()),
    }


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
    eq,tr,ev,bm=outputs[best_name]
    eq.to_csv(out/"equity.csv")
    tr.to_csv(out/"campaigns.csv",index=False)
    ev.to_csv(out/"events.csv",index=False)
    yearly(eq).to_csv(out/"yearly.csv",index=False)

    wf=walk_forward(data,costs,rules,args.initial)
    wf.to_csv(out/"walk_forward.csv",index=False)

    # 2x transaction-cost stress.
    stress_costs=CostModel(args.fee_bps*2,args.slippage_bps*2)
    stress_rows=[]; stress_map={}
    for s in CANDIDATES:
        e,t,v,x=backtest(data,s,stress_costs,rules,args.initial)
        m=metrics(e,t,x,args.initial)
        stress_rows.append({"strategy":s.name,**m}); stress_map[s.name]=m
    pd.DataFrame(stress_rows).to_csv(out/"cost_stress_2x.csv",index=False)

    # One additional 4H bar of signal latency.
    delay_rows=[]; delay_map={}
    for s in CANDIDATES:
        e,t,v,x=backtest(data,s,costs,rules,args.initial,signal_delay_bars=2)
        m=metrics(e,t,x,args.initial)
        delay_rows.append({"strategy":s.name,**m}); delay_map[s.name]=m
    pd.DataFrame(delay_rows).to_csv(out/"execution_delay_stress.csv",index=False)

    # Core-only and no-short attribution for the selected candidate.
    ecore,tcore,vcore,xcore=backtest(data,best_s,costs,rules,args.initial,tactical_enabled=False)
    mcore=metrics(ecore,tcore,xcore,args.initial)
    enoshort,tnoshort,vnoshort,xnoshort=backtest(data,best_s,costs,rules,args.initial,shorts_enabled=False)
    mnoshort=metrics(enoshort,tnoshort,xnoshort,args.initial)
    pd.DataFrame([
        {"variant":"V5 full",**bm},
        {"variant":"core only",**mcore},
        {"variant":"no shorts",**mnoshort},
    ]).to_csv(out/"attribution.csv",index=False)

    robust=robustness_grid(data,best_s,costs,rules,args.initial)
    robust.to_csv(out/"robustness_grid.csv",index=False)

    bh=benchmark_buyhold(data,args.initial)
    sma=benchmark_sma200(data,args.initial)
    gates=acceptance(bm,wf,stress_map[best_name],delay_map[best_name],robust)

    krw=pd.DataFrame(index=eq.index)
    krw["strategy_equity_krw"]=10_000_000*(eq.equity/args.initial)
    krw.to_csv(out/"krw_10m_projection.csv")

    summary={
        "version":"V5 Core + Tactical",
        "data_start":str(data.index.min()),"data_end":str(data.index.max()),
        "rows_1h":len(data),"best_full_sample":best_name,
        "best_metrics":bm,"core_only_metrics":mcore,"no_short_metrics":mnoshort,
        "buy_hold":bh,"sma200_long_cash":sma,"acceptance":gates,
        "assumptions":{
            "price_source":"Binance BTCUSDT spot 1H price proxy",
            "fee_bps_per_rebalance":costs.fee_bps,
            "slippage_bps_per_rebalance":costs.slippage_bps,
            "funding_included":False,
            "signal_timing":"completed daily/4H data -> later 4H open",
            "martingale":False,"averaging_down":False,"simultaneous_hedge":False,
        }
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str))

    lines=[
        "# BTC AI EA V5 — Core + Tactical Backtest Report","",
        f"- Data: {summary['data_start']} → {summary['data_end']} ({len(data):,} 1H bars)",
        f"- Best full-sample candidate: **{best_name}**",
        f"- CAGR: **{pct(bm['cagr'])}**",
        f"- Total return: **{pct(bm['total_return'])}**",
        f"- Max drawdown: **{pct(bm['max_drawdown'])}**",
        f"- Sharpe: **{bm['sharpe_365']:.2f}**",
        f"- Daily profit factor: **{bm['profit_factor_daily']:.2f}**",
        f"- Rebalances: **{bm['rebalances']}**",
        f"- Costs: **${bm['costs_paid_usd']:.2f}**",
        f"- Core-only CAGR: **{pct(mcore['cagr'])}**",
        f"- No-short CAGR: **{pct(mnoshort['cagr'])}**",
        f"- Buy & hold CAGR / MDD: **{pct(bh['cagr'])} / {pct(bh['max_drawdown'])}**",
        f"- 200D long/cash CAGR / MDD: **{pct(sma['cagr'])} / {pct(sma['max_drawdown'])}**",
        "","## Candidate comparison","",cand.to_markdown(index=False),
        "","## Walk-forward OOS","",wf.to_markdown(index=False) if not wf.empty else "Not enough windows.",
        "","## Attribution","",
        pd.DataFrame([
            {"variant":"V5 full",**bm},
            {"variant":"core only",**mcore},
            {"variant":"no shorts",**mnoshort},
        ]).to_markdown(index=False),
        "","## Acceptance gates","","```json",json.dumps(gates,indent=2,default=str),"```",
        "","## Decision rule","",
        "V5 is not live-ready unless OOS, 2x-cost, latency and nearby-parameter tests remain acceptable. "
        "Even a strong backtest must still pass perpetual-futures/funding validation and paper trading."
    ]
    (out/"REPORT.md").write_text("\n".join(lines))
    print("\n=== V5 COMPLETE ===")
    print((out/"REPORT.md").read_text())


if __name__ == "__main__":
    main()
