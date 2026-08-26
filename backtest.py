#!/usr/bin/env python3
"""
BTC AI Trend EA Research Backtester
===================================
Purpose
-------
Research-grade first-pass validation for:
4H Regime -> 1H breakout entry -> anti-martingale -> ATR stop/trailing
-> risk controls -> fees/slippage -> walk-forward selection.

Important
---------
- This is research code, not a profit guarantee.
- Price data: Binance public BTCUSDT spot klines (1h), used as a long-history
  price proxy. Bybit funding/basis are intentionally NOT included in Phase 1.
- Trading-cost defaults model Bybit VIP-0 perpetual taker trading fees as a
  configurable 5.5 bps per fill plus 2 bps slippage per fill.
- All signals use completed candles. Entries happen no earlier than the next
  1-hour bar open to avoid look-ahead bias.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BINANCE_VISION = "https://data.binance.vision/data/spot"
KLINE_COLS = [
    "open_time","open","high","low","close","volume","close_time",
    "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
]

@dataclass(frozen=True)
class Strategy:
    name: str
    breakout_lookback: int
    adx_threshold: float
    stop_atr: float
    trail_atr: float
    ema_fast: int = 50
    ema_slow: int = 200
    slope_bars: int = 6
    shock_quantile: float = 0.95
    risk_pct: float = 0.005
    add1_atr: float = 0.5
    add2_atr: float = 1.0

CANDIDATES = [
    Strategy("V1", 20, 20.0, 2.0, 3.0),
    Strategy("V2", 30, 22.0, 2.2, 3.2),
    Strategy("V3", 55, 18.0, 2.5, 4.0),
]

@dataclass
class CostModel:
    fee_bps: float = 5.5
    slippage_bps: float = 2.0

@dataclass
class RiskRules:
    notional_cap: float = 1.0
    daily_loss_lock: float = 0.02
    weekly_loss_lock: float = 0.05
    soft_dd: float = 0.10
    hard_dd: float = 0.15

def month_starts(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    last = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= last:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)

def _fetch_bytes(url: str, retries: int = 3) -> bytes:
    last = None
    req = urllib.request.Request(url, headers={"User-Agent": "btc-ai-ea/1.0"})
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
        raise ValueError("Unexpected Binance kline file")
    df = df.iloc[:, :12]
    df.columns = KLINE_COLS[:df.shape[1]]
    # Binance Vision has used millisecond and (for some archives) microsecond
    # epoch precision. Detect dynamically.
    t = pd.to_numeric(df["open_time"], errors="coerce")
    unit = "us" if t.dropna().median() > 1e14 else "ms"
    idx = pd.to_datetime(t, unit=unit, utc=True, errors="coerce")
    out = pd.DataFrame(index=idx)
    for c in ["open","high","low","close","volume"]:
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    return out.dropna().sort_index()

def download_btc_1h(start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    last_complete_month = pd.Timestamp(now.year, now.month, 1, tz="UTC") - pd.Timedelta(hours=1)
    frames = []

    # Monthly archives for all complete months.
    month_end = min(end_ts, last_complete_month)
    for m in month_starts(start_ts, month_end):
        ym = m.strftime("%Y-%m")
        cache_zip = cache_dir / f"BTCUSDT-1h-{ym}.zip"
        url = f"{BINANCE_VISION}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{ym}.zip"
        if not cache_zip.exists():
            try:
                cache_zip.write_bytes(_fetch_bytes(url))
                print(f"[DATA] downloaded {ym}", flush=True)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"[DATA] missing {ym}; skipped", flush=True)
                    continue
                raise
        with zipfile.ZipFile(cache_zip) as z:
            csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
            frames.append(_parse_kline_csv(z.read(csv_name)))

    # Current month daily archives, up to yesterday/T-1.
    current_month_start = pd.Timestamp(now.year, now.month, 1, tz="UTC")
    if end_ts >= current_month_start and end_ts >= start_ts:
        d0 = max(start_ts.normalize(), current_month_start)
        d1 = min(end_ts.normalize(), (now - pd.Timedelta(days=1)).normalize())
        for d in pd.date_range(d0, d1, freq="1D", tz="UTC"):
            ds = d.strftime("%Y-%m-%d")
            cache_zip = cache_dir / f"BTCUSDT-1h-{ds}.zip"
            url = f"{BINANCE_VISION}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{ds}.zip"
            if not cache_zip.exists():
                try:
                    cache_zip.write_bytes(_fetch_bytes(url))
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        continue
                    raise
            with zipfile.ZipFile(cache_zip) as z:
                csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
                frames.append(_parse_kline_csv(z.read(csv_name)))

    if not frames:
        raise RuntimeError("No Binance data downloaded.")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    # Require OHLC integrity.
    bad = (df["high"] < df[["open","close","low"]].max(axis=1)) | \
          (df["low"] > df[["open","close","high"]].min(axis=1))
    if bad.any():
        raise RuntimeError(f"OHLC integrity failure on {int(bad.sum())} rows")
    print(f"[DATA] rows={len(df):,} from {df.index.min()} to {df.index.max()}")
    return df

def wilder_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return wilder_ema(tr, n)

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, n)
    plus_di = 100 * wilder_ema(plus_dm, n) / a.replace(0, np.nan)
    minus_di = 100 * wilder_ema(minus_dm, n) / a.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder_ema(dx, n)

def build_features(df1h: pd.DataFrame, s: Strategy) -> pd.DataFrame:
    x = df1h.copy()
    x["atr1h"] = atr(x, 14)
    x["don_hi"] = x["high"].shift(1).rolling(s.breakout_lookback).max()
    x["don_lo"] = x["low"].shift(1).rolling(s.breakout_lookback).min()

    # 4H bar [00:00,04:00) is labelled 04:00. At 04:00 its OHLC is known.
    h4 = x[["open","high","low","close","volume"]].resample(
        "4h", closed="left", label="right"
    ).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    h4["ema_fast"] = h4["close"].ewm(span=s.ema_fast, adjust=False,
                                     min_periods=s.ema_fast).mean()
    h4["ema_slow"] = h4["close"].ewm(span=s.ema_slow, adjust=False,
                                     min_periods=s.ema_slow).mean()
    h4["adx"] = adx(h4, 14)
    h4["atr"] = atr(h4, 14)
    h4["atr_pct"] = h4["atr"] / h4["close"]
    h4["shock_cut"] = h4["atr_pct"].rolling(540, min_periods=180).quantile(s.shock_quantile)
    h4["slope"] = h4["ema_slow"] - h4["ema_slow"].shift(s.slope_bars)

    shock = (h4["atr_pct"] > h4["shock_cut"]) | \
            (h4["close"].pct_change().abs() >
             3.0 * h4["close"].pct_change().rolling(90, min_periods=30).std())
    bull = (
        (h4["close"] > h4["ema_slow"]) &
        (h4["ema_fast"] > h4["ema_slow"]) &
        (h4["slope"] > 0) &
        (h4["adx"] >= s.adx_threshold)
    )
    bear = (
        (h4["close"] < h4["ema_slow"]) &
        (h4["ema_fast"] < h4["ema_slow"]) &
        (h4["slope"] < 0) &
        (h4["adx"] >= s.adx_threshold)
    )
    h4["regime"] = "RANGE"
    h4.loc[bull, "regime"] = "BULL"
    h4.loc[bear, "regime"] = "BEAR"
    h4.loc[shock.fillna(False), "regime"] = "SHOCK"

    # merge_asof backward ensures only completed 4H information is available.
    left = x.reset_index().rename(columns={x.index.name or "index":"time"})
    right = h4[["regime","adx","atr_pct"]].reset_index()
    right = right.rename(columns={right.columns[0]:"time",
                                  "adx":"adx4h","atr_pct":"atr_pct4h"})
    merged = pd.merge_asof(left.sort_values("time"), right.sort_values("time"),
                           on="time", direction="backward")
    return merged.set_index("time")

def apply_slippage(price: float, side: int, bps: float) -> float:
    # side +1 = buy; -1 = sell
    return price * (1 + side * bps / 10_000.0)

def fee(notional: float, bps: float) -> float:
    return abs(notional) * bps / 10_000.0

def week_key(ts: pd.Timestamp):
    iso = ts.isocalendar()
    return (iso.year, iso.week)

def backtest(df: pd.DataFrame, strategy: Strategy, costs: CostModel,
             rules: RiskRules, initial: float = 10_000.0):
    d = build_features(df, strategy).dropna(subset=["atr1h","regime"])
    if len(d) < 1000:
        raise RuntimeError("Too little usable history.")

    cash = initial
    qty = 0.0
    direction = 0
    avg_entry = 0.0
    initial_entry = 0.0
    entry_atr = 0.0
    stop = np.nan
    max_fav = np.nan
    min_fav = np.nan
    add1 = add2 = False
    campaign_start_cash = initial
    campaign_entry_time = None
    total_fees = 0.0
    hard_stopped = False
    pending = 0
    pending_signal_time = None

    equity_peak = initial
    day_start_equity = initial
    week_start_equity = initial
    current_day = d.index[0].date()
    current_week = week_key(d.index[0])
    day_locked = False
    week_locked = False
    bars_in_market = 0
    trades = []
    events = []
    equity_rows = []

    def mark(px):
        return cash + (direction * qty * (px - avg_entry) if qty > 0 else 0.0)

    def max_qty_for_notional(eq, px):
        return max(0.0, rules.notional_cap * max(eq, 0.0) / px)

    def enter(ts, side, px, a, tranche_frac, reason):
        nonlocal cash, qty, direction, avg_entry, initial_entry, entry_atr
        nonlocal stop, max_fav, min_fav, campaign_start_cash, campaign_entry_time
        nonlocal total_fees

        eq = mark(px)
        risk_budget = max(eq, 0.0) * strategy.risk_pct
        # Initial tranche consumes 50% of campaign risk budget; add tranches 25%.
        distance = strategy.stop_atr * a
        desired = (risk_budget * tranche_frac / distance) if distance > 0 else 0.0
        cap_qty = max_qty_for_notional(eq, px) - qty
        q = min(desired, max(0.0, cap_qty))
        if q <= 0:
            return False
        fill = apply_slippage(px, side, costs.slippage_bps)
        f = fee(fill * q, costs.fee_bps)
        cash -= f
        total_fees += f
        if qty == 0:
            direction = side
            qty = q
            avg_entry = fill
            initial_entry = fill
            entry_atr = a
            stop = fill - side * strategy.stop_atr * a
            max_fav = fill
            min_fav = fill
            campaign_start_cash = cash + f
            campaign_entry_time = ts
        else:
            avg_entry = (avg_entry * qty + fill * q) / (qty + q)
            qty += q
        events.append({"time":ts, "event":reason, "side":side,
                       "price":fill, "qty":q, "fee":f})
        return True

    def close_all(ts, px, reason):
        nonlocal cash, qty, direction, avg_entry, stop, max_fav, min_fav
        nonlocal add1, add2, total_fees, campaign_entry_time
        if qty <= 0:
            return
        exit_side = -direction
        fill = apply_slippage(px, exit_side, costs.slippage_bps)
        pnl = direction * qty * (fill - avg_entry)
        f = fee(fill * qty, costs.fee_bps)
        cash += pnl - f
        total_fees += f
        campaign_pnl = cash - campaign_start_cash
        trades.append({
            "entry_time": campaign_entry_time,
            "exit_time": ts,
            "direction": "LONG" if direction > 0 else "SHORT",
            "exit_reason": reason,
            "entry_avg": avg_entry,
            "exit_price": fill,
            "qty": qty,
            "net_pnl": campaign_pnl,
            "return_on_start_equity": campaign_pnl / max(campaign_start_cash, 1e-12),
        })
        events.append({"time":ts, "event":"EXIT_"+reason, "side":exit_side,
                       "price":fill, "qty":qty, "fee":f})
        qty = 0.0
        direction = 0
        avg_entry = 0.0
        stop = np.nan
        max_fav = min_fav = np.nan
        add1 = add2 = False
        campaign_entry_time = None

    prev = None
    for ts, r in d.iterrows():
        o, h, l, c, a = map(float, [r.open, r.high, r.low, r.close, r.atr1h])
        regime = r.regime

        # Reset day/week risk locks using equity known at current open.
        eq_open = mark(o)
        if ts.date() != current_day:
            current_day = ts.date()
            day_start_equity = eq_open
            day_locked = False
        wk = week_key(ts)
        if wk != current_week:
            current_week = wk
            week_start_equity = eq_open
            week_locked = False

        # Execute previous completed-bar signal at this bar's open.
        if pending and qty == 0 and not day_locked and not week_locked and not hard_stopped:
            if (pending > 0 and regime == "BULL") or (pending < 0 and regime == "BEAR"):
                initial_entry = 0
                if enter(ts, pending, o, a, 0.50, "ENTRY"):
                    add1 = add2 = False
            pending = 0

        # Conservative stop handling: pre-existing stop is checked BEFORE using
        # this bar's new extreme to update the trailing stop.
        if qty > 0:
            bars_in_market += 1
            if direction > 0 and l <= stop:
                px = min(o, stop) if o < stop else stop
                close_all(ts, px, "STOP")
            elif direction < 0 and h >= stop:
                px = max(o, stop) if o > stop else stop
                close_all(ts, px, "STOP")

        # Regime reversal exit at next available close; no instant flip.
        if qty > 0:
            if (direction > 0 and regime == "BEAR") or (direction < 0 and regime == "BULL"):
                close_all(ts, c, "REGIME_FLIP")

        # Winner-only adds. Stop checks were already processed, so same-bar
        # adverse excursion is conservatively given priority.
        if qty > 0:
            if direction > 0:
                if not add1 and h >= initial_entry + strategy.add1_atr * entry_atr:
                    if enter(ts, +1, initial_entry + strategy.add1_atr*entry_atr,
                             entry_atr, 0.25, "ADD1"):
                        add1 = True
                        stop = max(stop, initial_entry - 1.0*entry_atr)
                if add1 and not add2 and h >= initial_entry + strategy.add2_atr * entry_atr:
                    if enter(ts, +1, initial_entry + strategy.add2_atr*entry_atr,
                             entry_atr, 0.25, "ADD2"):
                        add2 = True
                        stop = max(stop, avg_entry)
            else:
                if not add1 and l <= initial_entry - strategy.add1_atr * entry_atr:
                    if enter(ts, -1, initial_entry - strategy.add1_atr*entry_atr,
                             entry_atr, 0.25, "ADD1"):
                        add1 = True
                        stop = min(stop, initial_entry + 1.0*entry_atr)
                if add1 and not add2 and l <= initial_entry - strategy.add2_atr * entry_atr:
                    if enter(ts, -1, initial_entry - strategy.add2_atr*entry_atr,
                             entry_atr, 0.25, "ADD2"):
                        add2 = True
                        stop = min(stop, avg_entry)

        # Update trailing stop for NEXT bar only.
        if qty > 0:
            if direction > 0:
                max_fav = max(max_fav, h)
                trail_mult = 1.5 if regime == "SHOCK" else strategy.trail_atr
                stop = max(stop, max_fav - trail_mult * a)
            else:
                min_fav = min(min_fav, l)
                trail_mult = 1.5 if regime == "SHOCK" else strategy.trail_atr
                stop = min(stop, min_fav + trail_mult * a)

        eq = mark(c)
        equity_peak = max(equity_peak, eq)
        dd = (eq / equity_peak - 1.0) if equity_peak > 0 else -1.0

        # Account risk locks.
        if day_start_equity > 0 and eq/day_start_equity - 1 <= -rules.daily_loss_lock:
            day_locked = True
            if qty > 0:
                close_all(ts, c, "DAILY_LOCK")
                eq = cash
        if week_start_equity > 0 and eq/week_start_equity - 1 <= -rules.weekly_loss_lock:
            week_locked = True
            if qty > 0:
                close_all(ts, c, "WEEKLY_LOCK")
                eq = cash
        if dd <= -rules.hard_dd:
            if qty > 0:
                close_all(ts, c, "HARD_DD")
                eq = cash
            hard_stopped = True

        # Completed-bar signal for next bar. Soft DD halves risk by simply
        # suppressing lower-conviction breakout entries in alternating bars.
        if qty == 0 and not pending and not day_locked and not week_locked and not hard_stopped:
            soft = dd <= -rules.soft_dd
            allow = (not soft) or (ts.hour % 2 == 0)
            if allow:
                if regime == "BULL" and c > float(r.don_hi):
                    pending = +1
                    pending_signal_time = ts
                elif regime == "BEAR" and c < float(r.don_lo):
                    pending = -1
                    pending_signal_time = ts

        equity_rows.append({
            "time": ts, "equity": eq, "cash": cash, "position_qty": qty,
            "direction": direction, "regime": regime, "drawdown": dd,
            "stop": stop if qty else np.nan
        })
        prev = r

    if qty > 0:
        ts = d.index[-1]
        close_all(ts, float(d.iloc[-1].close), "END")
        if equity_rows:
            equity_rows[-1]["equity"] = cash
            equity_rows[-1]["cash"] = cash
            equity_rows[-1]["position_qty"] = 0.0
            equity_rows[-1]["direction"] = 0

    eqdf = pd.DataFrame(equity_rows).set_index("time")
    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(events)
    return eqdf, tdf, edf, {"fees": total_fees, "hard_stopped": hard_stopped,
                             "bars_in_market": bars_in_market, "usable_bars": len(d)}

def calc_metrics(eq: pd.DataFrame, trades: pd.DataFrame, extra: dict,
                 initial: float = 10_000.0):
    if eq.empty:
        return {}
    final = float(eq.equity.iloc[-1])
    days = max((eq.index[-1] - eq.index[0]).total_seconds()/86400, 1)
    years = days / 365.2425
    total_ret = final/initial - 1
    cagr = (final/initial)**(1/years) - 1 if final > 0 else -1.0
    dd = eq.equity/eq.equity.cummax() - 1
    mdd = float(dd.min())
    daily = eq.equity.resample("1D").last().dropna().pct_change().dropna()
    sharpe = float(np.sqrt(365)*daily.mean()/daily.std()) if daily.std() > 0 else np.nan
    downside = daily[daily < 0].std()
    sortino = float(np.sqrt(365)*daily.mean()/downside) if downside and downside > 0 else np.nan
    calmar = cagr/abs(mdd) if mdd < 0 else np.nan
    if trades.empty:
        pf = np.nan; win = np.nan; n = 0
    else:
        wins = trades.loc[trades.net_pnl > 0, "net_pnl"].sum()
        losses = -trades.loc[trades.net_pnl < 0, "net_pnl"].sum()
        pf = float(wins/losses) if losses > 0 else np.inf
        win = float((trades.net_pnl > 0).mean())
        n = len(trades)
    return {
        "start": str(eq.index[0]),
        "end": str(eq.index[-1]),
        "initial_usd": initial,
        "final_usd": final,
        "total_return": total_ret,
        "cagr": cagr,
        "max_drawdown": mdd,
        "sharpe_365": sharpe,
        "sortino_365": sortino,
        "calmar": calmar,
        "profit_factor": pf,
        "win_rate": win,
        "campaigns": n,
        "fees_paid_usd": extra["fees"],
        "market_exposure": extra["bars_in_market"]/max(extra["usable_bars"],1),
        "hard_stopped": extra["hard_stopped"],
    }

def score_metric(m: dict):
    if not m or m.get("hard_stopped"):
        return -1e9
    if m.get("campaigns", 0) < 8:
        return -1e6
    c = m.get("cagr", -1)
    dd = abs(m.get("max_drawdown", -1))
    sh = m.get("sharpe_365", 0)
    pf = m.get("profit_factor", 0)
    if not np.isfinite(sh): sh = -1
    if not np.isfinite(pf): pf = 5
    # Deliberately penalize drawdown; not a raw-return optimizer.
    return c - 1.5*dd + 0.05*sh + 0.02*min(pf, 5)

def run_candidates(data, costs, rules, initial):
    rows = []
    outputs = {}
    for s in CANDIDATES:
        print(f"[TEST] {s.name}", flush=True)
        eq, tr, ev, ex = backtest(data, s, costs, rules, initial)
        m = calc_metrics(eq, tr, ex, initial)
        rows.append({"strategy":s.name, **m, "score":score_metric(m)})
        outputs[s.name] = (eq, tr, ev, m)
    return pd.DataFrame(rows).sort_values("score", ascending=False), outputs

def walk_forward(data, costs, rules, initial):
    # 2-year train -> 1-year OOS test, rolled one year.
    start = data.index.min().normalize()
    end = data.index.max().normalize()
    rows = []
    anchor = start
    while anchor + pd.DateOffset(years=3) <= end + pd.Timedelta(days=1):
        train_end = anchor + pd.DateOffset(years=2) - pd.Timedelta(hours=1)
        test_start = train_end + pd.Timedelta(hours=1)
        test_end = anchor + pd.DateOffset(years=3) - pd.Timedelta(hours=1)
        train = data.loc[(data.index >= anchor) & (data.index <= train_end)]
        test = data.loc[(data.index >= test_start) & (data.index <= test_end)]
        if len(train) < 8_000 or len(test) < 3_000:
            anchor = anchor + pd.DateOffset(years=1)
            continue
        train_scores = []
        for s in CANDIDATES:
            eq, tr, ev, ex = backtest(train, s, costs, rules, initial)
            m = calc_metrics(eq, tr, ex, initial)
            train_scores.append((score_metric(m), s, m))
        train_scores.sort(key=lambda z:z[0], reverse=True)
        _, chosen, train_m = train_scores[0]
        eq, tr, ev, ex = backtest(test, chosen, costs, rules, initial)
        test_m = calc_metrics(eq, tr, ex, initial)
        rows.append({
            "train_start":anchor, "train_end":train_end,
            "test_start":test_start, "test_end":test_end,
            "chosen":chosen.name,
            "train_cagr":train_m["cagr"], "train_mdd":train_m["max_drawdown"],
            "test_return":test_m["total_return"], "test_cagr":test_m["cagr"],
            "test_mdd":test_m["max_drawdown"], "test_sharpe":test_m["sharpe_365"],
            "test_pf":test_m["profit_factor"], "test_trades":test_m["campaigns"],
            "test_hard_stopped":test_m["hard_stopped"]
        })
        anchor = anchor + pd.DateOffset(years=1)
    return pd.DataFrame(rows)

def yearly_from_equity(eq: pd.DataFrame):
    rows = []
    for year, g in eq.groupby(eq.index.year):
        if len(g) < 2: continue
        ret = g.equity.iloc[-1]/g.equity.iloc[0]-1
        dd = (g.equity/g.equity.cummax()-1).min()
        rows.append({"year":int(year), "return":ret, "within_year_mdd":dd})
    return pd.DataFrame(rows)

def fmt_pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x*100:.2f}%"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-08-17")
    ap.add_argument("--end", default=(datetime.now(timezone.utc)-timedelta(days=1)).strftime("%Y-%m-%d"))
    ap.add_argument("--initial", type=float, default=10_000.0)
    ap.add_argument("--fee-bps", type=float, default=5.5)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    outdir = Path(args.results); outdir.mkdir(parents=True, exist_ok=True)
    data = download_btc_1h(args.start, args.end, Path(args.cache))
    costs = CostModel(args.fee_bps, args.slippage_bps)
    rules = RiskRules()

    candidates, outputs = run_candidates(data, costs, rules, args.initial)
    candidates.to_csv(outdir/"candidate_summary.csv", index=False)
    best_name = candidates.iloc[0].strategy
    eq, tr, ev, best_m = outputs[best_name]
    eq.to_csv(outdir/"equity.csv")
    tr.to_csv(outdir/"trades.csv", index=False)
    ev.to_csv(outdir/"events.csv", index=False)
    yearly_from_equity(eq).to_csv(outdir/"yearly.csv", index=False)

    wf = walk_forward(data, costs, rules, args.initial)
    wf.to_csv(outdir/"walk_forward.csv", index=False)

    # 2x transaction-cost stress.
    stress_costs = CostModel(args.fee_bps*2, args.slippage_bps*2)
    stress_rows = []
    for s in CANDIDATES:
        e2, t2, v2, x2 = backtest(data, s, stress_costs, rules, args.initial)
        m2 = calc_metrics(e2, t2, x2, args.initial)
        stress_rows.append({"strategy":s.name, **m2})
    pd.DataFrame(stress_rows).to_csv(outdir/"cost_stress_2x.csv", index=False)

    bh_ret = data.close.iloc[-1]/data.close.iloc[0]-1
    summary = {
        "data_start": str(data.index.min()),
        "data_end": str(data.index.max()),
        "rows": len(data),
        "best_in_full_sample": best_name,
        "best_metrics": best_m,
        "buy_hold_total_return": float(bh_ret),
        "assumptions": {
            "fee_bps_per_fill": costs.fee_bps,
            "slippage_bps_per_fill": costs.slippage_bps,
            "funding_included": False,
            "price_source": "Binance public BTCUSDT spot 1h as price proxy",
            "entry_timing": "signal on completed 1h close, entry next 1h open",
            "notional_cap": rules.notional_cap,
            "base_campaign_risk_pct": CANDIDATES[0].risk_pct,
        }
    }
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2, default=str))

    md = []
    md.append("# BTC AI EA — Phase 1 Backtest Result\n")
    md.append(f"- Data: {summary['data_start']} → {summary['data_end']} ({len(data):,} 1h bars)")
    md.append(f"- Best full-sample candidate: **{best_name}**")
    md.append(f"- CAGR: **{fmt_pct(best_m['cagr'])}**")
    md.append(f"- Max drawdown: **{fmt_pct(best_m['max_drawdown'])}**")
    md.append(f"- Sharpe: **{best_m['sharpe_365']:.2f}**")
    md.append(f"- Profit factor: **{best_m['profit_factor']:.2f}**")
    md.append(f"- Campaigns: **{best_m['campaigns']}**")
    md.append(f"- Hard DD stop triggered: **{best_m['hard_stopped']}**")
    md.append("\n## Candidate table\n")
    md.append(candidates.to_markdown(index=False))
    md.append("\n## Walk-forward OOS\n")
    md.append(wf.to_markdown(index=False) if not wf.empty else "Not enough windows.")
    md.append("\n## Interpretation gate\n")
    md.append("Phase 1 is a research screen only. Do not deploy live unless OOS, 2x-cost "
              "stress, funding-aware futures validation, and paper trading also pass.")
    (outdir/"REPORT.md").write_text("\n".join(md))

    print("\n=== COMPLETE ===")
    print((outdir/"REPORT.md").read_text())

if __name__ == "__main__":
    main()
