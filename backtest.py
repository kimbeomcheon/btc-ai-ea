#!/usr/bin/env python3
"""
BTC AI EA — V4 Macro Trend Research Backtester
================================================

V4 design objective
-------------------
Fix the main V1-V3 failure modes:
- too many 1H trades
- transaction-cost drag
- weak OOS persistence
- insufficient participation in major BTC trends

Architecture
------------
1D regime -> 4H breakout -> asymmetric long/short -> winner-only pyramiding
-> ATR stop/trailing -> volatility sizing -> account loss locks

Research rules
--------------
- Signals only use completed bars.
- 4H signal is executed at the NEXT 4H bar open.
- Long and short are asymmetric: BTC long risk is larger; bear shorts are smaller.
- No martingale, no averaging down, no simultaneous long/short.
- No parameter optimization inside a single window.
- Fixed V4A/V4B/V4C families are compared, then walk-forward selected.
- Binance BTCUSDT spot 1H is a long-history PRICE PROXY.
- Funding/basis/perpetual-specific data are intentionally deferred to Phase 2.
"""

from __future__ import annotations

import argparse
import io
import json
import math
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
    "open_time","open","high","low","close","volume","close_time",
    "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
]

@dataclass(frozen=True)
class Strategy:
    name: str
    ema_fast: int
    ema_slow: int
    adx_min: float
    entry_lookback_4h: int
    exit_lookback_4h: int
    initial_stop_atr: float
    trail_atr: float
    breakout_buffer_atr: float
    vol_target: float
    max_leverage: float
    long_risk_pct: float
    short_risk_pct: float
    short_scale: float
    add1_atr: float = 1.0
    add2_atr: float = 2.0
    cooldown_bars: int = 6
    shock_quantile: float = 0.95

CANDIDATES = [
    Strategy(
        "V4A", ema_fast=50, ema_slow=200, adx_min=18,
        entry_lookback_4h=30, exit_lookback_4h=14,
        initial_stop_atr=3.0, trail_atr=5.0, breakout_buffer_atr=0.10,
        vol_target=0.35, max_leverage=1.25,
        long_risk_pct=0.008, short_risk_pct=0.004, short_scale=0.50,
    ),
    Strategy(
        "V4B", ema_fast=75, ema_slow=200, adx_min=20,
        entry_lookback_4h=55, exit_lookback_4h=20,
        initial_stop_atr=3.5, trail_atr=6.0, breakout_buffer_atr=0.10,
        vol_target=0.35, max_leverage=1.25,
        long_risk_pct=0.008, short_risk_pct=0.004, short_scale=0.45,
    ),
    Strategy(
        "V4C", ema_fast=100, ema_slow=250, adx_min=18,
        entry_lookback_4h=80, exit_lookback_4h=30,
        initial_stop_atr=4.0, trail_atr=7.0, breakout_buffer_atr=0.05,
        vol_target=0.30, max_leverage=1.50,
        long_risk_pct=0.008, short_risk_pct=0.0035, short_scale=0.40,
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
    soft_dd: float = 0.10
    hard_dd: float = 0.15

def _fetch_bytes(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "btc-ai-ea-v4/1.0"})
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
    for c in ["open","high","low","close","volume"]:
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

    # current month via completed daily archives
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
    bad = ((x["high"] < x[["open","close","low"]].max(axis=1)) |
           (x["low"] > x[["open","close","high"]].min(axis=1)))
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

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * wilder(plus_dm, n) / a.replace(0, np.nan)
    mdi = 100 * wilder(minus_dm, n) / a.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n)

def build_features(df1h: pd.DataFrame, s: Strategy) -> pd.DataFrame:
    # 4H bars labeled at OPEN time. A signal on one row's CLOSE is executed
    # at the next row's OPEN.
    h4 = df1h[["open","high","low","close","volume"]].resample(
        "4h", closed="left", label="left"
    ).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    h4["atr"] = atr(h4, 14)
    h4["entry_hi"] = h4["high"].shift(1).rolling(s.entry_lookback_4h).max()
    h4["entry_lo"] = h4["low"].shift(1).rolling(s.entry_lookback_4h).min()
    h4["exit_hi"] = h4["high"].shift(1).rolling(s.exit_lookback_4h).max()
    h4["exit_lo"] = h4["low"].shift(1).rolling(s.exit_lookback_4h).min()

    # Daily bars labeled at the NEXT midnight: at that timestamp the full
    # previous day's OHLC is known.
    d = df1h[["open","high","low","close","volume"]].resample(
        "1D", closed="left", label="right"
    ).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    d["ema_fast"] = d["close"].ewm(span=s.ema_fast, adjust=False,
                                    min_periods=s.ema_fast).mean()
    d["ema_slow"] = d["close"].ewm(span=s.ema_slow, adjust=False,
                                    min_periods=s.ema_slow).mean()
    d["adx"] = adx(d, 14)
    d["slow_slope"] = d["ema_slow"] - d["ema_slow"].shift(10)

    ret = d["close"].pct_change()
    d["rv30"] = ret.rolling(30, min_periods=20).std() * np.sqrt(365)
    d["atr_pct"] = atr(d, 14) / d["close"]
    d["shock_cut"] = d["atr_pct"].rolling(540, min_periods=180).quantile(s.shock_quantile)
    sig = ret.rolling(60, min_periods=30).std()
    shock = ((d["atr_pct"] > d["shock_cut"]) |
             (ret.abs() > 3.0 * sig))

    bull = (
        (d["close"] > d["ema_slow"]) &
        (d["ema_fast"] > d["ema_slow"]) &
        (d["slow_slope"] > 0) &
        (d["adx"] >= s.adx_min)
    )
    bear = (
        (d["close"] < d["ema_slow"]) &
        (d["ema_fast"] < d["ema_slow"]) &
        (d["slow_slope"] < 0) &
        (d["adx"] >= s.adx_min)
    )
    d["regime"] = "RANGE"
    d.loc[bull, "regime"] = "BULL"
    d.loc[bear, "regime"] = "BEAR"
    d.loc[shock.fillna(False), "regime"] = "SHOCK"

    left = h4.reset_index().rename(columns={h4.index.name or "index":"time"})
    right = d[["regime","rv30","adx","ema_fast","ema_slow"]].reset_index()
    right = right.rename(columns={
        right.columns[0]:"time", "adx":"adx1d",
        "ema_fast":"ema_fast1d", "ema_slow":"ema_slow1d"
    })
    x = pd.merge_asof(left.sort_values("time"), right.sort_values("time"),
                      on="time", direction="backward")
    return x.set_index("time")

def slip(price: float, side: int, bps: float) -> float:
    return price * (1 + side * bps / 10_000.0)

def commission(notional: float, bps: float) -> float:
    return abs(notional) * bps / 10_000.0

def wk(ts: pd.Timestamp):
    i = ts.isocalendar()
    return (int(i.year), int(i.week))

def target_leverage(eq: float, px: float, a: float, rv: float,
                    side: int, s: Strategy, soft_dd: bool) -> float:
    if not np.isfinite(rv) or rv <= 0 or not np.isfinite(a) or a <= 0 or px <= 0:
        return 0.0
    vol_lev = s.vol_target / rv
    risk_pct = s.long_risk_pct if side > 0 else s.short_risk_pct
    stop_pct = s.initial_stop_atr * a / px
    risk_lev = risk_pct / stop_pct if stop_pct > 0 else 0.0
    lev = min(s.max_leverage, vol_lev, risk_lev)
    if side < 0:
        lev *= s.short_scale
    if soft_dd:
        lev *= 0.50
    return max(0.0, lev)

def backtest(df1h: pd.DataFrame, s: Strategy, costs: CostModel,
             rules: RiskRules, initial: float = 10_000.0, entry_delay_bars: int = 1):
    d = build_features(df1h, s).dropna(subset=["atr","regime","rv30"])
    if len(d) < 1500:
        raise RuntimeError("Insufficient usable 4H history")

    cash = initial
    qty = 0.0
    side = 0
    avg_entry = 0.0
    first_entry = 0.0
    entry_atr = 0.0
    stop = np.nan
    fav = np.nan
    target_qty = 0.0
    add1 = add2 = False
    campaign_start = initial
    campaign_time = None
    cooldown = 0
    pending = 0
    pending_count = 0

    fees = 0.0
    peak = initial
    hard_stopped = False
    bars_market = 0
    current_day = d.index[0].date()
    current_week = wk(d.index[0])
    day_start = initial
    week_start = initial
    day_lock = week_lock = False

    trades = []
    events = []
    equity = []

    def mark(px):
        if qty <= 0:
            return cash
        return cash + side * qty * (px - avg_entry)

    def open_campaign(ts, new_side, px, a, rv, regime, soft):
        nonlocal cash, qty, side, avg_entry, first_entry, entry_atr, stop, fav
        nonlocal target_qty, add1, add2, campaign_start, campaign_time, fees
        eq = mark(px)
        lev = target_leverage(eq, px, a, rv, new_side, s, soft)
        if lev <= 0:
            return False
        tq = eq * lev / px
        q = tq * 0.50
        if q <= 0:
            return False
        fill = slip(px, new_side, costs.slippage_bps)
        f = commission(fill*q, costs.fee_bps)
        cash -= f
        fees += f
        side = new_side
        qty = q
        avg_entry = fill
        first_entry = fill
        entry_atr = a
        target_qty = tq
        stop = fill - new_side * s.initial_stop_atr * a
        fav = fill
        add1 = add2 = False
        campaign_start = cash + f
        campaign_time = ts
        events.append({
            "time":ts, "event":"ENTRY", "direction":"LONG" if side>0 else "SHORT",
            "price":fill, "qty":q, "leverage_target":lev, "regime":regime, "fee":f
        })
        return True

    def add_to(ts, fraction, px, tag):
        nonlocal cash, qty, avg_entry, fees
        desired_total = target_qty * fraction
        q = max(0.0, desired_total - qty)
        if q <= 0:
            return False
        fill = slip(px, side, costs.slippage_bps)
        f = commission(fill*q, costs.fee_bps)
        cash -= f
        fees += f
        avg_entry = (avg_entry*qty + fill*q) / (qty+q)
        qty += q
        events.append({
            "time":ts, "event":tag, "direction":"LONG" if side>0 else "SHORT",
            "price":fill, "qty":q, "fee":f
        })
        return True

    def close_all(ts, px, reason):
        nonlocal cash, qty, side, avg_entry, stop, fav, target_qty
        nonlocal add1, add2, campaign_time, cooldown, fees
        if qty <= 0:
            return
        exit_side = -side
        fill = slip(px, exit_side, costs.slippage_bps)
        pnl = side * qty * (fill - avg_entry)
        f = commission(fill*qty, costs.fee_bps)
        cash += pnl - f
        fees += f
        trades.append({
            "entry_time":campaign_time, "exit_time":ts,
            "direction":"LONG" if side>0 else "SHORT",
            "exit_reason":reason, "entry_avg":avg_entry, "exit_price":fill,
            "qty":qty, "net_pnl":cash-campaign_start,
            "return_on_start_equity":(cash-campaign_start)/max(campaign_start,1e-12)
        })
        events.append({
            "time":ts, "event":"EXIT_"+reason,
            "direction":"LONG" if side>0 else "SHORT",
            "price":fill, "qty":qty, "fee":f
        })
        qty=0.0; side=0; avg_entry=0.0; stop=np.nan; fav=np.nan
        target_qty=0.0; add1=False; add2=False; campaign_time=None
        cooldown = s.cooldown_bars

    for ts, r in d.iterrows():
        o,h,l,c,a = map(float, [r.open,r.high,r.low,r.close,r.atr])
        regime = str(r.regime)
        rv = float(r.rv30)

        if cooldown > 0:
            cooldown -= 1

        eq_open = mark(o)
        if ts.date() != current_day:
            current_day = ts.date()
            day_start = eq_open
            day_lock = False
        ww = wk(ts)
        if ww != current_week:
            current_week = ww
            week_start = eq_open
            week_lock = False

        peak = max(peak, eq_open)
        dd_open = eq_open/peak - 1 if peak>0 else -1
        soft = dd_open <= -rules.soft_dd

        # Existing position exits on regime invalidation at the current 4H open.
        if qty > 0:
            if side > 0 and regime not in ("BULL","SHOCK"):
                close_all(ts, o, "REGIME")
            elif side < 0 and regime not in ("BEAR","SHOCK"):
                close_all(ts, o, "REGIME")

        # Deferred entry from completed 4H signal.
        if pending != 0:
            pending_count -= 1
            if pending_count <= 0:
                if qty == 0 and cooldown == 0 and not day_lock and not week_lock and not hard_stopped:
                    if ((pending > 0 and regime=="BULL") or
                        (pending < 0 and regime=="BEAR")):
                        open_campaign(ts, pending, o, a, rv, regime, soft)
                pending = 0
                pending_count = 0

        # Stop FIRST: conservative same-bar sequencing.
        if qty > 0:
            bars_market += 1
            if side > 0 and l <= stop:
                px = o if o < stop else stop
                close_all(ts, px, "STOP")
            elif side < 0 and h >= stop:
                px = o if o > stop else stop
                close_all(ts, px, "STOP")

        # Counter-channel exits use the completed 4H close.
        if qty > 0:
            if side > 0 and c < float(r.exit_lo):
                close_all(ts, c, "CHANNEL")
            elif side < 0 and c > float(r.exit_hi):
                close_all(ts, c, "CHANNEL")

        # Winner-only anti-martingale. Stop had priority above.
        if qty > 0:
            if side > 0:
                if not add1 and h >= first_entry + s.add1_atr*entry_atr:
                    if add_to(ts, 0.75, first_entry+s.add1_atr*entry_atr, "ADD1"):
                        add1=True
                        stop=max(stop, first_entry-entry_atr)
                if add1 and not add2 and h >= first_entry + s.add2_atr*entry_atr:
                    if add_to(ts, 1.00, first_entry+s.add2_atr*entry_atr, "ADD2"):
                        add2=True
                        stop=max(stop, avg_entry)
            else:
                if not add1 and l <= first_entry - s.add1_atr*entry_atr:
                    if add_to(ts, 0.75, first_entry-s.add1_atr*entry_atr, "ADD1"):
                        add1=True
                        stop=min(stop, first_entry+entry_atr)
                if add1 and not add2 and l <= first_entry - s.add2_atr*entry_atr:
                    if add_to(ts, 1.00, first_entry-s.add2_atr*entry_atr, "ADD2"):
                        add2=True
                        stop=min(stop, avg_entry)

        # Trailing stop becomes active for NEXT 4H bar.
        if qty > 0:
            trail_mult = min(s.trail_atr, 3.0) if regime=="SHOCK" else s.trail_atr
            if side > 0:
                fav=max(fav,h)
                stop=max(stop, fav-trail_mult*a)
            else:
                fav=min(fav,l)
                stop=min(stop, fav+trail_mult*a)

        eq = mark(c)
        peak = max(peak, eq)
        dd = eq/peak - 1 if peak>0 else -1

        # Portfolio locks.
        if day_start>0 and eq/day_start-1 <= -rules.daily_loss_lock:
            day_lock=True
            if qty>0:
                close_all(ts,c,"DAILY_LOCK")
                eq=cash
        if week_start>0 and eq/week_start-1 <= -rules.weekly_loss_lock:
            week_lock=True
            if qty>0:
                close_all(ts,c,"WEEKLY_LOCK")
                eq=cash
        if dd <= -rules.hard_dd:
            if qty>0:
                close_all(ts,c,"HARD_DD")
                eq=cash
            hard_stopped=True

        # Generate new signal from THIS completed 4H candle.
        # It cannot execute until a future 4H open.
        if qty==0 and pending==0 and cooldown==0 and not day_lock and not week_lock and not hard_stopped:
            buf = s.breakout_buffer_atr * a
            if regime=="BULL" and c > float(r.entry_hi)+buf:
                pending=+1
                pending_count=max(1,entry_delay_bars)
            elif regime=="BEAR" and c < float(r.entry_lo)-buf:
                pending=-1
                pending_count=max(1,entry_delay_bars)

        equity.append({
            "time":ts, "equity":eq, "cash":cash, "qty":qty,
            "direction":side, "regime":regime, "drawdown":dd,
            "stop":stop if qty>0 else np.nan
        })

    if qty>0:
        close_all(d.index[-1], float(d.iloc[-1].close), "END")
        if equity:
            equity[-1]["equity"]=cash
            equity[-1]["cash"]=cash
            equity[-1]["qty"]=0.0
            equity[-1]["direction"]=0

    eqdf=pd.DataFrame(equity).set_index("time")
    trdf=pd.DataFrame(trades)
    evdf=pd.DataFrame(events)
    return eqdf,trdf,evdf,{
        "fees":fees, "hard_stopped":hard_stopped,
        "bars_market":bars_market, "bars_total":len(d)
    }

def metrics(eq: pd.DataFrame, trades: pd.DataFrame, extra: dict, initial: float):
    final=float(eq.equity.iloc[-1])
    days=max((eq.index[-1]-eq.index[0]).total_seconds()/86400,1)
    years=days/365.2425
    cagr=(final/initial)**(1/years)-1 if final>0 else -1
    dd=eq.equity/eq.equity.cummax()-1
    daily=eq.equity.resample("1D").last().dropna().pct_change().dropna()
    std=daily.std()
    sharpe=float(np.sqrt(365)*daily.mean()/std) if std and std>0 else np.nan
    downside=daily[daily<0].std()
    sortino=float(np.sqrt(365)*daily.mean()/downside) if downside and downside>0 else np.nan
    mdd=float(dd.min())
    calmar=cagr/abs(mdd) if mdd<0 else np.nan
    if trades.empty:
        pf=np.nan; win=np.nan; n=0; longs=shorts=0
    else:
        gp=trades.loc[trades.net_pnl>0,"net_pnl"].sum()
        gl=-trades.loc[trades.net_pnl<0,"net_pnl"].sum()
        pf=float(gp/gl) if gl>0 else np.inf
        win=float((trades.net_pnl>0).mean())
        n=len(trades)
        longs=int((trades.direction=="LONG").sum())
        shorts=int((trades.direction=="SHORT").sum())
    return {
        "start":str(eq.index[0]), "end":str(eq.index[-1]),
        "initial_usd":initial, "final_usd":final,
        "total_return":final/initial-1, "cagr":cagr,
        "max_drawdown":mdd, "sharpe_365":sharpe,
        "sortino_365":sortino, "calmar":calmar,
        "profit_factor":pf, "win_rate":win,
        "campaigns":n, "long_campaigns":longs, "short_campaigns":shorts,
        "fees_paid_usd":extra["fees"],
        "market_exposure":extra["bars_market"]/max(extra["bars_total"],1),
        "hard_stopped":extra["hard_stopped"],
    }

def objective(m):
    if not m or m.get("hard_stopped"):
        return -1e9
    if m.get("campaigns",0)<8:
        return -1e6
    sh=m.get("sharpe_365",0)
    pf=m.get("profit_factor",0)
    if not np.isfinite(sh): sh=-1
    if not np.isfinite(pf): pf=5
    # Penalize DD and reward OOS-sensible risk adjusted performance.
    return m["cagr"] - 1.2*abs(m["max_drawdown"]) + 0.08*sh + 0.02*min(pf,5)

def run_all(data,costs,rules,initial,delay=1):
    rows=[]; out={}
    for s in CANDIDATES:
        print(f"[TEST] {s.name}",flush=True)
        eq,tr,ev,ex=backtest(data,s,costs,rules,initial,delay)
        m=metrics(eq,tr,ex,initial)
        rows.append({"strategy":s.name,**m,"score":objective(m)})
        out[s.name]=(eq,tr,ev,m)
    return pd.DataFrame(rows).sort_values("score",ascending=False),out

def walk_forward(data,costs,rules,initial):
    # Slower V4 requires more training history: 3y train -> 1y OOS.
    start=data.index.min().normalize()
    end=data.index.max().normalize()
    rows=[]
    anchor=start
    while anchor+pd.DateOffset(years=4) <= end+pd.Timedelta(days=1):
        train_end=anchor+pd.DateOffset(years=3)-pd.Timedelta(hours=1)
        test_start=train_end+pd.Timedelta(hours=1)
        test_end=anchor+pd.DateOffset(years=4)-pd.Timedelta(hours=1)
        train=data.loc[(data.index>=anchor)&(data.index<=train_end)]
        test=data.loc[(data.index>=test_start)&(data.index<=test_end)]
        if len(train)<15000 or len(test)<4000:
            anchor+=pd.DateOffset(years=1); continue
        scored=[]
        for s in CANDIDATES:
            eq,tr,ev,ex=backtest(train,s,costs,rules,initial)
            m=metrics(eq,tr,ex,initial)
            scored.append((objective(m),s,m))
        scored.sort(key=lambda x:x[0],reverse=True)
        _,chosen,tm=scored[0]
        eq,tr,ev,ex=backtest(test,chosen,costs,rules,initial)
        om=metrics(eq,tr,ex,initial)
        rows.append({
            "train_start":anchor,"train_end":train_end,
            "test_start":test_start,"test_end":test_end,
            "chosen":chosen.name,
            "train_cagr":tm["cagr"],"train_mdd":tm["max_drawdown"],
            "test_return":om["total_return"],"test_cagr":om["cagr"],
            "test_mdd":om["max_drawdown"],"test_sharpe":om["sharpe_365"],
            "test_pf":om["profit_factor"],"test_trades":om["campaigns"],
            "test_hard_stopped":om["hard_stopped"]
        })
        anchor+=pd.DateOffset(years=1)
    return pd.DataFrame(rows)

def robustness_grid(data,best:Strategy,costs,rules,initial):
    rows=[]
    for lb_mult in (0.8,1.0,1.2):
        for trail_mult in (0.8,1.0,1.2):
            s=replace(
                best,
                name=f"{best.name}_LB{lb_mult:.1f}_TR{trail_mult:.1f}",
                entry_lookback_4h=max(12,int(round(best.entry_lookback_4h*lb_mult))),
                trail_atr=max(2.0,best.trail_atr*trail_mult)
            )
            eq,tr,ev,ex=backtest(data,s,costs,rules,initial)
            m=metrics(eq,tr,ex,initial)
            rows.append({
                "lookback_mult":lb_mult,"trail_mult":trail_mult,
                "entry_lookback":s.entry_lookback_4h,"trail_atr":s.trail_atr,
                "cagr":m["cagr"],"mdd":m["max_drawdown"],
                "sharpe":m["sharpe_365"],"pf":m["profit_factor"],
                "campaigns":m["campaigns"],"hard_stopped":m["hard_stopped"]
            })
    return pd.DataFrame(rows)

def benchmark_200d(df1h,initial):
    d=df1h[["close"]].resample("1D",closed="left",label="right").last().dropna()
    d["sma200"]=d.close.rolling(200).mean()
    # Signal known at daily close, applied to next day's return via shift.
    sig=(d.close>d.sma200).astype(float).shift(1).fillna(0)
    r=d.close.pct_change().fillna(0)
    eq=initial*(1+sig*r).cumprod()
    return eq

def yearly(eq):
    rows=[]
    for y,g in eq.groupby(eq.index.year):
        if len(g)<2: continue
        rows.append({
            "year":int(y),
            "return":float(g.equity.iloc[-1]/g.equity.iloc[0]-1),
            "within_year_mdd":float((g.equity/g.equity.cummax()-1).min())
        })
    return pd.DataFrame(rows)

def pass_fail(m,wf,stress,robust):
    oos_ret = float((1+wf.test_return).prod()-1) if not wf.empty else np.nan
    oos_years = len(wf)
    oos_cagr = (1+oos_ret)**(1/oos_years)-1 if oos_years>0 and 1+oos_ret>0 else np.nan
    oos_mdd = float(wf.test_mdd.min()) if not wf.empty else np.nan
    # Aggregate gates intentionally strict.
    return {
        "full_sample_cagr_ge_25pct": bool(m["cagr"]>=0.25),
        "full_sample_mdd_le_15pct": bool(abs(m["max_drawdown"])<=0.15),
        "full_sample_sharpe_ge_1_3": bool(m["sharpe_365"]>=1.3),
        "full_sample_pf_ge_1_4": bool(m["profit_factor"]>=1.4),
        "oos_compound_cagr": oos_cagr,
        "oos_max_window_mdd": oos_mdd,
        "oos_positive_windows_ratio": float((wf.test_return>0).mean()) if not wf.empty else np.nan,
        "stress_2x_not_hard_stopped": bool(not stress["hard_stopped"]),
        "robustness_all_not_hard_stopped": bool((~robust.hard_stopped).all()),
        "robustness_positive_cagr_ratio": float((robust.cagr>0).mean()),
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

    cand,outputs=run_all(data,costs,rules,args.initial)
    cand.to_csv(out/"candidate_summary.csv",index=False)
    best_name=str(cand.iloc[0].strategy)
    best_s=next(s for s in CANDIDATES if s.name==best_name)
    eq,tr,ev,bm=outputs[best_name]
    eq.to_csv(out/"equity.csv")
    tr.to_csv(out/"trades.csv",index=False)
    ev.to_csv(out/"events.csv",index=False)
    yearly(eq).to_csv(out/"yearly.csv",index=False)

    wf=walk_forward(data,costs,rules,args.initial)
    wf.to_csv(out/"walk_forward.csv",index=False)

    # Transaction-cost stress
    stress_costs=CostModel(args.fee_bps*2,args.slippage_bps*2)
    stress_rows=[]
    stress_map={}
    for s in CANDIDATES:
        e,t,v,x=backtest(data,s,stress_costs,rules,args.initial)
        m=metrics(e,t,x,args.initial)
        stress_rows.append({"strategy":s.name,**m})
        stress_map[s.name]=m
    pd.DataFrame(stress_rows).to_csv(out/"cost_stress_2x.csv",index=False)

    # One extra 4H execution delay
    delay_rows=[]
    for s in CANDIDATES:
        e,t,v,x=backtest(data,s,costs,rules,args.initial,entry_delay_bars=2)
        m=metrics(e,t,x,args.initial)
        delay_rows.append({"strategy":s.name,**m})
    pd.DataFrame(delay_rows).to_csv(out/"execution_delay_stress.csv",index=False)

    robust=robustness_grid(data,best_s,costs,rules,args.initial)
    robust.to_csv(out/"robustness_grid.csv",index=False)

    bh=float(data.close.iloc[-1]/data.close.iloc[0]-1)
    smaeq=benchmark_200d(data,args.initial)
    sma_ret=float(smaeq.iloc[-1]/args.initial-1)

    gates=pass_fail(bm,wf,stress_map[best_name],robust)

    # KRW projection is simply the strategy return path applied to KRW 10m.
    krw=pd.DataFrame(index=eq.index)
    krw["strategy_equity_krw"]=10_000_000*(eq.equity/args.initial)
    krw.to_csv(out/"krw_10m_projection.csv")

    summary={
        "version":"V4 Macro Trend",
        "data_start":str(data.index.min()),"data_end":str(data.index.max()),
        "one_hour_rows":len(data),
        "best_full_sample":best_name,
        "best_metrics":bm,
        "buy_hold_total_return":bh,
        "sma200_long_cash_total_return":sma_ret,
        "gates":gates,
        "assumptions":{
            "price_source":"Binance BTCUSDT spot 1h price proxy",
            "funding_included":False,
            "fee_bps_per_fill":costs.fee_bps,
            "slippage_bps_per_fill":costs.slippage_bps,
            "signal_execution":"completed 4H signal -> next 4H open",
            "martingale":False,
            "averaging_down":False,
            "simultaneous_long_short":False,
        }
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,default=str))

    lines=[
        "# BTC AI EA V4 — Backtest Report","",
        f"- Data: {summary['data_start']} → {summary['data_end']} ({len(data):,} 1H bars)",
        f"- Best full-sample candidate: **{best_name}**",
        f"- CAGR: **{pct(bm['cagr'])}**",
        f"- Max drawdown: **{pct(bm['max_drawdown'])}**",
        f"- Sharpe: **{bm['sharpe_365']:.2f}**",
        f"- Profit factor: **{bm['profit_factor']:.2f}**",
        f"- Campaigns: **{bm['campaigns']}** (Long {bm['long_campaigns']} / Short {bm['short_campaigns']})",
        f"- Fees paid: **${bm['fees_paid_usd']:.2f}**",
        f"- Buy & hold total return: **{pct(bh)}**",
        f"- Simple 200D long/cash total return: **{pct(sma_ret)}**",
        "",
        "## Candidate comparison","",
        cand.to_markdown(index=False),
        "",
        "## Walk-forward OOS","",
        wf.to_markdown(index=False) if not wf.empty else "Not enough windows.",
        "",
        "## Acceptance gates","",
        "```json",json.dumps(gates,indent=2,default=str),"```",
        "",
        "## Decision rule","",
        "Do NOT deploy live merely because full-sample performance is attractive. "
        "A viable model must survive OOS, 2x transaction costs, one-bar delay, "
        "nearby-parameter robustness, then perpetual-futures/funding validation "
        "and at least four weeks of paper trading."
    ]
    (out/"REPORT.md").write_text("\n".join(lines))

    print("\n=== V4 COMPLETE ===")
    print((out/"REPORT.md").read_text())

if __name__=="__main__":
    main()
