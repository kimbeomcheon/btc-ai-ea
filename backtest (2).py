#!/usr/bin/env python3
"""
BTC BOOK+TURTLE HYBRID BACKTEST
===============================

Purpose
-------
Compare a book-derived multi-timeframe trend strategy, a Turtle-style breakout
strategy, and a hybrid strategy on BTCUSDT Binance USD-M futures 1H public data.

Design principles
-----------------
- No lookahead: all signals use completed bars only.
- Major trend change: Daily/Weekly/Monthly prior high/low close breakout.
- 12H/4H/1H are execution/add-on layers, not major regime flips.
- Pyramiding only in the profitable direction; no martingale / averaging down.
- Turtle-style 20/55 breakout + ATR sizing is tested separately.
- Costs: fee + slippage per position change; optional historical funding if available.
- Outputs: summary.csv, yearly.csv, equity curves, and config.json.

Important
---------
This is a research backtest, not a promise of lossless trading.
Binance USD-M futures history starts in 2019, so --start 2017-08-17 will be
automatically clipped to 2019-09-08.
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
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

VISION = "https://data.binance.vision/data/futures/um"
FUTURES_LAUNCH = pd.Timestamp("2019-09-08", tz="UTC")
KLINE_COLS = [
    "open_time","open","high","low","close","volume","close_time",
    "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
]

@dataclass(frozen=True)
class Costs:
    fee_bps: float = 5.5
    slippage_bps: float = 2.0
    funding_mult: float = 1.0

@dataclass(frozen=True)
class Params:
    # Book-derived major swing lengths
    daily_swing: int = 20
    weekly_swing: int = 10
    monthly_swing: int = 6

    # Lower-timeframe short-term extreme windows
    h12_swing: int = 10
    h4_swing: int = 20
    h1_swing: int = 20

    # Break confirmation
    penetration_atr: float = 0.10
    min_body_ratio: float = 0.45
    min_clv: float = 0.60

    # Turtle
    turtle_fast: int = 20
    turtle_slow: int = 55
    turtle_exit: int = 10

    # Risk / sizing
    base_exposure: float = 0.55
    add_exposure: float = 0.18
    max_exposure: float = 1.00
    vol_target: float = 0.55
    max_adds: int = 2
    add_trigger_atr: float = 0.75

    # Drawdown control
    soft_dd: float = 0.10
    hard_dd: float = 0.15

def fetch_bytes(url: str, retries: int = 4) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent":"btc-book-turtle-backtest/1.0"})
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
        time.sleep(1.2*(i+1))
    raise RuntimeError(f"download failed: {url}: {last}")

def parse_kline(raw: bytes) -> pd.DataFrame:
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

def parse_funding(raw: bytes) -> pd.Series:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)

    cols = {str(c).lower(): c for c in df.columns}
    tcol = next((cols[k] for k in ["calc_time","funding_time","fundingtime","time","timestamp"] if k in cols), None)
    rcol = next((cols[k] for k in ["last_funding_rate","funding_rate","fundingrate"] if k in cols), None)
    if tcol is None or rcol is None:
        return pd.Series(dtype=float)

    t = pd.to_numeric(df[tcol], errors="coerce")
    unit = "us" if t.dropna().median() > 1e14 else "ms"
    idx = pd.to_datetime(t, unit=unit, utc=True, errors="coerce")
    rate = pd.to_numeric(df[rcol], errors="coerce")
    s = pd.Series(rate.to_numpy(), index=idx).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()

def month_starts(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    last = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= last:
        yield cur
        cur += pd.offsets.MonthBegin(1)

def download_data(start: str, end: str, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    s = max(pd.Timestamp(start, tz="UTC"), FUTURES_LAUNCH)
    e = pd.Timestamp(end, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    current_month = pd.Timestamp(now.year, now.month, 1, tz="UTC")

    frames = []
    funding = []

    monthly_end = min(e, current_month - pd.Timedelta(hours=1))
    for m in month_starts(s, monthly_end):
        ym = m.strftime("%Y-%m")

        kp = cache / f"BTCUSDT-1h-{ym}.zip"
        ku = f"{VISION}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{ym}.zip"
        if not kp.exists():
            try:
                kp.write_bytes(fetch_bytes(ku))
            except urllib.error.HTTPError as ex:
                if ex.code == 404:
                    continue
                raise
        with zipfile.ZipFile(kp) as z:
            n = next(x for x in z.namelist() if x.endswith(".csv"))
            frames.append(parse_kline(z.read(n)))

        fp = cache / f"BTCUSDT-fundingRate-{ym}.zip"
        fu = f"{VISION}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{ym}.zip"
        if not fp.exists():
            try:
                fp.write_bytes(fetch_bytes(fu))
            except urllib.error.HTTPError as ex:
                if ex.code != 404:
                    raise
        if fp.exists():
            try:
                with zipfile.ZipFile(fp) as z:
                    n = next(x for x in z.namelist() if x.endswith(".csv"))
                    funding.append(parse_funding(z.read(n)))
            except Exception:
                pass

    if e >= current_month:
        d0 = max(s.normalize(), current_month)
        d1 = min(e.normalize(), (now - pd.Timedelta(days=1)).normalize())
        if d1 >= d0:
            for d in pd.date_range(d0, d1, freq="1D", tz="UTC"):
                ds = d.strftime("%Y-%m-%d")
                kp = cache / f"BTCUSDT-1h-{ds}.zip"
                ku = f"{VISION}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{ds}.zip"
                if not kp.exists():
                    try:
                        kp.write_bytes(fetch_bytes(ku))
                    except urllib.error.HTTPError as ex:
                        if ex.code == 404:
                            continue
                        raise
                with zipfile.ZipFile(kp) as z:
                    n = next(x for x in z.namelist() if x.endswith(".csv"))
                    frames.append(parse_kline(z.read(n)))

    if not frames:
        raise RuntimeError("No futures data downloaded")

    px = pd.concat(frames).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    px = px.loc[(px.index >= s) & (px.index <= e)]

    fr = pd.concat(funding).sort_index() if funding else pd.Series(dtype=float)
    if len(fr):
        fr = fr[~fr.index.duplicated(keep="last")]
        fr = fr.loc[(fr.index >= s) & (fr.index <= e)]

    return px, fr

def resample_ohlc(px: pd.DataFrame, rule: str, label="right") -> pd.DataFrame:
    return px.resample(rule, closed="left", label=label).agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    ).dropna()

def wilder(s: pd.Series, n: int):
    return s.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def atr(df: pd.DataFrame, n: int = 14):
    prev = df.close.shift(1)
    tr = pd.concat([
        (df.high-df.low).abs(),
        (df.high-prev).abs(),
        (df.low-prev).abs()
    ], axis=1).max(axis=1)
    return wilder(tr, n)

def candle_quality(df: pd.DataFrame):
    rng = (df.high-df.low).replace(0, np.nan)
    body = (df.close-df.open).abs()
    body_ratio = body / rng
    clv_long = (df.close-df.low) / rng
    clv_short = (df.high-df.close) / rng
    return body_ratio, clv_long, clv_short

def add_swings(df: pd.DataFrame, n: int, prefix: str) -> pd.DataFrame:
    x = df.copy()
    x[f"{prefix}_prev_hi"] = x.high.shift(1).rolling(n, min_periods=n).max()
    x[f"{prefix}_prev_lo"] = x.low.shift(1).rolling(n, min_periods=n).min()
    x[f"{prefix}_atr"] = atr(x, 14)
    br, cl, cs = candle_quality(x)
    x[f"{prefix}_body"] = br
    x[f"{prefix}_clv_long"] = cl
    x[f"{prefix}_clv_short"] = cs
    return x

def confirmed_break_up(df: pd.DataFrame, prefix: str, p: Params):
    lvl = df[f"{prefix}_prev_hi"]
    a = df[f"{prefix}_atr"]
    return (
        (df.close > lvl + p.penetration_atr*a) &
        (df[f"{prefix}_body"] >= p.min_body_ratio) &
        (df[f"{prefix}_clv_long"] >= p.min_clv)
    )

def confirmed_break_down(df: pd.DataFrame, prefix: str, p: Params):
    lvl = df[f"{prefix}_prev_lo"]
    a = df[f"{prefix}_atr"]
    return (
        (df.close < lvl - p.penetration_atr*a) &
        (df[f"{prefix}_body"] >= p.min_body_ratio) &
        (df[f"{prefix}_clv_short"] >= p.min_clv)
    )

def build_book_regime(px: pd.DataFrame, p: Params):
    d = add_swings(resample_ohlc(px, "1D"), p.daily_swing, "d")
    w = add_swings(resample_ohlc(px, "1W"), p.weekly_swing, "w")
    m = add_swings(resample_ohlc(px, "MS"), p.monthly_swing, "m")

    d["up"] = confirmed_break_up(d, "d", p)
    d["dn"] = confirmed_break_down(d, "d", p)
    w["up"] = confirmed_break_up(w, "w", p)
    w["dn"] = confirmed_break_down(w, "w", p)
    m["up"] = confirmed_break_up(m, "m", p)
    m["dn"] = confirmed_break_down(m, "m", p)

    # persistent regimes: +1 long / -1 short / 0 neutral
    def persistent(sig_up, sig_dn):
        out = np.zeros(len(sig_up), dtype=float)
        state = 0.0
        for i, (u, dn) in enumerate(zip(sig_up.fillna(False), sig_dn.fillna(False))):
            if u:
                state = 1.0
            elif dn:
                state = -1.0
            out[i] = state
        return out

    d["regime"] = persistent(d["up"], d["dn"])
    w["regime"] = persistent(w["up"], w["dn"])
    m["regime"] = persistent(m["up"], m["dn"])

    # Daily is execution regime; higher frames bias confidence.
    book = d[["regime"]].rename(columns={"regime":"d_regime"}).copy()
    book = pd.merge_asof(
        book.reset_index().rename(columns={book.index.name or "index":"time"}).sort_values("time"),
        w[["regime"]].reset_index().rename(columns={w.index.name or "index":"time","regime":"w_regime"}).sort_values("time"),
        on="time", direction="backward"
    )
    book = pd.merge_asof(
        book.sort_values("time"),
        m[["regime"]].reset_index().rename(columns={m.index.name or "index":"time","regime":"m_regime"}).sort_values("time"),
        on="time", direction="backward"
    ).set_index("time")

    # Weighted hierarchy: daily determines side, weekly/monthly strengthen or weaken.
    agree = (
        (book.d_regime != 0).astype(float) +
        (book.w_regime == book.d_regime).astype(float) +
        (book.m_regime == book.d_regime).astype(float)
    )
    book["confidence"] = (agree / 3.0).clip(0,1)
    return book

def lower_tf_add_signals(px: pd.DataFrame, p: Params):
    h12 = add_swings(resample_ohlc(px, "12h"), p.h12_swing, "h12")
    h4 = add_swings(resample_ohlc(px, "4h"), p.h4_swing, "h4")
    h1 = add_swings(px.copy(), p.h1_swing, "h1")

    # Book logic: lower-TF prior low break can mark short-term low -> add long;
    # prior high break can mark short-term high -> add short.
    def contrarian_add(df, prefix):
        long_add = confirmed_break_down(df, prefix, p)
        short_add = confirmed_break_up(df, prefix, p)
        return pd.DataFrame({"long_add":long_add.astype(int),"short_add":short_add.astype(int)}, index=df.index)

    a12 = contrarian_add(h12, "h12").rename(columns={"long_add":"l12","short_add":"s12"})
    a4 = contrarian_add(h4, "h4").rename(columns={"long_add":"l4","short_add":"s4"})
    a1 = contrarian_add(h1, "h1").rename(columns={"long_add":"l1","short_add":"s1"})

    idx = px.index
    out = pd.DataFrame(index=idx)
    for src, cols in [(a12,["l12","s12"]),(a4,["l4","s4"]),(a1,["l1","s1"])]:
        tmp = pd.merge_asof(
            pd.DataFrame({"time":idx}),
            src.reset_index().rename(columns={src.index.name or "index":"time"}).sort_values("time"),
            on="time", direction="backward"
        ).set_index("time")
        for c in cols:
            out[c] = tmp[c].fillna(0)
    # require at least one completed lower-TF extreme event recently
    out["long_add_score"] = out[["l12","l4","l1"]].sum(axis=1)
    out["short_add_score"] = out[["s12","s4","s1"]].sum(axis=1)
    return out

def build_turtle(px: pd.DataFrame, p: Params):
    h4 = resample_ohlc(px, "4h", label="left")
    h4["atr"] = atr(h4, 20)
    h4["hi20"] = h4.high.shift(1).rolling(p.turtle_fast).max()
    h4["lo20"] = h4.low.shift(1).rolling(p.turtle_fast).min()
    h4["hi55"] = h4.high.shift(1).rolling(p.turtle_slow).max()
    h4["lo55"] = h4.low.shift(1).rolling(p.turtle_slow).min()
    h4["exit_hi"] = h4.high.shift(1).rolling(p.turtle_exit).max()
    h4["exit_lo"] = h4.low.shift(1).rolling(p.turtle_exit).min()

    state = 0.0
    out = []
    entry = np.nan
    for _, r in h4.iterrows():
        if state == 0:
            if np.isfinite(r.hi55) and r.close > r.hi55:
                state = 1.0
                entry = r.close
            elif np.isfinite(r.lo55) and r.close < r.lo55:
                state = -1.0
                entry = r.close
        elif state > 0:
            if np.isfinite(r.exit_lo) and r.close < r.exit_lo:
                state = 0.0
                entry = np.nan
        else:
            if np.isfinite(r.exit_hi) and r.close > r.exit_hi:
                state = 0.0
                entry = np.nan
        out.append(state)
    h4["turtle_regime"] = out
    return h4[["turtle_regime","atr"]]

def merge_features(px: pd.DataFrame, p: Params):
    book = build_book_regime(px, p)
    adds = lower_tf_add_signals(px, p)
    turtle = build_turtle(px, p)

    base = pd.DataFrame(index=px.index)
    base["close"] = px.close
    base["ret"] = px.close.pct_change().fillna(0.0)
    base["atr1h"] = atr(px, 14)
    rv = base["ret"].rolling(24*30, min_periods=24*10).std()*np.sqrt(24*365)
    base["rv"] = rv

    def asof_join(left, right):
        return pd.merge_asof(
            left.reset_index().rename(columns={left.index.name or "index":"time"}).sort_values("time"),
            right.reset_index().rename(columns={right.index.name or "index":"time"}).sort_values("time"),
            on="time", direction="backward"
        ).set_index("time")

    x = asof_join(base, book)
    x = x.join(adds, how="left")
    x = asof_join(x, turtle)
    return x.ffill()

def target_exposure(x: pd.DataFrame, p: Params, mode: str):
    exp = pd.Series(0.0, index=x.index)
    add_count = 0
    last_add_price = np.nan
    last_side = 0

    peak_equity_proxy = 1.0
    equity_proxy = 1.0

    for i in range(1, len(x)):
        r = x.iloc[i]
        prev = x.iloc[i-1]

        # Research drawdown proxy based on previous target
        equity_proxy *= max(1e-9, 1.0 + exp.iloc[i-1]*r.ret)
        peak_equity_proxy = max(peak_equity_proxy, equity_proxy)
        dd = equity_proxy/peak_equity_proxy - 1.0

        book_side = int(np.sign(r.d_regime)) if np.isfinite(r.d_regime) else 0
        turtle_side = int(np.sign(r.turtle_regime)) if np.isfinite(r.turtle_regime) else 0

        if mode == "BOOK":
            side = book_side
        elif mode == "TURTLE":
            side = turtle_side
        else:
            # HYBRID: Daily book regime is primary. Turtle can confirm or veto.
            if book_side == 0:
                side = turtle_side
            elif turtle_side == 0 or turtle_side == book_side:
                side = book_side
            else:
                side = 0

        if side != last_side:
            add_count = 0
            last_add_price = r.close if side != 0 else np.nan
            last_side = side

        if side == 0:
            raw = 0.0
        else:
            # volatility targeting
            vol_scale = 1.0
            if np.isfinite(r.rv) and r.rv > 0:
                vol_scale = min(1.25, max(0.35, p.vol_target/r.rv))

            conf = float(r.confidence) if np.isfinite(r.confidence) else 0.33
            raw = side * p.base_exposure * (0.65 + 0.35*conf) * vol_scale

            if mode != "TURTLE" and add_count < p.max_adds:
                atrv = r.atr1h
                favorable = (
                    (side > 0 and np.isfinite(last_add_price) and r.close >= last_add_price + p.add_trigger_atr*atrv) or
                    (side < 0 and np.isfinite(last_add_price) and r.close <= last_add_price - p.add_trigger_atr*atrv)
                )
                extreme = (side > 0 and r.long_add_score >= 1) or (side < 0 and r.short_add_score >= 1)
                if favorable and extreme:
                    add_count += 1
                    last_add_price = r.close

            raw += side * p.add_exposure * add_count

        raw = float(np.clip(raw, -p.max_exposure, p.max_exposure))

        # Drawdown defense, never adds risk while drawdown is severe.
        if dd <= -p.hard_dd:
            raw = 0.0
        elif dd <= -p.soft_dd:
            raw *= 0.5

        # execute next bar: shift applied outside
        exp.iloc[i] = raw

    return exp.shift(1).fillna(0.0)

def map_funding_to_hourly(index: pd.DatetimeIndex, funding: pd.Series):
    s = pd.Series(0.0, index=index)
    if funding is None or len(funding) == 0:
        return s
    # Map funding timestamp to nearest prior hourly index.
    pos = index.searchsorted(funding.index, side="right") - 1
    good = (pos >= 0) & (pos < len(index))
    for p, v in zip(pos[good], funding.iloc[np.where(good)[0]]):
        s.iloc[p] += float(v)
    return s

def performance(name, x, exposure, funding, costs: Costs, initial: float):
    ret = x["ret"].fillna(0.0)
    delta = exposure.diff().abs().fillna(exposure.abs())
    trading_cost = delta * (costs.fee_bps + costs.slippage_bps) / 10000.0

    fr = map_funding_to_hourly(x.index, funding)
    # positive funding: longs pay, shorts receive
    funding_cost = exposure * fr * costs.funding_mult

    strat_ret = exposure.shift(1).fillna(0.0)*ret - trading_cost - funding_cost
    equity = initial * (1.0 + strat_ret).cumprod()

    rollmax = equity.cummax()
    dd = equity/rollmax - 1.0
    years = max((equity.index[-1]-equity.index[0]).total_seconds()/(365.25*24*3600), 1/365)
    cagr = (equity.iloc[-1]/initial)**(1/years)-1
    mdd = dd.min()
    ann = np.sqrt(24*365)
    sharpe = (strat_ret.mean()/strat_ret.std()*ann) if strat_ret.std() > 0 else np.nan
    downside = strat_ret[strat_ret < 0].std()
    sortino = (strat_ret.mean()/downside*ann) if downside and downside > 0 else np.nan

    gross_profit = strat_ret[strat_ret > 0].sum()
    gross_loss = -strat_ret[strat_ret < 0].sum()
    pf = gross_profit/gross_loss if gross_loss > 0 else np.nan
    calmar = cagr/abs(mdd) if mdd < 0 else np.nan

    out = pd.DataFrame({
        "equity": equity,
        "exposure": exposure,
        "ret": strat_ret,
        "drawdown": dd,
        "trading_cost": trading_cost,
        "funding_cost": funding_cost
    })

    yearly = out["equity"].resample("YE").last().pct_change()
    if len(yearly):
        first_year_end = out["equity"].loc[:yearly.index[0]]
        yearly.iloc[0] = first_year_end.iloc[-1]/initial - 1.0

    summary = {
        "strategy": name,
        "initial": initial,
        "final": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1]/initial - 1.0),
        "CAGR": float(cagr),
        "MDD": float(mdd),
        "Sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "Sortino": float(sortino) if np.isfinite(sortino) else None,
        "Calmar": float(calmar) if np.isfinite(calmar) else None,
        "PF": float(pf) if np.isfinite(pf) else None,
        "avg_abs_exposure": float(exposure.abs().mean()),
        "max_abs_exposure": float(exposure.abs().max()),
        "turnover": float(delta.sum()),
        "trading_cost_sum": float((trading_cost * equity.shift(1).fillna(initial)).sum()),
        "funding_cost_sum": float((funding_cost * equity.shift(1).fillna(initial)).sum()),
    }
    return summary, yearly, out

def run(args):
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)

    end = args.end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    px, funding = download_data(args.start, end, cache)
    p = Params()
    c = Costs()

    x = merge_features(px, p)
    x = x.dropna(subset=["close"]).copy()

    results = []
    yearly_frames = []

    for mode in ["BOOK","TURTLE","HYBRID"]:
        exp = target_exposure(x, p, mode)
        summary, yearly, eq = performance(mode, x, exp, funding, c, args.initial)
        results.append(summary)
        ydf = yearly.rename(mode).to_frame()
        yearly_frames.append(ydf)
        eq.to_csv(outdir / f"equity_{mode.lower()}.csv")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(outdir / "summary.csv", index=False)

    yearly_df = pd.concat(yearly_frames, axis=1)
    yearly_df.index = yearly_df.index.year
    yearly_df.to_csv(outdir / "yearly.csv")

    # Benchmark gates against known V9.2 baseline.
    gate = summary_df.copy()
    gate["beat_v92_cagr"] = gate["CAGR"] > 0.159790195
    gate["mdd_le_15pct"] = gate["MDD"].abs() <= 0.15
    gate["stage1"] = (gate["CAGR"] >= 0.18) & gate["mdd_le_15pct"]
    gate["stage2"] = (gate["CAGR"] >= 0.20) & gate["mdd_le_15pct"]
    gate.to_csv(outdir / "gate.csv", index=False)

    (outdir / "config.json").write_text(json.dumps({
        "params": asdict(p),
        "costs": asdict(c),
        "start_requested": args.start,
        "start_effective": str(px.index.min()),
        "end": str(px.index.max()),
        "rows": len(px),
        "funding_observations": len(funding),
        "baseline_v92": {
            "CAGR": 0.159790195,
            "MDD": -0.146337566,
            "Sharpe": 1.1644946,
            "PF": 1.3191059
        }
    }, indent=2), encoding="utf-8")

    print("\n=== BTC BOOK+TURTLE HYBRID BACKTEST ===")
    print(summary_df[["strategy","final","total_return","CAGR","MDD","Sharpe","PF"]].to_string(index=False))
    print("\n=== GATES ===")
    print(gate[["strategy","beat_v92_cagr","mdd_le_15pct","stage1","stage2"]].to_string(index=False))
    print(f"\nResults saved to: {outdir.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-08-17")
    ap.add_argument("--end", default=None)
    ap.add_argument("--initial", type=float, default=10000.0)
    ap.add_argument("--cache", default="data_cache_book_turtle")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
