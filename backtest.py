#!/usr/bin/env python3
"""
BTC AI EA — V10.5 Hybrid Trend/Convex Futures Backtester
==========================================================

V10.5 hybrid objective:
- Binance USD-M BTCUSDT perpetual futures price proxy with historical funding.
- Preserve V10 frozen-decision and adaptive-cost stress integrity.
- Restore a persistent V9-style trend core so base exposure is not blocked by a
  short-horizon cost-admission gate.
- Apply the economic admission gate only to tactical/convex risk additions.
- Strong and weak bull regimes carry different core exposure.
- Convex additions require confirmed favorable breakout continuation.
- Risk reductions always execute immediately.
- No martingale, averaging down, simultaneous long/short hedge, or lookahead.
"""

from __future__ import annotations
import argparse, io, json, time, urllib.error, urllib.request, zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

VISION = "https://data.binance.vision/data/futures/um"
KLINE_COLS = [
    "open_time","open","high","low","close","volume","close_time",
    "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
]
FUTURES_LAUNCH = pd.Timestamp("2019-09-08", tz="UTC")

@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 5.5
    slippage_bps: float = 2.0
    funding_mult: float = 1.0

@dataclass(frozen=True)
class Rules:
    hard_drawdown: float = 0.15
    soft_drawdown: float = 0.10
    max_long: float = 0.70
    admission_margin: float = 1.50
    min_order_delta: float = 0.035
    hold_bars: int = 2
    edge_horizon_bars: int = 42
    trend_edge_weight: float = 0.30
    breakout_edge_weight: float = 0.14

BASE_RULES = Rules()

@dataclass(frozen=True)
class Candidate:
    name: str
    admission_margin: float
    min_order_delta: float
    hold_bars: int
    strong_core: float = 0.30
    weak_core: float = 0.10
    tactical_add: float = 0.14

CANDIDATES = [
    Candidate("V105A_CORE30_W10", 1.50, 0.035, 2, 0.30, 0.10, 0.14),
    Candidate("V105B_CORE28_W10", 1.50, 0.035, 2, 0.28, 0.10, 0.14),
    Candidate("V105C_CORE30_W08", 1.75, 0.035, 2, 0.30, 0.08, 0.12),
]

def fetch_bytes(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent":"btc-ai-ea-v10.5/1.0"})
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
        time.sleep(1.5*(i+1))
    raise RuntimeError(f"download failed: {url}: {last}")

def month_starts(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    last = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= last:
        yield cur
        cur += pd.offsets.MonthBegin(1)

def parse_kline(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), header=None)
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
    # Binance funding archives have changed column labels over time.
    # Parse defensively by locating a timestamp-like column and a small decimal-rate column.
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    cols = {str(c).lower(): c for c in df.columns}
    tcol = None
    for key in ["calc_time","funding_time","fundingtime","time","timestamp"]:
        if key in cols:
            tcol = cols[key]; break
    rcol = None
    for key in ["last_funding_rate","funding_rate","fundingrate"]:
        if key in cols:
            rcol = cols[key]; break
    if tcol is None or rcol is None:
        # Headerless fallback.
        df = pd.read_csv(io.BytesIO(raw), header=None)
        if df.shape[1] < 2:
            return pd.Series(dtype=float)
        best_t = None
        for c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() > 0.8 and x.dropna().median() > 1e11:
                best_t = c; break
        best_r = None
        for c in df.columns:
            if c == best_t: continue
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() > 0.8 and x.dropna().abs().median() < 0.01:
                best_r = c; break
        if best_t is None or best_r is None:
            return pd.Series(dtype=float)
        tcol, rcol = best_t, best_r
    t = pd.to_numeric(df[tcol], errors="coerce")
    unit = "us" if t.dropna().median() > 1e14 else "ms"
    idx = pd.to_datetime(t, unit=unit, utc=True, errors="coerce")
    rate = pd.to_numeric(df[rcol], errors="coerce")
    s = pd.Series(rate.to_numpy(), index=idx).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

def download_data(start: str, end: str, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    s = max(pd.Timestamp(start, tz="UTC"), FUTURES_LAUNCH)
    e = pd.Timestamp(end, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    current_month = pd.Timestamp(now.year, now.month, 1, tz="UTC")
    frames, funding = [], []

    monthly_end = min(e, current_month - pd.Timedelta(hours=1))
    for m in month_starts(s, monthly_end):
        ym = m.strftime("%Y-%m")
        kp = cache / f"BTCUSDT-1h-{ym}.zip"
        ku = f"{VISION}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{ym}.zip"
        if not kp.exists():
            try: kp.write_bytes(fetch_bytes(ku))
            except urllib.error.HTTPError as ex:
                if ex.code == 404: continue
                raise
        with zipfile.ZipFile(kp) as z:
            n = next(x for x in z.namelist() if x.endswith(".csv"))
            frames.append(parse_kline(z.read(n)))

        fp = cache / f"BTCUSDT-fundingRate-{ym}.zip"
        fu = f"{VISION}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{ym}.zip"
        if not fp.exists():
            try: fp.write_bytes(fetch_bytes(fu))
            except urllib.error.HTTPError as ex:
                if ex.code != 404: raise
        if fp.exists():
            try:
                with zipfile.ZipFile(fp) as z:
                    n = next(x for x in z.namelist() if x.endswith(".csv"))
                    funding.append(parse_funding(z.read(n)))
            except Exception as ex:
                print(f"[WARN] funding parse {ym}: {ex}")

    # completed daily archives for current month
    if e >= current_month:
        d0 = max(s.normalize(), current_month)
        d1 = min(e.normalize(), (now - pd.Timedelta(days=1)).normalize())
        if d1 >= d0:
            for d in pd.date_range(d0, d1, freq="1D", tz="UTC"):
                ds = d.strftime("%Y-%m-%d")
                kp = cache / f"BTCUSDT-1h-{ds}.zip"
                ku = f"{VISION}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{ds}.zip"
                if not kp.exists():
                    try: kp.write_bytes(fetch_bytes(ku))
                    except urllib.error.HTTPError as ex:
                        if ex.code == 404: continue
                        raise
                with zipfile.ZipFile(kp) as z:
                    n = next(x for x in z.namelist() if x.endswith(".csv"))
                    frames.append(parse_kline(z.read(n)))

    if not frames:
        raise RuntimeError("No BTC perpetual futures data available.")
    px = pd.concat(frames).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    px = px.loc[(px.index >= s) & (px.index <= e)]
    bad = ((px.high < px[["open","close","low"]].max(axis=1)) |
           (px.low > px[["open","close","high"]].min(axis=1)))
    if bad.any():
        raise RuntimeError(f"OHLC integrity failure: {int(bad.sum())}")
    fr = pd.concat(funding).sort_index() if funding else pd.Series(dtype=float)
    if len(fr):
        fr = fr[~fr.index.duplicated(keep="last")]
        fr = fr.loc[(fr.index >= s) & (fr.index <= e)]
    print(f"[DATA] futures rows={len(px):,} {px.index.min()} -> {px.index.max()}")
    print(f"[DATA] funding observations={len(fr):,}")
    return px, fr

def wilder(s: pd.Series, n: int):
    return s.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def atr(df: pd.DataFrame, n: int=14):
    prev = df.close.shift(1)
    tr = pd.concat([(df.high-df.low).abs(),
                    (df.high-prev).abs(),
                    (df.low-prev).abs()], axis=1).max(axis=1)
    return wilder(tr,n)

def build_features(px1h: pd.DataFrame):
    h4 = px1h.resample("4h", closed="left", label="left").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    h4["atr"] = atr(h4,14)
    h4["entry_hi"] = h4.high.shift(1).rolling(40).max()
    h4["exit_lo"] = h4.low.shift(1).rolling(20).min()
    h4["ret4"] = h4.close.pct_change()
    h4["rv4"] = h4.ret4.rolling(42).std()*np.sqrt(6*365)

    d = px1h.resample("1D", closed="left", label="right").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    d["fast"] = d.close.ewm(span=100, adjust=False, min_periods=100).mean()
    d["slow"] = d.close.ewm(span=250, adjust=False, min_periods=250).mean()
    d["slow_slope"] = d.slow/d.slow.shift(30)-1
    dr = d.close.pct_change()
    d["rv30"] = dr.rolling(30,min_periods=20).std()*np.sqrt(365)
    d["rv90med"] = d.rv30.rolling(90,min_periods=45).median()
    d["rv_ratio"] = d.rv30/d.rv90med.replace(0,np.nan)
    d["ret20"] = d.close.pct_change(20)
    d["high20"] = d.close.shift(1).rolling(20,min_periods=10).max()
    d["dd20"] = d.close/d.high20-1
    d["dclose"] = d.close

    left = h4.reset_index().rename(columns={h4.index.name or "index":"time"})
    right = d[["fast","slow","slow_slope","rv30","rv_ratio","ret20","dd20","dclose"]].reset_index()
    right = right.rename(columns={right.columns[0]:"time"})
    x = pd.merge_asof(left.sort_values("time"), right.sort_values("time"),
                      on="time", direction="backward").set_index("time")
    return x

def target_components(r, cand: Candidate):
    if not np.isfinite(r.slow) or not np.isfinite(r.fast):
        return 0.0, 0.0
    above_slow = bool(r.dclose > r.slow)
    strong = above_slow and (r.fast > r.slow) and (r.slow_slope > 0)
    weak = above_slow and not strong
    if not (strong or weak):
        return 0.0, 0.0

    core = cand.strong_core if strong else cand.weak_core
    if np.isfinite(r.rv30) and r.rv30 > 0:
        core *= float(np.clip(0.55/r.rv30, 0.55, 1.0))
    if np.isfinite(r.rv_ratio) and r.rv_ratio > 1.45:
        core *= 0.68
    if np.isfinite(r.ret20) and r.ret20 < -0.10:
        core *= 0.50
    if np.isfinite(r.dd20) and r.dd20 < -0.15:
        return 0.0, 0.0

    tactical = 0.0
    if strong and np.isfinite(r.entry_hi) and np.isfinite(r.atr) and r.atr > 0 and r.close > r.entry_hi:
        strength = (r.close-r.entry_hi)/r.atr
        if strength >= 0.25:
            tactical += cand.tactical_add
        if strength >= 1.25:
            tactical += 0.10
        if strength >= 2.50:
            tactical += 0.08

    core = float(np.clip(core, 0.0, BASE_RULES.max_long))
    tactical = float(np.clip(tactical, 0.0, BASE_RULES.max_long-core))
    return core, tactical


def desired_target(r, cand: Candidate):
    core, tactical = target_components(r, cand)
    return float(np.clip(core+tactical, 0.0, BASE_RULES.max_long))


def expected_edge_bps(r):
    # Conservative, causal edge proxy over the planned 24h horizon.
    if not np.isfinite(r.fast) or not np.isfinite(r.slow) or r.slow <= 0:
        return 0.0
    trend = max(0.0, r.fast/r.slow - 1.0)
    breakout = 0.0
    if np.isfinite(r.entry_hi) and np.isfinite(r.atr) and r.atr > 0:
        breakout = max(0.0, (r.close-r.entry_hi)/r.atr)
    return float(10000*(0.30*trend) + 10.0*0.14*min(breakout,6.0))

def map_funding_to_4h(index, funding: pd.Series):
    out = pd.Series(0.0,index=index)
    if funding is None or len(funding)==0:
        return out
    # Funding timestamp belongs to the nearest prior 4H decision interval.
    bucket = funding.groupby(funding.index.floor("4h")).sum()
    return bucket.reindex(index,fill_value=0.0)

def simulate(x, funding, cand: Candidate, cost: CostModel,
             initial=10000.0, adaptive_cost=True, frozen_trades=None):
    idx = x.index
    funding4 = map_funding_to_4h(idx,funding)
    eq = float(initial); peak=eq; exposure=0.0
    last_trade_bar = -10**9
    trades=[]; rows=[]; hard_stop=False
    one_way = cost.fee_bps + cost.slippage_bps

    for i in range(1,len(x)):
        r0=x.iloc[i-1]; r=x.iloc[i]
        ret = float(r.open/r0.open-1.0) if r0.open>0 else 0.0

        # PnL from exposure held over the previous 4H interval.
        eq *= max(1e-12, 1.0 + exposure*ret)
        fr = float(funding4.iloc[i-1]) if i-1 < len(funding4) else 0.0
        if exposure > 0 and fr != 0:
            eq *= max(1e-12, 1.0 - exposure*fr*cost.funding_mult)

        peak=max(peak,eq)
        dd=eq/peak-1.0
        if dd <= -BASE_RULES.hard_drawdown:
            target=0.0
            hard_stop=True
        else:
            target=desired_target(r0, cand)
            if dd <= -BASE_RULES.soft_drawdown:
                target=min(target, exposure*0.5)

        if frozen_trades is not None:
            # Exact base decisions replayed by bar index.
            new_exp = frozen_trades.get(i, exposure)
            if abs(new_exp-exposure)>1e-12:
                turn=abs(new_exp-exposure)
                eq *= max(1e-12,1.0-turn*one_way/10000.0)
                exposure=float(new_exp)
        else:
            delta=target-exposure
            urgent = target < exposure - 1e-12
            execute=False
            if urgent and abs(delta)>1e-12:
                execute=True
            elif delta > 0:
                core_target, tactical_target = target_components(r0, cand)
                enough_time = (i-last_trade_bar) >= cand.hold_bars
                if exposure + 1e-12 < core_target:
                    core_delta = core_target - exposure
                    if core_delta >= cand.min_order_delta and enough_time:
                        target = core_target
                        delta = target - exposure
                        execute = True
                elif tactical_target > 0:
                    threshold = cand.admission_margin * (2.0*one_way)
                    edge = expected_edge_bps(r0)
                    enough_size = delta >= cand.min_order_delta
                    if adaptive_cost and edge >= threshold and enough_time and enough_size:
                        execute=True
            if execute:
                turn=abs(delta)
                eq *= max(1e-12,1.0-turn*one_way/10000.0)
                exposure=float(target)
                last_trade_bar=i
                trades.append((i,exposure))
        peak=max(peak,eq)
        rows.append((idx[i],eq,exposure,eq/peak-1.0,fr))

    curve=pd.DataFrame(rows,columns=["time","equity","exposure","drawdown","funding_rate"]).set_index("time")
    trade_map={i:e for i,e in trades}
    return curve, trade_map, hard_stop

def metrics(curve, initial=10000.0):
    if curve.empty:
        return dict(cagr=np.nan,mdd=np.nan,sharpe=np.nan,pf=np.nan,final=np.nan)
    years=max((curve.index[-1]-curve.index[0]).total_seconds()/(365.25*86400),1e-9)
    final=float(curve.equity.iloc[-1])
    cagr=(final/initial)**(1/years)-1
    mdd=float(curve.drawdown.min())
    rr=curve.equity.pct_change().dropna()
    sharpe=float(rr.mean()/rr.std()*np.sqrt(6*365)) if len(rr)>2 and rr.std()>0 else np.nan
    gains=float(rr[rr>0].sum()); losses=float(-rr[rr<0].sum())
    pf=gains/losses if losses>0 else np.nan
    return dict(cagr=cagr,mdd=mdd,sharpe=sharpe,pf=pf,final=final)

def run_one(x,funding,cand,initial):
    base_cost=CostModel()
    base, trades, hard = simulate(x,funding,cand,base_cost,initial,adaptive_cost=True)
    adaptive2, _, hard_a = simulate(x,funding,cand,CostModel(11.0,4.0,2.0),initial,adaptive_cost=True)
    frozen2, _, hard_f = simulate(x,funding,cand,CostModel(11.0,4.0,2.0),initial,
                                  adaptive_cost=False,frozen_trades=trades)
    delay = x.iloc[1:].copy()
    # +4H delay by shifting decision features one extra row while preserving prices.
    signal_cols=["fast","slow","slow_slope","rv30","rv_ratio","ret20","dd20","dclose","entry_hi","atr"]
    xd=x.copy()
    xd[signal_cols]=xd[signal_cols].shift(1)
    delayed,_,hard_d=simulate(xd,funding,cand,base_cost,initial,adaptive_cost=True)
    return {
        "base":metrics(base,initial),"adaptive2x":metrics(adaptive2,initial),
        "frozen2x":metrics(frozen2,initial),"+4h_delay":metrics(delayed,initial),
        "hard_stop":{"base":hard,"adaptive2x":hard_a,"frozen2x":hard_f,"delay":hard_d},
        "_curve":base,"_trades":trades
    }

def robustness(x,funding,cand,initial):
    variants=[
        ("M125",Candidate(cand.name+"_M125",1.25,cand.min_order_delta,cand.hold_bars,cand.strong_core,cand.weak_core,cand.tactical_add)),
        ("BASE",cand),
        ("M200",Candidate(cand.name+"_M200",2.00,cand.min_order_delta,cand.hold_bars,cand.strong_core,cand.weak_core,cand.tactical_add)),
        ("D025",Candidate(cand.name+"_D025",cand.admission_margin,0.025,cand.hold_bars,cand.strong_core,cand.weak_core,cand.tactical_add)),
        ("D050",Candidate(cand.name+"_D050",cand.admission_margin,0.050,cand.hold_bars,cand.strong_core,cand.weak_core,cand.tactical_add)),
    ]
    out=[]
    for label,v in variants:
        r=run_one(x,funding,v,initial)
        passed=(r["base"]["mdd"]>-0.15 and r["adaptive2x"]["mdd"]>-0.15 and
                r["frozen2x"]["mdd"]>-0.15 and not any(r["hard_stop"].values()))
        out.append({"variant":label,"pass":bool(passed),
                    "base_cagr":r["base"]["cagr"],"base_mdd":r["base"]["mdd"]})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",default="2017-08-17")
    ap.add_argument("--end",default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    ap.add_argument("--initial",type=float,default=10000)
    ap.add_argument("--cache",default=".cache_v105")
    ap.add_argument("--results",default="results")
    a=ap.parse_args()
    outdir=Path(a.results); outdir.mkdir(parents=True,exist_ok=True)
    px,funding=download_data(a.start,a.end,Path(a.cache))
    x=build_features(px).dropna(subset=["open","close"]).copy()
    if len(x)<1000:
        raise RuntimeError("insufficient feature history")

    results=[]
    for c in CANDIDATES:
        print(f"[RUN] {c.name}",flush=True)
        r=run_one(x,funding,c,a.initial)
        rob=robustness(x,funding,c,a.initial)
        split=int(len(x)*0.70)
        oos=run_one(x.iloc[split:].copy(),funding,c,a.initial)
        rob_rate=sum(v["pass"] for v in rob)/len(rob)
        row={"candidate":asdict(c),"base":r["base"],"adaptive2x":r["adaptive2x"],
             "frozen2x":r["frozen2x"],"+4h_delay":r["+4h_delay"],
             "hard_stop":r["hard_stop"],"robustness":rob,"robustness_pass_rate":rob_rate,
             "oos":{"base":oos["base"],"adaptive2x":oos["adaptive2x"],"frozen2x":oos["frozen2x"]},
             "funding_observations":int(len(funding))}
        gate=(r["base"]["cagr"]>=0.10 and r["base"]["mdd"]>-0.15 and
              r["adaptive2x"]["mdd"]>-0.15 and r["frozen2x"]["mdd"]>-0.15 and
              r["+4h_delay"]["mdd"]>-0.15 and rob_rate>=0.60 and
              not any(r["hard_stop"].values()))
        row["development_gate"]=bool(gate)
        results.append(row)
        r["_curve"].to_csv(outdir/f"{c.name}_equity.csv")

    # Prefer gate pass, then robustness, then base Sharpe.
    def key(z):
        return (int(z["development_gate"]),z["robustness_pass_rate"],
                z["base"]["sharpe"] if np.isfinite(z["base"]["sharpe"]) else -99)
    selected=max(results,key=key)
    summary={
        "version":"V10.5",
        "architecture":"cost-admissible futures execution",
        "selected":selected["candidate"]["name"],
        "development_gate_passed":selected["development_gate"],
        "funding_available":bool(len(funding)),
        "results":results,
    }
    (outdir/"v10_summary.json").write_text(json.dumps(summary,indent=2,default=float))
    pd.DataFrame([{
        "candidate":z["candidate"]["name"],
        "gate":z["development_gate"],
        "base_cagr":z["base"]["cagr"],"base_mdd":z["base"]["mdd"],
        "base_sharpe":z["base"]["sharpe"],"base_pf":z["base"]["pf"],
        "adaptive2x_mdd":z["adaptive2x"]["mdd"],
        "frozen2x_mdd":z["frozen2x"]["mdd"],
        "delay_mdd":z["+4h_delay"]["mdd"],
        "robustness_pass_rate":z["robustness_pass_rate"],
        "oos_cagr":z["oos"]["base"]["cagr"],"oos_mdd":z["oos"]["base"]["mdd"],
    } for z in results]).to_csv(outdir/"v10_scorecard.csv",index=False)
    print(json.dumps({
        "selected":summary["selected"],
        "gate":summary["development_gate_passed"],
        "base":selected["base"],
        "adaptive2x":selected["adaptive2x"],
        "frozen2x":selected["frozen2x"],
        "delay":selected["+4h_delay"],
        "robustness":selected["robustness_pass_rate"],
        "funding_observations":selected["funding_observations"],
    },indent=2,default=float))

if __name__=="__main__":
    main()
