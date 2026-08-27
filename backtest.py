#!/usr/bin/env python3
"""
BTC AI EA — V11.0 V9-Engine / Futures-Execution Backtester
==========================================================

Purpose
-------
V11 stops tuning the V10 admission gate and restores the proven V9 return engine:
- V9 strong/weak bull regime core.
- V9 20/50/200-day market-risk states with recovery hysteresis.
- V9 ATR-normalized convex 3-stage additions.
- 4H breakout tactical overlay.
- Binance USD-M BTCUSDT perpetual futures price proxy and historical funding.
- Risk reductions always execute.
- Cost admission applies only to tactical/convex risk additions.
- Adaptive 2x and frozen-decision 2x cost stress are both required.
- +4H execution delay, OOS and local robustness are measured.
- No martingale, averaging down, simultaneous long/short hedge, or lookahead.
"""

from __future__ import annotations
import argparse, io, json, time, urllib.error, urllib.request, zipfile
from dataclasses import dataclass, asdict, replace
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
class CostModel:
    fee_bps: float = 5.5
    slippage_bps: float = 2.0
    funding_mult: float = 1.0

@dataclass(frozen=True)
class Strategy:
    name: str
    fast_days: int = 100
    slow_days: int = 250
    slope_days: int = 30
    strong_long: float = 0.256
    weak_long: float = 0.088
    tactical_long: float = 0.144
    breakout_4h: int = 40
    exit_4h: int = 20
    vol_target: float = 0.55
    vol_floor_scale: float = 0.55
    max_long: float = 0.704
    convex_add1: float = 0.128
    convex_add2: float = 0.136
    convex_add3: float = 0.144
    convex_trigger1_atr: float = 1.44
    convex_trigger2_atr: float = 3.36
    convex_trigger3_atr: float = 5.76
    convex_stepdown_atr: float = 2.16
    tactical_min_stage: int = 2
    risk_fast_days: int = 20
    risk_mid_days: int = 50
    risk_slow_days: int = 200
    mom5_cut: float = -0.10
    mom20_cut: float = -0.17
    high20_cut: float = -0.15
    rv_ratio_cut: float = 1.45
    caution_scale: float = 0.68
    defense_scale: float = 0.32
    panic_scale: float = 0.0
    recovery_days: int = 5
    admission_margin: float = 1.50
    min_add_delta: float = 0.025
    rebalance_deadband: float = 0.040
    min_rebalance_bars: int = 1

CANDIDATES = [
    Strategy("V11A_V9CORE"),
    Strategy("V11B_CORE95", strong_long=0.2432, weak_long=0.0836,
             convex_add1=0.1216, convex_add2=0.1292, convex_add3=0.1368),
    Strategy("V11C_CORE105", strong_long=0.2688, weak_long=0.0924,
             convex_add1=0.1344, convex_add2=0.1428, convex_add3=0.1512,
             max_long=0.72),
]

@dataclass(frozen=True)
class RiskRules:
    soft_drawdown: float = 0.10
    hard_drawdown: float = 0.15

def fetch_bytes(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent":"btc-ai-ea-v11/1.0"})
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
    cur = pd.Timestamp(start.year,start.month,1,tz="UTC")
    last = pd.Timestamp(end.year,end.month,1,tz="UTC")
    while cur <= last:
        yield cur
        cur += pd.offsets.MonthBegin(1)

def parse_kline(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if df.shape[1] < 6:
        raise ValueError("Unexpected Binance kline archive")
    df = df.iloc[:, :min(12,df.shape[1])]
    df.columns = KLINE_COLS[:df.shape[1]]
    t = pd.to_numeric(df["open_time"],errors="coerce")
    unit = "us" if t.dropna().median() > 1e14 else "ms"
    idx = pd.to_datetime(t,unit=unit,utc=True,errors="coerce")
    out = pd.DataFrame(index=idx)
    for c in ["open","high","low","close","volume"]:
        out[c] = pd.to_numeric(df[c],errors="coerce").to_numpy()
    return out.dropna().sort_index()

def parse_funding(raw: bytes) -> pd.Series:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    cols = {str(c).lower():c for c in df.columns}
    tcol = next((cols[k] for k in ["calc_time","funding_time","fundingtime","time","timestamp"] if k in cols),None)
    rcol = next((cols[k] for k in ["last_funding_rate","funding_rate","fundingrate"] if k in cols),None)
    if tcol is None or rcol is None:
        df = pd.read_csv(io.BytesIO(raw),header=None)
        if df.shape[1] < 2:
            return pd.Series(dtype=float)
        tcol = None; rcol = None
        for c in df.columns:
            x = pd.to_numeric(df[c],errors="coerce")
            if x.notna().mean() > .8 and x.dropna().median() > 1e11:
                tcol = c; break
        for c in df.columns:
            if c == tcol: continue
            x = pd.to_numeric(df[c],errors="coerce")
            if x.notna().mean() > .8 and x.dropna().abs().median() < .01:
                rcol = c; break
        if tcol is None or rcol is None:
            return pd.Series(dtype=float)
    t = pd.to_numeric(df[tcol],errors="coerce")
    unit = "us" if t.dropna().median() > 1e14 else "ms"
    idx = pd.to_datetime(t,unit=unit,utc=True,errors="coerce")
    rate = pd.to_numeric(df[rcol],errors="coerce")
    s = pd.Series(rate.to_numpy(),index=idx).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()

def download_data(start: str, end: str, cache: Path):
    cache.mkdir(parents=True,exist_ok=True)
    s = max(pd.Timestamp(start,tz="UTC"),FUTURES_LAUNCH)
    e = pd.Timestamp(end,tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    current_month = pd.Timestamp(now.year,now.month,1,tz="UTC")
    frames=[]; funding=[]
    monthly_end = min(e,current_month-pd.Timedelta(hours=1))
    for m in month_starts(s,monthly_end):
        ym=m.strftime("%Y-%m")
        kp=cache/f"BTCUSDT-1h-{ym}.zip"
        ku=f"{VISION}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{ym}.zip"
        if not kp.exists():
            try: kp.write_bytes(fetch_bytes(ku))
            except urllib.error.HTTPError as ex:
                if ex.code == 404: continue
                raise
        with zipfile.ZipFile(kp) as z:
            n=next(x for x in z.namelist() if x.endswith(".csv"))
            frames.append(parse_kline(z.read(n)))
        fp=cache/f"BTCUSDT-fundingRate-{ym}.zip"
        fu=f"{VISION}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{ym}.zip"
        if not fp.exists():
            try: fp.write_bytes(fetch_bytes(fu))
            except urllib.error.HTTPError as ex:
                if ex.code != 404: raise
        if fp.exists():
            try:
                with zipfile.ZipFile(fp) as z:
                    n=next(x for x in z.namelist() if x.endswith(".csv"))
                    funding.append(parse_funding(z.read(n)))
            except Exception as ex:
                print(f"[WARN] funding {ym}: {ex}")
    if e >= current_month:
        d0=max(s.normalize(),current_month)
        d1=min(e.normalize(),(now-pd.Timedelta(days=1)).normalize())
        if d1 >= d0:
            for d in pd.date_range(d0,d1,freq="1D",tz="UTC"):
                ds=d.strftime("%Y-%m-%d")
                kp=cache/f"BTCUSDT-1h-{ds}.zip"
                ku=f"{VISION}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{ds}.zip"
                if not kp.exists():
                    try: kp.write_bytes(fetch_bytes(ku))
                    except urllib.error.HTTPError as ex:
                        if ex.code == 404: continue
                        raise
                with zipfile.ZipFile(kp) as z:
                    n=next(x for x in z.namelist() if x.endswith(".csv"))
                    frames.append(parse_kline(z.read(n)))
    if not frames:
        raise RuntimeError("No futures data")
    px=pd.concat(frames).sort_index()
    px=px[~px.index.duplicated(keep="last")]
    px=px.loc[(px.index>=s)&(px.index<=e)]
    bad=((px.high<px[["open","close","low"]].max(axis=1))|
         (px.low>px[["open","close","high"]].min(axis=1)))
    if bad.any():
        raise RuntimeError(f"OHLC integrity failure: {int(bad.sum())}")
    fr=pd.concat(funding).sort_index() if funding else pd.Series(dtype=float)
    if len(fr):
        fr=fr[~fr.index.duplicated(keep="last")]
        fr=fr.loc[(fr.index>=s)&(fr.index<=e)]
    print(f"[DATA] futures rows={len(px):,} {px.index.min()} -> {px.index.max()}")
    print(f"[DATA] funding observations={len(fr):,}")
    return px,fr

def wilder(s: pd.Series,n:int):
    return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def atr(df: pd.DataFrame,n:int=14):
    prev=df.close.shift(1)
    tr=pd.concat([(df.high-df.low).abs(),(df.high-prev).abs(),(df.low-prev).abs()],axis=1).max(axis=1)
    return wilder(tr,n)

def build_features(px1h: pd.DataFrame,s:Strategy):
    h4=px1h.resample("4h",closed="left",label="left").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    h4["atr4h"]=atr(h4,14)
    h4["entry_hi"]=h4.high.shift(1).rolling(s.breakout_4h).max()
    h4["exit_lo"]=h4.low.shift(1).rolling(s.exit_4h).min()

    d=px1h.resample("1D",closed="left",label="right").agg(
        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    d["fast"]=d.close.ewm(span=s.fast_days,adjust=False,min_periods=s.fast_days).mean()
    d["slow"]=d.close.ewm(span=s.slow_days,adjust=False,min_periods=s.slow_days).mean()
    d["slow_slope"]=d.slow-d.slow.shift(s.slope_days)
    strong=(d.close>d.slow)&(d.fast>d.slow)&(d.slow_slope>0)
    weak=(d.close>d.slow)&~strong
    d["regime"]="NEUTRAL"
    d.loc[weak,"regime"]="BULL_WEAK"
    d.loc[strong,"regime"]="BULL_STRONG"

    d["risk_fast"]=d.close.ewm(span=s.risk_fast_days,adjust=False,min_periods=s.risk_fast_days).mean()
    d["risk_mid"]=d.close.ewm(span=s.risk_mid_days,adjust=False,min_periods=s.risk_mid_days).mean()
    d["risk_slow"]=d.close.ewm(span=s.risk_slow_days,adjust=False,min_periods=s.risk_slow_days).mean()
    ret=d.close.pct_change()
    d["ret5"]=d.close.pct_change(5)
    d["ret20"]=d.close.pct_change(20)
    d["high20"]=d.close.shift(1).rolling(20,min_periods=10).max()
    d["dd20"]=d.close/d.high20-1
    d["rv30"]=ret.rolling(30,min_periods=20).std()*np.sqrt(365)
    d["rv90_med"]=d.rv30.rolling(90,min_periods=45).median()
    d["rv_ratio"]=d.rv30/d.rv90_med.replace(0,np.nan)
    d["atr_pct"]=atr(d,14)/d.close
    d["shock_cut"]=d.atr_pct.rolling(540,min_periods=180).quantile(.95)
    d["shock"]=((d.atr_pct>d.shock_cut)|(ret.abs()>3*ret.rolling(60,min_periods=30).std())).fillna(False)
    d["dclose"]=d.close
    d["daily_seq"]=np.arange(len(d),dtype=int)

    left=h4.reset_index().rename(columns={h4.index.name or "index":"time"})
    right=d[["regime","rv30","rv_ratio","shock","risk_fast","risk_mid","risk_slow",
             "ret5","ret20","dd20","dclose","daily_seq"]].reset_index()
    right=right.rename(columns={right.columns[0]:"time"})
    return pd.merge_asof(left.sort_values("time"),right.sort_values("time"),
                         on="time",direction="backward").set_index("time")

STATE_LEVEL={"NORMAL":0,"CAUTION":1,"DEFENSIVE":2,"PANIC":3}
LEVEL_STATE={v:k for k,v in STATE_LEVEL.items()}

def raw_market_risk_state(r,s):
    warnings=0
    vals=[
        np.isfinite(r.dclose) and np.isfinite(r.risk_fast) and r.dclose<r.risk_fast,
        np.isfinite(r.dclose) and np.isfinite(r.risk_mid) and r.dclose<r.risk_mid,
        np.isfinite(r.dclose) and np.isfinite(r.risk_slow) and r.dclose<r.risk_slow,
        np.isfinite(r.risk_fast) and np.isfinite(r.risk_mid) and r.risk_fast<r.risk_mid,
        np.isfinite(r.risk_mid) and np.isfinite(r.risk_slow) and r.risk_mid<r.risk_slow,
        np.isfinite(r.ret5) and r.ret5<-0.05,
        np.isfinite(r.ret20) and r.ret20<-0.10,
        np.isfinite(r.rv_ratio) and r.rv_ratio>s.rv_ratio_cut,
    ]
    warnings=sum(bool(x) for x in vals)
    panic=((np.isfinite(r.ret5) and r.ret5<=s.mom5_cut) or
           (np.isfinite(r.ret20) and r.ret20<=s.mom20_cut) or
           (np.isfinite(r.dd20) and r.dd20<=s.high20_cut and
            np.isfinite(r.dclose) and np.isfinite(r.risk_mid) and r.dclose<r.risk_mid) or
           (bool(r.shock) and np.isfinite(r.ret5) and r.ret5<-0.035))
    if panic: return "PANIC"
    defensive=(warnings>=4 or
               (np.isfinite(r.dclose) and np.isfinite(r.risk_mid) and r.dclose<r.risk_mid and
                np.isfinite(r.ret20) and r.ret20<-0.07))
    if defensive: return "DEFENSIVE"
    if warnings>=1: return "CAUTION"
    return "NORMAL"

def hysteresis_update(current,raw,count,recovery_days):
    if current is None: return raw,0
    c=STATE_LEVEL[current]; rr=STATE_LEVEL[raw]
    if rr>=c: return raw,0
    count+=1
    if count<recovery_days: return current,count
    return LEVEL_STATE[max(rr,c-1)],0

def market_scale(state,s):
    return {"NORMAL":1.0,"CAUTION":s.caution_scale,
            "DEFENSIVE":s.defense_scale,"PANIC":s.panic_scale}[state]

def vol_scale(rv,s):
    if not np.isfinite(rv) or rv<=0: return 1.0
    return float(np.clip(s.vol_target/rv,s.vol_floor_scale,1.0))

def core_exposure(regime,s):
    if regime=="BULL_STRONG": return s.strong_long
    if regime=="BULL_WEAK": return s.weak_long
    return 0.0

def expected_tactical_edge_bps(r,stage,s):
    trend_bonus=0.0
    if r.regime=="BULL_STRONG": trend_bonus=25.0
    elif r.regime=="BULL_WEAK": trend_bonus=8.0
    breakout=0.0
    if np.isfinite(r.entry_hi) and np.isfinite(r.atr4h) and r.atr4h>0:
        breakout=max(0.0,(r.close-r.entry_hi)/r.atr4h)
    return float(trend_bonus+12.0*min(breakout,6.0)+8.0*stage)

def map_funding_to_4h(index,funding):
    out=pd.Series(0.0,index=index)
    if funding is None or len(funding)==0: return out
    bucket=funding.groupby(funding.index.floor("4h")).sum()
    return bucket.reindex(index,fill_value=0.0)

def model_target(r,s,risk_state,stage,tactical_on):
    vs=vol_scale(float(r.rv30),s)
    base=core_exposure(str(r.regime),s)
    adds=[s.convex_add1,s.convex_add2,s.convex_add3]
    convex=sum(adds[:max(0,min(stage,3))])
    tactical=s.tactical_long if tactical_on and stage>=s.tactical_min_stage else 0.0
    exp=(base+convex+tactical)*vs*market_scale(risk_state,s)
    return float(np.clip(exp,0.0,s.max_long))

def simulate(x,funding,s,cost,initial=10000.0,adaptive_cost=True,frozen_trades=None):
    funding4=map_funding_to_4h(x.index,funding)
    eq=float(initial); peak=eq; exposure=0.0
    stage=0; anchor=np.nan; peak_px=np.nan
    tactical=False
    risk_state=None; rec_count=0; last_daily_seq=None
    last_trade_i=-10**9
    trades=[]; rows=[]; hard=False
    turnover=0.0; costs_paid=0.0; rebals=0
    one_way=cost.fee_bps+cost.slippage_bps

    for i in range(1,len(x)):
        r0=x.iloc[i-1]; r=x.iloc[i]
        ret=float(r.open/r0.open-1.0) if r0.open>0 else 0.0
        eq*=max(1e-12,1.0+exposure*ret)
        fr=float(funding4.iloc[i-1])
        if exposure>0 and fr!=0:
            eq*=max(1e-12,1.0-exposure*fr*cost.funding_mult)

        peak=max(peak,eq); dd=eq/peak-1.0

        if int(r0.daily_seq)!=last_daily_seq:
            raw=raw_market_risk_state(r0,s)
            risk_state,rec_count=hysteresis_update(risk_state,raw,rec_count,s.recovery_days)
            last_daily_seq=int(r0.daily_seq)
        rs=risk_state or "NORMAL"

        # Causal convex campaign state from completed prior 4H bar.
        if str(r0.regime) not in ("BULL_STRONG","BULL_WEAK") or rs=="PANIC":
            stage=0; anchor=np.nan; peak_px=np.nan; tactical=False
        else:
            if not np.isfinite(anchor):
                if np.isfinite(r0.entry_hi) and r0.close>r0.entry_hi:
                    anchor=float(r0.close); peak_px=float(r0.close)
            if np.isfinite(anchor) and np.isfinite(r0.atr4h) and r0.atr4h>0:
                peak_px=max(float(peak_px),float(r0.close))
                fav=(float(r0.close)-anchor)/float(r0.atr4h)
                new_stage=stage
                if fav>=s.convex_trigger3_atr: new_stage=3
                elif fav>=s.convex_trigger2_atr: new_stage=max(new_stage,2)
                elif fav>=s.convex_trigger1_atr: new_stage=max(new_stage,1)
                giveback=(peak_px-float(r0.close))/float(r0.atr4h)
                if giveback>=s.convex_stepdown_atr and new_stage>0:
                    new_stage-=1
                    peak_px=float(r0.close)
                stage=max(0,min(3,new_stage))
                if stage>=s.tactical_min_stage and np.isfinite(r0.entry_hi) and r0.close>r0.entry_hi:
                    tactical=True
                if np.isfinite(r0.exit_lo) and r0.close<r0.exit_lo:
                    tactical=False
                    stage=0; anchor=np.nan; peak_px=np.nan

        target=0.0 if dd<=-RiskRules().hard_drawdown else model_target(r0,s,rs,stage,tactical)
        if dd<=-RiskRules().hard_drawdown:
            hard=True
        elif dd<=-RiskRules().soft_drawdown:
            target=min(target,exposure*0.5)

        if frozen_trades is not None:
            new_exp=frozen_trades.get(i,exposure)
            delta=float(new_exp-exposure)
            if abs(delta)>1e-12:
                cst=abs(delta)*one_way/10000.0
                eq*=max(1e-12,1.0-cst)
                costs_paid+=cst; turnover+=abs(delta); rebals+=1
                exposure=float(new_exp)
        else:
            delta=float(target-exposure)
            execute=False
            urgent=delta< -1e-12
            if urgent:
                execute=True
            elif delta>1e-12:
                strategic_core=core_exposure(str(r0.regime),s)*vol_scale(float(r0.rv30),s)*market_scale(rs,s)
                enough_time=(i-last_trade_i)>=s.min_rebalance_bars
                if exposure+1e-12 < strategic_core:
                    if strategic_core-exposure>=s.min_add_delta and enough_time:
                        target=strategic_core; delta=target-exposure; execute=True
                else:
                    threshold=s.admission_margin*2.0*one_way
                    edge=expected_tactical_edge_bps(r0,stage,s)
                    if adaptive_cost and edge>=threshold and abs(delta)>=s.min_add_delta and enough_time:
                        execute=True
            if abs(delta)<s.rebalance_deadband and not urgent:
                execute=False
            if execute:
                cst=abs(delta)*one_way/10000.0
                eq*=max(1e-12,1.0-cst)
                costs_paid+=cst; turnover+=abs(delta); rebals+=1
                exposure=float(target); last_trade_i=i
                trades.append((i,exposure))

        peak=max(peak,eq)
        rows.append((x.index[i],eq,exposure,eq/peak-1.0,fr,stage,rs))

    curve=pd.DataFrame(rows,columns=["time","equity","exposure","drawdown","funding_rate","convex_stage","risk_state"]).set_index("time")
    return curve,{i:e for i,e in trades},hard,{"turnover":turnover,"rebalances":rebals,"cost_fraction":costs_paid}

def metrics(curve,initial=10000.0):
    if curve.empty:
        return dict(cagr=np.nan,mdd=np.nan,sharpe=np.nan,pf=np.nan,final=np.nan,
                    exposure_time=np.nan,avg_exposure=np.nan,max_exposure=np.nan)
    years=max((curve.index[-1]-curve.index[0]).total_seconds()/(365.25*86400),1e-9)
    final=float(curve.equity.iloc[-1])
    cagr=(final/initial)**(1/years)-1
    mdd=float(curve.drawdown.min())
    rr=curve.equity.pct_change().dropna()
    sharpe=float(rr.mean()/rr.std()*np.sqrt(6*365)) if len(rr)>2 and rr.std()>0 else np.nan
    gains=float(rr[rr>0].sum()); losses=float(-rr[rr<0].sum())
    pf=gains/losses if losses>0 else np.nan
    return dict(cagr=cagr,mdd=mdd,sharpe=sharpe,pf=pf,final=final,
                exposure_time=float((curve.exposure>1e-12).mean()),
                avg_exposure=float(curve.exposure.mean()),
                max_exposure=float(curve.exposure.max()))

def run_one(x,funding,s,initial):
    base_cost=CostModel()
    base,trades,hb,execb=simulate(x,funding,s,base_cost,initial,True)
    adaptive2,_,ha,execa=simulate(x,funding,s,CostModel(11,4,2),initial,True)
    frozen2,_,hf,execf=simulate(x,funding,s,CostModel(11,4,2),initial,False,trades)
    xd=x.copy()
    sigcols=["regime","rv30","rv_ratio","shock","risk_fast","risk_mid","risk_slow",
             "ret5","ret20","dd20","dclose","daily_seq","entry_hi","exit_lo","atr4h"]
    xd[sigcols]=xd[sigcols].shift(1)
    delayed,_,hd,execd=simulate(xd,funding,s,base_cost,initial,True)
    return {
        "base":metrics(base,initial),"adaptive2x":metrics(adaptive2,initial),
        "frozen2x":metrics(frozen2,initial),"+4h_delay":metrics(delayed,initial),
        "hard_stop":{"base":hb,"adaptive2x":ha,"frozen2x":hf,"delay":hd},
        "execution":{"base":execb,"adaptive2x":execa,"frozen2x":execf,"delay":execd},
        "_curve":base
    }

def robustness(x,funding,s,initial):
    variants=[
        ("CORE95",replace(s,strong_long=s.strong_long*.95,weak_long=s.weak_long*.95)),
        ("BASE",s),
        ("CORE105",replace(s,strong_long=s.strong_long*1.05,weak_long=s.weak_long*1.05,
                           max_long=min(.75,s.max_long*1.03))),
        ("TRIG90",replace(s,convex_trigger1_atr=s.convex_trigger1_atr*.9,
                          convex_trigger2_atr=s.convex_trigger2_atr*.9,
                          convex_trigger3_atr=s.convex_trigger3_atr*.9)),
        ("TRIG110",replace(s,convex_trigger1_atr=s.convex_trigger1_atr*1.1,
                           convex_trigger2_atr=s.convex_trigger2_atr*1.1,
                           convex_trigger3_atr=s.convex_trigger3_atr*1.1)),
    ]
    out=[]
    for label,v in variants:
        r=run_one(x,funding,v,initial)
        passed=(r["base"]["mdd"]>-0.15 and r["adaptive2x"]["mdd"]>-0.15 and
                r["frozen2x"]["mdd"]>-0.15 and r["+4h_delay"]["mdd"]>-0.15 and
                not any(r["hard_stop"].values()))
        out.append({"variant":label,"pass":bool(passed),
                    "base_cagr":r["base"]["cagr"],"base_mdd":r["base"]["mdd"]})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",default="2017-08-17")
    ap.add_argument("--end",default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    ap.add_argument("--initial",type=float,default=10000)
    ap.add_argument("--cache",default=".cache_v11")
    ap.add_argument("--results",default="results")
    a=ap.parse_args()
    outdir=Path(a.results); outdir.mkdir(parents=True,exist_ok=True)
    px,funding=download_data(a.start,a.end,Path(a.cache))

    results=[]
    for s in CANDIDATES:
        print(f"[RUN] {s.name}",flush=True)
        x=build_features(px,s).dropna(subset=["open","close","rv30","atr4h","risk_slow"]).copy()
        if len(x)<1000: raise RuntimeError("insufficient feature history")
        r=run_one(x,funding,s,a.initial)
        rob=robustness(x,funding,s,a.initial)
        split=int(len(x)*.70)
        oos=run_one(x.iloc[split:].copy(),funding,s,a.initial)
        rob_rate=sum(v["pass"] for v in rob)/len(rob)
        row={"candidate":asdict(s),"base":r["base"],"adaptive2x":r["adaptive2x"],
             "frozen2x":r["frozen2x"],"+4h_delay":r["+4h_delay"],
             "hard_stop":r["hard_stop"],"execution":r["execution"],
             "robustness":rob,"robustness_pass_rate":rob_rate,
             "oos":{"base":oos["base"],"adaptive2x":oos["adaptive2x"],
                    "frozen2x":oos["frozen2x"]},
             "funding_observations":int(len(funding))}
        gate=(r["base"]["cagr"]>=.10 and r["base"]["mdd"]>-.15 and
              r["base"]["sharpe"]>=.8 and r["adaptive2x"]["mdd"]>-.15 and
              r["frozen2x"]["mdd"]>-.15 and r["+4h_delay"]["mdd"]>-.15 and
              oos["base"]["cagr"]>=.07 and rob_rate>=.60 and
              not any(r["hard_stop"].values()))
        row["development_gate"]=bool(gate)
        results.append(row)
        r["_curve"].to_csv(outdir/f"{s.name}_equity.csv")

    def key(z):
        return (int(z["development_gate"]),z["robustness_pass_rate"],
                z["base"]["sharpe"] if np.isfinite(z["base"]["sharpe"]) else -99,
                z["base"]["cagr"])
    selected=max(results,key=key)
    summary={
        "version":"V11.0",
        "architecture":"V9 regime/risk/convex engine on futures/funding execution",
        "selected":selected["candidate"]["name"],
        "development_gate_passed":selected["development_gate"],
        "funding_available":bool(len(funding)),
        "results":results,
    }
    (outdir/"v11_summary.json").write_text(json.dumps(summary,indent=2,default=float))
    pd.DataFrame([{
        "candidate":z["candidate"]["name"],"gate":z["development_gate"],
        "base_cagr":z["base"]["cagr"],"base_mdd":z["base"]["mdd"],
        "base_sharpe":z["base"]["sharpe"],"base_pf":z["base"]["pf"],
        "adaptive2x_mdd":z["adaptive2x"]["mdd"],"frozen2x_mdd":z["frozen2x"]["mdd"],
        "delay_mdd":z["+4h_delay"]["mdd"],"robustness_pass_rate":z["robustness_pass_rate"],
        "oos_cagr":z["oos"]["base"]["cagr"],"oos_mdd":z["oos"]["base"]["mdd"],
        "exposure_time":z["base"]["exposure_time"],"avg_exposure":z["base"]["avg_exposure"],
        "turnover":z["execution"]["base"]["turnover"],"rebalances":z["execution"]["base"]["rebalances"],
    } for z in results]).to_csv(outdir/"v11_scorecard.csv",index=False)

    print(json.dumps({
        "selected":summary["selected"],"gate":summary["development_gate_passed"],
        "base":selected["base"],"adaptive2x":selected["adaptive2x"],
        "frozen2x":selected["frozen2x"],"delay":selected["+4h_delay"],
        "oos":selected["oos"]["base"],"robustness":selected["robustness_pass_rate"],
        "execution":selected["execution"]["base"],
        "funding_observations":selected["funding_observations"],
    },indent=2,default=float))

if __name__=="__main__":
    main()
