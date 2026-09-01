#!/usr/bin/env python3
from __future__ import annotations
import json, math
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

R=Path("results"); O=Path("results_auto"); O.mkdir(exist_ok=True)
INIT=10000.0; FEE=5.5; SLIP=2.0

def load():
    b=pd.read_csv(R/"integrated_equity_v92_baseline_stream.csv").rename(columns=lambda c:"time" if c.startswith("Unnamed") else c)
    t=pd.read_csv(R/"TURTLE_ORIGINAL_SIGNAL_4H.csv")
    b["time"]=pd.to_datetime(b["time"],utc=True); t["time"]=pd.to_datetime(t["time"],utc=True)
    x=pd.merge(t[["time","close","turtle_side","turtle_N"]],b[["time","exposure","drawdown"]],on="time").sort_values("time").set_index("time")
    x["ret"]=x.close.pct_change().fillna(0); x["n_pct"]=x.turtle_N/x.close
    return x

def units(x,max_units,spacing,max_n_pct,dd_block):
    out=[]; u=0; last=np.nan; prev=0
    for _,r in x.iterrows():
        side=int(np.sign(r.turtle_side)) if np.isfinite(r.turtle_side) else 0
        if side<=0:
            u=0; last=np.nan; prev=side; out.append(0); continue
        if prev<=0:
            u=1; last=float(r.close); prev=side; out.append(u); continue
        ok=True
        if max_n_pct is not None and np.isfinite(r.n_pct) and r.n_pct>max_n_pct: ok=False
        if dd_block is not None and np.isfinite(r.drawdown) and r.drawdown<=-abs(dd_block): ok=False
        if ok and u<max_units and np.isfinite(r.turtle_N) and r.turtle_N>0 and r.close>=last+spacing*r.turtle_N:
            u+=1; last=float(r.close)
        out.append(u); prev=side
    return pd.Series(out,index=x.index,dtype=float)

def exposure(x,p):
    mu,sp,apu,mx,mn,dd=p
    u=units(x,mu,sp,mn,dd)
    add=(u-1).clip(lower=0)*apu
    e=x.exposure.astype(float).copy()
    m=(x.turtle_side>0)&(e>=0)
    e.loc[m]=e.loc[m]+add.loc[m]
    return e.clip(0,mx)

def evalm(x,e,fee=FEE,slip=SLIP,delay=0,start=0.0):
    if delay:e=e.shift(delay).fillna(0)
    i=int(len(x)*start); xx=x.iloc[i:]; ee=e.iloc[i:]
    held=ee.shift(1).fillna(0); turn=ee.diff().abs().fillna(ee.abs())
    rr=held*xx.ret-(fee+slip)/10000*turn
    eq=INIT*(1+rr).cumprod(); dd=eq/eq.cummax()-1
    yrs=max((eq.index[-1]-eq.index[0]).total_seconds()/(365.25*86400),1/365)
    cagr=(eq.iloc[-1]/INIT)**(1/yrs)-1
    sd=rr.std(); sh=rr.mean()/sd*math.sqrt(6*365) if sd and sd>0 else np.nan
    d=(1+rr).resample("1D").prod()-1; gp=d[d>0].sum(); gl=-d[d<0].sum()
    return dict(final=float(eq.iloc[-1]),cagr=float(cagr),mdd=float(dd.min()),sharpe=float(sh),pf=float(gp/gl) if gl>0 else np.nan,
                avg_exp=float(ee.abs().mean()),max_exp=float(ee.abs().max()),turn=float(turn.sum()))

def wf(x,e):
    rows=[]
    for k in range(4):
        a,b=int(len(x)*k/4),int(len(x)*(k+1)/4)
        z=x.iloc[a:b]; q=e.iloc[a:b]
        held=q.shift(1).fillna(0); tr=q.diff().abs().fillna(q.abs())
        rr=held*z.ret-(FEE+SLIP)/10000*tr; eq=INIT*(1+rr).cumprod(); dd=eq/eq.cummax()-1
        yrs=max((eq.index[-1]-eq.index[0]).total_seconds()/(365.25*86400),1/365)
        rows.append(((eq.iloc[-1]/INIT)**(1/yrs)-1,float(dd.min())))
    return np.mean([a>0 for a,b in rows]),np.median([a for a,b in rows]),min(b for a,b in rows)

def main():
    x=load(); rows=[]
    grid=product([2,3],[0.75,1.0,1.25],[0.08,0.12,0.16,0.20],[0.85,0.95,1.05,1.15],[None,0.07,0.09],[None,0.08,0.10])
    for p in grid:
        e=exposure(x,p); b=evalm(x,e); c2=evalm(x,e,FEE*2,SLIP*2); d4=evalm(x,e,delay=1); o=evalm(x,e,start=.75); wp,wm,ww=wf(x,e)
        r=dict(max_units=p[0],spacing_n=p[1],add_per_unit=p[2],max_exposure=p[3],max_n_pct=p[4],dd_block=p[5],**b,
               cost2x_cagr=c2["cagr"],cost2x_mdd=c2["mdd"],delay4h_cagr=d4["cagr"],delay4h_mdd=d4["mdd"],
               oos25_cagr=o["cagr"],oos25_mdd=o["mdd"],wf_positive=wp,wf_median_cagr=wm,wf_worst_mdd=ww)
        r["gate20"]=abs(r["mdd"])<=.20 and abs(r["oos25_mdd"])<=.20 and r["cost2x_cagr"]>0 and r["delay4h_cagr"]>0
        r["score"]=r["cagr"]+.25*r["cost2x_cagr"]+.25*r["delay4h_cagr"]+.20*r["oos25_cagr"]+.10*r["wf_median_cagr"]-.70*abs(r["mdd"])
        rows.append(r)
    df=pd.DataFrame(rows)
    pool=df[df.gate20].copy()
    if pool.empty: pool=df.copy()
    best=pool.sort_values(["score","cagr"],ascending=False).iloc[0]
    df=df.sort_values(["gate20","score","cagr"],ascending=False)
    df.to_csv(O/"AUTO_RESEARCH_COMPARISON.csv",index=False); df.head(50).to_csv(O/"TOP50.csv",index=False)

    ref=pd.read_csv(R/"INTEGRATED_V2_COMPARISON.csv")
    v92=ref[ref.strategy=="V92_BASELINE_STREAM"].iloc[0]; old=ref[ref.strategy=="V92_TURTLE_PYRAMID"].iloc[0]
    report=f"""# BTC AUTO HIGH-CAGR RESEARCH

- V9.2 baseline CAGR/MDD: {v92.cagr:.2%} / {v92.max_drawdown:.2%}
- Prior unrestricted pyramid CAGR/MDD: {old.cagr:.2%} / {old.max_drawdown:.2%}

## Best limited-pyramid candidate
- CAGR: {best.cagr:.2%}
- MDD: {best.mdd:.2%}
- Cost 2x CAGR/MDD: {best.cost2x_cagr:.2%} / {best.cost2x_mdd:.2%}
- +4H delay CAGR/MDD: {best.delay4h_cagr:.2%} / {best.delay4h_mdd:.2%}
- OOS25 CAGR/MDD: {best.oos25_cagr:.2%} / {best.oos25_mdd:.2%}
- WF positive / median CAGR / worst MDD: {best.wf_positive:.0%} / {best.wf_median_cagr:.2%} / {best.wf_worst_mdd:.2%}
- Params: units={int(best.max_units)}, spacing={best.spacing_n:.2f}N, add={best.add_per_unit:.2f}, max_exp={best.max_exposure:.2f}x, vol_cap={best.max_n_pct}, dd_block={best.dd_block}
- MDD<=20% strict gate: {'PASS' if bool(best.gate20) else 'FAIL'}

No live trading is enabled.
"""
    (O/"REPORT.md").write_text(report,encoding="utf-8")
    (O/"BEST.json").write_text(json.dumps(best.to_dict(),indent=2,default=str),encoding="utf-8")
    print(report); print("candidates:",len(df),"strict_pass:",int(df.gate20.sum()))

if __name__=="__main__": main()
