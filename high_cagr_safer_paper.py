#!/usr/bin/env python3
"""HIGH_CAGR_FUNDING_AWARE_SAFER_V1 — public-data DRY-RUN paper only."""
from __future__ import annotations
import argparse, inspect, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import backtest
from paper_trader import fetch_klines, fetch_funding, KLINE_PATH, FUNDING_PATH

MODE="DRY_RUN_PAPER_ONLY"; STRATEGY="HIGH_CAGR_FUNDING_AWARE_SAFER_V1"
BASE="V92C_COOLDOWN2"; INIT=10000.0; FEE=5.5; SLIP=2.0
SP=0.65; ERTH=0.045; SOFT=0.10; ADD=1.70; CAP=1.12
FWIN=2; FTH=0.00048; FSCALE=0.0


def iso(t):
    t=pd.Timestamp(t); t=t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.isoformat().replace("+00:00","Z")

def fnum(x, default=0.0):
    x=float(x); return x if math.isfinite(x) else default

def base_strategy():
    m=[s for s in backtest.CANDIDATES if s.name==BASE]
    if len(m)!=1: raise RuntimeError(f"missing/ambiguous {BASE}")
    return m[0]

def decision_history(raw, t):
    t=pd.Timestamp(t).tz_convert("UTC"); prior=raw.loc[raw.index<t].copy()
    if prior.empty: raise RuntimeError("no completed 1H history")
    op=float(raw.loc[t,"open"]) if t in raw.index else float(prior.iloc[-1].close)
    ph=pd.DataFrame({"open":[op],"high":[op],"low":[op],"close":[op],"volume":[0.]},index=[t])
    return pd.concat([prior,ph]).sort_index(),op

def pyramid(x):
    c=x.close.to_numpy(float); side=x.turtle_side.to_numpy(float)
    n=x.turtle_N.to_numpy(float); b=x.exposure.to_numpy(float)
    out=np.zeros(len(x),dtype=np.int8); units=0; prev=0; last=np.nan
    for i in range(len(x)):
        s=int(np.sign(side[i])) if np.isfinite(side[i]) else 0
        if s<=0 or not np.isfinite(b[i]) or b[i]<=0:
            units=0; prev=s; last=np.nan; continue
        if prev<=0:
            units=1; last=c[i]; out[i]=units; prev=s; continue
        if units<4 and np.isfinite(n[i]) and n[i]>0 and c[i]>=last+SP*n[i]:
            units+=1; last=c[i]
        out[i]=units; prev=s
    return pd.Series(out,index=x.index,dtype=float).shift(1).fillna(0.)

def funding_feature(fund, idx):
    if fund.empty: return pd.Series(np.nan,index=idx,dtype=float)
    return fund.sort_index().rolling(FWIN,min_periods=FWIN).mean().reindex(idx).ffill().shift(1)

def target_stream(px, fund):
    eq,_,_,_=backtest.backtest(px,base_strategy(),backtest.CostModel(FEE,SLIP),
        backtest.RiskRules(),initial=INIT,signal_delay_bars=1,enforce_hard_stop=False)
    tur=backtest.build_original_turtle_4h(px,0.0075,4,1.50)
    book=backtest.build_book_hierarchy_4h(px); h4=backtest._ohlc_resample(px,"4h","left")
    idx=h4.index.intersection(eq.index).intersection(tur.index).intersection(book.index)
    if len(idx)<500: raise RuntimeError(f"insufficient merged 4H history {len(idx)}")
    x=pd.DataFrame(index=idx); x["open"]=h4.open.reindex(idx); x["close"]=h4.close.reindex(idx)
    x["turtle_side"]=tur.turtle_side.reindex(idx); x["turtle_N"]=tur.turtle_N.reindex(idx)
    x["exposure"]=eq.target_exposure.reindex(idx); x["drawdown"]=eq.drawdown.reindex(idx)
    x["book_conf"]=book.book_conf_live.reindex(idx).fillna(0.)
    er=x.close.diff(360).abs()/x.close.diff().abs().rolling(360).sum()
    x["er"]=er.replace([np.inf,-np.inf],np.nan).shift(1); x["units"]=pyramid(x)
    stage=(x.units-1).clip(lower=0); fac=pd.Series(np.where(x.er>=ERTH,1.,1.-SOFT),index=idx)
    elig=(x.turtle_side>0)&(x.exposure>0); extra=stage*ADD*x.book_conf*fac.fillna(1.-SOFT)
    tgt=x.exposure.copy(); tgt.loc[elig]+=extra.loc[elig]; tgt=tgt.clip(-CAP,CAP)
    x["fund_mean"]=funding_feature(fund,idx); x["riskoff"]=(x.fund_mean>FTH).fillna(False)
    tgt.loc[x.riskoff&(tgt>0)]*=FSCALE
    x["base"]=x.exposure; x["extra"]=extra.where(elig,0.); x["target"]=tgt
    return x

def signal(raw,fund,t):
    hist,op=decision_history(raw,t); x=target_stream(hist,fund); t=pd.Timestamp(t).tz_convert("UTC")
    u=x.loc[x.index<=t]
    if u.empty: raise RuntimeError("no causal target")
    ts=u.index[-1]; r=u.iloc[-1]
    return dict(timestamp=ts,decision=t,mark=op if ts==t else float(r.open),target=float(r.target),
      base=float(r.base),extra=float(r.extra),turtle=int(np.sign(r.turtle_side)),units=int(r.units),
      er=fnum(r.er,float("nan")),book=fnum(r.book_conf),fund_mean=fnum(r.fund_mean,float("nan")),
      riskoff=bool(r.riskoff),stale=bool(ts!=t),rows=len(x))

def load_state(path):
    if not path.exists(): return dict(mode=MODE,strategy=STRATEGY,paper_equity=INIT,peak_equity=INIT,
      target_exposure=0.,mark_price=None,timestamp=None,cumulative_turnover=0.,cumulative_cost=0.,cumulative_funding_paid=0.)
    s=json.loads(path.read_text());
    if s.get("mode")!=MODE or s.get("strategy")!=STRATEGY: raise RuntimeError("incompatible paper state")
    return s

def update(prev,sig,fund):
    ts=pd.Timestamp(sig["timestamp"]); pt=pd.Timestamp(prev["timestamp"]) if prev.get("timestamp") else None
    if pt is not None: pt=pt.tz_localize("UTC") if pt.tzinfo is None else pt.tz_convert("UTC")
    if pt is not None and ts<pt: raise RuntimeError("paper time went backward")
    same=pt is not None and ts==pt; pe=float(prev.get("paper_equity",INIT)); pp=float(prev.get("peak_equity",pe))
    pexp=float(prev.get("target_exposure",0.)); pm=prev.get("mark_price"); cm=float(sig["mark"])
    pret=0. if same or not pm else cm/float(pm)-1.
    obs=fund.iloc[0:0] if same or pt is None else fund.loc[(fund.index>pt)&(fund.index<=ts)]
    fr=float(obs.sum()) if len(obs) else 0.; eb=pe*max(1e-12,1+pexp*pret); paid=eb*pexp*fr; eb=max(1e-12,eb-paid)
    tgt=float(sig["target"]); chg=0. if same else tgt-pexp; turn=abs(chg); cost=eb*turn*(FEE+SLIP)/10000.; eq=max(1e-12,eb-cost); peak=max(pp,eq)
    st=dict(schema_version=1,mode=MODE,strategy=STRATEGY,symbol="BTCUSDT",timestamp=iso(ts),
      generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),paper_equity=fnum(eq),peak_equity=fnum(peak),
      drawdown=fnum(eq/peak-1),mark_price=fnum(cm),target_exposure=fnum(tgt),exposure_change=fnum(chg),turnover=fnum(turn),cost=fnum(cost),
      cumulative_turnover=fnum(float(prev.get("cumulative_turnover",0))+turn),cumulative_cost=fnum(float(prev.get("cumulative_cost",0))+cost),
      funding_paid=fnum(paid),cumulative_funding_paid=fnum(float(prev.get("cumulative_funding_paid",0))+paid))
    last2=fund.loc[fund.index<ts].tail(2); rep={**st,"latest_completed_4h_bar":iso(sig["decision"]-pd.Timedelta(hours=4)),
      "prior_timestamp":prev.get("timestamp"),"prior_exposure":pexp,"price_return_since_prior_run":pret,
      "base_exposure":sig["base"],"overlay_extra_before_funding":sig["extra"],"turtle_side":sig["turtle"],"overlay_units":sig["units"],
      "er60_live":None if not math.isfinite(sig["er"]) else sig["er"],"book_conf_live":sig["book"],
      "funding_observations_since_prior_run":len(obs),"realized_funding_rate_sum":fr,"last_two_settled_funding_rates":[float(v) for v in last2],
      "last_two_settled_funding_mean":float(last2.mean()) if len(last2)==2 else None,
      "funding_mean_live_used_by_signal":None if not math.isfinite(sig["fund_mean"]) else sig["fund_mean"],"funding_risk_off":sig["riskoff"],
      "stale_decision":sig["stale"],"safety":{"api_key_used":False,"orders_enabled":False,"private_endpoints_present":False,
      "allowed_public_endpoints":[KLINE_PATH,FUNDING_PATH]},"frozen_parameters":{"base_strategy":BASE,"spacing_N":SP,"er60_threshold":ERTH,
      "soft_gate_cut":SOFT,"add_weight":ADD,"max_exposure":CAP,"funding_window_settlements":FWIN,"funding_threshold":FTH}}
    return st,rep

def self_test():
    idx=pd.date_range("2026-01-01",periods=6,freq="4h",tz="UTC"); f=pd.Series([.0005,.0005],index=[idx[0],idx[2]])
    z=funding_feature(f,idx); assert not(z.iloc[2]>FTH) and z.iloc[3]>FTH
    src=inspect.getsource(fetch_klines)+inspect.getsource(fetch_funding)
    for q in ("/v5/order","/v5/position","/v5/account","/v5/execution"): assert q not in src
    print("SELF_TEST_PASS")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state-dir",default="paper_state_high_cagr_safer"); ap.add_argument("--lookback-days",type=int,default=800); ap.add_argument("--self-test",action="store_true"); a=ap.parse_args()
    if a.self_test: self_test(); return
    if a.lookback_days<600: raise ValueError("lookback must be >=600 days")
    d=Path(a.state_dir); d.mkdir(parents=True,exist_ok=True); spath=d/"state.json"; prev=load_state(spath)
    now=pd.Timestamp.now(tz="UTC"); t=now.floor("4h"); raw=fetch_klines(t-pd.Timedelta(days=a.lookback_days),now.floor("1h")); fund=fetch_funding(t-pd.Timedelta(days=a.lookback_days),now)
    sig=signal(raw,fund,t)
    if sig["stale"]: raise RuntimeError(f"stale signal {sig['timestamp']} vs {t}")
    st,rep=update(prev,sig,fund); spath.write_text(json.dumps(st,indent=2,allow_nan=False)); (d/"report.json").write_text(json.dumps(rep,indent=2,allow_nan=False)); print(json.dumps(rep,indent=2,allow_nan=False))
if __name__=="__main__": main()
