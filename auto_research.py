#!/usr/bin/env python3
"""Causal V9.2 + winner-only Turtle pyramid exposure research."""
from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

INIT = 10_000.0
FEE_BPS = 5.5
SLIPPAGE_BPS = 2.0


def load(results: Path) -> pd.DataFrame:
    baseline = pd.read_csv(results / "integrated_equity_v92_baseline_stream.csv")
    baseline = baseline.rename(columns=lambda c: "time" if c.startswith("Unnamed") else c)
    turtle = pd.read_csv(results / "TURTLE_ORIGINAL_SIGNAL_4H.csv")
    baseline["time"] = pd.to_datetime(baseline["time"], utc=True)
    turtle["time"] = pd.to_datetime(turtle["time"], utc=True)
    x = pd.merge(turtle[["time", "close", "turtle_side", "turtle_N"]],
                 baseline[["time", "exposure", "drawdown"]], on="time", validate="one_to_one")
    x = x.sort_values("time").set_index("time")
    x["ret"] = x.close.pct_change().fillna(0.0)
    x["n_pct"] = x.turtle_N / x.close
    # Compare today's completed ATR only with a strictly prior distribution.
    x["n_median_90d"] = x.n_pct.rolling(6 * 90, min_periods=6 * 30).median().shift(1)
    return x


def pyramid_units(x: pd.DataFrame, spacing: float, dd_block: float | None,
                  vol_spike: float | None, trail_n: float | None,
                  max_units: int = 4) -> pd.Series:
    """State machine using only information known at each completed 4H bar."""
    close = x.close.to_numpy(float); side0 = x.turtle_side.to_numpy(float)
    natr = x.turtle_N.to_numpy(float); dd = x.drawdown.to_numpy(float)
    n_pct = x.n_pct.to_numpy(float); n_med = x.n_median_90d.to_numpy(float)
    out = np.zeros(len(x), dtype=np.int8)
    units = 0; previous_side = 0; last_add = np.nan; peak = np.nan
    for i in range(len(x)):
        side = int(np.sign(side0[i])) if np.isfinite(side0[i]) else 0
        # Additions are long-only and disappear when the Turtle state is flat/short.
        if side <= 0:
            units = 0; last_add = peak = np.nan; previous_side = side
            continue
        if previous_side <= 0:
            units = 1; last_add = peak = close[i]
            out[i] = units; previous_side = side
            continue
        peak = max(peak, close[i]) if np.isfinite(peak) else close[i]
        # De-risk removes only added units. Re-entry needs a fresh favorable move.
        if units > 1 and trail_n is not None and np.isfinite(natr[i]) and natr[i] > 0 \
                and close[i] <= peak - trail_n * natr[i]:
            units = 1; last_add = peak = close[i]
        blocked = dd_block is not None and np.isfinite(dd[i]) and dd[i] <= -dd_block
        if vol_spike is not None and np.isfinite(n_pct[i]) and np.isfinite(n_med[i]):
            blocked = blocked or n_pct[i] > vol_spike * n_med[i]
        if not blocked and units < max_units and np.isfinite(natr[i]) and natr[i] > 0:
            # One add per completed bar avoids optimistic gap sequencing.
            if close[i] >= last_add + spacing * natr[i]:
                units += 1; last_add = close[i]; peak = max(peak, close[i])
        out[i] = units; previous_side = side
    return pd.Series(out, index=x.index, dtype=float)


def exposure(x: pd.DataFrame, units: pd.Series, add_weight: float,
             max_exposure: float) -> pd.Series:
    add = (units - 1.0).clip(lower=0.0) * add_weight
    e = x.exposure.astype(float).copy()
    eligible = (x.turtle_side > 0) & (e >= 0)
    e.loc[eligible] += add.loc[eligible]
    return e.clip(-max_exposure, max_exposure)


def evaluate(x: pd.DataFrame, e: pd.Series, fee_bps: float = FEE_BPS,
             slip_bps: float = SLIPPAGE_BPS, delay: int = 0,
             start: float = 0.0, end: float = 1.0) -> dict:
    if delay:
        e = e.shift(delay).fillna(0.0)
    lo, hi = int(len(x) * start), int(len(x) * end)
    ee = e.to_numpy(float)[lo:hi]; raw_ret = x.ret.to_numpy(float)[lo:hi]
    held = np.empty_like(ee); held[0] = 0.0; held[1:] = ee[:-1]
    turnover = np.empty_like(ee); turnover[0] = abs(ee[0]); turnover[1:] = np.abs(np.diff(ee))
    rr = held * raw_ret - (fee_bps + slip_bps) / 10_000.0 * turnover
    eq = INIT * np.cumprod(1.0 + rr); drawdown = eq / np.maximum.accumulate(eq) - 1.0
    idx = x.index[lo:hi]
    years = max((idx[-1] - idx[0]).total_seconds() / (365.2425 * 86400), 1 / 365)
    cagr = (eq[-1] / INIT) ** (1 / years) - 1 if eq[-1] > 0 else -1.0
    sd = np.std(rr, ddof=1); sharpe = np.mean(rr) / sd * math.sqrt(6 * 365.2425) if sd > 0 else np.nan
    gp, gl = rr[rr > 0].sum(), -rr[rr < 0].sum()
    return {"final": float(eq[-1]), "cagr": float(cagr), "mdd": float(drawdown.min()),
            "sharpe": float(sharpe), "pf": float(gp / gl) if gl > 0 else np.nan,
            "avg_exp": float(np.mean(np.abs(ee))), "max_exp": float(np.max(np.abs(ee))),
            "turn": float(turnover.sum())}


def walk_forward(x: pd.DataFrame, e: pd.Series) -> tuple[float, float, float, float]:
    rows = [evaluate(x, e, start=k / 4, end=(k + 1) / 4) for k in range(4)]
    return (float(np.mean([r["cagr"] > 0 for r in rows])),
            float(np.median([r["cagr"] for r in rows])),
            float(min(r["cagr"] for r in rows)), float(min(r["mdd"] for r in rows)))


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated rows for maximizing CAGR and minimizing absolute MDD."""
    ordered = df.sort_values(["mdd", "cagr"], ascending=[False, False])
    keep, best_cagr = [], -np.inf
    for idx, row in ordered.iterrows():
        if row.cagr > best_cagr:
            keep.append(idx); best_cagr = row.cagr
    return df.loc[keep].sort_values("cagr", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--output", type=Path, default=Path("results_auto"))
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    x = load(args.results); rows = []
    state_grid = product([0.75, 1.0, 1.25], [None, 0.05, 0.075, 0.10],
                         [None, 1.25, 1.50, 1.75], [None, 0.75, 1.0, 1.25])
    for spacing, dd_block, vol_spike, trail_n in state_grid:
        units = pyramid_units(x, spacing, dd_block, vol_spike, trail_n)
        for add_weight, max_exp in product(np.arange(0.02, 0.201, 0.02),
                                           np.arange(0.70, 1.101, 0.10)):
            e = exposure(x, units, float(add_weight), float(max_exp))
            base = evaluate(x, e); cost2 = evaluate(x, e, FEE_BPS * 2, SLIPPAGE_BPS * 2)
            delay = evaluate(x, e, delay=1); oos = evaluate(x, e, start=0.75)
            wf_pos, wf_med, wf_worst_cagr, wf_worst_mdd = walk_forward(x, e)
            row = dict(spacing_n=spacing, add_per_unit=add_weight, max_exposure=max_exp,
                       dd_block=dd_block, vol_spike_ratio=vol_spike, trail_derisk_n=trail_n,
                       **base, cost2x_cagr=cost2["cagr"], cost2x_mdd=cost2["mdd"],
                       delay4h_cagr=delay["cagr"], delay4h_mdd=delay["mdd"],
                       oos25_cagr=oos["cagr"], oos25_mdd=oos["mdd"], wf_positive=wf_pos,
                       wf_median_cagr=wf_med, wf_worst_cagr=wf_worst_cagr,
                       wf_worst_mdd=wf_worst_mdd)
            row["gate20"] = abs(row["mdd"]) <= 0.20
            row["stress_gate20"] = (row["gate20"] and abs(cost2["mdd"]) <= 0.20 and
                                    abs(delay["mdd"]) <= 0.20 and abs(oos["mdd"]) <= 0.20)
            row["robust"] = (cost2["cagr"] > 0 and delay["cagr"] > 0 and
                             oos["cagr"] > 0 and wf_pos >= 0.75)
            rows.append(row)
    df = pd.DataFrame(rows); eligible = df[df.gate20]
    stress_eligible = df[df.stress_gate20]
    choice = eligible if not eligible.empty else pareto_frontier(df)
    best = choice.sort_values(["cagr", "robust", "mdd"], ascending=[False, False, False]).iloc[0]
    frontier = pareto_frontier(df)
    df.sort_values(["gate20", "cagr", "mdd"], ascending=[False, False, False]).to_csv(
        args.output / "AUTO_RESEARCH_COMPARISON.csv", index=False)
    frontier.to_csv(args.output / "PARETO_FRONTIER.csv", index=False)
    eligible.sort_values("cagr", ascending=False).head(50).to_csv(args.output / "TOP50_MDD20.csv", index=False)
    stress_eligible.sort_values("cagr", ascending=False).head(50).to_csv(
        args.output / "TOP50_STRESS_MDD20.csv", index=False)
    reference = pd.read_csv(args.results / "INTEGRATED_V2_COMPARISON.csv")
    v92 = reference[reference.strategy == "V92_BASELINE_STREAM"].iloc[0]
    old = reference[reference.strategy == "V92_TURTLE_PYRAMID"].iloc[0]
    stress_best = stress_eligible.sort_values("cagr", ascending=False).iloc[0] if len(stress_eligible) else None
    stress_line = (f"- All-scenario MDD<=20% best CAGR/MDD: {stress_best.cagr:.2%} / "
                   f"{stress_best.mdd:.2%} (cost2x MDD {stress_best.cost2x_mdd:.2%})"
                   if stress_best is not None else "- All-scenario MDD<=20% candidate: none")
    vol_label = "off" if pd.isna(best.vol_spike_ratio) else f"{best.vol_spike_ratio:.2f}x"
    report = f"""# BTC causal limited-pyramid research

- V9.2 baseline CAGR/MDD: {v92.cagr:.2%} / {v92.max_drawdown:.2%}
- Prior unrestricted pyramid CAGR/MDD: {old.cagr:.2%} / {old.max_drawdown:.2%}
- Searched candidates: {len(df):,}; MDD<=20%: {len(eligible):,}
{stress_line}

## Decision-rule winner
- CAGR / MDD: {best.cagr:.2%} / {best.mdd:.2%}
- Cost 2x CAGR/MDD: {best.cost2x_cagr:.2%} / {best.cost2x_mdd:.2%}
- +4H delay CAGR/MDD: {best.delay4h_cagr:.2%} / {best.delay4h_mdd:.2%}
- OOS25 CAGR/MDD: {best.oos25_cagr:.2%} / {best.oos25_mdd:.2%}
- WF positive / median / worst CAGR / worst MDD: {best.wf_positive:.0%} / {best.wf_median_cagr:.2%} / {best.wf_worst_cagr:.2%} / {best.wf_worst_mdd:.2%}
- Params: spacing={best.spacing_n:.2f}N, add={best.add_per_unit:.2f}, max_exp={best.max_exposure:.2f}x, DD block={best.dd_block:.1%}, ATR spike={vol_label}, trailing de-risk={best.trail_derisk_n:.2f}N
- MDD<=20%: {'PASS' if bool(best.gate20) else 'FAIL (see Pareto frontier)'}; robustness checks: {'PASS' if bool(best.robust) else 'REVIEW'}

No live trading is enabled. Results are research evidence, not pristine untouched OOS evidence.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    best_json = {k: (None if pd.isna(v) else v) for k, v in best.to_dict().items()}
    (args.output / "BEST.json").write_text(
        json.dumps(best_json, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
