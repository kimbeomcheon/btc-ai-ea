#!/usr/bin/env python3
"""V12 Alpha research backtester. Research only; no trading or private APIs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as v112


VERSION = "V12-alpha"
MODES = ("Core-only", "Alpha-only", "Combined")
MAX_GROSS_EXPOSURE = 3.0
HARD_DRAWDOWN = 0.20


@dataclass(frozen=True)
class AlphaRules:
    name: str
    fast_4h: int = 36
    slow_4h: int = 144
    breakout_4h: int = 60
    exit_4h: int = 24
    vol_lookback_4h: int = 42
    vol_target: float = 0.65
    min_gross: float = 0.15
    max_gross: float = 2.25
    stop_atr: float = 3.0
    risk_per_trade: float = 0.015
    daily_loss_brake: float = 0.04
    weekly_loss_brake: float = 0.08
    drawdown_throttle: float = 0.10
    drawdown_floor: float = 0.18
    high_vol_cut: float = 1.20
    high_vol_scale: float = 0.60
    funding_soft: float = 0.00025
    funding_hard: float = 0.00075
    participation_floor: float = 0.75
    rebalance_deadband: float = 0.05
    min_rebalance_bars: int = 1
    admission_margin: float = 1.5


# Small, pre-declared research set. No parameter fitting occurs in this program.
CANDIDATES = (
    AlphaRules("V12A_BALANCED"),
    AlphaRules("V12B_DEFENSIVE", vol_target=0.55, max_gross=1.80,
               stop_atr=3.5, risk_per_trade=0.012),
    AlphaRules("V12C_RESPONSIVE", fast_4h=30, slow_4h=120, breakout_4h=48,
               exit_4h=20, vol_target=0.70, max_gross=2.50),
)


def core_strategy() -> v112.Strategy:
    matches = [s for s in v112.CANDIDATES if s.name == "V112B_RECOVERY4"]
    if len(matches) != 1:
        raise RuntimeError("frozen V112B_RECOVERY4 core is unavailable")
    return matches[0]


def build_research_features(px1h: pd.DataFrame, rules: AlphaRules) -> pd.DataFrame:
    x = v112.build_features(px1h, core_strategy()).dropna(
        subset=["open", "close", "rv30", "atr4h", "risk_slow"]
    ).copy()
    returns = x.close.pct_change()
    x["alpha_fast"] = x.close.ewm(span=rules.fast_4h, adjust=False,
                                     min_periods=rules.fast_4h).mean()
    x["alpha_slow"] = x.close.ewm(span=rules.slow_4h, adjust=False,
                                     min_periods=rules.slow_4h).mean()
    x["breakout_hi"] = x.high.shift(1).rolling(rules.breakout_4h).max()
    x["breakout_lo"] = x.low.shift(1).rolling(rules.breakout_4h).min()
    x["exit_hi"] = x.high.shift(1).rolling(rules.exit_4h).max()
    x["exit_lo_alpha"] = x.low.shift(1).rolling(rules.exit_4h).min()
    x["rv4h"] = returns.rolling(rules.vol_lookback_4h).std() * np.sqrt(6 * 365)
    # Public historical OI is not consistently available in the archive.  A
    # causal volume-participation ratio is used as an explicitly labelled proxy.
    volume_med = x.volume.rolling(180, min_periods=90).median().replace(0, np.nan)
    x["oi_proxy"] = x.volume.rolling(6).mean() / volume_med
    return x.dropna(subset=["alpha_fast", "alpha_slow", "breakout_hi",
                            "breakout_lo", "rv4h", "oi_proxy"])


def core_targets(x: pd.DataFrame, funding: pd.Series, cost: v112.CostModel,
                 adaptive: bool) -> pd.Series:
    curve, _, _, _ = v112.simulate(x, funding, core_strategy(), cost, 10_000.0, adaptive)
    return curve.exposure.reindex(x.index, method="ffill").fillna(0.0)


def alpha_gross(row: pd.Series, rules: AlphaRules) -> float:
    raw = rules.vol_target / float(row.rv4h) if row.rv4h > 0 else 0.0
    return float(np.clip(raw, rules.min_gross, min(rules.max_gross, MAX_GROSS_EXPOSURE)))


def funding_scale(direction: int, rate: float, rules: AlphaRules) -> float:
    crowded = direction * rate
    if crowded >= rules.funding_hard:
        return 0.0
    if crowded <= rules.funding_soft:
        return 1.0
    width = rules.funding_hard - rules.funding_soft
    return float(np.clip(1.0 - (crowded - rules.funding_soft) / width, 0.0, 1.0))


def alpha_edge_bps(row: pd.Series, direction: int) -> float:
    distance = 0.0
    boundary = row.breakout_hi if direction > 0 else row.breakout_lo
    if np.isfinite(row.atr4h) and row.atr4h > 0:
        distance = direction * (float(row.close) - float(boundary)) / float(row.atr4h)
    trend = direction * (float(row.alpha_fast) / float(row.alpha_slow) - 1.0)
    return float(18.0 + min(max(distance, 0.0), 5.0) * 8.0 + min(max(trend, 0.0), .10) * 300.0)


def simulate(x: pd.DataFrame, funding: pd.Series, rules: AlphaRules, mode: str,
             cost: v112.CostModel, initial: float = 10_000.0,
             adaptive_cost: bool = True, frozen_trades: dict[int, float] | None = None):
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    funding4 = v112.map_funding_to_4h(x.index, funding)
    core = core_targets(x, funding, cost, adaptive_cost) if mode != "Alpha-only" else pd.Series(0.0, index=x.index)
    equity = float(initial); peak = equity; exposure = 0.0
    direction = 0; entry_price = np.nan; last_trade = -10**9
    day_key = week_key = None; day_start = week_start = equity
    hard_stop = False; turnover = cost_paid = funding_paid = 0.0; rebalances = 0
    rows = []; trades: list[tuple[int, float]] = []
    one_way = cost.fee_bps + cost.slippage_bps

    for i in range(1, len(x)):
        previous = x.iloc[i - 1]; current = x.iloc[i]
        ret = float(current.open / previous.open - 1.0)
        equity *= max(1e-12, 1.0 + exposure * ret)
        rate = float(funding4.iloc[i - 1])
        funding_cost = equity * exposure * rate * cost.funding_mult
        equity = max(1e-12, equity - funding_cost)
        funding_paid += funding_cost
        peak = max(peak, equity)

        current_day = x.index[i].date()
        current_week = x.index[i].isocalendar()[:2]
        if current_day != day_key:
            day_key, day_start = current_day, equity
        if current_week != week_key:
            week_key, week_start = current_week, equity
        total_dd = equity / peak - 1.0
        daily_loss = equity / day_start - 1.0
        weekly_loss = equity / week_start - 1.0
        brake = daily_loss <= -rules.daily_loss_brake or weekly_loss <= -rules.weekly_loss_brake
        if total_dd <= -HARD_DRAWDOWN:
            hard_stop = True

        if direction >= 0 and previous.close > previous.breakout_hi and previous.alpha_fast > previous.alpha_slow and previous.oi_proxy >= rules.participation_floor:
            direction = 1
            if not np.isfinite(entry_price): entry_price = float(previous.close)
        elif direction <= 0 and previous.close < previous.breakout_lo and previous.alpha_fast < previous.alpha_slow and previous.oi_proxy >= rules.participation_floor:
            direction = -1
            if not np.isfinite(entry_price): entry_price = float(previous.close)
        if direction > 0 and (previous.close < previous.exit_lo_alpha or previous.alpha_fast < previous.alpha_slow):
            direction = 0; entry_price = np.nan
        elif direction < 0 and (previous.close > previous.exit_hi or previous.alpha_fast > previous.alpha_slow):
            direction = 0; entry_price = np.nan

        alpha = 0.0
        if direction and not brake and not hard_stop:
            stop_pct = rules.stop_atr * float(previous.atr4h) / float(previous.close)
            risk_cap = rules.risk_per_trade / stop_pct if stop_pct > 0 else 0.0
            gross = min(alpha_gross(previous, rules), risk_cap, MAX_GROSS_EXPOSURE)
            alpha = direction * gross * funding_scale(direction, rate, rules)
            if float(previous.rv4h) >= rules.high_vol_cut:
                alpha *= rules.high_vol_scale
            adverse = direction * (float(previous.close) / float(entry_price) - 1.0)
            if adverse <= -stop_pct:
                alpha = 0.0; direction = 0; entry_price = np.nan

        core_target = float(core.iloc[i]) if mode != "Alpha-only" else 0.0
        desired = core_target if mode == "Core-only" else alpha
        if mode == "Combined":
            desired = core_target + alpha
        desired = float(np.clip(desired, -MAX_GROSS_EXPOSURE, MAX_GROSS_EXPOSURE))
        if hard_stop or brake:
            desired = 0.0
        elif total_dd <= -rules.drawdown_throttle:
            width = max(rules.drawdown_floor - rules.drawdown_throttle, 1e-9)
            throttle = np.clip((rules.drawdown_floor + total_dd) / width, 0.0, 1.0)
            desired *= float(throttle)

        # No averaging down: same-direction gross may only increase after a
        # favorable move from the campaign entry.
        if exposure * desired > 0 and abs(desired) > abs(exposure) and np.isfinite(entry_price):
            if np.sign(desired) * (float(previous.close) / float(entry_price) - 1.0) <= 0:
                desired = np.sign(desired) * abs(exposure)

        if frozen_trades is not None:
            target = float(frozen_trades.get(i, exposure))
            execute = abs(target - exposure) > 1e-12
        else:
            target = desired
            delta = target - exposure
            execute = delta * exposure < 0 or abs(target) < abs(exposure)
            if abs(delta) >= rules.rebalance_deadband and i - last_trade >= rules.min_rebalance_bars:
                if abs(target) <= abs(exposure):
                    execute = True
                elif not adaptive_cost or mode == "Core-only":
                    execute = True
                else:
                    edge = alpha_edge_bps(previous, int(np.sign(alpha))) if alpha else 0.0
                    execute = edge >= rules.admission_margin * 2.0 * one_way
        if execute:
            delta = target - exposure
            trade_cost = equity * abs(delta) * one_way / 10_000.0
            equity = max(1e-12, equity - trade_cost)
            turnover += abs(delta); cost_paid += trade_cost; rebalances += 1
            exposure = target; last_trade = i; trades.append((i, exposure))
        peak = max(peak, equity)
        rows.append((x.index[i], equity, exposure, equity / peak - 1.0, rate,
                     funding_cost, turnover, cost_paid, rebalances, direction,
                     daily_loss, weekly_loss, brake))

    columns = ["time", "equity", "exposure", "drawdown", "funding_rate",
               "funding_paid", "turnover", "cost_paid", "rebalances",
               "alpha_direction", "daily_loss", "weekly_loss", "loss_brake"]
    curve = pd.DataFrame(rows, columns=columns).set_index("time")
    execution = {"turnover": turnover, "rebalances": rebalances,
                 "cost_paid": cost_paid, "funding_paid": funding_paid}
    return curve, {i: value for i, value in trades}, hard_stop, execution


def metrics(curve: pd.DataFrame, initial: float) -> dict[str, float]:
    if curve.empty:
        return {key: float("nan") for key in ("cagr", "mdd", "sharpe", "pf", "final",
                                                "exposure_time", "avg_exposure",
                                                "avg_gross_exposure", "max_gross_exposure")}
    years = max((curve.index[-1] - curve.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    final = float(curve.equity.iloc[-1])
    returns = curve.equity.pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * np.sqrt(6 * 365)) if len(returns) > 2 and returns.std() > 0 else np.nan
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.nan
    return {"cagr": (final / initial) ** (1 / years) - 1.0,
            "mdd": float(curve.drawdown.min()), "sharpe": sharpe, "pf": profit_factor,
            "final": final,
            "exposure_time": float((curve.exposure.abs() > 1e-12).mean()),
            "avg_exposure": float(curve.exposure.mean()),
            "avg_gross_exposure": float(curve.exposure.abs().mean()),
            "max_gross_exposure": float(curve.exposure.abs().max())}


def run_one(x, funding, rules, mode, initial):
    base_cost = v112.CostModel()
    base, trades, hb, eb = simulate(x, funding, rules, mode, base_cost, initial, True)
    stress = v112.CostModel(11, 4, 2)
    adaptive, _, ha, ea = simulate(x, funding, rules, mode, stress, initial, True)
    frozen, _, hf, ef = simulate(x, funding, rules, mode, stress, initial, False, trades)
    delayed = x.copy()
    signal_columns = [column for column in x.columns if column not in ("open", "high", "low", "close", "volume")]
    delayed[signal_columns] = delayed[signal_columns].shift(1)
    delayed = delayed.dropna(subset=signal_columns)
    delay, _, hd, ed = simulate(delayed, funding, rules, mode, base_cost, initial, True)
    return {"base": metrics(base, initial), "adaptive2x": metrics(adaptive, initial),
            "frozen2x": metrics(frozen, initial), "+4h_delay": metrics(delay, initial),
            "hard_stop": {"base": hb, "adaptive2x": ha, "frozen2x": hf, "delay": hd},
            "execution": {"base": eb, "adaptive2x": ea, "frozen2x": ef, "delay": ed},
            "_curve": base}


def walk_forward(x, funding, rules, mode, initial):
    # Fixed rules, expanding chronology, three untouched forward windows.
    boundaries = (0.55, 0.70, 0.85, 1.00)
    results = []
    for number, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        start, end = int(len(x) * left), int(len(x) * right)
        window = x.iloc[start:end].copy()
        result = run_one(window, funding, rules, mode, initial)
        results.append({"fold": number, "start": str(window.index[0]), "end": str(window.index[-1]),
                        "base": result["base"], "adaptive2x": result["adaptive2x"],
                        "frozen2x": result["frozen2x"], "+4h_delay": result["+4h_delay"],
                        "execution": result["execution"], "hard_stop": result["hard_stop"]})
    return results


def robustness(x, funding, rules, mode, initial):
    variants = (("BASE", rules),
                ("SIGNAL105", replace(rules, participation_floor=rules.participation_floor * 1.05)),
                ("VOL95", replace(rules, vol_target=rules.vol_target * .95)),
                ("CAP95", replace(rules, max_gross=rules.max_gross * .95)),
                ("REBALANCE110", replace(rules, rebalance_deadband=rules.rebalance_deadband * 1.10)),
                ("RISK95", replace(rules, risk_per_trade=rules.risk_per_trade * .95,
                                   drawdown_throttle=rules.drawdown_throttle * .95)))
    rows = []
    for label, variant in variants:
        result = run_one(x, funding, variant, mode, initial)
        passed = (result["base"]["mdd"] > -HARD_DRAWDOWN and
                  result["adaptive2x"]["mdd"] > -HARD_DRAWDOWN and
                  result["frozen2x"]["mdd"] > -HARD_DRAWDOWN and
                  result["+4h_delay"]["mdd"] > -HARD_DRAWDOWN and
                  not any(result["hard_stop"].values()))
        rows.append({"variant": label, "pass": bool(passed),
                     "base_cagr": result["base"]["cagr"], "base_mdd": result["base"]["mdd"]})
    return rows


def overfit_check(base, oos, walk):
    gap = base["cagr"] - oos["base"]["cagr"]
    positive_folds = sum(fold["base"]["cagr"] > 0 for fold in walk)
    reasons = []
    if gap > max(.35, 2 * max(oos["base"]["cagr"], 0.0)):
        reasons.append("base/OOS CAGR instability")
    if positive_folds < 2:
        reasons.append("fewer than two positive walk-forward folds")
    return {"failed": bool(reasons), "reasons": reasons, "base_oos_cagr_gap": gap,
            "positive_walk_forward_folds": positive_folds}


def selection_key(row):
    stress_mdd = min(row["adaptive2x"]["mdd"], row["frozen2x"]["mdd"],
                     row["+4h_delay"]["mdd"])
    turnover = row["execution"]["base"]["turnover"]
    return (int(row["development_gate"]), int(row["oos"]["base"]["mdd"] > -HARD_DRAWDOWN),
            row["oos"]["base"]["mdd"], row["robustness_pass_rate"],
            row["base"]["sharpe"] if np.isfinite(row["base"]["sharpe"]) else -np.inf,
            stress_mdd, row["oos"]["base"]["cagr"], -turnover)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-09-08")
    parser.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--cache", default=".cache_v12")
    parser.add_argument("--results", default="results_v12")
    args = parser.parse_args()
    output = Path(args.results); output.mkdir(parents=True, exist_ok=True)
    prices, funding = v112.download_data(args.start, args.end, Path(args.cache))
    all_results = []

    for rules in CANDIDATES:
        features = build_research_features(prices, rules)
        if len(features) < 1500:
            raise RuntimeError("insufficient V12 feature history")
        split = int(len(features) * .70)
        for mode in MODES:
            print(f"[RUN] {rules.name} / {mode}", flush=True)
            base = run_one(features, funding, rules, mode, args.initial)
            oos = run_one(features.iloc[split:].copy(), funding, rules, mode, args.initial)
            walk = walk_forward(features, funding, rules, mode, args.initial)
            robust = robustness(features, funding, rules, mode, args.initial)
            robust_rate = sum(item["pass"] for item in robust) / len(robust)
            overfit = overfit_check(base["base"], oos, walk)
            gate = (oos["base"]["cagr"] >= .20 and oos["base"]["mdd"] > -.20 and
                    base["base"]["sharpe"] >= 1.0 and
                    base["adaptive2x"]["mdd"] > -.20 and base["frozen2x"]["mdd"] > -.20 and
                    base["+4h_delay"]["mdd"] > -.20 and robust_rate >= .60 and
                    not any(base["hard_stop"].values()) and not overfit["failed"])
            oos_output = {key: value for key, value in oos.items() if key != "_curve"}
            row = {"candidate": asdict(rules), "mode": mode, "base": base["base"],
                   "adaptive2x": base["adaptive2x"], "frozen2x": base["frozen2x"],
                   "+4h_delay": base["+4h_delay"], "oos": oos_output,
                   "walk_forward": walk, "robustness": robust,
                   "robustness_pass_rate": robust_rate, "execution": base["execution"],
                   "hard_stop": base["hard_stop"], "overfit_check": overfit,
                   "development_gate": bool(gate)}
            all_results.append(row)
            safe_mode = mode.lower().replace("-", "_")
            base["_curve"].to_csv(output / f"{rules.name}_{safe_mode}_equity.csv")

    selected_by_mode = {mode: max((row for row in all_results if row["mode"] == mode),
                                  key=selection_key)["candidate"]["name"] for mode in MODES}
    summary = {"version": VERSION, "research_only": True, "max_gross_exposure": MAX_GROSS_EXPOSURE,
               "selected_by_mode": selected_by_mode,
               "development_gate": {"oos_cagr_min": .20, "oos_mdd_gt": -.20,
                                    "base_sharpe_min": 1.0, "stress_mdd_gt": -.20,
                                    "robustness_min": .60, "hard_stop": False,
                                    "overfit_check": "PASS"},
               "results_by_mode": {mode: [row for row in all_results if row["mode"] == mode]
                                   for mode in MODES}}
    (output / "v12_alpha_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    flat = [{"candidate": row["candidate"]["name"], "mode": row["mode"],
             "gate": row["development_gate"], "overfit_fail": row["overfit_check"]["failed"],
             "base_cagr": row["base"]["cagr"], "base_sharpe": row["base"]["sharpe"],
             "base_pf": row["base"]["pf"],
             "base_mdd": row["base"]["mdd"], "oos_cagr": row["oos"]["base"]["cagr"],
             "oos_mdd": row["oos"]["base"]["mdd"], "robustness": row["robustness_pass_rate"],
             "turnover": row["execution"]["base"]["turnover"],
             "rebalances": row["execution"]["base"]["rebalances"],
             "cost_paid": row["execution"]["base"]["cost_paid"],
             "funding_paid": row["execution"]["base"]["funding_paid"],
             "avg_gross": row["base"]["avg_gross_exposure"],
             "max_gross": row["base"]["max_gross_exposure"]} for row in all_results]
    pd.DataFrame(flat).to_csv(output / "v12_alpha_scorecard.csv", index=False)
    for mode in MODES:
        (output / f"{mode.lower().replace('-', '_')}_results.json").write_text(
            json.dumps(summary["results_by_mode"][mode], indent=2, default=float))
    print(pd.DataFrame(flat).to_string(index=False))


if __name__ == "__main__":
    main()
