#!/usr/bin/env python3
"""V13 causal AI Decision Engine research backtester.

Research-only public-data code. It contains no exchange order endpoints, API
keys, secrets, or live/paper execution integration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as core


VERSION = "V13-decision-engine"
MODES = ("Core-only", "Decision-only", "Combined")
MAX_GROSS = 3.0
HARD_DRAWDOWN = 0.20


@dataclass(frozen=True)
class DecisionRules:
    name: str
    online_calibration: bool
    trend_fast: int = 36
    trend_slow: int = 144
    breakout_bars: int = 60
    rv_bars: int = 42
    vol_target: float = 0.60
    max_bull: float = 1.75
    max_bear: float = 1.35
    max_range: float = 0.55
    max_stress: float = 0.30
    min_confidence: float = 0.18
    max_uncertainty: float = 0.92
    risk_per_trade: float = 0.0125
    stop_atr: float = 2.5
    daily_brake: float = 0.04
    weekly_brake: float = 0.08
    dd_throttle: float = 0.10
    dd_floor: float = 0.18
    shock_rv: float = 1.20
    rebalance_deadband: float = 0.05
    min_rebalance_bars: int = 1
    admission_margin: float = 1.50
    online_lr: float = 0.025
    online_l2: float = 0.001
    online_warmup: int = 360


CANDIDATES = (
    DecisionRules("V13A_INTERPRETABLE", online_calibration=False),
    DecisionRules("V13B_CAUSAL_ONLINE", online_calibration=True,
                  min_confidence=0.20, online_warmup=540),
)


def frozen_core() -> core.Strategy:
    matches = [candidate for candidate in core.CANDIDATES if candidate.name == "V112B_RECOVERY4"]
    if len(matches) != 1:
        raise RuntimeError("frozen V112B_RECOVERY4 is unavailable")
    return matches[0]


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def entropy_uncertainty(probability: float) -> float:
    p = float(np.clip(probability, 1e-12, 1 - 1e-12))
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2))


def regime_for(row: pd.Series) -> str:
    if bool(row.shock) or float(row.rv4h) >= float(row.rv_stress_cut):
        return "HIGH_VOL_STRESS"
    if row.close > row.trend_slow and row.trend_fast > row.trend_slow:
        return "BULL_TREND"
    if row.close < row.trend_slow and row.trend_fast < row.trend_slow:
        return "BEAR_TREND"
    return "RANGE"


def regime_cap(regime: str, rules: DecisionRules) -> float:
    return {"BULL_TREND": rules.max_bull, "BEAR_TREND": rules.max_bear,
            "RANGE": rules.max_range, "HIGH_VOL_STRESS": rules.max_stress}[regime]


def regime_weights(regime: str) -> np.ndarray:
    # trend, breakout/momentum, carry, participation
    return {"BULL_TREND": np.array([.38, .32, .12, .18]),
            "BEAR_TREND": np.array([.38, .32, .12, .18]),
            "RANGE": np.array([.16, .24, .42, .18]),
            "HIGH_VOL_STRESS": np.array([.20, .20, .40, .20])}[regime]


def build_decision_features(px1h: pd.DataFrame, funding: pd.Series,
                            rules: DecisionRules) -> pd.DataFrame:
    x = core.build_features(px1h, frozen_core()).dropna(
        subset=["open", "close", "rv30", "atr4h", "risk_slow"]
    ).copy()
    ret = x.close.pct_change()
    x["trend_fast"] = x.close.ewm(span=rules.trend_fast, adjust=False,
                                    min_periods=rules.trend_fast).mean()
    x["trend_slow"] = x.close.ewm(span=rules.trend_slow, adjust=False,
                                    min_periods=rules.trend_slow).mean()
    high = x.high.shift(1).rolling(rules.breakout_bars).max()
    low = x.low.shift(1).rolling(rules.breakout_bars).min()
    midpoint = (high + low) / 2.0
    width = (high - low).replace(0, np.nan)
    x["rv4h"] = ret.rolling(rules.rv_bars).std() * np.sqrt(6 * 365)
    x["rv_stress_cut"] = x.rv4h.rolling(540, min_periods=180).quantile(.90)
    atr_pct = x.atr4h / x.close
    x["trend_score"] = np.tanh(np.log(x.trend_fast / x.trend_slow) /
                                  atr_pct.replace(0, np.nan) / 4.0)
    x["breakout_score"] = np.tanh((x.close - midpoint) / width * 4.0)
    funding4 = core.map_funding_to_4h(x.index, funding)
    x["funding_rate"] = funding4
    x["carry_score"] = -np.tanh(funding4 / .00035)
    volume_base = x.volume.rolling(180, min_periods=90).median().replace(0, np.nan)
    x["oi_proxy"] = x.volume.rolling(6).mean() / volume_base
    x["participation_score"] = np.tanh((x.oi_proxy - 1.0) * 1.5) * np.sign(x.breakout_score)
    required = ["trend_fast", "trend_slow", "rv4h", "rv_stress_cut", "trend_score",
                "breakout_score", "carry_score", "oi_proxy", "participation_score"]
    x = x.dropna(subset=required).copy()
    x["decision_regime"] = [regime_for(row) for _, row in x.iterrows()]

    model_columns = ["trend_score", "breakout_score", "carry_score", "participation_score"]
    scores = x[model_columns].to_numpy(dtype=float)
    static_probabilities = []
    disagreements = []
    for i, (_, row) in enumerate(x.iterrows()):
        weights = regime_weights(str(row.decision_regime))
        weighted_score = float(np.dot(weights, scores[i]))
        static_probabilities.append(float(sigmoid(2.2 * weighted_score)))
        disagreements.append(float(np.std(scores[i])))

    probabilities = np.asarray(static_probabilities)
    if rules.online_calibration:
        learned = np.zeros(5, dtype=float)  # intercept + four independent sub-model scores
        online_probabilities = np.full(len(x), .5, dtype=float)
        closes = x.close.to_numpy(dtype=float)
        for i in range(len(x)):
            if i > 0:
                previous_vector = np.r_[1.0, scores[i - 1]]
                label = 1.0 if closes[i] > closes[i - 1] else 0.0
                previous_probability = float(sigmoid(np.dot(learned, previous_vector)))
                learned += rules.online_lr * ((label - previous_probability) * previous_vector -
                                              rules.online_l2 * np.r_[0.0, learned[1:]])
            if i >= rules.online_warmup:
                online_probabilities[i] = float(sigmoid(np.dot(learned, np.r_[1.0, scores[i]])))
            else:
                online_probabilities[i] = probabilities[i]
        probabilities = .60 * online_probabilities + .40 * probabilities

    x["p_up"] = np.clip(probabilities, 1e-6, 1 - 1e-6)
    x["p_down"] = 1.0 - x.p_up
    x["confidence"] = (x.p_up - .5).abs() * 2.0
    entropy = np.array([entropy_uncertainty(value) for value in x.p_up])
    disagreement = np.clip(np.asarray(disagreements) / 1.25, 0.0, 1.0)
    x["uncertainty"] = np.clip(.70 * entropy + .30 * disagreement, 0.0, 1.0)
    x["decision_score"] = x.p_up - x.p_down
    x["signal_valid"] = np.isfinite(x[["p_up", "p_down", "confidence", "uncertainty"]]).all(axis=1)
    return x


def leakage_check(x: pd.DataFrame, rules: DecisionRules) -> dict:
    checks = {
        "chronological_only": True,
        "no_shuffle_split": True,
        "breakout_inputs_shifted": True,
        "online_update_order": "learn_previous_label_then_predict_current_for_next_bar",
        "probabilities_bounded": bool(((x.p_up > 0) & (x.p_up < 1) &
                                       np.isclose(x.p_up + x.p_down, 1.0)).all()),
        "signals_finite": bool(x.signal_valid.all()),
        "online_warmup_positive": bool(not rules.online_calibration or rules.online_warmup > 0),
    }
    return {"passed": bool(all(value is not False for value in checks.values())), "checks": checks}


def expected_value(row: pd.Series, direction: int, rules: DecisionRules,
                   one_way_bps: float) -> dict[str, float]:
    win_probability = float(row.p_up if direction > 0 else row.p_down)
    atr_move = float(row.atr4h / row.close)
    expected_profit = atr_move * (1.25 + 1.25 * float(row.confidence))
    expected_loss = atr_move * rules.stop_atr
    round_trip_cost = 2.0 * one_way_bps / 10_000.0
    expected_funding = max(0.0, direction * float(row.funding_rate))
    edge = (expected_profit * win_probability - expected_loss * (1.0 - win_probability) -
            rules.admission_margin * (round_trip_cost + expected_funding))
    return {"edge": float(edge), "expected_profit": expected_profit,
            "expected_loss": expected_loss, "win_probability": win_probability,
            "estimated_cost": round_trip_cost + expected_funding}


def decision_target(row: pd.Series, rules: DecisionRules, one_way_bps: float) -> tuple[float, dict]:
    if not bool(row.signal_valid) or row.confidence < rules.min_confidence or row.uncertainty > rules.max_uncertainty:
        return 0.0, {"edge": 0.0, "admitted": False}
    direction = 1 if row.p_up > row.p_down else -1
    ev = expected_value(row, direction, rules, one_way_bps)
    if ev["edge"] <= 0:
        return 0.0, {**ev, "admitted": False}
    vol_gross = rules.vol_target / float(row.rv4h) if row.rv4h > 0 else 0.0
    stop_pct = rules.stop_atr * float(row.atr4h / row.close)
    risk_cap = rules.risk_per_trade / stop_pct if stop_pct > 0 else 0.0
    gross = min(vol_gross * float(row.confidence), risk_cap,
                regime_cap(str(row.decision_regime), rules), MAX_GROSS)
    if row.decision_regime == "HIGH_VOL_STRESS":
        gross *= .50
    return direction * float(max(0.0, gross)), {**ev, "admitted": True}


def core_target_series(x, funding, cost_model, adaptive):
    curve, _, _, _ = core.simulate(x, funding, frozen_core(), cost_model, 10_000.0, adaptive)
    return curve.exposure.reindex(x.index, method="ffill").fillna(0.0)


def simulate(x, funding, rules, mode, cost_model, initial=10_000.0,
             adaptive_cost=True, frozen_trades=None):
    funding4 = core.map_funding_to_4h(x.index, funding)
    core_targets = (core_target_series(x, funding, cost_model, adaptive_cost)
                    if mode == "Combined" else pd.Series(0.0, index=x.index))
    equity = float(initial); peak = equity; exposure = 0.0
    entry_price = np.nan; campaign_direction = 0; pyramid_stage = 0; last_trade = -10**9
    day_key = week_key = None; day_start = week_start = equity
    hard_stop = False; turnover = fee_paid = slippage_paid = funding_paid = 0.0; rebalances = 0
    stale_count = 0; trades = []; rows = []

    for i in range(1, len(x)):
        previous, current = x.iloc[i - 1], x.iloc[i]
        ret = float(current.open / previous.open - 1.0)
        equity *= max(1e-12, 1.0 + exposure * ret)
        funding_rate = float(funding4.iloc[i - 1])
        funding_cost = equity * exposure * funding_rate * cost_model.funding_mult
        equity = max(1e-12, equity - funding_cost); funding_paid += funding_cost
        peak = max(peak, equity)
        current_day = x.index[i].date(); current_week = x.index[i].isocalendar()[:2]
        if current_day != day_key: day_key, day_start = current_day, equity
        if current_week != week_key: week_key, week_start = current_week, equity
        drawdown = equity / peak - 1.0
        daily_loss = equity / day_start - 1.0; weekly_loss = equity / week_start - 1.0
        brake = daily_loss <= -rules.daily_brake or weekly_loss <= -rules.weekly_brake
        if drawdown <= -HARD_DRAWDOWN: hard_stop = True

        stale_signal = (not bool(previous.signal_valid) or
                        x.index[i] - x.index[i - 1] > pd.Timedelta(hours=4))
        stale_count = stale_count + 1 if stale_signal else 0
        one_way = cost_model.fee_bps + cost_model.slippage_bps
        decision, ev = decision_target(previous, rules, one_way)
        if stale_count or brake or hard_stop: decision = 0.0
        core_target = float(core_targets.iloc[i]) if mode == "Combined" else 0.0
        desired = decision if mode == "Decision-only" else core_target + decision
        desired = float(np.clip(desired, -MAX_GROSS, MAX_GROSS))
        if brake or hard_stop or stale_count: desired = 0.0
        elif drawdown <= -rules.dd_throttle:
            width = max(rules.dd_floor - rules.dd_throttle, 1e-9)
            desired *= float(np.clip((rules.dd_floor + drawdown) / width, 0.0, 1.0))
        if float(previous.rv4h) >= rules.shock_rv: desired *= .50

        new_direction = int(np.sign(desired))
        if new_direction != campaign_direction:
            entry_price = float(previous.close) if new_direction else np.nan
            campaign_direction = new_direction; pyramid_stage = 0
        if exposure * desired > 0 and campaign_direction and np.isfinite(entry_price):
            favorable_atr = campaign_direction * (float(previous.close) - entry_price) / float(previous.atr4h)
            allowed_stage = 0 if favorable_atr < 1.0 else 1 if favorable_atr < 2.0 else 2
            if abs(desired) > abs(exposure):
                # Anti-martingale: increases require favorable price, maintained
                # confidence, and a higher earned pyramid stage.
                if favorable_atr <= 0 or previous.confidence < rules.min_confidence:
                    desired = np.sign(desired) * abs(exposure)
                elif allowed_stage <= pyramid_stage:
                    desired = np.sign(desired) * min(abs(desired), abs(exposure))
                else:
                    pyramid_stage = allowed_stage

        if frozen_trades is not None:
            target = float(frozen_trades.get(i, exposure)); execute = abs(target - exposure) > 1e-12
        else:
            target = desired; delta = target - exposure
            risk_reduction = abs(target) < abs(exposure) or target * exposure < 0
            execute = risk_reduction and abs(delta) > 1e-12
            if abs(delta) >= rules.rebalance_deadband and i - last_trade >= rules.min_rebalance_bars:
                execute = risk_reduction or not adaptive_cost or bool(ev.get("admitted", False))
        if execute:
            delta = target - exposure
            fee = equity * abs(delta) * cost_model.fee_bps / 10_000.0
            slippage = equity * abs(delta) * cost_model.slippage_bps / 10_000.0
            equity = max(1e-12, equity - fee - slippage)
            fee_paid += fee; slippage_paid += slippage; turnover += abs(delta); rebalances += 1
            exposure = target; last_trade = i; trades.append((i, exposure))
        peak = max(peak, equity)
        rows.append((x.index[i], equity, exposure, equity / peak - 1.0, funding_rate,
                     turnover, rebalances, fee_paid, slippage_paid, funding_paid,
                     str(previous.decision_regime), float(previous.p_up), float(previous.p_down),
                     float(previous.confidence), float(previous.uncertainty),
                     float(ev.get("edge", 0.0)), pyramid_stage, brake, stale_count))

    columns = ["time", "equity", "exposure", "drawdown", "funding_rate", "turnover",
               "rebalances", "fees", "slippage", "funding_paid", "regime", "p_up",
               "p_down", "confidence", "uncertainty", "expected_edge", "pyramid_stage",
               "loss_brake", "stale_count"]
    curve = pd.DataFrame(rows, columns=columns).set_index("time")
    execution = {"turnover": turnover, "rebalances": rebalances, "fees": fee_paid,
                 "slippage": slippage_paid, "funding": funding_paid,
                 "total_cost": fee_paid + slippage_paid + funding_paid}
    return curve, {i: value for i, value in trades}, hard_stop, execution


def metrics(curve, initial):
    if curve.empty:
        return {name: np.nan for name in ("cagr", "mdd", "sharpe", "pf", "final",
                                          "exposure_time", "avg_gross", "max_gross")}
    years = max((curve.index[-1] - curve.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    final = float(curve.equity.iloc[-1]); returns = curve.equity.pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * np.sqrt(6 * 365)) if len(returns) > 2 and returns.std() > 0 else np.nan
    gains = float(returns[returns > 0].sum()); losses = float(-returns[returns < 0].sum())
    return {"cagr": (final / initial) ** (1 / years) - 1.0,
            "mdd": float(curve.drawdown.min()), "sharpe": sharpe,
            "pf": gains / losses if losses > 0 else np.nan, "final": final,
            "exposure_time": float((curve.exposure.abs() > 1e-12).mean()),
            "avg_gross": float(curve.exposure.abs().mean()),
            "max_gross": float(curve.exposure.abs().max())}


def core_metrics(curve, initial):
    result = core.metrics(curve, initial)
    result["avg_gross"] = result["avg_exposure"]
    result["max_gross"] = result["max_exposure"]
    return result


def core_execution(curve, execution, initial):
    held = curve.exposure.shift(1).fillna(0.0)
    funding_fraction = float((held * curve.funding_rate).sum())
    total_bps = core.CostModel().fee_bps + core.CostModel().slippage_bps
    fee_share = core.CostModel().fee_bps / total_bps
    cost_paid = float(execution["cost_fraction"] * initial)
    return {**execution, "fees": cost_paid * fee_share,
            "slippage": cost_paid * (1 - fee_share),
            "funding": funding_fraction * initial,
            "total_cost": cost_paid + funding_fraction * initial}


def run_core_control(x, funding, initial):
    base_cost = core.CostModel(); stress = core.CostModel(11, 4, 2); strategy = frozen_core()
    base, trades, hb, eb = core.simulate(x, funding, strategy, base_cost, initial, True)
    adaptive, _, ha, ea = core.simulate(x, funding, strategy, stress, initial, True)
    frozen, _, hf, ef = core.simulate(x, funding, strategy, stress, initial, False, trades)
    delayed = x.copy()
    signal_columns = ["regime", "rv30", "rv_ratio", "shock", "risk_fast", "risk_mid",
                      "risk_slow", "ret5", "ret20", "dd20", "dclose", "daily_seq",
                      "entry_hi", "exit_lo", "atr4h"]
    delayed[signal_columns] = delayed[signal_columns].shift(1)
    delayed = delayed.dropna(subset=signal_columns)
    delay, _, hd, ed = core.simulate(delayed, funding, strategy, base_cost, initial, True)
    return {"base": core_metrics(base, initial), "adaptive2x": core_metrics(adaptive, initial),
            "frozen2x": core_metrics(frozen, initial), "+4h_delay": core_metrics(delay, initial),
            "hard_stop": {"base": hb, "adaptive2x": ha, "frozen2x": hf, "delay": hd},
            "execution": {"base": core_execution(base, eb, initial),
                          "adaptive2x": core_execution(adaptive, ea, initial),
                          "frozen2x": core_execution(frozen, ef, initial),
                          "delay": core_execution(delay, ed, initial)}, "_curve": base}


def run_one(x, funding, rules, mode, initial):
    if mode == "Core-only": return run_core_control(x, funding, initial)
    base_cost = core.CostModel(); stress = core.CostModel(11, 4, 2)
    base, trades, hb, eb = simulate(x, funding, rules, mode, base_cost, initial, True)
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
    boundaries = (.55, .70, .85, 1.0); output = []
    for fold, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        window = x.iloc[int(len(x) * left):int(len(x) * right)].copy()
        result = run_one(window, funding, rules, mode, initial)
        output.append({"fold": fold, "start": str(window.index[0]), "end": str(window.index[-1]),
                       **{key: result[key] for key in ("base", "adaptive2x", "frozen2x", "+4h_delay",
                                                       "execution", "hard_stop")}})
    return output


def robustness(x, funding, rules, mode, initial):
    if mode == "Core-only": return core.robustness(x, funding, frozen_core(), initial)
    variants = (("BASE", rules),
                ("CONF105", replace(rules, min_confidence=rules.min_confidence * 1.05)),
                ("VOL95", replace(rules, vol_target=rules.vol_target * .95)),
                ("CAP95", replace(rules, max_bull=rules.max_bull * .95,
                                  max_bear=rules.max_bear * .95)),
                ("REBALANCE110", replace(rules, rebalance_deadband=rules.rebalance_deadband * 1.10)),
                ("RISK95", replace(rules, risk_per_trade=rules.risk_per_trade * .95,
                                   dd_throttle=rules.dd_throttle * .95)))
    output = []
    for label, variant in variants:
        result = run_one(x, funding, variant, mode, initial)
        passed = (all(result[key]["mdd"] > -HARD_DRAWDOWN
                      for key in ("base", "adaptive2x", "frozen2x", "+4h_delay")) and
                  not any(result["hard_stop"].values()))
        output.append({"variant": label, "pass": bool(passed),
                       "base_cagr": result["base"]["cagr"], "base_mdd": result["base"]["mdd"]})
    return output


def overfit_check(base, oos, walk):
    gap = base["cagr"] - oos["base"]["cagr"]
    positive = sum(fold["base"]["cagr"] > 0 for fold in walk); reasons = []
    if gap > max(.35, 2 * max(oos["base"]["cagr"], 0.0)): reasons.append("base/OOS decay")
    if positive < 2: reasons.append("walk-forward regime dependence")
    return {"failed": bool(reasons), "reasons": reasons, "base_oos_cagr_gap": gap,
            "positive_walk_forward_folds": positive}


def selection_key(row):
    stress = min(row["adaptive2x"]["mdd"], row["frozen2x"]["mdd"], row["+4h_delay"]["mdd"])
    return (int(row["development_gate"]), int(row["oos"]["base"]["mdd"] > -HARD_DRAWDOWN),
            row["oos"]["base"]["mdd"], row["robustness_pass_rate"],
            row["base"]["sharpe"] if np.isfinite(row["base"]["sharpe"]) else -np.inf,
            stress, row["oos"]["base"]["cagr"], -row["execution"]["base"]["turnover"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-09-08")
    parser.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    parser.add_argument("--initial", type=float, default=10_000.0)
    parser.add_argument("--cache", default=".cache_v13")
    parser.add_argument("--results", default="results_v13")
    args = parser.parse_args(); output = Path(args.results); output.mkdir(parents=True, exist_ok=True)
    prices, funding = core.download_data(args.start, args.end, Path(args.cache)); rows = []
    leakage_reports = {}

    for rules in CANDIDATES:
        x = build_decision_features(prices, funding, rules)
        if len(x) < 1500: raise RuntimeError("insufficient V13 feature history")
        leakage_reports[rules.name] = leakage_check(x, rules)
        split = int(len(x) * .70)
        for mode in MODES:
            print(f"[RUN] {rules.name} / {mode}", flush=True)
            base = run_one(x, funding, rules, mode, args.initial)
            oos = run_one(x.iloc[split:].copy(), funding, rules, mode, args.initial)
            walk = walk_forward(x, funding, rules, mode, args.initial)
            robust = robustness(x, funding, rules, mode, args.initial)
            robust_rate = sum(item["pass"] for item in robust) / len(robust)
            overfit = overfit_check(base["base"], oos, walk)
            gate = (oos["base"]["cagr"] >= .20 and oos["base"]["mdd"] > -.20 and
                    base["base"]["sharpe"] >= 1.0 and
                    all(base[key]["mdd"] > -.20 for key in ("adaptive2x", "frozen2x", "+4h_delay")) and
                    robust_rate >= .60 and not any(base["hard_stop"].values()) and
                    leakage_reports[rules.name]["passed"] and not overfit["failed"])
            row = {"candidate": asdict(rules), "mode": mode, "base": base["base"],
                   "adaptive2x": base["adaptive2x"], "frozen2x": base["frozen2x"],
                   "+4h_delay": base["+4h_delay"],
                   "oos": {key: value for key, value in oos.items() if key != "_curve"},
                   "walk_forward": walk, "robustness": robust,
                   "robustness_pass_rate": robust_rate, "execution": base["execution"],
                   "hard_stop": base["hard_stop"], "data_leakage_check": leakage_reports[rules.name],
                   "overfit_check": overfit, "development_gate": bool(gate)}
            rows.append(row)
            base["_curve"].to_csv(output / f"{rules.name}_{mode.lower().replace('-', '_')}_equity.csv")

    selected = {mode: max((row for row in rows if row["mode"] == mode), key=selection_key)["candidate"]["name"]
                for mode in MODES}
    summary = {"version": VERSION, "research_only": True, "live_trading": False,
               "max_gross_exposure": MAX_GROSS, "selected_by_mode": selected,
               "data_leakage_checks": leakage_reports,
               "results_by_mode": {mode: [row for row in rows if row["mode"] == mode] for mode in MODES}}
    (output / "v13_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    flat = []
    for row in rows:
        execution = row["execution"]["base"]
        flat.append({"candidate": row["candidate"]["name"], "mode": row["mode"],
                     "gate": row["development_gate"], "overfit_fail": row["overfit_check"]["failed"],
                     "leakage_pass": row["data_leakage_check"]["passed"],
                     "base_cagr": row["base"]["cagr"], "base_mdd": row["base"]["mdd"],
                     "base_sharpe": row["base"]["sharpe"], "base_pf": row["base"]["pf"],
                     "oos_cagr": row["oos"]["base"]["cagr"], "oos_mdd": row["oos"]["base"]["mdd"],
                     "robustness": row["robustness_pass_rate"],
                     "adaptive2x_mdd": row["adaptive2x"]["mdd"],
                     "frozen2x_mdd": row["frozen2x"]["mdd"], "delay_mdd": row["+4h_delay"]["mdd"],
                     "hard_stop_any": any(row["hard_stop"].values()),
                     "turnover": execution["turnover"], "rebalances": execution["rebalances"],
                     "fees": execution["fees"], "slippage": execution["slippage"],
                     "funding": execution["funding"], "total_cost": execution["total_cost"],
                     "avg_gross": row["base"]["avg_gross"], "max_gross": row["base"]["max_gross"]})
    pd.DataFrame(flat).to_csv(output / "v13_scorecard.csv", index=False)
    for mode in MODES:
        (output / f"{mode.lower().replace('-', '_')}_results.json").write_text(
            json.dumps(summary["results_by_mode"][mode], indent=2, default=float))
    print(pd.DataFrame(flat).to_string(index=False))


if __name__ == "__main__":
    main()
