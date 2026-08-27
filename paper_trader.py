#!/usr/bin/env python3
"""Bybit public-data paper runner for the frozen V112B_RECOVERY4 strategy.

This module is deliberately incapable of placing orders.  It accepts no API
credentials and its HTTP client permits only two public market-data endpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import backtest


MODE = "DRY_RUN_PAPER_ONLY"
BYBIT_PUBLIC_BASE = "https://api.bybit.com"
KLINE_PATH = "/v5/market/kline"
FUNDING_PATH = "/v5/market/funding/history"
ALLOWED_PUBLIC_PATHS = frozenset((KLINE_PATH, FUNDING_PATH))
SYMBOL = "BTCUSDT"
CATEGORY = "linear"
INTERVAL = "60"
STRATEGY_NAME = "V112B_RECOVERY4"
INITIAL_EQUITY = 10_000.0
ONE_WAY_COST_BPS = backtest.CostModel().fee_bps + backtest.CostModel().slippage_bps


def utc_iso(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def finite(value: float, fallback: float = 0.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else fallback


def public_get(path: str, params: dict[str, object], retries: int = 3) -> dict:
    if path not in ALLOWED_PUBLIC_PATHS:
        raise RuntimeError(f"non-public or unsupported Bybit endpoint: {path}")
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{BYBIT_PUBLIC_BASE}{path}?{query}",
        headers={"Accept": "application/json", "User-Agent": "btc-ai-ea-paper/1.0"},
        method="GET",
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
            if payload.get("retCode") != 0:
                raise RuntimeError(f"Bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Bybit public request failed: {path}: {last_error}")


def fetch_klines(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    cursor = int(end.timestamp() * 1000)
    rows: list[list[str]] = []
    while cursor >= start_ms:
        payload = public_get(KLINE_PATH, {
            "category": CATEGORY, "symbol": SYMBOL, "interval": INTERVAL,
            "start": start_ms, "end": cursor, "limit": 1000,
        })
        page = payload["result"].get("list", [])
        if not page:
            break
        rows.extend(page)
        oldest = min(int(row[0]) for row in page)
        if oldest <= start_ms:
            break
        cursor = oldest - 1
    if not rows:
        raise RuntimeError("Bybit returned no BTCUSDT linear klines")
    frame = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "turnover"])
    frame["time"] = pd.to_datetime(pd.to_numeric(frame.time), unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.set_index("time")[["open", "high", "low", "close", "volume"]]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index().dropna()
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def fetch_funding(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    start_ms = int(start.timestamp() * 1000)
    cursor = int(end.timestamp() * 1000)
    observations: list[tuple[int, float]] = []
    while cursor >= start_ms:
        payload = public_get(FUNDING_PATH, {
            "category": CATEGORY, "symbol": SYMBOL, "endTime": cursor, "limit": 200,
        })
        page = payload["result"].get("list", [])
        if not page:
            break
        parsed = [(int(row["fundingRateTimestamp"]), float(row["fundingRate"])) for row in page]
        observations.extend((ts, rate) for ts, rate in parsed if ts >= start_ms)
        oldest = min(ts for ts, _ in parsed)
        if oldest <= start_ms:
            break
        cursor = oldest - 1
    if not observations:
        return pd.Series(dtype=float)
    series = pd.Series(
        [rate for _, rate in observations],
        index=pd.to_datetime([ts for ts, _ in observations], unit="ms", utc=True),
        dtype=float,
    )
    return series[~series.index.duplicated(keep="last")].sort_index()


def frozen_strategy() -> backtest.Strategy:
    matches = [strategy for strategy in backtest.CANDIDATES if strategy.name == STRATEGY_NAME]
    if len(matches) != 1:
        raise RuntimeError(f"frozen strategy {STRATEGY_NAME} is missing or ambiguous")
    strategy = matches[0]
    if strategy.recovery_days != 4:
        raise RuntimeError("V112B_RECOVERY4 parameters no longer match the frozen definition")
    return strategy


def latest_signal(px: pd.DataFrame, funding: pd.Series) -> dict[str, object]:
    strategy = frozen_strategy()
    features = backtest.build_features(px, strategy).dropna(
        subset=["open", "close", "rv30", "atr4h", "risk_slow"]
    ).copy()
    completed_cutoff = pd.Timestamp.now(tz="UTC").floor("4h") - pd.Timedelta(hours=4)
    features = features.loc[features.index <= completed_cutoff]
    if len(features) < 1000:
        raise RuntimeError(f"insufficient completed feature history: {len(features)} rows")
    curve, _, _, execution = backtest.simulate(
        features, funding, strategy, backtest.CostModel(), INITIAL_EQUITY, True
    )
    if curve.empty:
        raise RuntimeError("strategy replay produced no paper signal")
    latest = curve.iloc[-1]
    mark_price = float(features.loc[curve.index[-1], "open"])
    return {
        "timestamp": curve.index[-1],
        "mark_price": mark_price,
        "target_exposure": float(latest.exposure),
        "risk_state": str(latest.risk_state),
        "convex_stage": int(latest.convex_stage),
        "replay_execution": execution,
        "feature_rows": int(len(features)),
    }


def load_state(path: Path, initial_equity: float) -> dict[str, object]:
    if not path.exists():
        return {
            "paper_equity": initial_equity,
            "peak_equity": initial_equity,
            "target_exposure": 0.0,
            "mark_price": None,
            "timestamp": None,
            "cumulative_turnover": 0.0,
            "cumulative_cost": 0.0,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("mode") != MODE or state.get("strategy") != STRATEGY_NAME:
        raise RuntimeError("refusing incompatible prior paper state")
    return state


def update_paper_state(previous: dict[str, object], signal: dict[str, object],
                       funding: pd.Series, initial_equity: float) -> tuple[dict, dict]:
    timestamp = signal["timestamp"]
    prior_timestamp = pd.Timestamp(previous["timestamp"]) if previous.get("timestamp") else None
    if prior_timestamp is not None and timestamp < prior_timestamp:
        raise RuntimeError("latest market bar predates prior paper state")

    prior_equity = float(previous.get("paper_equity", initial_equity))
    prior_peak = float(previous.get("peak_equity", prior_equity))
    prior_exposure = float(previous.get("target_exposure", 0.0))
    prior_mark = previous.get("mark_price")
    current_mark = float(signal["mark_price"])
    price_return = current_mark / float(prior_mark) - 1.0 if prior_mark else 0.0
    interval_funding = 0.0
    if prior_timestamp is not None and len(funding):
        interval_funding = float(funding.loc[(funding.index > prior_timestamp) &
                                             (funding.index <= timestamp)].sum())

    equity_before_trade = prior_equity * max(1e-12, 1.0 + prior_exposure * price_return)
    funding_paid = equity_before_trade * prior_exposure * interval_funding
    equity_before_trade = max(1e-12, equity_before_trade - funding_paid)
    target = float(signal["target_exposure"])
    exposure_change = target - prior_exposure
    turnover = abs(exposure_change)
    cost = equity_before_trade * turnover * ONE_WAY_COST_BPS / 10_000.0
    equity = max(1e-12, equity_before_trade - cost)
    peak = max(prior_peak, equity)
    drawdown = equity / peak - 1.0

    state = {
        "schema_version": 1,
        "mode": MODE,
        "strategy": STRATEGY_NAME,
        "symbol": SYMBOL,
        "category": CATEGORY,
        "timestamp": utc_iso(timestamp),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "paper_equity": finite(equity),
        "peak_equity": finite(peak),
        "drawdown": finite(drawdown),
        "mark_price": finite(current_mark),
        "target_exposure": finite(target),
        "exposure_change": finite(exposure_change),
        "funding": {"interval_rate": finite(interval_funding), "paid": finite(funding_paid)},
        "turnover": finite(turnover),
        "cost": finite(cost),
        "cumulative_turnover": finite(float(previous.get("cumulative_turnover", 0.0)) + turnover),
        "cumulative_cost": finite(float(previous.get("cumulative_cost", 0.0)) + cost),
        "risk_state": signal["risk_state"],
        "convex_stage": signal["convex_stage"],
    }
    report = {
        **state,
        "price_return_since_prior_run": finite(price_return),
        "prior_timestamp": previous.get("timestamp"),
        "prior_exposure": finite(prior_exposure),
        "feature_rows": signal["feature_rows"],
        "funding_observations": int(len(funding)),
        "strategy_parameters": asdict(frozen_strategy()),
        "safety": {
            "api_key_used": False,
            "private_endpoints_present": False,
            "orders_enabled": False,
            "allowed_public_endpoints": sorted(ALLOWED_PUBLIC_PATHS),
        },
    }
    return state, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", default="paper_state")
    parser.add_argument("--lookback-days", type=int, default=500)
    parser.add_argument("--initial-equity", type=float, default=INITIAL_EQUITY)
    args = parser.parse_args()
    if args.lookback_days < 400:
        raise ValueError("lookback-days must be at least 400 for frozen daily features")

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    previous = load_state(state_path, args.initial_equity)
    end = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta(hours=1)
    start = end - pd.Timedelta(days=args.lookback_days)
    px = fetch_klines(start, end)
    funding = fetch_funding(start, end)
    signal = latest_signal(px, funding)
    state, report = update_paper_state(previous, signal, funding, args.initial_equity)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    (state_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
