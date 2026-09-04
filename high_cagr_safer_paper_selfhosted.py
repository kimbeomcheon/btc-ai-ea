#!/usr/bin/env python3
"""Korea self-hosted launcher for HIGH_CAGR_FUNDING_AWARE_SAFER_V1.

Frozen strategy logic remains in high_cagr_safer_paper.py.
This launcher only adds a strict Bybit-specific funding cache:
- first successful run seeds the requested funding history;
- later runs fetch only a 24-hour overlap from the latest cached settlement;
- another exchange's funding is never substituted.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import high_cagr_safer_paper as core

ORIGINAL_FETCH_FUNDING = core.fetch_public_funding
CACHE_FILENAME = "funding_cache.csv"
INCREMENTAL_OVERLAP = pd.Timedelta(hours=24)

def _state_dir() -> Path:
    return Path(os.environ.get("SAFER_STATE_DIR", "paper_state_high_cagr_safer"))

def _cache_path() -> Path:
    d = _state_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / CACHE_FILENAME

def _load_cache() -> pd.Series:
    p = _cache_path()
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    if not {"timestamp", "funding_rate"}.issubset(df.columns):
        raise RuntimeError("invalid funding cache schema")
    idx = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    vals = pd.to_numeric(df["funding_rate"], errors="raise").astype(float)
    s = pd.Series(vals.to_numpy(), index=idx, dtype=float)
    return s[~s.index.duplicated(keep="last")].sort_index()

def _save_cache(s: pd.Series) -> None:
    p = _cache_path()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame({
        "timestamp": [core.iso(x) for x in s.index],
        "funding_rate": [float(v) for v in s.to_numpy()],
    })
    tmp = p.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(p)

def fetch_cached_bybit_funding(start, end):
    """Fetch/merge Bybit settled funding only."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    cached = _load_cache()
    if cached.empty:
        fetch_start = start
        mode = "seed-full-history"
    else:
        fetch_start = max(start, cached.index[-1] - INCREMENTAL_OVERLAP)
        mode = "incremental-24h-overlap"

    fresh, source = ORIGINAL_FETCH_FUNDING(fetch_start, end)
    if not cached.empty and fresh.empty:
        raise RuntimeError("Bybit incremental funding fetch returned no observations")

    merged = pd.concat([cached, fresh]) if not cached.empty else fresh.copy()
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged = merged.loc[(merged.index >= start) & (merged.index <= end)]

    if len(merged) < 2:
        raise RuntimeError("insufficient Bybit settled funding observations after cache merge")

    _save_cache(merged)
    return merged, f"{source} ({mode}; cached={len(cached)}; fresh={len(fresh)}; merged={len(merged)})"

core.fetch_public_funding = fetch_cached_bybit_funding

if __name__ == "__main__":
    core.main()
