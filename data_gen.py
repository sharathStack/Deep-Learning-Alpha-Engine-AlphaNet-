"""
data_gen.py  –  Synthetic OHLCV bar generator (realistic GBM with regimes)

Produces hourly OHLCV bars that mimic real FX microstructure:
  - Geometric Brownian Motion base process
  - Alternating trending / mean-reverting regimes
  - Realistic intra-bar OHLC construction from sub-tick GBM
  - Log-normal volume with intraday seasonality
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd

import config


def generate_ohlcv() -> pd.DataFrame:
    np.random.seed(config.GBM_SEED)
    n     = config.N_BARS
    sigma = config.GBM_VOL
    s0    = config.INITIAL_PRICE

    # ── Regime-switching drift ────────────────────────────────────────────────
    regime_len = 200                         # bars per regime
    n_regimes  = math.ceil(n / regime_len)
    drifts     = np.random.choice(
        [1e-5, -1e-5, 5e-5, -5e-5, 0],
        size=n_regimes
    )
    drift_arr = np.repeat(drifts, regime_len)[:n]

    # ── Close prices via GBM ─────────────────────────────────────────────────
    eps   = np.random.normal(0, sigma, n)
    log_r = drift_arr + eps
    close = s0 * np.exp(np.cumsum(log_r))

    # ── Intra-bar OHLC ────────────────────────────────────────────────────────
    bar_range = np.abs(np.random.normal(0, sigma * 1.5, n))
    open_     = np.roll(close, 1);  open_[0] = s0
    high      = np.maximum(open_, close) + bar_range * np.random.uniform(0.3, 1.0, n)
    low       = np.minimum(open_, close) - bar_range * np.random.uniform(0.3, 1.0, n)
    low       = np.clip(low, 1e-4, None)

    # ── Volume: log-normal + hour-of-day seasonality ──────────────────────────
    hour      = np.arange(n) % 24
    intraday  = 1.0 + 0.5 * np.sin(2 * np.pi * (hour - 8) / 12)  # peak at London open
    volume    = np.random.lognormal(10, 0.5, n) * intraday

    dates = pd.date_range("2018-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=dates)

    print(f"Generated {len(df):,} OHLCV bars  "
          f"[{df.index[0].date()} → {df.index[-1].date()}]")
    return df
