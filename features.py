"""
features.py  –  40+ feature engineering pipeline for FX price data

Feature categories:
  1. Returns        – 1/2/5/10-bar, log-returns
  2. OHLC structure – body ratio, wick ratios, HL range
  3. Trend          – EMA(5/10/20/50/100/200), MACD, ADX
  4. Momentum       – RSI(7/14/21), Stochastic(14/21), ROC(5/10/20)
  5. Volatility     – ATR(5/14), Bollinger(20/40), Realized vol, Parkinson
  6. Microstructure – Volume ratio, VWAP distance, session dummy

Targets:
  target_dir  – 1 if next bar close > current close, else 0
  target_ret  – next bar log-return (regression target)
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]

    # ── 1. Returns ──────────────────────────────────────────────────────────
    d["log_ret"]  = np.log(c / c.shift(1))
    d["ret_1"]    = c.pct_change(1)
    d["ret_2"]    = c.pct_change(2)
    d["ret_5"]    = c.pct_change(5)
    d["ret_10"]   = c.pct_change(10)

    # ── 2. OHLC structure ───────────────────────────────────────────────────
    rng            = (h - l).replace(0, 1e-8)
    d["body"]      = (c - o) / rng
    d["upper_wick"]= (h - c.clip(lower=o)) / rng
    d["lower_wick"]= (c.clip(upper=o) - l) / rng
    d["hl_pct"]    = rng / c

    # ── 3. EMA & MACD ────────────────────────────────────────────────────────
    for p in [5, 10, 20, 50, 100, 200]:
        ema           = c.ewm(span=p, adjust=False).mean()
        d[f"ema_{p}"] = ema
        d[f"dist_{p}"]= (c - ema) / c

    ema12          = c.ewm(span=12).mean()
    ema26          = c.ewm(span=26).mean()
    macd           = ema12 - ema26
    signal         = macd.ewm(span=9).mean()
    d["macd"]      = macd / c
    d["macd_sig"]  = signal / c
    d["macd_hist"] = (macd - signal) / c

    # ── 4. RSI ───────────────────────────────────────────────────────────────
    for p in [7, 14, 21]:
        delta   = c.diff()
        gain    = delta.clip(lower=0).ewm(span=p).mean()
        loss    = (-delta.clip(upper=0)).ewm(span=p).mean()
        d[f"rsi_{p}"] = 100 - 100 / (1 + gain / (loss + 1e-8))

    # ── 5. Stochastic ────────────────────────────────────────────────────────
    for p in [14, 21]:
        L, H        = l.rolling(p).min(), h.rolling(p).max()
        stk          = (c - L) / (H - L + 1e-8)
        d[f"stk_{p}"]= stk
        d[f"std_{p}"]= stk.rolling(3).mean()

    # ── 6. Bollinger Bands ───────────────────────────────────────────────────
    for p in [20, 40]:
        sma          = c.rolling(p).mean()
        std          = c.rolling(p).std()
        d[f"bb_pos_{p}"]  = (c - sma) / (2 * std + 1e-8)
        d[f"bb_wid_{p}"]  = 4 * std / sma

    # ── 7. ATR & Realised Vol ────────────────────────────────────────────────
    tr = pd.concat([h - l, (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr_14"]    = tr.ewm(span=14).mean() / c
    d["atr_5"]     = tr.ewm(span=5).mean()  / c
    d["rvol_5"]    = d["log_ret"].rolling(5).std()  * math.sqrt(252 * 24)
    d["rvol_20"]   = d["log_ret"].rolling(20).std() * math.sqrt(252 * 24)
    d["rvol_ratio"]= d["rvol_5"] / (d["rvol_20"] + 1e-8)

    # Parkinson volatility (OHLC estimator)
    d["park_vol"]  = np.sqrt(
        (1 / (4 * math.log(2))) * (np.log(h / l + 1e-8)**2).rolling(20).mean()
    ) * math.sqrt(252 * 24)

    # ── 8. ADX ───────────────────────────────────────────────────────────────
    plus_dm  = (h.diff()).clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    d["adx"] = ((plus_dm.ewm(span=14).mean() - minus_dm.ewm(span=14).mean()).abs()
                / (tr.ewm(span=14).mean() + 1e-8))

    # ── 9. ROC ───────────────────────────────────────────────────────────────
    for p in [5, 10, 20]:
        d[f"roc_{p}"] = c.pct_change(p)

    # ── 10. Volume microstructure ─────────────────────────────────────────────
    vol_ma5        = v.rolling(5).mean()
    d["vol_ratio"] = v / (vol_ma5 + 1)
    vwap           = ((v * (h + l + c) / 3).rolling(20).sum()
                      / (v.rolling(20).sum() + 1))
    d["vwap_dist"] = (c - vwap) / c

    # ── 11. Session dummy (London = hours 8–17) ───────────────────────────────
    if hasattr(df.index, "hour"):
        d["london_session"] = ((df.index.hour >= 8) & (df.index.hour < 17)).astype(float)
    else:
        d["london_session"] = 0.0

    # ── Targets ───────────────────────────────────────────────────────────────
    d["target_dir"] = (d["log_ret"].shift(-1) > 0).astype(int)
    d["target_ret"] = d["log_ret"].shift(-1)

    return d.dropna()


def feature_cols(df: pd.DataFrame) -> list[str]:
    excl = {"open", "high", "low", "close", "volume", "target_dir", "target_ret"}
    return [c for c in df.columns if c not in excl]
