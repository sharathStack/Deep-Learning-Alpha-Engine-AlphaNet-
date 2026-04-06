#  Deep Learning Alpha Engine (AlphaNet)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/DL-PyTorch-ee4c2c?logo=pytorch)
![Architecture](https://img.shields.io/badge/Model-LSTM%20%2B%20Transformer-informational)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
> Multi-task LSTM + Transformer model for directional forecasting and return magnitude regression on FX hourly bars — with 44 engineered features and strict walk-forward validation across 5 folds.
---
Project Structure
```
project3_dl_alpha/
├── config.py       ← All hyperparameters (model dims, training, walk-forward)
├── data_gen.py     ← Regime-switching GBM OHLCV with intraday seasonality
├── features.py     ← 44 technical + microstructure features
├── dataset.py      ← PyTorch Dataset + DataLoader factory
├── model.py        ← AlphaNet: BiLSTM + Transformer + GRN fusion
├── trainer.py      ← Multi-task training loop + OneCycleLR + checkpointing
├── backtest.py     ← Walk-forward OOS backtest with TC + slippage model
├── dashboard.py    ← Training curves + fold analytics dashboard
├── main.py         ← Entry point
└── requirements.txt
```
---
How to Run
```bash
cd project3_dl_alpha
pip install -r requirements.txt
python main.py
```
Expected terminal output:
```
[1] Generated 6,000 OHLCV bars [2018-01-01 → 2018-09-22]
[2] 44 features  |  5,847 bars after dropna
[3] Training AlphaNet  |  epochs=30  |  device=cuda
    Parameters: 1,243,521
    Ep   5/30  tr_loss=0.6821  val_acc=0.5312  val_f1=0.5198
    Ep  10/30  tr_loss=0.6543  val_acc=0.5489  val_f1=0.5341
    Ep  30/30  tr_loss=0.6211  val_acc=0.5621  val_f1=0.5489
    Best val accuracy: 0.5621

[4] Walk-Forward Backtest  |  folds=5
    Fold 1  acc=0.5421  strat=+4.21%  B&H=+2.11%  sharpe=1.42
    Fold 2  acc=0.5218  strat=+2.87%  B&H=+1.53%  sharpe=0.98
    ...
    Avg OOS Accuracy: 0.5398
    Avg Sharpe: 1.24

Dashboard saved → alphanet_dashboard.png
```
---
Model Architecture
```
Input (44 features, seq_len=60)
  │
  ├─→ Bidirectional LSTM (128 hidden × 2 layers → 256 out)
  │     └─→ Linear projection → d_model (128)
  │
  └─→ Transformer encoder (2 layers, 4 heads, d_model=128)
          + Positional encoding
                │
           GRN Gated Fusion  ← Concat LSTM + Transformer states
                │             (Temporal Fusion Transformer gate, Lim 2021)
           Cross-Attention   ← Last timestep queries full context
                │
           Shared GRN (d_model → d_model/2)
               / \
    Direction head    Return-magnitude head
    (BCEWithLogits)        (MSE)
```
Total parameters: 1,243,521
---
## Feature Engineering (44 features)

Category	Features

Returns	1/2/5/10-bar returns, log-returns

OHLC structure	Body ratio, upper/lower wick, HL range %

Trend	EMA(5/10/20/50/100/200), MACD, MACD signal, MACD hist, ADX

Momentum	RSI(7/14/21), Stochastic(14/21), ROC(5/10/20)

Volatility	ATR(5/14), Bollinger(20/40), Realised vol(5/20), Parkinson vol

Microstructure	Volume ratio, VWAP distance, London session dummy

---
## Walk-Forward Backtest

5 folds — expanding training window, fixed-size OOS test

No look-ahead bias — scaler fit on train, applied to test

Transaction cost model: 2 bps round-trip + 1 bp slippage

Signal: logit > 0 → long (+1), logit ≤ 0 → short (−1)

---
## Dashboard Output

`alphanet_dashboard.png` — 8-panel dashboard:

Training loss (train vs val) by epoch

Validation accuracy by epoch

Validation F1 score by epoch

Walk-forward OOS equity curves (all 5 folds overlaid)

Architecture summary panel

Strategy vs Buy & Hold return by fold

Sharpe ratio by fold

OOS accuracy by fold

---
## References

Vaswani et al. (2017). Attention Is All You Need. NeurIPS.

Lim et al. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. IJF.

Sezer et al. (2020). Financial time series forecasting with deep learning. Applied Soft Computing.

---
Requirements
```
torch>=2.2
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4
matplotlib>=3.8
```
