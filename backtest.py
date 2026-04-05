"""
backtest.py  –  Walk-forward backtest engine with transaction costs

Walk-forward scheme:
  fold 0: train on [0, fold_size*2),  test on [fold_size*2, fold_size*3)
  fold 1: train on [0, fold_size*3),  test on [fold_size*3, fold_size*4)
  ...
  Each fold retrains from scratch (no data leakage).

Signal: logit > 0 → long (+1),  logit < 0 → short (-1)
P&L:    signal × next_bar_return − |Δsignal| × round_trip_cost
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score

import config
from dataset import make_loaders, TimeSeriesDataset
from model import AlphaNet
from trainer import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WalkForwardBacktest:

    def __init__(self):
        self.tc   = config.TRANSACTION_COST
        self.slip = config.SLIPPAGE

    def run(self, X_all: np.ndarray, y_dir_all: np.ndarray,
            y_ret_all: np.ndarray, n_features: int) -> tuple[pd.DataFrame, list]:

        fold_size   = len(X_all) // (config.WF_FOLDS + 1)
        results     = []
        equity_curves = []
        scaler      = RobustScaler()

        print(f"\n  Walk-Forward Backtest  |  folds={config.WF_FOLDS}")
        print(f"  {'─'*55}")

        for fold in range(config.WF_FOLDS):
            train_end = fold_size * (fold + 2)
            test_end  = min(train_end + fold_size, len(X_all))
            min_needed = config.SEQ_LEN + 20

            if (test_end - train_end) < min_needed:
                continue

            # Scale on train, apply to test (strict OOS)
            X_tr = scaler.fit_transform(X_all[:train_end])
            X_te = scaler.transform(X_all[train_end:test_end])
            y_dir_tr = y_dir_all[:train_end]
            y_dir_te = y_dir_all[train_end:test_end]
            y_ret_tr = y_ret_all[:train_end]
            y_ret_te = y_ret_all[train_end:test_end]

            tr_ld, va_ld = make_loaders(X_tr, y_dir_tr, y_ret_tr,
                                         X_te, y_dir_te, y_ret_te)
            if len(tr_ld) == 0 or len(va_ld) == 0:
                continue

            model   = AlphaNet(n_features)
            trainer = Trainer(model, steps_per_epoch=len(tr_ld))
            trainer.fit(tr_ld, va_ld, epochs=config.WF_EPOCHS_PER_FOLD)

            # ── OOS prediction ────────────────────────────────────────────
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for X, yd, _ in va_ld:
                    lg, _ = model(X.to(DEVICE))
                    preds.extend((torch.sigmoid(lg) > 0.5).cpu().numpy().astype(int))
                    labels.extend(yd.numpy())

            preds  = np.array(preds)
            labels = np.array(labels)
            signals = np.where(preds == 1, 1, -1)

            raw_rets = y_ret_te[config.SEQ_LEN: config.SEQ_LEN + len(signals)]
            cost     = (self.tc + self.slip)
            sig_change = np.abs(np.diff(signals, prepend=signals[0]))
            strat_rets = signals * raw_rets - sig_change * cost

            cum   = np.cumprod(1 + strat_rets)
            bh    = np.cumprod(1 + raw_rets)
            dd    = float(np.min(cum / np.maximum.accumulate(cum) - 1))
            sharpe = (strat_rets.mean() / (strat_rets.std() + 1e-9)
                      * math.sqrt(252 * 24))

            res = {
                "fold":         fold + 1,
                "train_bars":   train_end,
                "test_bars":    len(preds),
                "accuracy":     round(accuracy_score(labels, preds), 4),
                "f1":           round(f1_score(labels, preds, zero_division=0), 4),
                "strat_return": round(float(cum[-1] - 1), 4),
                "bh_return":    round(float(bh[-1] - 1), 4),
                "sharpe":       round(sharpe, 3),
                "max_dd":       round(dd, 4),
                "n_trades":     int(sig_change.sum()),
            }
            results.append(res)
            equity_curves.append(cum)

            print(f"  Fold {fold+1}  acc={res['accuracy']:.4f}  "
                  f"strat={res['strat_return']:+.2%}  "
                  f"B&H={res['bh_return']:+.2%}  "
                  f"sharpe={res['sharpe']:.2f}  dd={res['max_dd']:.2%}")

        return pd.DataFrame(results), equity_curves
