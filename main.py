"""
main.py  –  Deep Learning Alpha Engine entry point

Run order:
  1. Generate synthetic OHLCV
  2. Compute 40+ features
  3. Single train/val split → train AlphaNet, save best weights
  4. Walk-forward backtest (5 folds, strict OOS)
  5. Print summary & render dashboard
"""
import torch
import numpy as np
from sklearn.preprocessing import RobustScaler

import config
from data_gen  import generate_ohlcv
from features  import compute as compute_features, feature_cols
from dataset   import make_loaders
from model     import AlphaNet
from trainer   import Trainer
from backtest  import WalkForwardBacktest
import dashboard

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


def main():
    print("═" * 60)
    print("  DEEP LEARNING ALPHA ENGINE  –  AlphaNet")
    print(f"  LSTM + Transformer  |  {config.SYMBOL}")
    print("═" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1] Generating OHLCV data…")
    raw = generate_ohlcv()

    # ── 2. Features ───────────────────────────────────────────────────────────
    print("[2] Engineering features…")
    df      = compute_features(raw)
    fcols   = feature_cols(df)
    print(f"    {len(fcols)} features  |  {len(df):,} bars after dropna")

    X_all    = df[fcols].values.astype(np.float32)
    y_dir_all = df["target_dir"].values.astype(np.int64)
    y_ret_all = df["target_ret"].values.astype(np.float32)

    # ── 3. Single-fold training ───────────────────────────────────────────────
    print("\n[3] Single train/val training pass…")
    split   = int(len(X_all) * config.TRAIN_FRAC)
    scaler  = RobustScaler()
    X_tr    = scaler.fit_transform(X_all[:split])
    X_va    = scaler.transform(X_all[split:])

    tr_ld, va_ld = make_loaders(
        X_tr, y_dir_all[:split], y_ret_all[:split],
        X_va, y_dir_all[split:], y_ret_all[split:],
    )

    model   = AlphaNet(n_features=len(fcols))
    print(f"    Parameters: {model.n_params():,}")
    trainer = Trainer(model, steps_per_epoch=len(tr_ld))
    trainer.fit(tr_ld, va_ld)

    final_metrics = trainer.evaluate(va_ld)
    print(f"\n    Final val  →  acc={final_metrics['accuracy']:.4f}  "
          f"f1={final_metrics['f1']:.4f}")
    trainer.save()

    # ── 4. Walk-forward backtest ──────────────────────────────────────────────
    print("\n[4] Walk-forward backtest…")
    wf = WalkForwardBacktest()
    results, equity_curves = wf.run(X_all, y_dir_all, y_ret_all, len(fcols))

    print("\n  Walk-Forward Summary:")
    if not results.empty:
        print(results.to_string(index=False))
        print(f"\n  Avg OOS Accuracy : {results['accuracy'].mean():.4f}")
        print(f"  Avg Sharpe       : {results['sharpe'].mean():.3f}")
        print(f"  Avg Strategy Ret : {results['strat_return'].mean():+.2%}")
        print(f"  Avg B&H Ret      : {results['bh_return'].mean():+.2%}")

    # ── 5. Dashboard ──────────────────────────────────────────────────────────
    print("\n[5] Rendering dashboard…")
    dashboard.plot(trainer.history, results, equity_curves, fcols)

    print("\n  Done ✓")


if __name__ == "__main__":
    main()
