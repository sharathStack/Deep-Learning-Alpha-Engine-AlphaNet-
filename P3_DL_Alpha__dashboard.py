"""
dashboard.py  –  AlphaNet results visualisation
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

import config

DARK, GRID_C = "#0f1117", "#1a1d2e"
WHITE        = "#e8eaf6"
GREEN        = "#00d4aa"
RED          = "#ff4d6d"
BLUE         = "#4d9de0"
AMBER        = "#f7b731"
PURPLE       = "#a55eea"
COLORS       = [GREEN, BLUE, AMBER, RED, PURPLE]


def _style(ax):
    ax.set_facecolor(GRID_C)
    ax.tick_params(colors=WHITE, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#2a2d3e")
    ax.title.set_color(WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.grid(alpha=0.18, color="#3a3d50")


def plot(history: dict, results_df: pd.DataFrame,
         equity_curves: list, feature_names: list) -> None:

    fig = plt.figure(figsize=(20, 12), facecolor=DARK)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.40)

    # ── 1. Training loss ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1)
    ax1.plot(history["train_loss"], color=RED,   linewidth=1.5, label="Train")
    ax1.plot(history["val_loss"],   color=BLUE,  linewidth=1.5, label="Val")
    ax1.set_title("Multi-Task Loss", fontweight="bold")
    ax1.legend(fontsize=8, facecolor=GRID_C, labelcolor=WHITE)

    # ── 2. Validation accuracy ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2)
    ax2.plot(history["val_acc"], color=GREEN, linewidth=1.5, marker="o", markersize=3)
    ax2.axhline(0.5, color=AMBER, linestyle="--", alpha=0.6, label="Random baseline")
    ax2.set_title("Validation Accuracy", fontweight="bold")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE)

    # ── 3. Validation F1 ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _style(ax3)
    ax3.plot(history["val_f1"], color=PURPLE, linewidth=1.5, marker="s", markersize=3)
    ax3.set_title("Validation F1 Score", fontweight="bold")
    ax3.set_ylabel("F1")

    # ── 4. Walk-forward equity curves ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    _style(ax4)
    for i, curve in enumerate(equity_curves):
        col = COLORS[i % len(COLORS)]
        ax4.plot(curve, color=col, linewidth=1.3, label=f"Fold {i+1}", alpha=0.85)
    ax4.axhline(1.0, color=WHITE, linestyle="--", alpha=0.3)
    ax4.set_title("Walk-Forward OOS Equity Curves", fontweight="bold")
    ax4.set_ylabel("Equity (start=1.0)")
    ax4.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE)

    # ── 5. Fold bar chart: strategy vs B&H ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    _style(ax5)
    if not results_df.empty:
        folds = results_df["fold"].astype(str)
        x     = np.arange(len(folds))
        ax5.bar(x - 0.2, results_df["strat_return"] * 100, 0.38,
                label="Strategy", color=GREEN, alpha=0.85)
        ax5.bar(x + 0.2, results_df["bh_return"]    * 100, 0.38,
                label="Buy & Hold", color=BLUE, alpha=0.85)
        ax5.axhline(0, color=WHITE, linestyle="--", alpha=0.3)
        ax5.set_xticks(x)
        ax5.set_xticklabels([f"F{f}" for f in folds], color=WHITE, fontsize=8)
        ax5.set_title("Return by Fold (%)", fontweight="bold")
        ax5.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE)

    # ── 6. Sharpe by fold ─────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    _style(ax6)
    if not results_df.empty:
        bar_colors = [GREEN if s > 0 else RED for s in results_df["sharpe"]]
        ax6.bar(results_df["fold"].astype(str), results_df["sharpe"],
                color=bar_colors, alpha=0.85)
        ax6.axhline(0, color=WHITE, linestyle="--", alpha=0.3)
        ax6.set_title("Sharpe Ratio by Fold (ann.)", fontweight="bold")

    # ── 7. Accuracy by fold ───────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    _style(ax7)
    if not results_df.empty:
        ax7.bar(results_df["fold"].astype(str), results_df["accuracy"] * 100,
                color=AMBER, alpha=0.85)
        ax7.axhline(50, color=RED, linestyle="--", alpha=0.6, label="50% baseline")
        ax7.set_title("OOS Accuracy by Fold (%)", fontweight="bold")
        ax7.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE)

    # ── 8. Feature count & architecture summary ───────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    _style(ax8)
    ax8.axis("off")
    lines = [
        ("AlphaNet Architecture", ""),
        ("", ""),
        ("  Input features",   str(len(feature_names))),
        ("  Sequence length",  str(config.SEQ_LEN)),
        ("  LSTM hidden",      f"{config.LSTM_HIDDEN} (bi)"),
        ("  LSTM layers",      str(config.LSTM_LAYERS)),
        ("  Transformer d",    str(config.D_MODEL)),
        ("  Attn heads",       str(config.N_HEADS)),
        ("  WF folds",         str(config.WF_FOLDS)),
        ("  Epochs/fold",      str(config.WF_EPOCHS_PER_FOLD)),
        ("  TC + slippage",    f"{(config.TRANSACTION_COST+config.SLIPPAGE)*10000:.1f}bps"),
    ]
    for i, (k, v) in enumerate(lines):
        color = AMBER if i == 0 else WHITE
        weight= "bold" if i == 0 else "normal"
        ax8.text(0.04, 0.97 - i * 0.086, f"{k:<24}{v}",
                 transform=ax8.transAxes, fontsize=8.5, color=color,
                 fontweight=weight, family="monospace")
    ax8.set_title("Model Summary", fontweight="bold")

    fig.suptitle(f"AlphaNet Deep Learning Alpha Engine  ─  {config.SYMBOL}",
                 fontsize=14, fontweight="bold", color=WHITE, y=1.01)
    plt.savefig(config.CHART_OUTPUT, dpi=config.CHART_DPI,
                bbox_inches="tight", facecolor=DARK)
    print(f"\nDashboard saved → {config.CHART_OUTPUT}")
