"""
trainer.py  –  Multi-task training loop for AlphaNet

Loss = BCE(direction) + λ · MSE(return magnitude)

Features:
  - AdamW optimiser with OneCycleLR schedule
  - Gradient clipping
  - Best-model checkpoint (by val accuracy)
  - Training history logging
"""
from __future__ import annotations
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

import config
from model import AlphaNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(self, model: AlphaNet, steps_per_epoch: int):
        self.model     = model.to(DEVICE)
        self.opt       = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt,
            max_lr=config.LEARNING_RATE * 10,
            steps_per_epoch=steps_per_epoch,
            epochs=config.EPOCHS,
        )
        self.lambda_ret = config.LAMBDA_RET
        self.history: dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []
        }
        self._best_acc    = 0.0
        self._best_state  = None

    # ── Loss ──────────────────────────────────────────────────────────────────
    def _loss(self, logit, pred_ret, y_dir, y_ret):
        l_dir = F.binary_cross_entropy_with_logits(logit, y_dir.float())
        l_ret = F.mse_loss(pred_ret, y_ret.float())
        return l_dir + self.lambda_ret * l_ret

    # ── Single epoch ──────────────────────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for X, yd, yr in loader:
            X, yd, yr = X.to(DEVICE), yd.to(DEVICE), yr.to(DEVICE)
            self.opt.zero_grad()
            logit, pred_ret = self.model(X)
            loss = self._loss(logit, pred_ret, yd, yr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP)
            self.opt.step()
            try:
                self.scheduler.step()
            except Exception:
                pass
            total += loss.item()
        return total / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_pred, all_label, total = [], [], 0.0
        for X, yd, yr in loader:
            X, yd, yr = X.to(DEVICE), yd.to(DEVICE), yr.to(DEVICE)
            logit, pred_ret = self.model(X)
            total      += self._loss(logit, pred_ret, yd, yr).item()
            preds       = (torch.sigmoid(logit) > 0.5).cpu().numpy()
            all_pred.extend(preds)
            all_label.extend(yd.cpu().numpy())
        acc = accuracy_score(all_label, all_pred)
        f1  = f1_score(all_label, all_pred, average="binary", zero_division=0)
        return {"loss": total / max(len(loader), 1), "accuracy": acc, "f1": f1}

    # ── Full fit ──────────────────────────────────────────────────────────────
    def fit(self, tr_loader: DataLoader, va_loader: DataLoader,
            epochs: int = None) -> None:
        epochs = epochs or config.EPOCHS
        print(f"\n  Training AlphaNet  |  epochs={epochs}  |  device={DEVICE}")
        print(f"  Params: {self.model.n_params():,}")
        print(f"  {'─'*52}")

        for ep in range(1, epochs + 1):
            t0         = time.time()
            tr_loss    = self.train_epoch(tr_loader)
            va_metrics = self.evaluate(va_loader)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_metrics["loss"])
            self.history["val_acc"].append(va_metrics["accuracy"])
            self.history["val_f1"].append(va_metrics["f1"])

            if va_metrics["accuracy"] > self._best_acc:
                self._best_acc   = va_metrics["accuracy"]
                self._best_state = {k: v.clone()
                                    for k, v in self.model.state_dict().items()}

            if ep % 5 == 0 or ep <= 3:
                print(f"  Ep {ep:>3}/{epochs}  "
                      f"tr_loss={tr_loss:.4f}  "
                      f"val_acc={va_metrics['accuracy']:.4f}  "
                      f"val_f1={va_metrics['f1']:.4f}  "
                      f"({time.time()-t0:.1f}s)")

        if self._best_state:
            self.model.load_state_dict(self._best_state)
        print(f"\n  Best val accuracy: {self._best_acc:.4f}")

    def save(self, path: str = None) -> None:
        path = path or config.MODEL_SAVE_PATH
        torch.save(self.model.state_dict(), path)
        print(f"  Model saved → {path}")
