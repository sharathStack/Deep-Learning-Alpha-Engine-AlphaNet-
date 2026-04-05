"""
dataset.py  –  PyTorch Dataset and DataLoader factory for time-series windows

Each sample: (X[t-seq:t], y_dir[t], y_ret[t])
  X       shape: (seq_len, n_features)
  y_dir   scalar int  {0, 1}
  y_ret   scalar float
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


class TimeSeriesDataset(Dataset):

    def __init__(self, X: np.ndarray, y_dir: np.ndarray,
                 y_ret: np.ndarray):
        self.X     = torch.FloatTensor(X)
        self.y_dir = torch.LongTensor(y_dir)
        self.y_ret = torch.FloatTensor(y_ret)
        self.seq   = config.SEQ_LEN

    def __len__(self) -> int:
        return len(self.X) - self.seq

    def __getitem__(self, idx: int):
        x = self.X[idx: idx + self.seq]              # (seq_len, features)
        return (x,
                self.y_dir[idx + self.seq],
                self.y_ret[idx + self.seq])


def make_loaders(X_tr: np.ndarray, y_dir_tr: np.ndarray, y_ret_tr: np.ndarray,
                 X_va: np.ndarray, y_dir_va: np.ndarray, y_ret_va: np.ndarray,
                 ) -> tuple[DataLoader, DataLoader]:

    tr_ds = TimeSeriesDataset(X_tr, y_dir_tr, y_ret_tr)
    va_ds = TimeSeriesDataset(X_va, y_dir_va, y_ret_va)

    tr_ld = DataLoader(tr_ds, batch_size=config.BATCH_SIZE,
                       shuffle=True, drop_last=True, pin_memory=False)
    va_ld = DataLoader(va_ds, batch_size=config.BATCH_SIZE,
                       shuffle=False, pin_memory=False)
    return tr_ld, va_ld
