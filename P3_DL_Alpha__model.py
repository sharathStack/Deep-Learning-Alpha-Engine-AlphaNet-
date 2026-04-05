"""
model.py  –  AlphaNet: LSTM + Transformer Temporal Fusion Architecture

Architecture:
  Input  →  Linear projection  →  LayerNorm
      │
      ├─→  Bidirectional LSTM encoder   (captures sequential patterns)
      │
      └─→  Transformer encoder          (positional self-attention)
                │
           GRN gated fusion            (Temporal Fusion gate, Lim 2021)
                │
           Cross-attention             (last step queries full context)
                │
           Shared GRN representation
               / \
   Direction head  Return-magnitude head
   (BCEWithLogits)    (MSE)

References:
  Vaswani et al. (2017) – "Attention Is All You Need"
  Lim et al. (2021)     – "Temporal Fusion Transformers"
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class GatedResidualNetwork(nn.Module):
    """GRN from Temporal Fusion Transformer (gating + skip connection)."""
    def __init__(self, d_in: int, d_hid: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.fc1   = nn.Linear(d_in, d_hid)
        self.fc2   = nn.Linear(d_hid, d_out)
        self.gate  = nn.Linear(d_hid, d_out)
        self.norm  = nn.LayerNorm(d_out)
        self.skip  = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = F.elu(self.fc1(x))
        h    = self.drop(h)
        gate = torch.sigmoid(self.gate(h))
        out  = gate * self.fc2(h)
        return self.norm(out + self.skip(x))


class AlphaNet(nn.Module):

    def __init__(self, n_features: int):
        super().__init__()
        D   = config.D_MODEL
        H   = config.LSTM_HIDDEN
        L   = config.LSTM_LAYERS
        nh  = config.N_HEADS
        dr  = config.DROPOUT

        # Input
        self.input_proj = nn.Linear(n_features, D)
        self.input_norm = nn.LayerNorm(D)

        # LSTM
        self.lstm = nn.LSTM(D, H, num_layers=L, batch_first=True,
                            dropout=dr if L > 1 else 0.0,
                            bidirectional=True)
        self.lstm_proj = nn.Linear(H * 2, D)

        # Transformer
        self.pos_enc = PositionalEncoding(D, dropout=dr,
                                          max_len=config.SEQ_LEN + 10)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=D, nhead=nh, dim_feedforward=D * 4,
            dropout=dr, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Gated fusion: LSTM state ‖ Transformer state → D
        self.grn_fusion = GatedResidualNetwork(D * 2, D * 2, D, dr)

        # Cross-attention: last timestep queries full sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=nh, dropout=dr, batch_first=True
        )

        # Shared representation
        self.shared_grn = GatedResidualNetwork(D, D, D // 2, dr)

        # Task heads
        self.head_dir = nn.Sequential(
            nn.Linear(D // 2, 64), nn.ReLU(), nn.Dropout(dr), nn.Linear(64, 1)
        )
        self.head_ret = nn.Sequential(
            nn.Linear(D // 2, 64), nn.GELU(), nn.Dropout(dr), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, n_features)
        h = self.input_norm(F.relu(self.input_proj(x)))   # (B, T, D)

        lstm_out, _ = self.lstm(h)                         # (B, T, 2H)
        lstm_out    = self.lstm_proj(lstm_out)             # (B, T, D)

        tf_out = self.transformer(self.pos_enc(h))         # (B, T, D)

        fused   = self.grn_fusion(torch.cat([lstm_out, tf_out], dim=-1))  # (B, T, D)
        query   = fused[:, -1:, :]                         # (B, 1, D)
        ctx, _  = self.cross_attn(query, fused, fused)     # (B, 1, D)
        ctx     = ctx.squeeze(1)                           # (B, D)

        shared  = self.shared_grn(ctx)                     # (B, D//2)
        return self.head_dir(shared).squeeze(-1), self.head_ret(shared).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
