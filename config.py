"""
config.py  –  Deep Learning Alpha Engine parameters
"""

# ── Data ──────────────────────────────────────────────────────────────────────
SYMBOL        = "GBPUSD"
N_BARS        = 6000          # synthetic hourly bars to generate
INITIAL_PRICE = 1.3000
GBM_VOL       = 0.0015        # hourly vol
GBM_SEED      = 99

# ── Feature Engineering ───────────────────────────────────────────────────────
SEQ_LEN       = 60            # lookback window fed into model (60 bars)
TRAIN_FRAC    = 0.80

# ── Model Architecture ────────────────────────────────────────────────────────
LSTM_HIDDEN   = 128           # LSTM hidden units (bidirectional → ×2)
LSTM_LAYERS   = 2
D_MODEL       = 128           # Transformer / projection dimension
N_HEADS       = 4             # attention heads
DROPOUT       = 0.20

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS        = 30
BATCH_SIZE    = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
LAMBDA_RET    = 0.30          # weight on regression (return magnitude) task
GRAD_CLIP     = 1.0
SEED          = 42

# ── Walk-Forward Backtest ─────────────────────────────────────────────────────
WF_FOLDS           = 5
WF_EPOCHS_PER_FOLD = 15
TRANSACTION_COST   = 0.0002   # round-trip spread + commission
SLIPPAGE           = 0.0001

# ── Output ────────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH = "alphanet_best.pt"
CHART_OUTPUT    = "alphanet_dashboard.png"
CHART_DPI       = 150
