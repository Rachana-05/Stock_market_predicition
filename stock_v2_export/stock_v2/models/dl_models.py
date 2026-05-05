# models/dl_models.py — DNN, Transformer, CNN-LSTM, BiLSTM, TCN
import os, time, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import sys
sys.path.insert(0, "/workspace/stock_v2")
from config import *
from models.ml_models import compute_metrics, train_torch

os.makedirs(SAVED_MODELS, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────────────────────────────────
# DNN (Dense Neural Network) — 5 layers with BatchNorm
# Professor focus: fit_score vs predict_score
# ────────────────────────────────────────────────────
class DNNModel(nn.Module):
    def __init__(self, n_feat=45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        if x.dim() == 3:          # if sequence input, use last timestep
            x = x[:, -1, :]
        return self.net(x)


def run_dnn(X_tr_flat, y_tr, X_te_flat, y_te):
    print("\n--- DNN (5-layer Dense) ---")
    model = DNNModel(n_feat=X_tr_flat.shape[1])
    return train_torch(model, X_tr_flat, y_tr, X_te_flat, y_te, "DNN", seq=False)


# ────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ────────────────────────────────────────────────────
# TRANSFORMER
# ────────────────────────────────────────────────────
class StockTransformer(nn.Module):
    def __init__(self, n_feat=45, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        self.proj  = nn.Linear(n_feat, d_model)
        self.pe    = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                          dropout=0.2, batch_first=True, norm_first=True)
        self.enc   = nn.TransformerEncoder(enc, num_layers)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.head  = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.pe(self.proj(x))
        x = self.enc(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.head(x)


def run_transformer(X_tr_seq, y_tr, X_te_seq, y_te):
    print("\n--- Transformer ---")
    model = StockTransformer(n_feat=X_tr_seq.shape[2])
    return train_torch(model, X_tr_seq, y_tr, X_te_seq, y_te, "Transformer")


# ────────────────────────────────────────────────────
# CNN-LSTM
# ────────────────────────────────────────────────────
class CNNLSTMModel(nn.Module):
    def __init__(self, n_feat=45):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feat, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(128, 128, 2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(),
                                   nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def run_cnn_lstm(X_tr_seq, y_tr, X_te_seq, y_te):
    print("\n--- CNN-LSTM ---")
    model = CNNLSTMModel(n_feat=X_tr_seq.shape[2])
    return train_torch(model, X_tr_seq, y_tr, X_te_seq, y_te, "CNNLSTM")


# ────────────────────────────────────────────────────
# BILSTM
# ────────────────────────────────────────────────────
class BiLSTMModel(nn.Module):
    def __init__(self, n_feat=45, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, 2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.bn   = nn.BatchNorm1d(hidden * 2)
        self.head = nn.Sequential(nn.Dropout(0.3),
                                   nn.Linear(hidden * 2, 64),
                                   nn.GELU(), nn.Linear(64, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.bn(out[:, -1, :]))


def run_bilstm(X_tr_seq, y_tr, X_te_seq, y_te):
    print("\n--- BiLSTM ---")
    model = BiLSTMModel(n_feat=X_tr_seq.shape[2])
    return train_torch(model, X_tr_seq, y_tr, X_te_seq, y_te, "BiLSTM")


# ────────────────────────────────────────────────────
# TCN
# ────────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, padding):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=padding))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=padding))
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(0.2)
        self.down  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        o = self.drop(self.act(self.conv1(x)))
        o = self.drop(self.act(self.conv2(o)))
        r = x if self.down is None else self.down(x)
        return self.act(o[:, :, :r.size(2)] + r)


class TCNModel(nn.Module):
    def __init__(self, n_feat=45, channels=[128, 128, 128, 128]):
        super().__init__()
        layers = []
        in_ch  = n_feat
        for i, ch in enumerate(channels):
            dil = 2 ** i
            pad = (3 - 1) * dil
            layers.append(TemporalBlock(in_ch, ch, 3, dil, pad))
            in_ch = ch
        self.net  = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(channels[-1], 64),
                                   nn.GELU(), nn.Dropout(0.2),
                                   nn.Linear(64, 2))

    def forward(self, x):
        out = self.net(x.permute(0, 2, 1))
        return self.head(out[:, :, -1])


def run_tcn(X_tr_seq, y_tr, X_te_seq, y_te):
    print("\n--- TCN ---")
    model = TCNModel(n_feat=X_tr_seq.shape[2])
    return train_torch(model, X_tr_seq, y_tr, X_te_seq, y_te, "TCN")
