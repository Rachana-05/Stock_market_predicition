# models/qnn_models.py — Hybrid QNN
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import sys
sys.path.insert(0, "/workspace/stock_v2")
from config import *
from models.ml_models import compute_metrics

os.makedirs(SAVED_MODELS, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import pennylane as qml
    PENNYLANE_OK = True
except ImportError:
    PENNYLANE_OK = False
    print("  WARNING: PennyLane not installed. HybridQNN will use classical-only fallback.")

weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}


def _make_circuit():
    if not PENNYLANE_OK:
        return None, None
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def hybrid_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Z")
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(inputs[i] * inputs[i+1], wires=i+1)
            qml.CNOT(wires=[i, i+1])
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
    return dev, hybrid_circuit


class HybridQNN(nn.Module):
    def __init__(self, n_feat=45):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(n_feat, 64), nn.BatchNorm1d(64),
            nn.GELU(), nn.Dropout(0.2), nn.Linear(64, N_QUBITS),
        )
        if PENNYLANE_OK:
            _, circuit = _make_circuit()
            self.ql = qml.qnn.TorchLayer(circuit, weight_shapes)
            self.use_quantum = True
        else:
            self.ql = nn.Linear(N_QUBITS, N_QUBITS)
            self.use_quantum = False
        self.post = nn.Sequential(nn.Linear(N_QUBITS, 16), nn.GELU(), nn.Linear(16, 2))

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]
        x = torch.tanh(self.pre(x)) * np.pi
        x = self.ql(x)
        return self.post(x)


def run_hybrid_qnn(X_tr_flat, y_tr, X_te_flat, y_te):
    print("\n--- Hybrid QNN (ZZ encoding + StronglyEntangling) ---")
    t0 = time.time()

    n_qnn = min(1500, len(X_tr_flat))
    idx   = np.random.choice(len(X_tr_flat), n_qnn, replace=False)
    Xtr_s, ytr_s = X_tr_flat[idx], y_tr[idx]

    model = HybridQNN(n_feat=X_tr_flat.shape[1]).to(DEVICE)
    Xt = torch.tensor(Xtr_s,  dtype=torch.float32)
    yt = torch.tensor(ytr_s,  dtype=torch.long)
    Xe = torch.tensor(X_te_flat, dtype=torch.float32)

    loader  = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)
    crit    = nn.CrossEntropyLoss()
    opt     = torch.optim.Adam(model.parameters(), lr=0.005)
    best_acc, best_state = 0, None

    epochs = min(EPOCHS, 20)
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val = model(Xe.to(DEVICE)).argmax(1).cpu().numpy()
            val_acc = accuracy_score(y_te, val)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    [HybridQNN] epoch {epoch+1}  val={val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred_tr = model(Xt.to(DEVICE)).argmax(1).cpu().numpy()
        y_pred_te = model(Xe.to(DEVICE)).argmax(1).cpu().numpy()

    metrics = compute_metrics(y_te, y_pred_te, ytr_s, y_pred_tr, "HybridQNN")
    metrics["train_time_s"] = round(time.time() - t0, 1)
    metrics["n_qubits"]     = N_QUBITS
    metrics["encoding"]     = "ZZ AngleEmbedding + CNOT"
    metrics["ansatz"]       = "StronglyEntanglingLayers"
    metrics["quantum_used"] = PENNYLANE_OK

    path = os.path.join(SAVED_MODELS, "HybridQNN.pt")
    torch.save(model.state_dict(), path)
    return metrics, path, y_pred_te
