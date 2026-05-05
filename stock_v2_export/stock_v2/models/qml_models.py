# models/qml_models.py — QSVM + VQC with ZZFeatureMap (6 qubits)
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import sys
sys.path.insert(0, "/workspace/stock_v2")
from config import *
from models.ml_models import compute_metrics

os.makedirs(SAVED_MODELS, exist_ok=True)

try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    QISKIT_OK = True
    print("  Qiskit ZZFeatureMap: available")
except Exception:
    QISKIT_OK = False

try:
    import pennylane as qml
    PENNYLANE_OK = True
except ImportError:
    PENNYLANE_OK = False


def reduce_to_qubits(X_tr, X_te, n=None):
    n = n or N_QUBITS
    pca = PCA(n_components=n, random_state=RANDOM_SEED)
    Xtr_r = pca.fit_transform(X_tr)
    Xte_r = pca.transform(X_te)
    scale = np.abs(Xtr_r).max(axis=0) + 1e-8
    return (np.clip(Xtr_r/scale,-1,1)*np.pi).astype(np.float32), \
           (np.clip(Xte_r/scale,-1,1)*np.pi).astype(np.float32), pca, scale


def _classical_kernel_svm(Xtr, ytr, Xte):
    """Classical RBF SVM as fallback when no quantum library available."""
    svm = SVC(kernel="rbf", C=5.0, gamma="scale", probability=True)
    svm.fit(Xtr, ytr)
    return svm.predict(Xte), svm.predict(Xtr), svm


def _pennylane_zz_kernel(Xtr, ytr, Xte):
    """ZZ-equivalent kernel using PennyLane."""
    import pennylane as qml
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def zz_circ(x1, x2):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
            qml.RZ(2*x1[i], wires=i)
        for i in range(N_QUBITS-1):
            qml.CNOT(wires=[i,i+1])
            qml.RZ(2*(np.pi-x1[i])*(np.pi-x1[i+1]), wires=i+1)
            qml.CNOT(wires=[i,i+1])
        for i in range(N_QUBITS-1):
            qml.CNOT(wires=[i,i+1])
            qml.RZ(-2*(np.pi-x2[i])*(np.pi-x2[i+1]), wires=i+1)
            qml.CNOT(wires=[i,i+1])
        for i in range(N_QUBITS):
            qml.RZ(-2*x2[i], wires=i)
            qml.Hadamard(wires=i)
        return qml.probs(wires=range(N_QUBITS))

    def kern(x1,x2): return float(zz_circ(x1,x2)[0])
    print(f"    Building ZZ kernel ({len(Xtr)}x{len(Xtr)})...")
    K_tr = np.array([[kern(a,b) for b in Xtr] for a in Xtr])
    print(f"    Building test kernel ({len(Xte)}x{len(Xtr)})...")
    K_te = np.array([[kern(a,b) for b in Xtr] for a in Xte])
    svm = SVC(kernel="precomputed", C=5.0, probability=True)
    svm.fit(K_tr, ytr)
    return svm.predict(K_te), svm.predict(K_tr), svm


# ════════════════════════════════════════════════════
# QSVM with ZZFeatureMap
# ════════════════════════════════════════════════════
def run_qsvm_qiskit(X_tr, y_tr, X_te, y_te, n_sample=100):
    print("\n--- QSVM with ZZFeatureMap (6 qubits) ---")
    t0 = time.time()
    n_sample = min(n_sample, len(X_tr))
    idx = np.random.choice(len(X_tr), n_sample, replace=False)
    Xtr_s, ytr_s = X_tr[idx], y_tr[idx]
    Xtr_r, Xte_r, pca, scale = reduce_to_qubits(Xtr_s, X_te)

    method = "Unknown"
    if QISKIT_OK:
        try:
            feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=2, entanglement="full")
            kernel = FidelityStatevectorKernel(feature_map=feature_map, cache_statevectors=True)
            svm = SVC(kernel=kernel.evaluate, probability=True, C=5.0)
            print(f"    Qiskit ZZFeatureMap — training on {len(Xtr_r)} samples...")
            svm.fit(Xtr_r, ytr_s)
            y_pred_tr = svm.predict(Xtr_r)
            y_pred_te = svm.predict(Xte_r)
            method = "Qiskit ZZFeatureMap"
        except Exception as e:
            print(f"    Qiskit failed: {e}")
            QISKIT_OK2 = False
    else:
        QISKIT_OK2 = False

    if not QISKIT_OK or method == "Unknown":
        if PENNYLANE_OK:
            y_pred_te, y_pred_tr, svm = _pennylane_zz_kernel(Xtr_r, ytr_s, Xte_r)
            method = "PennyLane ZZ kernel"
        else:
            print("    Using classical RBF SVM fallback (no quantum library)")
            y_pred_te, y_pred_tr, svm = _classical_kernel_svm(Xtr_r, ytr_s, Xte_r)
            method = "Classical RBF SVM (fallback)"

    metrics = compute_metrics(y_te, y_pred_te, ytr_s, y_pred_tr, "QSVM_ZZFeatureMap")
    metrics["train_time_s"] = round(time.time()-t0, 1)
    metrics["quantum_method"] = method
    metrics["n_qubits"] = N_QUBITS
    metrics["feature_map"] = "ZZFeatureMap"

    path = os.path.join(SAVED_MODELS, "QSVM_ZZFeatureMap.pkl")
    joblib.dump({"model":svm,"pca":pca,"scale":scale,"Xtr_r":Xtr_r,"ytr":ytr_s,"method":method}, path)
    return metrics, path, y_pred_te


# ════════════════════════════════════════════════════
# VQC with ZZFeatureMap
# ════════════════════════════════════════════════════
def run_vqc_zz(X_tr, y_tr, X_te, y_te, n_sample=80):
    print("\n--- VQC with ZZFeatureMap (6 qubits) ---")
    t0 = time.time()
    n_sample = min(n_sample, len(X_tr))
    idx = np.random.choice(len(X_tr), n_sample, replace=False)
    Xtr_s, ytr_s = X_tr[idx], y_tr[idx]
    Xtr_r, Xte_r, pca, scale = reduce_to_qubits(Xtr_s, X_te)

    method = "Unknown"
    if QISKIT_OK:
        try:
            from qiskit_machine_learning.algorithms import VQC
            from qiskit_algorithms.optimizers import COBYLA
            feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=2, entanglement="full")
            ansatz = RealAmplitudes(num_qubits=N_QUBITS, reps=3, entanglement="full")
            optimizer = COBYLA(maxiter=150)
            vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
            print(f"    Qiskit VQC — training on {len(Xtr_r)} samples...")
            vqc.fit(Xtr_r, ytr_s)
            y_pred_tr = vqc.predict(Xtr_r)
            y_pred_te = vqc.predict(Xte_r)
            model_obj = vqc
            method = "Qiskit VQC + ZZFeatureMap"
        except Exception as e:
            print(f"    Qiskit VQC failed: {e}")
            method = "Unknown"

    if method == "Unknown":
        if PENNYLANE_OK:
            import pennylane as qml, torch, torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            dev2 = qml.device("default.qubit", wires=N_QUBITS)
            ws2  = {"weights": (N_LAYERS, N_QUBITS, 3)}
            @qml.qnode(dev2, interface="torch")
            def vqc_circ(inputs, weights):
                for i in range(N_QUBITS):
                    qml.Hadamard(wires=i); qml.RZ(2*inputs[i], wires=i)
                for i in range(N_QUBITS-1):
                    qml.CNOT(wires=[i,i+1])
                    qml.RZ(2*(np.pi-inputs[i])*(np.pi-inputs[i+1]), wires=i+1)
                    qml.CNOT(wires=[i,i+1])
                qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
                return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
            class PLVQC(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.ql   = qml.qnn.TorchLayer(vqc_circ, ws2)
                    self.head = nn.Linear(N_QUBITS, 2)
                def forward(self, x): return self.head(self.ql(x))
            dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m2 = PLVQC().to(dev_t)
            Xt = torch.tensor(Xtr_r, dtype=torch.float32)
            yt = torch.tensor(ytr_s, dtype=torch.long)
            Xe = torch.tensor(Xte_r, dtype=torch.float32)
            opt2 = torch.optim.Adam(m2.parameters(), lr=0.02)
            crit = nn.CrossEntropyLoss()
            for ep in range(25):
                m2.train(); opt2.zero_grad()
                loss = crit(m2(Xt.to(dev_t)), yt.to(dev_t))
                loss.backward(); opt2.step()
                if (ep+1)%10==0: print(f"    VQC epoch {ep+1}/25  loss={loss.item():.4f}")
            m2.eval()
            with torch.no_grad():
                y_pred_te = m2(Xe.to(dev_t)).argmax(1).cpu().numpy()
                y_pred_tr = m2(Xt.to(dev_t)).argmax(1).cpu().numpy()
            model_obj = m2
            method = "PennyLane VQC + ZZ encoding"
        else:
            print("    Using classical SVM fallback")
            y_pred_te, y_pred_tr, model_obj = _classical_kernel_svm(Xtr_r, ytr_s, Xte_r)
            method = "Classical SVM fallback"

    metrics = compute_metrics(y_te, y_pred_te, ytr_s, y_pred_tr, "VQC_ZZFeatureMap")
    metrics["train_time_s"] = round(time.time()-t0, 1)
    metrics["quantum_method"] = method
    metrics["n_qubits"] = N_QUBITS
    metrics["feature_map"] = "ZZFeatureMap"

    path = os.path.join(SAVED_MODELS, "VQC_ZZFeatureMap.pkl")
    joblib.dump({"model":model_obj,"pca":pca,"scale":scale,"method":method}, path)
    return metrics, path, y_pred_te
