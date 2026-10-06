import csv
import warnings
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TILE_CUDA, TILE_TRITON = 16, 32
POWER_LIMIT = 300.0

def traffic(kernel, S):
    """Return (on-chip global bytes, shared-memory bytes, tile size) estimates."""
    if kernel == "vector_add_cuda":
        return 0.0, 0.0, 0
    if kernel == "reduction_cuda":
        return 0.0, 16.0 * S, 0                                   # tree: ~4 shared accesses/element
    if kernel == "naive_cuda":
        return 8.0 * S**3 - 8.0 * S * S, 0.0, 1                   # re-reads, no shared memory
    if kernel == "tiled_cuda":
        return 8.0 * S**3 / TILE_CUDA - 8.0 * S * S, 8.0 * S**3 + 8.0 * S**3 / TILE_CUDA, TILE_CUDA
    if kernel == "triton":                                        # shared-memory estimate is a guess
        return 8.0 * S**3 / TILE_TRITON - 8.0 * S * S, 8.0 * S**3 + 8.0 * S**3 / TILE_TRITON, TILE_TRITON
    return None                                                   # cuBLAS: unknown

kernels, y, X_rate, X_static = [], [], [], []
for r in csv.DictReader(open("energy_results.csv")):
    S = int(r["size"])
    t = traffic(r["kernel"], S)
    if t is None:
        continue
    onchip, shared, tile = t
    seconds = float(r["ms_per_call"]) / 1000
    flops = float(r["flops"])
    dram = float(r["bytes_min"])
    ai = flops / dram

    kernels.append(r["kernel"])
    y.append(float(r["avg_watts"]))
    X_rate.append([flops / seconds / 1e9, dram / seconds / 1e9, onchip / seconds / 1e9,
                   shared / seconds / 1e9, np.log10(ai), tile])
    X_static.append([np.log10(flops), np.log10(dram), np.log10(onchip + 1),
                     np.log10(shared + 1), np.log10(ai), tile])

kernels = np.array(kernels)
y = np.array(y)
feature_sets = {"activity rates": np.array(X_rate), "static (no timing)": np.array(X_static)}

linear = make_pipeline(StandardScaler(), LinearRegression())
models = {
    "mean baseline": None,
    "linear": linear,
    "linear, capped": linear,          # same model; predictions clipped to the power limit below
    "gradient boosting": GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                                   learning_rate=0.05, random_state=0),
    "MLP (16 units)": make_pipeline(StandardScaler(),
                                    MLPRegressor(hidden_layer_sizes=(16,), alpha=1e-2,
                                                 max_iter=20000, random_state=0)),
}

def leave_one_kernel_out(X, name, model):
    preds = np.zeros_like(y)
    for k in sorted(set(kernels)):
        test = kernels == k
        train = ~test
        if model is None:
            preds[test] = y[train].mean()
        else:
            m = clone(model)
            m.fit(X[train], y[train])
            preds[test] = m.predict(X[test])
    if name == "linear, capped":
        preds = np.minimum(preds, POWER_LIMIT)
    return preds

print(f"Power range in data: {y.min():.0f}-{y.max():.0f} W over {len(y)} runs, {len(set(kernels))} kernels\n")
print(f"{'model':>20} | " + " | ".join(f"{fs:>22}" for fs in feature_sets))
print(f"{'':>20} | " + " | ".join(f"{'MAE W   mean %err':>22}" for _ in feature_sets))

all_preds = {}
for name, model in models.items():
    cells = []
    for fs, X in feature_sets.items():
        p = leave_one_kernel_out(X, name, model)
        all_preds[(name, fs)] = p
        mae = np.mean(np.abs(p - y))
        mape = 100 * np.mean(np.abs(p - y) / y)
        cells.append(f"{mae:8.1f}   {mape:9.1f}%")
    print(f"{name:>20} | " + " | ".join(f"{c:>22}" for c in cells))

print("\nPer held-out kernel, activity-rate features (MAE in W):")
print(f"{'held-out kernel':>16} | " + " ".join(f"{n:>18}" for n in models))
for k in sorted(set(kernels)):
    mask = kernels == k
    row = [np.mean(np.abs(all_preds[(n, 'activity rates')][mask] - y[mask])) for n in models]
    print(f"{k:>16} | " + " ".join(f"{v:18.1f}" for v in row))
