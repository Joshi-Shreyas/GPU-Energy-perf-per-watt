import csv
import numpy as np

TILE_CUDA, TILE_TRITON = 16, 32
IDLE_START, IDLE_END = 59.1, 65.4

def algo_bytes(kernel, S):
    if kernel == "vector_add_cuda": return 12 * S
    if kernel == "reduction_cuda":  return 4 * S
    if kernel == "naive_cuda":      return 8 * S**3 + 4 * S * S
    if kernel == "tiled_cuda":      return 8 * S**3 / TILE_CUDA + 4 * S * S
    if kernel == "triton":          return 8 * S**3 / TILE_TRITON + 4 * S * S
    return None   # cuBLAS: unknown

raw = list(csv.DictReader(open("energy_results.csv")))
n = len(raw)
data = []
for i, r in enumerate(raw):
    S = int(r["size"])
    algo = algo_bytes(r["kernel"], S)
    if algo is None:
        continue
    seconds = float(r["ms_per_call"]) / 1000
    idle = IDLE_START + (IDLE_END - IDLE_START) * i / (n - 1)
    dram = float(r["bytes_min"])
    data.append({
        "kernel": r["kernel"], "size": S,
        "dyn_joules": float(r["joules_per_call"]) - idle * seconds,
        "flops": float(r["flops"]),
        "dram_bytes": dram,
        "onchip_bytes": algo - dram,
        "throttled": r["throttled"] == "True",
    })

def fit(rows):
    X = np.array([[d["flops"], d["dram_bytes"], d["onchip_bytes"]] for d in rows])
    y = np.array([d["dyn_joules"] for d in rows])
    Xw = X / y[:, None]
    scale = np.abs(Xw).max(axis=0)
    coef_s, *_ = np.linalg.lstsq(Xw / scale, np.ones(len(rows)), rcond=None)
    coef = coef_s / scale
    return coef, (X @ coef - y) / y

for title, rows in [("unthrottled runs only", [d for d in data if not d["throttled"]]),
                    ("all runs (excluding cuBLAS)", data)]:
    coef, rel = fit(rows)
    print(f"\n=== {title} ({len(rows)} runs) ===")
    print(f"  pJ per FLOP:        {coef[0]*1e12:8.2f}")
    print(f"  pJ per DRAM byte:   {coef[1]*1e12:8.2f}")
    print(f"  pJ per on-chip byte:{coef[2]*1e12:8.2f}")
    if min(coef) < 0:
        print("  WARNING: negative coefficient")
    print(f"  mean |error| = {100*np.mean(np.abs(rel)):.1f}%, worst = {100*np.max(np.abs(rel)):.1f}%")
    for k in sorted(set(d["kernel"] for d in rows)):
        errs = [rel[j] for j, d in enumerate(rows) if d["kernel"] == k]
        print(f"    {k:>16}: mean error {100*np.mean(errs):+6.1f}%")
