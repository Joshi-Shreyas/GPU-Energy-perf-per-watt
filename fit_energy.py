import csv
import numpy as np

TILE_CUDA = 16                  # your tiled kernel's TILE_WIDTH
TILE_TRITON = 32                # BLOCK_SIZE used by triton_matmul
PEAK_BW = 900e9                 # bytes/s
SHORT_KERNEL_MS = 0.05

def algo_bytes(kernel, S):
    if kernel == "vector_add_cuda":
        return 12 * S                          # read A, read B, write C
    if kernel == "reduction_cuda":
        return 4 * S                           # read input once
    if kernel == "naive_cuda":
        return 8 * S**3 + 4 * S * S            # no reuse: every multiply-add reads A and B
    if kernel == "tiled_cuda":
        return 8 * S**3 / TILE_CUDA + 4 * S * S
    if kernel == "triton":
        return 8 * S**3 / TILE_TRITON + 4 * S * S
    return None                                # cuBLAS: internal tiling unknown

data = []
for r in csv.DictReader(open("energy_results.csv")):
    S = int(r["size"])
    ms = float(r["ms_per_call"])
    data.append({
        "kernel": r["kernel"], "size": S,
        "seconds": ms / 1000,
        "joules": float(r["joules_per_call"]),
        "flops": float(r["flops"]),
        "bytes_min": float(r["bytes_min"]),
        "bytes_algo": algo_bytes(r["kernel"], S),
        "short": ms < SHORT_KERNEL_MS,
    })

print("Implied bandwidth under the bytes_algo model (above 900 GB/s = caches must have helped):")
for d in data:
    if d["bytes_algo"] is not None and d["kernel"] not in ("vector_add_cuda", "reduction_cuda"):
        gbs = d["bytes_algo"] / d["seconds"] / 1e9
        note = "  <-- impossible as DRAM traffic" if gbs > PEAK_BW / 1e9 else ""
        print(f"  {d['kernel']:>12} {d['size']:>5}: {gbs:9.0f} GB/s{note}")

def fit(rows, byte_key):
    X = np.array([[d["flops"], d[byte_key], d["seconds"]] for d in rows])
    y = np.array([d["joules"] for d in rows])
    Xw = X / y[:, None]                        # divide by energy -> fit relative error
    yw = np.ones(len(rows))
    scale = np.abs(Xw).max(axis=0)             # rescale columns for numerical stability
    coef_scaled, *_ = np.linalg.lstsq(Xw / scale, yw, rcond=None)
    coef = coef_scaled / scale
    rel_err = (X @ coef - y) / y
    return coef, rel_err

cases = [
    ("bytes_min, all kernels", [d for d in data], "bytes_min"),
    ("bytes_min, no short kernels", [d for d in data if not d["short"]], "bytes_min"),
    ("bytes_algo, no cuBLAS", [d for d in data if d["bytes_algo"] is not None], "bytes_algo"),
    ("bytes_algo, no cuBLAS, no short", [d for d in data if d["bytes_algo"] is not None and not d["short"]], "bytes_algo"),
]

for title, rows, key in cases:
    coef, rel = fit(rows, key)
    a, b, c = coef
    print(f"\n=== {title} ({len(rows)} runs) ===")
    print(f"  a = {a*1e12:8.2f} pJ per FLOP")
    print(f"  b = {b*1e12:8.2f} pJ per byte")
    print(f"  c = {c:8.2f} W  (time term; expect roughly idle, ~60 W)")
    if min(coef) < 0:
        print("  WARNING: negative coefficient -> the fit can't separate these effects")
    print(f"  mean |error| = {100*np.mean(np.abs(rel)):.1f}%, worst = {100*np.max(np.abs(rel)):.1f}%")
    for k in sorted(set(d["kernel"] for d in rows)):
        errs = [rel[i] for i, d in enumerate(rows) if d["kernel"] == k]
        print(f"    {k:>16}: mean error {100*np.mean(errs):+6.1f}%")
