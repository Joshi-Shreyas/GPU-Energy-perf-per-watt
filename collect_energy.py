import csv
import time
import torch
from torch.utils.cpp_extension import load
from power_measure import PowerMeter
from triton_matmul import triton_matmul

kern = load(name="kernels_ext", sources=["kernels_ext.cu"], verbose=False)
tiled = load(name="tiled_matmul", sources=["matmul_ext.cu"], verbose=False)

meter = PowerMeter()
torch.zeros(1, device="cuda")              # open the CUDA context before measuring idle
idle_start = meter.measure_idle()
print(f"Idle (start): {idle_start:.1f} W\n")

rows = []

def record(kernel, size, fn, flops, bytes_min):
    r = meter.measure(f"{kernel} {size}", fn)
    r.update(kernel=kernel, size=size, flops=flops, bytes_min=bytes_min, idle_watts=idle_start)
    rows.append(r)
    print(f"{kernel:>16} {size:>10} | {r['ms_per_call']:9.3f} ms | {r['avg_watts']:6.1f} W | "
          f"{r['joules_per_call']*1000:9.2f} mJ | {r['sm_clock_mhz_avg']:5.0f} MHz | "
          f"throttled={r['throttled']}")
    time.sleep(1)                           # let clocks and power settle between runs

# --- Memory-bound kernels ---
for N in [2**20, 2**22, 2**24, 2**26]:
    x = torch.randn(N, device="cuda")
    y = torch.randn(N, device="cuda")
    assert torch.allclose(kern.vector_add(x, y), x + y), "vector_add mismatch"
    record("vector_add_cuda", N, lambda: kern.vector_add(x, y),
           flops=N, bytes_min=3 * N * 4)            # read A, read B, write C

    xr = torch.rand(N, device="cuda")               # positive values keep the relative check stable
    assert torch.allclose(kern.reduce_sum_partials(xr).sum(), xr.sum(), rtol=1e-4), "reduction mismatch"
    record("reduction_cuda", N, lambda: kern.reduce_sum_partials(xr),
           flops=N, bytes_min=N * 4)                # read the input once (partials are tiny)

# --- Matmul kernels ---
for S in [512, 1024, 2048, 4096]:
    a = torch.randn(S, S, device="cuda")
    b = torch.randn(S, S, device="cuda")
    ref = a @ b
    for name, fn in [("naive_cuda", kern.naive_matmul),
                     ("tiled_cuda", tiled.tiled_matmul),
                     ("triton", triton_matmul)]:
        assert torch.allclose(fn(a, b), ref, atol=1e-2, rtol=1e-3), f"{name} mismatch at {S}"

    flops = 2 * S**3
    bytes_min = 3 * S * S * 4                       # read A and B once, write C once (the minimum)
    record("naive_cuda", S, lambda: kern.naive_matmul(a, b), flops, bytes_min)
    record("tiled_cuda", S, lambda: tiled.tiled_matmul(a, b), flops, bytes_min)
    record("triton", S, lambda: triton_matmul(a, b), flops, bytes_min)
    record("cublas", S, lambda: a @ b, flops, bytes_min)

idle_end = meter.measure_idle()
print(f"\nIdle (end): {idle_end:.1f} W  (start was {idle_start:.1f} W)")

with open("energy_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"Saved {len(rows)} rows to energy_results.csv")
