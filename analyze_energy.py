import csv
import matplotlib
matplotlib.use("Agg")                  # no display on the cluster; save to file
import matplotlib.pyplot as plt

PEAK_BW = 900e9                        # V100 memory bandwidth, bytes/s
CORES = 5120                           # V100 FP32 CUDA cores
FLOPS_PER_CORE_PER_CYCLE = 2           # one fused multiply-add
IDLE_START, IDLE_END = 59.1, 65.4      # from collect_energy.py output
SHORT_KERNEL_MS = 0.05                 # under 50 us, launch overhead may matter

rows = list(csv.DictReader(open("energy_results.csv")))
n = len(rows)
results = []

for i, r in enumerate(rows):
    flops = float(r["flops"])
    nbytes = float(r["bytes_min"])
    ms = float(r["ms_per_call"])
    s = ms / 1000
    watts = float(r["avg_watts"])
    joules = float(r["joules_per_call"])
    clock = float(r["sm_clock_mhz_avg"])

    idle = IDLE_START + (IDLE_END - IDLE_START) * i / (n - 1)   # interpolate drift by run order
    dyn_watts = watts - idle
    gflops = flops / s / 1e9
    ai = flops / nbytes                                           # FLOPs per byte (minimum traffic)
    peak_compute = CORES * FLOPS_PER_CORE_PER_CYCLE * clock * 1e6 # at the clock actually measured
    attainable = min(peak_compute, ai * PEAK_BW)

    results.append({
        "kernel": r["kernel"], "size": int(r["size"]), "ms": ms,
        "gflops": gflops, "gbs": nbytes / s / 1e9, "ai": ai,
        "bound": "memory" if ai * PEAK_BW < peak_compute else "compute",
        "pct_of_roofline": 100 * (flops / s) / attainable,
        "watts": watts, "idle_watts": idle, "dyn_watts": dyn_watts,
        "joules": joules, "dyn_joules": dyn_watts * s,
        "pj_per_flop": joules / flops * 1e12,
        "pj_per_byte": joules / nbytes * 1e12,
        "gflop_per_joule": gflops / watts,
        "dyn_gflop_per_joule": gflops / dyn_watts,
        "clock_mhz": clock, "throttled": r["throttled"],
        "short_kernel": ms < SHORT_KERNEL_MS,
    })

with open("metrics_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

print(f"{'kernel':>16} {'size':>9} | {'GFLOPS':>8} {'GB/s':>7} {'AI':>7} {'%roof':>6} | "
      f"{'W':>6} {'dynW':>6} {'pJ/FLOP':>9} {'GFLOP/J':>8} | flags")
for m in results:
    flags = []
    if m["throttled"] == "True":
        flags.append("throttled")
    if m["short_kernel"]:
        flags.append("short")
    print(f"{m['kernel']:>16} {m['size']:>9} | {m['gflops']:8.1f} {m['gbs']:7.1f} {m['ai']:7.2f} "
          f"{m['pct_of_roofline']:5.1f}% | {m['watts']:6.1f} {m['dyn_watts']:6.1f} "
          f"{m['pj_per_flop']:9.1f} {m['gflop_per_joule']:8.2f} | {','.join(flags)}")

# How much does the idle drift matter? Compare dynamic power using start vs end idle.
worst = max((IDLE_END - IDLE_START) / (m["watts"] - IDLE_START) for m in results)
print(f"\nIdle drift sensitivity: dynamic power changes by up to {100*worst:.1f}% "
      f"depending on whether start or end idle is used")

# --- Roofline plot ---
fig, ax = plt.subplots(figsize=(8, 6))
x = [0.05, 2000]
ax.plot(x, [v * PEAK_BW / 1e9 for v in x], "k-", lw=1)
peak_1530 = CORES * FLOPS_PER_CORE_PER_CYCLE * 1530e6 / 1e9
min_clock = min(m["clock_mhz"] for m in results)
peak_min = CORES * FLOPS_PER_CORE_PER_CYCLE * min_clock * 1e6 / 1e9
ax.axhline(peak_1530, color="k", lw=1, label=f"compute peak @1530 MHz ({peak_1530:.0f} GFLOPS)")
ax.axhline(peak_min, color="k", lw=1, ls="--", label=f"compute peak @{min_clock:.0f} MHz (throttled)")

markers = {"vector_add_cuda": "o", "reduction_cuda": "s", "naive_cuda": "^",
           "tiled_cuda": "v", "triton": "D", "cublas": "*"}
for k, mk in markers.items():
    pts = [m for m in results if m["kernel"] == k]
    ax.scatter([p["ai"] for p in pts], [p["gflops"] for p in pts], marker=mk, s=60, label=k)
    for p in pts:
        ax.annotate(str(p["size"]), (p["ai"], p["gflops"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1, 30000)
ax.set_xlabel("Arithmetic intensity (FLOPs per byte, minimum traffic)")
ax.set_ylabel("Achieved GFLOPS")
ax.set_title("V100 roofline: measured kernels")
ax.legend(fontsize=7, loc="lower right")
ax.grid(True, which="both", ls=":", lw=0.5)
fig.tight_layout()
fig.savefig("roofline.png", dpi=150)
print("Saved metrics_results.csv and roofline.png")
