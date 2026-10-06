import time
import threading
import torch
import pynvml


def get_nvml_handle():
    """Match the CUDA device PyTorch is using to the right NVML device."""
    n = pynvml.nvmlDeviceGetCount()
    if n == 1:
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    # Several GPUs visible to NVML: match by UUID (NVML ignores CUDA_VISIBLE_DEVICES)
    uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if uuid is not None:
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            u = pynvml.nvmlDeviceGetUUID(h)
            u = u.decode() if isinstance(u, bytes) else u
            if u.endswith(str(uuid)):
                return h
    raise RuntimeError("Could not match the CUDA device to an NVML device")


# Throttle reasons that mean the clock was actually reduced
THROTTLE_MASK = 0
for name, default in [
    ("nvmlClocksThrottleReasonSwPowerCap", 0x04),
    ("nvmlClocksThrottleReasonHwSlowdown", 0x08),
    ("nvmlClocksThrottleReasonSwThermalSlowdown", 0x20),
    ("nvmlClocksThrottleReasonHwThermalSlowdown", 0x40),
    ("nvmlClocksThrottleReasonHwPowerBrakeSlowdown", 0x80),
]:
    THROTTLE_MASK |= getattr(pynvml, name, default)


class PowerMeter:
    def __init__(self, sample_interval=0.01):
        pynvml.nvmlInit()
        self.h = get_nvml_handle()
        self.interval = sample_interval
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.has_energy_counter = True
        except pynvml.NVMLError:
            self.has_energy_counter = False
        self.idle_watts = None

    def _energy_mj(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)  # millijoules

    def _sample_loop(self, samples, stop):
        while not stop.is_set():
            samples.append((
                pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0,  # watts
                pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),  # MHz
                pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU),
                pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.h),
            ))
            time.sleep(self.interval)

    def _run_measured(self, work):
        """Run work() while sampling; return (seconds, joules, samples)."""
        samples, stop = [], threading.Event()
        sampler = threading.Thread(target=self._sample_loop, args=(samples, stop))
        torch.cuda.synchronize()
        e0 = self._energy_mj() if self.has_energy_counter else None
        sampler.start()
        t0 = time.time()
        work()
        torch.cuda.synchronize()
        t1 = time.time()
        stop.set()
        sampler.join()
        seconds = t1 - t0
        if self.has_energy_counter:
            joules = (self._energy_mj() - e0) / 1000.0
        else:  # fallback: average sampled power x time
            joules = sum(s[0] for s in samples) / max(len(samples), 1) * seconds
        return seconds, joules, samples

    def measure_idle(self, seconds=5.0):
        seconds, joules, _ = self._run_measured(lambda: time.sleep(seconds))
        self.idle_watts = joules / seconds
        return self.idle_watts

    def measure(self, name, fn, min_seconds=3.0):
        # Warm up, then estimate how many calls fill min_seconds
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        per_call = (time.time() - t0) / 5
        iters = max(10, int(min_seconds / per_call))

        def work():
            for _ in range(iters):
                fn()

        seconds, joules, samples = self._run_measured(work)
        avg_w = joules / seconds
        dyn_j = joules - (self.idle_watts or 0.0) * seconds
        clocks = [s[1] for s in samples] or [0]
        temps = [s[2] for s in samples] or [0]
        throttled = any(s[3] & THROTTLE_MASK for s in samples)
        return {
            "name": name,
            "iters": iters,
            "ms_per_call": seconds / iters * 1000,
            "avg_watts": avg_w,
            "watts_above_idle": avg_w - (self.idle_watts or 0.0),
            "joules_per_call": joules / iters,
            "dynamic_joules_per_call": dyn_j / iters,
            "sm_clock_mhz_min": min(clocks),
            "sm_clock_mhz_avg": sum(clocks) / len(clocks),
            "temp_c_max": max(temps),
            "throttled": throttled,
        }


if __name__ == "__main__":
    meter = PowerMeter()
    name = pynvml.nvmlDeviceGetName(meter.h)
    name = name.decode() if isinstance(name, bytes) else name
    limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(meter.h) / 1000.0
    print(f"GPU: {name} | power limit: {limit_w:.0f} W | "
          f"energy counter supported: {meter.has_energy_counter}")

    # Create a CUDA context first, so idle includes it (it's present during real runs too)
    torch.zeros(1, device="cuda")
    print(f"Idle power: {meter.measure_idle():.1f} W\n")

    # Two sanity-check workloads with opposite behavior
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    x = torch.randn(64 * 1024 * 1024, device="cuda")
    y = torch.randn(64 * 1024 * 1024, device="cuda")

    for r in [meter.measure("vector add (memory-bound)", lambda: x + y),
              meter.measure("cuBLAS matmul 4096 (compute-bound)", lambda: a @ b)]:
        print(f"{r['name']}")
        print(f"  {r['ms_per_call']:.3f} ms/call over {r['iters']} calls")
        print(f"  {r['avg_watts']:.1f} W avg ({r['watts_above_idle']:.1f} W above idle)")
        print(f"  {r['joules_per_call']*1000:.2f} mJ/call total, "
              f"{r['dynamic_joules_per_call']*1000:.2f} mJ/call above idle")
        print(f"  SM clock min/avg: {r['sm_clock_mhz_min']}/{r['sm_clock_mhz_avg']:.0f} MHz, "
              f"max temp {r['temp_c_max']} C, throttled: {r['throttled']}\n")
