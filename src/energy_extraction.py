"""
energy_extraction.py
--------------------
Reads power and system telemetry through unprivileged Android kernel
interfaces. No root access or dangerous permissions required.

Interfaces used (Section 4.2 of paper):
  - /sys/class/power_supply/battery/  -> voltage, current, instantaneous power
  - /sys/devices/system/cpu/cpu*/cpufreq/ -> CPU frequency per core
  - /proc/stat                         -> CPU utilization
  - /sys/class/kgsl/kgsl-3d0/          -> Adreno GPU load
  - /proc/meminfo                       -> memory bandwidth (differential)

On non-Android systems (e.g. during offline experiments) this module
falls back to simulated readings so the rest of the pipeline still runs.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional
from scipy.fft import rfft, rfftfreq


# ------------------------------------------------------------------
# Kernel interface paths
# ------------------------------------------------------------------

POWER_SUPPLY_PATH   = "/sys/class/power_supply/battery"
CPU_FREQ_PATH       = "/sys/devices/system/cpu"
PROC_STAT_PATH      = "/proc/stat"
GPU_LOAD_PATH       = "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage"
MEMINFO_PATH        = "/proc/meminfo"


# ------------------------------------------------------------------
# Low-level readers
# ------------------------------------------------------------------

def _read_file(path: str, default=None):
    """Read a single value from a sysfs / procfs file."""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError):
        return default


def read_battery_power_uw() -> float:
    """
    Return instantaneous battery power in microwatts.
    Falls back to voltage * current if power_now is unavailable.
    """
    power = _read_file(f"{POWER_SUPPLY_PATH}/power_now")
    if power is not None:
        return float(power)

    voltage = _read_file(f"{POWER_SUPPLY_PATH}/voltage_now", "0")
    current = _read_file(f"{POWER_SUPPLY_PATH}/current_now", "0")
    return abs(float(voltage) * float(current)) / 1e6


def read_cpu_freq_khz() -> List[float]:
    """Return current frequency (kHz) for each CPU core."""
    freqs = []
    try:
        cores = sorted([
            d for d in os.listdir(CPU_FREQ_PATH)
            if d.startswith("cpu") and d[3:].isdigit()
        ])
        for core in cores:
            path = os.path.join(
                CPU_FREQ_PATH, core,
                "cpufreq", "scaling_cur_freq"
            )
            val = _read_file(path, "0")
            freqs.append(float(val))
    except FileNotFoundError:
        freqs = [0.0]
    return freqs


def read_cpu_utilization() -> float:
    """
    Parse /proc/stat and return overall CPU utilization [0, 1].
    Uses delta between two consecutive reads (100ms apart).
    """
    def _parse_stat():
        line = _read_file(PROC_STAT_PATH, "cpu 0 0 0 0 0 0 0")
        if line is None:
            return None
        parts = line.splitlines()[0].split()
        vals = list(map(int, parts[1:8]))  # user nice sys idle iowait irq softirq
        return vals

    s1 = _parse_stat()
    time.sleep(0.1)
    s2 = _parse_stat()

    if s1 is None or s2 is None:
        return 0.0

    idle1, idle2 = s1[3], s2[3]
    total1, total2 = sum(s1), sum(s2)

    delta_idle  = idle2 - idle1
    delta_total = total2 - total1

    if delta_total == 0:
        return 0.0
    return 1.0 - (delta_idle / delta_total)


def read_gpu_load() -> float:
    """Return GPU busy percentage [0, 100]. Returns 0 if unavailable."""
    val = _read_file(GPU_LOAD_PATH, "0")
    try:
        return float(val.replace("%", "").strip())
    except ValueError:
        return 0.0


def read_mem_bandwidth_kb() -> float:
    """
    Estimate memory bandwidth via differential /proc/meminfo sampling.
    Returns KB transferred in ~100ms window.
    """
    def _get_active():
        content = _read_file(MEMINFO_PATH, "")
        if not content:
            return 0
        for line in content.splitlines():
            if line.startswith("Active:"):
                return int(line.split()[1])
        return 0

    m1 = _get_active()
    time.sleep(0.1)
    m2 = _get_active()
    return abs(m2 - m1)


# ------------------------------------------------------------------
# Simulation fallback (for offline / non-Android environments)
# ------------------------------------------------------------------

def _simulate_reading(is_member: bool = False) -> Dict[str, float]:
    """
    Simulate energy readings for offline testing.
    Member inputs produce lower, tighter energy profiles
    (matching the membership-correlated divergence in the paper).
    """
    rng = np.random.default_rng()
    if is_member:
        power     = rng.normal(850, 30)    # lower, tighter
        cpu_freq  = rng.normal(1200, 50)
        cpu_util  = rng.normal(0.35, 0.03)
        gpu_load  = rng.normal(12, 2)
        mem_bw    = rng.normal(400, 20)
    else:
        power     = rng.normal(1100, 80)   # higher, noisier
        cpu_freq  = rng.normal(1600, 120)
        cpu_util  = rng.normal(0.55, 0.06)
        gpu_load  = rng.normal(25, 5)
        mem_bw    = rng.normal(700, 60)

    return {
        "power_uw":   max(0, power),
        "cpu_freq_khz": max(0, cpu_freq),
        "cpu_util":   np.clip(cpu_util, 0, 1),
        "gpu_load":   np.clip(gpu_load, 0, 100),
        "mem_bw_kb":  max(0, mem_bw),
    }


ANDROID = os.path.exists(POWER_SUPPLY_PATH)


# ------------------------------------------------------------------
# Single probe
# ------------------------------------------------------------------

def single_probe(is_member: bool = False) -> Dict[str, float]:
    """
    Take one energy reading snapshot.
    Uses real kernel interfaces on Android, simulation otherwise.
    """
    if ANDROID:
        return {
            "power_uw":     read_battery_power_uw(),
            "cpu_freq_khz": np.mean(read_cpu_freq_khz()),
            "cpu_util":     read_cpu_utilization(),
            "gpu_load":     read_gpu_load(),
            "mem_bw_kb":    read_mem_bandwidth_kb(),
        }
    else:
        return _simulate_reading(is_member=is_member)


# ------------------------------------------------------------------
# K-probe averaging  (Eq. 3 / Section 4.2 of paper)
# ------------------------------------------------------------------

def probe_energy(
    K: int = 50,
    interval_ms: int = 100,
    idle_between_s: float = 2.0,
    is_member: bool = False
) -> np.ndarray:
    """
    Issue K background probes and average readings to reduce noise
    by sqrt(K) as described in the paper.

    Args:
        K            : number of probes (default 50)
        interval_ms  : polling interval in milliseconds
        idle_between_s: idle gap between probes to let governors settle
        is_member    : simulation hint (ignored on real device)

    Returns:
        e_bar : averaged reading vector [K, 5] -> shape [5]
                [power_uw, cpu_freq_khz, cpu_util, gpu_load, mem_bw_kb]
    """
    readings = []
    keys = ["power_uw", "cpu_freq_khz", "cpu_util", "gpu_load", "mem_bw_kb"]

    for k in range(K):
        snap = single_probe(is_member=is_member)
        readings.append([snap[key] for key in keys])
        time.sleep(interval_ms / 1000.0)

        # Idle gap every 10 probes to let CPU governors return to baseline
        if (k + 1) % 10 == 0 and idle_between_s > 0:
            time.sleep(idle_between_s)

    arr = np.array(readings)          # [K, 5]
    e_bar = arr.mean(axis=0)          # [5]  noise reduced by sqrt(K)
    return e_bar


# ------------------------------------------------------------------
# Feature extraction  (Eq. 5 in paper)
# ------------------------------------------------------------------

def extract_features(
    e_bar: np.ndarray,
    trace: Optional[np.ndarray] = None,
    t_exit: float = 0.0,
    n_fft_components: int = 8
) -> np.ndarray:
    """
    Build the full feature vector F(q) from averaged energy readings.

    F(q) = [total_energy, peak_power, t_exit, spectral_components,
            cpu_util, gpu_load, mem_bw]

    Args:
        e_bar             : averaged probe vector [5]
        trace             : optional full power trace [K] for FFT
        t_exit            : early-exit timing in seconds (if available)
        n_fft_components  : number of FFT spectral components to include

    Returns:
        feature vector as 1-D numpy array
    """
    power_uw, cpu_freq, cpu_util, gpu_load, mem_bw = e_bar

    total_energy = power_uw          # proxy (no T integral without trace)
    peak_power   = power_uw

    # Spectral components via FFT on power trace (if available)
    if trace is not None and len(trace) > 1:
        fft_vals = np.abs(rfft(trace))
        spectral = fft_vals[:n_fft_components]
        # Pad if trace was short
        if len(spectral) < n_fft_components:
            spectral = np.pad(spectral, (0, n_fft_components - len(spectral)))
    else:
        spectral = np.zeros(n_fft_components)

    features = np.concatenate([
        [total_energy, peak_power, t_exit],
        spectral,
        [cpu_util, gpu_load, mem_bw]
    ])
    return features


# ------------------------------------------------------------------
# High-level: collect feature vector for one query
# ------------------------------------------------------------------

def collect_query_features(
    K: int = 50,
    t_exit: float = 0.0,
    is_member: bool = False
) -> np.ndarray:
    """
    Full pipeline: probe -> average -> extract features.

    Args:
        K         : number of probes
        t_exit    : early-exit timing captured during model inference
        is_member : simulation hint

    Returns:
        feature vector as 1-D numpy array
    """
    e_bar = probe_energy(K=K, is_member=is_member)
    features = extract_features(e_bar, t_exit=t_exit)
    return features


# ------------------------------------------------------------------
# Quick sanity test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running on Android device: {ANDROID}")
    print("Collecting features for a simulated MEMBER query...")
    f_member = collect_query_features(K=10, t_exit=0.12, is_member=True)
    print(f"  Feature vector (dim={len(f_member)}): {np.round(f_member, 3)}")

    print("Collecting features for a simulated NON-MEMBER query...")
    f_nonmember = collect_query_features(K=10, t_exit=0.21, is_member=False)
    print(f"  Feature vector (dim={len(f_nonmember)}): {np.round(f_nonmember, 3)}")

    print(f"\nMean power difference (member vs non-member): "
          f"{abs(f_member[0] - f_nonmember[0]):.1f} uW")
