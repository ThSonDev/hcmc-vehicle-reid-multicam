"""System resource monitor for the local pipeline.

Periodically samples CPU / RAM / swap / disk I/O / GPU and writes to the console +
`logs/system.jsonl`. The DRAM-less SATA disk is this machine's known bottleneck
(sustained write ~150 MB/s once the pSLC cache is exhausted), so disk is emphasized:
crossing the threshold logs at WARNING.

Run:  python monitor.py [--interval 2.0] [--disk-warn 140]
"""
import time
import argparse

import psutil
from log_utils import setup_logging

log = setup_logging("system")

# GPU via NVML (nvidia-ml-py). If there's no GPU / it's not installed, skip the GPU part.
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml = None
    _GPU = None


def gpu_fields():
    if _GPU is None:
        return {}
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(_GPU)
        mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(_GPU) / 1000.0  # mW -> W
        temp = pynvml.nvmlDeviceGetTemperature(_GPU, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "gpu_pct": util.gpu,
            "gpu_mem_mb": round(mem.used / 1024 / 1024),
            "gpu_pow_w": round(power, 1),
            "gpu_temp": temp,
        }
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description="System resource monitor.")
    ap.add_argument("--interval", type=float, default=2.0, help="Sampling interval (seconds)")
    ap.add_argument("--disk-warn", type=float, default=140.0,
                    help="Disk write threshold (MB/s) to warn at (near the pSLC floor ~150)")
    ap.add_argument("--ram-warn", type=float, default=90.0, help="RAM %% threshold to warn at")
    args = ap.parse_args()

    log.info("Monitor started", extra={"event": "init", "interval": args.interval,
                                       "gpu": _GPU is not None})

    # Prime the delta-based counters (the first call returns 0 / is meaningless).
    psutil.cpu_percent(interval=None)
    psutil.cpu_percent(percpu=True, interval=None)
    prev_disk = psutil.disk_io_counters()
    prev_t = time.time()

    try:
        while True:
            time.sleep(args.interval)
            now = time.time()
            dt = max(1e-6, now - prev_t)

            disk = psutil.disk_io_counters()
            read_mbs = (disk.read_bytes - prev_disk.read_bytes) / 1e6 / dt
            write_mbs = (disk.write_bytes - prev_disk.write_bytes) / 1e6 / dt
            prev_disk, prev_t = disk, now

            cpu = psutil.cpu_percent(interval=None)
            cores = psutil.cpu_percent(percpu=True, interval=None)
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()

            fields = {
                "event": "sample",
                "cpu_pct": round(cpu, 1),
                "cpu_peak": round(max(cores), 1) if cores else 0.0,
                "ram_pct": vm.percent,
                "ram_used_gb": round(vm.used / 1e9, 2),
                "swap_used_gb": round(sw.used / 1e9, 2),
                "disk_r_mbs": round(read_mbs, 1),
                "disk_w_mbs": round(write_mbs, 1),
            }
            fields.update(gpu_fields())

            if write_mbs >= args.disk_warn:
                log.warning("High disk write (near pSLC floor ~150 MB/s)", extra=fields)
            elif vm.percent >= args.ram_warn:
                log.warning("High RAM usage", extra=fields)
            else:
                log.info("sample", extra=fields)
    except KeyboardInterrupt:
        log.info("Monitor stopped", extra={"event": "shutdown"})
    finally:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
