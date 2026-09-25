"""Logging helpers: console + file loggers, machine info, per-stage runtime and peak memory."""
import json
import logging
import sys
import threading
import time
from datetime import datetime

import psutil

from .config import LOG_DIR

GB = 1024 ** 3
MB = 1024 ** 2
STAGE_METRICS_FILE = "stage_metrics.jsonl"


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to stdout and to logs/<name>.log (idempotent)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def machine_info() -> dict:
    """Return total/available RAM (GB) and physical/logical CPU core counts."""
    vm = psutil.virtual_memory()
    return {
        "ram_total_gb": round(vm.total / GB, 2),
        "ram_available_gb": round(vm.available / GB, 2),
        "ram_used_pct": vm.percent,
        "cpu_physical": psutil.cpu_count(logical=False),
        "cpu_logical": psutil.cpu_count(logical=True),
    }


def rss_mb() -> float:
    """Return the current process resident memory in MB."""
    return psutil.Process().memory_info().rss / MB


class StageTimer:
    """Context manager that logs a stage's wall time and peak RSS.

    Peak RSS is sampled by a background thread (the OS process-lifetime peak
    cannot be reset per stage). One JSON line per stage is appended to
    logs/stage_metrics.jsonl.
    """

    def __init__(self, stage: str, logger: logging.Logger, interval: float = 0.1, **extra):
        """Store the stage name, logger, sampling interval and extra fields to record."""
        self.stage, self.logger, self.interval, self.extra = stage, logger, interval, extra
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self.peak_mb = 0.0

    def _sample(self) -> None:
        """Poll RSS until stopped, keeping the maximum."""
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / MB)
            self._stop.wait(self.interval)

    def __enter__(self) -> "StageTimer":
        """Start the clock and the memory sampler."""
        self.start_mb = rss_mb()
        self.peak_mb = self.start_mb
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self.logger.info("[%s] start (rss %.0f MB)", self.stage, self.start_mb)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Stop sampling, log the metrics and append them to the JSONL file."""
        self._stop.set()
        self._thread.join()
        seconds = time.perf_counter() - self._t0
        end_mb = rss_mb()
        self.peak_mb = max(self.peak_mb, end_mb)
        status = "ok" if exc_type is None else f"error: {exc_type.__name__}"
        self.logger.info("[%s] %s in %.1fs | peak rss %.0f MB | end rss %.0f MB",
                         self.stage, status, seconds, self.peak_mb, end_mb)
        record = {"time": datetime.now().isoformat(timespec="seconds"), "stage": self.stage,
                  "status": status, "seconds": round(seconds, 2), "start_rss_mb": round(self.start_mb),
                  "peak_rss_mb": round(self.peak_mb), "end_rss_mb": round(end_mb), **self.extra}
        with open(LOG_DIR / STAGE_METRICS_FILE, "a", encoding="utf-8", newline="") as f:
            f.write(json.dumps(record) + "\n")
        return False
