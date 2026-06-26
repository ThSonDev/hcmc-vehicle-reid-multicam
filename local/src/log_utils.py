"""Standardized logging for the local pipeline.

`setup_logging("cam1")` returns a logger that writes to both:
  - Console: human-readable, colorized (when a terminal), format
             `HH:MM:SS LEVEL [component] msg key=val ...`
  - File:    `logs/<component>.jsonl`, one JSON event per line (re-readable by tools/AI),
             size-rotated (RotatingFileHandler) to avoid filling the disk / wasting TBW.

Pass structured fields via `extra=`:
    log.info("match", extra={"event": "match", "cam1_id": 5, "score": 0.81})
These become columns in the JSONL and `key=val` pairs on the console.
"""
import os
import sys
import json
import logging
import datetime
import faulthandler
from logging.handlers import RotatingFileHandler

# Standard LogRecord attributes -> used to separate the "extra" fields we add.
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}

_COLORS = {
    "DEBUG": "\033[37m",     # gray
    "INFO": "\033[36m",      # cyan
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[41m",  # red background
}
_RESET = "\033[0m"

# Default log size limits (DRAM-less SATA disk + low TBW -> keep small).
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

# Project root (parent of local/) so logs land in <root>/logs regardless of the cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per-run timestamp baked into log filenames (time_day-month-year) so each run is
# separated. run.py exports REID_LOG_STAMP so every component of one launch shares it;
# a standalone script falls back to its own start time.
_RUN_STAMP = os.environ.get("REID_LOG_STAMP") or datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")


def _extras(record):
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


_crash_installed = False


def install_crash_handler(logger):
    """Make abnormal deaths visible in the logs instead of vanishing to stderr.

    - `sys.excepthook`: any uncaught Python exception is logged as a CRITICAL
      `event=crash` record (with traceback) to the JSONL before the process dies,
      then re-raised to the default hook so it still prints to stderr.
    - `faulthandler`: native crashes (e.g. a CUDA/cuDNN segfault, which Python
      can't catch) dump a C-level traceback to stderr — which run.py now captures.

    Idempotent: installs once per process.
    """
    global _crash_installed
    if _crash_installed:
        return
    _crash_installed = True

    faulthandler.enable(all_threads=True)

    _prev_hook = sys.excepthook

    def _hook(etype, exc, tb):
        if not issubclass(etype, KeyboardInterrupt):
            logger.critical("Uncaught exception", exc_info=(etype, exc, tb),
                            extra={"event": "crash", "error": str(exc)})
        _prev_hook(etype, exc, tb)

    sys.excepthook = _hook


class JsonlFormatter(logging.Formatter):
    """One JSON object per log record."""

    def format(self, record):
        data = {
            "ts": datetime.datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "component": record.name,
            "msg": record.getMessage(),
        }
        data.update(_extras(record))
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact, colorized format for the terminal."""

    def __init__(self, use_color=True):
        super().__init__()
        self.use_color = use_color

    def format(self, record):
        ts = datetime.datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        color = _COLORS.get(record.levelname, "") if self.use_color else ""
        reset = _RESET if self.use_color else ""
        line = f"{ts} {color}{record.levelname:<7}{reset} [{record.name}] {record.getMessage()}"
        extras = _extras(record)
        if extras:
            line += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(component, level=logging.INFO, log_dir=None, capture_crashes=True):
    """Return a configured logger for `component` (e.g. "cam1", "producer", "system").

    Calling repeatedly with the same component returns the same logger without
    attaching duplicate handlers. Console follows `level`; the JSONL file always
    logs from DEBUG to keep full detail. When `capture_crashes` is set (default),
    uncaught exceptions and native faults are routed to the logs (see
    `install_crash_handler`) so a process can't die silently.
    """
    log_dir = log_dir or os.environ.get("REID_LOG_DIR") or os.path.join(_ROOT, "logs")
    logger = logging.getLogger(component)
    if logger.handlers:  # already configured -> idempotent
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(ConsoleFormatter(use_color=sys.stderr.isatty()))
    ch.setLevel(level)
    logger.addHandler(ch)

    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(log_dir, f"{_RUN_STAMP}_{component}.jsonl"),
            maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8",
        )
        fh.setFormatter(JsonlFormatter())
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    except OSError as e:
        logger.warning("Could not create log file, console only", extra={"error": str(e)})

    if capture_crashes:
        install_crash_handler(logger)

    return logger
