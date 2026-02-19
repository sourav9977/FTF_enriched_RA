"""Centralized pipeline logging for the FTF prototype.

Provides structured, step-aware logging that captures:
  - Step entry/exit with timing
  - Input/output shapes and key stats
  - Per-item calculation traces (for debugging individual RA items)
  - Warnings and data quality flags

Usage:
    from src.pipeline_logger import get_logger, StepLogger

    log = get_logger(__name__)
    with StepLogger("4a", "Overlap Scoring") as step:
        step.log_input("RA", ra_df)
        step.log_calc("overlap_score", ra_id="RA_001", value=0.85,
                       detail="pattern=1, color=1, fit=0, sleeve=1, neck=0, length=1")
        step.log_output("best_matches", best_df)
"""

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_INITIALIZED = False


def setup_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_dir: Path = None,
) -> None:
    """Initialize logging for the full pipeline.

    Creates two handlers:
      - Console: INFO level (concise summaries)
      - File: DEBUG level (every calculation detail)
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    log_path = log_dir or _LOG_DIR
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"ftf_pipeline_{timestamp}.log"

    root = logging.getLogger("src")
    root.setLevel(logging.DEBUG)

    if not root.handlers:
        # Console handler — concise
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, console_level.upper()))
        ch.setFormatter(logging.Formatter(
            "%(levelname)-5s | %(message)s"
        ))
        root.addHandler(ch)

        # File handler — detailed
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(getattr(logging, file_level.upper()))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(fh)

    _INITIALIZED = True
    root.info(f"Logging initialized — file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module. Auto-initializes if needed."""
    if not _INITIALIZED:
        setup_logging()
    return logging.getLogger(name)


class StepLogger:
    """Context manager for step-level logging with timing and stats.

    Usage:
        with StepLogger("4a", "Overlap Scoring") as step:
            step.log_input("RA", df)
            step.log_calc(...)
            step.log_output("matches", result_df)
    """

    def __init__(self, step_id: str, step_name: str, logger_name: str = None):
        self.step_id = step_id
        self.step_name = step_name
        self.logger = get_logger(logger_name or f"src.step_{step_id}")
        self._start = None
        self._calc_count = 0

    def __enter__(self):
        self._start = time.time()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"STEP {self.step_id}: {self.step_name} — START")
        self.logger.info(f"{'='*60}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        if exc_type:
            self.logger.error(
                f"STEP {self.step_id}: {self.step_name} — FAILED after {elapsed:.2f}s: {exc_val}"
            )
        else:
            self.logger.info(
                f"STEP {self.step_id}: {self.step_name} — DONE in {elapsed:.2f}s "
                f"({self._calc_count} calculations logged)"
            )
        self.logger.info("")
        return False

    def log_input(self, name: str, data: Any, detail: str = None):
        """Log an input dataset's shape and key stats."""
        if isinstance(data, pd.DataFrame):
            msg = f"  INPUT  {name}: {data.shape[0]} rows × {data.shape[1]} cols"
            if detail:
                msg += f" | {detail}"
            self.logger.info(msg)
            # Log column-level nulls at DEBUG
            null_pcts = data.isnull().mean()
            high_null = null_pcts[null_pcts > 0.5]
            if not high_null.empty:
                self.logger.debug(f"  INPUT  {name} high-null cols: {dict(high_null.round(2))}")
        elif isinstance(data, pd.Series):
            self.logger.info(f"  INPUT  {name}: {len(data)} values, mean={data.mean():.3f}")
        elif isinstance(data, dict):
            self.logger.info(f"  INPUT  {name}: {len(data)} entries")
            self.logger.debug(f"  INPUT  {name}: {data}")
        else:
            self.logger.info(f"  INPUT  {name}: {data}")

    def log_output(self, name: str, data: Any, detail: str = None):
        """Log an output dataset's shape and key stats."""
        if isinstance(data, pd.DataFrame):
            msg = f"  OUTPUT {name}: {data.shape[0]} rows × {data.shape[1]} cols"
            if detail:
                msg += f" | {detail}"
            self.logger.info(msg)
        elif isinstance(data, pd.Series):
            self.logger.info(
                f"  OUTPUT {name}: {len(data)} values, "
                f"min={data.min():.3f}, mean={data.mean():.3f}, max={data.max():.3f}"
            )
        else:
            msg = f"  OUTPUT {name}: {data}"
            if detail:
                msg += f" | {detail}"
            self.logger.info(msg)

    def log_calc(self, calc_name: str, detail: str = None, **kwargs):
        """Log a single calculation with named values.

        Example:
            step.log_calc("overlap_score", ra_id="RA_001", brick="SHIRTS",
                          value=0.85, detail="pattern=1, color=1, fit=0")
        """
        self._calc_count += 1
        parts = [f"  CALC   {calc_name}"]
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        if detail:
            parts.append(f"| {detail}")
        self.logger.debug(" ".join(parts))

    def log_decision(self, item_id: str, decision: str, reason: str):
        """Log a classification or routing decision for an item."""
        self._calc_count += 1
        self.logger.debug(f"  DECIDE {item_id}: {decision} — {reason}")

    def log_warning(self, message: str):
        """Log a data quality or logic warning."""
        self.logger.warning(f"  WARN   {message}")

    def log_summary(self, **kwargs):
        """Log a step-level summary with key metrics."""
        parts = [f"  SUMMARY"]
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.3f}")
            else:
                parts.append(f"{k}={v}")
        self.logger.info(" ".join(parts))


class ItemTracer:
    """Trace a specific item through the entire pipeline for debugging.

    Usage:
        tracer = ItemTracer("RA_001")
        tracer.trace("Step 4a", "overlap_score", 0.85, "matched with Trend_003")
        ...
        print(tracer.report())
    """

    def __init__(self, item_id: str):
        self.item_id = item_id
        self.events = []
        self.logger = get_logger(f"src.trace.{item_id}")

    def trace(self, step: str, event: str, value: Any = None, detail: str = None):
        entry = {
            "step": step,
            "event": event,
            "value": value,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        }
        self.events.append(entry)
        msg = f"TRACE [{self.item_id}] {step} | {event}"
        if value is not None:
            msg += f" = {value}"
        if detail:
            msg += f" | {detail}"
        self.logger.debug(msg)

    def report(self) -> str:
        lines = [f"=== Item Trace: {self.item_id} ({len(self.events)} events) ==="]
        for e in self.events:
            val = f" = {e['value']}" if e['value'] is not None else ""
            det = f" | {e['detail']}" if e['detail'] else ""
            lines.append(f"  [{e['step']}] {e['event']}{val}{det}")
        return "\n".join(lines)
