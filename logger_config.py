"""
logger_config.py
────────────────
Centralized logging factory for the job market pipeline.

Usage in any module:
    from logger_config import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")

Log levels:
    DEBUG    — internal state, watermark registrations, cache hits
    INFO     — progress milestones, row counts, merge completions
    WARNING  — non-fatal skips, missing optional fields, retries
    ERROR    — caught exceptions, failed stages
    CRITICAL — unrecoverable failures (rarely used)

Environment variable override:
    Set JOB_PIPELINE_LOG_LEVEL=WARNING in production jobs to suppress noise.
    Default is INFO for balanced output in notebooks.
"""

import logging
import os
import sys

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s :: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Allow override via environment variable (useful for production jobs)
_DEFAULT_LEVEL = os.environ.get("JOB_PIPELINE_LOG_LEVEL", "INFO").upper()

# Root logger name — all pipeline loggers are children of this
_ROOT_NAME = "job_pipeline"

# Track whether the root handler has been configured
_configured = False


def _configure_root():
    """
    One-time setup of the root pipeline logger.
    Adds a StreamHandler to stdout so output appears in
    Databricks notebook cells and driver logs.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(getattr(logging, _DEFAULT_LEVEL, logging.INFO))

    # Avoid duplicate handlers if module is re-imported
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)

    # Prevent propagation to Spark's root logger (avoids duplicate lines)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger under the job_pipeline namespace.

    Args:
        name: Module name (typically __name__). Will be prefixed
              with 'job_pipeline.' if not already.

    Examples:
        get_logger(__name__)
            -> logger named "job_pipeline.scrape_google_careers"
        get_logger("watermark")
            -> logger named "job_pipeline.watermark"
    """
    _configure_root()

    # Strip common prefixes for cleaner names
    short = name.replace("job_market_analysis.", "")

    if short.startswith(_ROOT_NAME + "."):
        logger_name = short
    else:
        logger_name = f"{_ROOT_NAME}.{short}"

    return logging.getLogger(logger_name)


def set_level(level: str):
    """
    Dynamically change the log level for all pipeline loggers.

    Args:
        level: One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

    Usage:
        from logger_config import set_level
        set_level("DEBUG")   # verbose mode for troubleshooting
        set_level("WARNING") # quiet mode for production
    """
    _configure_root()
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
