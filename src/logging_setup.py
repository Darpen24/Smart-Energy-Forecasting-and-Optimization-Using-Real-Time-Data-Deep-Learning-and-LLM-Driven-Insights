# logging_setup.py
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

def get_logger(name: str, prefix: str = "pipeline", logs_dir: str = "logs") -> logging.Logger:
    """
    Returns a logger that writes to logs/<prefix>-YYYY-MM-DD.log
    and also to the console. Rotates at midnight, keeps 14 days.
    """
    os.makedirs(logs_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(logs_dir, f"{prefix}-{date_str}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate handlers if called twice

    # If no handlers, attach file + console
    if not logger.handlers:
        # File handler (rotates daily)
        fh = TimedRotatingFileHandler(log_path, when="midnight", backupCount=14, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger
